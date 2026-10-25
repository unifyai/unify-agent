"""The child half of the sandboxed Python worker.

This file runs inside bubblewrap as ``python -I -S`` and imports nothing from
``unify``: the harness starts it by path (``runpy.run_path``), so no package
``__init__`` runs, and the only things it can reach in the harness are the
names the harness installs through the protocol below. The harness imports the
same file for :func:`encode` and :func:`decode`, so both halves share one
codec.

Protocol: one JSON object per line. The harness writes to the worker's stdin
and reads its stdout; the worker moves both onto private descriptors first, so
nothing a cell prints or a subprocess writes can reach the channel.

Harness -> worker::

    {"op": "init", "sys_path": [...], "builtins": [...], "globals": {...},
                                              (a global is ["module", m], ["attr", m, a] or
                                              ["lazy_module", m], imported on first use)
     "help": bool, "audit": {"roots": [...], "path": str}, "no_bytecode": true,
     "sweep": [root, ...]}                    (audit, no_bytecode: only under UNIFY_MEMORY; sweep: only
                                              the memory agents' interpreters, :meth:`Worker.sweep_modules`)
    {"op": "exec", "id": n, "source": str, "sync": {...}}
    {"op": "reply", "id": k, "value": ..., "coroutine": bool}
    {"op": "reply", "id": k, "error": {"type", "module", "message", "args"}}
    {"op": "interrupt", "id": n}              (a cancel: cell n stops where it is, the namespace is kept)

Worker -> harness::

    {"op": "ready", "missing": {name: error}, "audit": "on" | error}
                                              (audit: only when init asked)
    {"op": "call" | "describe" | "dir", "id": k, "target": {...}, ...}
    {"op": "doc", "id": k, "target": {...} | None, "label": str}
                                              (``help(obj)`` for a harness object: unify/actor/core_surface.py)
    {"op": "done", "id": n, "result": ..., "error": str | None, ...,
     "audit": {"records", "dropped", "failed", "bytes"}}   (when installed)

Values cross as JSON with a few tagged forms (``{"__unify__": tag, ...}``):
tuples, sets, bytes, non-string dictionary keys, dates and times, decimals,
references to harness objects (``ref``), pydantic models sent by value with a
reference back to the original (``model``) and, for a cell's result only, a
worker object the harness can see only as its repr (``opaque``). JSON rather
than pickle because the harness decodes what model-written code sends: a JSON
decoder builds only lists, dictionaries and scalars, whereas unpickling is safe
only as far as a ``find_class`` allowlist and every ``__reduce__`` it admits.
"""

from __future__ import annotations

import asyncio
import base64
import builtins
import datetime as _dt
import decimal
import importlib
import io
import itertools
import json
import os
import re
import sys
import signal
import tempfile
import threading
import time
import traceback
import types
from typing import Any, Callable, Optional

TAG = "__unify__"
#: How much earlier than a cell's start a file time may be and still count as
#: written during it (file times lag the clock by up to a timer tick).
_SWEEP_MARGIN_NS = 2_000_000_000

#: The module whose dict is the session's globals (``__name__`` of every cell).
SESSION_MODULE = "__sandbox_worker__"
MAX_DEPTH = 64
MAX_REPR = 500
MAX_FD_OUTPUT = 256 * 1024


__all__ = [
    "BoundaryRefusal",
    "MAX_DEPTH",
    "TAG",
    "decode",
    "json_values",
    "encode",
    "short_repr",
]


class BoundaryRefusal(TypeError):
    """A value or attribute that cannot cross the worker boundary, named."""


#: The signal the channel's reader sends this process's main thread to interrupt the running cell.
INTERRUPT_SIGNAL = signal.SIGUSR1


class CellInterrupted(BaseException):
    """Raised in a cell the requester cancelled while it ran: the cell stops where it is, and everything the
    session's earlier cells bound is kept. A ``BaseException``, so a cell's ``except Exception`` does not
    swallow it."""


# ---------------------------------------------------------------------------
# The JSON values a text carries (the memory package reads a request's with it)
# ---------------------------------------------------------------------------

_JSON_DECODER = json.JSONDecoder()
_JSON_OPEN = re.compile(r"[\[{]")


def json_values(text: str) -> list:
    """Every top-level JSON object and array in *text*, parsed, in order.

    The text is scanned from the left: at each ``{`` or ``[`` the standard
    json module reads one value; a value read is kept and the scan goes on
    after it. Where no valid value starts, the scan goes on from the point
    where the text stopped being JSON. So a fenced block and JSON written in
    prose are found alike, a nested array is one value (the outermost),
    brackets inside a JSON string belong to the string, invalid JSON is
    ignored (with any value inside it before the point where it fails), and
    the scan takes time linear in the text. Brackets nested deeper than the
    json module reads end the scan. Scalars outside an object or array are
    not collected.
    """
    values: list = []
    index = 0
    while True:
        found = _JSON_OPEN.search(text, index)
        if found is None:
            return values
        start = found.start()
        try:
            value, end = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            index = max(exc.pos, start + 1)
            continue
        except RecursionError:
            return values
        values.append(value)
        index = end


def short_repr(value: Any, limit: int = MAX_REPR) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a broken __repr__ must not break a call
        text = f"<{type(value).__name__} object>"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def type_name(value: Any) -> str:
    cls = type(value)
    module = getattr(cls, "__module__", "") or ""
    name = getattr(cls, "__qualname__", cls.__name__)
    if module in ("builtins", "") or module.startswith("__sandbox"):
        return name
    return f"{module}.{name}"


def _cell_class(value: Any) -> bool:
    """A class defined by cell code in the cells' namespace."""
    module = getattr(value, "__module__", None)
    return isinstance(value, type) and isinstance(module, str) and module.startswith("__sandbox_")


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def encode(
    value: Any,
    *,
    pre: Optional[Callable[[Any], Any]] = None,
    unknown: Callable[[Any, str], Any],
    where: str = "value",
    _depth: int = 0,
) -> Any:
    """*value* as JSON-ready data.

    ``pre(value)`` may return a tagged form for types the caller knows (a
    reference, a model) or None; ``unknown(value, where)`` handles anything
    else, by returning a tagged form or raising :class:`BoundaryRefusal`.
    """
    if value is None or type(value) in (bool, int, float, str):
        return value
    if _depth > MAX_DEPTH:
        raise BoundaryRefusal(f"{where} is nested more than {MAX_DEPTH} levels deep")
    if pre is not None:
        tagged = pre(value)
        if tagged is not None:
            return tagged
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str(value)

    def rec(v: Any) -> Any:
        return encode(v, pre=pre, unknown=unknown, where=where, _depth=_depth + 1)

    if isinstance(value, dict):
        if all(type(k) is str for k in value) and TAG not in value:
            return {k: rec(v) for k, v in value.items()}
        return {TAG: "dict", "items": [[rec(k), rec(v)] for k, v in value.items()]}
    if isinstance(value, list):
        return [rec(v) for v in value]
    if isinstance(value, tuple):
        return {TAG: "tuple", "items": [rec(v) for v in value]}
    if isinstance(value, frozenset):
        return {TAG: "frozenset", "items": [rec(v) for v in value]}
    if isinstance(value, set):
        return {TAG: "set", "items": [rec(v) for v in value]}
    if isinstance(value, (bytes, bytearray)):
        return {TAG: "bytes", "b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, _dt.datetime):
        return {TAG: "datetime", "iso": value.isoformat()}
    if isinstance(value, _dt.date):
        return {TAG: "date", "iso": value.isoformat()}
    if isinstance(value, _dt.time):
        return {TAG: "time", "iso": value.isoformat()}
    if isinstance(value, _dt.timedelta):
        return {
            TAG: "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, decimal.Decimal):
        return {TAG: "decimal", "s": str(value)}
    return unknown(value, where)


def decode(
    value: Any,
    *,
    tagged: Callable[[str, dict], Any],
    _depth: int = 0,
) -> Any:
    """The inverse of :func:`encode`; ``tagged(tag, payload)`` handles the
    caller's own tags (``ref``, ``model``, ``opaque``, ``target``)."""
    if _depth > MAX_DEPTH + 2:
        raise BoundaryRefusal(f"value is nested more than {MAX_DEPTH} levels deep")
    if isinstance(value, list):
        return [decode(v, tagged=tagged, _depth=_depth + 1) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(TAG)
    if tag is None:
        return {
            k: decode(v, tagged=tagged, _depth=_depth + 1) for k, v in value.items()
        }

    def items() -> list:
        return [decode(v, tagged=tagged, _depth=_depth + 1) for v in value["items"]]

    if tag == "dict":
        return {
            decode(k, tagged=tagged, _depth=_depth + 1): decode(
                v,
                tagged=tagged,
                _depth=_depth + 1,
            )
            for k, v in value["items"]
        }
    if tag == "tuple":
        return tuple(items())
    if tag == "set":
        return set(items())
    if tag == "frozenset":
        return frozenset(items())
    if tag == "bytes":
        return base64.b64decode(value["b64"])
    if tag == "datetime":
        return _dt.datetime.fromisoformat(value["iso"])
    if tag == "date":
        return _dt.date.fromisoformat(value["iso"])
    if tag == "time":
        return _dt.time.fromisoformat(value["iso"])
    if tag == "timedelta":
        return _dt.timedelta(
            days=value["days"],
            seconds=value["seconds"],
            microseconds=value["microseconds"],
        )
    if tag == "decimal":
        return decimal.Decimal(value["s"])
    return tagged(str(tag), value)


# ---------------------------------------------------------------------------
# Worker-side stand-ins for harness objects
# ---------------------------------------------------------------------------


class RemoteError(Exception):
    """An exception raised in the harness whose type does not exist here.

    Subclasses are made on demand, named like the harness type, so a
    traceback reads ``unify.common.tool_errors.ToolInputError: ...``.
    """

    harness_type = ""


#: Where the harness defines the request error a cell's model call raises.
MODEL_REQUEST_ERROR_MODULE = "unify.common.cell_models"


class ModelRequestError(RemoteError, ValueError):
    """The provider refused a model call this code made because of the
    request's own content: a ``response_format`` schema it does not accept, a
    parameter the endpoint does not take, a prompt over the model's limit.

    The message is the provider's. Nothing was generated; fix the request and
    call again. The harness's error of that name arrives as this class, which
    is the cell's ``ModelRequestError`` global.
    """

    harness_type = f"{MODEL_REQUEST_ERROR_MODULE}.ModelRequestError"


ModelRequestError.__module__ = MODEL_REQUEST_ERROR_MODULE


class ModelCallFailed(RemoteError, RuntimeError):
    """A model call this code made failed for a reason other than the
    request's own content: access, a rate limit, the provider's own fault.

    The text is fixed: the status and the error's type, never the provider's
    reply. Nothing was generated. The harness's error of that name arrives as
    this class, which is the cell's ``ModelCallFailed`` global.
    """

    harness_type = f"{MODEL_REQUEST_ERROR_MODULE}.ModelCallFailed"


ModelCallFailed.__module__ = MODEL_REQUEST_ERROR_MODULE


class _Remote:
    """A harness object reached through the worker's channel."""

    def __init__(
        self,
        worker: "Worker",
        target: dict,
        label: str,
        repr_text: str,
        is_async: bool = False,
    ) -> None:
        object.__setattr__(self, "_w", worker)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_repr", repr_text)
        object.__setattr__(self, "_async", is_async)

    def __getattr__(self, name: str) -> Any:
        return self._w.attribute(self, name)

    def __getitem__(self, key: Any) -> Any:
        return self._w.item(self, key)

    # Without this, ``for x in proxy`` and ``x in proxy`` would fall back to
    # ``proxy[0]``, ``proxy[1]``, ... across the boundary; they stay a
    # TypeError, as before subscripts crossed.
    __iter__ = None

    def __setattr__(self, name: str, value: Any) -> None:
        raise BoundaryRefusal(
            f"cannot set {name!r} on {self._label}: harness objects cannot be "
            "changed from the worker",
        )

    def __dir__(self) -> list:
        return list(self._w.request_sync("dir", target=self._target)[0])

    def __repr__(self) -> str:
        return self._repr


class RemoteNamespace(_Remote):
    """A harness object with attributes (``primitives``, a manager, a handle)."""


class RemoteCallable(RemoteNamespace):
    """A harness callable; calling it runs it in the harness."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._w.invoke(self, args, kwargs)


class RemoteRecord(dict):
    """A pydantic model the harness sent by value.

    Reads like the model (``r.field`` and ``r["field"]``); passed back to the
    harness unchanged it becomes the original model again.
    """

    def __init__(self, data: dict, type_name_: str, ref: Optional[int]) -> None:
        super().__init__(data)
        self._unify_type = type_name_
        self._unify_ref = ref
        self._unify_snapshot = dict(data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_unify_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{self._unify_type} (sent from the harness by value) has no "
                f"field {name!r}",
            ) from None

    def model_dump(self, **_: Any) -> dict:
        return dict(self)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in self.items())
        return f"{self._unify_type}({fields})"


class _Refused:
    """A harness name that cannot be used in the worker, saying why."""

    def __init__(self, name: str, reason: str) -> None:
        self._name = name
        self._reason = reason

    def _refuse(self) -> BoundaryRefusal:
        return BoundaryRefusal(
            f"{self._name!r} is not available in the sandboxed Python worker: "
            f"{self._reason}",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self._refuse()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        raise self._refuse()

    def __repr__(self) -> str:
        return f"<unavailable in the worker: {self._name}>"


class _LazyModule:
    """A module name bound in the cells' globals that imports the module on its first attribute access.

    Nothing of the module runs when the worker starts: its code runs inside the cell that first uses it, under
    that cell's timeout, as an ``import`` line in the cell would: before the cell's own code when the cell's code names
    it (``Worker.load_named``), else at its first attribute access. Then the name in the namespace is rebound to
    the real module, when it still holds this object. A failed import raises :class:`ImportError` naming the
    module, chained to the cause, as the cell's error (an interrupt or an exit passes unchanged); the next
    access tries again. ``import <name>`` in a cell goes through the normal import system as before.
    """

    __slots__ = ("_name", "_ns")

    def __init__(self, name: str, ns: dict) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_ns", ns)

    def _load(self) -> Any:
        name = self._name
        try:
            module = importlib.import_module(name)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:  # noqa: BLE001 - any failure is the cell's error
            raise ImportError(
                f"importing the module {name!r} failed ({type(exc).__name__})",
                name=name,
            ) from exc
        if self._ns.get(name) is self:
            self._ns[name] = module
        return module

    def __getattr__(self, attr: str) -> Any:
        # Dunder probes (copying, introspection) do not import the module.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return getattr(self._load(), attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._load(), attr, value)

    def __dir__(self) -> list:
        return dir(self._load())

    def __repr__(self) -> str:
        return f"<module {self._name!r}, imported on first use>"


_NO_ARGUMENT = object()


async def _ready(value: Any) -> Any:
    return value


# ``query_llm(..., response_format=Model)`` with a model a cell defined: the
# class cannot cross to the harness, so its JSON schema goes in the OpenAI
# form the harness already takes as a dict, and the answer (a dict) is
# validated here into the cell's own class.
_QUERY_LLM = "query_llm"
MAX_SCHEMA_CHARS = 64 * 1024


def _cell_response_model(label: str, kwargs: dict) -> Any:
    """The cell-defined pydantic model a ``query_llm`` call names as its
    ``response_format``, or None (anything else crosses as it does today)."""
    if label != _QUERY_LLM:
        return None
    value = kwargs.get("response_format")
    if not _cell_class(value):
        return None
    from pydantic import BaseModel

    return value if issubclass(value, BaseModel) else None


def _response_format_of(model: Any) -> dict:
    """*model*'s JSON schema as plain data, refused when it cannot be one."""
    where = f"response_format {model.__name__} of {_QUERY_LLM}"
    try:
        text = json.dumps(model.model_json_schema())
    except Exception as exc:  # noqa: BLE001 - the schema is the model's own
        raise BoundaryRefusal(
            f"{where}: its JSON schema could not be built "
            f"({type(exc).__name__}: {exc})",
        ) from None
    if len(text) > MAX_SCHEMA_CHARS:
        raise BoundaryRefusal(
            f"{where}: its JSON schema is {len(text)} characters, over the "
            f"{MAX_SCHEMA_CHARS} that may cross to the harness",
        )
    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__[:64], "schema": json.loads(text)},
    }


def _validated(model: Any, value: Any) -> Any:
    """The answer as an instance of the cell's *model*; pydantic's
    ``ValidationError`` when it does not fit (as in-process). An answer that
    is not a dict passes as it is."""
    return model.model_validate(value) if isinstance(value, dict) else value


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------


class Worker:
    def __init__(self, rfile: Any, wfile: Any, out_fd: int) -> None:
        self._rfile = rfile
        self._wfile = wfile
        self._wlock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, tuple] = {}
        self._describe_cache: dict[str, Any] = {}
        self._exc_classes: dict[tuple[str, str], type] = {
            (MODEL_REQUEST_ERROR_MODULE, "ModelRequestError"): ModelRequestError,
            (MODEL_REQUEST_ERROR_MODULE, "ModelCallFailed"): ModelCallFailed,
        }
        self._out_fd = out_fd
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # The session's globals are a registered module's dict, as in-process
        # (``PythonExecutionSession``): a class a cell defines resolves names
        # through ``sys.modules[cls.__module__]`` (pydantic's annotations do).
        # Emptied so the namespace holds only what init and the cells bind.
        module = types.ModuleType(SESSION_MODULE)
        module.__dict__.clear()
        sys.modules[SESSION_MODULE] = module
        self.ns: dict[str, Any] = module.__dict__
        self.installed: dict[str, Any] = {}
        self._stdout: list[dict] = []
        self._stderr: list[dict] = []
        self._real_print = builtins.print
        # UNIFY_MEMORY (spec §3a): the work-tree audit hook, only when the
        # harness asks for it in init; ``audit_state`` is "on" or why not.
        self.audit: Any = None
        self.audit_state: Optional[str] = None
        # The memory agents' persistent interpreters: the roots whose modules
        # are checked for changed files before each cell (init's "sweep"),
        # each module's file and stamp when first seen, the lazily bound
        # names (rebound when their package is dropped) and when the last
        # cell started (nanoseconds).
        self.sweep_roots: tuple[str, ...] = ()
        self._module_stamps: dict[str, tuple] = {}
        self._lazy: dict[str, str] = {}
        self._cell_started_ns = 0
        # The cell running now (its id and task) and the id of one the harness asked to interrupt (a cancel).
        self._running: Optional[tuple[Any, asyncio.Task]] = None
        self._interrupt_id: Any = None

    # -- channel -------------------------------------------------------------
    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._wlock:
            self._wfile.write(data)
            self._wfile.flush()


    def reader(self) -> None:
        for line in self._rfile:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("op") == "interrupt":
                self.interrupt(msg.get("id"))
                continue
            if msg.get("op") == "reply":
                entry = self._pending.pop(msg.get("id"), None)
                if entry is None:
                    continue
                if entry[0] == "sync":
                    entry[2].append(msg)
                    entry[1].set()
                else:
                    loop, fut = entry[1], entry[2]
                    loop.call_soon_threadsafe(_resolve, fut, msg)
            else:
                assert self.loop is not None
                self.loop.call_soon_threadsafe(self.handle, msg)
        # The harness closed the channel: nothing more can be asked of us.
        os._exit(0)

    def interrupt(self, cid: Any) -> None:
        """On the reader thread: stop cell *cid* if it is still running. A cell waiting on an ``await`` is
        cancelled there; one running code is interrupted by :data:`INTERRUPT_SIGNAL` (:meth:`on_interrupt`). Either
        way it ends as an error and the namespace is kept. A cell that already ended is left alone."""
        running = self._running
        if running is None or running[0] != cid or self.loop is None:
            return
        self._interrupt_id = cid
        self.loop.call_soon_threadsafe(running[1].cancel)
        try:
            signal.pthread_kill(threading.main_thread().ident, INTERRUPT_SIGNAL)
        except (OSError, ValueError):
            pass

    def on_interrupt(self, signum: int, frame: Any) -> None:
        """:data:`INTERRUPT_SIGNAL`'s handler, on the main thread: raise :class:`CellInterrupted` in the cell the
        harness asked to interrupt, when its code is what runs (otherwise it waits, and its cancellation stops
        it)."""
        running = self._running
        if running is None or self._interrupt_id is None or running[0] != self._interrupt_id:
            return
        while frame is not None:
            if frame.f_code.co_name == "__exec_wrapper":
                self._interrupt_id = None
                raise CellInterrupted(
                    "the requester cancelled the request while this cell ran: it was interrupted here, and the "
                    "session's variables are kept",
                )
            frame = frame.f_back


    def request_sync(self, op: str, **fields: Any) -> tuple[Any, dict]:
        rid = next(self._ids)
        event = threading.Event()
        box: list = []
        self._pending[rid] = ("sync", event, box)
        self.send({"op": op, "id": rid, **fields})
        event.wait()
        return self._unwrap(box[0]), box[0]

    async def request_async(self, op: str, **fields: Any) -> tuple[Any, dict]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        rid = next(self._ids)
        self._pending[rid] = ("async", loop, fut)
        try:
            self.send({"op": op, "id": rid, **fields})
            msg = await fut
        finally:
            self._pending.pop(rid, None)
        return self._unwrap(msg), msg

    def _unwrap(self, msg: dict) -> Any:
        if "error" in msg:
            raise self._exception(msg["error"])
        return self.decode(msg.get("value"))

    def _exception(self, err: dict) -> BaseException:
        name = str(err.get("type") or "Exception")
        module = str(err.get("module") or "")
        message = str(err.get("message") or "")
        if module == "builtins":
            cls = getattr(builtins, name, None)
            if isinstance(cls, type) and issubclass(cls, Exception):
                try:
                    args = err.get("args")
                    args = self.decode(args) if args is not None else [message]
                    return cls(*args)
                except Exception:  # noqa: BLE001 - fall back to the named class
                    pass
        key = (module, name)
        cls = self._exc_classes.get(key)
        if cls is None:
            cls = type(
                name,
                (RemoteError,),
                {"__module__": module or "harness", "harness_type": f"{module}.{name}"},
            )
            self._exc_classes[key] = cls
        return cls(message)

    # -- codec hooks -----------------------------------------------------------
    def decode(self, value: Any) -> Any:
        return decode(value, tagged=self._tagged)

    def _tagged(self, tag: str, payload: dict) -> Any:
        if tag == "ref":
            target = {"ref": int(payload["id"]), "path": []}
            label = f"<harness {payload.get('type', 'object')}>"
            repr_text = str(payload.get("repr") or label)
            if payload.get("callable"):
                return RemoteCallable(
                    self,
                    target,
                    label,
                    repr_text,
                    bool(payload.get("async")),
                )
            return RemoteNamespace(self, target, label, repr_text)
        if tag == "model":
            data = self.decode(payload.get("data") or {})
            ref = payload.get("ref")
            return RemoteRecord(
                data if isinstance(data, dict) else {"value": data},
                str(payload.get("type") or "Model"),
                int(ref) if ref is not None else None,
            )
        raise BoundaryRefusal(f"the harness sent an unknown value form {tag!r}")

    def _pre(self, value: Any) -> Any:
        if isinstance(value, _Remote):
            target = object.__getattribute__(value, "_target")
            if "ref" in target and not target["path"]:
                return {TAG: "ref", "id": target["ref"]}
            return {TAG: "target", "target": target}
        if isinstance(value, RemoteRecord):
            if value._unify_ref is not None and dict(value) == value._unify_snapshot:
                return {TAG: "ref", "id": value._unify_ref}
        return None

    def encode_arg(self, value: Any, where: str) -> Any:
        def unknown(v: Any, w: str) -> Any:
            raise BoundaryRefusal(
                f"{w} is a {type_name(v)} ({short_repr(v, 80)}), which cannot be "
                "passed to the harness: only data (str, int, float, bool, None, "
                "list, tuple, dict, set, bytes, dates, Decimal) and objects the "
                "harness handed out cross the worker boundary",
            )

        return encode(value, pre=self._pre, unknown=unknown, where=where)

    def encode_result(self, value: Any) -> Any:
        def unknown(v: Any, w: str) -> Any:
            return {TAG: "opaque", "type": type_name(v), "repr": short_repr(v)}

        return encode(value, pre=self._pre, unknown=unknown, where="the result")

    # -- proxies ---------------------------------------------------------------
    def attribute(self, proxy: _Remote, name: str) -> Any:
        label = f"{proxy._label}.{name}"
        if name.startswith("_"):
            raise AttributeError(
                f"{label}: attributes starting with '_' of harness objects do not "
                "cross the worker boundary",
            )
        base = object.__getattribute__(proxy, "_target")
        return self._remote_at({**base, "path": [*base["path"], name]}, label)

    def item(self, proxy: _Remote, key: Any) -> Any:
        """``proxy[key]``, resolved in the harness, for a ``str`` or ``int`` key.

        The key is one more element of the target's path, ``{"item": key}``;
        the harness checks it as it checks an attribute name.
        """
        label = f"{proxy._label}[{key!r}]"
        if type(key) not in (str, int):
            raise BoundaryRefusal(
                f"{proxy._label}[...]: only str and int subscripts of harness "
                f"objects cross the worker boundary, not {type(key).__name__}",
            )
        if isinstance(key, str) and key.startswith("_"):
            raise KeyError(
                f"{label}: keys starting with '_' of harness objects do not "
                "cross the worker boundary",
            )
        base = object.__getattribute__(proxy, "_target")
        return self._remote_at({**base, "path": [*base["path"], {"item": key}]}, label)

    def _remote_at(self, target: dict, label: str) -> Any:
        key = json.dumps(target, sort_keys=True)
        desc = self._describe_cache.get(key)
        if desc is None:
            desc, _ = self.request_sync("describe", target=target)
            self._describe_cache[key] = desc
        kind = desc.get("kind")
        if kind == "callable":
            return RemoteCallable(
                self,
                target,
                label,
                f"<harness callable {label}>",
                bool(desc.get("async")),
            )
        if kind == "namespace":
            return RemoteNamespace(self, target, label, str(desc.get("repr") or label))
        return desc.get("value")

    def invoke(self, proxy: _Remote, args: tuple, kwargs: dict) -> Any:
        label = proxy._label
        target = object.__getattribute__(proxy, "_target")
        model = _cell_response_model(label, kwargs)
        if model is not None:
            kwargs = {**kwargs, "response_format": _response_format_of(model)}
        enc_args = [
            self.encode_arg(a, f"argument {i + 1} of {label}")
            for i, a in enumerate(args)
        ]
        enc_kwargs = {
            str(k): self.encode_arg(v, f"argument {k!r} of {label}")
            for k, v in kwargs.items()
        }
        fields = {"target": target, "args": enc_args, "kwargs": enc_kwargs}
        if object.__getattribute__(proxy, "_async"):
            return self._call_async(fields, model)
        value, msg = self.request_sync("call", **fields)
        if model is not None:
            value = _validated(model, value)
        return _ready(value) if msg.get("coroutine") else value

    async def _call_async(self, fields: dict, model: Any = None) -> Any:
        value, _ = await self.request_async("call", **fields)
        return value if model is None else _validated(model, value)

    # -- output ----------------------------------------------------------------
    def _write(self, parts: list, text: str) -> None:
        if not text:
            return
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += text
        else:
            parts.append({"type": "text", "text": text})

    def _print(
        self,
        *args: Any,
        sep: Any = " ",
        end: Any = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:  # noqa: E501
        if file is not None:
            self._real_print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        sep = " " if sep is None else sep
        end = "\n" if end is None else end
        self._write(self._stdout, sep.join(str(a) for a in args) + end)

    def display(self, obj: Any) -> None:
        module = type(obj).__module__ or ""
        if module.startswith("PIL.") and hasattr(obj, "save"):
            buf = io.BytesIO()
            obj.save(buf, format="PNG")
            self._stdout.append(
                {
                    "type": "image",
                    "mime": "image/png",
                    "data": base64.b64encode(buf.getvalue()).decode("ascii"),
                },
            )
            return
        self._write(self._stdout, (obj if isinstance(obj, str) else str(obj)) + "\n")

    def _fd_offset(self) -> int:
        return os.fstat(self._out_fd).st_size

    def _fd_output(self, start: int) -> str:
        end = os.fstat(self._out_fd).st_size
        size = end - start
        if size <= 0:
            return ""
        if size <= MAX_FD_OUTPUT:
            raw = os.pread(self._out_fd, size, start)
        else:
            half = MAX_FD_OUTPUT // 2
            raw = (
                os.pread(self._out_fd, half, start)
                + f"\n[... {size - 2 * half} bytes omitted ...]\n".encode()
                + os.pread(self._out_fd, half, end - half)
            )
        return raw.decode("utf-8", errors="replace")

    # -- namespace -------------------------------------------------------------
    def init(self, msg: dict) -> dict:
        if msg.get("no_bytecode") is True:
            # UNIFY_MEMORY: importing the memory export writes no __pycache__ into it
            sys.dont_write_bytecode = True
        sys.path[:] = [p for p in msg.get("sys_path", []) if isinstance(p, str) and p]
        names = msg.get("builtins") or []
        safe = {n: getattr(builtins, n) for n in names if hasattr(builtins, n)}
        if "print" in safe:
            safe["print"] = self._print
        self.ns["__builtins__"] = safe
        self.ns["__name__"] = SESSION_MODULE
        missing: dict[str, str] = {}
        for name, spec in (msg.get("globals") or {}).items():
            if (
                isinstance(spec, list)
                and len(spec) == 2
                and spec[0] == "lazy_module"
                and isinstance(spec[1], str)
            ):
                # imported by the first cell that uses it (:class:`_LazyModule`)
                self.ns[name] = _LazyModule(spec[1], self.ns)
                self._lazy[name] = spec[1]
                continue
            value = self._import(spec)
            if isinstance(value, _Refused):
                missing[name] = value._reason
            else:
                self.ns[name] = value
        if msg.get("help"):
            # Under -S there is no site-installed help(); this one prints the
            # harness objects' documentation too.
            self.ns["help"] = self.help
        if isinstance(msg.get("audit"), dict):
            self.audit_state = self._install_audit(msg["audit"])
        sweep = msg.get("sweep")
        if isinstance(sweep, list):
            self.sweep_roots = tuple(
                r.rstrip("/") for r in sweep if isinstance(r, str) and r.startswith("/") and r.rstrip("/")
            )
        return missing

    def _under_sweep(self, path: Any) -> bool:
        return isinstance(path, str) and any(path.startswith(r + "/") for r in self.sweep_roots)

    def sweep_modules(self) -> list[str]:
        """Before a cell in a persistent memory interpreter: drop every module
        loaded from a file under :attr:`sweep_roots` whose file changed since
        it was loaded, with every module of the same top-level package (a
        package re-imported without its submodules would lose them as
        attributes), so the next import reads the files as they are now. A
        lazily bound name whose package was dropped is bound lazily again; a
        name a cell bound is left as it is (it keeps the code it was bound
        to). Fails safe: a module whose file cannot be read, or that was first
        seen after a cell during which its file may have been written, is
        dropped too. Standard library only; reads ``os.stat`` of paths in the
        box. Returns the dropped names."""
        now = time.time_ns()
        changed: set[str] = set()
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if not self._under_sweep(path):
                continue
            try:
                st = os.stat(path)
                stamp: Optional[tuple] = (path, st.st_size, st.st_mtime_ns)
            except (OSError, ValueError):
                stamp = None
            known = self._module_stamps.get(name)
            if known is None:
                # loaded during the last cell: its file may have been written
                # after it was read, so only a file older than that cell's start
                # is known to match its code (file times come from a coarse
                # clock, hence the margin; a needless reload is harmless)
                if stamp is None or stamp[2] >= self._cell_started_ns - _SWEEP_MARGIN_NS:
                    changed.add(name)
                else:
                    self._module_stamps[name] = stamp
            elif known != stamp:
                changed.add(name)
        tops = {n.split(".", 1)[0] for n in changed}
        dropped: list[str] = []
        for name in list(sys.modules):
            if name.split(".", 1)[0] in tops and self._under_sweep(getattr(sys.modules[name], "__file__", None)):
                sys.modules.pop(name, None)
                self._module_stamps.pop(name, None)
                dropped.append(name)
        for name, target in self._lazy.items():
            value = self.ns.get(name)
            if (
                target.split(".", 1)[0] in tops
                and isinstance(value, types.ModuleType)
                and getattr(value, "__name__", None) == target
            ):
                self.ns[name] = _LazyModule(target, self.ns)
        if dropped:
            importlib.invalidate_caches()
        self._cell_started_ns = now
        return dropped

    def _install_audit(self, spec: dict) -> str:
        """Install the memory audit hook (``unify/memory/integration/
        adapters/audit.py``), loaded by its file path so no ``unify`` package
        code runs here; "on", or why it is not installed."""
        try:
            import importlib.util

            roots = [r for r in spec.get("roots") or [] if isinstance(r, str)]
            found = importlib.util.spec_from_file_location(
                "_unify_memory_audit",
                str(spec.get("path")),
            )
            module = importlib.util.module_from_spec(found)
            found.loader.exec_module(module)
            self.audit = module.install(roots)
            return "on"
        except Exception as exc:  # noqa: BLE001 - the cells run without it
            self.audit = None
            return f"{type(exc).__name__}: {exc}"[:300]

    def _audit(self, control: str) -> None:
        """``begin`` or ``end`` on the audit hook; a tampered hook never changes the cell."""
        try:
            getattr(self.audit, control)()
        except Exception:  # noqa: BLE001 - the records are hints only
            pass

    def _audit_drain(self) -> Any:
        """The hook's drained records if they can be sent, else None: the ``done`` message is never lost."""
        try:
            drained = self.audit.drain()
            if isinstance(drained, dict):
                json.dumps(drained)
                return drained
        except Exception:  # noqa: BLE001 - a tampered hook never loses the cell
            pass
        return None

    @staticmethod
    def _import(spec: list) -> Any:
        try:
            value: Any = importlib.import_module(spec[1])
            if spec[0] == "attr":
                for part in spec[2].split("."):
                    value = getattr(value, part)
            return value
        except Exception as exc:  # noqa: BLE001 - surfaced when it is used
            return _Refused(
                str(spec[-1]),
                f"importing it in the worker failed ({type(exc).__name__}: {exc})",
            )

    def _local(self, which: str) -> Any:
        if which == "display":
            return self.display
        if which == "run_coro_sync":
            return _run_coro_sync
        if which == "model_request_error":
            return ModelRequestError
        if which == "model_call_failed":
            return ModelCallFailed
        raise BoundaryRefusal(f"unknown worker-local name {which!r}")


    def apply_sync(self, sync: dict) -> None:
        for name in sync.get("remove") or []:
            obj = self.installed.pop(name, None)
            if obj is not None and self.ns.get(name) is obj:
                del self.ns[name]
        for name, desc in (sync.get("set") or {}).items():
            kind = desc.get("kind")
            target = {"root": name, "path": []}
            repr_text = str(desc.get("repr") or name)
            if kind == "remote":
                if desc.get("callable"):
                    obj: Any = RemoteCallable(
                        self,
                        target,
                        name,
                        repr_text,
                        bool(desc.get("async")),
                    )
                else:
                    obj = RemoteNamespace(self, target, name, repr_text)
            elif kind == "value":
                obj = self.decode(desc.get("value"))
            elif kind == "local":
                obj = self._local(desc["local"])
            elif kind == "import":
                obj = self._import(desc["spec"])
            else:
                obj = _Refused(name, str(desc.get("reason") or "it cannot cross"))
            self.ns[name] = obj
            self.installed[name] = obj


    def help(self, obj: Any = _NO_ARGUMENT) -> None:
        """Print the documentation of *obj*: a harness object's from the
        harness, anything else's as ``pydoc`` renders it."""
        if obj is _NO_ARGUMENT:
            text, _ = self.request_sync("doc", target=None, label="")
        elif isinstance(obj, _Remote):
            text, _ = self.request_sync(
                "doc",
                target=object.__getattribute__(obj, "_target"),
                label=object.__getattribute__(obj, "_label"),
            )
        else:
            import pydoc

            if isinstance(obj, _LazyModule):
                obj = obj._load()
            text = pydoc.render_doc(obj, title="%s", renderer=pydoc.plaintext)
        self._write(self._stdout, str(text).rstrip("\n") + "\n")

    # -- cells -----------------------------------------------------------------
    def handle(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "exec":
            assert self.loop is not None
            self._running = (msg.get("id"), self.loop.create_task(self.run_cell(msg)))

    def load_named(self, source: str) -> None:
        """Import each lazy module whose name the cell's code uses (:func:`_cell_names`) before the cell runs, inside
        the cell: under its clock and its audit, as the first use would. The cell then sees the module itself wherever
        it uses the name, so ``help(memory)``, ``memory.__doc__`` or ``type(memory)`` in the first cell that names it
        describe the module, never the stand-in. A failed import leaves the name lazy: the cell's own use of it
        raises, as before."""
        if not self._lazy:
            return
        names = _cell_names(source)
        for name in self._lazy:
            held = self.ns.get(name)
            if name in names and isinstance(held, _LazyModule):
                try:
                    held._load()
                except ImportError:
                    pass  # the cell's own use raises it, chained to the cause

    async def run_cell(self, msg: dict) -> None:
        cid = msg.get("id")
        self._describe_cache.clear()
        importlib.invalidate_caches()  # packages the harness installed meanwhile
        self._stdout, self._stderr = [], []
        start = self._fd_offset()
        result: Any = None
        error: Optional[str] = None
        shown_error: Optional[str] = None
        try:
            if self.sweep_roots:
                try:
                    self.sweep_modules()
                except Exception:  # noqa: BLE001 - never the cell's error; the next cell sweeps again
                    pass
            self.apply_sync(msg.get("sync") or {})
            ns = self.ns
            try:
                if self.audit is not None:
                    self._audit("begin")
                self.load_named(msg["source"])
                exec(compile(msg["source"], "<string>", "exec"), ns)
                result = await ns["__exec_wrapper"]()
            finally:
                if self.audit is not None:
                    self._audit("end")
                ns.pop("__exec_wrapper", None)
        except BaseException as exc:  # noqa: BLE001 - every failure is the cell's
            error = _format_cell_error(exc)
            # what the model reads: the same traceback without the harness's own
            # frames; the full one still goes to the harness for its logs
            shown_error = _format_cell_error(exc, _harness_files())
            result = None
        # the cell has ended: an interrupt that comes now finds nothing to stop
        self._running, self._interrupt_id = None, None
        try:
            encoded = self.encode_result(result)
        except Exception as exc:  # noqa: BLE001
            encoded = None
            error = error or f"The result could not be sent to the harness: {exc}"
        self._write(self._stdout, self._fd_output(start))
        self.send(
            {
                "op": "done",
                "id": cid,
                "result": encoded,
                "error": error,
                "shown_error": shown_error if shown_error != error else None,
                "stdout": self._stdout,
                "stderr": self._stderr,
                **({"audit": self._audit_drain()} if self.audit is not None else {}),
            },
        )


#: Where the host's cell wrapper (``unify.common.box_interpreter.wrap_cell``) puts the cell's own code: the string it
#: assigns to this name inside ``__exec_wrapper``, which it compiles and runs at the namespace's top level.
CELL_CODE_NAME = "__s"


def _cell_names(source: str) -> set:
    """The names a cell's code uses: the code the wrapper carries (:data:`CELL_CODE_NAME`; else *source* itself),
    parsed, never run, for every name it reads or binds. Code that does not parse names nothing."""
    import ast

    flags = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    try:
        tree = compile(source, "<cell>", "exec", flags, dont_inherit=True)
        code = next(
            (
                node.value.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == CELL_CODE_NAME
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ),
            None,
        )
        if code is not None:
            tree = compile(code, "<cell>", "exec", flags, dont_inherit=True)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def _harness_files() -> frozenset[str]:
    """The files of the harness's own namespaces in this box, whose frames a
    cell's reader cannot act on: the tools stub and its channel client, and
    the library package's own module and contract runtime. The files of the
    library's stored functions are not among them."""
    files = {"<tools>", "<tools: channel client>"}
    for name in ("tools", "memory", "memory._contract"):
        path = getattr(sys.modules.get(name), "__file__", None)
        if isinstance(path, str):
            files.add(path)
    return frozenset(files)


def _format_cell_error(exc: BaseException, hide: frozenset[str] = frozenset()) -> str:
    """The traceback without this file's frames (and without the frames of
    *hide*'s files): the cell's and the stored functions' frames are what
    the model can act on."""
    te = traceback.TracebackException.from_exception(exc)
    pending = [te]
    while pending:
        current = pending.pop()
        current.stack = traceback.StackSummary.from_list(
            [f for f in current.stack if f.filename != __file__ and f.filename not in hide],
        )
        pending += [e for e in (current.__cause__, current.__context__) if e]
    return "".join(te.format())


def _resolve(fut: asyncio.Future, msg: dict) -> None:
    if not fut.done():
        fut.set_result(msg)


def _run_coro_sync(coro: Any, timeout: Optional[float] = None) -> Any:
    """Drive *coro* to completion from synchronous code, under a running loop."""
    box: dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"run_coro_sync timed out after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


class _Capture(io.TextIOBase):
    def __init__(self, worker: Worker, which: str) -> None:
        self._worker = worker
        self._which = which

    def write(self, text: str) -> int:
        parts = self._worker._stdout if self._which == "out" else self._worker._stderr
        self._worker._write(parts, text)
        return len(text)

    def isatty(self) -> bool:
        return False


def main() -> None:
    # Named for the model reading a traceback, not for this file's run name.
    BoundaryRefusal.__module__ = "sandbox"
    # The channel moves to private descriptors; stdin reads EOF and whatever
    # reaches descriptors 1 and 2 (a subprocess, C code) lands in a private
    # file the worker returns with the cell's output.
    proto_in, proto_out = os.dup(0), os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    out_fd, out_path = tempfile.mkstemp(prefix="unify-worker-output-")
    os.unlink(out_path)
    os.dup2(out_fd, 1)
    os.dup2(out_fd, 2)
    rfile = os.fdopen(proto_in, "rb")
    wfile = os.fdopen(proto_out, "wb")
    worker = Worker(rfile, wfile, out_fd)
    sys.stdout = _Capture(worker, "out")  # type: ignore[assignment]
    sys.stderr = _Capture(worker, "err")  # type: ignore[assignment]
    first = rfile.readline()
    if not first:
        return
    missing = worker.init(json.loads(first))
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    worker.loop = loop
    signal.signal(INTERRUPT_SIGNAL, worker.on_interrupt)
    threading.Thread(target=worker.reader, daemon=True).start()
    ready = {"op": "ready", "missing": missing, "pid": os.getpid()}
    if worker.audit_state is not None:
        ready["audit"] = worker.audit_state
    worker.send(ready)
    loop.run_forever()


if __name__ == "__unify_worker__":
    main()
