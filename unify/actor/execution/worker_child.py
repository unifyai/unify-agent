"""The child half of the sandboxed Python worker (``UNIFY_WORKSPACE_PYTHON=worker``).

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
     "help": bool}
    {"op": "exec", "id": n, "source": str, "sync": {...}, "scratch": bool}
    {"op": "reply", "id": k, "value": ..., "coroutine": bool}
    {"op": "reply", "id": k, "error": {"type", "module", "message", "args"}}

Worker -> harness::

    {"op": "ready", "missing": {name: error}}
    {"op": "call" | "describe" | "dir", "id": k, "target": {...}, ...}
    {"op": "fn_begin", "id": k, "name": str, "mode": "run" | "call",
     "args": [...], "kwargs": {...}}
    {"op": "fn_end", "id": k, "token": int, "result": ..., "error": str | None,
     "abandoned": bool}
    {"op": "doc", "id": k, "target": {...} | None, "label": str}
    {"op": "note", "event": str, ...}           (no reply)

``fn_begin``/``fn_end``/``doc`` and the ``help`` flag exist only under
``UNIFY_TOOL_SURFACE=core`` (unify/actor/core_surface.py): a stored function
run by ``functions.run``, or called by name, runs here and the harness records
the call between its begin and its end. While one runs, every request carries
``"cases": [token, ...]``, the recordings it belongs to, so the harness adds
the environment calls it serves to those cases.
    {"op": "done", "id": n, "result": ..., "error": str | None, ...}

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
import contextvars
import datetime as _dt
import decimal
import importlib
import inspect
import io
import itertools
import json
import linecache
import os
import sys
import tempfile
import threading
import traceback
from typing import Any, Callable, Optional

TAG = "__unify__"
MAX_DEPTH = 64
MAX_REPR = 500
MAX_FD_OUTPUT = 256 * 1024

#: The states ``functions.run`` takes (as ``execute_function`` did).
RUN_STATES = ("stateless", "stateful", "read_only")

# UNIFY_TOOL_SURFACE=core: the recordings (harness tokens) of the stored-
# function calls running in this context, outermost first.
_CASES: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "unify_worker_cases",
    default=(),
)

__all__ = [
    "BoundaryRefusal",
    "MAX_DEPTH",
    "TAG",
    "decode",
    "encode",
    "short_repr",
]


class BoundaryRefusal(TypeError):
    """A value or attribute that cannot cross the worker boundary, named."""


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


class SteerableToolHandle:
    """Stand-in for the harness's handle base class.

    Handles live in the harness. One a harness call returns arrives as a
    proxy that is an instance of this class, so ``isinstance`` checks hold;
    a handle defined in the worker cannot be handed to the harness.
    """


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


class RemoteHandle(RemoteNamespace, SteerableToolHandle):
    """A harness steerable handle."""


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


class _StoredFunction:
    """A stored function defined in the worker; tells the harness it ran.

    With ``record`` (``UNIFY_TOOL_SURFACE=core``) each call is recorded by the
    harness as the in-process boundary wrapper records it -- usage, trust
    evidence, a case with the environment calls it makes -- instead of only
    being noted as used.
    """

    def __init__(
        self,
        raw: Callable[..., Any],
        name: str,
        worker: "Worker",
        record: bool = False,
    ) -> None:
        self.__wrapped__ = raw
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = getattr(raw, "__doc__", None)
        self._worker = worker
        self._record = record

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._record:
            return self._worker.call_recorded(self, args, kwargs)
        self._worker.note("function_used", name=self.__name__)
        return self.__wrapped__(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__wrapped__, name)

    def __repr__(self) -> str:
        return f"<stored function {self.__name__}>"


#: Steering position probes. They return nothing and only feed progress
#: reports, so the worker posts them without waiting for an answer.
RUNTIME_NOTES = frozenset(
    {
        "increment_loop_iteration",
        "start_loop_context",
        "end_loop_context",
        "push_path_context",
        "pop_path_context",
    },
)


class _RuntimeProxy(RemoteNamespace):
    def __getattr__(self, name: str) -> Any:
        if name in RUNTIME_NOTES:
            worker = self._w

            def _post(*args: Any) -> None:
                worker.note("runtime", method=name, args=list(args))

            return _post
        return super().__getattr__(name)


class _FunctionsProxy(RemoteNamespace):
    """``functions`` (``UNIFY_TOOL_SURFACE=core``): the harness's function
    library, except ``run``, which runs the stored code here, in the worker."""

    def __getattr__(self, name: str) -> Any:
        if name == "run":
            return self._w.run_function
        return super().__getattr__(name)


_NO_ARGUMENT = object()


async def _ready(value: Any) -> Any:
    return value


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
        self._exc_classes: dict[tuple[str, str], type] = {}
        self._out_fd = out_fd
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.ns: dict[str, Any] = {}
        self.installed: dict[str, Any] = {}
        # The globals every namespace starts from (init's), for the fresh
        # globals of ``functions.run(..., state="stateless")``.
        self.base_ns: dict[str, Any] = {}
        self._stdout: list[dict] = []
        self._stderr: list[dict] = []
        self._real_print = builtins.print

    # -- channel -------------------------------------------------------------
    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._wlock:
            self._wfile.write(data)
            self._wfile.flush()

    def note(self, event: str, **fields: Any) -> None:
        self.send({"op": "note", "event": event, **fields})

    def reader(self) -> None:
        for line in self._rfile:
            try:
                msg = json.loads(line)
            except ValueError:
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

    @staticmethod
    def _with_cases(fields: dict) -> dict:
        cases = _CASES.get()
        return {**fields, "cases": list(cases)} if cases else fields

    def request_sync(self, op: str, **fields: Any) -> tuple[Any, dict]:
        rid = next(self._ids)
        event = threading.Event()
        box: list = []
        self._pending[rid] = ("sync", event, box)
        self.send({"op": op, "id": rid, **self._with_cases(fields)})
        event.wait()
        return self._unwrap(box[0]), box[0]

    async def request_async(self, op: str, **fields: Any) -> tuple[Any, dict]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        rid = next(self._ids)
        self._pending[rid] = ("async", loop, fut)
        try:
            self.send({"op": op, "id": rid, **self._with_cases(fields)})
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
            if payload.get("handle"):
                return RemoteHandle(self, target, label, repr_text)
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
        target = {**base, "path": [*base["path"], name]}
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
            return self._call_async(fields)
        value, msg = self.request_sync("call", **fields)
        return _ready(value) if msg.get("coroutine") else value

    async def _call_async(self, fields: dict) -> Any:
        value, _ = await self.request_async("call", **fields)
        return value

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
        sys.path[:] = [p for p in msg.get("sys_path", []) if isinstance(p, str) and p]
        names = msg.get("builtins") or []
        safe = {n: getattr(builtins, n) for n in names if hasattr(builtins, n)}
        if "print" in safe:
            safe["print"] = self._print
        self.ns["__builtins__"] = safe
        self.ns["__name__"] = "__sandbox_worker__"
        missing: dict[str, str] = {}
        for name, spec in (msg.get("globals") or {}).items():
            value = self._import(spec)
            if isinstance(value, _Refused):
                missing[name] = value._reason
            else:
                self.ns[name] = value
        if msg.get("help"):
            # Under -S there is no site-installed help(); this one prints the
            # harness objects' documentation too.
            self.ns["help"] = self.help
        self.base_ns = dict(self.ns)
        return missing

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
        if which == "around_cp":
            return self._around_cp
        if which == "run_coro_sync":
            return _run_coro_sync
        if which == "handle_class":
            return SteerableToolHandle
        raise BoundaryRefusal(f"unknown worker-local name {which!r}")

    async def _around_cp(self, label: str, awaitable: Any) -> Any:
        cp = self.ns.get("_cp")
        await cp(f"Before: {label}")
        try:
            if inspect.isawaitable(awaitable):
                return await awaitable
            return awaitable
        finally:
            await cp(f"After: {label}")

    def _define_function(
        self,
        name: str,
        source: str,
        filename: str,
        record: bool = False,
    ) -> Any:
        lines = source.splitlines(keepends=True)
        linecache.cache[filename] = (len(source), None, lines, filename)
        try:
            exec(compile(source, filename, "exec"), self.ns)
        except Exception as exc:  # noqa: BLE001 - surfaced when it is used
            return _Refused(
                name,
                f"defining it in the worker failed ({type(exc).__name__}: {exc})",
            )
        raw = self.ns.get(name)
        if not callable(raw):
            return _Refused(name, "its source does not define it")
        return _StoredFunction(raw, name, self, record=record)

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
                if desc.get("runtime"):
                    obj: Any = _RuntimeProxy(self, target, name, repr_text)
                elif desc.get("library") == "functions":
                    obj = _FunctionsProxy(self, target, name, repr_text)
                elif desc.get("callable"):
                    obj = RemoteCallable(
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
            elif kind == "function":
                obj = self._define_function(
                    name,
                    desc["source"],
                    desc["filename"],
                    record=bool(desc.get("record")),
                )
            elif kind == "local":
                obj = self._local(desc["local"])
            elif kind == "import":
                obj = self._import(desc["spec"])
            elif kind == "model":
                current = self.ns.get(name)
                if name in self.ns and self.installed.get(name) is not current:
                    continue  # the worker's own definition stands
                obj = _Refused(name, str(desc.get("reason") or "model-written"))
            else:
                obj = _Refused(name, str(desc.get("reason") or "it cannot cross"))
            self.ns[name] = obj
            self.installed[name] = obj

    def variables(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in self.ns.items():
            if not isinstance(name, str) or name.startswith("_"):
                continue
            if name in self.installed or callable(value) or inspect.ismodule(value):
                continue
            out[name] = short_repr(value)
        return out

    # -- stored functions (UNIFY_TOOL_SURFACE=core) ----------------------------
    def _record_value(self, value: Any) -> Any:
        """*value* for the harness's recording: data as data, anything else
        as its repr, so recording never refuses a call."""
        return self.encode_result(value)

    def _begin_fields(self, name: str, mode: str, args: tuple, kwargs: dict) -> dict:
        return {
            "name": name,
            "mode": mode,
            "args": [self._record_value(a) for a in args],
            "kwargs": {str(k): self._record_value(v) for k, v in kwargs.items()},
        }

    def _end_fields(
        self,
        token: int,
        result: Any = None,
        error: Optional[BaseException] = None,
    ) -> dict:
        if error is None:
            return {"token": token, "result": self._record_value(result)}
        if not isinstance(error, Exception):
            # Cancelled or stopped: the call says nothing about the function.
            return {"token": token, "abandoned": True}
        return {"token": token, "error": _format_cell_error(error)}

    @staticmethod
    def _add_note(error: BaseException, reply: Any) -> None:
        note = reply.get("note") if isinstance(reply, dict) else None
        if note and isinstance(error, Exception):
            try:
                error.add_note(str(note))
            except Exception:  # noqa: BLE001 - a note never replaces the error
                pass

    def call_recorded(self, fn: "_StoredFunction", args: tuple, kwargs: dict) -> Any:
        """A direct call of a stored function, recorded by the harness as the
        in-process boundary wrapper records one."""
        raw = fn.__wrapped__
        reply, _ = self.request_sync(
            "fn_begin",
            **self._begin_fields(fn.__name__, "call", args, kwargs),
        )
        token = reply.get("token") if isinstance(reply, dict) else None
        if token is None:
            return raw(*args, **kwargs)
        mark = _CASES.set(_CASES.get() + (token,))
        try:
            result = raw(*args, **kwargs)
        except BaseException as exc:
            _CASES.reset(mark)
            self._add_note(
                exc,
                self.request_sync("fn_end", **self._end_fields(token, error=exc))[0],
            )
            raise
        _CASES.reset(mark)
        if inspect.isawaitable(result):
            return self._finish_recorded(token, result)
        self.request_sync("fn_end", **self._end_fields(token, result=result))
        return result

    async def _finish_recorded(self, token: int, awaitable: Any) -> Any:
        mark = _CASES.set(_CASES.get() + (token,))
        try:
            value = await awaitable
        except BaseException as exc:
            _CASES.reset(mark)
            reply, _ = await self.request_async(
                "fn_end",
                **self._end_fields(token, error=exc),
            )
            self._add_note(exc, reply)
            raise
        _CASES.reset(mark)
        await self.request_async("fn_end", **self._end_fields(token, result=value))
        return value

    def _resolve_name(self, name: str) -> Any:
        """``name`` (dotted for ``primitives.*``) in this worker's namespace."""
        head, *rest = str(name).split(".")
        if head not in self.ns:
            raise NameError(
                f"name {name!r} is not a stored function and not defined in "
                "this session",
            )
        value = self.ns[head]
        for part in rest:
            value = getattr(value, part)
        return value

    async def run_function(
        self,
        name: str,
        /,
        *,
        state: str = "stateless",
        **kwargs: Any,
    ) -> Any:
        """``functions.run``: call a stored function by name, here, recorded."""
        if state not in RUN_STATES:
            raise ValueError(
                f"state must be one of {list(RUN_STATES)}, not {state!r}",
            )
        reply, _ = await self.request_async(
            "fn_begin",
            **self._begin_fields(str(name), "run", (), kwargs),
        )
        token = reply.get("token") if isinstance(reply, dict) else None
        if token is None:
            # A primitive, or a function this session defined: called as it is.
            out = self._resolve_name(name)(**kwargs)
            return (await out) if inspect.isawaitable(out) else out
        fn_name = str(reply.get("fn_name") or name)
        source = str(reply.get("source") or "")
        filename = str(reply.get("filename") or f"<function:{fn_name}>")
        if state == "stateful":
            ns = self.ns
        elif state == "read_only":
            ns = dict(self.ns)
        else:
            ns = {**self.base_ns, **self.installed}
        mark = _CASES.set(_CASES.get() + (token,))
        try:
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(keepends=True),
                filename,
            )
            exec(compile(source, filename, "exec"), ns)
            fn = ns.get(fn_name)
            if not callable(fn):
                raise NameError(f"the stored source does not define {fn_name!r}")
            out = fn(**kwargs)
            if inspect.isawaitable(out):
                out = await out
        except BaseException as exc:
            _CASES.reset(mark)
            end, _ = await self.request_async(
                "fn_end",
                **self._end_fields(token, error=exc),
            )
            self._add_note(exc, end)
            raise
        _CASES.reset(mark)
        await self.request_async("fn_end", **self._end_fields(token, result=out))
        if state == "stateful" and ns.get(fn_name) is fn:
            # Defined in the session now; later calls by name are recorded too.
            ns[fn_name] = _StoredFunction(fn, fn_name, self, record=True)
        return out

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
        elif getattr(obj, "__func__", None) is Worker.run_function:
            text, _ = self.request_sync(
                "doc",
                target={"root": "functions", "path": ["run"]},
                label="functions.run",
            )
        else:
            import pydoc

            target = obj.__wrapped__ if isinstance(obj, _StoredFunction) else obj
            text = pydoc.render_doc(target, title="%s", renderer=pydoc.plaintext)
        self._write(self._stdout, str(text).rstrip("\n") + "\n")

    # -- cells -----------------------------------------------------------------
    def handle(self, msg: dict) -> None:
        op = msg.get("op")
        if op == "exec":
            assert self.loop is not None
            self.loop.create_task(self.run_cell(msg))
        elif op == "variables":
            self.send(
                {"op": "done", "id": msg.get("id"), "variables": self.variables()},
            )

    async def run_cell(self, msg: dict) -> None:
        cid = msg.get("id")
        self._describe_cache.clear()
        importlib.invalidate_caches()  # packages the harness installed meanwhile
        self._stdout, self._stderr = [], []
        start = self._fd_offset()
        result: Any = None
        error: Optional[str] = None
        error_type: Optional[str] = None
        remote = False
        message = ""
        try:
            self.apply_sync(msg.get("sync") or {})
            ns = dict(self.ns) if msg.get("scratch") else self.ns
            try:
                exec(compile(msg["source"], "<string>", "exec"), ns)
                result = await ns["__exec_wrapper"]()
            finally:
                ns.pop("__exec_wrapper", None)
        except BaseException as exc:  # noqa: BLE001 - every failure is the cell's
            error = _format_cell_error(exc)
            error_type = type(exc).__name__
            remote = isinstance(exc, RemoteError)
            message = str(exc)
            result = None
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
                "error_type": error_type,
                "remote": remote,
                "message": message,
                "stdout": self._stdout,
                "stderr": self._stderr,
            },
        )


def _format_cell_error(exc: BaseException) -> str:
    """The traceback without this file's frames: the cell's and the stored
    functions' frames are what the model can act on."""
    te = traceback.TracebackException.from_exception(exc)
    pending = [te]
    while pending:
        current = pending.pop()
        current.stack = traceback.StackSummary.from_list(
            [f for f in current.stack if f.filename != __file__],
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
    threading.Thread(target=worker.reader, daemon=True).start()
    worker.send({"op": "ready", "missing": missing, "pid": os.getpid()})
    loop.run_forever()


if __name__ == "__unify_worker__":
    main()
