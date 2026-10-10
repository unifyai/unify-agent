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
     "help": bool, "audit": {"roots": [...], "path": str}, "no_bytecode": true}
                                              (audit, no_bytecode: only under UNIFY_MEMORY_V2)
    {"op": "exec", "id": n, "source": str, "sync": {...}, "scratch": bool,
     "inventory": true}                       (only when asked)
    {"op": "reply", "id": k, "value": ..., "coroutine": bool}
    {"op": "reply", "id": k, "error": {"type", "module", "message", "args"}}

Worker -> harness::

    {"op": "ready", "missing": {name: error}, "audit": "on" | error}
                                              (audit: only when init asked)
    {"op": "call" | "describe" | "dir", "id": k, "target": {...}, ...}
    {"op": "fn_begin", "id": k, "name": str, "mode": "run" | "call",
     "args": [...], "kwargs": {...}}
    {"op": "fn_end", "id": k, "token": int, "result": ..., "error": str | None,
     "abandoned": bool}
    {"op": "doc", "id": k, "target": {...} | None, "label": str}
    {"op": "note", "event": str, ...}           (no reply)

``fn_begin``/``fn_end``/``doc`` and the ``help`` flag exist only under
the core tool surface (unify/actor/core_surface.py): a stored function
run by ``functions.run``, or called by name, runs here and the harness records
the call between its begin and its end. While one runs, every request carries
``"cases": [token, ...]``, the recordings it belongs to, so the harness adds
the environment calls it serves to those cases.
    {"op": "done", "id": n, "result": ..., "error": str | None, ...,
     "inventory": str | None,                 (when the exec asked for it)
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
import re
import sys
import tempfile
import threading
import traceback
import types
from typing import Any, Callable, Optional

TAG = "__unify__"
#: The module whose dict is the session's globals (``__name__`` of every cell).
SESSION_MODULE = "__sandbox_worker__"
MAX_DEPTH = 64
MAX_REPR = 500
MAX_FD_OUTPUT = 256 * 1024

#: The states ``functions.run`` takes (as ``execute_function`` did).
RUN_STATES = ("stateless", "stateful", "read_only")

# The core tool surface: the recordings (harness tokens) of the stored-
# function calls running in this context, outermost first.
_CASES: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "unify_worker_cases",
    default=(),
)

__all__ = [
    "BoundaryRefusal",
    "CellReply",
    "Inventory",
    "Reply",
    "Request",
    "MAX_DEPTH",
    "TAG",
    "decode",
    "describe_value",
    "json_values",
    "encode",
    "short_repr",
]


class BoundaryRefusal(TypeError):
    """A value or attribute that cannot cross the worker boundary, named."""


# ---------------------------------------------------------------------------
# UNIFY_REPLY_CHANNEL=code+text: reply(text) from a cell
# ---------------------------------------------------------------------------
# Defined here, with the standard library only, so the worker and the
# in-process sandbox share one implementation (the harness side is
# unify/common/_async_tool/cell_reply.py).

#: Every stored function is compiled under ``<function:NAME>``
#: (unify/function_manager/source_labels.py).
STORED_FUNCTION_PREFIX = "<function:"


class CellReply(BaseException):
    """Raised by ``reply(text)``: the cell ends and *text* is the turn's reply.

    A ``BaseException``, so a cell's ``except Exception`` does not swallow it.
    """

    def __init__(self, text: str, from_value: bool) -> None:
        super().__init__("reply() ended the cell")
        self.text = text
        self.from_value = from_value


def stored_function_on_stack(frame: Any) -> Optional[str]:
    """The name of a stored function running below *frame*, if any."""
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(STORED_FUNCTION_PREFIX) and filename.endswith(">"):
            return filename[len(STORED_FUNCTION_PREFIX) : -1]
        frame = frame.f_back
    return None


class Reply:
    """reply(text): send ``text`` as your reply and end your turn.

    ``text`` must be a str; it is sent exactly as given, as if you had
    replied with it. The cell stops at the call (what it printed so far is
    kept) and no later step of the turn runs. One reply per turn; not
    available inside a stored function, which returns its result to the cell
    instead.
    """

    def __init__(self, precheck: Optional[Callable[[], None]] = None) -> None:
        self._precheck = precheck
        self._used = False

    def __call__(self, text: str) -> None:
        self._send(text, True, sys._getframe(1))

    def _literal_reply(self, text: str) -> None:
        """``reply("...")`` with the text written out in the call."""
        self._send(text, False, sys._getframe(1))

    def new_cell(self) -> None:
        self._used = False

    def _send(self, text: Any, from_value: bool, caller: Any) -> None:
        if not isinstance(text, str):
            raise TypeError(
                f"reply() takes a str, not {type(text).__name__}: pass the exact "
                "text of your reply, e.g. reply(str(value)) or "
                "reply(json.dumps(obj))",
            )
        name = stored_function_on_stack(caller)
        if name is not None:
            raise RuntimeError(
                f"reply() cannot be called inside a stored function ({name}): "
                "return the text from the function and call reply() in the cell",
            )
        if self._used:
            raise RuntimeError(
                "reply() was already called in this turn; a turn has one reply",
            )
        if self._precheck is not None:
            self._precheck()
        if self._precheck is None:
            # In the worker the harness checks the turn; this cell's own
            # second call is refused here.
            self._used = True
        raise CellReply(text, from_value)

    def __repr__(self) -> str:
        return "<reply(text): send text as your reply and end your turn>"


# ---------------------------------------------------------------------------
# UNIFY_BIND_REQUEST=on: the current request as ``request`` in a cell
# ---------------------------------------------------------------------------
# Defined here, with the standard library only, so the worker and the
# in-process sandbox read a request the same way (the harness side is
# unify/common/_async_tool/bound_request.py).

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


class Request:
    """request: the current request, read-only.

    ``request.text`` is the request's text, as you received it;
    ``request.data`` is the list of JSON objects and arrays it contains, in
    order of appearance. Each cell gets a fresh copy, so what a cell does to
    it does not reach a later cell. A variable of your own named ``request``
    is never replaced; after ``del request`` the next cell has the current
    request again.
    """

    __slots__ = ("_text", "_data")

    def __init__(self, text: str) -> None:
        object.__setattr__(self, "_text", str(text))
        object.__setattr__(self, "_data", json_values(self._text))

    @property
    def text(self) -> str:
        return self._text

    @property
    def data(self) -> list:
        return self._data

    def renewed(self) -> "Request":
        """A fresh copy, for the next cell."""
        return Request(self._text)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"request is read-only: request.{name} cannot be set; copy what you "
            "need into a variable of your own",
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"request is read-only: request.{name} cannot be deleted")

    def __repr__(self) -> str:
        count = len(self._data)
        return (
            f"<request: {len(self._text)} characters of text (request.text), "
            f"{count} JSON value{'' if count == 1 else 's'} in request.data>"
        )


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
# UNIFY_VARIABLE_INVENTORY=on: the variables a session's cells bound, in a line
# ---------------------------------------------------------------------------
# Defined here, with the standard library only, so a cell's namespace is
# described the same way in the worker and in the in-process sandbox (the
# harness side is unify/actor/execution/session.py).

INVENTORY_LABEL = "[variables]"
INVENTORY_NAMES = 12
INVENTORY_CHARS = 400
#: A scalar's or string's value is shown when its repr is at most this long.
_VALUE_CHARS = 24
#: An int this wide is described by its width, not its digits.
_BIG_INT_BITS = 128
#: A list or tuple this long is not scanned for equal rows.
_ROW_SCAN = 10_000
_SIGNATURE_CHARS = 40
#: The harness's names in a cell; never listed, even after a cell rebinds one.
HARNESS_NAMES = frozenset(
    {
        "primitives",
        "request",
        "reply",
        "record",
        "agents",
        "display",
        "functions",
        "guidance",
        "steering",
        "help",
        "install",
    },
)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell_written(value: Any) -> bool:
    """A function or class defined by cell code in a session's namespace,
    not a stored function the harness compiled there (``<function:NAME>``)."""
    module = getattr(value, "__module__", None)
    if not (isinstance(module, str) and module.startswith("__sandbox_")):
        return False
    if isinstance(value, type):
        return True
    try:
        code = getattr(inspect.unwrap(value), "__code__", None)
    except ValueError:
        return False
    filename = getattr(code, "co_filename", "")
    return isinstance(filename, str) and not filename.startswith(
        STORED_FUNCTION_PREFIX,
    )


def _signature(fn: Any) -> str:
    kind = "async function" if inspect.iscoroutinefunction(fn) else "function"
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return kind
    names = []
    for param in params:
        if param.kind is param.VAR_POSITIONAL:
            names.append(f"*{param.name}")
        elif param.kind is param.VAR_KEYWORD:
            names.append(f"**{param.name}")
        else:
            names.append(param.name)
    return f"{kind}({_clip(', '.join(names), _SIGNATURE_CHARS)})"


def _rows(value: Any) -> str:
    """``N``, or ``NxM`` for N rows (lists or tuples) of equal length M."""
    count = len(value)
    if 0 < count <= _ROW_SCAN and isinstance(value[0], (list, tuple)):
        width = len(value[0])
        if all(isinstance(row, (list, tuple)) and len(row) == width for row in value):
            return f"{count}x{width}"
    return str(count)


def _array(value: Any, cls: type) -> Optional[str]:
    """A numpy or pandas value: its shape, and its dtype where it has one."""
    shape = getattr(value, "shape", None)
    if not isinstance(shape, tuple) or not all(isinstance(n, int) for n in shape):
        return None
    if not shape:
        return f"{cls.__name__} = {_clip(str(value), _VALUE_CHARS)}"
    text = f"{cls.__name__}[{'x'.join(str(n) for n in shape)}]"
    if hasattr(cls, "dtype"):
        text += f" {getattr(value, 'dtype', '')}"
    return text


def describe_value(value: Any) -> Optional[str]:
    """*value*'s type and a short shape, without printing it whole.

    A scalar, or a string whose repr is short, shows its value (``int =
    3``, ``str = 'abc'``); a longer string, bytes, a list, tuple or set its
    length (``str[120]``, ``list[3]``), a list or tuple of equal rows its
    rows and columns (``list[10x10]``), a dict its key count (``dict[4
    keys]``), a numpy or pandas value its shape and dtype; a function or
    class a cell defined its kind (``function(grid, k)``, ``class``); any
    other object its type's name. None: not a variable to list (a module, or
    a callable the cell did not define).
    """
    cls = type(value)
    if inspect.ismodule(value):
        return None
    if isinstance(value, type) or inspect.isfunction(value):
        if not _cell_written(value):
            return None
        return "class" if isinstance(value, type) else _signature(value)
    module = getattr(cls, "__module__", "") or ""
    if callable(value) and not module.startswith("__sandbox_"):
        return None
    name = cls.__name__
    if module.split(".")[0] in ("numpy", "pandas"):
        # Before the scalars: numpy's float64 is a float.
        return _array(value, cls) or name
    if value is None:
        return "None"
    if isinstance(value, int):
        if not isinstance(value, bool) and value.bit_length() > _BIG_INT_BITS:
            return f"{name} ({value.bit_length()} bits)"
        return f"{name} = {_clip(repr(value), _VALUE_CHARS)}"
    if isinstance(value, (float, complex)):
        return f"{name} = {_clip(repr(value), _VALUE_CHARS)}"
    if isinstance(value, str):
        if len(value) <= _VALUE_CHARS:
            text = repr(value)
            if len(text) <= _VALUE_CHARS + 2:
                return f"{name} = {text}"
        return f"{name}[{len(value)}]"
    if isinstance(value, (bytes, bytearray, set, frozenset)):
        return f"{name}[{len(value)}]"
    if isinstance(value, dict):
        return f"{name}[{len(value)} keys]"
    if isinstance(value, (list, tuple)):
        return f"{name}[{_rows(value)}]"
    return name


def render_inventory(entries: list) -> str:
    """The line for *entries* (``(name, description)``, in order)."""
    head = INVENTORY_LABEL + " "
    parts: list = []
    used = len(head)
    for index, (name, text) in enumerate(entries):
        if len(parts) == INVENTORY_NAMES:
            break
        piece = f"{name}: {text}"
        rest = len(entries) - index - 1
        tail = len(f"; …and {rest} more") if rest else 0
        cost = (2 if parts else 0) + len(piece)
        if used + cost + tail > INVENTORY_CHARS:
            break
        parts.append(piece)
        used += cost
    line = head + "; ".join(parts)
    more = len(entries) - len(parts)
    if more:
        line += ("; " if parts else "") + f"…and {more} more"
    return line


class Inventory:
    """The variables one session's cells bound, as a line for the cell's result.

    Around each cell the caller takes :meth:`snapshot` of the namespace and
    then calls :meth:`after_cell`: a name whose value the cell bound or
    rebound is the cell's, and its most recent binding (or change of shape)
    orders the line. A name the cell did not bind (the harness's) is never
    listed. The line is returned only when the listed names or their
    descriptions differ from the last line returned.
    """

    def __init__(self) -> None:
        self._cell = 0
        self._bound: dict = {}  # name -> the cell that last bound or reshaped it
        self._shapes: dict = {}  # name -> its description after that cell
        self._shown: frozenset = frozenset()

    @staticmethod
    def snapshot(ns: dict) -> dict:
        return {name: id(value) for name, value in list(ns.items())}

    def after_cell(
        self,
        ns: dict,
        before: dict,
        installed: Optional[dict] = None,
    ) -> Optional[str]:
        self._cell += 1
        for name, value in list(ns.items()):
            if not isinstance(name, str) or name.startswith("_"):
                continue
            if name in HARNESS_NAMES:
                continue
            if before.get(name) != id(value):
                self._bound[name] = self._cell
        entries: dict = {}
        for name in list(self._bound):
            if name not in ns:
                del self._bound[name]
                self._shapes.pop(name, None)
                continue
            value = ns[name]
            if installed and installed.get(name) is value:
                continue
            try:
                text = describe_value(value)
            except Exception:  # noqa: BLE001 - a value's own code failed
                text = type(value).__name__
            if text is None:
                continue
            if self._shapes.get(name, text) != text:
                self._bound[name] = self._cell
            self._shapes[name] = text
            entries[name] = text
        shown = frozenset(entries.items())
        if shown == self._shown:
            return None
        self._shown = shown
        if not entries:
            return None
        order = sorted(entries, key=lambda n: (-self._bound[n], n))
        return render_inventory([(n, entries[n]) for n in order])


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

    With ``record`` (the core tool surface) each call is recorded by the
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
    """``functions`` (the core tool surface): the harness's function
    library, except ``run``, which runs the stored code here, in the worker."""

    def __getattr__(self, name: str) -> Any:
        if name == "run":
            return self._w.run_function
        return super().__getattr__(name)


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
    if not (isinstance(value, type) and _cell_written(value)):
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
        self._exc_classes: dict[tuple[str, str], type] = {}
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
        # The globals every namespace starts from (init's), for the fresh
        # globals of ``functions.run(..., state="stateless")``.
        self.base_ns: dict[str, Any] = {}
        self._stdout: list[dict] = []
        self._stderr: list[dict] = []
        self._real_print = builtins.print
        # UNIFY_REPLY_CHANNEL=code+text: the cells' reply(); the harness
        # checks the turn when the cell reports its reply.
        self.reply = Reply()
        # UNIFY_VARIABLE_INVENTORY=on: what this namespace's cells bound.
        self.inventory = Inventory()
        # UNIFY_MEMORY_V2 (spec §3a): the work-tree audit hook, only when the
        # harness asks for it in init; ``audit_state`` is "on" or why not.
        self.audit: Any = None
        self.audit_state: Optional[str] = None

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
            # UNIFY_MEMORY_V2: importing the memory export writes no __pycache__ into it
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
        self.base_ns = dict(self.ns)
        return missing

    def _install_audit(self, spec: dict) -> str:
        """Install the memory-v2 audit hook (``unify/memory_v2/integration/
        adapters/audit.py``), loaded by its file path so no ``unify`` package
        code runs here; "on", or why it is not installed."""
        try:
            import importlib.util

            roots = [r for r in spec.get("roots") or [] if isinstance(r, str)]
            found = importlib.util.spec_from_file_location(
                "_unify_memory_v2_audit",
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
        if which == "around_cp":
            return self._around_cp
        if which == "run_coro_sync":
            return _run_coro_sync
        if which == "handle_class":
            return SteerableToolHandle
        if which == "reply":
            return self.reply
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
            current = self.ns.get(name)
            if obj is not None and (
                current is obj
                # UNIFY_BIND_REQUEST=on: any harness Request, as in process.
                or (isinstance(obj, Request) and isinstance(current, Request))
            ):
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
            elif kind == "request":
                # UNIFY_BIND_REQUEST=on: renewed before each cell, and bound
                # only where the model has no ``request`` of its own.
                obj = Request(str(desc.get("text") or ""))
                self.installed[name] = obj
                if self._request_unclaimed(name):
                    self.ns[name] = obj
                continue
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

    def _request_unclaimed(self, name: str) -> bool:
        """The name is unbound or holds a harness ``Request``, not a variable
        of the model's own."""
        return name not in self.ns or isinstance(self.ns[name], Request)

    def renew_requests(self) -> None:
        """UNIFY_BIND_REQUEST=on: each cell gets a fresh ``request``, so what
        a cell did to it does not last; a variable of the model's own named
        ``request`` is left as it is."""
        for name, obj in list(self.installed.items()):
            if isinstance(obj, Request):
                fresh = obj.renewed()
                self.installed[name] = fresh
                if self._request_unclaimed(name):
                    self.ns[name] = fresh

    def variables(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in self.ns.items():
            if not isinstance(name, str) or name.startswith("_"):
                continue
            if name in self.installed or callable(value) or inspect.ismodule(value):
                continue
            out[name] = short_repr(value)
        return out

    # -- stored functions (the core tool surface) ----------------------------
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
            self.request_sync("fn_end", **self._end_fields(token, error=exc))
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
            await self.request_async("fn_end", **self._end_fields(token, error=exc))
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

    def _define_helpers(self, helpers: list, ns: dict) -> None:
        """Define the stored functions a ``functions.run`` entry point calls in
        the namespace it runs in, each recorded when called, as a read that
        loads the entry point binds them. All are defined before any is
        wrapped, so they find one another by name."""
        raws = []
        for helper in helpers:
            name = str(helper.get("name"))
            source = str(helper.get("source") or "")
            filename = str(helper.get("filename") or f"<function:{name}>")
            linecache.cache[filename] = (
                len(source),
                None,
                source.splitlines(keepends=True),
                filename,
            )
            exec(compile(source, filename, "exec"), ns)
            if callable(ns.get(name)):
                raws.append((name, ns[name]))
        for name, raw in raws:
            ns[name] = _StoredFunction(raw, name, self, record=True)

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
        # The harness may just have installed its dependencies.
        importlib.invalidate_caches()
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
            self._define_helpers(reply.get("helpers") or [], ns)
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
            await self.request_async("fn_end", **self._end_fields(token, error=exc))
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
        reply: Optional[dict] = None
        self.reply.new_cell()
        # UNIFY_VARIABLE_INVENTORY=on: asked only for a cell that keeps
        # what it binds.
        listing = bool(msg.get("inventory")) and not msg.get("scratch")
        inventory: Optional[str] = None
        try:
            self.apply_sync(msg.get("sync") or {})
            self.renew_requests()
            ns = dict(self.ns) if msg.get("scratch") else self.ns
            before = Inventory.snapshot(ns) if listing else None
            try:
                if self.audit is not None:
                    self._audit("begin")
                exec(compile(msg["source"], "<string>", "exec"), ns)
                result = await ns["__exec_wrapper"]()
            finally:
                if self.audit is not None:
                    self._audit("end")
                ns.pop("__exec_wrapper", None)
                if before is not None:
                    try:
                        inventory = self.inventory.after_cell(
                            ns,
                            before,
                            self.installed,
                        )
                    except Exception:  # noqa: BLE001 - never the cell's error
                        inventory = None
        except CellReply as replied:
            reply = {"text": replied.text, "from_value": replied.from_value}
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
                **({"reply": reply} if reply is not None else {}),
                **({"inventory": inventory} if listing else {}),
                **({"audit": self._audit_drain()} if self.audit is not None else {}),
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
    ready = {"op": "ready", "missing": missing, "pid": os.getpid()}
    if worker.audit_state is not None:
        ready["audit"] = worker.audit_state
    worker.send(ready)
    loop.run_forever()


if __name__ == "__unify_worker__":
    main()
