"""Adapter 1, the tool kind (spec §C2): environment calls recorded through the observer seam.

:class:`RecordingObserver` implements ``unify.function_manager.primitives.observers.EnvObserver``. It only
records. ``before()`` always returns ``None``, so it never answers a call. ``complete`` is ``False``, so it
never makes the raw-global proxy refuse an access it cannot route. Each call that ran becomes one
``Action(kind="tool")``:

- ``channel``: the primitives namespace. For a raw-global call (``via == "global"``, e.g.
  ``apis.spotify.login``) it is the first attribute (the app, ``spotify``), and ``method`` is the rest.
- ``args``, ``kwargs`` and ``response``: JSON-safe deep copies (results arrive by reference). Each is
  built under a node and character budget, with cycles cut and every string redacted before any cut,
  and then the whole value is capped in size.
- ``status`` ``ok`` or ``error``; ``error`` is the exception type and the first line of its message.
- ``effect``: the declared effect, ``unknown`` when undeclared (every raw-global call).

``after()`` runs synchronously on the agent's call path, so all of its work is bounded and it never
raises; a failure to record is counted. Calls another observer intercepted are not observed evidence
and are not recorded. At most ``max_calls`` actions are kept; the rest are counted.

The harness tells the observer which cell is running with :meth:`RecordingObserver.set_cell`. A call is
attributed to the cell that was current when it started (``before()``). Cells are assumed to run one at
a time per observer, so create one observer per request.

Nothing here is wired into the actor.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from unify.memory_v2.episodes import Action
from unify.memory_v2.fingerprint import fingerprint, shape
from unify.memory_v2.redact import Redactor

__all__ = [
    "BUDGET",
    "CYCLE",
    "TRUNCATED",
    "RecordingObserver",
    "ToolDrain",
    "action_fingerprints",
    "error_text",
    "jsonable",
    "split_channel",
]

#: Key of the marker that replaces a value over the size cap.
TRUNCATED = "__truncated__"
#: Marker for a container met again on its own path (a reference cycle).
CYCLE = {"__cycle__": True}
#: Marker for whatever is left once the node or character budget is spent.
BUDGET = {"__more__": "budget"}

_MAX_DEPTH = 8
_MAX_ITEMS = 1000
_MAX_NODES = 10_000
_MAX_CHARS = 65_536
_REPR_CHARS = 2000


def split_channel(call: Any) -> tuple[str, str]:
    """``(channel, method)``: the app of a dotted raw-global call, otherwise the namespace."""
    method = str(call.method)
    if call.via == "global" and "." in method:
        app, rest = method.split(".", 1)
        return app, rest
    return str(call.namespace), method


def _safe_cut(redactor: Redactor, text: str, limit: int) -> str:
    """``text`` redacted as a whole, then cut to ``limit`` characters.

    The whole string is redacted, not a window past the cut: each replacement shortens the text, so a
    window can come up short of the cut and let a secret that straddles the window's end through. The
    string is already in memory and redaction is linear in its length."""
    return redactor.text(text)[:limit]


class _Walk:
    """One bounded, redacting deep copy: a shared node and character budget and a cycle guard."""

    def __init__(self, redactor: Redactor, *, max_nodes: int, max_chars: int) -> None:
        self.redactor = redactor
        self.nodes = max_nodes
        self.chars = max_chars
        self.path: set[int] = set()

    def spent(self) -> bool:
        return self.nodes <= 0 or self.chars <= 0

    def text(self, s: str, limit: int | None = None) -> str:
        cut = min(len(s), self.chars if limit is None else min(limit, self.chars))
        out = _safe_cut(self.redactor, s, max(cut, 0))
        self.chars -= len(out)
        return out

    def repr_of(self, value: Any) -> dict:
        try:
            text = repr(value)
        except Exception:  # noqa: BLE001 - a hostile __repr__ must not break recording
            text = f"<{type(value).__name__}>"
        return {"__repr__": self.text(text, _REPR_CHARS)}

    def copy(self, value: Any, depth: int) -> Any:
        if self.spent():
            return dict(BUDGET)
        self.nodes -= 1
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            finite = value == value and value not in (float("inf"), float("-inf"))
            return value if finite else repr(value)
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"__bytes__": len(value)}
        container = isinstance(value, (dict, list, tuple, set, frozenset))
        if depth <= 0:
            # Never repr a container: its repr is as unbounded as the walk (a shared DAG).
            if container:
                return {"__depth__": type(value).__name__, "len": len(value)}
            return self.repr_of(value)
        if container:
            key = id(value)
            if key in self.path:
                return dict(CYCLE)
            self.path.add(key)
            try:
                return self._container(value, depth)
            finally:
                self.path.discard(key)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            try:
                fields = {
                    f.name: getattr(value, f.name) for f in dataclasses.fields(value)
                }
            except Exception:  # noqa: BLE001 - fall back to repr
                return self.repr_of(value)
            return self.copy(fields, depth)
        try:
            dump = getattr(value, "model_dump", None)
            if callable(dump):
                return self.copy(dump(mode="json"), depth - 1)
        except Exception:  # noqa: BLE001 - fall back to repr
            pass
        return self.repr_of(value)

    def _container(self, value: Any, depth: int) -> Any:
        total = len(value)
        if isinstance(value, dict):
            out: dict = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= _MAX_ITEMS or self.spent():
                    out["__more__"] = total - i
                    break
                if isinstance(k, str):
                    name = self.text(k, _REPR_CHARS)
                else:
                    name = self.repr_of(k)["__repr__"]
                if name in out:  # two keys with the same text: keep both
                    name = f"{name}#{i}"
                out[name] = self.copy(v, depth - 1)
            return out
        if isinstance(value, (set, frozenset)):
            items = list(value)[:_MAX_ITEMS]
            try:
                items.sort(key=repr)
            except Exception:  # noqa: BLE001 - unordered is fine
                pass
        else:
            items = value[:_MAX_ITEMS]
        out_list = []
        for i, v in enumerate(items):
            if self.spent():
                out_list.append({"__more__": total - i})
                return out_list
            out_list.append(self.copy(v, depth - 1))
        if total > len(items):
            out_list.append({"__more__": total - len(items)})
        return out_list


def jsonable(
    value: Any,
    depth: int = _MAX_DEPTH,
    *,
    redactor: Redactor | None = None,
    max_nodes: int = _MAX_NODES,
    max_chars: int = _MAX_CHARS,
) -> Any:
    """A redacted, JSON-safe deep copy of ``value``.

    Containers are rebuilt, pydantic models dumped and other objects replaced by a bounded ``repr``. The
    copy is bounded in depth, in items per container, in total nodes and in total string characters. A
    container met again on its own path becomes ``{"__cycle__": true}``. Strings are redacted before any
    cut."""
    walk = _Walk(
        redactor if redactor is not None else Redactor(),
        max_nodes=max_nodes,
        max_chars=max_chars,
    )
    return walk.copy(value, depth)


def error_text(
    error: BaseException,
    limit: int = 500,
    redactor: Redactor | None = None,
) -> str:
    """The exception's type and the first line of its message, redacted and then bounded."""
    redactor = redactor if redactor is not None else Redactor()
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - a hostile __str__ still gives a record
        message = "<unprintable>"
    # Redacted before the first line is taken, so the first line of a multi-line secret is not kept.
    lines = redactor.text(message).strip().splitlines()
    first = lines[0] if lines else ""
    text = type(error).__name__ + (f": {first}" if first else "")
    return _safe_cut(redactor, text, limit)


@dataclass
class ToolDrain:
    """What :meth:`RecordingObserver.drain` hands over: the actions and the counts since the last drain."""

    actions: list[Action] = field(default_factory=list)
    dropped: int = 0
    """Calls that ran but were not kept because ``max_calls`` was reached."""
    intercepted: int = 0
    """Calls another observer answered (not recorded)."""
    failed: int = 0
    """Calls that could not be recorded (an internal error, swallowed)."""


class RecordingObserver:
    """Records every environment call that ran as a tool action; never intercepts and never raises."""

    complete = False

    def __init__(
        self,
        redactor: Redactor | None = None,
        *,
        max_calls: int = 2000,
        max_value_bytes: int = 16384,
        max_error_chars: int = 500,
        max_nodes: int = _MAX_NODES,
    ) -> None:
        self.redactor = redactor if redactor is not None else Redactor()
        self.max_calls = max_calls
        self.max_value_bytes = max_value_bytes
        self.max_error_chars = max_error_chars
        self.max_nodes = max_nodes
        self._cell = 0
        self._started_in: dict[int, int] = {}
        self._pending = ToolDrain()
        self._lock = threading.Lock()

    # -- harness side -------------------------------------------------------------------------------

    def set_cell(self, index: int) -> None:
        """The index of the cell now running; calls that start from now on are attributed to it."""
        with self._lock:
            self._cell = int(index)

    @property
    def cell(self) -> int:
        return self._cell

    def drain(self) -> ToolDrain:
        """The actions recorded so far (in call-completion order) and the counts; all are reset."""
        with self._lock:
            out, self._pending = self._pending, ToolDrain()
        return out

    # -- EnvObserver --------------------------------------------------------------------------------

    def before(self, call: Any) -> None:
        with self._lock:
            # Bounded: a call whose after() never comes (another observer's before() raised) leaks one
            # entry at most until the bound.
            if len(self._started_in) < 4 * self.max_calls:
                self._started_in[id(call)] = self._cell
        return None

    def after(
        self,
        call: Any,
        *,
        result: Any,
        error: BaseException | None,
        intercepted: bool,
        started: float,
        elapsed_s: float,
    ) -> None:
        with self._lock:
            cell = self._started_in.pop(id(call), self._cell)
            if intercepted:
                self._pending.intercepted += 1
                return
            if len(self._pending.actions) >= self.max_calls:
                self._pending.dropped += 1
                return
        try:
            action = self._action(call, cell, result, error)
        except Exception:  # noqa: BLE001 - recording never changes the call's outcome
            with self._lock:
                self._pending.failed += 1
            return
        with self._lock:
            if len(self._pending.actions) >= self.max_calls:
                self._pending.dropped += 1
                return
            self._pending.actions.append(action)

    def _action(
        self,
        call: Any,
        cell: int,
        result: Any,
        error: BaseException | None,
    ) -> Action:
        channel, method = split_channel(call)
        failed = error is not None
        args = self._value(list(call.args))
        return Action(
            cell=cell,
            channel=channel,
            method=method,
            args=args if isinstance(args, list) else [args],
            kwargs=self._value(dict(call.kwargs)),
            response=None if failed else self._value(result),
            status="error" if failed else "ok",
            effect=call.effect or "unknown",
            error=(
                error_text(error, self.max_error_chars, self.redactor)
                if failed
                else None
            ),
            kind="tool",
        )

    # -- values -------------------------------------------------------------------------------------

    def _value(self, value: Any) -> Any:
        # Bounded first (nodes and characters), redacted while copying, then capped as a whole.
        clean = jsonable(
            value,
            redactor=self.redactor,
            max_nodes=self.max_nodes,
            max_chars=4 * self.max_value_bytes,
        )
        text = json.dumps(clean, ensure_ascii=False, default=str)
        size = len(text.encode("utf-8"))
        if size <= self.max_value_bytes:
            return clean
        preview = text.encode("utf-8")[: self.max_value_bytes // 2].decode(
            "utf-8",
            errors="ignore",
        )
        return {
            TRUNCATED: {
                "bytes": size,
                "shape": shape(clean)[:_REPR_CHARS],
                "preview": preview,
            },
        }


def _truncated_shape(response: Any) -> str | None:
    if isinstance(response, dict) and set(response) == {TRUNCATED}:
        marker = response[TRUNCATED]
        if isinstance(marker, dict) and isinstance(marker.get("shape"), str):
            return marker["shape"]
    return None


def action_fingerprints(actions: list[Action]) -> dict[str, dict[str, list[str]]]:
    """``fingerprint`` of the drained tool actions; a response over the size cap counts with the shape
    of the full response it replaced, not the shape of the marker."""
    tool = [a for a in actions if a.kind == "tool"]
    capped: dict[str, set[str]] = {}
    plain: list[Action] = []
    for a in tool:
        s = _truncated_shape(a.response) if a.status == "ok" else None
        if s is None:
            plain.append(a)
        else:
            capped.setdefault(f"{a.channel}.{a.method}", set()).add(s)
    out = fingerprint(plain)
    for key, shapes in capped.items():
        slot = out.setdefault(key, {"shapes": [], "errors": []})
        slot["shapes"] = sorted(set(slot["shapes"]) | shapes)
    return out
