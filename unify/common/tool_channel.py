"""The host side of the box-to-host tool channel: an agent's cells call ``tools.<name>(...)`` and each call reaches
the host as one framed request on an inherited pipe; the host checks it, runs the tool's handler and answers.

Transport. One channel per interpreter box: two anonymous pipes, a request pipe (the box holds the write end, the
host reads) and a reply pipe (the host writes, the box holds the read end). Both are created non-inheritable; only
the two box ends are handed to the box (``pass_fds``) and named in its environment as ``TOOLS_CHANNEL=fd:<request>,
<reply>``. The host closes its copies of the box ends right after the spawn (:meth:`Channel.after_spawn`), so either
side sees end of file when the other is gone. Pipes carry no descriptors and have no name to reconnect to.

Framing. A frame is a 4-byte big-endian length and that many bytes of UTF-8 JSON. The host checks, in this order and
before any handler runs: the length (at most :data:`MAX_REQUEST_BYTES`), reading exactly that many bytes (each frame
within :data:`FRAME_READ_TIMEOUT_S` of its first byte), UTF-8, the nesting depth (a lexical scan before parsing, at
most :data:`MAX_DEPTH`), ``json.loads``, the envelope's keys exactly ``{v, id, tool, args}``, ``v == 1``, ``id`` an
integer greater than the previous id, ``tool`` a string and ``args`` an object. Any failure is a malformed frame: the
channel closes and one ``channel_closed`` event row names the reason. Then the call: the tool must be on this box's
allow-list (else the general ``unknown_tool`` refusal) and the arguments must validate against the tool's registry
schema, with unknown arguments refused on every object (``bad_arguments``).

Authority. Who is calling (agent, name, role, pass, roots, limits) comes only from the binding the channel was opened
with, never from a frame; a frame has no field for it and any extra key closes the channel.

Concurrency and time. One call in flight per box: the next frame is read only after the reply to the current one
is written. Every call has a deadline: the tool's own, never past the binding's. A handler that is a coroutine runs
on the event loop and is cancelled at its deadline; a synchronous handler marked blocking runs in the worker pool
(:class:`HandlerThreads`) and, past its deadline, is waited for up to :data:`HANDLER_GRACE_S`; if it is still running
the call is answered ``not run: timed out``, the channel closes with ``handler_stuck`` and :attr:`Channel.stuck` tells
the owning loop to stop (no second handler ever runs on that binding). Only short bookkeeping runs on the loop.

Shared state (one owner or one lock each; handlers of up to 17 boxes of a pass may run at once):

- the counters (reader calls per turn and per pass, checks, the step guard, cells): checked and incremented on the
  event loop, before dispatch, by the binding's ``counters.admit`` (so a flood of parallel calls cannot pass a limit);
- the pass's record (:class:`unify.memory.messaging.PassRecord`): the event loop; the record tools are coroutine
  handlers and never touch it from a thread;
- the tool-events writer and ``refusal-codes.json``: the event loop; every event row goes through ``on_event``, called
  on the loop in the order calls end, and the codes file is written once per pass by the pass itself;
- ``/memory`` as a whole (the draft manifest and pool decisions, ``write``/``edit``, the fixture files and their
  listing, the quota measure and the copy ``check`` judges): the pass's ``memory_lock`` (a :class:`threading.Lock`);
- the pool store: read-only during a pass (``/inputs/pool``); the pass alone changes it, after the loop;
- each helper's staging directory: its own channel (one call in flight), and the export into the writer's
  ``/inputs`` happens on the loop when the helper ends;
- each channel's own fields: the channel's serve task.

C4: :meth:`Channel.handler_threads_alive` and :meth:`HandlerThreads.alive` say whether a handler thread is still
running; the pass waits for none to be alive before its gate or landing starts (:meth:`HandlerThreads.wait_idle`).
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import functools
import hashlib
import importlib
import inspect
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Mapping

#: Request frame limit: the 1 MiB file limit under /memory plus 64 KiB of envelope, so ``write`` of a full file fits.
MAX_REQUEST_BYTES = 1_114_112
#: Reply frame limit.
MAX_REPLY_BYTES = 1 << 20
#: Nesting depth of a request (the envelope is depth 1), the manifest's own limit.
MAX_DEPTH = 16
#: Each frame is read within this many seconds of its first byte (a stalled frame closes the channel).
FRAME_READ_TIMEOUT_S = 10.0
#: How long a reply may wait for the box to read it.
REPLY_WRITE_TIMEOUT_S = 10.0
#: How long a handler thread past its deadline is waited for before the channel closes and the pass stops.
HANDLER_GRACE_S = 5.0
#: The envelope's keys, exactly.
ENVELOPE_KEYS = frozenset({"v", "id", "tool", "args"})
VERSION = 1
#: The reason codes a ``channel_closed`` row can carry.
CLOSE_REASONS = (
    "oversize",
    "eof_mid_frame",
    "frame_timeout",
    "not_utf8",
    "too_deep",
    "not_json",
    "bad_envelope",
    "bad_version",
    "id_not_increasing",
    "between_cells",
    "reply_timeout",
    "handler_stuck",
)

_HEADER = 4
#: the channel serving the current call (handlers reach the call's thread tracker and event extras through it)
_CURRENT: contextvars.ContextVar["_CallScope | None"] = contextvars.ContextVar(
    "tool_channel_call",
    default=None,
)


@dataclass
class CallContext:
    """Who is calling and what the handler may use, built by the host from the channel's binding (never a frame).

    ``roots`` maps a box root path (``/memory``, ``/workspace``, ...) to an open directory descriptor. ``state`` is
    the owning pass or request object, for handlers that need its methods. ``deadline`` is a ``time.monotonic()``
    time; for a call it is that call's own deadline.
    """

    agent: str
    name: str
    role: str
    pass_id: str | None
    request_id: str | None
    roots: dict[str, int]
    limits: object
    counters: object
    record: object | None
    episodes: object | None
    deadline: float
    state: object


class Malformed(Exception):
    """A frame that ends the channel; ``reason`` is one of :data:`CLOSE_REASONS`."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _EndOfInput(Exception):
    """The box closed its end between frames."""


# --- framing -------------------------------------------------------------------------------------------------


def too_deep(text: str, limit: int = MAX_DEPTH) -> bool:
    """Whether the JSON text *text* nests arrays and objects deeper than *limit*, from a lexical scan that never
    parses (brackets inside strings do not count)."""
    depth = 0
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > limit:
                return True
        elif ch in "]}":
            depth -= 1
    return False


def _depth(value: object) -> int:
    """The nesting depth of a parsed value, without recursion."""
    best, stack = 0, [(value, 1)]
    while stack:
        v, d = stack.pop()
        if isinstance(v, dict):
            best = max(best, d)
            stack.extend((x, d + 1) for x in v.values())
        elif isinstance(v, list):
            best = max(best, d)
            stack.extend((x, d + 1) for x in v)
    return best


def parse_request(payload: bytes, last_id: int | None) -> dict:
    """The checked envelope of one request frame's payload, or :class:`Malformed`. *last_id*: the previous id on
    this channel (None before the first)."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise Malformed("not_utf8") from None
    if too_deep(text):
        raise Malformed("too_deep")
    try:
        env = json.loads(text)
    except RecursionError:
        raise Malformed("too_deep") from None
    except ValueError:
        raise Malformed("not_json") from None
    if _depth(env) > MAX_DEPTH:
        raise Malformed("too_deep")
    if not isinstance(env, dict) or set(env) != ENVELOPE_KEYS:
        raise Malformed("bad_envelope")
    if isinstance(env["v"], bool) or env["v"] != VERSION:
        raise Malformed("bad_version")
    call_id = env["id"]
    if not isinstance(call_id, int) or isinstance(call_id, bool):
        raise Malformed("bad_envelope")
    if last_id is not None and call_id <= last_id:
        raise Malformed("id_not_increasing")
    if not isinstance(env["tool"], str) or not isinstance(env["args"], dict):
        raise Malformed("bad_envelope")
    return env


def frame(obj: object) -> bytes:
    """One frame: the 4-byte big-endian length, then the compact UTF-8 JSON of *obj*."""
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return len(data).to_bytes(_HEADER, "big") + data


def canonical_args(args: dict) -> bytes:
    """The canonical JSON of validated arguments (sorted keys, compact), which ``args_bytes`` and ``args_sha256``
    measure."""
    return json.dumps(
        args,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


# --- argument validation -------------------------------------------------------------------------------------

_JSON_TYPES: dict[str, Callable[[object], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(schema: Mapping, value: object, where: str = "arguments") -> str | None:
    """The first way *value* breaks *schema* (a problem text naming only argument names), or None.

    The registry's subset of JSON schema: ``type`` (a name or a list), ``enum``, ``const``, ``properties`` with
    ``required`` (every object refuses names it does not declare, whatever ``additionalProperties`` says),
    ``items``, ``minItems``, ``maxItems`` and ``anyOf``. Iterative, so a value's depth cannot exhaust the stack.
    """
    stack: list[tuple[Mapping, object, str]] = [(schema, value, where)]
    while stack:
        sch, val, at = stack.pop()
        if not isinstance(sch, Mapping):
            continue
        if "anyOf" in sch:
            options = [s for s in sch["anyOf"] if isinstance(s, Mapping)]
            if not any(validate(s, val, at) is None for s in options):
                return f"{at} does not match any allowed form"
        types = sch.get("type")
        if types is not None:
            names = types if isinstance(types, list) else [types]
            if not any(_JSON_TYPES.get(t, lambda v: False)(val) for t in names):
                return f"{at} must be {' or '.join(map(str, names))}"
        if "enum" in sch and val not in sch["enum"]:
            return f"{at} must be one of {', '.join(map(str, sch['enum']))}"
        if "const" in sch and val != sch["const"]:
            return f"{at} must be {sch['const']!r}"
        if isinstance(val, dict) and ("properties" in sch or sch.get("type") == "object"):
            props = sch.get("properties") or {}
            unknown = sorted(k for k in val if k not in props)
            if unknown:
                shown = ", ".join(repr(str(k)[:40]) for k in unknown[:5])
                return f"unknown argument{'s' if len(unknown) > 1 else ''} {shown} in {at}"
            for req in sch.get("required") or ():
                if req not in val:
                    return f"{at} needs {req!r}"
            for k, v in val.items():
                stack.append((props[k], v, f"{k!r}" if at == where else f"{at}.{k}"))
        if isinstance(val, list):
            if "minItems" in sch and len(val) < sch["minItems"]:
                return f"{at} needs at least {sch['minItems']} items"
            if "maxItems" in sch and len(val) > sch["maxItems"]:
                return f"{at} takes at most {sch['maxItems']} items"
            items = sch.get("items")
            if isinstance(items, Mapping):
                stack.extend((items, v, f"{at}[{i}]") for i, v in enumerate(val))
    return None


# --- replies ---------------------------------------------------------------------------------------------------


def _reply_class() -> Any:
    from unify.memory.writer_tools import Reply

    return Reply


def _general(form: str, **fields: object) -> Any:
    from unify.memory.writer_tools import general

    return general(form, **fields)


def as_reply(value: object) -> Any:
    """A handler's answer as a ``Reply``: a ``Reply`` as it is; a text, ``refused`` when it carries a form code
    (:class:`unify.memory.tool_spec.Refusal`), else ``ok``."""
    reply_cls = _reply_class()
    if isinstance(value, reply_cls):
        return value
    text = "" if value is None else str(value)
    form = getattr(value, "form", None)
    return reply_cls(text, "refused" if form else "ok", form)


def frame_status(reply: Any) -> str:
    """The reply frame's status: ``ok``, ``refused``, ``not_run`` (a refusal whose text starts ``not run: ``) or
    ``error``."""
    if reply.status == "error":
        return "error"
    if reply.status == "refused":
        return "not_run" if reply.text.startswith("not run: ") else "refused"
    return "ok"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tool_row(
    binding: CallContext,
    tool: str,
    args_canonical: bytes,
    reply: Any,
    *,
    cell: int | None,
    box_gone: bool,
    extra: Mapping | None = None,
) -> dict:
    """One ``tool_call`` row for an in-code call, with the writer's fields (``call`` is the pass's own count and is
    set by the pass's event writer) and the additive ``via``, ``cell``, ``args_sha256`` and ``box_gone``. Built only
    from the binding, the handler's reply and the digested arguments; never from text the box wrote."""
    text_bytes = len(reply.text.encode("utf-8", errors="replace"))
    row = {
        "event": "tool_call",
        "agent": binding.agent,
        "name": binding.name,
        "pass_id": binding.pass_id,
        "utc": _utc(),
        "tool": tool[:80],
        "status": reply.status,
        "refusal": (
            {"form": reply.form, "text_bytes": text_bytes}
            if reply.status == "refused"
            else None
        ),
        "paths": [dict(p) for p in reply.paths],
        "args_bytes": len(args_canonical),
        "result_bytes": text_bytes,
        "result_truncated": "; next: offset=" in reply.text,
        "usage": None,
        "via": "channel",
        "cell": cell,
        "args_sha256": hashlib.sha256(args_canonical).hexdigest(),
        "box_gone": bool(box_gone),
        **dict(extra or {}),
    }
    if binding.request_id is not None:
        row["request_id"] = binding.request_id
    return row


# --- handler threads (C4) ----------------------------------------------------------------------------------------


class HandlerThreads:
    """The worker-pool threads running tool handlers, for one channel and (through *parent*) for the whole pass.

    A thread is tracked from submission until its function returns, also after its call stopped waiting for it
    (a handler past its deadline): the gate and landing wait until :meth:`alive` is false.
    """

    def __init__(self, parent: "HandlerThreads | None" = None) -> None:
        self._futures: set[asyncio.Future] = set()
        self._parent = parent

    def _track(self, fut: asyncio.Future) -> None:
        self._futures.add(fut)
        fut.add_done_callback(self._futures.discard)
        if self._parent is not None:
            self._parent._track(fut)

    def submit(self, fn: Callable[..., Any], *args: Any) -> asyncio.Future:
        """*fn(*args)* in the handlers' worker pool, with the caller's context; a tracked future."""
        loop = asyncio.get_running_loop()
        run = functools.partial(contextvars.copy_context().run, fn, *args)
        fut = loop.run_in_executor(_pool(), run)
        self._track(fut)
        return fut

    def alive(self) -> bool:
        return any(not f.done() for f in self._futures)

    async def wait_idle(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for every tracked thread to return; True when none is running."""
        pending = [f for f in self._futures if not f.done()]
        if pending:
            await asyncio.wait(pending, timeout=max(0.0, timeout))
        return not self.alive()


#: The handlers' own bounded worker pool, apart from the loop's default executor, so blocking work elsewhere (cells,
#: model calls) never starves a handler and a handler never starves it.
HANDLER_WORKERS = 32
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(
                max_workers=HANDLER_WORKERS,
                thread_name_prefix="tool-handler",
            )
        return _POOL


@dataclass
class _CallScope:
    channel: "Channel"
    extra: dict


async def in_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run the blocking *fn(*args)* in the worker pool from a coroutine handler, tracked by the calling channel's
    threads (C4). Outside a call it is :func:`asyncio.to_thread`. Cancelling the caller stops the wait, never the
    thread, which stays tracked until it returns."""
    scope = _CURRENT.get()
    if scope is None:
        return await asyncio.to_thread(fn, *args)
    fut = scope.channel.threads.submit(fn, *args)
    return await asyncio.shield(fut)


def event_extra() -> dict:
    """The current call's extra event fields (e.g. ``messaging``), which a handler fills; a throwaway dict outside a
    call."""
    scope = _CURRENT.get()
    return scope.extra if scope is not None else {}


def resolve_handler(ref: str) -> Callable[..., Any]:
    """The function a registry ``handler`` string (``module:function``) names."""
    module, _, name = ref.partition(":")
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError(f"{ref} is not callable")
    return fn


# --- the channel -------------------------------------------------------------------------------------------------


class Channel:
    """One box's tool channel (module docstring). Open it before the box starts, hand :meth:`box_fds` to the box with
    :meth:`env_value` in its environment, call :meth:`after_spawn` once it runs, and run :meth:`serve` as a task.
    """

    def __init__(
        self,
        binding: CallContext,
        allow: tuple[str, ...],
        on_event: Callable[[dict], None] | None,
        *,
        specs: Mapping[str, Any] | None = None,
        threads: HandlerThreads | None = None,
    ) -> None:
        self.binding = binding
        self.allow = tuple(allow)
        self._on_event = on_event
        self._specs = specs
        self.threads = HandlerThreads(parent=threads)
        # host end of the request pipe (read), box end (write); box end of the reply pipe (read), host end (write)
        self._req_r, self._req_w = os.pipe()
        self._rep_r, self._rep_w = os.pipe()
        for fd in (self._req_r, self._req_w, self._rep_r, self._rep_w):
            os.set_inheritable(fd, False)
        os.set_blocking(self._req_r, False)
        os.set_blocking(self._rep_w, False)
        self._box_ends_open = True
        self._task: asyncio.Task | None = None
        self._last_id: int | None = None
        self._cell: int | None = None
        self._in_flight: str | None = None
        self._box_gone = False
        self.closed: str | None = None
        self.stuck = False
        self.calls = 0
        #: seconds spent in host calls per cell (the cell clock pauses during them)
        self.host_seconds: dict[int, float] = {}

    @classmethod
    def open(
        cls,
        binding: CallContext,
        allow: tuple[str, ...],
        on_event: Callable[[dict], None] | None,
        *,
        specs: Mapping[str, Any] | None = None,
        threads: HandlerThreads | None = None,
    ) -> "Channel":
        """A channel for one box. *specs*: the registry by name (default :func:`unify.common.tool_registry.registry`);
        *threads*: the pass's tracker, so the pass sees every channel's handler threads."""
        return cls(binding, allow, on_event, specs=specs, threads=threads)

    # -- the box's side of the pipes --------------------------------------------------------------------------

    def box_fds(self) -> tuple[int, int]:
        """(the request pipe's write end, the reply pipe's read end): the two descriptors the box gets."""
        return self._req_w, self._rep_r

    def env_value(self) -> str:
        """``TOOLS_CHANNEL``'s value in the box: ``fd:<request>,<reply>`` (the box keeps the descriptor numbers)."""
        return f"fd:{self._req_w},{self._rep_r}"

    def after_spawn(self) -> None:
        """Close the host's copies of the box ends, so end of file reaches either side when the other is gone."""
        if self._box_ends_open:
            self._box_ends_open = False
            for fd in (self._req_w, self._rep_r):
                _close_fd(fd)

    # -- cells --------------------------------------------------------------------------------------------------

    def cell_started(self, cell: int) -> None:
        """Frames are accepted from now until :meth:`cell_ended`; *cell* is the cell row's ``call`` number."""
        self._cell = cell
        self.host_seconds.setdefault(cell, 0.0)

    def cell_ended(self, cell: int) -> None:
        if self._cell == cell:
            self._cell = None

    def call_in_flight(self) -> str | None:
        """The tool of the call being answered now, or None (the interpreter pauses the cell clock while one is)."""
        return self._in_flight

    def box_gone(self) -> None:
        """The box was killed or ended: a reply still to be written is dropped, and the call's row says so."""
        self._box_gone = True

    def handler_threads_alive(self) -> bool:
        return self.threads.alive()

    # -- serving -----------------------------------------------------------------------------------------------

    def _spec_table(self) -> Mapping[str, Any]:
        if self._specs is None:
            from unify.common import tool_registry

            self._specs = tool_registry.registry()
        return self._specs

    async def serve(self) -> None:
        """Read and answer frames until the channel closes or the box closes its end."""
        self._task = asyncio.current_task()
        try:
            while self.closed is None:
                try:
                    payload = await self._read_frame()
                except _EndOfInput:
                    await self.close("eof", record=False)
                    return
                except Malformed as exc:
                    await self.close(exc.reason)
                    return
                if self._cell is None:
                    await self.close("between_cells")
                    return
                try:
                    env = parse_request(payload, self._last_id)
                except Malformed as exc:
                    await self.close(exc.reason)
                    return
                self._last_id = env["id"]
                await self._answer(env)
        except asyncio.CancelledError:
            await self.close("cancelled", record=False)
            raise

    async def _answer(self, env: dict) -> None:
        cell = self._cell
        started = time.monotonic()
        tool, args = env["tool"], env["args"]
        self._in_flight = tool
        self._box_gone = False
        scope = _CallScope(self, {})
        token = _CURRENT.set(scope)
        try:
            reply = await self._dispatch(tool, args)
        finally:
            _CURRENT.reset(token)
            self._in_flight = None
        self.calls += 1
        if cell is not None:
            self.host_seconds[cell] = self.host_seconds.get(cell, 0.0) + (
                time.monotonic() - started
            )
        out = {
            "v": VERSION,
            "id": env["id"],
            "status": frame_status(reply),
            "text": reply.text,
            "form": reply.form,
            "paths": [dict(p) for p in reply.paths],
            "next": None,
        }
        data = frame(out)
        if len(data) - _HEADER > MAX_REPLY_BYTES:
            data = frame(_cut(out))
        if not self._box_gone:
            try:
                await self._write_all(data)
            except BrokenPipeError:
                self._box_gone = True
            except Malformed as exc:
                self._emit(tool, args, reply, cell, scope.extra)
                await self.close(exc.reason)
                return
        self._emit(tool, args, reply, cell, scope.extra)
        if self.stuck and self.closed is None:
            await self.close("handler_stuck")

    def _emit(self, tool: str, args: dict, reply: Any, cell: int | None, extra: dict) -> None:
        if self._on_event is None:
            return
        try:
            canon = canonical_args(args)
        except (TypeError, ValueError):
            canon = b""
        self._on_event(
            tool_row(
                self.binding,
                tool,
                canon,
                reply,
                cell=cell,
                box_gone=self._box_gone,
                extra=extra,
            ),
        )

    async def _dispatch(self, tool: str, args: dict) -> Any:
        """The answer to one well-formed call: the allow-list, the schema, the wall clock, the counters (on the
        loop, before anything runs), then the handler under its deadline."""
        specs = self._spec_table()
        spec = specs.get(tool) if tool in self.allow else None
        if spec is None:
            return _general(
                "unknown_tool",
                name=tool[:80],
                tools=", ".join(self.allow),
            )
        problem = validate(getattr(spec, "parameters", {}) or {}, args)
        if problem is not None:
            return _general("bad_arguments", reason=problem[:300])
        now = time.monotonic()
        if self.binding.deadline - now <= 0:
            return _general("stopped", reason="the wall clock has ended")
        admit = getattr(self.binding.counters, "admit", None)
        if admit is not None:
            refused = admit(tool, getattr(spec, "kind", None), args)
            if refused is not None:
                return as_reply(refused)
        own = getattr(spec, "deadline_s", None)
        deadline = self.binding.deadline
        if own is not None:
            deadline = min(deadline, now + float(own))
        ctx = dataclasses.replace(self.binding, deadline=deadline)
        try:
            handler = spec.handler
            fn = handler if callable(handler) else resolve_handler(handler)
        except Exception as exc:  # noqa: BLE001 - a missing handler is an error reply, never a crash
            return _reply_class()(f"error: {type(exc).__name__} in tools.{tool}", "error")
        return await self._run_handler(tool, fn, ctx, args, bool(getattr(spec, "blocking", False)), deadline)

    async def _run_handler(
        self,
        tool: str,
        fn: Callable[..., Any],
        ctx: CallContext,
        args: dict,
        blocking: bool,
        deadline: float,
    ) -> Any:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            if inspect.iscoroutinefunction(fn):
                task = asyncio.ensure_future(fn(ctx, args))
                try:
                    return as_reply(await asyncio.wait_for(task, timeout))
                except asyncio.TimeoutError:
                    return _general("stopped", reason="timed out")
            if not blocking:
                return as_reply(fn(ctx, args))
            fut = self.threads.submit(fn, ctx, args)
            done, _ = await asyncio.wait({fut}, timeout=timeout)
            if not done:
                done, _ = await asyncio.wait({fut}, timeout=HANDLER_GRACE_S)
            if not done:
                # the thread cannot be cancelled: no further call on this binding, and the owner stops the pass
                self.stuck = True
                return _general("stopped", reason="timed out")
            return as_reply(fut.result())
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken handler is answered, never ends the channel
            return _reply_class()(f"error: {type(exc).__name__} in tools.{tool}", "error")

    # -- pipe I/O ------------------------------------------------------------------------------------------------

    async def _ready(self, fd: int, reading: bool, deadline: float | None) -> None:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def wake() -> None:
            if not fut.done():
                fut.set_result(None)

        (loop.add_reader if reading else loop.add_writer)(fd, wake)
        try:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise Malformed("frame_timeout" if reading else "reply_timeout") from None
        finally:
            if self.closed is None:  # a closed channel's descriptor number may already be reused
                (loop.remove_reader if reading else loop.remove_writer)(fd)

    async def _read_some(self, n: int, deadline: float | None) -> bytes:
        while True:
            try:
                return os.read(self._req_r, n)
            except BlockingIOError:
                await self._ready(self._req_r, True, deadline)
            except OSError:
                return b""

    async def _read_exact(self, n: int, deadline: float) -> bytes:
        out = bytearray()
        while len(out) < n:
            chunk = await self._read_some(min(n - len(out), 1 << 16), deadline)
            if not chunk:
                raise Malformed("eof_mid_frame")
            out += chunk
        return bytes(out)

    async def _read_frame(self) -> bytes:
        first = await self._read_some(1, None)  # no deadline before a frame starts
        if not first:
            raise _EndOfInput()
        deadline = time.monotonic() + FRAME_READ_TIMEOUT_S
        header = first + await self._read_exact(_HEADER - 1, deadline)
        length = int.from_bytes(header, "big")
        if length > MAX_REQUEST_BYTES:
            raise Malformed("oversize")
        return await self._read_exact(length, deadline)

    async def _write_all(self, data: bytes) -> None:
        deadline = time.monotonic() + REPLY_WRITE_TIMEOUT_S
        view = memoryview(data)
        while view:
            try:
                view = view[os.write(self._rep_w, view) :]
            except BlockingIOError:
                await self._ready(self._rep_w, False, deadline)

    # -- closing -------------------------------------------------------------------------------------------------

    async def close(self, reason: str, *, record: bool = True) -> None:
        """Close the channel: a best-effort closing notice to the box (never waited for), both host ends shut, and
        one ``channel_closed`` row when *record* (every malformed frame, stall and stuck handler)."""
        if self.closed is not None:
            return
        self.closed = reason
        task = self._task
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        try:
            loop = asyncio.get_running_loop()
            loop.remove_reader(self._req_r)
            loop.remove_writer(self._rep_w)
        except (RuntimeError, ValueError, OSError):
            pass
        try:
            os.write(
                self._rep_w,
                frame({"v": VERSION, "id": 0, "status": "closed", "text": reason, "form": reason}),
            )
        except OSError:
            pass
        for fd in (self._req_r, self._rep_w):
            _close_fd(fd)
        self.after_spawn()
        if record and self._on_event is not None:
            b = self.binding
            row = {
                "event": "channel_closed",
                "agent": b.agent,
                "name": b.name,
                "pass_id": b.pass_id,
                "utc": _utc(),
                "reason": reason,
                "cell": self._cell,
                "via": "channel",
            }
            if b.request_id is not None:
                row["request_id"] = b.request_id
            self._on_event(row)


def _cut(out: dict) -> dict:
    """A reply over the frame limit: its head, with a note saying how much there was (never a silent cut)."""
    raw = out["text"].encode("utf-8")
    keep = MAX_REPLY_BYTES - 65536
    head = raw[:keep].decode("utf-8", errors="ignore")
    note = f"\n(reply cut: {len(raw)} bytes, the first {len(head.encode('utf-8'))} shown)"
    return {**out, "text": head + note, "paths": out["paths"][:200]}


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass

