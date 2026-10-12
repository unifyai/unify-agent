"""The memory agents' tools as host handlers for in-code calls (``tools.<name>(...)`` through the tool channel,
:mod:`unify.common.tool_channel`): one function per tool, named as the tool, ``handler(ctx, args) -> Reply``.

Each handler keeps the logic, reply texts and refusal forms of the same tool as a JSON tool call
(``writer_pass.WriterPass._run``'s dispatch and ``helpers._tool``); only how the call arrives changes. The calling
agent's pass state is ``ctx.state``: a :class:`WriterCalls` for the writer or the curator, a :class:`HelperCalls` for
a helper. The channel has already checked the allow-list, the arguments against the tool's schema and the counters
(``state.admit``, on the event loop) before a handler runs.

Where each runs: a synchronous handler here does blocking work and is registered ``blocking`` (the channel runs it in
the worker pool); a coroutine handler runs on the event loop and sends its blocking parts to the pool with
:func:`unify.common.tool_channel.in_thread`. Work on ``/memory`` as a whole (the draft manifest, ``write``/``edit``,
fixtures, the quota measure, the copy ``check`` judges) holds the pass's ``memory_lock``.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from unify.common.tool_channel import HANDLER_GRACE_S, event_extra, in_thread

from . import edit_tools as _et
from . import writer_tools as _wt
from .tool_spec import form_of

#: What each memory tool is, for the registry's in-code fields: (kind, blocking, deadline in seconds or None for the
#: pass's wall clock). Coroutine handlers are not ``blocking``: they run on the loop and send heavy work to the pool.
TOOL_FIELDS: dict[str, tuple[str, bool, float | None]] = {
    "read": ("reader", True, 30.0),
    "grep": ("reader", True, 30.0),
    "overview": ("reader", True, 30.0),
    "read_episode": ("reader", True, 30.0),
    "pool": ("reader", True, 30.0),
    "similar": ("reader", False, 30.0),
    "fixture": ("action", True, 60.0),
    "replay": ("action", False, None),
    "check": ("action", False, None),
    "finish": ("terminal", False, None),
    "write": ("action", True, 60.0),
    "edit": ("action", True, 60.0),
    "delegate": ("record", False, 60.0),
    "send": ("record", False, 30.0),
    "messages": ("record", False, 30.0),
    "wait": ("record", False, None),
    "who": ("record", False, 30.0),
    "reply": ("record", False, 30.0),
    "stop": ("record", False, None),
    "ask": ("record", False, None),
    "stage": ("action", True, 60.0),
    "done": ("terminal", False, None),
}

#: The draft manifest under /memory.
_MANIFEST_SHOWN = "/memory/.pass/manifest.json"


# --- pass state ----------------------------------------------------------------------------------------------------


@dataclass
class WriterCalls:
    """The writer's (or the curator's) pass state its in-code tool handlers use, owned by the pass's event loop and
    by its one channel (one call in flight): the pass loop copies its counters in before a cell and back after it.
    """

    writer: Any
    agent: str
    pass_id: str
    box: Path
    inputs: Path
    tmp: Path
    roots: dict[str, Path]
    parent: str
    uncapped: bool
    max_calls: int
    max_reads: int
    remaining: Callable[[], float]
    cause: Callable[[str], None]
    writing: Callable[[], None]
    check_file: Callable[..., Any]
    phase: dict
    notes: list
    deadline_s: float
    #: the pass's wall clock end (``time.monotonic()``) and its spend (helpers' spend joins it)
    deadline: float = 0.0
    spend: Any = None
    calls: int = 0
    checks: int = 0
    reads: int = 0
    readers: int = 0
    stop: str | None = None
    finished: bool = False
    summary: str = ""
    base: Any = None
    #: the accepted finish's or the refused finish's reply text, shown as the cell's last line
    finish_text: str | None = None
    memory_lock: threading.Lock = field(default_factory=threading.Lock)
    #: kills the agent's box (verified) before finish judges /memory
    kill_box: Callable[[], Awaitable[None]] | None = None
    #: the pass's handler-thread tracker (C4)
    threads: Any = None
    check_copies: int = 0
    kind: str = "writer"

    def admit(self, tool: str, kind: str | None, args: dict) -> "_wt.Reply | None":
        """The counters, checked and incremented on the event loop before the handler runs (today's order and
        texts); a refusal or None."""
        from .writer_pass import CHECK_FLOOR_S, CODE_DEADLINE, MAX_CHECKS

        if self.finished:
            return _wt.general("pass_finished")
        if tool == "finish":
            return None
        if tool == "check":
            if self.stop is not None:
                return _wt.general("stopped", reason=self.stop)
            if not self.uncapped and self.checks >= MAX_CHECKS:
                return _wt.general("check_limit", limit=MAX_CHECKS)
            if self.calls >= self.max_calls:
                return _wt.general("call_limit")
            if self.remaining() < CHECK_FLOOR_S:
                return _wt.refused("check", "too_late")
            self.checks += 1
            self.calls += 1  # a check is a call against max_calls
            self.writing()
            return None
        if tool in ("replay", "fixture"):
            self.calls += 1  # a writer action, against the step guard
            self.writing()
            return None
        if tool in _et.NAMES:
            if self.stop is not None:
                return _wt.general("stopped", reason=self.stop)
            if self.remaining() <= 0:
                self.stop = f"pass deadline of {self.deadline_s} s reached"
                self.cause(CODE_DEADLINE)
                return _wt.general("stopped", reason=self.stop)
            self.calls += 1  # a writer action, against the step guard
            self.writing()
            return None
        if tool in ("read", "grep", "overview", "read_episode", "pool"):
            if self.readers >= _wt.READERS_PER_TURN:
                return _wt.general("readers_per_turn")
            if not self.uncapped and self.reads >= int(self.max_reads):
                return _wt.general("reads_limit", limit=int(self.max_reads))
            self.readers += 1
            self.reads += 1
        return None


@dataclass
class HelperCalls:
    """A helper's state its in-code tool handlers use: the delegate context, its own name and its box's folders
    (``st`` as ``helpers._prepare`` makes it), owned by its one channel."""

    ctx: Any
    call: str
    st: dict
    resolve: Callable[..., Any]
    doc: dict
    finished: bool = False
    finish_text: str | None = None
    kill_box: Callable[[], Awaitable[None]] | None = None
    threads: Any = None
    kind: str = "helper"

    def admit(self, tool: str, kind: str | None, args: dict) -> "_wt.Reply | None":
        if self.finished:
            from .tool_spec import GENERAL, refusal

            text = refusal(GENERAL, "task_done")
            return _wt.Reply(text, "refused", text.form)
        return None


# --- shared helpers ------------------------------------------------------------------------------------------------


def _reply_of(content: object, paths: list | tuple = ()) -> "_wt.Reply":
    """A helper tool's (text, paths) as a Reply: refused when the text carries a form code."""
    form = form_of(content)
    return _wt.Reply(str(content), "refused" if form else "ok", form, tuple(paths))


def write_draft_manifest(box: Path, manifest: object) -> None:
    """Write the draft manifest ``/memory/.pass/manifest.json`` from /memory's descriptor, component by component
    without following a link, through a new ``O_EXCL`` file renamed over it (``edit_tools._replace``): a link swapped
    in at the leaf is replaced, never written through. ``OSError`` on any refusal."""
    import json

    text = (
        manifest
        if isinstance(manifest, str)
        else json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    parts = _wt.MANIFEST.split("/")
    try:
        root = os.open(Path(box), _et._DIR_FLAGS)
    except OSError:
        raise OSError("/memory cannot be opened") from None
    try:
        dir_fd = _et._parent(root, parts, _wt.MANIFEST, create=True)
        try:
            _et._check_leaf(_et._leaf_kind(dir_fd, parts[-1]), _wt.MANIFEST)
            _et._replace(dir_fd, parts[-1], text.encode("utf-8"), _wt.MANIFEST)
        finally:
            os.close(dir_fd)
    except _et._Refused as r:
        raise OSError(f"/memory/.pass/manifest.json: {r.form}") from None
    finally:
        os.close(root)


def _writer_reader(ctx: Any, name: str, args: dict) -> "_wt.Reply":
    st: WriterCalls = ctx.state
    return st.writer._reader_call(name, args, st.roots, agent=st.agent)


async def _helper_tool(ctx: Any, name: str, args: dict) -> "_wt.Reply":
    from . import helpers as _helpers

    hc: HelperCalls = ctx.state
    content, paths = await _helpers._tool(hc.ctx, name, args, hc.st, hc.resolve)
    return _reply_of(content, paths)


# --- readers -------------------------------------------------------------------------------------------------------


def _reader(name: str) -> Callable[[Any, dict], Any]:
    def handler(ctx: Any, args: dict) -> "_wt.Reply":
        if isinstance(ctx.state, HelperCalls):
            from . import helpers as _helpers

            hc: HelperCalls = ctx.state
            content, paths = _helpers._tool_sync(hc.ctx, name, args, hc.st, hc.resolve)
            return _reply_of(content, paths)
        return _writer_reader(ctx, name, args)

    handler.__name__ = handler.__qualname__ = name
    handler.__doc__ = f"The ``{name}`` tool (a reader): today's reader, from the worker pool."
    return handler


read = _reader("read")
grep = _reader("grep")


async def overview(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``overview`` tool."""
    if isinstance(ctx.state, HelperCalls):
        return await _helper_tool(ctx, "overview", args)
    return await in_thread(_writer_reader, ctx, "overview", args)


async def read_episode(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``read_episode`` tool."""
    if isinstance(ctx.state, HelperCalls):
        return await _helper_tool(ctx, "read_episode", args)
    return await in_thread(_writer_reader, ctx, "read_episode", args)


def pool(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``pool`` tool; ``decide`` writes the draft manifest descriptor-relatively (:func:`write_draft_manifest`),
    under the pass's /memory lock."""
    st: WriterCalls = ctx.state
    with st.memory_lock:
        return _wt.pool(
            args,
            st.roots["/inputs"] / "pool",
            st.roots["/memory"],
            write=write_draft_manifest,
        )


async def similar(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``similar`` tool."""
    if isinstance(ctx.state, HelperCalls):
        return await _helper_tool(ctx, "similar", args)
    st: WriterCalls = ctx.state
    return await in_thread(st.writer._similar_call, args)


# --- the writer's actions ------------------------------------------------------------------------------------------


def fixture(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``fixture`` tool, under the pass's /memory lock."""
    st: WriterCalls = ctx.state
    w = st.writer
    with st.memory_lock:
        return _wt.fixture(args, w._episode, st.box, w.gate._blob, w._batch)


async def replay(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``replay`` tool: today's verifier, from the worker pool."""
    st: WriterCalls = ctx.state
    content = await in_thread(st.writer._replay_call, args, st.box, st.tmp)
    return _wt.Reply(
        content,
        (
            "error"
            if content.startswith("replay error")
            else ("refused" if content.startswith("not run") else "ok")
        ),
        "not_run" if content.startswith("not run") else None,
    )


def _edit_call(name: str) -> Callable[[Any, dict], Any]:
    def handler(ctx: Any, args: dict) -> "_wt.Reply":
        from .writer_pass import _measure

        st: WriterCalls = ctx.state
        with st.memory_lock:
            reply = _et.call(name, args, st.box)
            if reply.status == "ok":
                over_quota = _measure(st.box)
                if over_quota is not None:
                    st.stop = over_quota
                    reply = _wt.Reply(
                        f"{reply.text}\n({over_quota}; the pass is refused)",
                        reply.status,
                        reply.form,
                        reply.paths,
                    )
        return reply

    handler.__name__ = handler.__qualname__ = name
    handler.__doc__ = f"The ``{name}`` tool on /memory (descriptor-relative), then the quota measure."
    return handler


write = _edit_call("write")
edit = _edit_call("edit")


def _check_copy(st: WriterCalls, given: object) -> tuple["_wt.Reply | None", Path | None, bool]:
    """Under the /memory lock: write a given manifest to the live draft, then copy /memory (descriptor-relative,
    never following a link) for the check to judge. (refusal or None, the copy, whether a manifest was written)."""
    from .writer_pass import _check_manifest, _measure, _mirror

    where = [{"path": _MANIFEST_SHOWN, "mode": "read"}]
    wrote = False
    with st.memory_lock:
        if given not in (None, ""):
            parsed, problem = _check_manifest(given)
            if problem is not None:
                return _wt.refused("check", "manifest_problem", tuple(where), reason=problem), None, False
            try:
                write_draft_manifest(st.box, given if isinstance(given, str) else parsed)
            except OSError as exc:
                return (
                    _wt.refused(
                        "check",
                        "manifest_problem",
                        tuple(where),
                        reason=f"cannot write /memory/.pass/manifest.json ({type(exc).__name__})",
                    ),
                    None,
                    False,
                )
            wrote = True
        _measure(st.box)  # makes every entry readable again, as after a cell
        st.check_copies += 1
        copy = Path(st.tmp) / f"check-copy-{st.check_copies}"
        copy.mkdir()
        _mirror(st.box, copy)
    return None, copy, wrote


def _check_work(st: WriterCalls, args: dict) -> "_wt.Reply":
    """``check`` from the worker pool: the parent snapshot once, the copy, then today's check on the copy (its
    heavy work in killable children, as today)."""
    import shutil

    if st.base is None:
        try:
            st.base = st.writer.gate.parent_snapshot(st.parent, Path(st.tmp) / "check-parent")
        except Exception:  # noqa: BLE001 - GitError, OSError: the preview retries
            st.base = None
        st.base = st.base or st.parent
    refused, copy, wrote = _check_copy(st, args.get("manifest"))
    if refused is not None:
        return refused
    try:
        reply = st.check_file({}, st.base, box=copy)
    finally:
        shutil.rmtree(copy, ignore_errors=True)
    if wrote:
        reply = _wt.Reply(
            reply.text,
            reply.status,
            reply.form,
            ({"path": _MANIFEST_SHOWN, "mode": "write"}, *reply.paths),
        )
    return reply


async def check(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``check`` tool: judged on a copy of /memory taken when the call arrives, so a cell still writing cannot
    change what is checked; never on the event loop."""
    return await in_thread(_check_work, ctx.state, args)


async def finish(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``finish`` tool: the box is killed first (nothing writes /memory while it is judged), then today's finish
    logic; an accepted finish ends every helper and waits for every handler thread of the pass (C4)."""
    from .writer_pass import FINISH_G1_REFUSALS, _accepted_manifest

    st: WriterCalls = ctx.state
    if st.kill_box is not None:
        await st.kill_box()
    library = bool(st.writer.cfg.library_mode)
    if library and (st.uncapped or st.phase["g1_refusals"] < FINISH_G1_REFUSALS):
        g1, empty = await in_thread(_locked, st, st.writer._finish_manifest_problem, st.box)
    else:
        g1, empty = None, False
    manifest_path = ({"path": _MANIFEST_SHOWN, "mode": "read"},)
    if g1 is not None:
        st.phase["g1_refusals"] += 1  # bounded: past it the final check refuses by name
        content = _wt.refusal("finish", "manifest_problem", reason=g1)
        reply = _wt.Reply(content, "refused", "manifest_problem", manifest_path)
    else:
        st.summary, st.finished = str(args.get("summary", "")), True
        st.phase.update(finish_accepted=True, nothing_to_store=empty)
        st.writer.final_manifest = await in_thread(_locked, st, _accepted_manifest, st.box)
        # finishing ends every helper still running; what each staged is kept
        await st.writer._close_record()
        # C4: the gate never starts while a handler thread of this pass runs
        if st.threads is not None and not await st.threads.wait_idle(HANDLER_GRACE_S):
            st.stop = "a tool handler of this pass did not end; the gate does not start"
        reply = _wt.Reply("ok", paths=manifest_path if library else ())
    st.finish_text = reply.text
    return reply


def _locked(st: WriterCalls, fn: Callable[..., Any], *args: Any) -> Any:
    with st.memory_lock:
        return fn(*args)


# --- the record and helpers ----------------------------------------------------------------------------------------


async def delegate(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``delegate`` tool: starts one helper; its model calls are made on the host through the helper route and
    booking (``helpers.handle``); nothing about the route reaches the reply or the event row."""
    from . import helpers as _helpers
    from .writer_pass import _delegate_reply, _messaging_extra

    st: WriterCalls = ctx.state
    w = st.writer
    hctx = w._helper_context(
        st.pass_id,
        st.inputs,
        st.box,
        Path(st.tmp),
        st.deadline,
        st.spend,
    )
    info: dict = {}
    reply = _delegate_reply(await _helpers.handle(args, hctx, info))
    event_extra().update(_messaging_extra(info))
    return reply


def _record_call(name: str) -> Callable[[Any, dict], Any]:
    async def handler(ctx: Any, args: dict) -> "_wt.Reply":
        from . import messaging as _messaging
        from .writer_pass import _delegate_reply, _messaging_extra

        if name == "stop" and "name" in args and "helper" not in args:
            args = {**{k: v for k, v in args.items() if k != "name"}, "helper": args["name"]}
        if isinstance(ctx.state, HelperCalls):
            hc: HelperCalls = ctx.state
            info: dict = {}
            text = await _messaging.call(hc.ctx.pass_record, hc.call, name, args, info)
            event_extra().update({"messaging": dict(info), "helper_call": hc.call})
            return _reply_of(text)
        st: WriterCalls = ctx.state
        pr = st.writer._pass_record
        info = {}
        reply = _delegate_reply(await _messaging.call(pr, pr.root, name, args, info))
        event_extra().update(_messaging_extra(info))
        return reply

    handler.__name__ = handler.__qualname__ = name
    handler.__doc__ = f"The record's ``{name}`` tool, on the event loop (the record's owner)."
    return handler


send = _record_call("send")
messages = _record_call("messages")
wait = _record_call("wait")
who = _record_call("who")
reply = _record_call("reply")
stop = _record_call("stop")
ask = _record_call("ask")


def stage(ctx: Any, args: dict) -> "_wt.Reply":
    """A helper's ``stage`` tool, into its own staging directory."""
    from . import helpers as _helpers

    hc: HelperCalls = ctx.state
    content, paths = _helpers._tool_sync(hc.ctx, "stage", args, hc.st, hc.resolve)
    return _reply_of(content, paths)


async def done(ctx: Any, args: dict) -> "_wt.Reply":
    """A helper's ``done`` tool: today's claim checks; accepted, the helper ends after the cell."""
    from . import helpers as _helpers

    hc: HelperCalls = ctx.state
    try:
        got = await in_thread(_helpers.check_claims, args, hc.resolve, hc.ctx.shared)
    except Exception as exc:  # noqa: BLE001 - a broken check is answered, never ends the helper
        got = f"error: {type(exc).__name__}"
    if isinstance(got, str):
        status = "error" if got.startswith("error:") else "refused"
        out = _wt.Reply(str(got), status, form_of(got) if status == "refused" else None)
    else:
        hc.doc["summary"] = str(args.get("summary", ""))[:4000]
        hc.doc["claims"] = got
        hc.finished = True
        if hc.kill_box is not None:
            await hc.kill_box()
        out = _wt.Reply("ok")
    hc.finish_text = out.text
    return out


# --- registry fields and the evidence store's reads from the worker pool --------------------------------------------


@dataclass(frozen=True)
class SpecView:
    """A memory tool's spec with the in-code fields the channel reads (used until the shared registry provides
    them): the base spec's fields plus ``kind``, ``blocking``, ``deadline_s`` and ``handler``."""

    name: str
    parameters: dict
    kind: str
    blocking: bool
    deadline_s: float | None
    handler: str


def channel_specs() -> dict[str, Any]:
    """The channel's spec table: the shared registry's when it exists, else the memory registry's specs with this
    module's handlers and :data:`TOOL_FIELDS` (every object refuses unknown arguments either way)."""
    try:
        from unify.common import tool_registry  # type: ignore[attr-defined]

        return dict(tool_registry.registry())
    except ImportError:
        pass
    from .tool_spec import registry

    out: dict[str, Any] = {}
    for s in registry():
        if s.name not in TOOL_FIELDS:
            continue
        kind, blocking, deadline = TOOL_FIELDS[s.name]
        out[s.name] = SpecView(
            s.name,
            s.parameters,
            kind,
            blocking,
            deadline,
            f"unify.memory.tool_handlers:{s.name}",
        )
    return out


def in_code_json_tools(agent: str) -> list[dict]:
    """The one JSON tool every agent's requests carry with tools in code: ``execute_code``."""
    try:
        from unify.common import tool_registry  # type: ignore[attr-defined]

        return list(tool_registry.json_tools(agent))
    except ImportError:
        return [_wt.EXECUTE_CODE.as_tool()]


def stub_source(names: tuple[str, ...]) -> str | None:
    """The box module ``tools`` for *names* (the registry's renderer), or None until it exists."""
    try:
        from unify.common import tool_registry  # type: ignore[attr-defined]

        return tool_registry.render_stub(tuple(names))
    except ImportError:
        return None


class _ThreadReads:
    """An evidence store connection that the worker pool can read from: the owning thread keeps the store's own
    connection; any other thread gets its own read-only connection to the same file (SQLite connections belong to
    the thread that made them). A write from another thread fails, as it should: handlers only read the store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._owner = threading.get_ident()
        self._local = threading.local()
        self._opened: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._path = ""
        for row in conn.execute("PRAGMA database_list"):
            if row[1] == "main":
                self._path = row[2] or ""

    def _pick(self) -> sqlite3.Connection:
        if threading.get_ident() == self._owner or not self._path:
            return self._conn
        own = getattr(self._local, "conn", None)
        if own is None:
            uri = Path(self._path).as_uri() + "?mode=ro"
            own = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30.0)
            self._local.conn = own
            with self._lock:
                self._opened.append(own)
        return own

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pick(), name)

    def __enter__(self) -> Any:
        return self._pick().__enter__()

    def __exit__(self, *exc: Any) -> Any:
        return self._pick().__exit__(*exc)

    def close_readers(self) -> None:
        with self._lock:
            for c in self._opened:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._opened.clear()


def thread_reads(*stores: Any) -> Callable[[], None]:
    """Let the worker pool read each distinct evidence store in *stores* (objects with a SQLite ``db``) for the rest
    of the pass; returns the undo, which restores each store's own connection and closes the readers."""
    done: list[tuple[Any, _ThreadReads]] = []
    seen: set[int] = set()
    for s in stores:
        db = getattr(s, "db", None)
        if s is None or id(s) in seen or not isinstance(db, sqlite3.Connection):
            continue
        seen.add(id(s))
        proxy = _ThreadReads(db)
        s.db = proxy
        done.append((s, proxy))

    def undo() -> None:
        for s, proxy in done:
            s.db = proxy._conn
            proxy.close_readers()

    return undo


__all__ = [n for n in TOOL_FIELDS] + [
    "WriterCalls",
    "HelperCalls",
    "channel_specs",
    "in_code_json_tools",
    "stub_source",
    "thread_reads",
    "write_draft_manifest",
    "TOOL_FIELDS",
]


# --- one agent's persistent interpreter and its channel ---------------------------------------------------------------

#: What a cell's output says when the interpreter had to start again.
RESTARTED = "(the interpreter was restarted: variables are gone, files remain)"


def _default_start() -> Callable[..., Awaitable[Any]]:
    from unify.common.box_interpreter import BoxInterpreter  # type: ignore[import-not-found]

    return BoxInterpreter.start


def _open_roots(roots: dict[str, Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for box_path, host in roots.items():
        try:
            out[box_path] = os.open(Path(host), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError:
            continue
    return out


class InterpreterSession:
    """One memory agent's persistent confined interpreter and its tool channel (tools in code).

    The interpreter starts at the agent's first cell with a fresh channel bound to the agent (its name, role, pass,
    roots and state; never anything from the box), and starts again, with a new channel, after the channel closed or
    the interpreter was stopped. :meth:`cell` runs one cell, accepting tool frames only while it runs. :meth:`kill`
    stops the box (``finish``/``done`` call it before judging). :meth:`close` ends both at the pass's end.
    """

    def __init__(
        self,
        *,
        agent: str,
        name: str,
        role: str,
        pass_id: str,
        state: Any,
        allow: tuple[str, ...],
        binds: dict,
        env: dict,
        roots: dict[str, Path],
        deadline: float,
        cell_limit_s: float,
        on_event: Callable[[dict], None],
        threads: Any,
        start: Callable[..., Awaitable[Any]] | None = None,
        specs: dict | None = None,
        profile: Any = None,
    ) -> None:
        from unify.common.tool_channel import CallContext

        self.allow = tuple(allow)
        self.binds, self.env, self.profile = binds, dict(env), profile
        self.cell_limit_s = float(cell_limit_s)
        self.on_event = on_event
        self.threads = threads
        self._start = start
        self._specs = specs
        self._root_fds = _open_roots(roots)
        self.binding = CallContext(
            agent=agent,
            name=name,
            role=role,
            pass_id=pass_id,
            request_id=None,
            roots=dict(self._root_fds),
            limits=None,
            counters=state,
            record=None,
            episodes=None,
            deadline=deadline,
            state=state,
        )
        self.channel: Any = None
        self.interp: Any = None
        self._serve: Any = None
        self._note = ""
        self.stuck = False
        self.box_dead = False

    async def _open(self) -> None:
        from unify.common.tool_channel import Channel

        if self._specs is None:
            self._specs = channel_specs()
        start = self._start or _default_start()
        self.channel = Channel.open(
            self.binding,
            self.allow,
            self.on_event,
            specs=self._specs,
            threads=self.threads,
        )
        self.interp = await start(
            agent=self.binding.agent,
            profile=self.profile,
            binds=self.binds() if callable(self.binds) else self.binds,
            env={**self.env, "TOOLS_CHANNEL": self.channel.env_value()},
            channel=self.channel,
            stub_source=stub_source(self.allow),
            cell_limit_s=self.cell_limit_s,
        )
        self.channel.after_spawn()
        self._serve = asyncio.ensure_future(self.channel.serve())
        self.box_dead = False

    async def _stop_box(self) -> None:
        if self.channel is not None:
            self.channel.box_gone()
        if self.interp is not None and not self.box_dead:
            self.box_dead = True
            await self.interp.stop()

    async def kill(self) -> None:
        """Stop the box now (verified kill); a reply still to be written is dropped."""
        await self._stop_box()

    async def cell(self, code: str, cell_no: int, timeout_s: float) -> tuple[str, Any]:
        """Run one cell: (the text before the host's notes, the interpreter's cell result or None when it failed to
        run). Starts the interpreter first when needed."""
        if self.channel is not None and self.channel.stuck:
            self.stuck = True
        if self.interp is None or self.box_dead or self.channel is None or self.channel.closed is not None:
            if self.interp is not None:
                why = self.channel.closed if self.channel is not None else None
                await self._stop_box()
                await self._close_channel()
                if why and why != "eof":
                    self._note = f"(tool channel closed: {why}) {RESTARTED}\n"
                elif not self._note:
                    self._note = f"{RESTARTED}\n"
            await self._open()
        note, self._note = self._note, ""
        self.channel.cell_started(cell_no)
        try:
            res = await asyncio.wait_for(self.interp.run_cell(code), timeout=max(1.0, timeout_s) + 30.0)
        except asyncio.TimeoutError:
            await self._stop_box()
            return note, None
        finally:
            if self.channel is not None:
                self.channel.cell_ended(cell_no)
        if getattr(res, "timed_out", False) or getattr(res, "restarted", False):
            self.box_dead = True  # the interpreter restarts before the next cell
        if self.channel is not None and self.channel.stuck:
            self.stuck = True
        return note, res

    def closed_reason(self) -> str | None:
        return None if self.channel is None else self.channel.closed

    async def _close_channel(self) -> None:
        if self.channel is not None:
            await self.channel.close("ended", record=False)
        if self._serve is not None:
            self._serve.cancel()
            try:
                await self._serve
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - the serve task ends with the channel
                pass
        self._serve = None

    async def close(self) -> None:
        """At the pass's end: stop the box, close the channel and the root descriptors."""
        try:
            await self._stop_box()
        finally:
            await self._close_channel()
            for fd in self._root_fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._root_fds.clear()


def cell_parts(
    note: str,
    res: Any,
    timeout_s: float,
    closed: str | None,
    finish_text: str | None,
) -> tuple[str, str]:
    """A cell's output in two parts: the body (a host note, then the interpreter's output and error), which the
    agent's own output handling may cut and save; and the host's notes after it (a timeout, a closed channel, the
    end call's result as the last line), which are never cut."""
    tail = ""
    if res is None:
        body = note or "(no output)"
        tail += f"\n(cell timed out after {timeout_s:.0f} s; {RESTARTED[1:-1]})"
    else:
        out = str(getattr(res, "output", "") or "")
        err = str(getattr(res, "error", "") or "")
        body = note + ((out + ("\n" + err if err else "")) or "(no output)")
        if getattr(res, "timed_out", False):
            tail += f"\n(cell timed out after {timeout_s:.0f} s; {RESTARTED[1:-1]})"
    if closed is not None and closed not in ("eof", "ended", "cancelled"):
        tail += f"\n(tool channel closed: {closed})"
    if finish_text is not None:
        tail += "\n" + finish_text
    return body, tail
