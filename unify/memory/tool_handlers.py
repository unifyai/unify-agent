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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from unify.common.tool_channel import HANDLER_GRACE_S, event_extra, in_thread

from . import edit_tools as _et
from . import writer_tools as _wt
from .tool_spec import form_of

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
    #: the interpreter's /memory quota measure (a breach stops the box); None: measured here
    measure_quota: Callable[[], Awaitable[str | None]] | None = None
    #: the pass's handler-thread tracker (the gate waits until none runs)
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
        if tool in ("replay", "fixture", "land_capture"):
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


async def _after_write(st: WriterCalls, reply: "_wt.Reply") -> "_wt.Reply":
    """After a call that wrote /memory: the quota measure (the interpreter's, which stops the box on a breach), and
    today's note and stop on a breach."""
    from .writer_pass import _measure

    if reply.status != "ok":
        return reply
    if st.measure_quota is not None:
        over_quota = await st.measure_quota()
    else:
        over_quota = await in_thread(_locked, st, _measure, st.box)
    if over_quota is None:
        return reply
    st.stop = over_quota
    return _wt.Reply(
        f"{reply.text}\n({over_quota}; the pass is refused)",
        reply.status,
        reply.form,
        reply.paths,
    )


def _fixture_work(st: WriterCalls, args: dict) -> "_wt.Reply":
    w = st.writer
    with st.memory_lock:
        return _wt.fixture(
            args,
            w._episode,
            st.box,
            w.gate._blob,
            w._batch,
            python=getattr(w.gate, "python", None),
        )


async def fixture(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``fixture`` tool, from the worker pool under the pass's /memory lock, then the quota measure."""
    st: WriterCalls = ctx.state
    return await _after_write(st, await in_thread(_fixture_work, st, args))


def _land_work(st: WriterCalls, args: dict) -> "_wt.Reply":
    from .land_capture import land

    w = st.writer
    with st.memory_lock:
        return land(
            args,
            store=w._captures_root(),
            box=st.box,
            episode=w._episode,
            blob=w.gate._blob,
            batch=w._batch,
            python=getattr(w.gate, "python", None),
            pass_id=st.pass_id,
        )


async def land_capture(ctx: Any, args: dict) -> "_wt.Reply":
    """The ``land_capture`` tool (:mod:`.land_capture`): the capture read from the harness's store, the files written
    through the ``write`` and ``fixture`` paths under the pass's /memory lock, then the quota measure."""
    st: WriterCalls = ctx.state
    return await _after_write(st, await in_thread(_land_work, st, args))


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
    async def handler(ctx: Any, args: dict) -> "_wt.Reply":
        st: WriterCalls = ctx.state
        reply = await in_thread(_locked, st, _et.call, name, args, st.box)
        return await _after_write(st, reply)

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
    logic; an accepted finish ends every helper and waits for every handler thread of the pass."""
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
        # the gate never starts while a handler thread of this pass runs
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


def in_code_json_tools(agent: str) -> list[dict]:
    """The one model-API tool every agent's requests carry with tools in code (the registry's ``execute_code``)."""
    from unify.common import tool_registry

    return tool_registry.json_tools(agent)


def stub_source(names: tuple[str, ...], agent: str | None = None) -> str:
    """The box module ``tools`` for *agent* allowed *names* (the registry's renderer)."""
    from unify.common import tool_registry

    return tool_registry.render_stub(tuple(names), agent)


class _ThreadReads:
    """An evidence store connection the worker pool can use: the owning thread keeps the store's own connection; any
    other thread gets its own connection to the same file (SQLite connections belong to the thread that made them),
    with the store's busy timeout, so reads and any write from a pool thread take SQLite's own file locks."""

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
        # the same busy timeout as the store's own connection, so a pool thread waits on a lock as the loop does
        busy = conn.execute("PRAGMA busy_timeout").fetchone()
        self._timeout_s = (busy[0] / 1000.0) if busy and busy[0] else 5.0

    def _pick(self) -> sqlite3.Connection:
        if threading.get_ident() == self._owner or not self._path:
            return self._conn
        own = getattr(self._local, "conn", None)
        if own is None:
            own = sqlite3.connect(self._path, check_same_thread=False, timeout=self._timeout_s)
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
    """Let the worker pool use each distinct evidence store in *stores* (objects with a SQLite ``db``; the writer's,
    the gate's and the episode lookup's, one store or several) for the rest of the pass; returns the undo, which
    restores each store's own connection and closes the pool's."""
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


#: The memory tools this module answers (each registry spec's handler names one of them).
HANDLED = (
    "read",
    "grep",
    "overview",
    "read_episode",
    "pool",
    "similar",
    "fixture",
    "replay",
    "land_capture",
    "check",
    "finish",
    "write",
    "edit",
    "delegate",
    "send",
    "messages",
    "wait",
    "who",
    "reply",
    "stop",
    "ask",
    "stage",
    "done",
)

__all__ = [
    *HANDLED,
    "WriterCalls",
    "HelperCalls",
    "in_code_json_tools",
    "stub_source",
    "thread_reads",
    "write_draft_manifest",
]


# --- one agent's persistent interpreter and its channel ---------------------------------------------------------------


def _default_start() -> Callable[..., Awaitable[Any]]:
    from unify.common.box_interpreter import BoxInterpreter  # type: ignore[import-not-found]

    return BoxInterpreter.start


def memory_profile(**fields: Any) -> Any:
    """The interpreter's ``MemoryProfile`` with *fields* (a plain namespace where the interpreter is not present)."""
    try:
        from unify.common.box_interpreter import MemoryProfile  # type: ignore[import-not-found]
    except ImportError:
        from types import SimpleNamespace

        return SimpleNamespace(**fields)
    return MemoryProfile(**fields)


def memory_globals(agent: str, memory_root: Path) -> dict:
    """The names bound in *agent*'s cells besides ``tools`` (which its stub binds): the profile's ``globals``.
    *memory_root* is the host directory bound at ``/memory``; ``memory`` is bound (lazily) when it holds the library
    package (``memory/__init__.py``)."""
    from unify.common import tool_registry

    facts = {"memory": (Path(memory_root) / "memory" / "__init__.py").is_file()}
    return {k: v for k, v in tool_registry.prebound(agent, facts).items() if k != "tools"}


def _open_roots(roots: dict[str, Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for box_path, host in roots.items():
        try:
            out[box_path] = os.open(Path(host), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError:
            continue
    return out


class InterpreterSession:
    """One memory agent's persistent confined interpreter (``BoxInterpreter``) and its tool channels.

    The interpreter starts at the agent's first cell. It takes a channel factory (every box it starts gets a new
    channel bound to the agent: its name, role, pass, roots and state; never anything from the box), starts the
    channel's ``serve`` and brackets each cell with ``cell_started``/``cell_ended``. :meth:`kill` stops the box from a
    handler (``finish``/``done`` judge nothing while it lives); the next cell then starts a new interpreter.
    :meth:`close` ends it at the pass's end.

    *tools* False (tools in code off: the agent's tools stay JSON tools): no channel and no ``tools`` stub, so the
    box has no route to the host at all; everything else is the same. Under ``UNIFY_CELL_STATE=fresh``
    (:func:`unify.common.box_interpreter.persistent` False) the interpreter is stopped after each cell, with no note,
    as the actor's is.
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
        profile: Any,
        start: Callable[..., Awaitable[Any]] | None = None,
        specs: dict | None = None,
        tools: bool = True,
        record: Any = None,
    ) -> None:
        """*specs*: tool specs by name for the channel (None: the shared registry's, as in a pass). *tools*: the
        box gets a tool channel and the ``tools`` stub (False: neither). *record*: the agent's view of the run record
        (``UNIFY_AGENTS_RUN_RECORD=on``: its comms tools' record), else None."""
        from unify.common.tool_channel import CallContext

        self.tools = bool(tools)
        self.allow = tuple(allow) if self.tools else ()
        self.binds, self.env, self.profile = binds, dict(env), profile
        self.cell_limit_s = float(cell_limit_s)
        self.on_event = on_event
        self.threads = threads
        self._start = start
        self._specs = specs
        # the channel's roots (descriptors the host's handlers resolve paths from); no channel, none opened
        self._root_fds = _open_roots(roots) if self.tools else {}
        self.binding = CallContext(
            agent=agent,
            name=name,
            role=role,
            pass_id=pass_id,
            request_id=None,
            roots=dict(self._root_fds),
            limits=None,
            counters=state,
            record=record,
            episodes=None,
            deadline=deadline,
            state=state,
        )
        self.channel: Any = None
        self.interp: Any = None
        self._row_cell: int | None = None
        self._killed_by_call = False
        self._note = ""
        self.stuck = False

    def _new_channel(self) -> Any:
        """The interpreter's channel factory: a fresh channel for every box it starts."""
        from unify.common.tool_channel import Channel

        if self.channel is not None and self.channel.stuck:
            self.stuck = True
        self.channel = Channel.open(
            self.binding,
            self.allow,
            self.on_event,
            specs=self._specs,
            threads=self.threads,
            row_cell=lambda: self._row_cell,
        )
        return self.channel

    async def kill(self) -> None:
        """Stop the box now (verified kill), from the handler of a call the box made; its reply is dropped. The next
        cell starts a new interpreter."""
        self._killed_by_call = True
        if self.channel is not None:
            self.channel.box_gone()
        interp, self.interp = self.interp, None
        if interp is not None:
            self._keep_stopped(interp)
            await interp.stop()
            self._note = RESTARTED

    async def measure_quota(self) -> str | None:
        """The /memory quota now (after write, edit, fixture); a breach stops the box. Without a box, the profile's
        measure alone (nothing to stop); None when the profile has none."""
        interp = self.interp
        measure = getattr(interp, "measure_quota", None)
        if measure is not None:
            return await measure()
        alone = getattr(self.profile, "measure", None)
        return await asyncio.to_thread(alone) if alone is not None else None

    def box_alive(self) -> bool:
        """Whether a box of this session is alive: the running interpreter's, or any interpreter it stopped (by an
        end call's :meth:`kill` or by :meth:`close`), so a box that survived its stop is seen by the check that keeps
        the gate from judging /memory."""
        for interp in (self.interp, *getattr(self, "_stopped_boxes", ())):
            alive = getattr(interp, "box_alive", None)
            if alive is not None and alive():
                return True
        return False

    def _keep_stopped(self, interp: Any) -> None:
        """Remember an interpreter this session stops, so :meth:`box_alive` still reads its box's liveness."""
        self.__dict__.setdefault("_stopped_boxes", []).append(interp)

    async def cell(self, code: str, cell_no: int) -> Any:
        """Run one cell: the interpreter's cell result, or None when the pass's wall clock ended first. *cell_no*:
        the cell's event row number (in-code calls' rows name it as their ``cell``)."""
        from unify.common.box_interpreter import persistent

        if self.interp is None:
            self.interp = await (self._start or _default_start())(
                agent=self.binding.agent,
                profile=self.profile,
                binds=self.binds,
                env=self.env,
                channel=self._new_channel if self.tools else None,
                stub_source=stub_source(self.allow, self.binding.agent) if self.tools else None,
                cell_limit_s=self.cell_limit_s,
            )
        self._row_cell = cell_no
        self._killed_by_call = False
        if self.channel is not None:
            self.channel.gone_during = None
        interp = self.interp
        try:
            try:
                res = await asyncio.wait_for(
                    interp.run_cell(code, call=cell_no),
                    timeout=max(1.0, self.binding.deadline - time.monotonic() + CELL_MARGIN_S),
                )
            except asyncio.TimeoutError:
                res = None
                await self.kill()
                self._killed_by_call = False
            if self._killed_by_call and self.channel is not None:
                # the call that stopped the box (finish, done) finishes its handler before the cell has ended
                await self.channel.settled(self.binding.deadline - time.monotonic())
            if self.channel is not None and self.channel.stuck:
                self.stuck = True
        finally:
            if not persistent():
                # UNIFY_CELL_STATE=fresh: every cell starts a new interpreter, as the setting already says
                self._note = ""
                interp, self.interp = self.interp, None
                if interp is not None:
                    self._keep_stopped(interp)
                    await interp.stop()
        return res

    def gone_during(self) -> tuple[str, bool] | None:
        """(tool, completed) of a call the box went away during in the last cell, else None."""
        return None if self.channel is None else self.channel.gone_during

    def take_note(self) -> str:
        note, self._note = self._note, ""
        return note

    async def close(self) -> None:
        """At the pass's end: stop the interpreter (verified kill, which closes its channel) and the root
        descriptors."""
        try:
            interp, self.interp = self.interp, None
            if interp is not None:
                # kept so box_alive() reads the stopped box's own liveness after the stop
                self._keep_stopped(interp)
                await interp.stop()
            if self.channel is not None and not self.channel.closed:
                await self.channel.close("ended")
        finally:
            for fd in self._root_fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._root_fds.clear()


#: How long past the pass's wall clock the host waits for a cell before it stops the box itself.
CELL_MARGIN_S = 5.0
#: What a cell's output says when the interpreter will start again (an end call stopped it).
RESTARTED = "(the interpreter was stopped by the end of the last call; variables are gone, files remain)"


def cell_parts(
    res: Any,
    finish_text: str | None,
    gone: tuple[str, bool] | None = None,
    note: str = "",
) -> tuple[str, str]:
    """A cell's output in two parts: the body (the interpreter's notes about an earlier restart, what the cell
    printed and its traceback), which the agent's own output handling may cut and save; and the notes after it (how
    it ended, a call the box went away during, the end call's result as the last line), never cut."""
    if res is None:
        body = "\n".join(x for x in (note, "(no output)") if x)
        tail = "\n(the pass's wall clock ended during the cell)"
    else:
        out = str(getattr(res, "output", "") or "")
        err = str(getattr(res, "error", "") or "")
        printed = (f"{out}\n{err}" if out and err else out or err).rstrip("\n") or "(no output)"
        before = [x for x in (note, *(getattr(res, "notes_before", None) or [])) if x]
        body = "\n".join([*before, printed])
        after = list(getattr(res, "notes_after", None) or [])
        tail = "".join("\n" + x for x in after)
    if gone is not None and finish_text is None:
        tool, completed = gone
        tail += (
            f"\n(the cell ended during tools.{tool}; that call "
            f"{'completed' if completed else 'did not run'})"
        )
    if finish_text is not None:
        tail += "\n" + finish_text
    return body, tail
