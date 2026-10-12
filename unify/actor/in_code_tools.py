"""The actor's and a teammate's tools called from code, for one request (``UNIFY_TOOLS_IN_CODE=on``).

With the switch on, the request's one model-API tool is ``execute_code(code)``. Each cell runs in the agent's
persistent interpreter (:class:`unify.common.box_interpreter.BoxInterpreter` over the agent's sandboxed
:class:`~unify.actor.execution.worker.PythonWorker`), and every other tool is a call ``tools.<name>(...)`` from the
cell, through the tool channel (:mod:`unify.common.tool_channel`) to a host handler of
:mod:`unify.actor.tool_handlers`. What an agent may call is the registry's allow-list for it
(:func:`unify.actor.tool_handlers.actor_binding`); who is calling comes from the binding, never from the box.

One :class:`InCodeTools` serves one request of one agent:

* the binding: the agent (``actor`` or ``teammate``), its name on the agent record, the request's id, the workspace
  as an open directory descriptor under the box root ``/workspace``, the record
  (:class:`unify.common.record_adapters.AgentRecordAdapter`) and the request's state
  (:class:`unify.actor.tool_handlers.ActorRequestState`: the loop's reply slot that ``tools.finish`` fills, the
  response format's check, the storage review's starter);
* the channel table: each allowed tool's registry spec with the actor's handler (``tool_registry.handler_for``) and
  the agent's own parameter schema (``tool_registry.strict_parameters``);
* the box module ``tools``: ``tool_registry.render_stub`` for the allow-list, built in the box when the interpreter
  starts;
* the evidence: ``tool-events.jsonl`` in the request's own folder (:func:`request_folder`, under the harness's state
  directory, which no box sees), one ``tool_call`` row per cell and one per in-code call, written by the host as the
  memory pass writes its rows (a file created new, never through a link, then appended);
* the end (:meth:`InCodeTools.close`): the interpreter is stopped with the verified kill, the handler threads are
  waited for, and host work a call started (a storage review) is awaited, or cancelled when the request did not end
  normally; each is checked to have ended.

``UNIFY_CELL_STATE=persistent`` keeps the interpreter between cells; ``fresh`` stops it after every cell, so each
cell starts in a new one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import os
import stat
import time
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "EventFile",
    "InCodeTools",
    "SESSIONS",
    "enabled",
    "open_for_act",
    "request_folder",
]

#: The request has no wall clock of its own (the loop bounds it); this only keeps the channel's deadline arithmetic
#: finite, and is never reached by a request.
NO_WALL_CLOCK_S = 30 * 86400.0
#: How long the end of a request waits for a handler thread still running (a handler past its deadline).
THREADS_GRACE_S = 10.0
#: Every session opened in this process (weakly held), so a test can check each one ended.
SESSIONS: "weakref.WeakSet[InCodeTools]" = weakref.WeakSet()


def enabled() -> bool:
    """Whether ``UNIFY_TOOLS_IN_CODE=on``."""
    from unify.common import in_code_switch

    return in_code_switch.tools_in_code()


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_request_id() -> str:
    """A UTC stamp and 128 random bits: unique across processes and hosts."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex}"


def request_folder(request_id: str) -> Path:
    """The request's own folder for its evidence, under the harness's state directory (hidden from every box)."""
    from unify.db import store_home

    return store_home() / "requests" / request_id


class EventFile:
    """``<folder>/tool-events.jsonl``, written as the memory pass writes its own (``WriterPass._write_event``): the
    file is created new at the first row (never through a link, never over an existing file), then appended without
    following a link; a failure stops the file, never the request."""

    NAME = "tool-events.jsonl"

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._path: Optional[Path] = None
        self._broken = False

    @property
    def path(self) -> Path:
        return self.folder / self.NAME

    def append(self, row: dict) -> None:
        if self._broken:
            return
        try:
            if self._path is None:
                self._broken = True
                for d in (self.folder.parent, self.folder):
                    d.mkdir(parents=True, exist_ok=True)
                    if d.is_symlink() or not d.is_dir():
                        return
                path = self.path
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            else:
                path = self._path
                flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC
            fd = os.open(path, flags, 0o644)
            with os.fdopen(fd, "a") as fh:
                if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
                    self._broken = True
                    return
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            self._path, self._broken = path, False
        except OSError:
            self._broken = True  # a measurement never stops a request


def channel_specs(agent: str, allow: tuple[str, ...]) -> dict[str, Any]:
    """The channel's spec table for *agent*: each allowed host tool's registry spec with the handler that serves
    *agent* (``tool_registry.handler_for``) and *agent*'s own parameter schema (``tool_registry.strict_parameters``).
    Tools that run in the box (``sh``) are not on it: the stub runs them."""
    from unify.common import tool_registry as R

    out: dict[str, Any] = {}
    for name in allow:
        spec = R.registry()[name]
        if spec.kind == "local":
            continue
        handler, blocking = R.handler_for(agent, name)
        out[name] = dataclasses.replace(
            spec,
            handler=handler,
            blocking=blocking,
            parameters=R.strict_parameters(agent, name),
        )
    return out


class InCodeTools:
    """One request's in-code tools for the actor or a teammate (module docstring)."""

    def __init__(
        self,
        *,
        agent: str,
        name: str,
        record: Any,
        workspace_root: str,
        shadow: dict,
        cell_limit_s: Optional[float],
        can_store: bool = False,
        store_skills: Optional[Callable[[str], Awaitable[Any]]] = None,
        validate: Optional[Callable[[Any], str]] = None,
        spawner: Optional[str] = None,
        policy: Any = None,
        request_id: Optional[str] = None,
        events_folder: Optional[Path] = None,
        start: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> None:
        from unify.actor import tool_handlers as H
        from unify.common import tool_registry as R
        from unify.common.tool_channel import HandlerThreads

        self.agent = agent
        self.request_id = request_id or new_request_id()
        self.events = EventFile(events_folder or request_folder(self.request_id))
        self.shadow = shadow
        self.cell_limit_s = float(cell_limit_s) if cell_limit_s else math.inf
        self._start = start
        self._root_fd = os.open(workspace_root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        self.state = H.ActorRequestState(
            validate=validate,
            store_skills=store_skills if can_store else None,
            workspace_root=str(workspace_root),
            policy=policy,
            spawner=spawner,
        )
        self.binding, self.allow = H.actor_binding(
            agent=agent,
            name=name,
            request_id=self.request_id,
            roots={H.WORKSPACE: self._root_fd},
            record=record,
            deadline=time.monotonic() + NO_WALL_CLOCK_S,
            state=self.state,
            can_store=can_store and store_skills is not None,
        )
        self.specs = channel_specs(agent, self.allow)
        self.channel_allow = tuple(n for n in self.allow if n in self.specs)
        self.stub = R.render_stub(self.allow, agent)
        self.threads = HandlerThreads()
        self.channel: Any = None
        self.interp: Any = None
        self.calls = 0
        self._cell_call: Optional[int] = None
        self._lock = asyncio.Lock()
        self.closed = False
        SESSIONS.add(self)

    # -- the model-API tool ------------------------------------------------------------------------------------

    def tool(self) -> Any:
        """The request's ``execute_code`` for the loop: one argument, ``code`` (the schema the model is sent is the
        registry's, ``tool_registry.json_tools``)."""
        from unify.common.tool_spec import ToolSpec

        session = self

        async def execute_code(code: str) -> Any:
            return await session.run_cell(code)

        execute_code.__doc__ = _execute_code_doc()
        return ToolSpec(fn=execute_code, display_label="Running code")

    # -- the channel and the interpreter -----------------------------------------------------------------------

    def _on_event(self, row: dict) -> None:
        if row.get("event") == "tool_call":
            row["call"] = self._next_call()
        self.events.append(row)

    def _next_call(self) -> int:
        n = self.calls
        self.calls += 1
        return n

    def _new_channel(self) -> Any:
        from unify.common.tool_channel import Channel

        self.channel = Channel.open(
            self.binding,
            self.channel_allow,
            self._on_event,
            specs=self.specs,
            threads=self.threads,
            row_cell=lambda: self._cell_call,
        )
        return self.channel

    async def _interpreter(self) -> Any:
        if self.interp is None:
            from unify.common.box_interpreter import ActorProfile, BoxInterpreter

            start = self._start or BoxInterpreter.start
            self.interp = await start(
                agent=self.agent,
                profile=ActorProfile(shadow=self.shadow),
                binds=None,
                env={},
                channel=self._new_channel,
                stub_source=self.stub,
                cell_limit_s=self.cell_limit_s,
            )
        return self.interp

    def box_alive(self) -> bool:
        interp = self.interp
        return bool(interp is not None and interp.box_alive())

    def threads_alive(self) -> bool:
        return self.threads.alive()

    # -- cells -------------------------------------------------------------------------------------------------

    async def run_cell(self, code: str) -> Any:
        """Run one cell; its output as the loop shows it (an ``ExecutionResult``). Its event row is written when it
        ends, with the number reserved when it started (the in-code calls it made are numbered after it)."""
        from unify.common._async_tool import bound_request, cell_reply
        from unify.common.box_interpreter import persistent

        if self.closed:
            raise RuntimeError("this request's tools have ended")
        async with self._lock:
            # the loop's slot for the turn's reply, which tools.finish fills (the loop that runs this call set it)
            self.state.reply_slot = cell_reply.current()
            call = self._next_call()
            self._cell_call = call
            started = time.monotonic()
            res: Any = None
            raised: Optional[BaseException] = None
            try:
                interp = await self._interpreter()
                if self.channel is not None:
                    self.channel.gone_during = None
                bound_request.install(self.shadow)
                res = await interp.run_cell(code, call=call)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # a cell's reply() (a BaseException) or a control exception
                raised = exc
            finally:
                self._cell_call = None
                if not persistent() and self.interp is not None:
                    interp, self.interp = self.interp, None
                    await interp.stop()
            out = self._result(res, raised, started)
            self._cell_row(call, code, out)
            if raised is not None and not _is_reply(raised):
                raise raised
            return out

    def _result(self, res: Any, raised: Optional[BaseException], started: float) -> Any:
        from unify.actor.execution.types import ExecutionResult, TextPart, parts_to_text
        from unify.common._async_tool import cell_reply

        stdout: list = []
        stderr: list = []
        error: Optional[str] = None
        result: Any = None
        if res is not None:
            stdout += [TextPart(text=n + "\n") for n in res.notes_before]
            parts = list(res.parts or [])
            stdout += parts
            printed = parts_to_text(parts)
            rest = res.output[len(printed) :].lstrip("\n") if res.output.startswith(printed) else ""
            if rest:
                stderr.append(TextPart(text=rest))
            error, result = res.error, res.result
            after = list(res.notes_after)
        else:
            after = []
        gone = self.channel.gone_during if self.channel is not None else None
        if gone is not None:
            tool, completed = gone
            after.append(
                f"(the cell ended during tools.{tool}; that call {'completed' if completed else 'did not run'})",
            )
        if raised is not None and _is_reply(raised):
            refused = cell_reply.deliver(raised.text, raised.from_value)
            error = refused if refused is not None else error
        stdout += [TextPart(text=n + "\n") for n in after]
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            result=_plain(result),
            error=error,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    def _cell_row(self, call: int, code: str, out: Any) -> None:
        from unify.actor.execution.types import parts_to_text
        from unify.common.tool_channel import canonical_args

        text = parts_to_text(out.stdout) + parts_to_text(out.stderr) + (out.error or "")
        args = canonical_args({"code": code})
        row = {
            "event": "tool_call",
            "agent": self.binding.agent,
            "name": self.binding.name,
            "pass_id": None,
            "request_id": self.request_id,
            "call": call,
            "utc": _utc(),
            "tool": "execute_code",
            "status": "ok",
            "refusal": None,
            "paths": [],
            "args_bytes": len(args),
            "result_bytes": len(text.encode("utf-8", errors="replace")),
            "result_truncated": False,
            "usage": None,
            "via": "model",
            "error": out.error is not None,
        }
        self.events.append(row)

    # -- the end of the request --------------------------------------------------------------------------------

    async def close(self, *, completed: bool = True) -> None:
        """Stop the interpreter (verified kill), wait for every handler thread, and end the host work calls started:
        awaited when the request ended normally (*completed*), else cancelled; each is checked to have ended. Safe
        to repeat. Raises when something could not be ended."""
        if self.closed:
            return
        self.closed = True
        problems: list[str] = []
        try:
            interp, self.interp = self.interp, None
            if interp is not None:
                await interp.stop()
                if interp.box_alive():
                    problems.append("the interpreter's box is still alive")
            if self.channel is not None and not self.channel.closed:
                await self.channel.close("ended", record=False)
            if not await self.threads.wait_idle(THREADS_GRACE_S):
                problems.append("a tool handler thread is still running")
            tasks = list(self.state.host_tasks)
            if tasks:
                if not completed:
                    for t in tasks:
                        t.cancel()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                        logger.warning("work a tool call started ended with %s", type(r).__name__)
                if any(not t.done() for t in tasks):
                    problems.append("work a tool call started is still running")
        finally:
            try:
                os.close(self._root_fd)
            except OSError:
                pass
        if problems:
            raise RuntimeError("; ".join(problems))


#: The objects a cell held before that the registry's tools replace with the switch on: a cell calls
#: ``tools.install``, ``tools.read``, ``tools.grep`` and the record tools instead.
REPLACED_OBJECTS = ("install", "read_file", "grep", "record", "agents")


def open_for_act(
    *,
    agents: Any,
    sandbox: Any,
    store_skills: Optional[Callable[[str], Awaitable[Any]]],
    response_format: Any,
    cell_limit_s: Optional[float],
) -> Optional[InCodeTools]:
    """The in-code tools of one ``act()`` (the actor's request, or a teammate's when *agents* is a teammate's
    binding), or None while ``UNIFY_TOOLS_IN_CODE`` is off. *agents*: the act's
    :class:`unify.agents.binding.Binding`; *sandbox*: its ``PythonExecutionSession`` (the namespace the worker's
    proxy serves, without the objects :data:`REPLACED_OBJECTS` names); *store_skills*: the storage review's
    starter, None where the request may not store; *response_format*: the request's, whose check ``tools.finish``
    applies."""
    if not enabled():
        return None
    from unify import sandbox as _sandbox
    from unify.actor import tool_handlers as H
    from unify.common.record_adapters import AgentRecordAdapter

    validate = None
    if response_format is not None:
        from unify.common._async_tool.response_format import normalize_response_format

        validate = H.response_validator(normalize_response_format(response_format))
    for holder in (sandbox.global_state, getattr(sandbox, "core_globals", None)):
        if isinstance(holder, dict):
            for key in REPLACED_OBJECTS:
                holder.pop(key, None)
    pool = agents.pool
    root = pool.record.root
    participant = pool.record.participants.get(agents.name)
    policy = _sandbox.build_policy()
    return InCodeTools(
        agent="actor" if agents.name == root else "teammate",
        name=agents.name,
        record=AgentRecordAdapter(pool),
        workspace_root=str(policy.workspace),
        shadow=sandbox.global_state,
        cell_limit_s=cell_limit_s,
        can_store=store_skills is not None and agents.name == root,
        store_skills=store_skills,
        validate=validate,
        spawner=getattr(participant, "spawner", None),
        policy=policy,
    )


def _is_reply(exc: BaseException) -> bool:
    from unify.actor.execution.worker_child import CellReply

    return isinstance(exc, CellReply)


def _plain(value: Any) -> Any:
    """A cell's result as the loop can show it (JSON values as they are, anything else as its text)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)[:4000]


def _execute_code_doc() -> str:
    from unify.common.tool_registry import EXECUTE_CODE

    return EXECUTE_CODE.description
