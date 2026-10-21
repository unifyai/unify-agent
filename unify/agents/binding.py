"""Wire one ``act()`` to the run record: as the request's main agent, or as a helper."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from unify.agents.pool import Pool
from unify.agents.record import USER, PostRefused
from unify.agents.views import AgentsView, RecordView

_CURRENT: ContextVar[Optional[tuple[Pool, str]]] = ContextVar(
    "unify_agents_current",
    default=None,
)
_LAST_ROOT: list[Optional[Pool]] = [None]


def helper_request(name: str, spawner: str, request: str, request_seq: int) -> str:
    """A helper's first message: who it is, who asked it (and in which record entry) and what."""
    return (
        f"You are `{name}`, a helper in a team. `{spawner}` asked you (record entry "
        f"#{request_seq}):\n\n{request}\n\nYour reply goes to `{spawner}`."
    )


def spawn_permitted() -> tuple[bool, str]:
    """Helpers start only where each agent's code runs in its own confined
    worker: always, outside a test that runs Python in this process."""
    from unify.actor.execution import worker

    if worker.enabled():
        return True, ""
    return False, (
        "starting helpers needs Python in the sandboxed worker, so that each "
        "agent's code runs in its own confined process"
    )


def current_root_pool() -> Optional[Pool]:
    """The pool of the most recent main agent in this process (for hosts and tests)."""
    return _LAST_ROOT[0]


@dataclass
class Binding:
    pool: Pool
    name: str
    record_view: RecordView
    agents_view: AgentsView
    #: whether this act() is the request's main agent (a teammate a pool started is not)
    is_main: bool = True
    #: this agent's role on the run record: "actor", or "helper" for a teammate
    run_role: str = "actor"

    def globals(self) -> dict[str, object]:
        return {"record": self.record_view, "agents": self.agents_view}

    async def on_turn_boundary(self) -> Optional[str]:
        return self.pool.record.take_block(self.name)

    def prompt_section(self) -> str:
        """The run record's section of the first user message: the common section every agent reads and this agent's
        own roster lines."""
        from unify.agents.run_record import prompt_lines

        participant = self.pool.record.participants.get(self.name)
        return prompt_lines(self.name, self.run_role, getattr(participant, "spawner", None),
                            home=getattr(self.pool.record, "home", None))  # fmt: skip

    def comms_view(self) -> Any:
        """The record the in-code tools use: the run record's view of this agent."""
        from unify.agents.run_record import RunRecordView

        participant = self.pool.record.participants.get(self.name)
        return RunRecordView(
            self.pool.record,
            self.name,
            self.run_role,
            pool=self.pool,
            spawner=getattr(participant, "spawner", None),
        )

    def leave(self) -> None:
        """At the request's end, on the run record: this agent has ended (its name stays taken)."""
        try:
            self.pool.record.leave(self.name)
        except Exception:  # noqa: BLE001 - leaving never costs the caller its result
            logging.getLogger(__name__).warning("could not leave the run record", exc_info=True)


async def _start_helper(
    pool: Pool,
    name: str,
    spawner: str,
    request: str,
    request_seq: int,
) -> str:
    """Run one helper as a sub-actor bounded by the spawner's grants; return its reply."""
    from unify.actor.environments.actor import _build_inner_actor

    actor, guidelines = _build_inner_actor(
        guidelines=None,
        prompt_guidance=None,
        guidance_scope=None,
        prompt_functions=None,
        discovery_scope=None,
        timeout=None,
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
    )
    token = _CURRENT.set((pool, name))
    try:
        handle = await actor.act(
            helper_request(name, spawner, request, request_seq),
            guidelines=guidelines,
            persist=False,
            can_store=False,
            clarification_enabled=False,
        )
    finally:
        _CURRENT.reset(token)
    try:
        return str(await handle.result())
    finally:
        await actor.close()


def _episode() -> Optional[str]:
    """The memory run's episode id when one is open, else None."""
    try:
        from unify.memory.integration import hooks

        run = hooks._run()
        eid = getattr(run, "episode_id", "") if run is not None else ""
    except Exception:  # noqa: BLE001 - no memory run here
        eid = ""
    return str(eid) if eid else None


def _run_home() -> Path:
    from unify.db import store_home

    return Path(store_home())


def bind_run_record(*, request: str, user_reads: bool) -> Binding:
    """The request's main agent on the run record, as ``actor-<hex of its episode>``; the request is the user's entry
    addressed to it, and the agent's cursor starts after it."""
    from unify.agents.run_record import join_run_record

    episode = _episode()
    # the actor's name: 4 hex digits of its episode id (a random id where no memory run is open)
    naming = episode or uuid.uuid4().hex[:16]
    record = join_run_record(_run_home())
    record.user_reads = user_reads
    name = record.unique_name("actor", naming)
    for _ in range(20):
        try:
            record.join(name, "actor", episode=episode)
            break
        except PostRefused:
            name = record.unique_name("actor", naming)
    else:
        raise PostRefused("no free name for the actor on the run record")
    record.root = name
    allowed, why = spawn_permitted()
    pool: Pool

    async def start(helper: str, spawner: str, task: str, request_seq: int) -> str:
        return await _start_helper(pool, helper, spawner, task, request_seq)

    pool = Pool(record, start_helper=start, spawn_allowed=allowed, spawn_refusal=why)
    entry = record.append_harness(
        USER,
        str(request or ""),
        mentions=[name],
        full_text_at="the main agent's first message",
    )
    record.set_cursor(name, entry.seq)
    _LAST_ROOT[0] = pool
    return Binding(pool, name, RecordView(record, name), AgentsView(pool, name), True, "actor")


def bind_for_act(*, request: str, user_reads: bool) -> Binding:
    """The helper this act() was started for, or a new main agent."""
    current = _CURRENT.get()
    if current is not None:
        pool, name = current
        return Binding(pool, name, RecordView(pool.record, name), AgentsView(pool, name), False, "helper")
    return bind_run_record(request=request, user_reads=user_reads)
