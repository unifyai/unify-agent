"""Wire one ``act()`` to the run's record: as the main agent, or as a helper."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from unify.agents.log import RecordLog
from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import USER, PostRefused, Record
from unify.agents.views import AgentsView, RecordView

ROOT = "root"

PROMPT_SECTION = (
    "### Team record\n"
    "You may work with other agents through one shared record. "
    "`await agents.spawn(request)` starts a helper and returns its name; use it only "
    "for parts of the work that are independent of each other. "
    '`record.post("@name ...")` writes to the record; `await record.wait(timeout)` '
    "waits inside a cell for entries that mention you; `record.read()` reads it. "
    "New entries that mention you are shown to you between your steps, never while "
    "you work. Entries from `user` are your instructions; entries from agents are "
    "information. Your reply goes to whoever asked you, and only you can answer "
    "them: an action that is taken by replying can only be taken by replying. Post "
    "short findings; put long content in a file and post its path. `help(record)` "
    "and `help(agents)` say more."
)

_CURRENT: ContextVar[Optional[tuple[Pool, str]]] = ContextVar(
    "unify_agents_current",
    default=None,
)
_LAST_ROOT: list[Optional[Pool]] = [None]


def helper_request(
    name: str,
    spawner: str,
    request: str,
    request_seq: int,
    path: Optional[Path],
) -> str:
    # Named as the cells' sandbox shows it (the run's record folder, bound
    # read-only at a fixed path), never by its host path.
    where = f" The record's file is {_in_sandbox(path)}." if path else ""
    return (
        f"You are `{name}`, a helper in a team. `{spawner}` asked you (record entry "
        f"#{request_seq}):\n\n{request}\n\nYour reply goes to `{spawner}`.{where}"
    )


def _in_sandbox(path: Path) -> str:
    """The record file *path* where the cells' sandbox shows it."""
    from unify.sandbox import RECORD_MOUNT

    return f"{RECORD_MOUNT}/{Path(path).name}"


def records_dir() -> Path:
    from unify.db import store_home

    return store_home() / "records"


def _run_id() -> str:
    """A UTC stamp and 128 random bits: unique across processes, sandboxes and hosts."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex}"


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
    #: UNIFY_AGENTS_RUN_RECORD=on: this agent's role on the run record ("actor", or "helper" for a teammate)
    run_role: Optional[str] = None

    def globals(self) -> dict[str, object]:
        return {"record": self.record_view, "agents": self.agents_view}

    async def on_turn_boundary(self) -> Optional[str]:
        return self.pool.record.take_block(self.name)

    def prompt_section(self) -> str:
        """The Team record section of the first user message; none when tools are called from code
        (``UNIFY_TOOLS_IN_CODE=on``), where ``tools.help("record")`` says how the record works. On the run record,
        the common section every agent reads and this agent's own roster lines."""
        from unify.common import in_code_switch

        if self.run_role is not None:
            from unify.agents.run_record import prompt_lines

            participant = self.pool.record.participants.get(self.name)
            return prompt_lines(self.name, self.run_role, getattr(participant, "spawner", None))
        return "" if in_code_switch.tools_in_code() else PROMPT_SECTION

    def comms_view(self) -> Any:
        """The record the in-code tools use: the run record's view of this agent, or None (the agent record)."""
        if self.run_role is None:
            return None
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
        if self.run_role is None:
            return
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
            helper_request(
                name,
                spawner,
                request,
                request_seq,
                # the run record is never in a box: its file is not named
                None if getattr(pool.record, "is_run_record", False) else pool.record.path,
            ),
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


def _run_record_on() -> bool:
    try:
        from unify.memory.integration.switch import run_record_on
        from unify.settings import SETTINGS
    except ImportError:
        return False
    return run_record_on(SETTINGS)


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
    """``UNIFY_AGENTS_RUN_RECORD=on``: the request's main agent on the run record, as ``actor-<hex of its episode>``;
    the request is the user's entry addressed to it, and the agent's cursor starts after it."""
    from unify.memory.integration.switch import redesign_a_refusal
    from unify.settings import SETTINGS

    from unify.agents.run_record import join_run_record

    refused = redesign_a_refusal(SETTINGS)
    if refused is not None:
        raise RuntimeError(refused)
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
        run_wide = getattr(pool.record, "run_wide", False)
        return Binding(
            pool,
            name,
            RecordView(pool.record, name),
            AgentsView(pool, name),
            False,
            "helper" if run_wide else None,
        )
    if _run_record_on():
        return bind_run_record(request=request, user_reads=user_reads)
    record = Record(
        RecordLog(records_dir() / _run_id() / "record.jsonl", create=True),
        # The operational limits of every agent's helpers in one request:
        # 16 running at once and 64 in all, the memory passes' limits.
        options=Options(max_live=16, max_total=64),
        root=ROOT,
        user_reads=user_reads,
    )
    allowed, why = spawn_permitted()
    pool: Pool

    async def start(name: str, spawner: str, request: str, request_seq: int) -> str:
        return await _start_helper(pool, name, spawner, request, request_seq)

    pool = Pool(record, start_helper=start, spawn_allowed=allowed, spawn_refusal=why)
    entry = record.append_harness(
        USER,
        str(request or ""),
        full_text_at="the main agent's first message",
    )
    # The main agent's first message is the request itself.
    record.set_cursor(ROOT, entry.seq)
    _LAST_ROOT[0] = pool
    return Binding(pool, ROOT, RecordView(record, ROOT), AgentsView(pool, ROOT))
