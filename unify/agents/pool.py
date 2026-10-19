"""Helpers: started by any agent, answered through the record, stopped with the request."""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Optional

from unify.agents.entry import MAX_TEXT_BYTES, RESERVED_NAMES, valid_name
from unify.agents.record import HARNESS, USER, PostRefused, Record
from unify.transcripts import scrub

StartHelper = Callable[[str, str, str, int], Awaitable[str]]
#: How many free names a pool on the run record tries when other processes keep taking the one it picked.
_NAME_TRIES = 20
_PLACEHOLDER = re.compile(
    r"^\s*(no-?op|n/?a|not applicable|none|nothing|todo|tbd)\s*[.!]?\s*$",
    re.I,
)


class Pool:
    def __init__(
        self,
        record: Record,
        *,
        start_helper: StartHelper,
        spawn_allowed: bool,
        spawn_refusal: str = "",
    ) -> None:
        self.record = record
        self._start_helper = start_helper
        self._spawn_allowed = spawn_allowed
        self._spawn_refusal = (
            spawn_refusal or "starting helpers is not allowed in this run"
        )
        self._tasks: dict[str, asyncio.Task] = {}
        #: every helper this pool started or tried to start, in order (the run record's limits count these)
        self._started: list[str] = []

    def _helpers(self) -> list[str]:
        if getattr(self.record, "run_wide", False):
            # the run record (unify/agents/run_record.py) holds the whole run's agents: this pool's limits count only
            # the helpers it started
            return list(self._started)
        return [n for n, p in self.record.participants.items() if p.spawner is not None]

    def _live(self) -> list[str]:
        rec = self.record
        if getattr(rec, "run_wide", False):
            return [n for n in self._started if rec.participants.get(n) is not None and rec.participants[n].state == "running"]
        return [n for n in rec.live_agents() if n != rec.root]

    async def spawn(
        self,
        spawner: str,
        request: str,
        name: Optional[str] = None,
    ) -> str:
        rec, opts = self.record, self.record.options
        # Any agent may start helpers; the record's limits (live and total)
        # bound them all, whoever starts them.
        if not self._spawn_allowed:
            raise PermissionError(self._spawn_refusal)
        if opts.max_total == 0:
            raise PermissionError("starting helpers is off in this run")
        request = str(request or "")
        if len("".join(request.split())) < 16 or _PLACEHOLDER.match(request):
            raise ValueError(
                "the request must say what to do (at least 16 characters)",
            )
        live = self._live()
        if len(live) >= opts.max_live:
            raise RuntimeError(
                f"{len(live)} helpers are running ({', '.join(live)}); "
                "wait for a reply or stop one first",
            )
        if len(self._helpers()) >= opts.max_total:
            scope = "this request or pass" if getattr(rec, "run_wide", False) else "this run"
            raise RuntimeError(
                f"{scope} has started its limit of {opts.max_total} helpers",
            )
        picked = name is None and getattr(rec, "run_wide", False)
        if picked:
            # a name no agent of the whole run has had (other processes name theirs too)
            name = rec.next_name("h")
        if name is None:
            k = 1
            while f"h{k}" in rec.participants:
                k += 1
            name = f"h{k}"
        if not valid_name(name) or name in RESERVED_NAMES or name in rec.participants:
            raise ValueError(
                f"{name!r} cannot be a helper's name: use lower-case letters, digits "
                "and dashes, not user, harness, all or root, and not a name already "
                "in use",
            )
        text = f"@{name} {request}"
        size = len(scrub(text).encode("utf-8"))
        if size > MAX_TEXT_BYTES:
            raise PostRefused(
                f"the request is {size} bytes with its mention and the limit is "
                f"{MAX_TEXT_BYTES}; write it to a file and give the helper its path",
            )
        if opts.max_entries is not None and len(rec.entries) >= opts.max_entries:
            raise PostRefused(f"the record is full ({opts.max_entries} entries)")
        for _ in range(_NAME_TRIES):
            try:
                participant = rec.add_agent(name, spawner=spawner)
                break
            except PostRefused:
                if not picked:
                    raise
                # another process took the name between the look and the join: the next free one
                name = rec.next_name("h")
                text = f"@{name} {request}"
        else:
            raise PostRefused("no free helper name could be taken; try again")
        self._started.append(name)
        try:
            entry, _ = rec.append(spawner, text)
        except Exception:
            # The name stays taken (it is never reused); the helper never ran.
            participant.state = "failed"
            raise
        participant.request_seq = entry.seq
        # The helper's first message already carries its request.
        rec.set_cursor(name, entry.seq)
        self._tasks[name] = asyncio.get_running_loop().create_task(
            self._run(name, spawner, request, entry.seq),
            name=f"unify-agents:{name}",
        )
        return name

    async def _run(
        self,
        name: str,
        spawner: str,
        request: str,
        request_seq: int,
    ) -> None:
        participant = self.record.participants[name]
        try:
            text = await self._start_helper(name, spawner, request, request_seq)
        except asyncio.CancelledError:
            if participant.state == "running":
                participant.state = "stopped"
            raise
        except Exception as exc:
            participant.state = "failed"
            self.record.append_harness(
                HARNESS,
                f"@{spawner} {name} ended without replying: {type(exc).__name__}: {exc}",
                kind="system",
                mentions=[spawner],
            )
            return
        participant.state = "replied"
        self.record.append_harness(
            name,
            str(text or ""),
            kind="reply",
            mentions=[spawner],
            full_text_at="this helper's transcript",
        )

    def stop(self, by: str, name: str, reason: str = "") -> None:
        participant = self.record.participants.get(name)
        if participant is None or participant.spawner is None:
            raise ValueError(f"there is no helper named {name!r}")
        if by != participant.spawner and by not in (USER, HARNESS):
            raise PermissionError(f"only the agent that started {name} can stop it")
        if participant.state != "running":
            return
        text = f"@{name} stop" + (f": {reason}" if reason else "")
        self.record.append_harness(by, text, kind="cancel", mentions=[name])
        participant.state = "stopped"
        task = self._tasks.get(name)
        if task is not None:
            task.cancel()

    def list(self, spawner: str) -> list[dict]:
        return [
            {
                "name": p.name,
                "state": p.state,
                "request_seq": p.request_seq,
                "last_seq": p.last_seq,
            }
            for p in self.record.participants.values()
            if p.spawner == spawner
        ]

    async def finish_request(
        self,
        reply_text: str,
        *,
        reason: str = "the main agent replied",
    ) -> dict[str, str]:
        """Record the main agent's reply to the user, then end every helper still running."""
        self.record.append_harness(
            self.record.root,
            str(reply_text or ""),
            kind="reply",
            mentions=[USER],
            full_text_at="the session's result",
        )
        return await self.close(reason)

    async def close(self, reason: str, timeout: float = 30.0) -> dict[str, str]:
        """Stop running helpers and check that each task has ended (verified termination)."""
        for name, task in self._tasks.items():
            participant = self.record.participants[name]
            if not task.done() and participant.state == "running":
                participant.state = "stopped"
                self.record.append_harness(
                    HARNESS,
                    f"@{participant.spawner} {name} stopped: {reason}",
                    kind="system",
                    mentions=[participant.spawner],
                )
                task.cancel()
        if self._tasks:
            await asyncio.wait(list(self._tasks.values()), timeout=timeout)
        return {
            n: ("ended" if t.done() else "still running")
            for n, t in self._tasks.items()
        }
