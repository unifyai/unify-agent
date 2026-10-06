"""Helpers: started by the main agent, answered through the record, stopped with the request."""

from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Optional

from unify.agents.entry import MAX_TEXT_BYTES, RESERVED_NAMES, valid_name
from unify.agents.record import HARNESS, USER, Record

StartHelper = Callable[[str, str, str, int], Awaitable[str]]
_PLACEHOLDER = re.compile(
    r"^\s*(no-?op|n/?a|not applicable|none|nothing|todo|tbd)\s*[.!]?\s*$",
    re.I,
)


def _fit(text: str) -> str:
    """A reply cut to the record's limit, saying where the rest is."""
    data = text.encode("utf-8")
    if len(data) <= MAX_TEXT_BYTES:
        return text
    note = (
        f"\n[cut: {len(data)} bytes in all; the full reply is in this helper's "
        "transcript]"
    )
    room = MAX_TEXT_BYTES - len(note.encode("utf-8"))
    return data[:room].decode("utf-8", errors="ignore") + note


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

    def _helpers(self) -> list[str]:
        return [n for n, p in self.record.participants.items() if p.spawner is not None]

    async def spawn(self, spawner: str, task: str, name: Optional[str] = None) -> str:
        rec, opts = self.record, self.record.options
        if spawner != rec.root:
            raise PermissionError("only the main agent can start helpers in this run")
        if not self._spawn_allowed:
            raise PermissionError(self._spawn_refusal)
        if opts.max_total == 0:
            raise PermissionError("starting helpers is off in this run")
        task = str(task or "")
        if len("".join(task.split())) < 16 or _PLACEHOLDER.match(task):
            raise ValueError("the task must say what to do (at least 16 characters)")
        live = [n for n in rec.live_agents() if n != rec.root]
        if len(live) >= opts.max_live:
            raise RuntimeError(
                f"{len(live)} helpers are running ({', '.join(live)}); "
                "wait for a reply or stop one first",
            )
        if len(self._helpers()) >= opts.max_total:
            raise RuntimeError(
                f"this run has started its limit of {opts.max_total} helpers",
            )
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
        rec.add_agent(name, spawner=spawner)
        entry, _ = rec.append(spawner, f"@{name} {task}")
        rec.participants[name].task_seq = entry.seq
        self._tasks[name] = asyncio.get_running_loop().create_task(
            self._run(name, spawner, task, entry.seq),
            name=f"unify-agents:{name}",
        )
        return name

    async def _run(self, name: str, spawner: str, task: str, task_seq: int) -> None:
        participant = self.record.participants[name]
        try:
            text = await self._start_helper(name, spawner, task, task_seq)
        except asyncio.CancelledError:
            if participant.state == "running":
                participant.state = "stopped"
            raise
        except Exception as exc:
            participant.state = "failed"
            self.record.append(
                HARNESS,
                f"@{spawner} {name} ended without replying: {type(exc).__name__}: {exc}",
                kind="system",
                mentions=[spawner],
            )
            return
        participant.state = "replied"
        self.record.append(
            name,
            _fit(str(text or "")),
            kind="reply",
            mentions=[spawner],
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
        self.record.append(by, text, kind="cancel", mentions=[name])
        participant.state = "stopped"
        task = self._tasks.get(name)
        if task is not None:
            task.cancel()

    def list(self, spawner: str) -> list[dict]:
        return [
            {
                "name": p.name,
                "state": p.state,
                "task_seq": p.task_seq,
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
        self.record.append(
            self.record.root,
            _fit(str(reply_text or "")),
            kind="reply",
            mentions=[USER],
        )
        return await self.close(reason)

    async def close(self, reason: str, timeout: float = 30.0) -> dict[str, str]:
        """Stop running helpers and check that each task has ended (verified termination)."""
        for name, task in self._tasks.items():
            participant = self.record.participants[name]
            if not task.done() and participant.state == "running":
                participant.state = "stopped"
                self.record.append(
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
