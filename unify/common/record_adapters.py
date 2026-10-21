"""The two records behind the record tools, each seen through :class:`unify.common.tool_registry.Record`.

* :class:`PassRecordAdapter`: a memory pass's in-memory record (:class:`unify.memory.messaging.PassRecord`), shared
  by the writer or curator and the helpers it starts.
* :class:`AgentRecordAdapter`: the run's agent record (:mod:`unify.agents.record` and its :class:`Pool`), shared by
  the actor and its teammates.

Neither changes what the record does: each method calls the record's own functions and turns a refusal into
:class:`unify.common.tool_registry.RecordRefused` with the record tool's form code, so the tools' handlers answer
both records the same way: its ``text`` is either the finished refusal (a ``Refusal``, which keeps its form) or the
reason, which the handler fills into the tool's template for that form. Entries are returned as plain dicts (``seq``, ``ts``, ``author``, ``kind``, ``text``,
``mentions``).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from unify.common.tool_registry import RecordRefused


def _entry(e: Any) -> dict:
    return e.to_dict() if hasattr(e, "to_dict") else dict(e)


class PassRecordAdapter:
    """A memory pass's record (:class:`unify.memory.messaging.PassRecord`)."""

    def __init__(self, pr: Any) -> None:
        self.pr = pr

    def post(self, sender: str, text: str, to: Optional[list] = None) -> dict:
        from unify.memory import messaging

        if to:
            text = " ".join(f"@{n}" for n in to) + " " + str(text)
        info: dict = {}
        reply, sent_to = messaging.post(self.pr, sender, text, messaging.SEND_SPEC, info)
        form = getattr(reply, "form", None)
        if form is not None:
            raise RecordRefused(form, reply)
        return {"seq": info.get("seq"), "to": list(sent_to), "text": str(reply)}

    def unseen(self, reader: str) -> list:
        from unify.memory import messaging

        return [_entry(e) for e in messaging.take(self.pr, reader)]

    async def wait(self, reader: str, timeout: float) -> list:
        """The unseen entries for *reader*; waits for one at most *timeout* seconds and never past the pass's wall
        clock, and returns at once when no other agent is running (none can arrive) or *reader* is asked to stop.
        """
        from unify.memory import messaging

        pr = self.pr
        end = time.monotonic() + max(0.0, float(timeout))
        while True:
            new = messaging.take(pr, reader)
            if new:
                return [_entry(e) for e in new]
            if not [n for n in pr.running() if n != reader] or reader in pr.stopping:
                return []
            left = min(pr.left(), end - time.monotonic())
            if left <= 0:
                return []
            await pr.record.wait_for(reader, timeout=left)

    def roster(self) -> list:
        from unify.memory import messaging

        return messaging.roster(self.pr)

    async def start(self, spawner: str, task: str, **kw: Any) -> dict:
        """Start a helper through the pass's pool (the pass's own lane, booking and wall clock apply)."""
        pr = self.pr
        if pr.pool is None:
            raise RecordRefused("no_lane", "this pass has no helper lane")
        name = kw.get("name")
        if name is None:
            k = 1
            while f"h{k}" in pr.record.participants:
                k += 1
            name = f"h{k}"
        pr.add_helper(name, task)
        try:
            started = await pr.pool.spawn(spawner, task, name)
        except Exception as exc:  # the pool's own refusals, with the reason
            raise RecordRefused(_spawn_form(exc), str(exc)) from None
        return {"name": started}

    async def stop(self, by: str, name: str) -> dict:
        from unify.memory import messaging

        info: dict = {}
        reply = await messaging.stop(self.pr, by, {"helper": name}, info)
        form = getattr(reply, "form", None)
        if form is not None:
            raise RecordRefused(form, reply)
        return {"name": name, "text": str(reply), **info}


def _post_form(exc: Exception) -> str:
    text = str(exc)
    if text.startswith("posting too fast"):
        return "too_fast"
    if text.startswith("the record is full"):
        return "record_full"
    if "bytes" in text and "limit" in text:
        return "too_large"
    return "not_posted"


def _spawn_form(exc: Exception) -> str:
    from unify.agents.record import PostRefused

    text = str(exc)
    if isinstance(exc, PostRefused):
        return "record_full" if text.startswith("the record is full") else "too_large"
    if isinstance(exc, PermissionError):
        return "not_main_agent" if text.startswith("only the main agent") else "spawn_off"
    if isinstance(exc, ValueError):
        return "short_request" if text.startswith("the request must say") else "bad_name"
    if isinstance(exc, RuntimeError):
        return "helper_limit"
    return "bad_request"


class AgentRecordAdapter:
    """The run's agent record (:class:`unify.agents.record.Record`) and the pool that starts teammates."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        self.record = pool.record

    def post(self, sender: str, text: str, to: Optional[list] = None) -> dict:
        from unify.agents.record import PostRefused

        if to:
            text = " ".join(f"@{n}" for n in to) + " " + str(text)
        rec = self.record
        try:
            rec.check_rate(sender)
            entry, unresolved = rec.append(sender, str(text))
        except PostRefused as exc:
            raise RecordRefused(_post_form(exc), str(exc)) from None
        return {"seq": entry.seq, "to": list(entry.mentions), "unresolved": list(unresolved)}

    def unseen(self, reader: str) -> list:
        """The entries for *reader* it has not been given, marked given (as ``wait`` marks them, so the turn
        boundary does not show them again)."""
        rec = self.record
        with rec._lock:
            found = rec._unseen(reader, True)
            rec.participants[reader].returned.update(e.seq for e in found)
        return [_entry(e) for e in found]

    async def wait(self, reader: str, timeout: float) -> list:
        from unify.agents.record import MAX_WAIT_S

        timeout = min(float(timeout), MAX_WAIT_S)
        if timeout <= 0:
            return self.unseen(reader)
        return [_entry(e) for e in await self.record.wait_for(reader, timeout=timeout)]

    def roster(self) -> list:
        rec = self.record
        return [
            {
                "name": p.name,
                "role": "actor" if p.name == rec.root else "teammate",
                "state": p.state,
                "help": "" if p.spawner is None else f"started by {p.spawner}",
            }
            for p in rec.participants.values()
            if p.role == "agent"
        ]

    async def start(self, spawner: str, task: str, **kw: Any) -> dict:
        try:
            name = await self.pool.spawn(spawner, task, kw.get("name"))
        except Exception as exc:  # the pool's own refusals, with the reason
            raise RecordRefused(_spawn_form(exc), str(exc)) from None
        return {"name": name}

    async def stop(self, by: str, name: str) -> dict:
        from unify.common.tool_registry import refusal

        participant = self.record.participants.get(name)
        if participant is None or participant.spawner is None:
            helpers = [p.name for p in self.record.participants.values() if p.spawner is not None]
            raise RecordRefused(
                "unknown_helper",
                refusal("stop", "unknown_helper", name=str(name)[:80], valid=", ".join(helpers) or "(none)"),
            )
        if participant.state != "running":
            # as the pool does: stopping an agent that already ended changes nothing
            return {"name": name, "state": participant.state}
        try:
            self.pool.stop(by, name, f"{by} stopped it")
        except PermissionError as exc:
            raise RecordRefused("not_allowed", str(exc)) from None
        return {"name": name, "state": participant.state}


__all__ = ["AgentRecordAdapter", "PassRecordAdapter"]
