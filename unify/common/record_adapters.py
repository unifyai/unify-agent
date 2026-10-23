"""The record behind the actor's record tools, seen through :class:`unify.common.tool_registry.Record`.

:class:`AgentRecordAdapter`: the run's agent record (:mod:`unify.agents.record` and its :class:`Pool`), shared by
the actor and its teammates.

It does not change what the record does: each method calls the record's own functions and turns a refusal into
:class:`unify.common.tool_registry.RecordRefused` with the record tool's form code, so the tools' handlers answer
both records the same way: its ``text`` is either the finished refusal (a ``Refusal``, which keeps its form) or the
reason, which the handler fills into the tool's template for that form. Entries are returned as plain dicts (``seq``, ``ts``, ``author``, ``kind``, ``text``,
``mentions``).
"""

from __future__ import annotations

from typing import Any, Optional

from unify.common.tool_registry import RecordRefused


def _entry(e: Any) -> dict:
    return e.to_dict() if hasattr(e, "to_dict") else dict(e)


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
        return "spawn_off"
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


__all__ = ["AgentRecordAdapter"]
