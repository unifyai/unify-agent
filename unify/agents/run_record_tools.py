"""The comms tools on the run record: ``send``, ``messages``, ``wait`` and ``who``,
one definition for every agent: the actor, a teammate, the librarian, the writer, the curator and every helper call
them from their cells (``tools.send(...)``), and the fork through its relay (:mod:`unify.agents.fork_relay`).

Each host handler has the registry's signature ``handler(ctx, args) -> Reply``: ``ctx.record`` is the calling agent's
:class:`unify.agents.run_record.RunRecordView` (bound by the host, never from a frame). The text functions
(:func:`send_text`, :func:`messages_text`, :func:`wait_text`, :func:`who_text`) are what both the handlers and the
fork's relay answer with, so every agent reads the same replies.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from unify.agents.record import MAX_WAIT_S
from unify.agents.run_record import ALL_WAITING, RunRecordView, row_of

#: The comms tools every agent has on the run record.
NAMES = ("send", "messages", "wait", "who")
#: Every tool that only moves messages: a cell that called these and nothing else is not evidence work (the storm
#: notice's count, :meth:`unify.agents.run_record.RunRecord.note_act`).
MESSAGE_TOOLS = frozenset({"send", "messages", "wait", "who", "reply", "ask"})
#: The entries one ``messages(all=True)`` call shows.
ALL_PAGE = 50

_TAGS = {"reply": " (reply)", "cancel": " (stop)", "system": " (notice)"}


def _reply(text: str, status: str = "ok", form: Optional[str] = None) -> Any:
    from unify.memory.writer_tools import Reply

    return Reply(str(text), status, form, ())


def _refused(tool: str, form: str, **fields: Any) -> Any:
    from unify.common.tool_registry import form_of, refusal

    text = refusal(tool, form, **fields)
    return _reply(text, "refused", form_of(text))


def _running(view: RunRecordView) -> str:
    names = [
        r["name"]
        for r in view.record.roster_rows()
        if r["state"] in ("running", "waiting")
    ]
    return ", ".join(names) or "(none)"


def render(view: RunRecordView, rows: list[dict]) -> str:
    lines = [
        f"#{r['seq']} {r['author']}{_TAGS.get(r.get('kind'), '')}: {r['text']}"
        for r in rows
    ]
    lines.append(f"(running now: {_running(view)})")
    return "\n".join(lines)


# --- the texts ------------------------------------------------------------------------------------------------------


def send_text(
    view: RunRecordView,
    text: Any,
    *,
    default_to: Optional[str] = None,
    info: Optional[dict] = None,
) -> tuple[str, Optional[str]]:
    """Post *text* as the view's agent: ``(reply text, refusal form or None)``. A message that addresses no agent goes
    to *default_to* (a helper's starter) when given, and is refused otherwise."""
    from unify.common.tool_registry import RecordRefused, form_of, refusal

    if not isinstance(text, str) or not text.strip():
        return refusal("send", "no_text"), "no_text"
    reply_to = None
    try:
        out = view.post(view.name, text, default_to=default_to, reply_to=reply_to)
    except RecordRefused as exc:
        if form_of(exc.text) is not None:
            return str(exc.text), exc.form
        try:
            return refusal("send", exc.form, why=exc.text), exc.form
        except (KeyError, IndexError):
            return refusal("send", "not_posted", why=exc.text), "not_posted"
    if info is not None:
        info.update(to=list(out["to"]), seq=out["seq"])
    said = f"sent to {', '.join(out['to'])} (#{out['seq']})"
    if out.get("unresolved"):
        said += f"; these names match no agent: {', '.join('@' + n for n in out['unresolved'])}"
    if out.get("ended"):
        said += (
            f"; ended, so nobody reads it for them now: {', '.join('@' + n for n in out['ended'])}"
            + (
                " (the writer reads it when it next runs)"
                if "writer" in out["ended"]
                else ""
            )
        )
    return said, None


def messages_text(
    view: RunRecordView, *, all_entries: bool = False, info: Optional[dict] = None
) -> str:
    """The new messages for the view's agent, without waiting; with *all_entries*, the next page of every entry of the
    record."""
    if all_entries:
        rows, more = view.messages_all(ALL_PAGE)
        if info is not None:
            info.update(entries=len(rows), more=more, all=True)
        if not rows:
            return f"no entries you have not been shown (running now: {_running(view)})"
        text = render(view, rows)
        if more:
            text += f"\n({more} more entries: call tools.messages(all=True) again)"
        return text
    rows = view.unseen(view.name)
    if info is not None:
        info.update(entries=len(rows), **{"from": sorted({r["author"] for r in rows})})
    if rows:
        return render(view, rows)
    return f"no new messages (running now: {_running(view)})"


async def wait_text(
    view: RunRecordView,
    timeout: Optional[float] = None,
    *,
    deadline: Optional[float] = None,
    stop: Optional[Callable[[], Optional[str]]] = None,
    info: Optional[dict] = None,
) -> str:
    """Wait for the next messages for the view's agent: at most *timeout* seconds (default and limit
    :data:`MAX_WAIT_S`), never past *deadline* (a ``time.monotonic()`` time: the pass's or the request's), returning at
    once when no one who could answer is running or when *stop* names a reason the agent is asked to stop.
    """
    started = time.monotonic()
    limit = MAX_WAIT_S if timeout is None else min(float(timeout), MAX_WAIT_S)
    if deadline is not None and deadline - time.monotonic() <= 0:
        rows, why = view.unseen(view.name), "wall_clock"
        rows_text = render(view, rows) if rows else None
        if info is not None:
            info.update(outcome="message" if rows else why, seconds=0.0)
        return rows_text or "no new messages; the wall clock has ended"
    rows, why = await view.wait_outcome(limit, deadline=deadline, stop=stop)
    if info is not None:
        info.update(outcome=why, seconds=round(time.monotonic() - started, 3))
    if rows:
        return render(view, rows)
    if why == "nothing_can_arrive":
        return f"no new messages; {ALL_WAITING} (running now: {_running(view)})"
    if why == "asked_to_stop":
        reason = stop() if stop is not None else None
        return f"no new messages; you are asked to stop: {reason or 'the agent that started you stopped you'}"
    if why == "wall_clock":
        return "no new messages; the wall clock is reached"
    return f"no new messages in {round(time.monotonic() - started)} s (running now: {_running(view)})"


def who_text(view: RunRecordView, *, info: Optional[dict] = None) -> str:
    """Who is on the record now (agents that ended are counted, not listed), and whom this agent should mention."""
    doc = view.record.roster_doc()
    rows = [r for r in doc["agents"] if r["state"] != "ended"]
    ended = len(doc["agents"]) - len(rows)
    if info is not None:
        info.update(
            roster=[
                {k: r.get(k) for k in ("name", "role", "state", "episode")}
                for r in rows
            ]
        )
    lines = ["agents on the record now:"]
    for r in rows:
        where = f", episode {r['episode']}" if r.get("episode") else ""
        note = f"; {r['note']}" if r.get("note") else ""
        lines.append(
            f"- {r['name']} ({r.get('role')}, {r['state']}{where}): {r.get('helps_with', '')}{note}"
        )
    if ended:
        lines.append(f"({ended} agents of this run have ended)")
    me = next((r for r in rows if r["name"] == view.name), None)
    lines.append(f"You are {view.name} ({view.role}).")
    from unify.agents.run_record import INSTANCE_ROLES, role_seeded, run_has_memory
    from unify.common.tool_registry import MEMORY_ROLES

    home = getattr(view.record, "home", None)
    has = {r for r in INSTANCE_ROLES if home is not None and role_seeded(home, r)}
    # a run without memory (its setting and its record) has no writer, fork or librarian: none is offered
    memory = run_has_memory(home, view.role, getattr(view, "spawner", None))
    for m in (me or {}).get("should_mention") or ():
        if m.get("who") in INSTANCE_ROLES and m.get("who") not in has:
            continue  # a role this run does not have is never offered
        if m.get("who") in MEMORY_ROLES and not memory:
            continue
        lines.append(f"Mention @{m.get('who')} when {m.get('when')}.")
    roles = ("@writer, @actor, " + ("@librarian, " if "librarian" in has else "") + "@fork (of your own episode), @helpers"
             if memory else "@actor, @helpers")
    lines.append(f"Address an agent with @<name> while it runs, or a role: {roles}. Anyone may address anyone.")
    return "\n".join(lines)


# --- the host handlers -----------------------------------------------------------------------------------------------


def _view(ctx: Any) -> Optional[RunRecordView]:
    view = getattr(ctx, "record", None)
    return view if isinstance(view, RunRecordView) else None


def _extra(info: dict) -> None:
    from unify.common.tool_channel import event_extra

    try:
        event_extra().update({"messaging": dict(info)})
    except Exception:  # noqa: BLE001 - the event row is reporting only
        pass


def _no_record(tool: str) -> Any:
    from unify.common.tool_registry import REFUSED, Refusal

    return _reply(
        Refusal(f"{REFUSED}this agent is not on the run record", tool, "not_posted"),
        "refused",
        "not_posted",
    )


async def send(ctx: Any, args: dict) -> Any:
    view = _view(ctx)
    if view is None:
        return _no_record("send")
    info: dict = {}
    text, form = await asyncio.to_thread(
        send_text,
        view,
        args.get("text"),
        default_to=view.spawner if view.role == "helper" else None,
        info=info,
    )
    _extra(info)
    return _reply(text) if form is None else _reply(text, "refused", form)


async def messages(ctx: Any, args: dict) -> Any:
    view = _view(ctx)
    if view is None:
        return _no_record("messages")
    info: dict = {}
    text = await asyncio.to_thread(
        messages_text, view, all_entries=bool(args.get("all")), info=info
    )
    _extra(info)
    return _reply(text)


async def wait(ctx: Any, args: dict) -> Any:
    view = _view(ctx)
    if view is None:
        return _no_record("wait")
    timeout = args.get("timeout")
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not 0 < timeout <= MAX_WAIT_S
    ):
        return _refused("wait", "bad_timeout")
    info: dict = {}
    text = await wait_text(
        view,
        timeout,
        deadline=getattr(ctx, "deadline", None),
        stop=getattr(view, "stop_reason", None),
        info=info,
    )
    _extra(info)
    return _reply(text)


async def who(ctx: Any, args: dict) -> Any:
    view = _view(ctx)
    if view is None:
        return _no_record("who")
    info: dict = {}
    text = await asyncio.to_thread(who_text, view, info=info)
    _extra(info)
    return _reply(text)


__all__ = [
    "MESSAGE_TOOLS",
    "NAMES",
    "messages",
    "messages_text",
    "render",
    "row_of",
    "send",
    "send_text",
    "wait",
    "wait_text",
    "who",
    "who_text",
]
