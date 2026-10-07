"""Symbolic: ``UNIFY_REVIEW_LAST_REPLY`` lets the storage review read the agent's
last reply when a host ends a persistent session.

A persistent session ends by a stop, and its loop's result is then the stop
notice "processed stopped early, no result". As shipped the storage review
reads that notice as the session's "Final Result" unless ``UNIFY_OUTCOME``
is on. In the Overhauled Unify office cells every one of about 120 storage
reviews read it, and on the live office screen the review then called 6
correct answers failures: a host that ends its sessions between tasks
without posting an outcome, which is all real day-to-day work, tells the
review that every session failed.

With the switch on and no outcome channel, a session ended normally
(``SESSION_ENDED``: ``/quit``, ``{"quit": true}`` or end of input) is
reviewed with the agent's last reply as its final result, followed by one
plain status line. Every other stop keeps the notice, and with
``UNIFY_OUTCOME`` on the outcome channel decides, as before.

The decision is the pure function ``review_final_result``, which a replay of
recorded sessions can call as the live handle does. Requests are captured at
unillm's transport (``tests/cache_discipline_helpers.py``); nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests import cache_discipline_helpers as h
from unify import outcome as outcome_mod
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

NOTICE = "processed stopped early, no result"
EMPTY = "(the agent's last reply had no text)"
STATUS = "(The host then ended the session; no outcome was posted.)"
ENDED = "session ended"

SESSION_REPLY = "All done: the email was sent to Kim."
CLOSING = "Thanks, that is everything."
CLOSING_REPLY = "Glad to help."
REVIEW_SUMMARY = "Nothing worth storing."
SOLVED = {"solved": True, "source": "grader"}

# Each wait on a session is bounded, so a failing run ends.
WAIT_S = 2.0


@pytest.fixture
def switches(monkeypatch):
    def set_(*, last_reply: bool = False, outcome: bool = False) -> None:
        monkeypatch.setattr(
            SETTINGS,
            "UNIFY_REVIEW_LAST_REPLY",
            last_reply,
            raising=False,
        )
        monkeypatch.setattr(SETTINGS, "UNIFY_OUTCOME", outcome)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "")

    set_()
    return set_


# ── the setting ──────────────────────────────────────────────────────────


def test_the_switch_is_off_as_shipped_and_parsed_as_a_bool():
    from unify.settings import ProductionSettings as Settings

    assert Settings.model_fields["UNIFY_REVIEW_LAST_REPLY"].default is False
    for raw, value in (("on", True), ("true", True), ("", False), ("off", False)):
        assert Settings(UNIFY_REVIEW_LAST_REPLY=raw).UNIFY_REVIEW_LAST_REPLY is value


# ── the decision, as a pure function ─────────────────────────────────────


def _decide(original=NOTICE, **kwargs) -> str:
    args = {
        "last_reply": SESSION_REPLY,
        "stop_reason": ENDED,
        "outcome_active": False,
        "reply_at_outcome": None,
        "last_reply_switch": True,
    }
    args.update(kwargs)
    return caa.review_final_result(original, **args)


def test_helper_off_reads_the_session_result():
    assert _decide(last_reply_switch=False) == NOTICE
    assert _decide("Done.", last_reply_switch=False) == "Done."


def test_helper_on_a_normal_end_reads_the_last_reply_and_one_status_line():
    assert _decide() == f"{SESSION_REPLY}\n\n{STATUS}"
    assert caa.SESSION_ENDED == ENDED and caa._STOPPED_NOTICE == NOTICE


def test_helper_on_an_empty_last_reply_says_it_had_no_text():
    assert _decide(last_reply="") == f"{EMPTY}\n\n{STATUS}"


def test_helper_on_with_no_reply_yet_keeps_the_notice():
    assert _decide(last_reply=None) == NOTICE


@pytest.mark.parametrize(
    ("original", "stop_reason"),
    [
        # a stop for any other reason: a host's cancel, a closed session
        (NOTICE, "user cancelled"),
        (NOTICE, "session closed"),
        # the loop died on its own (a failed compression, say): no stop
        (NOTICE, None),
        # the step limit ended the loop with its own result
        ("🔚 Terminating early: max_steps (6) exceeded", None),
        ("🔚 Terminating early: max_steps (6) exceeded", ENDED),
        # a task that finished by itself has a real result
        ("Done.", ENDED),
    ],
)
def test_helper_on_any_other_end_keeps_the_session_result(original, stop_reason):
    assert _decide(original, stop_reason=stop_reason) == original


@pytest.mark.parametrize("switch", [False, True])
def test_helper_with_the_outcome_channel_decides_as_before(switch):
    # an outcome arrived: the reply it arrived after
    assert (
        _decide(outcome_active=True, reply_at_outcome="A.", last_reply_switch=switch)
        == "A."
    )
    assert (
        _decide(outcome_active=True, reply_at_outcome="", last_reply_switch=switch)
        == EMPTY
    )
    # no outcome: the last reply in place of the notice, with no status line
    assert _decide(outcome_active=True, last_reply_switch=switch) == SESSION_REPLY
    assert (
        _decide(outcome_active=True, last_reply="", last_reply_switch=switch) == EMPTY
    )
    assert (
        _decide(
            outcome_active=True,
            stop_reason="user cancelled",
            last_reply_switch=switch,
        )
        == SESSION_REPLY
    )
    assert _decide("Done.", outcome_active=True, last_reply_switch=switch) == "Done."


# ── scenarios: a persistent session and its review ──────────────────────


async def _next(handle, kinds) -> dict:
    async def find() -> dict:
        while True:
            note = await handle.next_notification()
            if isinstance(note, dict) and note.get("type") in kinds:
                return note

    return await asyncio.wait_for(find(), WAIT_S)


async def _persistent_review(
    *,
    closing_reply: str = CLOSING_REPLY,
    stop_reason: str = ENDED,
    outcome=None,
):
    """A persistent session: a turn that ran a tool, a closing turn, its end."""
    from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle
    from unify.common.async_tool_loop import start_async_tool_loop

    replies = (
        lambda: h.completion(calls=[("FunctionManager_list_functions", {})]),
        lambda: h.completion(content=SESSION_REPLY),
        lambda: h.completion(content=closing_reply),
        lambda: h.completion(content=REVIEW_SUMMARY),
    )
    actor = CodeActActor()
    try:
        with h.scripted(replies) as provider:
            inner = start_async_tool_loop(
                h.new_client("You are a scripted actor."),
                "Send the email to Kim.",
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                persist=True,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor, persist=True)
            await _next(handle, ("response",))
            if outcome is not None:
                outcome_mod.post(handle.outcome_session_id, outcome)
            await handle.interject(CLOSING)
            await _next(handle, ("response",))
            await handle.stop(stop_reason)
            note = await _next(
                handle,
                ("storage_review_complete", "storage_review_skipped"),
            )
            await asyncio.wait_for(handle._lifecycle_task, WAIT_S)
    finally:
        await actor.close()
    assert note["message"] == REVIEW_SUMMARY
    (review,) = [r for r in provider.requests if _is_review(r["messages"])]
    return _final_result(review["messages"]), handle


def _is_review(messages: list) -> bool:
    return "## Final Result" in json.dumps(messages, default=str)


def _final_result(messages: list) -> str:
    for message in messages:
        text = str(message.get("content") or "")
        if "## Final Result\n\n" in text:
            return text.split("## Final Result\n\n", 1)[1]
    raise AssertionError("no Final Result in the review")


@pytest.mark.asyncio
async def test_off_a_normal_end_reviews_the_stop_notice(switches):
    final, handle = await _persistent_review()
    assert handle._last_reply is None
    assert final == NOTICE


@pytest.mark.asyncio
async def test_on_a_normal_end_reviews_the_last_reply(switches):
    switches(last_reply=True)
    final, handle = await _persistent_review()
    assert handle.outcome_session_id is None
    assert final == f"{CLOSING_REPLY}\n\n{STATUS}"


@pytest.mark.asyncio
async def test_on_an_empty_last_reply_is_named(switches):
    switches(last_reply=True)
    final, _handle = await _persistent_review(closing_reply="")
    assert final == f"{EMPTY}\n\n{STATUS}"


@pytest.mark.asyncio
async def test_on_a_cancelled_session_keeps_the_notice(switches):
    switches(last_reply=True)
    final, _handle = await _persistent_review(stop_reason="user cancelled")
    assert final == NOTICE


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", [False, True])
async def test_with_the_outcome_channel_the_review_reads_as_before(switches, switch):
    switches(last_reply=switch, outcome=True)
    final, _handle = await _persistent_review(outcome=SOLVED)
    assert final == SESSION_REPLY
    final, _handle = await _persistent_review()
    assert final == CLOSING_REPLY


# ── a loop that ends on its own; turn reviews (mocked review loop) ──────


def _inner(notifications: asyncio.Queue, result: asyncio.Future) -> MagicMock:
    inner = MagicMock()

    async def _result():
        return await result

    async def _next_notification():
        return await notifications.get()

    inner.result = _result
    inner.next_notification = _next_notification
    inner.stop = AsyncMock()
    inner._client = MagicMock(
        messages=[
            {"role": "user", "content": "file the week 1 spend report"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "report filed"},
            {"role": "assistant", "content": "Filed week 1."},
        ],
    )
    inner._task = MagicMock()
    inner._task.get_ask_tools = MagicMock(return_value={})
    inner._task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._queue = asyncio.Queue()
    return inner


async def _until(predicate) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), WAIT_S)


def _review_handle(summary: str) -> MagicMock:
    review = MagicMock()

    async def _result():
        return summary

    review.result = _result
    return review


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", [None, ENDED])
async def test_on_turn_reviews_read_the_turn_and_the_end_reads_the_last_reply(
    switches,
    stop,
):
    """A turn review reads its turn's reply as shipped. At the end, a loop
    stopped by its host reads the last reply; one that died on its own with
    the same notice keeps it."""
    from unify.actor.code_act_actor import (
        _CURRENT_AGENT_CONTEXT,
        AgentContext,
        _StorageCheckHandle,
    )

    switches(last_reply=True)
    notifications: asyncio.Queue = asyncio.Queue()
    result = asyncio.get_event_loop().create_future()
    token = _CURRENT_AGENT_CONTEXT.set(AgentContext())
    try:
        with (
            patch.object(caa, "_start_storage_check_loop") as loop,
            patch.object(caa, "publish_manager_method_event", new_callable=AsyncMock),
        ):
            loop.return_value = _review_handle("Stored nothing.")
            handle = _StorageCheckHandle(
                inner=_inner(notifications, result),
                actor=SimpleNamespace(function_manager=None, guidance_manager=None),
                turn_reviews_enabled=True,
                persist=True,
            )
            await notifications.put({"type": "response", "content": "Filed week 1."})
            await _until(
                lambda: handle._turn_review_task is not None
                and handle._turn_review_task.done(),
            )
            assert loop.call_args.kwargs["live_session"] is True
            assert loop.call_args.kwargs["original_result"] == "Filed week 1."
            if stop is not None:
                await handle.stop(stop)
            result.set_result(NOTICE)
            await _until(handle.done)
    finally:
        _CURRENT_AGENT_CONTEXT.reset(token)
    assert loop.call_count == 2
    final = loop.call_args.kwargs["original_result"]
    assert final == (f"Filed week 1.\n\n{STATUS}" if stop else NOTICE)


# ── `unify act --persist --jsonl` ────────────────────────────────────────

FIRST, SECOND = "Total the March invoices.", "Now the April ones."
FIRST_REPLY, SECOND_REPLY = "March: 1,240.50 EUR.", "April: 980.00 EUR."
MAX_STEPS = 6


async def look() -> str:
    """Look at the invoices."""
    return "Nothing new."


class _Model:
    """Scripted replies: each request is answered after one look (or, to
    reach the step limit, never)."""

    def __init__(self, *, look_forever: bool = False) -> None:
        self.requests: list[list[dict]] = []
        self.look_forever = look_forever

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content=REVIEW_SUMMARY)
        asked = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and m.get("content") in (FIRST, SECOND)
        ]
        looked = any(m.get("role") == "tool" for m in messages[asked[-1] :])
        if looked and not self.look_forever:
            text = messages[asked[-1]]["content"]
            return h.completion(content=FIRST_REPLY if text == FIRST else SECOND_REPLY)
        return h.completion(calls=[("look", {})])


class _Actor:
    """The actor ``unify act`` starts, on a scripted persistent loop."""

    max_steps = 40

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client("You total invoices."),
            request,
            {"look": look},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=30,
            persist=persist,
            max_steps=self.max_steps,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch):
    """``unify act --persist --jsonl`` on the scripted actor; stdin is a pipe."""
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    async def start(self) -> None:
        self._actor = _Actor()

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=True,
            no_compose=False,
            no_store=False,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send
    os.close(write_fd)


def _responses(lines: list[dict]) -> list[str]:
    return [line["content"] for line in lines if line["type"] == "response"]


async def _two_requests(jsonl_session) -> list[list[dict]]:
    session, lines, send = jsonl_session
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(FIRST))
        await _until(lambda: len(_responses(lines)) == 1)
        send({"message": SECOND})
        await _until(lambda: len(_responses(lines)) == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, WAIT_S)
    assert code == 0
    assert _responses(lines) == [FIRST_REPLY, SECOND_REPLY]
    assert [line["type"] for line in lines if line["type"] != "storage"][-2:] == [
        "result",
        "ended",
    ]
    return [r for r in model.requests if _is_review(r)]


@pytest.mark.asyncio
async def test_cli_off_a_quit_session_reviews_the_notice(switches, jsonl_session):
    (review,) = await _two_requests(jsonl_session)
    assert _final_result(review) == NOTICE


@pytest.mark.asyncio
async def test_cli_on_a_quit_session_reviews_the_second_reply(
    switches,
    jsonl_session,
):
    switches(last_reply=True)
    (review,) = await _two_requests(jsonl_session)
    assert _final_result(review) == f"{SECOND_REPLY}\n\n{STATUS}"


@pytest.mark.asyncio
async def test_cli_on_a_session_ended_by_its_step_limit_keeps_its_result(
    switches,
    jsonl_session,
    monkeypatch,
):
    switches(last_reply=True)
    monkeypatch.setattr(_Actor, "max_steps", MAX_STEPS)
    session, lines, _send = jsonl_session
    model = _Model(look_forever=True)
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        code = await asyncio.wait_for(session.run(FIRST), WAIT_S)
    assert code == 0
    capped = f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded"
    assert [line["content"] for line in lines if line["type"] == "result"] == [capped]
    (review,) = [r for r in model.requests if _is_review(r)]
    assert _final_result(review) == capped
