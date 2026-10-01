"""Symbolic: a message sent after a persistent session ended never reaches its review.

A persistent session's task loop can end on its own: in the ScienceWorld
training cell of 30 September (``UNIFY_REVIEW_FORK`` on) it ended four times
at the step limit (``max_steps (300) exceeded``, the limit counting every
message). Its host took the stop notice as the agent's turn and sent the next
observation, ending in the benchmark's reply protocol ("Reply with one
command ... as a JSON object"), before the storage review made its first
call. The handle forwarded it to the review, which read it as a user
interjection it had to answer: after its library writes, two of the six
reviews that got one answered with a game command
(``{"action":"command","command":"look at green light bulb"}``) instead of a
summary. None of the forked reviews that got no such message did so.

With ``UNIFY_REVIEW_FORK`` on, the message is refused and the review, forked
or standalone, sees only its own conversation; off, it is forwarded as
shipped. A live persistent session still takes its messages, and a handle
that was not persistent keeps the shipped routing, where an interjection
steers the review. Requests are captured at unillm's transport
(``tests/cache_discipline_helpers.py``).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

# A ScienceWorld-style text route: the actor's reply protocol, and the host's
# message after the stop notice (as recorded on 30 September, the step count
# being this test's).
PROTOCOL = (
    "You are playing ScienceWorld, a text game. Each turn, reply with one "
    "game command (or a short sequence) as a JSON object on the last line: "
    '{"action":"command","command":"<command>"}.'
)
TASK = (
    "Your task is to turn on the green light bulb by powering it using a "
    "circuit. First, focus on the green light bulb.\n" + PROTOCOL
)
FEEDBACK = (
    "Feedback since your last message:\n"
    "> 🔚 Terminating early: max_steps (5) exceeded\n"
    "No known action matches that input.\n"
    "(step 61/100)\n"
    "Same episode.\n"
    "Steps used: 61/100.\n"
    "Reply with one command (or a short sequence) as a JSON object on the "
    "last line."
)
GAME_COMMAND = '{"action":"command","command":"look at green light bulb"}'
SUMMARY = "Nothing worth storing."


FIRST_COMMAND = '{"action":"command","command":"focus on green light bulb"}'
SECOND_COMMAND = '{"action":"command","command":"look around"}'
OBSERVATION = (
    "Observation: You focus on the green light bulb.\n"
    "Reply with one command (or a short sequence) as a JSON object on the "
    "last line."
)


def _session_replies(tool: str | None, persist: bool) -> tuple:
    """The session's calls. With *tool*, one tool call, whose reply takes the
    session to the step limit; the limit cancels the call while it is still
    pending, which leaves it unanswered and so skips the fork, as in the
    recorded cell. Without, a game command per turn: two for a persistent
    session (its first request and the host's next observation), one for a
    session that was not."""
    if tool is not None:
        return (lambda: h.completion(calls=[(tool, {})]),)
    first = (lambda: h.completion(content=FIRST_COMMAND),)
    return first + ((lambda: h.completion(content=SECOND_COMMAND),) if persist else ())


REVIEW_REPLIES = (
    lambda: h.completion(content=SUMMARY),
    # Only a review that got the host's message makes a second call: it
    # answers the message, as the recorded review did.
    lambda: h.completion(content=GAME_COMMAND),
)


@pytest.fixture
def switches(monkeypatch):
    def set_(*, discipline: bool, fork: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)

    return set_


@pytest.fixture
def info_lines(monkeypatch):
    lines: list[str] = []
    original = caa.logger.info

    def capture(msg, *args, **kwargs):
        lines.append(str(msg))
        return original(msg, *args, **kwargs)

    monkeypatch.setattr(caa.logger, "info", capture)
    return lines


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    return "## Storage Review" in text or "You are a skill librarian" in text


def _mentions_feedback(request: dict) -> bool:
    return "Feedback since your last message" in json.dumps(
        request["messages"],
        default=str,
    )


async def _late_message(
    *,
    persist: bool,
    tool: str | None = None,
    max_steps: int | None = None,
):
    """A session whose task loop ends at its step limit, then the host's message.

    Without *tool*, a persistent session answers its first request with a
    game command, takes the host's next observation, answers it, and meets
    the limit (five messages); its history then ends with no unanswered
    call, so the review can fork. With *tool*, the limit is four messages. The review's first call is held until the host's
    late message has been sent, so the message arrives while that call is in
    flight, as it did in the recorded cell. Returns the session's result, the
    handle's notifications and every request, in order.
    """
    import unillm.clients.uni_llm as uni_llm

    from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle
    from unify.common.async_tool_loop import start_async_tool_loop

    actor = CodeActActor()
    notifications: list[dict] = []
    try:
        tools = h.session_tools(actor)
        replies = _session_replies(tool, persist) + REVIEW_REPLIES
        with h.scripted(replies) as provider:
            transport = uni_llm._acompletion_with_transient_retry
            in_flight, sent = asyncio.Event(), asyncio.Event()

            async def held(**kw):
                if _is_review(kw.get("messages") or []) and not sent.is_set():
                    in_flight.set()
                    await sent.wait()
                return await transport(**kw)

            uni_llm._acompletion_with_transient_retry = held
            inner = start_async_tool_loop(
                h.new_client(PROTOCOL),
                TASK,
                tools,
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=30,
                persist=persist,
                max_steps=max_steps,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor, persist=persist)
            if persist and tool is None:
                while True:
                    notification = await asyncio.wait_for(
                        handle.next_notification(),
                        60,
                    )
                    notifications.append(notification)
                    if notification.get("type") == "response":
                        break
                await handle.interject(OBSERVATION)
            result = await asyncio.wait_for(handle.result(), 20)
            await asyncio.wait_for(in_flight.wait(), 20)
            await handle.interject(FEEDBACK)
            sent.set()
            while True:
                notification = await asyncio.wait_for(handle.next_notification(), 20)
                notifications.append(notification)
                if notification.get("type") in (
                    "storage_review_complete",
                    "storage_review_skipped",
                ):
                    break
            await asyncio.wait_for(handle._lifecycle_task, 20)
    finally:
        await actor.close()
    return result, notifications, provider.requests


def _summary(notifications: list[dict]) -> str | None:
    done = [n for n in notifications if n.get("type") == "storage_review_complete"]
    return done[-1].get("message") if done else None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "discipline, tool, forked",
    [
        (True, None, True),
        # the recorded case: the fork is on but skipped (unanswered call)
        (True, "FunctionManager_list_functions", False),
        (False, None, False),
    ],
    ids=["forked", "fork-skipped", "no-cache-discipline"],
)
async def test_a_message_after_the_session_ended_is_refused_not_answered(
    switches,
    info_lines,
    discipline,
    tool,
    forked,
):
    switches(discipline=discipline, fork=True)
    result, notifications, requests = await _late_message(
        persist=True,
        tool=tool,
        max_steps=5 if tool is None else 4,
    )

    # The session's task loop ended on its own; its stop notice was the result.
    assert result.startswith("🔚 Terminating early: max_steps (")
    session = [r for r in requests if not _is_review(r["messages"])]
    reviews = [r for r in requests if _is_review(r["messages"])]
    assert len(session) == (2 if tool is None else 1)
    # The review made one call, never saw the host's message, and its reply
    # is the summary, not a game command.
    assert len(reviews) == 1
    assert not any(_mentions_feedback(r) for r in reviews)
    assert _summary(notifications) == SUMMARY
    refused = [n for n in notifications if n.get("type") == "interjection_refused"]
    assert [n["message"] for n in refused] == [caa._LATE_SESSION_MESSAGE_REFUSAL]
    assert any(
        line.startswith("Interjection not delivered: the persistent session")
        for line in info_lines
    )
    assert FEEDBACK not in "\n".join(info_lines)  # its length, not its text
    skipped = [line for line in info_lines if "fork skipped" in line]
    assert bool(skipped) is not forked
    # Forked, the review still continues the session's conversation, whose
    # system prompt holds the reply protocol; standalone, it has its own.
    review = reviews[0]["messages"]
    assert (review[0]["content"] == PROTOCOL) is forked
    if forked:
        # The observation the live session took is in the conversation.
        assert OBSERVATION in [m.get("content") for m in review]
    assert review[-1]["role"] == "user"
    assert review[-1]["content"].startswith(
        "## Storage Review" if forked else "Review the trajectory",
    )


@pytest.mark.asyncio
async def test_off_the_message_reaches_the_review_as_shipped(switches):
    """The recorded failure: the review answers the host's message."""
    switches(discipline=False, fork=False)
    _result, notifications, requests = await _late_message(persist=True, max_steps=5)

    reviews = [r for r in requests if _is_review(r["messages"])]
    assert reviews and _mentions_feedback(reviews[-1])
    assert not any(n.get("type") == "interjection_refused" for n in notifications)
    assert _summary(notifications) == GAME_COMMAND


@pytest.mark.asyncio
async def test_a_handle_that_was_not_persistent_still_steers_its_review(switches):
    switches(discipline=True, fork=True)
    result, notifications, requests = await _late_message(persist=False)
    assert result == FIRST_COMMAND

    reviews = [r for r in requests if _is_review(r["messages"])]
    assert reviews and _mentions_feedback(reviews[-1])
    assert not any(n.get("type") == "interjection_refused" for n in notifications)


@pytest.mark.asyncio
async def test_a_live_persistent_session_still_takes_its_messages(switches):
    """Before its task loop ends, the session gets every message."""
    from unify.actor.code_act_actor import CodeActActor, _StorageCheckHandle
    from unify.common.async_tool_loop import start_async_tool_loop

    switches(discipline=True, fork=True)
    actor = CodeActActor()
    replies = (
        lambda: h.completion(content=GAME_COMMAND),
        lambda: h.completion(content='{"action":"command","command":"look around"}'),
        # the review after the stop
        lambda: h.completion(content=SUMMARY),
    )
    try:
        with h.scripted(replies) as provider:
            inner = start_async_tool_loop(
                h.new_client(PROTOCOL),
                TASK,
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=30,
                persist=True,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor, persist=True)
            responses: list[dict] = []
            while len(responses) < 2:
                notification = await asyncio.wait_for(handle.next_notification(), 20)
                if notification.get("type") == "interjection_refused":
                    pytest.fail("a live session refused its message")
                if notification.get("type") == "response":
                    responses.append(notification)
                    if len(responses) == 1:
                        await handle.interject(FEEDBACK)
            assert not handle._task_done_event.is_set()
            await handle.stop("test over")
            await asyncio.wait_for(handle._lifecycle_task, 20)
    finally:
        await actor.close()
    assert [r["content"] for r in responses][-1].endswith('"look around"}')
    assert _mentions_feedback(provider.requests[1])
