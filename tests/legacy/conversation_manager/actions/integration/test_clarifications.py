"""
Clarification ConversationManager → CodeActActor integration tests.

These validate that when an actor run needs clarification, CM can:
- receive the clarification request
- present it to the brain / route the brain's answer back into the handle
- continue execution deterministically

A question enters the actor's loop handle the way a running call's
``request_clarification`` puts it there (its ``_clar_q``). The CM's
clarification watcher is the handle's consumer: it takes the question from
``handle.next_clarification()`` and publishes it to the CM. The watchers
publish on the broker ``managers_utils`` bound when it was imported, which is
the manager's broker in production; the fixture's broker reset closed that
one, so these tests point it at the manager's broker again
(``watchers_reach_cm``).

The slow brain and the actor's model are scripted (``scripted_brain``). The
action stays in flight while its model call waits on a gate the test
releases (``held``).
"""

import asyncio

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    extract_actor_handle,
    get_action_data,
    get_actor_started_event,
    held,
    run_cm_until_wait,
    scripted_brain_and_actor,
    wait_for_condition,
    wait_for_actor_completion,
    wait_until_done,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    steer,
    turn,
    wait,
)
from tests.scripted_model import reply
from unify.legacy.conversation_manager.events import UnifyMessageReceived

pytestmark = [pytest.mark.integration]

QUESTION = "Which Aurora did you mean: the mobile app or the data platform?"
REQUEST = (
    "Write a short status summary for the Aurora project into a text "
    "file in my workspace, based on whatever notes you can find there."
)
QUERY = (
    "Write a short status summary of the Aurora project into a text file in "
    "the user's workspace, from the notes there."
)


@pytest.fixture
def watchers_reach_cm(initialized_cm_codeact, monkeypatch):
    """The action watchers publish to the manager's broker, as in production."""
    from unify.legacy.conversation_manager.domains import managers_utils

    monkeypatch.setattr(
        managers_utils,
        "event_broker",
        initialized_cm_codeact.cm.event_broker,
    )


def _ask(handle, call_id: str) -> None:
    """Put the question in the loop handle's clarification queue, as a
    running call's request_clarification does."""
    handle._inner._clar_q.put_nowait(
        {
            "type": "clarification",
            "call_id": call_id,
            "tool_name": "request_clarification",
            "question": QUESTION,
        },
    )


def _clarifications(cm, handle_id: int) -> list[dict]:
    return [
        a
        for a in get_action_data(cm, handle_id).get("handle_actions", [])
        if a.get("action_name") == "clarification_request"
    ]


async def _start(cm, model):
    model.next(turn(act(QUERY), say("Working on the Aurora summary."), wait()))
    result = await cm.step_until_wait(UnifyMessageReceived(content=REQUEST))
    handle_id = get_actor_started_event(result).handle_id
    await wait_for_condition(
        lambda: len(model.of("actor")) >= 1,
        timeout=60,
        timeout_message="The actor never made its first model call.",
    )
    return result, handle_id, extract_actor_handle(cm, handle_id)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_clarification_handle_contract(
    initialized_cm_codeact,
    watchers_reach_cm,
    scripted_brain,
):
    """
    Smoke: CM → CodeActActor clarification contract works end-to-end.

    We do NOT rely on any manager prompt behavior. Instead, we verify the public
    actor handle clarification surface that CM relies on:
    - `handle.next_clarification()` yields the question (taken by the CM's
      clarification watcher, which records it on the action)
    - `handle.answer_clarification(call_id, answer)` delivers the answer to
      the channel of the call that asked

    The action's handle wraps the loop handle for the post-completion storage
    review and delegates ``next_clarification`` and ``answer_clarification``
    to it.
    """
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    model = scripted_brain_and_actor(
        scripted_brain,
        held(gate, reply("Wrote Aurora_status.txt.")),
    )

    try:
        result, handle_id, handle = await _start(cm, model)

        # The question surfaces through next_clarification(), with its call id.
        call_id = "test-clar-0"
        _ask(handle, call_id)
        await wait_for_condition(
            lambda: _clarifications(cm, handle_id),
            timeout=60,
            timeout_message="The clarification never reached the CM.",
        )
        (clar,) = _clarifications(cm, handle_id)
        assert "which aurora" in clar["query"].lower(), clar
        assert clar["call_id"] == call_id

        # Answer via the public API: it reaches the asking call's channel.
        answers: asyncio.Queue[str] = asyncio.Queue()
        handle._inner._clarification_channels[call_id] = (asyncio.Queue(), answers)
        await handle.answer_clarification(call_id, "The mobile app")
        assert await asyncio.wait_for(answers.get(), timeout=10) == "The mobile app"

        # User asks to stop the action so the flow completes deterministically.
        model.next(
            turn(
                steer("stop", handle_id, reason="The user cancelled the summary."),
                say("Cancelled the summary."),
                wait(),
            ),
        )
        await cm.step_until_wait(
            UnifyMessageReceived(
                content="That's enough, cancel that summary.",
            ),
        )
        final = await wait_for_actor_completion(cm, handle_id, timeout=120)
        assert final.startswith("Task stopped as requested.")
        await wait_until_done(handle)

        assert_no_errors(result)
    finally:
        gate.set()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_clarification_cm_event_broker_path(
    initialized_cm_codeact,
    watchers_reach_cm,
    scripted_brain,
):
    """
    CM event-based clarification wiring works end-to-end (golden path).

    Validates the full production path:
      handle.next_clarification() → actor_watch_clarifications publishes event →
      EventHandler appends pending clarification → the brain is shown it with
      the answer_clarification_action to use → the brain's answer is recorded
      and routed to handle.answer_clarification(...)
    """
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    answer = "Wrote Aurora_status.txt with the mobile app's status."
    model = scripted_brain_and_actor(scripted_brain, held(gate, reply(answer)))

    result, handle_id, handle = await _start(cm, model)

    # A question enters the loop handle; the watcher publishes it to the CM.
    call_id = "0"
    _ask(handle, call_id)

    # Wait until CM has recorded a pending clarification request.
    await wait_for_condition(
        lambda: _clarifications(cm, handle_id),
        timeout=60,
        timeout_message="Timed out waiting for CM to record clarification_request in handle_actions.",
    )

    routed: list[tuple[str, str]] = []
    original_answer = handle.answer_clarification

    async def capturing_answer(call_id: str, answer: str) -> None:
        routed.append((call_id, answer))
        return await original_answer(call_id, answer)

    handle.answer_clarification = capturing_answer

    # The user's reply makes the brain answer via answer_clarification_action.
    model.next(
        turn(
            steer(
                "answer_clarification",
                handle_id,
                answer="The mobile app",
                call_id=call_id,
            ),
            say("Thanks, going with the mobile app."),
            wait(),
        ),
    )
    _ = await cm.step_until_wait(
        UnifyMessageReceived(
            content="I meant the mobile app.",
        ),
    )

    # The brain was shown the pending question and how to answer it.
    shown = model.last_state()
    assert QUESTION in shown
    assert (
        f"answer_clarification_action(handle_id={handle_id}, call_id='{call_id}'"
        in shown
    )

    # The answer was recorded on the action and routed to the handle.
    answered = [
        a
        for a in get_action_data(cm, handle_id).get("handle_actions", [])
        if str(a.get("action_name", "")).startswith("answer_clarification_")
    ]
    assert [a["query"] for a in answered] == ["The mobile app"]
    assert routed == [(call_id, "The mobile app")]

    # Continue until wait; the action completes without errors.
    model.next(turn(wait()))
    _ = await run_cm_until_wait(cm, max_steps=5)
    gate.set()
    final = await wait_for_actor_completion(cm, handle_id, timeout=120)
    assert final == answer
    await wait_until_done(handle)

    assert_no_errors(result)
