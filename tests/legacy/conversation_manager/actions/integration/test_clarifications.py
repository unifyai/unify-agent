"""
Clarification ConversationManager → CodeActActor integration tests.

These validate that when an actor run needs clarification, CM can:
- receive the clarification request
- present it / route it back into the handle
- continue execution deterministically
"""

import asyncio

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    extract_actor_handle,
    get_action_data,
    get_actor_started_event,
    inject_actor_clarification_request,
    run_cm_until_wait,
    wait_for_condition,
    wait_for_actor_completion,
)
from unify.legacy.conversation_manager.events import UnifyMessageReceived

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_clarification_handle_contract(initialized_cm_codeact):
    """
    Smoke: CM → CodeActActor clarification contract works end-to-end.

    We do NOT rely on any manager prompt behavior. Instead, we verify the public
    actor handle clarification surface that CM relies on:
    - `handle.next_clarification()` yields a question
    - `handle.answer_clarification(call_id, answer)` delivers the answer

    The action's handle wraps the loop handle for the post-completion storage
    review and delegates ``next_clarification`` to it. Clarification questions
    arrive in the loop handle's internal _clar_q (populated by the inner loop's
    _handle_clarification when nested tools request clarification).
    Answers are routed back through the inner loop's mirror mechanism.
    """
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Write a short status summary for the Aurora project into a text "
                "file in my workspace, based on whatever notes you can find there."
            ),
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = extract_actor_handle(cm, handle_id)

    # Inject a clarification question into the handle's internal clarification
    # queue — the same queue that the inner loop's _handle_clarification() populates
    # when a nested tool calls request_clarification().
    question = "Which Aurora did you mean: the mobile app or the data platform?"
    call_id = "test-clar-0"
    handle._inner._clar_q.put_nowait(
        {
            "type": "clarification",
            "call_id": call_id,
            "tool_name": "request_clarification",
            "question": question,
        },
    )

    # Verify next_clarification() surfaces the injected question.
    clar = await asyncio.wait_for(handle.next_clarification(), timeout=300)
    assert isinstance(clar, dict)
    q = str(clar.get("question") or "")
    assert (
        "which aurora" in q.lower()
    ), f"Expected clarification about Aurora, got: {q!r}"

    # Answer clarification via the public API.
    # The answer is routed through the inner loop's mirror mechanism.
    resp_call_id = str(clar.get("call_id") or "")
    await handle.answer_clarification(resp_call_id, "The mobile app")

    # User asks to stop the action so the flow completes deterministically.
    result_stop = await cm.step_until_wait(
        UnifyMessageReceived(
            content="That's enough, cancel that summary.",
        ),
    )
    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_clarification_cm_event_broker_path(initialized_cm_codeact):
    """
    CM event-based clarification wiring works end-to-end (golden path).

    Validates the full production path:
      handle.next_clarification() → actor_watch_clarifications publishes event →
      EventHandler appends pending clarification → CM brain selects answer_clarification_* →
      steering tool routes to handle.answer_clarification(...)
    """
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Write a short status summary for the Aurora project into a text "
                "file in my workspace, based on whatever notes you can find there."
            ),
        ),
    )
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id
    handle = extract_actor_handle(cm, handle_id)

    # Inject a clarification question into the handle's internal clarification queue
    # (same queue that the inner loop populates when nested tools request clarification).
    # Then deterministically apply the corresponding CM-level event, since the background
    # broker consumer is not running in step-driven tests.
    question = "Which Aurora did you mean: the mobile app or the data platform?"
    call_id = "0"
    handle._inner._clar_q.put_nowait(
        {
            "type": "clarification",
            "call_id": call_id,
            "tool_name": "request_clarification",
            "question": question,
        },
    )
    await inject_actor_clarification_request(
        cm,
        handle_id=handle_id,
        query=question,
        call_id=call_id,
    )

    # Wait until CM has recorded a pending clarification request.
    await wait_for_condition(
        lambda: any(
            a.get("action_name") == "clarification_request"
            for a in get_action_data(cm, handle_id).get("handle_actions", [])
        ),
        timeout=300,
        timeout_message="Timed out waiting for CM to record clarification_request in handle_actions.",
    )

    # Send a user message that should cause CM brain to answer via answer_clarification_* tool.
    _ = await cm.step_until_wait(
        UnifyMessageReceived(
            content="I meant the mobile app.",
        ),
    )

    # The act of answering should record an action in handle_actions.
    await wait_for_condition(
        lambda: any(
            str(a.get("action_name", "")).startswith("answer_clarification_")
            for a in get_action_data(cm, handle_id).get("handle_actions", [])
        ),
        timeout=300,
        timeout_message="Timed out waiting for CM brain to call answer_clarification_* steering tool.",
    )

    # Continue until wait; the action should complete or progress without errors.
    _ = await run_cm_until_wait(cm, max_steps=5)
    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)

    assert_no_errors(result)
