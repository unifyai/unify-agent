"""
Steerability ConversationManager → CodeActActor integration tests.

These validate the core SteerableToolHandle contract as exercised through CM:
- pause/resume
- stop
- interject
- steering isolation when multiple handles are in-flight

All steering is driven by realistic user messages through step_until_wait,
so the CM brain sees conversational provenance for every state mutation.
"""

import pytest

from tests.helpers import _handle_project
from tests.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    extract_actor_handle,
    get_actor_started_event,
    inject_actor_result,
    wait_for_actor_completion,
    wait_for_condition,
)
from unify.conversation_manager.events import UnifyMessageReceived
from unify.conversation_manager.domains.brain_action_tools import (
    get_handle_paused_state,
)

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
# A cached actor reply finishes in a fraction of a second, before the live
# pause decision lands, so the handle is never seen paused.
@pytest.mark.fresh_llm_calls
@_handle_project
async def test_pause_resume_inflight_handle(initialized_cm_codeact):
    """Pause and resume an in-flight actor handle via natural user messages."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Search my notes files for anything about the quarterly revenue figures.",
        ),
    )

    handle_id = get_actor_started_event(result).handle_id
    handle = extract_actor_handle(cm, handle_id)

    # User asks to pause.
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on, pause that search for a moment. I need to think about what I actually need.",
        ),
    )

    await wait_for_condition(
        lambda: get_handle_paused_state(handle) is True,
        timeout=300,
        poll=0.05,
        timeout_message="Timed out waiting for handle to enter paused state.",
    )

    # User asks to resume.
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Okay, go ahead and continue that revenue search.",
        ),
    )

    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_stop_inflight_handle(initialized_cm_codeact):
    """Stop an in-flight actor handle via a natural user message."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Look through my recent notes files for mentions of the product launch timeline.",
        ),
    )
    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    # User asks to cancel.
    result_stop = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Never mind, cancel that search. I already found what I needed.",
        ),
    )

    actor_result = await wait_for_actor_completion(cm, handle_id, timeout=300)

    # Inject the ActorResult to trigger CM bookkeeping
    # (CMStepDriver patches the event broker and doesn't auto-forward background events).
    await inject_actor_result(
        cm,
        handle_id=handle_id,
        result=actor_result,
        success=True,
    )

    assert handle_id not in cm.cm.in_flight_actions
    assert handle_id in cm.cm.completed_actions
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_interject_midflight_constraints(initialized_cm_codeact):
    """Interject constraints mid-flight via a natural user message."""
    cm = initialized_cm_codeact

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Summarize my recent notes files about the marketing campaign.",
        ),
    )
    handle_id = get_actor_started_event(result).handle_id

    # User adds a constraint mid-flight.
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Actually, only include items that mention a specific dollar amount or budget figure.",
        ),
    )

    _final = await wait_for_actor_completion(cm, handle_id, timeout=300)
    assert_no_errors(result)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
# As above: from the cache, action A and its review finish before the pause.
@pytest.mark.fresh_llm_calls
@_handle_project
async def test_two_concurrent_handles_pause_one_other_completes(initialized_cm_codeact):
    """
    Concurrency: CM can track two in-flight actor handles and steer them independently.

    Scenario:
    - User requests a research task (A).
    - User asks to pause it while they think.
    - User sends a second, unrelated request (B).
    - B completes while A remains paused.
    """
    cm = initialized_cm_codeact

    # Action A: research task.
    result_a = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Search my notes files for anything about the quarterly budget review "
                "and summarize the key discussion points."
            ),
        ),
    )
    handle_id_a = get_actor_started_event(result_a).handle_id
    handle_a = extract_actor_handle(cm, handle_id_a)

    # User asks to pause A.
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hold on, pause that budget search for now. I want to do something else first.",
        ),
    )
    await wait_for_condition(
        lambda: get_handle_paused_state(handle_a) is True,
        timeout=300,
        poll=0.05,
        timeout_message="Timed out waiting for handle A to enter paused state.",
    )

    # Action B: unrelated request while A is paused.
    result_b = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Look up Alice's phone number for me.",
        ),
    )
    handle_id_b = get_actor_started_event(result_b).handle_id

    # CM should be tracking both handles (in-flight or completed).
    all_actions = {**cm.cm.in_flight_actions, **cm.cm.completed_actions}
    assert handle_id_a in all_actions
    assert handle_id_b in all_actions
    assert handle_id_a != handle_id_b

    # B should complete successfully even while A is paused.
    _final_b = await wait_for_actor_completion(cm, handle_id_b, timeout=300)
    assert_no_errors(result_b)

    # A should still be paused (steering isolation; no cross-talk).
    assert get_handle_paused_state(handle_a) is True

    # User resumes A.
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Okay, you can continue that budget search now.",
        ),
    )
    _ = await wait_for_actor_completion(cm, handle_id_a, timeout=300)
    assert_no_errors(result_a)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_ask_completed_action_about_trajectory(initialized_cm_codeact):
    """
    User asks about a completed action's trajectory/reasoning.

    This validates the completed_actions registry: handles persist after
    completion and remain available for ask queries via ask_* steering tools.

    Flow:
    1. User requests an action (triggers act)
    2. Action completes and moves to completed_actions
    3. User asks about how the action was performed
    4. Verify ask_* tool is available and can query the completed action
    """
    cm = initialized_cm_codeact

    # Step 1: Start an action that will complete
    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="List every file in my workspace with its size.",
        ),
    )

    actor_event = get_actor_started_event(result)
    handle_id = actor_event.handle_id

    # Wait for action to complete
    actor_result = await wait_for_actor_completion(cm, handle_id, timeout=300)

    # Inject the ActorResult event to trigger CM's event handler
    # (test driver patches event broker and doesn't forward background events)
    await inject_actor_result(
        cm,
        handle_id=handle_id,
        result=actor_result,
        success=True,
    )

    # Verify handle moved to completed_actions
    assert (
        handle_id not in cm.cm.in_flight_actions
    ), f"Handle {handle_id} should be removed from in_flight_actions after completion"
    assert (
        handle_id in cm.cm.completed_actions
    ), f"Handle {handle_id} should be in completed_actions after completion"

    # Verify the completed action data is preserved
    completed_data = cm.cm.completed_actions[handle_id]
    assert "handle" in completed_data, "Completed action should preserve handle"
    assert "query" in completed_data, "Completed action should preserve query"
    assert completed_data["query"], "Completed action query should not be empty"

    # Verify the handle's ask method is still functional (trajectory preserved)
    handle = completed_data["handle"]
    assert handle is not None, "Completed action handle should not be None"

    # Test that ask() works on the completed handle
    ask_handle = await handle.ask("What files did you find?")
    ask_result = await ask_handle.result()
    assert ask_result is not None, "Ask on completed handle should return a result"
    assert len(ask_result) > 0, "Ask result should not be empty"

    assert_no_errors(result)
