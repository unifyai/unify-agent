"""
tests/conversation_manager/actions/test_persist_interactive_tutorial.py
========================================================================

Tests that the CM slow brain stops a persistent tutorial session when the
user clearly signals the walkthrough is over.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    assert_steering_called,
    filter_events_by_type,
)
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    ActorHandleStarted,
)

pytestmark = pytest.mark.eval


def _assert_context_setting_phase(result, description: str) -> None:
    """Context-setting may optionally open a persist session; act is allowed."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    if not actor_events:
        return
    for event in actor_events:
        assert event.action_name == "act", (
            f"Context-setting should only trigger act(), not other tools — "
            f"{description}. Got: {event.action_name}"
        )


def _assert_persist_true(cm, description: str) -> None:
    """Assert that the most recent act() call used persist=True."""
    actions = cm.cm.in_flight_actions
    assert actions, f"No in-flight actions found — {description}"
    last_action = list(actions.values())[-1]
    assert last_action.get("persist") is True, (
        f"Expected persist=True for interactive tutorial action, "
        f"but got persist={last_action.get('persist')}. {description}"
    )


def _get_handle_id(result) -> int:
    """Extract the handle_id from the ActorHandleStarted event."""
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert actor_events, "No ActorHandleStarted event found"
    return actor_events[0].handle_id


@pytest.mark.asyncio
@_handle_project
async def test_persistent_session_stopped_when_tutorial_ends(initialized_cm):
    """CM stops a persistent session when the tutorial is clearly over.

    Phase 1: Context-setting — no act.
    Phase 2: First instruction — act(persist=True).
    Phase 3: Simulate the actor completing its turn (awaiting_input).
    Phase 4: User signals the tutorial is done — the CM should call
    stop_* on the persistent action rather than leaving it alive.
    """
    from unify.common.prompt_helpers import now as prompt_now

    cm = initialized_cm

    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "I'm going to walk you through how to process returns in our "
                "system. Bear with me while I get set up."
            ),
        ),
    )
    _assert_context_setting_phase(result1, "Context-setting should not trigger act")

    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="First step: open the Returns tab at the top of the page.",
        ),
    )
    assert_act_triggered(
        result2,
        ActorHandleStarted,
        "First instruction should trigger act",
        cm=cm,
    )
    _assert_persist_true(cm, "Tutorial instruction should use persist=True")
    handle_id = _get_handle_id(result2)

    cm.cm.in_flight_actions[handle_id]["handle_actions"].append(
        {
            "action_name": "response",
            "query": "Done — the Returns tab is open. The returns dashboard is now showing.",
            "status": "awaiting_input",
            "timestamp": prompt_now(),
        },
    )

    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Great, that's everything. You've got the hang of it now.",
        ),
    )

    assert_steering_called(
        cm,
        "stop_",
        "CM should call stop_* when the tutorial is clearly over",
        result=result3,
    )
    assert_efficient(result3, 5)
