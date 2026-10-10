"""
ConversationManager → CodeActActor integration test for skill storage signals.

Verifies the slow brain correctly interprets "remember this" / "save this
workflow" as interjections (skill-storage requests relayed to the running
action) rather than stop signals that would kill the session.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    has_steering_tool_call,
)
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
)
from unify.legacy.conversation_manager.events import UnifyMessageReceived

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_memoize_signal_triggers_interject(initialized_cm_codeact):
    """User says "remember this for next time" during a guided session.

    Flow:
    1. User describes a workflow and gives the first instruction (triggers act)
    2. User gives another guided step (interject)
    3. User says "remember this for next time"
    4. Verify interject_* is called (not stop_*) — the session stays alive
       so the action can store the skill without losing progress
    """
    cm = initialized_cm_codeact

    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "I'm going to show you how to process a refund in our CRM. "
                "First, open the refunds spreadsheet in my workspace and find the order for John Smith."
            ),
        ),
    )
    actor_event = get_actor_started_event(result1)

    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Now add a refund row for John's order with today's date.",
        ),
    )

    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Perfect, remember this workflow for next time.",
        ),
    )

    assert has_steering_tool_call(cm, "interject_") or has_steering_tool_call(
        cm,
        "ask_",
    ), (
        "Skill-storage request should interject or ask the running action, not stop it. "
        f"Got: {cm.all_tool_calls}"
    )
    assert not has_steering_tool_call(cm, "stop_"), (
        f"CM should interject the skill-storage request, not stop the session. "
        f"Tool calls: {cm.all_tool_calls}"
    )
    assert_no_errors(result1)


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_save_workflow_signal_triggers_interject(initialized_cm_codeact):
    """User says "save this, I want you to do this on your own" — alternate phrasing.

    Flow:
    1. User gives a guided instruction (triggers act)
    2. User says they want the assistant to store the skill
    3. Verify interject_* is called (not stop_*) — the action handles
       skill storage while remaining alive
    """
    cm = initialized_cm_codeact

    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Let me walk you through how we onboard a new vendor. "
                "Start by creating a vendor folder for Acme Corp in my workspace."
            ),
        ),
    )
    actor_event = get_actor_started_event(result1)

    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Great, save this. I want you to be able to do this on your own next time."
            ),
        ),
    )

    assert has_steering_tool_call(cm, "interject_") or has_steering_tool_call(
        cm,
        "ask_",
    ), (
        "'Save this, do it on your own' should interject or ask to request skill storage. "
        f"Got: {cm.all_tool_calls}"
    )
    assert not has_steering_tool_call(cm, "stop_"), (
        f"CM should interject the skill-storage request, not stop the session. "
        f"Tool calls: {cm.all_tool_calls}"
    )
    assert_no_errors(result1)
