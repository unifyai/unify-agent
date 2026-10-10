"""
Notification plumbing ConversationManager → CodeActActor integration tests.

These validate that progress notifications emitted by in-flight actor handles
are received by the CM outer process and reflected in CM action state.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    inject_actor_notification,
    steer_action,
    wait_for_actor_completion,
    wait_for_condition,
)
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = [pytest.mark.integration, pytest.mark.eval]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_actor_progress_notification_e2e_wiring(initialized_cm_codeact):
    """
    CM receives actor progress notifications and can surface progress to the user.

    Flow:
    1. User starts a long-running action.
    2. Test applies an ActorNotification event to CM state deterministically.
    3. CM records progress in in_flight_actions[handle_id]["handle_actions"].
    4. User asks for an update; CM emits an outbound chat response.
    """
    cm = initialized_cm_codeact
    handle_id: int | None = None

    try:
        start = await cm.step_until_wait(
            UnifyMessageReceived(
                content=(
                    "Search my notes files for discussion about quarterly planning and "
                    "keep me updated as you make progress."
                ),
            ),
        )
        assert_no_errors(start)

        handle_id = get_actor_started_event(start).handle_id

        injected_message = "Searching the notes files and preparing interim findings."
        # Deterministically apply notification handling to CM state.
        await inject_actor_notification(
            cm,
            handle_id=handle_id,
            response=injected_message,
        )

        await wait_for_condition(
            lambda: any(
                a.get("action_name") == "progress"
                and injected_message in str(a.get("query", ""))
                for a in cm.cm.in_flight_actions.get(handle_id, {}).get(
                    "handle_actions",
                    [],
                )
            ),
            timeout=300,
            timeout_message=(
                "Timed out waiting for CM to record progress notification in "
                "handle_actions."
            ),
        )

        update = await cm.step_until_wait(
            UnifyMessageReceived(
                content="Any update on that notes search?",
            ),
        )
        assert_no_errors(update)

        message_events = [
            e for e in update.output_events if isinstance(e, UnifyMessageSent)
        ]
        assert (
            message_events
        ), "Expected a user-facing outbound message after progress notification."
        assert any((e.content or "").strip() for e in message_events)

    finally:
        if handle_id is not None and handle_id in cm.cm.in_flight_actions:
            try:
                await steer_action(
                    cm,
                    handle_id,
                    "stop",
                    reason="Test cleanup: stop notification wiring action.",
                )
            except Exception:
                pass
            try:
                await wait_for_actor_completion(cm, handle_id, timeout=300)
            except Exception:
                pass
