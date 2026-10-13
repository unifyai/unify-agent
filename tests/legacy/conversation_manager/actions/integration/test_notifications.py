"""
Notification plumbing ConversationManager → CodeActActor integration tests.

These validate that progress notifications emitted by in-flight actor handles
are received by the CM outer process, reflected in CM action state, and
shown to the brain when the user asks for an update.

The slow brain and the actor's model are scripted (``scripted_brain``). The
action stays in flight while its model call waits on a gate (``held``).
"""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.integration.helpers import (
    assert_no_errors,
    get_actor_started_event,
    held,
    inject_actor_notification,
    scripted_brain_and_actor,
    steer_action,
    wait_for_actor_completion,
    wait_for_condition,
    wait_until_done,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    turn,
    wait,
)
from tests.scripted_model import reply
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = [pytest.mark.integration]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_actor_progress_notification_e2e_wiring(
    initialized_cm_codeact,
    scripted_brain,
):
    """
    CM receives actor progress notifications and can surface progress to the user.

    Flow:
    1. User starts a long-running action.
    2. Test applies an ActorNotification event to CM state deterministically.
    3. CM records progress in in_flight_actions[handle_id]["handle_actions"].
    4. User asks for an update; the brain is shown the progress, and its
       update reaches the user as an outbound chat message.
    """
    cm = initialized_cm_codeact
    gate = asyncio.Event()
    model = scripted_brain_and_actor(
        scripted_brain,
        held(gate, reply("Quarterly planning notes: Q1 goals and hiring plan.")),
    )
    handle_id: int | None = None

    try:
        model.next(
            turn(
                act(
                    "Search the user's notes files for discussion about quarterly "
                    "planning; send progress updates along the way.",
                ),
                say("Searching your notes; I'll keep you posted."),
                wait(),
            ),
        )
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
            timeout=60,
            timeout_message=(
                "Timed out waiting for CM to record progress notification in "
                "handle_actions."
            ),
        )

        update_text = "Still searching: I'm going through the notes files and preparing interim findings."
        model.next(turn(say(update_text), wait()))
        update = await cm.step_until_wait(
            UnifyMessageReceived(
                content="Any update on that notes search?",
            ),
        )
        assert_no_errors(update)

        # The brain was shown the action's progress when the user asked.
        shown = model.last_state()
        assert "type='progress'" in shown
        assert f"<content>{injected_message}</content>" in shown

        message_events = [
            e for e in update.output_events if isinstance(e, UnifyMessageSent)
        ]
        assert [e.content for e in message_events] == [
            update_text,
        ], "Expected a user-facing outbound message after progress notification."

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
                await wait_for_actor_completion(cm, handle_id, timeout=120)
                await wait_until_done(
                    cm.cm.completed_actions[handle_id]["handle"],
                )
            except Exception:
                pass
        gate.set()
