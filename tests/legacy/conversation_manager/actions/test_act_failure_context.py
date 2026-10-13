"""
tests/conversation_manager/actions/test_act_failure_context.py
================================================================

When an act() call fails (e.g. the LLM API rejects a request because an
image in the tool result exceeds the provider's per-image size limit), the
ConversationManager shows the failure to the brain on its next turn, so a
retry can carry it to the actor and avoid the same approach.

Production failure (2026-03-11): display(img) produced a 13 MB base64 PNG
that exceeded Anthropic's 5 MB per-image limit.  The actor's tool loop
crashed on the LLM inference call (not a tool call), and the exception
propagated through AsyncToolLoopHandle.result().

_StorageCheckHandle._run_lifecycle() captures the exception and surfaces
it as the result string.  actor_watch_result detects "Error" in the result
and publishes ActorResult(success=False).

The brain's model is scripted (``scripted_brain``): how a live model words
the retry is a question about the model; this checks that the failure
reaches the brain and that the retry reaches the actor as a new action.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.actions.scripted_brain import (
    act,
    say,
    turn,
    wait,
)
from tests.legacy.conversation_manager.cm_helpers import (
    assert_act_triggered,
    filter_events_by_type,
)
from tests.legacy.conversation_manager.actions.integration.helpers import (
    get_actor_started_event,
    inject_actor_result,
    run_cm_until_wait,
)
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
)


@pytest.mark.asyncio
@_handle_project
async def test_second_act_includes_failure_context(initialized_cm, scripted_brain):
    """When act() fails with an error, the brain's next turn shows the
    failure, and the retry it starts reaches the actor as a new action.

    Sequence:
    1. User asks to extract properties from an image and create a spreadsheet.
    2. CM dispatches act().
    3. act() fails — _StorageCheckHandle surfaces the error as the result
       and actor_watch_result publishes ActorResult(success=False).
    4. The brain is shown the failed action with its error, and dispatches a
       second act() that names the failure.
    """
    cm = initialized_cm
    image = "Attachments/att-img-1_IMG_1019.png"
    first_query = (
        f"Read the properties in the image at {image}, put them into an Excel "
        "spreadsheet with each property's postcode, and plot them on a map."
    )
    retry_query = (
        f"Retry: read the properties in the image at {image}, put them into an "
        "Excel spreadsheet with postcodes, and plot them on a map. The previous "
        "attempt failed because the image exceeded the provider's 5 MB "
        "per-image limit; resize or compress the image before displaying it."
    )
    brain = scripted_brain()
    brain.next(turn(act(first_query), say("On it, reading the image now."), wait()))

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Could you please take these properties in the attached image "
                "and put them all into an Excel spreadsheet for me, with each "
                "of their postcodes listed? Could you also please plot them all "
                "on a map?"
            ),
            attachments=[image],
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Image property extraction should trigger act",
        cm=cm,
    )

    first_actor_event = get_actor_started_event(result)
    first_handle_id = first_actor_event.handle_id

    error_message = (
        "Error: inner task failed: Exception: LLM call failed: "
        "BadRequestError: AnthropicException - "
        '{"type":"error","error":{"type":"invalid_request_error",'
        '"message":"messages.10.content.0.tool_result.content.2'
        ".image.source.base64: image exceeds 5 MB maximum: "
        '13330552 bytes > 5242880 bytes"}}'
    )
    await inject_actor_result(
        cm,
        handle_id=first_handle_id,
        result=error_message,
        success=False,
    )
    # The failed action is no longer in flight; its history records the failure.
    assert first_handle_id not in cm.cm.in_flight_actions
    failed = cm.cm.completed_actions[first_handle_id]
    assert failed["handle_actions"][-1]["action_name"] == "act_failed"

    brain.next(
        turn(
            act(retry_query),
            say("The image was too large to process; retrying with a smaller copy."),
            wait(),
        ),
    )
    followup_events = await run_cm_until_wait(cm, max_steps=10)

    # The brain's next turn was shown the failed action with its error.
    state = brain.last_state()
    completed = state.partition("<completed_actions>")[2]
    assert f"<action id='{first_handle_id}'" in completed
    assert "status='failed'" in completed
    assert "image exceeds 5 MB maximum" in completed

    second_actor_events = filter_events_by_type(
        followup_events,
        ActorHandleStarted,
    )
    assert second_actor_events, (
        "Expected the CM to retry with a second act() after the first "
        "failed with a BadRequestError. "
        f"Events emitted: {[type(e).__name__ for e in followup_events]}"
    )

    # The retry is a new action, and its query (with the failure) reaches the actor.
    (second,) = second_actor_events
    assert second.handle_id != first_handle_id
    assert second.query == retry_query
    handle = cm.cm.in_flight_actions[second.handle_id]["handle"]
    assert handle._request == retry_query
