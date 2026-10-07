"""
tests/conversation_manager/actions/test_act_failure_context.py
================================================================

When an act() call fails (e.g. the LLM API rejects a request because an
image in the tool result exceeds the provider's per-image size limit), the
ConversationManager should include context about the failure in the *next*
act() request so the actor can avoid the same approach.

Production failure (2026-03-11): display(img) produced a 13 MB base64 PNG
that exceeded Anthropic's 5 MB per-image limit.  The actor's tool loop
crashed on the LLM inference call (not a tool call), and the exception
propagated through AsyncToolLoopHandle.result().

_StorageCheckHandle._run_lifecycle() captures the exception and surfaces
it as the result string.  actor_watch_result detects "Error" in the result
and publishes ActorResult(success=False).  The CM should then include this
failure context in the next act() request.
"""

import pytest

from tests.helpers import _handle_project
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

pytestmark = pytest.mark.eval


@pytest.mark.asyncio
@_handle_project
async def test_second_act_includes_failure_context(initialized_cm):
    """When act() fails with an error, the CM should include failure context
    in the second act() request so the actor can avoid repeating the same
    approach.

    Sequence:
    1. User asks to extract properties from an image and create a spreadsheet.
    2. CM dispatches act().
    3. act() fails — _StorageCheckHandle surfaces the error as the result
       and actor_watch_result publishes ActorResult(success=False).
    4. CM dispatches a second act() — this request should reference the
       previous failure so the actor can adjust its strategy.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content=(
                "Could you please take these properties in the attached image "
                "and put them all into an Excel spreadsheet for me, with each "
                "of their postcodes listed? Could you also please plot them all "
                "on a map?"
            ),
            attachments=["Attachments/att-img-1_IMG_1019.png"],
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

    followup_events = await run_cm_until_wait(cm, max_steps=10)

    second_actor_events = filter_events_by_type(
        followup_events,
        ActorHandleStarted,
    )
    assert second_actor_events, (
        "Expected the CM to retry with a second act() after the first "
        "failed with a BadRequestError. "
        f"Events emitted: {[type(e).__name__ for e in followup_events]}"
    )

    second_query = second_actor_events[0].query
    query_lower = second_query.lower()

    failure_indicators = [
        "5 mb",
        "too large",
        "size limit",
        "image exceeds",
        "failed",
        "error",
        "previous",
        "resize",
        "compress",
    ]
    matches = [kw for kw in failure_indicators if kw in query_lower]

    assert matches, (
        "The second act() request should reference the previous failure "
        "(e.g. image size limit, need to resize/compress) so the actor "
        "can avoid repeating the same approach.\n\n"
        f"Second act() query was:\n{second_query}\n\n"
        "Expected at least one of: "
        f"{failure_indicators}"
    )
