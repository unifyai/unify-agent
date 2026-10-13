"""
tests/conversation_manager/test_take_action.py
===================================================

Tests that verify ConversationManager delegates to ``act``: when the slow
brain calls ``act``, the request becomes an in-flight action on the actor,
and a file the user attached reaches the actor by its workspace path.

The brain's model is scripted (``scripted_brain``): which requests a live
model sends to ``act`` is a question about the model, not the harness.
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
    assert_efficient,
    filter_events_by_type,
)
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
    UnifyMessageSent,
)


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_query_triggers_act(initialized_cm, scripted_brain):
    """
    The brain's ``act`` call starts an action that runs the brain's query.

    Natural scenario: the user needs a policy detail the assistant has to
    look up.
    """
    cm = initialized_cm
    query = "Find our office hours in the workspace notes and stored guidance."
    brain = scripted_brain()
    brain.next(turn(act(query), say("Let me check our office hours."), wait()))

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What are our office hours again?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Knowledge query should trigger act",
        cm=cm,
    )
    (started,) = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert (started.handle_id, started.action_name, started.query) == (
        0,
        "act",
        query,
    )

    # The brain was shown the user's message as new.
    state = brain.last_state()
    assert "**NEW** [" in state and "What are our office hours again?" in state

    # The action is in flight on the actor, running the brain's query.
    action = cm.cm.in_flight_actions[0]
    assert action["query"] == query
    assert action["handle"]._request == query
    assert not action["handle"].done()
    assert [a["action_name"] for a in action["handle_actions"]] == ["act_started"]

    # The acknowledgement reaches the chat.
    sent = filter_events_by_type(result.output_events, UnifyMessageSent)
    assert [e.content for e in sent] == ["Let me check our office hours."]
    assert cm.cm.chat_history.recent()[-1].content == "Let me check our office hours."

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  File/attachment-related requests -> should trigger act with filepath
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_summarize_attachment_triggers_act_with_filepath(
    initialized_cm,
    scripted_brain,
):
    """
    Unify message with attachment + request to summarize -> act includes filepath.

    Natural scenario: the user sends a document through the chat and asks the
    assistant to summarize it. The rendered message shows
    "[Attachments: Attachments/att-1_quarterly_report.pdf]", so the brain
    can quote the attachment's path in the act query and the Actor can open
    the file.
    """
    cm = initialized_cm
    path = "Attachments/att-1_quarterly_report.pdf"
    query = f"Summarize the PDF at {path} for the user."
    brain = scripted_brain()
    brain.next(turn(act(query), say("Reading the report now."), wait()))

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Please summarize this PDF for me.",
            attachments=[path],
        ),
    )

    # The brain was shown the attachment's workspace path on the message.
    assert (
        f"Please summarize this PDF for me. [Attachments: {path}]"
        in brain.last_state()
    )

    # First verify that act was triggered
    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Summarize attachment request should trigger act",
        cm=cm,
    )

    # Now verify the act query includes the filepath
    actor_events = filter_events_by_type(result.output_events, ActorHandleStarted)
    assert len(actor_events) >= 1, "Expected at least one ActorHandleStarted event"

    # The query sent to act includes the attachment path, and reaches the actor.
    act_query = actor_events[0].query.lower()
    assert "attachments" in act_query and "quarterly_report.pdf" in act_query, (
        f"Expected act query to include filepath 'Attachments/att-1_quarterly_report.pdf', "
        f"got query: {actor_events[0].query}"
    )
    handle = cm.cm.in_flight_actions[actor_events[0].handle_id]["handle"]
    assert path in handle._request

    # Efficiency assertions at end
    assert_efficient(result, 3)
