"""
tests/conversation_manager/test_take_action.py
===================================================

Tests that verify ConversationManager correctly delegates to ``act`` for
requests that require the general-purpose Actor (knowledge, web search,
guidance, files, combined/research).
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    assert_act_triggered,
    assert_efficient,
    filter_events_by_type,
)
from unify.legacy.conversation_manager.events import (
    ActorHandleStarted,
    UnifyMessageReceived,
)

pytestmark = pytest.mark.eval

# ---------------------------------------------------------------------------
#  Knowledge-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_query_triggers_act(initialized_cm):
    """
    The user asks about company policy -> should call act to search knowledge.

    Natural scenario: the user needs to know a policy detail.
    """
    cm = initialized_cm

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

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_knowledge_about_product_triggers_act(initialized_cm):
    """
    The user asks about product information -> should call act.

    Natural scenario: the user needs warranty/product details for a customer.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="A customer is asking about Tesla warranty. What do we have on file?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Product knowledge query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_store_knowledge_triggers_act(initialized_cm):
    """
    The user asks to remember some information -> should call act.

    Natural scenario: the user wants to store a piece of information.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Make a note that our refund window is 30 days for unopened items.",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Storing knowledge should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Web search requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_weather_query_triggers_act(initialized_cm):
    """
    The user asks about current weather -> should call act for web search.

    Natural scenario: the user planning travel or outdoor activity.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What's the weather like in Berlin today?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Weather query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_news_query_triggers_act(initialized_cm):
    """
    The user asks about current news -> should call act for web search.

    Natural scenario: the user wants to stay informed.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What's happening in the news today? Any major headlines?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "News query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_current_events_query_triggers_act(initialized_cm):
    """
    The user asks about a recent event -> should call act for web search.

    Natural scenario: the user following industry developments.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Any notable AI announcements this week I should know about?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Current events query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Guidance-related requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_guidance_query_triggers_act(initialized_cm):
    """
    The user asks for guidance on a process -> should call act.

    Natural scenario: the user needs to follow a procedure.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="We might have a security incident. What's the protocol?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Guidance query should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  Combined/complex requests -> should trigger act
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_find_and_action_triggers_act(initialized_cm):
    """
    The user asks to find something and do something with it -> should call act.

    Natural scenario: the user wants information found and acted upon.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Find Bob's latest invoice and let me know if it's been paid.",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Find-and-action request should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


@pytest.mark.asyncio
@_handle_project
async def test_research_request_triggers_act(initialized_cm):
    """
    The user asks for research on a topic -> should call act.

    Natural scenario: the user needs background information compiled.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="I'm meeting with Contoso tomorrow. Can you pull together some background on them?",
        ),
    )

    assert_act_triggered(
        result,
        ActorHandleStarted,
        "Research request should trigger act",
        cm=cm,
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)


# ---------------------------------------------------------------------------
#  File/attachment-related requests -> should trigger act with filepath
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_summarize_attachment_triggers_act_with_filepath(
    initialized_cm,
):
    """
    Unify message with attachment + request to summarize -> act should include filepath.

    Natural scenario: the user sends a document through the chat and asks the
    assistant to summarize it. The assistant should call `act` with the
    attachment's filepath so the Actor can access and process the file.

    The rendered message shows: "[Attachments: Attachments/att-1_quarterly_report.pdf]"
    so the LLM should know the file location and include it in the act query.
    """
    cm = initialized_cm

    result = await cm.step_until_wait(
        UnifyMessageReceived(
            content="Please summarize this PDF for me.",
            attachments=["Attachments/att-1_quarterly_report.pdf"],
        ),
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

    # The query sent to act should include the attachment path
    act_query = actor_events[0].query.lower()
    assert "attachments" in act_query and "quarterly_report.pdf" in act_query, (
        f"Expected act query to include filepath 'Attachments/att-1_quarterly_report.pdf', "
        f"got query: {actor_events[0].query}"
    )

    # Efficiency assertions at end
    assert_efficient(result, 3)
