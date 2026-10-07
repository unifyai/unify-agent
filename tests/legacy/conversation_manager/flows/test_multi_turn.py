"""
tests/conversation_manager/test_multi_turn.py
==================================================

Tests for multi-turn conversation memory.

These tests verify that the ConversationManager maintains context across
multiple user messages - i.e., the assistant can recall information from
earlier turns in the conversation.

Unlike single-turn tests (one message in, one response out), these tests send multiple messages and verify the assistant reasons
over the full conversation history.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    assert_content_contains,
    get_exactly_one,
)
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
#  Multi-turn tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_two_turn_recall(initialized_cm):
    """
    Two-turn conversation: user mentions a word, then asks assistant to recall it.
    """
    cm = initialized_cm
    # Turn 1: User mentions a unique identifier
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="My order number is ABC-9876. Please confirm you have it.",
        ),
    )
    msg1 = get_exactly_one(result1.output_events, UnifyMessageSent)
    assert msg1.content

    # Turn 2: User asks assistant to recall it
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What was my order number?",
        ),
    )
    msg2 = get_exactly_one(result2.output_events, UnifyMessageSent)

    # Assistant should recall the order number
    assert_content_contains(
        msg2.content,
        "ABC-9876",
        "Assistant should recall order number from previous turn",
        cm=cm,
        result=result2,
    )


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_three_turn_recall(initialized_cm):
    """
    Three-turn conversation: verify context persists across multiple exchanges.
    """
    cm = initialized_cm
    # Turn 1: First piece of info
    result1 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="My favorite color is blue. Please acknowledge.",
        ),
    )
    get_exactly_one(result1.output_events, UnifyMessageSent)

    # Turn 2: Second piece of info
    result2 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="I live in Seattle. Please acknowledge.",
        ),
    )
    get_exactly_one(result2.output_events, UnifyMessageSent)

    # Turn 3: Ask about both
    result3 = await cm.step_until_wait(
        UnifyMessageReceived(
            content="What is my favorite color and where do I live?",
        ),
    )
    msg3 = get_exactly_one(result3.output_events, UnifyMessageSent)

    assert_content_contains(
        msg3.content,
        "blue",
        "Assistant should recall favorite color from earlier turns",
        cm=cm,
        result=result3,
    )
    assert_content_contains(
        msg3.content,
        "Seattle",
        "Assistant should recall location from earlier turns",
        cm=cm,
        result=result3,
    )
