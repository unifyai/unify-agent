"""
tests/conversation_manager/test_multi_turn.py
==================================================

Tests for multi-turn conversation memory.

These tests verify that the ConversationManager maintains context across
multiple user messages - i.e., the assistant can recall information from
earlier turns in the conversation.

Unlike single-turn tests (one message in, one response out), these tests send multiple messages and verify the assistant reasons
over the full conversation history.

The slow brain's model is scripted (``tests/scripted_model.py``). Its later
replies recall the earlier turns, and the tests check that the harness gave
it those turns to recall: each later request shows the whole conversation so
far, in order, and each reply reaches the chat once.
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import (
    assert_content_contains,
    get_exactly_one,
)
from tests.legacy.conversation_manager.scripted_brain import (
    WAIT,
    brain_requests,
    brain_turn,
    conversation,
    say,
)
from tests.scripted_model import ScriptedModel, scripted
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)


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
    stated = "My order number is ABC-9876. Please confirm you have it."
    confirmed = "Confirmed: I have your order number, ABC-9876."
    question = "What was my order number?"
    model = ScriptedModel(
        actor=[
            brain_turn(say(confirmed), WAIT),
            brain_turn(say("Your order number is ABC-9876."), WAIT),
        ],
    )
    with scripted(model):
        # Turn 1: User mentions a unique identifier
        result1 = await cm.step_until_wait(
            UnifyMessageReceived(
                content=stated,
            ),
        )
        msg1 = get_exactly_one(result1.output_events, UnifyMessageSent)
        assert msg1.content == confirmed

        # Turn 2: User asks assistant to recall it
        result2 = await cm.step_until_wait(
            UnifyMessageReceived(
                content=question,
            ),
        )
        msg2 = get_exactly_one(result2.output_events, UnifyMessageSent)

    # The second turn is shown the first exchange before the question.
    _, recall = brain_requests(model)
    assert conversation(recall) == [stated, confirmed, question]
    model.assert_used_up()

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
    color = "My favorite color is blue. Please acknowledge."
    place = "I live in Seattle. Please acknowledge."
    question = "What is my favorite color and where do I live?"
    model = ScriptedModel(
        actor=[
            brain_turn(say("Noted: your favorite color is blue."), WAIT),
            brain_turn(say("Noted: you live in Seattle."), WAIT),
            brain_turn(
                say("Your favorite color is blue, and you live in Seattle."),
                WAIT,
            ),
        ],
    )
    with scripted(model):
        # Turn 1: First piece of info
        result1 = await cm.step_until_wait(
            UnifyMessageReceived(
                content=color,
            ),
        )
        get_exactly_one(result1.output_events, UnifyMessageSent)

        # Turn 2: Second piece of info
        result2 = await cm.step_until_wait(
            UnifyMessageReceived(
                content=place,
            ),
        )
        get_exactly_one(result2.output_events, UnifyMessageSent)

        # Turn 3: Ask about both
        result3 = await cm.step_until_wait(
            UnifyMessageReceived(
                content=question,
            ),
        )
        msg3 = get_exactly_one(result3.output_events, UnifyMessageSent)

    # The third turn is shown both earlier exchanges before the question.
    _, _, recall = brain_requests(model)
    assert conversation(recall) == [
        color,
        "Noted: your favorite color is blue.",
        place,
        "Noted: you live in Seattle.",
        question,
    ]
    model.assert_used_up()

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
