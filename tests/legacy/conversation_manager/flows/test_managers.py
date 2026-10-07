"""
tests/conversation_manager/test_managers.py
=================================================

Tests for task-related conversation flows through the ConversationManager.

Uses the ConversationManager "single-step" API:
- Construct an input Event (typically UnifyMessageReceived)
- Call `await cm.step(event)`
- Assert on state changes and output events

These tests verify that the LLM processes various request types correctly.
The focus is on LLM behavior and immediate responses, not internal task
machinery (which varies between real and simulated implementations).
"""

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import filter_events_by_type
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

pytestmark = pytest.mark.eval


def _get_chat_response(events):
    """Get the first outbound chat message, if any."""
    sent = filter_events_by_type(events, UnifyMessageSent)
    return sent[0] if sent else None


# =============================================================================
# LLM Processing Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_task_request_processed(initialized_cm):
    """
    Test that task creation requests are processed by the LLM.
    """
    cm = initialized_cm
    result = await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # LLM should have processed the request
    assert result.llm_ran, "Expected LLM to process task request"


@pytest.mark.asyncio
@_handle_project
async def test_stop_request_processed(initialized_cm):
    """
    Test that stop requests are processed by the LLM.
    """
    cm = initialized_cm
    # Start a task first
    await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # Then send stop request
    stop_result = await cm.step(
        UnifyMessageReceived(
            content="Stop that task, I don't need it anymore",
        ),
    )

    assert stop_result.llm_ran, "Expected LLM to process stop request"


@pytest.mark.asyncio
@_handle_project
async def test_status_query_processed(initialized_cm):
    """
    Test that status queries are processed by the LLM.
    """
    cm = initialized_cm
    # Start a task
    await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # Ask about status
    status_result = await cm.step(
        UnifyMessageReceived(
            content="What's the status of that task you're working on?",
        ),
    )

    assert status_result.llm_ran, "Expected LLM to process status query"


@pytest.mark.asyncio
@_handle_project
async def test_modification_request_processed(initialized_cm):
    """
    Test that task modification requests are processed.
    """
    cm = initialized_cm
    # Start a task
    await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # Modify the task
    modify_result = await cm.step(
        UnifyMessageReceived(
            content="Actually, for that task, please skip any file over 1 MB",
        ),
    )

    assert modify_result.llm_ran, "Expected LLM to process modification request"


@pytest.mark.asyncio
@_handle_project
async def test_pause_request_processed(initialized_cm):
    """
    Test that pause requests are processed by the LLM.
    """
    cm = initialized_cm
    # Start a task
    await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # Pause request
    pause_result = await cm.step(
        UnifyMessageReceived(
            content="Pause that task for now",
        ),
    )

    assert pause_result.llm_ran, "Expected LLM to process pause request"


@pytest.mark.asyncio
@_handle_project
async def test_resume_request_processed(initialized_cm):
    """
    Test that resume requests are processed by the LLM.
    """
    cm = initialized_cm
    # Start a task
    await cm.step(
        UnifyMessageReceived(
            content="Go through every CSV in my workspace and list their column names.",
        ),
    )

    # Resume request
    resume_result = await cm.step(
        UnifyMessageReceived(
            content="Resume that task please",
        ),
    )

    assert resume_result.llm_ran, "Expected LLM to process resume request"


# =============================================================================
# Clarification Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_llm_asks_clarification_for_ambiguous_request(initialized_cm):
    """
    Test that the LLM asks for clarification when given an ambiguous request.

    When a user's request is vague, the LLM should ask for clarification
    rather than making assumptions. This produces an immediate chat response.
    """
    cm = initialized_cm
    # Send an ambiguous request
    result = await cm.step(
        UnifyMessageReceived(
            content="I need help with a file",  # Ambiguous: which? what help?
        ),
    )

    # Should produce a chat response asking for clarification
    reply = _get_chat_response(result.output_events)
    assert reply is not None, "Expected a chat response"

    content_lower = reply.content.lower()

    # Verify clarification indicators
    clarification_indicators = ["which", "what", "how", "would you like", "prefer", "?"]
    has_clarification = any(
        indicator in content_lower for indicator in clarification_indicators
    )
    assert (
        has_clarification
    ), f"Expected LLM to ask a clarifying question, but got: {reply.content}"


@pytest.mark.asyncio
@_handle_project
async def test_llm_processes_clarification_response(initialized_cm):
    """
    Test that the LLM processes a clarification response.

    After asking for clarification, when the user provides more details,
    the LLM should process those details.
    """
    cm = initialized_cm
    # Send an ambiguous request
    await cm.step(
        UnifyMessageReceived(
            content="I need help with a file",
        ),
    )

    # Provide clarification
    result = await cm.step(
        UnifyMessageReceived(
            content="Only the CSV files, and list their row counts.",
        ),
    )

    # LLM should process the clarified request
    assert result.llm_ran, "Expected LLM to process clarification"


# =============================================================================
# Multi-turn Conversation Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_multi_turn_task_conversation(initialized_cm):
    """
    Test a multi-turn conversation involving task management.

    Verifies that the LLM processes each turn appropriately.
    """
    cm = initialized_cm
    # Turn 1: Request a task
    result1 = await cm.step(
        UnifyMessageReceived(
            content="List the files in my workspace",
        ),
    )
    assert result1.llm_ran, "Expected LLM to run for task request"

    # Turn 2: Ask about status
    result2 = await cm.step(
        UnifyMessageReceived(
            content="How's that task going?",
        ),
    )
    assert result2.llm_ran, "Expected LLM to run for status query"

    # Turn 3: Stop the task
    result3 = await cm.step(
        UnifyMessageReceived(
            content="Stop that task please",
        ),
    )
    assert result3.llm_ran, "Expected LLM to run for stop request"
