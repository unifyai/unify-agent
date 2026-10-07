"""
tests/conversation_manager/handle/test_handle.py
=================================================

Comprehensive tests for ConversationManagerHandle, specifically the ask() method.

The ask() method is the bridge between the Actor (and other managers) and the live
conversation with the user. It enables nested tool loops that can:
- Infer answers from the recent conversation (PATH 1)
- Ask clarifying questions and wait for user replies (PATH 2)
- Return structured responses (Pydantic models, Enums)

These tests validate the complete flow including:
- PATH 1: Inference from transcript without user interaction
- PATH 2: Interactive question/answer with user reply routing
- Structured output parsing and validation
- active_ask_handle registration and input routing
- Error handling and edge cases

Test Categories:
- Unit tests: Test isolated components with mocked dependencies
- Integration tests: Test the full flow with the CMStepDriver

Note: These are primarily eval tests since they involve LLM behavior.
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.helpers import _handle_project
from unify.legacy.conversation_manager.events import (
    DirectMessageEvent,
    Event,
    UnifyMessageReceived,
)
from unify.legacy.conversation_manager.handle import ConversationManagerHandle

pytestmark = pytest.mark.eval


# =============================================================================
# Helper Functions for Deterministic Waiting
# =============================================================================

# Ceiling for one ``ask()`` loop to reach its final answer. A cold cache means
# several real reasoning turns, plus any nested manager ask, each of which can
# take tens of seconds; a replayed cache finishes the same loop in milliseconds.
# The ceiling therefore only ever binds on cache-refresh runs, where it must be
# generous enough not to read a slow provider as a broken loop.
_ASK_RESULT_TIMEOUT = 300.0


async def _wait_for_condition(predicate, *, timeout: float = 30.0, poll: float = 0.05):
    """Poll predicate() until it returns True or timeout expires."""
    import time as _time

    start = _time.perf_counter()
    while _time.perf_counter() - start < timeout:
        if predicate():
            return True
        await asyncio.sleep(poll)
    return False


def _capture_questions(cm) -> list[str]:
    """Record every question ``ask_question`` posts to the chat, in order.

    The direct-speech publish is what marks PATH 2 as genuinely under way:
    it happens inside ``ask_question``, immediately before it blocks on a
    reply. Waiting on the returned list is therefore the trigger that says
    "the assistant has asked, an interjection will now be consumed as its
    answer" — and it holds whether the turn took a cached millisecond or a
    live minute.
    """
    questions: list[str] = []
    original_publish = cm.cm.event_broker.publish

    async def _tracking_publish(channel, message):
        if channel == "app:comms:direct_speech":
            evt = Event.from_json(message)
            if isinstance(evt, DirectMessageEvent):
                questions.append(evt.content)
        return await original_publish(channel, message)

    cm.cm.event_broker.publish = _tracking_publish
    return questions


# =============================================================================
# Test Response Formats (Pydantic Models and Enums)
# =============================================================================


class UserPreference(BaseModel):
    """Structured response for user preference questions."""

    preference: str = Field(description="The user's stated preference")
    confidence: float = Field(
        description="Confidence level from 0 to 1",
        ge=0.0,
        le=1.0,
    )


class MeetingTime(BaseModel):
    """Structured response for meeting time questions."""

    hour: int = Field(
        description="Hour on the 24-hour clock, 0-23; 2 PM is 14, 9 AM is 9.",
        ge=0,
        le=23,
    )
    minute: int = Field(description="Minute of the meeting", ge=0, le=59)
    period: str = Field(description="'AM' or 'PM', consistent with the 24-hour hour.")


class IssueCategory(str, Enum):
    """Enum for categorizing user issues."""

    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    URGENT = "urgent"


class IssueCategoryResponse(BaseModel):
    """Wrapper for IssueCategory enum to use with response_format."""

    category: IssueCategory = Field(description="The category of the issue")


# =============================================================================
# Unit Tests: Handle Initialization
# =============================================================================


class TestHandleInitialization:
    """Tests for ConversationManagerHandle initialization."""

    def test_handle_requires_event_broker(self):
        """Handle cannot be created without an event broker."""
        with pytest.raises(TypeError):
            ConversationManagerHandle(
                conversation_manager=MagicMock(),
            )

    def test_handle_stores_conversation_context(self):
        """Handle stores the broker and the conversation manager."""
        mock_broker = MagicMock()
        cm = MagicMock()

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_manager=cm,
        )

        assert handle.conversation_manager is cm
        assert handle.event_broker is mock_broker
        assert not handle._stopped

    def test_handle_not_stopped_initially(self):
        """Handle starts in active (not stopped) state."""
        handle = ConversationManagerHandle(
            event_broker=MagicMock(),
            conversation_manager=MagicMock(),
        )

        assert handle.done() is False


# =============================================================================
# Unit Tests: interject
# =============================================================================


class TestInterject:
    """Tests for interject method (publishes NotificationInjectedEvent)."""

    @pytest.mark.asyncio
    async def test_interject_publishes_event(self):
        """interject publishes NotificationInjectedEvent to the steering channel."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_manager=MagicMock(),
        )

        interjection_id = await handle.interject("Task completed successfully")

        # Returns a non-empty interjection_id
        assert isinstance(interjection_id, str) and interjection_id

        mock_broker.publish.assert_called_once()

        # Verify the published event goes to the steering channel
        call_args = mock_broker.publish.call_args
        channel = call_args[0][0]
        assert channel == "app:comms:steering"

        # Verify event payload contains the message
        message_json = call_args[0][1]
        event_data = json.loads(message_json)
        assert event_data["payload"]["content"] == "Task completed successfully"
        assert event_data["payload"]["source"] == "interjection"

    @pytest.mark.asyncio
    async def test_interject_returns_empty_when_stopped(self):
        """interject returns empty string and does nothing when handle is stopped."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_manager=MagicMock(),
        )
        await handle.stop(reason="test")

        result = await handle.interject("Should be ignored")

        assert result == ""
        mock_broker.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_interject_accepts_extra_kwargs(self):
        """interject tolerates extra kwargs without crashing."""
        mock_broker = AsyncMock()
        mock_broker.publish = AsyncMock(return_value=1)

        handle = ConversationManagerHandle(
            event_broker=mock_broker,
            conversation_manager=MagicMock(),
        )

        # Should not raise even with plumbing kwargs
        interjection_id = await handle.interject(
            "User correction incoming",
            _parent_chat_context_cont=[{"role": "user", "content": "hi"}],
        )

        assert isinstance(interjection_id, str) and interjection_id
        mock_broker.publish.assert_called_once()


# =============================================================================
# Unit Tests: stop and lifecycle methods
# =============================================================================


class TestHandleLifecycle:
    """Tests for handle lifecycle methods (stop, done, result)."""

    @pytest.mark.asyncio
    async def test_stop_marks_handle_done(self):
        """stop() marks the handle as done."""
        handle = ConversationManagerHandle(
            event_broker=MagicMock(),
            conversation_manager=MagicMock(),
        )

        assert handle.done() is False
        await handle.stop(reason="user cancelled")
        assert handle.done() is True

    @pytest.mark.asyncio
    async def test_stop_returns_reason(self):
        """stop() returns message with reason."""
        handle = ConversationManagerHandle(
            event_broker=MagicMock(),
            conversation_manager=MagicMock(),
        )

        await handle.stop(reason="task completed")
        assert handle._stopped
        result = await handle.result()
        assert "task completed" in result

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """Calling stop() multiple times is safe."""
        handle = ConversationManagerHandle(
            event_broker=MagicMock(),
            conversation_manager=MagicMock(),
        )

        await handle.stop(reason="first")
        await handle.stop(reason="second")
        assert handle._stopped

    @pytest.mark.asyncio
    async def test_result_waits_for_stop(self):
        """result() blocks until handle is stopped."""
        handle = ConversationManagerHandle(
            event_broker=MagicMock(),
            conversation_manager=MagicMock(),
        )

        # Start waiting for result in background
        result_task = asyncio.create_task(handle.result())

        # Give it a moment to start waiting
        await asyncio.sleep(0.05)
        assert not result_task.done()

        # Stop the handle
        await handle.stop(reason="completed")

        # Result should now complete
        result = await asyncio.wait_for(result_task, timeout=1.0)
        assert "completed" in result


# =============================================================================
# Unit Tests: ask_question future management
# =============================================================================


@pytest.mark.asyncio
async def test_ask_question_resets_future_after_await():
    """
    ask_question has three code paths that finish with user_reply_future.
    All of them MUST leave behind a fresh, pending future so that a
    subsequent ask_question call blocks for new user input rather than
    returning stale data or observing a dead future.

      1. "Patient mode" — future already done when ask_question is called.
      2. "Await mode"   — future is pending, so ask_question awaits it.
      3. "Timeout mode" — nobody replies, so the await times out.

    Regression 1: the await path did not reset the future, causing the
    second ask_question call to immediately return the stale first reply.

    Regression 2: the timeout path did not reset the future either, and
    ``wait_for`` leaves a timed-out future *cancelled*.  The next
    ask_question then saw ``.done() is True`` and called ``.result()`` on
    it, which raises CancelledError — aborting the whole surrounding tool
    loop, whose caller then observed the ask as having produced no answer
    at all rather than as one unanswered question.

    This mirrors the exact ask_question / user_reply_future closure from
    ConversationManagerHandle.ask() in handle.py.  Every path must reset.
    """
    user_reply_future: asyncio.Future = asyncio.Future()

    # Mirrors handle.py ask_question (minus the event_broker publish,
    # which is a side-effect irrelevant to future management).
    async def ask_question(text: str):
        nonlocal user_reply_future
        if not user_reply_future.done():
            try:
                await asyncio.wait_for(user_reply_future, timeout=0.1)
            except asyncio.TimeoutError:
                user_reply_future = asyncio.Future()
                return "Timed out waiting for user reply."
        user_msg = user_reply_future.result()
        user_reply_future = asyncio.Future()
        return f"User replied: {user_msg}"

    # --- Path 2 (await): future resolved while ask_question is awaiting ---
    asyncio.get_event_loop().call_later(
        0.05,
        user_reply_future.set_result,
        "vague answer",
    )
    result1 = await ask_question("What is your phone number?")
    assert result1 == "User replied: vague answer"

    assert not user_reply_future.done(), (
        "user_reply_future was not reset after the await path returned. "
        "A second ask_question will see .done()==True and immediately return "
        "the stale first reply instead of blocking for new user input."
    )

    # --- Path 1 (patient mode): future resolved before ask_question is called ---
    user_reply_future.set_result("555-1234")
    result2 = await ask_question("Could you give me the actual number?")
    assert result2 == "User replied: 555-1234"

    assert (
        not user_reply_future.done()
    ), "user_reply_future was not reset after the patient-mode path returned."

    # --- Path 3 (timeout): nobody replies at all ---
    result3 = await ask_question("Are you still there?")
    assert result3 == "Timed out waiting for user reply."

    assert not user_reply_future.done(), (
        "user_reply_future was not reset after the timeout path returned. "
        "wait_for cancels the future it gives up on, so the next "
        "ask_question would read a cancelled future."
    )
    assert not user_reply_future.cancelled(), (
        "user_reply_future was left cancelled after the timeout path. "
        "Reading it raises CancelledError, which aborts the surrounding "
        "tool loop instead of returning one unanswered question."
    )

    # A re-ask after the timeout must behave like any other question: block
    # for the next reply, and deliver it rather than dying on the stale future.
    asyncio.get_event_loop().call_later(
        0.02,
        user_reply_future.set_result,
        "still here",
    )
    result4 = await ask_question("Are you still there?")
    assert result4 == "User replied: still here"


@pytest.mark.asyncio
@_handle_project
async def test_real_ask_question_survives_unanswered_question(monkeypatch):
    """An unanswered question must not poison the questions that follow it.

    This drives the *real* closure built by ``ask()`` rather than a copy of
    it, by intercepting the tool set on its way into the tool loop. The
    mirrored test above documents the invariant; this one is what actually
    holds ``handle.py`` to it.

    A question nobody answers leaves its reply future cancelled unless the
    timeout path replaces it. If the next question saw a future that was
    ``done()`` and read it, reading a cancelled future would raise
    ``CancelledError`` — unwinding the whole ask loop, so the caller would be
    told the ask produced no answer at all instead of that one question went
    unanswered.
    """
    captured_tools: dict = {}

    def _capture_tools(**kwargs):
        captured_tools.update(kwargs["tools"])
        return MagicMock()

    monkeypatch.setattr(
        "unify.legacy.conversation_manager.handle.start_async_tool_loop",
        _capture_tools,
    )
    monkeypatch.setattr(
        "unify.legacy.conversation_manager.handle.USER_REPLY_TIMEOUT_S",
        0.1,
    )

    conversation_manager = MagicMock()
    conversation_manager.get_recent_transcript.return_value = ([], None)
    handle = ConversationManagerHandle(
        event_broker=AsyncMock(),
        conversation_manager=conversation_manager,
    )

    ask_handle = await handle.ask("What is the user's name?")
    ask_question = captured_tools["ask_question"]

    assert await ask_question("What is your name?") == (
        "Timed out waiting for user reply."
    )

    # The user speaks up after giving up on the first question. It must land
    # on the next question rather than on the abandoned future.
    await ask_handle.interject("My name is Bob")
    assert await ask_question("Sorry, your name?") == "User replied: My name is Bob"


# =============================================================================
# Integration Tests: ask() with PATH 1 (Inference)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_path1_infers_from_transcript(initialized_cm):
    """
    PATH 1: LLM infers answer from the chat transcript without asking user.

    When the user has already stated something in the chat, ask() should
    infer from that context without needing to ask again.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="I prefer to have meetings in the morning, around 9 AM works best for me.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    # Ask about the user's preference - should infer from transcript
    ask_handle = await handle.ask(
        "What time does the user prefer for meetings?",
        response_format=MeetingTime,
    )

    result = await ask_handle.result()

    assert isinstance(result, MeetingTime)
    assert result.hour == 9
    assert result.period.upper() == "AM"


# =============================================================================
# Integration Tests: ask() with PATH 2 (Interactive)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_path2_asks_when_ambiguous(initialized_cm):
    """
    PATH 2: LLM asks clarifying question when transcript is ambiguous.

    When there's no clear answer in the transcript, the LLM should use
    the ask_question tool to ask the user directly.
    """
    cm = initialized_cm
    # User says something unrelated to the question about to be asked
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hello, I wanted to discuss something with you today.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    questions_asked = _capture_questions(cm)

    # Ask something not in transcript - should trigger PATH 2
    ask_handle = await handle.ask(
        "What time would the user like to schedule the meeting?",
        response_format=MeetingTime,
    )

    # Wait until the question has actually been posted. ``active_ask_handle``
    # is set synchronously by ask() before it returns, so polling it says
    # nothing about whether PATH 2 has begun.
    assert await _wait_for_condition(
        lambda: len(questions_asked) >= 1,
        timeout=_ASK_RESULT_TIMEOUT,
    ), "PATH 2 never asked the user anything"

    # The handle should be waiting for user reply
    assert not ask_handle.done()

    # Simulate user reply via interject
    await ask_handle.interject("Let's do 2 PM")

    # Now get result
    result = await asyncio.wait_for(
        ask_handle.result(),
        timeout=_ASK_RESULT_TIMEOUT,
    )

    assert isinstance(result, MeetingTime)
    assert result.hour == 14  # 2 PM in 24-hour
    assert result.period.upper() == "PM"


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_path2_inbound_chat_message_answers_the_question(initialized_cm):
    """
    PATH 2: the user's next chat message is the answer.

    While a question is open, an inbound ``UnifyMessageReceived`` is delivered
    to the ask handle by the event handler itself, so the reply resolves the
    ask without a brain turn in between.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Hey, quick question about tomorrow.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    questions_asked = _capture_questions(cm)

    ask_handle = await handle.ask(
        "What time would the user like to schedule the meeting?",
        response_format=MeetingTime,
    )

    assert await _wait_for_condition(
        lambda: len(questions_asked) >= 1,
        timeout=_ASK_RESULT_TIMEOUT,
    ), "PATH 2 never asked the user anything"
    assert not ask_handle.done()

    # The reply arrives as an ordinary chat message, not through the handle.
    step = await cm.step(
        UnifyMessageReceived(content="Let's do 2 PM"),
        run_llm=False,
    )
    assert not step.llm_requested, "an open ask owns the reply; no brain turn"

    result = await asyncio.wait_for(
        ask_handle.result(),
        timeout=_ASK_RESULT_TIMEOUT,
    )

    assert isinstance(result, MeetingTime)
    assert result.hour == 14
    assert result.period.upper() == "PM"


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_path2_multiple_followup_questions(initialized_cm):
    """
    PATH 2: LLM can ask multiple follow-up questions to get clarity.

    If the user's first reply doesn't answer the question, the LLM
    should ask a follow-up question.
    """
    cm = initialized_cm
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    questions_asked = _capture_questions(cm)

    # Ask something specific
    ask_handle = await handle.ask(
        "What is the user's phone number for callbacks?",
    )

    # Wait for first question to be asked (poll for direct_speech event)
    assert await _wait_for_condition(
        lambda: len(questions_asked) >= 1,
        timeout=_ASK_RESULT_TIMEOUT,
    ), "PATH 2 never asked the user anything"

    # Give vague reply - should trigger follow-up
    await ask_handle.interject("You can reach me anytime")

    # Wait for follow-up question (poll for second question)
    await _wait_for_condition(
        lambda: len(questions_asked) >= 2 or ask_handle.done(),
        timeout=_ASK_RESULT_TIMEOUT,
    )

    # Provide actual answer
    await ask_handle.interject("+1-555-123-4567")

    result = await asyncio.wait_for(
        ask_handle.result(),
        timeout=_ASK_RESULT_TIMEOUT,
    )

    # Should have extracted the phone number
    assert "555" in result or "123" in result or "4567" in result


# =============================================================================
# Integration Tests: Structured Output (Pydantic and Enum)
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_returns_pydantic_model(initialized_cm):
    """
    ask() returns validated Pydantic model when response_format is specified.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="I strongly prefer video calls over phone calls for all meetings.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What is the user's communication preference and how confident are you?",
        response_format=UserPreference,
    )

    result = await ask_handle.result()

    assert isinstance(result, UserPreference)
    assert "video" in result.preference.lower()
    assert 0 <= result.confidence <= 1


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_returns_enum_value(initialized_cm):
    """
    ask() returns correct Enum value when response_format is an Enum.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="Can you help me understand why my payment didn't go through?",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What category of issue is this?",
        response_format=IssueCategoryResponse,
    )

    result = await ask_handle.result()

    assert isinstance(result, IssueCategoryResponse)
    assert result.category == IssueCategory.BILLING


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_without_response_format_returns_string(initialized_cm):
    """
    ask() without response_format returns a string summary.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="The product code I'm asking about is SKU-12345-XYZ.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask(
        "What product code did the user mention?",
    )

    result = await ask_handle.result()

    # Without response_format, should return string/dict
    assert "SKU-12345-XYZ" in str(result) or "12345" in str(result)


# =============================================================================
# Integration Tests: InterceptingHandle Behavior
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_intercepting_handle_delegates_lifecycle_methods(initialized_cm):
    """
    The InterceptingHandle correctly delegates stop/pause/resume/done.

    Follows the standard steerable handle pattern:
    - stop() sets the cancel flag
    - await result() waits for clean completion (returns None when stopped)
    - done() returns True after the inner loop completes
    """
    cm = initialized_cm
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask("What is the user's name?")

    # done() should return False initially (loop is running)
    assert ask_handle.done() is False

    # stop() tells the inner loop to cancel - this is non-blocking
    await ask_handle.stop(reason="cancelled by test")

    # Following standard pattern: await result() for clean completion
    # When stopped, result() returns None
    result = await asyncio.wait_for(
        ask_handle.result(),
        timeout=_ASK_RESULT_TIMEOUT,
    )
    assert result is None

    # After the inner loop completes, done() should return True
    assert ask_handle.done() is True


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_intercepting_handle_clears_active_ask_handle_on_result(initialized_cm):
    """
    When result() completes, active_ask_handle is cleared.
    """
    cm = initialized_cm
    await cm.step_until_wait(
        UnifyMessageReceived(
            content="My name is Alice Johnson.",
        ),
    )

    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    ask_handle = await handle.ask("What is the user's name?")

    # active_ask_handle should be set
    assert cm.cm.active_ask_handle is ask_handle

    # Get result
    await ask_handle.result()

    # active_ask_handle should be cleared
    assert cm.cm.active_ask_handle is None


# =============================================================================
# Integration Tests: Error Handling and Edge Cases
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_ask_raises_when_handle_stopped(initialized_cm):
    """
    ask() raises RuntimeError when called on a stopped handle.
    """
    cm = initialized_cm
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    await handle.stop(reason="already done")

    with pytest.raises(RuntimeError, match="stopped"):
        await handle.ask("This should fail")


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_ask_handles_empty_transcript(initialized_cm):
    """
    ask() handles case where transcript is empty.

    Should fall back to PATH 2 and ask the user directly.
    """
    cm = initialized_cm
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    questions_asked = _capture_questions(cm)

    ask_handle = await handle.ask("What is the user's name?")

    # Wait until the question has actually been posted. ``active_ask_handle``
    # is set synchronously by ask() before it returns, so polling it says
    # nothing about whether PATH 2 has begun.
    assert await _wait_for_condition(
        lambda: len(questions_asked) >= 1,
        timeout=_ASK_RESULT_TIMEOUT,
    ), "PATH 2 never asked the user anything"

    # Should be waiting for user input (PATH 2)
    assert not ask_handle.done()

    # Provide answer via interject
    await ask_handle.interject("My name is Bob")

    result = await asyncio.wait_for(
        ask_handle.result(),
        timeout=_ASK_RESULT_TIMEOUT,
    )

    assert "bob" in result.lower()


# =============================================================================
# Integration Tests: Concurrent ask() calls
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_only_one_active_ask_handle_at_a_time(initialized_cm):
    """
    Only one ask handle can be active at a time.

    Starting a new ask() should replace the previous active_ask_handle.
    Follows established cleanup pattern: stop() + await result().
    """
    cm = initialized_cm
    handle = ConversationManagerHandle(
        event_broker=cm.event_broker,
        conversation_manager=cm.cm,
    )

    # Start first ask
    ask_handle_1 = await handle.ask("What is the user's favorite color?")

    # Verify it's active
    assert cm.cm.active_ask_handle is ask_handle_1

    # Stop first handle before starting second (clean replacement)
    await ask_handle_1.stop(reason="replaced by new ask")

    # Start second ask (should replace first)
    ask_handle_2 = await handle.ask("What is the user's favorite food?")

    # Second should now be active
    assert cm.cm.active_ask_handle is ask_handle_2

    # Complete second ask properly
    await ask_handle_2.interject("I love pizza")
    await ask_handle_2.result()

    # Cleanup: ensure first handle completes
    try:
        await asyncio.wait_for(ask_handle_1.result(), timeout=5.0)
    except Exception:
        pass  # May already be cleaned up
