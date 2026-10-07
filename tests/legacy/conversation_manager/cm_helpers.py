"""
tests/conversation_manager/cm_helpers.py
=============================================

Shared helper functions for ConversationManager tests.

These helpers are specific to the event-driven testing paradigm used by
ConversationManager tests, where we step through events and filter output
events by type.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypeVar

from tests.assertion_helpers import assertion_failed

if TYPE_CHECKING:
    from tests.legacy.conversation_manager.cm_test_driver import (
        CMStepDriver,
        StepResult,
    )

T = TypeVar("T")


def normalize_identifier_tokens(text: str) -> str:
    """Lowercase and treat spaces/underscores as equivalent (``email_id`` ≈ ``email id``)."""
    return re.sub(r"[\s_]+", "", text.lower())


# =============================================================================
# Event Filtering Helpers
# =============================================================================


def filter_events_by_type(events: list, typ: type[T]) -> list[T]:
    """Filter a list of events by type.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to filter by (e.g., UnifyMessageSent, ActorHandleStarted)

    Returns:
        List of events matching the given type.
    """
    return [e for e in events if isinstance(e, typ)]


def get_exactly_one(events: list, typ: type[T]) -> T:
    """Get exactly one event of the given type, asserting there is exactly one.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to find (e.g., UnifyMessageSent, ActorHandleStarted)

    Returns:
        The single event of the given type.

    Raises:
        AssertionError: If there is not exactly one event of the given type.
    """
    matches = filter_events_by_type(events, typ)
    assert len(matches) == 1, f"Expected 1 {typ.__name__}, got {len(matches)}"
    return matches[0]


def assert_has_one(events: list, typ: type) -> bool:
    """Assert exactly one event of the given type exists.

    Args:
        events: List of events (typically StepResult.output_events)
        typ: Event type to check for (e.g., UnifyMessageSent, ActorHandleStarted)

    Returns:
        True if exactly one event exists.

    Raises:
        AssertionError: If there is not exactly one event of the given type.
    """
    matches = filter_events_by_type(events, typ)
    count = len(matches)
    assert count == 1, f"Expected exactly 1 {typ.__name__}, got {count}"
    return True


# =============================================================================
# Efficiency Assertion Helpers
# =============================================================================


def assert_efficient(result: "StepResult", max_steps: int, context: str = "") -> None:
    """Assert that the LLM completed within the expected number of steps.

    The appropriate max_steps depends on the context:
    - 3 steps: Simple interactions (small talk, acknowledgments, synchronous steering)
    - 5 steps: Complex interactions (async ask_* with interim ack + result relay,
               proactive behavior from actionable context like time constraints)

    Args:
        result: StepResult from CMStepDriver.step_until_wait()
        max_steps: Maximum acceptable number of LLM steps for this context.
        context: Optional context string for error messages.

    Raises:
        AssertionError: If the LLM took more than max_steps.
    """
    assert result.llm_step_count <= max_steps, (
        f"Expected efficient tool calling (<= {max_steps} steps), "
        f"but took {result.llm_step_count} steps. {context}"
    )


# =============================================================================
# Action and Steering Helpers
# =============================================================================


def get_in_flight_action_count(cm: "CMStepDriver") -> int:
    """Get the number of in-flight actions from the ConversationManager.

    Args:
        cm: CMStepDriver instance wrapping the ConversationManager.

    Returns:
        Number of in-flight actions (actor handles currently doing work).
    """
    return len(cm.cm.in_flight_actions or {})


def has_steering_tool_call(cm: "CMStepDriver", operation_prefix: str) -> bool:
    """Check if any steering tool with the given operation prefix was called.

    Checks all tool calls made during the test (tracked by CMStepDriver),
    not just in-flight actions, since completed/stopped actions are removed from
    in_flight_actions.

    Args:
        cm: CMStepDriver instance wrapping the ConversationManager.
        operation_prefix: Prefix to match (e.g., "pause_", "stop_", "interject_").

    Returns:
        True if a steering tool with the given prefix was called.
    """
    for tool_name in cm.all_tool_calls:
        if tool_name.startswith(operation_prefix):
            return True
    return False


def get_steering_action(result: "StepResult", operation_prefix: str) -> str | None:
    """Check if the LLM called a steering tool with the given operation prefix.

    The action name pattern is: {operation}_{short_name}__{handle_id}

    Args:
        result: StepResult from CMStepDriver.step() or step_until_wait().
        operation_prefix: Prefix to match (e.g., "pause_", "stop_", "interject_").

    Returns:
        The action name if found, None otherwise.
    """
    if hasattr(result, "last_action") and result.last_action:
        action = result.last_action
        if action.startswith(operation_prefix):
            return action
    return None


# =============================================================================
# Assertion Helpers with Detailed Error Messages
# =============================================================================


def build_cm_context(
    cm: "CMStepDriver | None" = None,
    result: "StepResult | None" = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build context dictionary for assertion_failed from CM test state.

    Args:
        cm: CMStepDriver instance (optional).
        result: StepResult from step()/step_until_wait() (optional).
        **extra: Additional context key-value pairs.

    Returns:
        Context dictionary suitable for assertion_failed().
    """
    context: dict[str, Any] = {}

    if cm is not None:
        context["all_tool_calls"] = cm.all_tool_calls
        if hasattr(cm, "cm") and hasattr(cm.cm, "in_flight_actions"):
            context["in_flight_actions"] = list(cm.cm.in_flight_actions.keys())

    if result is not None:
        context["output_event_types"] = [type(e).__name__ for e in result.output_events]
        context["llm_step_count"] = result.llm_step_count
        context["llm_requested"] = result.llm_requested
        context["llm_ran"] = result.llm_ran

    context.update(extra)
    return context


def assert_act_triggered(
    result: "StepResult",
    event_type: type,
    description: str,
    cm: "CMStepDriver | None" = None,
) -> None:
    """Assert that the 'act' method was triggered (ActorHandleStarted event).

    Args:
        result: StepResult from step_until_wait().
        event_type: The event type to check for (e.g., ActorHandleStarted).
        description: Human-readable description of the expected action.
        cm: Optional CMStepDriver for additional context.

    Raises:
        AssertionError: With detailed context if the assertion fails.
    """
    actor_events = filter_events_by_type(result.output_events, event_type)
    assert len(actor_events) >= 1, assertion_failed(
        expected=f"At least 1 {event_type.__name__} event",
        actual=f"{len(actor_events)} events of type {event_type.__name__}",
        reasoning=[],  # CM tests don't have LLM reasoning steps in the same format
        description=description,
        context_data=build_cm_context(cm=cm, result=result),
    )


def assert_steering_called(
    cm: "CMStepDriver",
    operation_prefix: str,
    description: str,
    result: "StepResult | None" = None,
) -> None:
    """Assert that a steering tool with the given prefix was called.

    Args:
        cm: CMStepDriver instance.
        operation_prefix: Prefix to match (e.g., "ask_", "stop_", "pause_").
        description: Human-readable description of the expected action.
        result: Optional StepResult for additional context.

    Raises:
        AssertionError: With detailed context if the assertion fails.
    """
    assert has_steering_tool_call(cm, operation_prefix), assertion_failed(
        expected=f"Steering tool with prefix '{operation_prefix}'",
        actual=f"No tool call starting with '{operation_prefix}'",
        reasoning=[],
        description=description,
        context_data=build_cm_context(cm=cm, result=result),
    )


def assert_content_contains(
    content: str,
    expected_substring: str,
    description: str,
    cm: "CMStepDriver | None" = None,
    result: "StepResult | None" = None,
    case_sensitive: bool = False,
) -> None:
    """Assert that content contains the expected substring.

    Args:
        content: The content string to check.
        expected_substring: The substring that should be present.
        description: Human-readable description of what we're checking.
        cm: Optional CMStepDriver for additional context.
        result: Optional StepResult for additional context.
        case_sensitive: Whether the check is case-sensitive (default False).

    Raises:
        AssertionError: With detailed context if the assertion fails.
    """
    check_content = content if case_sensitive else content.lower()
    check_substring = (
        expected_substring if case_sensitive else expected_substring.lower()
    )

    assert check_substring in check_content, assertion_failed(
        expected=f"Content containing '{expected_substring}'",
        actual=content,
        reasoning=[],
        description=description,
        context_data=build_cm_context(cm=cm, result=result),
    )


def assert_response_type(
    result: Any,
    expected_type: type,
    description: str,
    cm: "CMStepDriver | None" = None,
) -> None:
    """Assert that a result is of the expected type.

    Args:
        result: The result to check.
        expected_type: The expected type.
        description: Human-readable description of what we're checking.
        cm: Optional CMStepDriver for additional context.

    Raises:
        AssertionError: With detailed context if the assertion fails.
    """
    assert isinstance(result, expected_type), assertion_failed(
        expected=f"Instance of {expected_type.__name__}",
        actual=f"Instance of {type(result).__name__}: {result!r}",
        reasoning=[],
        description=description,
        context_data=build_cm_context(cm=cm),
    )
