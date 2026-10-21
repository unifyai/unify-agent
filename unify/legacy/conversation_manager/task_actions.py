"""
Centralized utilities for action steering operations in ConversationManager.

This module provides a single source of truth for:
- Steering operations derived from SteerableToolHandle
- Short name derivation from action queries (pane labels)
- Rendered steering-tool invocations for the state panes

Steering itself happens through six fixed, handle_id-addressed brain tools
(``ConversationManagerBrainActionTools.build_action_steering_tools``); the
``in_flight_actions`` and ``completed_actions`` panes render ready-to-use
invocations such as ``interject_action(handle_id=3, ...)`` built here.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

from ...common.async_tool_loop import SteerableToolHandle

# Derived labels and call-id suffixes never contain a double underscore, so
# they stay visually unambiguous next to snake_case identifiers.
_DELIM = "__"


# ─────────────────────────────────────────────────────────────────────────────
# Steering operations derived from SteerableToolHandle
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SteeringOperation:
    """Describes a steering operation that can be performed on an active action."""

    name: str  # e.g., "ask", "stop"
    method_name: str  # Method name on SteerableToolHandle
    param_name: (
        str  # Primary parameter name for the action (e.g., "question", "reason")
    )
    requires_clarification: bool = False  # Whether this needs a call_id suffix

    def get_docstring(self) -> str:
        """Get the full docstring from SteerableToolHandle."""
        method = getattr(SteerableToolHandle, self.method_name, None)
        if method is None:
            return ""
        doc = inspect.getdoc(method)
        if not doc:
            return ""
        return doc.strip()


# Core steering operations - derived from SteerableToolHandle's abstract methods
STEERING_OPERATIONS: tuple[SteeringOperation, ...] = (
    SteeringOperation("ask", "ask", "question"),
    SteeringOperation("stop", "stop", "reason"),
    SteeringOperation("interject", "interject", "message"),
    SteeringOperation("pause", "pause", ""),
    SteeringOperation("resume", "resume", ""),
    SteeringOperation(
        "answer_clarification",
        "answer_clarification",
        "answer",
        requires_clarification=True,
    ),
)

# Operation name -> SteeringOperation mapping
OPERATION_MAP: dict[str, SteeringOperation] = {
    op.name: op for op in STEERING_OPERATIONS
}


# ─────────────────────────────────────────────────────────────────────────────
# Short name derivation
# ─────────────────────────────────────────────────────────────────────────────

# Maximum character length for short_name, keeping pane labels compact.
_MAX_SHORT_NAME_CHARS = 25

# Characters to strip entirely (no word boundary created).
# These appear within words (contractions, quotes) or terminate sentences.
# - Apostrophes: ' and Unicode variants ' '
# - Quotes: " and Unicode variants " " plus backtick `
# - Sentence terminators: ! ?
_STRIP_CHARS_PATTERN = re.compile(r"['\u2018\u2019\"\u201c\u201d`!?]")


def derive_short_name(query: str, max_words: int = 4) -> str:
    """Derive a short, descriptive label from an action query for the state panes.

    Takes the first few words, lowercased, joined by underscores. Punctuation is
    handled in two ways:
    - Apostrophes, quotes, and sentence terminators are stripped (no word boundary)
    - Other punctuation (slashes, hyphens, etc.) becomes word separators

    Ensures no __ appears in the label and enforces a character limit to keep
    pane labels compact.

    Examples:
        "List all invoices" -> "list_all_invoices"
        "What's the weather?" -> "whats_the_weather"
        "Get docs/files/data" -> "get_docs_files_data"
    """
    # First, strip apostrophes/quotes/terminators (they don't create word boundaries)
    query = _STRIP_CHARS_PATTERN.sub("", query)
    # Then replace remaining non-alphanumeric chars with spaces (word separators)
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", query.lower())
    words = normalized.split()[:max_words]
    result = "_".join(words) if words else "task"

    # Collapse any accidental double underscores (__ is our structural delimiter)
    while _DELIM in result:
        result = result.replace(_DELIM, "_")

    # Truncate to max length to guarantee tool names stay under 64 chars
    if len(result) > _MAX_SHORT_NAME_CHARS:
        result = result[:_MAX_SHORT_NAME_CHARS].rstrip("_")

    return result


def safe_call_id_suffix(call_id: str) -> str:
    """Abbreviate a clarification call_id to its last-8-character suffix.

    ``answer_clarification_action`` accepts the suffix as shorthand for the
    full call id. Never contains __ .
    """
    if not call_id:
        return "0"
    result = call_id.replace("-", "_")[-8:]
    # Collapse any accidental double underscores
    while _DELIM in result:
        result = result.replace(_DELIM, "_")
    return result


def iter_steering_tools_for_action(
    handle_id: int,
    query: str,
    pending_clarifications: list[dict] | None = None,
    is_paused: bool | None = None,
) -> list[tuple[str, str]]:
    """Generate (action_name, description) pairs for an action's steering tools.

    Args:
        handle_id: The action handle ID
        query: The original action query
        pending_clarifications: List of pending clarification dicts with "call_id" keys
        is_paused: If True, only include resume (skip pause).
                   If False, only include pause (skip resume).
                   If None, include both.

    Returns:
        List of (action_name, description) tuples, rendered as ready-to-use
        invocations of the fixed steering tools
        (``interject_action(handle_id=3, ...)``).
    """
    actions = []

    for op in STEERING_OPERATIONS:
        # Conditionally skip pause/resume based on current state
        if is_paused is not None:
            if op.name == "pause" and is_paused:
                continue  # Skip pause when already paused
            if op.name == "resume" and not is_paused:
                continue  # Skip resume when not paused (running)

        if op.requires_clarification:
            # Only surface answer_clarification if there are pending ones
            for clar in pending_clarifications or []:
                call_id = clar.get("call_id", "")
                name = (
                    f"{op.name}_action(handle_id={handle_id}, "
                    f"call_id='{call_id}', ...)"
                )
                desc = (
                    op.get_docstring()
                    or f"{op.name.replace('_', ' ').title()} this action"
                )
                actions.append((name, desc))
        else:
            name = f"{op.name}_action(handle_id={handle_id}, ...)"
            desc = (
                op.get_docstring() or f"{op.name.replace('_', ' ').title()} this action"
            )
            actions.append((name, desc))

    return actions


def iter_steering_tools_for_completed_action(
    handle_id: int,
    query: str,
) -> list[tuple[str, str]]:
    """Generate (action_name, description) pairs for a completed action.

    Completed actions expose `ask` for querying the preserved trajectory.

    Args:
        handle_id: The action handle ID
        query: The original action query

    Returns:
        List of (action_name, description) tuples
    """
    actions = []

    ask_op = OPERATION_MAP["ask"]
    ask_name = f"ask_action(handle_id={handle_id}, ...)"
    ask_desc = ask_op.get_docstring() or "Ask about this completed action"
    actions.append((ask_name, ask_desc))

    return actions
