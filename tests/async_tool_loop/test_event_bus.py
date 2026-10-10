# tests/async_tool_loop/test_event_bus.py
#
# These tests assume the project already contains
# ─  async_tool_use_loop.py   (with _async_tool_use_loop_inner / start_async_tool_loop)
# ─  event_bus.py            (with EventBus / Event)
#
# No stubs for “unify” are provided – the real library is expected to be
# importable in the test environment.

from __future__ import annotations

import asyncio
import re

import pytest

from unify.common.async_tool_loop import (
    start_async_tool_loop,
)
from unify.common._async_tool.loop import async_tool_loop_inner
from tests.helpers import _handle_project, capture_events
from unify.common.llm_client import new_llm_client

_SUFFIX_RE = re.compile(r"\(([0-9a-f]{4})\)$")

pytestmark = pytest.mark.llm_call


async def echo(text: str) -> str:  # noqa: D401 – simple echo tool
    # Avoid time-based sleeping; just return immediately
    return text.upper()


# --------------------------------------------------------------------------- #
#                         Integration-level expectations                       #
# --------------------------------------------------------------------------- #


_INFRASTRUCTURE_EVENT_KINDS = {"thinking_sentinel"}


def _filter_runtime_context(events: list) -> list:
    """Filter out internal runtime/infrastructure events that don't represent conversation turns."""
    return [
        evt
        for evt in events
        if not evt.payload.get("message", {}).get("_runtime_context")
        and evt.payload.get("kind") not in _INFRASTRUCTURE_EVENT_KINDS
    ]


@pytest.mark.asyncio
async def test_basic_event_flow() -> None:
    """
    End-to-end check:

        user/msg → assistant/tool-call → tool/result → assistant/final-text

    The loop announces nothing around a call (no lifecycle messages, no
    visibility guidance), so a loop making one call publishes exactly these
    four messages. The model is scripted: nothing leaves the process.
    """
    from tests import cache_discipline_helpers as h

    replies = [
        lambda: h.completion(calls=[("echo", {"text": "world"})]),
        lambda: h.completion(content="WORLD"),
    ]
    with h.scripted(replies):
        async with capture_events("ToolLoop") as captured_events:
            await async_tool_loop_inner(
                client=h.new_client("You are an automated test agent."),
                message="world",
                tools={"echo": echo},
                interject_queue=asyncio.Queue(),
                cancel_event=asyncio.Event(),
                prune_tool_duplicates=True,
                time_awareness=False,
            )

    events = _filter_runtime_context(captured_events)
    roles = [evt.payload["message"]["role"] for evt in events]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert events[0].payload["message"]["content"] == "world"
    assert events[2].payload["message"]["content"].strip("'").strip('"') == "WORLD"
    assert events[3].payload["message"]["content"] == "WORLD"


# --------------------------------------------------------------------------- #
#               Publishing still works while the loop is running              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_interjection_publishes_user_event(llm_config) -> None:
    """
    Verify that interjections are published to the event bus as user messages.

    This test is purely about event bus mechanics, not model behavior.
    Model response quality and instruction-following are tested separately
    in test_interjections.py.
    """
    client = new_llm_client(**llm_config)
    opening_turn = "Event bus check: this is the opening user turn."
    interjected_turn = "Event bus check: this is the interjected user turn."
    client.set_system_message(
        "You are in an automated event-bus test. "
        "Briefly acknowledge each user message. Do not call tools.",
    )

    async with capture_events("ToolLoop") as captured_events:
        handle = start_async_tool_loop(
            client=client,
            message=opening_turn,
            tools={},
            max_consecutive_failures=1,
        )

        await handle.submit(interjected_turn)

        # We don't need to verify model output - just let it complete
        await handle.result()

    # Filter out internal runtime context events
    events = _filter_runtime_context(captured_events)
    roles = [evt.payload["message"]["role"] for evt in events]

    # EVENT BUS ASSERTIONS ONLY - no model behavior checks
    assert (
        roles.count("user") == 2
    ), "Event bus should record both initial and interjected user messages"

    user_contents = [
        evt.payload["message"].get("content", "")
        for evt in events
        if evt.payload["message"]["role"] == "user"
    ]
    assert any(
        "opening user turn" in c for c in user_contents
    ), "Initial message should be recorded"
    assert any(
        "interjected user turn" in c for c in user_contents
    ), "Interjection should be recorded"


# --------------------------------------------------------------------------- #
#          Tool results contain actual content, not placeholders              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_content_is_not_placeholder(llm_config) -> None:
    """
    Verify that tool messages published to EventBus contain actual tool results,
    not placeholder text like "Streaming..." or "In progress...".

    This is a regression test for an issue where ToolData objects were passed
    by reference to the EventBus, and the content was still a placeholder when
    the event was published (before the actual result was available).
    """
    # Use a deterministic tool that returns known content
    expected_result = "EXPECTED_TOOL_OUTPUT_12345"

    async def deterministic_tool(input_text: str) -> str:
        """A tool that returns a predictable result for testing."""
        return expected_result

    client = new_llm_client(**llm_config).set_system_message(
        "You are an automated test agent.\n"
        "You MUST call the tool named `deterministic_tool` exactly once, "
        "passing any text as the `input_text` argument.\n"
        "After the tool returns, reply with the tool result.",
    )

    async with capture_events("ToolLoop") as captured_events:
        await async_tool_loop_inner(
            client=client,
            message="please call the tool",
            tools={"deterministic_tool": deterministic_tool},
            interject_queue=asyncio.Queue(),
            cancel_event=asyncio.Event(),
            prune_tool_duplicates=True,
        )

    # Filter out internal runtime context events
    events = _filter_runtime_context(captured_events)

    # Find tool result events
    tool_events = [evt for evt in events if evt.payload["message"]["role"] == "tool"]

    assert len(tool_events) >= 1, "Should have at least one tool result event"

    # Verify tool result content is the actual result, not a placeholder
    tool_content = tool_events[0].payload["message"]["content"]

    # Check it's NOT a placeholder
    placeholder_patterns = [
        "Streaming...",
        "In progress...",
        "Loading...",
        "Pending...",
        "...",  # Common placeholder suffix
    ]
    for pattern in placeholder_patterns:
        assert (
            pattern not in tool_content or expected_result in tool_content
        ), f"Tool content appears to be a placeholder: {tool_content!r}"

    # Check it IS the expected result
    assert (
        expected_result in tool_content
    ), f"Tool content should contain '{expected_result}', got: {tool_content!r}"


# --------------------------------------------------------------------------- #
#                     ask() boundary event tests                               #
# --------------------------------------------------------------------------- #


def _extract_suffix(hierarchy_label: str) -> str | None:
    """Extract the trailing 4-hex-char suffix from a hierarchy_label."""
    m = _SUFFIX_RE.search(hierarchy_label)
    return m.group(1) if m else None
