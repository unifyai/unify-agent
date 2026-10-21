"""
Tests for the `persist=True` mode in the async tool loop.

When `persist=True`, the loop does not terminate when the LLM produces content
without tool calls. Instead, it blocks waiting for the next interjection. This
enables a single persistent loop that can process multiple events over time.

The model is scripted (``tests/scripted_model.py``): each test writes the
model's replies, and nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
)

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.async_tool_loop import start_async_tool_loop


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #
async def echo(text: str) -> str:
    """Echo back the input text."""
    await asyncio.sleep(0.05)
    return f"echoed: {text}"


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def _add(x: int, y: int):
    """A scripted reply that calls ``add``."""
    return reply(calls=[("add", {"x": x, "y": y})])


# --------------------------------------------------------------------------- #
#  BASIC PERSIST MODE                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_persist_mode_waits_for_interjection(llm_config):
    """
    In persist mode, after the LLM produces content without tool calls,
    the loop blocks waiting for an interjection instead of returning.
    """
    model = ScriptedModel(
        query_llm=[
            reply("hello"),
            reply("goodbye"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        handle = start_async_tool_loop(
            client,
            message="Say hello and nothing else.",
            tools={},
            persist=True,
            timeout=60,
        )

        # Wait for the LLM to produce initial content
        async def _has_assistant_response() -> bool:
            return any(
                m.get("role") == "assistant" and m.get("content")
                for m in (client.messages or [])
            )

        await _wait_for_condition(_has_assistant_response, poll=0.05, timeout=30.0)

        # The loop should NOT be done - it should be waiting for an interjection
        await asyncio.sleep(0.2)  # Small delay to ensure loop has reached wait state
        assert (
            not handle.done()
        ), "Persist loop should not terminate after first response"

        # Now interject to give it something to process
        await handle.submit("Now say goodbye.")

        # Wait for a second assistant response
        async def _has_two_responses() -> bool:
            assistant_msgs = [
                m
                for m in (client.messages or [])
                if m.get("role") == "assistant" and m.get("content")
            ]
            return len(assistant_msgs) >= 2

        await _wait_for_condition(_has_two_responses, poll=0.05, timeout=30.0)

        # Still should not be done - waiting for more
        await asyncio.sleep(0.2)
        assert (
            not handle.done()
        ), "Persist loop should continue waiting after interjection"

        # Stop the loop explicitly
        await handle.stop()
        result = await handle.result()
        assert result == "processed stopped early, no result"
    assert model.kinds() == ["query_llm", "query_llm"]


@pytest.mark.asyncio
async def test_persist_mode_processes_multiple_interjections(llm_config):
    """
    The persist loop should process multiple interjections sequentially.
    """
    model = ScriptedModel(
        actor=[
            reply("ready"),
            _add(2, 3),
            reply("5"),
            _add(10, 20),
            reply("30"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        handle = start_async_tool_loop(
            client,
            message="Reply with 'ready' and wait for my instructions.",
            tools={"add": add},
            persist=True,
            timeout=60,
        )

        # Wait for initial response
        async def _has_response() -> bool:
            return any(
                m.get("role") == "assistant" and m.get("content")
                for m in (client.messages or [])
            )

        await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

        # First interjection - ask it to add
        await handle.submit(
            "Use the add tool to compute 2 + 3, then tell me the result.",
        )

        # Wait for the tool to be called and response generated
        await _wait_for_tool_request(client, "add")

        async def _has_tool_result() -> bool:
            return any(m.get("role") == "tool" for m in (client.messages or []))

        await _wait_for_condition(_has_tool_result, poll=0.05, timeout=30.0)

        # Wait for assistant to process tool result
        async def _has_result_response() -> bool:
            msgs = client.messages or []
            tool_idx = next(
                (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
                -1,
            )
            if tool_idx < 0:
                return False
            return any(
                m.get("role") == "assistant" and m.get("content")
                for m in msgs[tool_idx + 1 :]
            )

        await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

        # Verify the result is in the response
        assistant_msgs = [
            m.get("content", "")
            for m in (client.messages or [])
            if m.get("role") == "assistant" and m.get("content")
        ]
        assert any("5" in msg for msg in assistant_msgs), "Should have computed 2+3=5"

        # Second interjection
        await handle.submit("Now add 10 and 20.")

        # Wait for second tool call
        async def _has_second_tool_call() -> bool:
            tool_calls = [
                m
                for m in (client.messages or [])
                if m.get("role") == "tool" and m.get("name") == "add"
            ]
            return len(tool_calls) >= 2

        await _wait_for_condition(_has_second_tool_call, poll=0.05, timeout=30.0)

        # Wait for the reply to the second result before stopping, so that
        # every scripted reply is asked for.
        async def _has_second_result_response() -> bool:
            msgs = client.messages or []
            tool_idxs = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
            if len(tool_idxs) < 2:
                return False
            return any(
                m.get("role") == "assistant" and m.get("content")
                for m in msgs[tool_idxs[1] + 1 :]
            )

        await _wait_for_condition(_has_second_result_response, poll=0.05, timeout=30.0)

        # Stop and verify
        await handle.stop()
        await handle.result()

        # Verify both additions were performed
        tool_msgs = [m for m in (client.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) >= 2, "Should have made at least 2 tool calls"
    assert model.kinds() == ["actor"] * 5


@pytest.mark.asyncio
async def test_persist_mode_terminates_on_stop(llm_config):
    """
    A persist loop should terminate gracefully when handle.stop() is called.
    """
    model = ScriptedModel(
        query_llm=[
            reply("waiting"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        handle = start_async_tool_loop(
            client,
            message="Say 'waiting' and wait.",
            tools={},
            persist=True,
            timeout=60,
        )

        # Wait for initial response
        async def _has_response() -> bool:
            return any(m.get("role") == "assistant" for m in (client.messages or []))

        await _wait_for_condition(_has_response, poll=0.05, timeout=30.0)

        # Stop the loop
        await handle.stop()

        # Should terminate with the standard stop message
        result = await handle.result()
        assert result == "processed stopped early, no result"
        assert handle.done()
    assert model.kinds() == ["query_llm"]


# --------------------------------------------------------------------------- #
#  NON-PERSIST MODE REGRESSION TEST                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_non_persist_mode_terminates_normally(llm_config):
    """
    Without persist=True, the loop should terminate as normal when the LLM
    produces content without tool calls (regression test).
    """
    model = ScriptedModel(
        query_llm=[
            reply("hello"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        handle = start_async_tool_loop(
            client,
            message="Say hello and nothing else.",
            tools={},
            persist=False,  # Explicitly false (the default)
            timeout=60,
        )

        # The loop should terminate with the assistant's response
        result = await handle.result()

        assert result.strip(), "Should have a non-empty response"
        assert handle.done(), "Loop should be done"

        # Verify only one assistant message (no waiting for interjections)
        assistant_msgs = [
            m for m in (client.messages or []) if m.get("role") == "assistant"
        ]
        assert len(assistant_msgs) == 1, "Should have exactly one assistant response"
    assert model.kinds() == ["query_llm"]


@pytest.mark.asyncio
async def test_persist_mode_with_tool_calls(llm_config):
    """
    Persist mode should handle tool calls normally and only wait after
    the LLM produces content without tool calls.
    """
    model = ScriptedModel(
        actor=[
            _add(1, 1),
            reply("2"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        handle = start_async_tool_loop(
            client,
            message="Use the add tool to compute 1 + 1, then tell me the result.",
            tools={"add": add},
            persist=True,
            timeout=60,
        )

        # Wait for the tool to be called
        await _wait_for_tool_request(client, "add")

        # Wait for the assistant to respond with the result
        async def _has_result_response() -> bool:
            msgs = client.messages or []
            tool_idx = next(
                (i for i, m in enumerate(msgs) if m.get("role") == "tool"),
                -1,
            )
            if tool_idx < 0:
                return False
            return any(
                m.get("role") == "assistant" and m.get("content")
                for m in msgs[tool_idx + 1 :]
            )

        await _wait_for_condition(_has_result_response, poll=0.05, timeout=30.0)

        # Verify the result contains "2"
        assistant_msgs = [
            m.get("content", "")
            for m in (client.messages or [])
            if m.get("role") == "assistant" and m.get("content")
        ]
        assert any("2" in msg for msg in assistant_msgs), "Should have computed 1+1=2"

        # Loop should still be alive, waiting for next interjection
        await asyncio.sleep(0.2)
        assert not handle.done(), "Persist loop should wait after tool call completes"

        # Clean up
        await handle.stop()
        await handle.result()
    assert model.kinds() == ["actor", "actor"]
