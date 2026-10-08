"""
Tests for the `persist=True` mode in the async tool loop.

When `persist=True`, the loop does not terminate when the LLM produces content
without tool calls. Instead, it blocks waiting for the next interjection. This
enables a single persistent loop that can process multiple events over time.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_condition,
    _wait_for_tool_request,
)

from unify.common.async_tool_loop import start_async_tool_loop

pytestmark = pytest.mark.llm_call


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


# --------------------------------------------------------------------------- #
#  RESPONSE FORMAT MODELS                                                     #
# --------------------------------------------------------------------------- #
class Greeting(BaseModel):
    """Simple structured response for persist-mode tests."""

    message: str = Field(..., description="A greeting message.")
    number: int = Field(..., description="Any integer chosen by the model.")


# --------------------------------------------------------------------------- #
#  BASIC PERSIST MODE                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_waits_for_interjection(llm_config):
    """
    In persist mode, after the LLM produces content without tool calls,
    the loop blocks waiting for an interjection instead of returning.
    """
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
    assert not handle.done(), "Persist loop should not terminate after first response"

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
    assert not handle.done(), "Persist loop should continue waiting after interjection"

    # Stop the loop explicitly
    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_processes_multiple_interjections(llm_config):
    """
    The persist loop should process multiple interjections sequentially.
    """
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

    # Stop and verify
    await handle.stop()
    await handle.result()

    # Verify both additions were performed
    tool_msgs = [m for m in (client.messages or []) if m.get("role") == "tool"]
    assert len(tool_msgs) >= 2, "Should have made at least 2 tool calls"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_terminates_on_stop(llm_config):
    """
    A persist loop should terminate gracefully when handle.stop() is called.
    """
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


# --------------------------------------------------------------------------- #
#  NON-PERSIST MODE REGRESSION TEST                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_non_persist_mode_terminates_normally(llm_config):
    """
    Without persist=True, the loop should terminate as normal when the LLM
    produces content without tool calls (regression test).
    """
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


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_with_tool_calls(llm_config):
    """
    Persist mode should handle tool calls normally and only wait after
    the LLM produces content without tool calls.
    """
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


# --------------------------------------------------------------------------- #
#  PERSIST MODE: send_response DOES NOT TERMINATE THE LOOP                    #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_does_not_terminate_on_send_response(llm_config):
    """In persist mode with ``response_format``, calling ``send_response``
    should NOT terminate the loop.

    When ``response_format`` is set the loop injects a ``send_response`` tool
    (named ``final_response`` in non-persist mode) and forces
    ``tool_choice=required``, so the LLM *must* call it.  In non-persist mode,
    ``final_response`` returns the structured payload and exits the loop.

    In **persist** mode the loop treats the ``send_response`` payload as the
    response for the *current* turn, then continues waiting for the next
    interjection — just like it does for a plain text-only response.
    """
    client = new_llm_client(**llm_config)

    client.set_system_message(
        "When asked, respond with a JSON object containing exactly two keys: "
        "'message' (a greeting) and 'number' (an integer). Do not include any "
        "extra keys or commentary.",
    )

    handle = start_async_tool_loop(
        client,
        message="Say hello and pick a number.",
        tools={},
        persist=True,
        response_format=Greeting,
        timeout=60,
    )

    # Wait for the LLM to call send_response (it must, because
    # tool_choice=required and send_response is the only tool available).
    await _wait_for_tool_request(client, "send_response")

    # Give the loop time to process the send_response call
    await asyncio.sleep(0.5)

    # In persist mode the loop should still be alive, waiting for the next
    # interjection.
    assert not handle.done(), (
        "Persist loop should NOT terminate after send_response — "
        "it should continue waiting for interjections."
    )

    # Interject and get a second response.
    await handle.submit("Now greet me in French and pick a different number.")

    # Wait for the LLM to produce a second send_response call
    async def _has_second_send_response() -> bool:
        count = sum(
            1
            for m in (client.messages or [])
            if m.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "send_response"
                for tc in (m.get("tool_calls") or [])
            )
        )
        return count >= 2

    await _wait_for_condition(_has_second_send_response, poll=0.05, timeout=30.0)

    # Still alive
    await asyncio.sleep(0.2)
    assert (
        not handle.done()
    ), "Persist loop should survive multiple send_response calls."

    # Explicit stop is the only way to end a persist loop
    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


# --------------------------------------------------------------------------- #
#  PERSIST MODE: response_format + tools (send_response masked during tools)  #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_response_format_with_tools(llm_config):
    """Persist mode with both ``response_format`` and real tools.

    This is the most realistic persist-mode pattern: a structured-output
    agent with tools that runs across multiple turns.  The test verifies:

    1. The LLM can call real tools (``add``) normally.
    2. ``send_response`` is only offered after tools complete (masked while
       tools are in-flight).
    3. After calling ``send_response``, the loop does NOT terminate.
    4. A second interjection triggers another tool-use → ``send_response``
       cycle.
    """
    client = new_llm_client(**llm_config)

    client.set_system_message(
        "You are a calculator assistant. When asked to compute something, "
        "use the add tool, then submit your structured response containing "
        "the 'message' (a short sentence with the result) and 'number' "
        "(the numeric result). Always use the tool first, then respond.",
    )

    handle = start_async_tool_loop(
        client,
        message="What is 3 + 4?",
        tools={"add": add},
        persist=True,
        response_format=Greeting,
        timeout=60,
    )

    # 1. Wait for the LLM to call the add tool
    await _wait_for_tool_request(client, "add")

    # 2. Wait for send_response to be called (appears after add completes)
    await _wait_for_tool_request(client, "send_response")

    # 3. Loop should still be alive
    await asyncio.sleep(0.3)
    assert (
        not handle.done()
    ), "Persist loop should NOT terminate after send_response with tools."

    # 4. Second turn: interject with another computation
    await handle.submit("Now what is 10 + 20?")

    # Wait for second add call
    async def _has_second_add() -> bool:
        count = sum(
            1
            for m in (client.messages or [])
            if m.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "add"
                for tc in (m.get("tool_calls") or [])
            )
        )
        return count >= 2

    await _wait_for_condition(_has_second_add, poll=0.05, timeout=30.0)

    # Wait for second send_response
    async def _has_second_send_response() -> bool:
        count = sum(
            1
            for m in (client.messages or [])
            if m.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "send_response"
                for tc in (m.get("tool_calls") or [])
            )
        )
        return count >= 2

    await _wait_for_condition(_has_second_send_response, poll=0.05, timeout=30.0)

    # 5. Still alive after second cycle
    await asyncio.sleep(0.2)
    assert (
        not handle.done()
    ), "Persist loop should survive multiple tool + send_response cycles."

    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


# --------------------------------------------------------------------------- #
#  NON-PERSIST: response_format uses final_response (not final_answer)        #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_non_persist_response_format_uses_final_response(llm_config):
    """In non-persist mode with ``response_format``, the loop should inject
    a tool named ``final_response`` (not the legacy ``final_answer``), and
    the loop should terminate normally when it is called.
    """
    client = new_llm_client(**llm_config)

    client.set_system_message(
        "When asked, respond with a JSON object containing exactly two keys: "
        "'message' (a greeting) and 'number' (an integer). Do not include any "
        "extra keys or commentary.",
    )

    handle = start_async_tool_loop(
        client,
        message="Say hello and pick a number.",
        tools={},
        persist=False,
        response_format=Greeting,
        timeout=60,
    )

    # The loop should terminate with a valid structured response.
    # result() returns a Pydantic model instance when response_format is set.
    result = await handle.result()
    assert isinstance(
        result,
        Greeting,
    ), f"Expected Greeting instance, got {type(result).__name__}: {result!r}"
    assert result.message.strip(), "Message must be non-empty"
    assert isinstance(result.number, int)

    # Verify the tool was called with the correct name
    assistant_tool_calls = [
        tc.get("function", {}).get("name")
        for m in (client.messages or [])
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    ]
    assert "final_response" in assistant_tool_calls, (
        "Non-persist mode with response_format should use 'final_response' tool, "
        f"but found: {assistant_tool_calls}"
    )
    assert (
        "final_answer" not in assistant_tool_calls
    ), "Legacy 'final_answer' name should not appear in tool calls."


# --------------------------------------------------------------------------- #
#  PERSIST MODE: handle.ask() should not break persist wait                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_survives_ask_mirror(llm_config):
    """Calling handle.ask() while in persist wait must not crash the loop.

    handle.ask() enqueues a _mirror sentinel to record the ask in the
    transcript. This sentinel wakes the persist wait, but because it
    carries no user message, the loop must not proceed to an LLM call
    with a trailing assistant message — models like Claude 4.6 Opus
    reject that with a BadRequestError ("does not support assistant
    message prefill").

    The loop should process the mirror for transcript fidelity, then
    return to the persist wait until a real interjection arrives.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={"echo": echo},
        persist=True,
        timeout=60,
    )

    # Wait for initial assistant response (loop enters persist wait after this)
    async def _has_assistant_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_assistant_response, poll=0.05, timeout=30.0)
    await asyncio.sleep(0.3)
    assert not handle.done(), "Loop should be in persist wait"

    # Dispatch an ask — this puts a _mirror sentinel into the queue
    ask_handle = await handle.ask("What tools do you have?")

    # Wait for the ask sub-loop to complete (it runs independently)
    ask_result = await ask_handle.result()
    assert ask_result, "Ask sub-loop should produce a response"

    # The outer loop must still be alive in persist wait — not crashed
    await asyncio.sleep(0.5)
    assert not handle.done(), (
        "Persist loop should survive handle.ask() without crashing. "
        "If this fails, the mirror sentinel likely triggered an invalid "
        "LLM call with a trailing assistant message."
    )

    # A real interjection should still work normally after the ask
    await handle.submit("Now use the echo tool with 'test'.")
    await _wait_for_tool_request(client, "echo")

    await asyncio.sleep(0.3)
    assert not handle.done(), "Loop should still be alive after real interjection"

    await handle.stop()
    result = await handle.result()
    assert result == "processed stopped early, no result"


@pytest.mark.asyncio
@_handle_project
async def test_persist_mode_transcript_note_appends_without_llm_turn(llm_config):
    """A ``_transcript_note`` sentinel leaves a loop-authored note in the
    transcript without granting an LLM turn.

    Background processes (e.g. a mid-session storage review) use this to
    inform the model of stored functions without waking it: the note must
    appear in ``client.messages`` while the loop stays in persist wait,
    and the assistant must produce no new message until a real
    interjection arrives.
    """
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="Say hello and nothing else.",
        tools={"echo": echo},
        persist=True,
        timeout=60,
    )

    async def _has_assistant_response() -> bool:
        return any(
            m.get("role") == "assistant" and m.get("content")
            for m in (client.messages or [])
        )

    await _wait_for_condition(_has_assistant_response, poll=0.05, timeout=30.0)
    await asyncio.sleep(0.3)
    assert not handle.done(), "Loop should be in persist wait"

    assistant_count_before = sum(
        1 for m in (client.messages or []) if m.get("role") == "assistant"
    )

    handle._queue.put_nowait(
        {"_transcript_note": {"text": "[background note] NOTE-MARKER-XYZ"}},
    )

    async def _note_appended() -> bool:
        return any(
            m.get("role") == "user" and "NOTE-MARKER-XYZ" in str(m.get("content", ""))
            for m in (client.messages or [])
        )

    await _wait_for_condition(_note_appended, poll=0.05, timeout=10.0)

    note_msg = next(
        m
        for m in client.messages
        if m.get("role") == "user" and "NOTE-MARKER-XYZ" in str(m.get("content", ""))
    )
    assert note_msg.get(
        "_loop_authored",
    ), "Note must be loop-authored, not a genuine user turn"

    # The note grants no LLM turn: the loop stays in persist wait with no
    # new assistant message. (A bounded quiet window is the only way to
    # assert an absence of activity.)
    await asyncio.sleep(1.0)
    assistant_count_after = sum(
        1 for m in (client.messages or []) if m.get("role") == "assistant"
    )
    assert (
        assistant_count_after == assistant_count_before
    ), "A transcript note must not wake the LLM"
    assert not handle.done(), "Loop should still be in persist wait"

    # A real interjection still works, and the model can now see the note.
    await handle.submit("Now use the echo tool with 'test'.")
    await _wait_for_tool_request(client, "echo")

    await handle.stop()
    await handle.result()


async def big_result() -> str:
    """Return a large blob of text."""
    return "DATA-" + ("y" * 2000)
