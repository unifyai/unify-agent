from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from tests.async_helpers import _wait_for_tool_result
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common._async_tool.time_context import TimeContext

# Module-level marker: all tests in this file are eval tests
pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (Neutral Names - No Speed Hints)                      #
# --------------------------------------------------------------------------- #


async def tool_alpha() -> str:
    """Performs operation alpha."""
    return "alpha_complete"


async def get_data() -> str:
    """Retrieves some data."""
    return "data retrieved successfully"


# --------------------------------------------------------------------------- #
#  HELPER: Count tool calls by name                                           #
# --------------------------------------------------------------------------- #


def count_tool_calls(messages: list, tool_name: str) -> int:
    """Count how many times a tool was called in the message history."""
    count = 0
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") == tool_name:
                    count += 1
    return count


@pytest.mark.asyncio
@_handle_project
async def test_conversation_start_awareness(llm_config, monkeypatch):
    """Verify the LLM reports conversation elapsed time from inline annotations.

    ``current_offset`` is patched to return ``"+30s"`` so the user message
    is annotated with ``[elapsed: +30s]``.  The LLM should read the
    annotation and report ~30 seconds.
    """
    monkeypatch.setattr(TimeContext, "current_offset", lambda self: "+30s")

    class ElapsedTimeResponse(BaseModel):
        elapsed_seconds: float = Field(
            ...,
            description="Number of seconds elapsed since the conversation started.",
        )

    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message="How long ago did this conversation start, in seconds?",
        tools={},
        response_format=ElapsedTimeResponse,
        time_awareness=True,
    ).result()

    assert isinstance(answer, ElapsedTimeResponse)
    assert (
        answer.elapsed_seconds == 30.0
    ), f"Expected 30s elapsed, got {answer.elapsed_seconds}"


@pytest.mark.asyncio
@_handle_project
async def test_tool_duration_awareness(llm_config, monkeypatch):
    """Verify the LLM can report a tool's execution duration from the
    inline metadata envelope on the tool result.

    ``duration_since`` is patched to return ``"2s"`` so the metadata
    envelope always shows a 2-second duration regardless of real wall time.
    """
    monkeypatch.setattr(TimeContext, "duration_since", lambda self, t: "2s")

    class ToolDurationResponse(BaseModel):
        tool_name: str = Field(
            ...,
            description="The name of the tool that was called.",
        )
        duration_seconds: float = Field(
            ...,
            description="How long the tool took to execute in seconds.",
        )

    client = new_llm_client(**llm_config)

    answer = await start_async_tool_loop(
        client,
        message=(
            "Call get_data to retrieve some data, then tell me "
            "how long that call took to execute."
        ),
        tools={"get_data": get_data},
        response_format=ToolDurationResponse,
        time_awareness=True,
    ).result()

    call_count = count_tool_calls(client.messages, "get_data")
    assert call_count >= 1, "get_data should have been called"

    assert isinstance(answer, ToolDurationResponse)
    assert "get_data" in answer.tool_name
    assert (
        answer.duration_seconds == 2.0
    ), f"Expected 2s duration, got {answer.duration_seconds}"


@pytest.mark.asyncio
@_handle_project
async def test_faster_tool_identification(llm_config, monkeypatch):
    """Verify the LLM can identify which tool was faster and re-call it.

    Uses neutral tool names so the LLM must infer speed solely from the
    duration annotations.  ``tool_alpha`` gets ``"5s"`` and ``tool_beta``
    gets ``"500ms"`` via an iterator-based ``duration_since`` patch.
    A gate on ``tool_beta`` guarantees ``tool_alpha`` completes first so
    the iterator yields in the correct order.
    """
    durations = iter(["5s", "500ms"])
    monkeypatch.setattr(TimeContext, "offset_at", lambda self, t: "+0s")
    monkeypatch.setattr(
        TimeContext,
        "duration_since",
        lambda self, t: next(durations, "500ms"),
    )

    beta_gate = asyncio.Event()

    async def gated_tool_beta() -> str:
        """Performs operation beta."""
        await asyncio.wait_for(beta_gate.wait(), timeout=300)
        return "beta_complete"

    class FasterToolResponse(BaseModel):
        faster_tool: str = Field(
            ...,
            description="The name of the tool with the shorter execution time.",
        )
        faster_duration_seconds: float = Field(
            ...,
            description="Execution duration of the faster tool in seconds.",
        )
        slower_tool: str = Field(
            ...,
            description="The name of the tool with the longer execution time.",
        )
        slower_duration_seconds: float = Field(
            ...,
            description="Execution duration of the slower tool in seconds.",
        )

    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message=(
            "Call tool_alpha first, then call tool_beta. "
            "After both complete, determine which tool was faster "
            "and call that faster tool again. "
            "Finally, report which tool was faster and which was slower, "
            "along with their durations."
        ),
        tools={"tool_alpha": tool_alpha, "tool_beta": gated_tool_beta},
        response_format=FasterToolResponse,
        time_awareness=True,
    )

    await _wait_for_tool_result(client, "tool_alpha")
    beta_gate.set()

    answer = await handle.result()

    alpha_count = count_tool_calls(client.messages, "tool_alpha")
    beta_count = count_tool_calls(client.messages, "tool_beta")

    # Both should be called at least once initially
    assert alpha_count >= 1, "tool_alpha should have been called at least once"
    assert beta_count >= 1, "tool_beta should have been called at least once"

    assert beta_count >= 2, (
        f"tool_beta should have been called twice (it's faster), "
        f"but was called {beta_count} times"
    )

    assert isinstance(answer, FasterToolResponse)
    assert answer.faster_tool == "tool_beta"
    assert answer.slower_tool == "tool_alpha"
    assert (
        answer.faster_duration_seconds == 0.5
    ), f"Expected 0.5s for beta, got {answer.faster_duration_seconds}"
    assert (
        answer.slower_duration_seconds == 5.0
    ), f"Expected 5s for alpha, got {answer.slower_duration_seconds}"
