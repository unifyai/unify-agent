from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone

import pytest

from tests.helpers import _handle_project
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.llm_client import new_llm_client
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common._async_tool.time_context import (
    format_offset,
    format_duration,
    TimeContext,
)

# --------------------------------------------------------------------------- #
#  SIMULATED TIME FIXTURE                                                     #
# --------------------------------------------------------------------------- #

# Fixed base datetime for simulated time
_SIMULATED_BASE = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def simulated_time(monkeypatch):
    """Patch perf_counter() for deterministic time.

    Each call to perf_counter() increments by 1 second so offsets and
    durations are reproducible across test runs.
    """
    state = {
        "perf_call_count": 0,
        "increment_seconds": 1,
    }

    def _simulated_perf_counter() -> float:
        value = state["perf_call_count"] * state["increment_seconds"]
        state["perf_call_count"] += 1
        return float(value)

    monkeypatch.setattr(
        "unify.common._async_tool.time_context.perf_counter",
        _simulated_perf_counter,
    )

    return state


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS                                                       #
# --------------------------------------------------------------------------- #


async def simple_tool() -> str:
    """A simple tool that returns immediately."""
    return "done"


async def timed_tool(duration: float = 0.1) -> str:
    """A tool that sleeps for a specified duration."""
    await asyncio.sleep(duration)
    return f"completed after {duration}s"


# --------------------------------------------------------------------------- #
#  UNIT TESTS – format_offset / format_duration                               #
# --------------------------------------------------------------------------- #


class TestFormatOffset:
    def test_zero(self):
        assert format_offset(0) == "+0s"

    def test_negative(self):
        assert format_offset(-5) == "+0s"

    def test_seconds_only(self):
        assert format_offset(45) == "+45s"

    def test_minutes_and_seconds(self):
        assert format_offset(192) == "+3m12s"

    def test_hours_minutes_seconds(self):
        assert format_offset(3750) == "+1h2m30s"

    def test_exact_minute(self):
        assert format_offset(60) == "+1m"

    def test_exact_hour(self):
        assert format_offset(3600) == "+1h"


class TestFormatDuration:
    def test_zero(self):
        assert format_duration(0) == "0s"

    def test_negative(self):
        assert format_duration(-1) == "0s"

    def test_milliseconds_only(self):
        assert format_duration(0.1) == "100ms"

    def test_seconds_and_ms(self):
        assert format_duration(2.045) == "2s45ms"

    def test_seconds_only(self):
        assert format_duration(5.0) == "5s"

    def test_minutes_and_seconds(self):
        assert format_duration(90) == "1m30s"

    def test_hours_suppress_ms(self):
        assert format_duration(5400.5) == "1h30m"


# --------------------------------------------------------------------------- #
#  UNIT TESTS – TimeContext methods                                           #
# --------------------------------------------------------------------------- #


class TestTimeContextWrapResult:
    def test_wraps_string_content(self):
        ctx = TimeContext(perf_counter_start=0.0)
        result = ctx.wrap_result('"hello"', scheduled_time=10.0)
        parsed = json.loads(result)
        assert parsed["tool_result"] == '"hello"'
        assert "called_at" in parsed["metadata"]
        assert "duration" in parsed["metadata"]

    def test_wraps_list_content_with_metadata_block(self):
        ctx = TimeContext(perf_counter_start=0.0)
        blocks = [{"type": "text", "text": "payload"}]
        result = ctx.wrap_result(blocks, scheduled_time=5.0)
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        meta = json.loads(result[0]["text"])
        assert "metadata" in meta
        assert result[1] == blocks[0]

    def test_called_at_uses_scheduled_time(self):
        ctx = TimeContext(perf_counter_start=100.0)
        result = ctx.wrap_result('"x"', scheduled_time=192.0)
        parsed = json.loads(result)
        assert parsed["metadata"]["called_at"] == "+1m32s"


class TestTimeContextPrefixUserMessage:
    def test_prefix_format(self):
        ctx = TimeContext(perf_counter_start=0.0)
        prefixed = ctx.prefix_user_message("hello")
        assert prefixed.startswith("[elapsed: +")
        assert prefixed.endswith("] hello")


class TestBuildExplanationPrompt:
    def test_contains_key_sections(self):
        prompt = TimeContext.build_explanation_prompt()
        assert "## Time Annotations" in prompt
        assert "tool_result" in prompt
        assert "called_at" in prompt
        assert "[elapsed:" in prompt
        assert "meta:started" in prompt


# --------------------------------------------------------------------------- #
#  HELPERS                                                                    #
# --------------------------------------------------------------------------- #


def _find_explanation_msg(messages: list) -> dict | None:
    for msg in messages:
        if msg.get("role") == "system" and msg.get("_time_explanation"):
            return msg
    return None


def _find_tool_results(messages: list) -> list[dict]:
    """Return all tool messages whose content contains 'tool_result'."""
    results = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "tool_result" in parsed:
                results.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def _find_user_messages(messages: list) -> list[dict]:
    return [m for m in messages if m.get("role") == "user" and not m.get("_ctx_header")]


_ELAPSED_RE = re.compile(r"^\[elapsed: \+\d+[hms\d]*\] ")

_SIMPLE_MESSAGE = "Call simple_tool and reply with 'ok'."


async def _run_loop(llm_config, message: str, tools: dict, calls: list):
    """Run a time-aware loop whose scripted model makes *calls* (one per
    turn, ``(name, args)``), then replies 'ok'. Returns the client and the
    model."""
    model = ScriptedModel(
        actor=[*(reply(calls=[c]) for c in calls), reply("ok")],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)
        handle = start_async_tool_loop(
            client,
            message=message,
            tools=tools,
            time_awareness=True,
        )
        assert await asyncio.wait_for(handle.result(), 60) == "ok"
    assert model.kinds() == ["actor"] * (len(calls) + 1)
    return client, model


# --------------------------------------------------------------------------- #
#  INTEGRATION TESTS (scripted model; the annotations are the loop's own)     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_explanation_prompt_injected(llm_config):
    """Static explanation system message is present when time_awareness=True."""
    client, model = await _run_loop(
        llm_config,
        _SIMPLE_MESSAGE,
        {"simple_tool": simple_tool},
        [("simple_tool", {})],
    )

    explanation = _find_explanation_msg(client.messages)
    assert explanation is not None, "Time explanation system message not found"
    assert "## Time Annotations" in explanation["content"]
    # The model receives it.
    assert any(
        "## Time Annotations" in str(m.get("content")) for m in model.calls[0].messages
    )
    assert model.kinds() == ["actor", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_tool_result_wrapped_with_metadata(llm_config):
    """Completed base-tool results contain tool_result + metadata envelope."""
    client, model = await _run_loop(
        llm_config,
        _SIMPLE_MESSAGE,
        {"simple_tool": simple_tool},
        [("simple_tool", {})],
    )

    wrapped = _find_tool_results(client.messages)
    assert len(wrapped) == 1, f"Expected 1 wrapped tool result, got {wrapped}"

    for item in wrapped:
        assert "tool_result" in item
        assert "done" in item["tool_result"]
        meta = item["metadata"]
        assert "called_at" in meta
        assert "duration" in meta
        assert meta["called_at"].startswith("+")
    # The model receives the wrapped result.
    assert any(
        m["role"] == "tool"
        and "tool_result" in str(m.get("content"))
        and "called_at" in str(m.get("content"))
        for m in model.calls[1].messages
    )
    assert model.kinds() == ["actor", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_user_message_prefixed_with_elapsed(llm_config):
    """Initial user message should have [elapsed: +XmYs] prefix."""
    client, model = await _run_loop(
        llm_config,
        _SIMPLE_MESSAGE,
        {"simple_tool": simple_tool},
        [("simple_tool", {})],
    )

    user_msgs = _find_user_messages(client.messages)
    assert len(user_msgs) >= 1, "No user messages found"

    first_user = user_msgs[0]
    assert _ELAPSED_RE.match(
        first_user["content"],
    ), f"User message not prefixed with elapsed: {first_user['content']!r}"
    assert first_user["content"].endswith(_SIMPLE_MESSAGE)
    assert model.kinds() == ["actor", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_simulated_time_deterministic_offsets(llm_config, simulated_time):
    """With simulated perf_counter, offsets are deterministic."""
    client, model = await _run_loop(
        llm_config,
        _SIMPLE_MESSAGE,
        {"simple_tool": simple_tool},
        [("simple_tool", {})],
    )

    assert simulated_time["perf_call_count"] > 0, "perf_counter() was not invoked"

    wrapped = _find_tool_results(client.messages)
    assert len(wrapped) == 1

    for item in wrapped:
        assert re.fullmatch(r"\+\d+[hms\d]*", item["metadata"]["called_at"])
        assert any(
            c in item["metadata"]["duration"] for c in ("s", "ms")
        ), f"Duration missing time unit: {item['metadata']['duration']}"
    assert model.kinds() == ["actor", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_multiple_tools_each_wrapped(llm_config):
    """Each tool invocation gets its own metadata envelope."""
    client, model = await _run_loop(
        llm_config,
        (
            "Call simple_tool first, then call timed_tool with duration=0.05. "
            "Reply with 'ok' after both complete."
        ),
        {"simple_tool": simple_tool, "timed_tool": timed_tool},
        [("simple_tool", {}), ("timed_tool", {"duration": 0.05})],
    )

    wrapped = _find_tool_results(client.messages)
    assert len(wrapped) == 2, f"Expected 2 wrapped results, got {len(wrapped)}"
    assert "done" in wrapped[0]["tool_result"]
    assert "completed after 0.05s" in wrapped[1]["tool_result"]
    for item in wrapped:
        assert item["metadata"]["called_at"].startswith("+")
        assert item["metadata"]["duration"]
    assert model.kinds() == ["actor", "actor", "actor"]
