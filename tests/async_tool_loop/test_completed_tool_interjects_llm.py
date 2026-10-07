from __future__ import annotations

import asyncio
import pytest

from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    make_gated_async_tool,
    _wait_for_tool_result,
    _is_synthetic_check_status_stub,
    _is_synthetic_check_status_tool_msg,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (interjections)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]

# ────────────────────────────────────────────────────────────────────────────
# Wait log helper – `wait` calls are pruned from transcript, so we watch logs
# ────────────────────────────────────────────────────────────────────────────

_WAIT_LOG_MESSAGE = "Assistant chose `wait` – no-op; not persisting to transcript."


# ────────────────────────────────────────────────────────────────────────────
# Dummy tools – one finishes almost instantly, the other a little later
# ────────────────────────────────────────────────────────────────────────────


async def fast_task() -> str:
    """Return very quickly."""
    await asyncio.sleep(0.05)
    return "FAST_RESULT"


async def slow_task() -> str:
    """Return after the fast task but (usually) before the LLM finishes thinking."""
    await asyncio.sleep(0.15)
    return "SLOW_RESULT"


async def very_slow_task() -> str:
    """
    Take long enough that the LLM has time to finish a thought
    after the fast task is done.
    """
    await asyncio.sleep(5.0)
    return "VERY_SLOW_RESULT"


# ────────────────────────────────────────────────────────────────────────────
# Test
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_wait_called_and_pruned_when_other_tool_is_very_slow(
    llm_config,
    monkeypatch,
) -> None:
    """
    When two tools are requested in one turn and only the fast one completes,
    the model will choose the `wait` helper to no-op until the very slow tool
    finishes. We assert that:

    - `wait` was indeed called (via log capture), and
    - `wait` does not appear in the final transcript (pruned from messages).

    This test uses a gated tool for very_slow_task to ensure deterministic timing:
    1. fast_task completes
    2. LLM gets a chance to respond (should call `wait`)
    3. Only then is very_slow_task's gate released
    4. LLM produces final answer
    """

    system_prompt = (
        "You have two tools: 'fast_task' and 'very_slow_task'. "
        "When asked to run them, always call BOTH tools in a single assistant turn.\n\n"
        "CRITICAL RULE: After calling tools, if any tool is still pending (shows '_placeholder': 'pending'), "
        "you MUST call the `wait` tool and produce NO other output. Do not explain, do not ask questions, "
        "do not produce any text - just call `wait`. Only after ALL tool results are available "
        "(no pending placeholders remain) should you produce your final text response summarizing the results."
    )

    client = new_llm_client(
        **llm_config,
        system_message=system_prompt,
    )

    # Create a gated tool for very_slow_task to control timing deterministically
    very_slow_gate, gated_very_slow_task = make_gated_async_tool(
        return_value="VERY_SLOW_RESULT",
    )

    tools = {"fast_task": fast_task, "very_slow_task": gated_very_slow_task}

    import unify.common._async_tool.loop as _loop_mod

    wait_logged = asyncio.Event()
    original_info = _loop_mod.LoopLogger.info

    def _capture_wait_log(self, msg, prefix=""):
        if _WAIT_LOG_MESSAGE in msg:
            wait_logged.set()
        return original_info(self, msg, prefix)

    monkeypatch.setattr(_loop_mod.LoopLogger, "info", _capture_wait_log)

    handle = start_async_tool_loop(
        client,
        message="Please run fast_task and very_slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    # Wait for fast_task result to be processed.
    await _wait_for_tool_result(client, "fast_task", min_results=1)

    # `wait` is deliberately pruned from the transcript, so observe the loop's
    # logger method directly instead of depending on pytest logging handlers.
    await asyncio.wait_for(wait_logged.wait(), timeout=120.0)

    # Now release the gate so very_slow_task can complete.
    very_slow_gate.set()

    # Wait for the loop to complete.
    await handle.result()

    # ── Assertions ───────────────────────────────────────────────────────

    # The loop may insert a small assistant→tool status pair (check_status_*) to
    # preserve ordering when earlier placeholders are no longer at the tail.
    is_status_assistant = lambda m: (
        m.get("role") == "assistant"
        and bool(m.get("tool_calls"))
        and any(
            tc.get("function", {}).get("name", "").startswith("check_status_")
            for tc in m["tool_calls"]
        )
    )
    is_status_tool = lambda m: (
        m.get("role") == "tool" and str(m.get("name", "")).startswith("check_status_")
    )

    non_stub_assistants = [
        m
        for m in client.messages
        if m.get("role") == "assistant" and not is_status_assistant(m)
    ]
    non_stub_tools = [
        m for m in client.messages if m.get("role") == "tool" and not is_status_tool(m)
    ]

    # 1) Assert that `wait` was called (via loop logger)
    assert wait_logged.is_set(), (
        "Expected LLM to call `wait` while very_slow_task was pending, but no "
        "`wait` log was found. This may indicate the LLM did not follow the prompt "
        "instructions to call `wait` when partial results are available."
    )

    # 2) Assert that `wait` is not persisted in the transcript
    #    - no assistant tool_call with function name 'wait'
    assert all(
        all(
            tc.get("function", {}).get("name") != "wait"
            for tc in (m.get("tool_calls") or [])
        )
        for m in client.messages
        if m.get("role") == "assistant"
    )
    #    - no tool message named 'wait'
    assert all(
        m.get("name") != "wait" for m in client.messages if m.get("role") == "tool"
    )

    # Basic health checks (non‑strict): initial + final assistants and at least two tools
    assert len(non_stub_assistants) >= 2
    assert len(non_stub_tools) >= 2

    # Tool names include fast & very_slow; placeholder duplicates don't hurt
    tool_names = {m["name"] for m in client.messages if m["role"] == "tool"}
    assert {"fast_task", "very_slow_task"}.issubset(tool_names)

    # Initial assistant turn requested BOTH tools – search robustly (index can vary)
    assistant_tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "assistant"
        and m.get("tool_calls")
        and not is_status_assistant(m)
    ]
    assert any(
        {"fast_task", "very_slow_task"}.issubset(
            {tc.get("function", {}).get("name") for tc in (m.get("tool_calls") or [])},
        )
        for m in assistant_tool_msgs
    )


@pytest.mark.asyncio
@_handle_project
async def test_late_tool_completion_spares_llm_step_when_preemption_disabled(
    llm_config,
    monkeypatch,
) -> None:
    """
    With ``interrupt_llm_on_tool_completion=False`` the same late completion that
    pre-empts the reasoning step in the test below must leave it alone, because
    the provider has already been billed for it.

    Asserts the two halves of that contract:

    - no in-flight LLM step is ever cancelled, and
    - both tool results still reach the transcript, so the model answers with
      the late result rather than without it.
    """
    import unify.common._async_tool.loop as _loop_mod

    _real_generate = _loop_mod.generate_with_preprocess
    stats = {"started": 0, "cancelled": 0}

    async def _counting_generate(*args, **kwargs):
        stats["started"] += 1
        try:
            return await _real_generate(*args, **kwargs)
        except asyncio.CancelledError:
            stats["cancelled"] += 1
            raise

    monkeypatch.setattr(_loop_mod, "generate_with_preprocess", _counting_generate)

    system_prompt = (
        "You have access to two tools called 'fast_task' and 'slow_task'. "
        "Always invoke *both* tools in the same assistant turn and wait for "
        "their results before replying to the user. Do not send any other "
        "assistant messages in between."
    )

    client = new_llm_client(
        **llm_config,
        system_message=system_prompt,
    )

    handle = start_async_tool_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools={"fast_task": fast_task, "slow_task": slow_task},
        interrupt_llm_with_interjections=True,
        interrupt_llm_on_tool_completion=False,
    )
    assert handle._loop_config["interrupt_llm_on_tool_completion"] is False

    await handle.result()

    assert stats["started"] >= 2, "expected at least an initial and a final LLM step"
    assert stats["cancelled"] == 0, (
        f"a late tool completion cancelled {stats['cancelled']} in-flight LLM "
        "step(s) despite interrupt_llm_on_tool_completion=False; those steps are "
        "billed by the provider and discarded"
    )

    non_stub_tools = [
        m
        for m in client.messages
        if m.get("role") == "tool" and not _is_synthetic_check_status_tool_msg(m)
    ]
    tool_names = {m["name"] for m in non_stub_tools}
    assert {"fast_task", "slow_task"}.issubset(tool_names), (
        "the late tool result never reached the transcript, so the model "
        f"concluded without it (saw {sorted(tool_names)})"
    )


@pytest.mark.asyncio
@_handle_project
async def test_llm_step_is_preempted_by_late_tool_completion(llm_config) -> None:
    """
    The model is instructed to call both tools in a single assistant turn. The fast
    task completes first, then the slow task completes while the model may still be
    thinking. The loop should pre-empt in-flight reasoning, deliver the late tool
    result, and run the model again to produce the final answer.

    Expected role shapes (excluding any synthetic check_status_* status stubs):

        0 user
        1 assistant (tool_calls fast & slow)
        2 tool  (fast_task result)
        3 tool  (slow_task result)
        4 assistant (final answer)

    The test asserts two assistant turns (initial + final) and two tool messages.
    """

    system_prompt = (
        "You have access to two tools called 'fast_task' and 'slow_task'. "
        "Always invoke *both* tools in the same assistant turn and wait for "
        "their results before replying to the user. Do not send any other "
        "assistant messages in between."
    )

    client = new_llm_client(
        **llm_config,
        system_message=system_prompt,
    )

    tools = {"fast_task": fast_task, "slow_task": slow_task}

    handle = start_async_tool_loop(
        client,
        message="Please run fast_task and slow_task, triggering them both **immediately** (at the same time)",
        tools=tools,
        interrupt_llm_with_interjections=True,
    )

    await handle.result()

    # ── Assertions ───────────────────────────────────────────────────────
    # Some real clients persist the system header as a first message.
    # Ignore any leading system messages when asserting the core skeleton.
    roles = [m["role"] for m in client.messages if m.get("role") != "system"]

    # Basic skeleton (excluding any system headers):
    #   user
    #   assistant(tool_calls fast & slow)
    #   tool  (fast_task result)
    #   assistant starts replying
    #   tool  (slow_task result)
    #   earlier assistant call is stopped
    #   assistant injests both results (final)
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    # Exclude status stubs (check_status_*) from strict counts using shared helpers
    non_stub_assistants = [
        m
        for m in client.messages
        if m.get("role") == "assistant" and not _is_synthetic_check_status_stub(m)
    ]
    non_stub_tools = [
        m
        for m in client.messages
        if m.get("role") == "tool" and not _is_synthetic_check_status_tool_msg(m)
    ]
    assert len(non_stub_assistants) == 2  # initial + final
    assert len(non_stub_tools) == 2  # fast + slow

    # The two tool results must correspond to the two tool names (excluding synthetic ones)
    tool_names = {m["name"] for m in non_stub_tools}
    assert {"fast_task", "slow_task"}.issubset(tool_names)

    # Initial assistant turn must have requested *both* tools – search robustly
    assistant_tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "assistant"
        and m.get("tool_calls")
        and not _is_synthetic_check_status_stub(m)
    ]
    assert any(
        {"fast_task", "slow_task"}.issubset(
            {tc.get("function", {}).get("name") for tc in (m.get("tool_calls") or [])},
        )
        for m in assistant_tool_msgs
    )
