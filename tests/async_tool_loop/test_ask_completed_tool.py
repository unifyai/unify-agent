"""
tests/async_tool_loop/test_ask_completed_tool.py
=================================================

Tests for the ``ask_about_completed_tool`` dynamic dispatcher that lets the
outer LLM retrospectively query completed inner steerable tools about their
internal reasoning and intermediate steps.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from unify.common.async_tool_loop import (
    start_async_tool_loop,
    AsyncToolLoopHandle,
)
from unify.common.llm_client import new_llm_client
from tests.baked_defaults import as_shipped  # noqa: F401


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_llm_uses_ask_about_completed_tool_after_inner_completes(llm_config):
    """
    The outer LLM should call ``ask_about_completed_tool`` to learn details
    about a completed inner tool that are not visible in the outer transcript.

    Setup:
    - An inner tool ``researcher`` performs opaque multi-step work
      (``gather_sources`` -> ``synthesize``) and returns only a final summary.
    - The outer loop is instructed to first run ``researcher``, then use
      ``ask_about_completed_tool`` to find out which sources were gathered.
    - The outer transcript only shows ``researcher -> "summary: ocean currents"``;
      the source list is only visible inside the inner transcript.
    """

    ask_dispatched = {"count": 0}

    def gather_sources() -> str:
        """Gather research sources for the topic."""
        return "sources: [NOAA-2024, OceanographyJournal-vol12, MarineBio-dataset-7]"

    def synthesize(sources: str) -> str:
        """Synthesize gathered sources into a final summary.

        Parameters
        ----------
        sources : str
            The sources string returned by gather_sources.
        """
        return "summary: ocean currents are driven by wind, temperature, and salinity gradients"

    async def researcher() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are a researcher. Perform these steps in order:\n"
            "1. Call `gather_sources` to find relevant sources.\n"
            "2. Call `synthesize` passing the sources you received.\n"
            "3. Reply with **only** the summary returned by `synthesize` — nothing else.",
        )
        h = start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={
                "gather_sources": gather_sources,
                "synthesize": synthesize,
            },
            max_steps=10,
            timeout=120,
        )
        return h

    researcher.__name__ = "researcher"
    researcher.__qualname__ = "researcher"

    # Wrap researcher so we can track calls to ask_about_completed_tool
    original_researcher = researcher

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are an assistant. Follow these steps EXACTLY in order:\n"
        "1. Call `researcher` with no arguments and wait for it to complete.\n"
        "2. After researcher completes, you MUST call `ask_about_completed_tool` "
        "to ask: 'What specific sources were gathered?'\n"
        '   Use the tool_id from the "[askable ...]" tail message announcing '
        "researcher's completion.\n"
        "3. After receiving the answer, reply with the sources that were found. "
        "Include the specific source names in your final reply.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"researcher": original_researcher},
        max_steps=25,
        timeout=300,
    )

    final_result = await outer_handle.result()
    assert final_result is not None, "Outer loop should complete"

    # Verify the LLM actually called ask_about_completed_tool by inspecting
    # the outer transcript for a tool_call with that name.
    msgs = client.messages or []
    ask_calls = [
        tc
        for m in msgs
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") == "ask_about_completed_tool"
    ]

    assert len(ask_calls) >= 1, (
        "The outer LLM should have called ask_about_completed_tool at least once, "
        f"but found {len(ask_calls)} calls. Tool calls in transcript: "
        + str(
            [
                tc.get("function", {}).get("name")
                for m in msgs
                if m.get("role") == "assistant"
                for tc in (m.get("tool_calls") or [])
            ],
        )
    )

    # The final reply should contain at least one of the source names,
    # proving the LLM successfully retrieved inner-only details.
    assert any(
        src in final_result for src in ["NOAA-2024", "OceanographyJournal", "MarineBio"]
    ), (
        f"Final reply should mention at least one source from the inner tool, "
        f"but got: {final_result!r}"
    )


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_ask_about_completed_tool_with_multiple_completed_tools(llm_config):
    """
    When multiple inner steerable tools have completed, the dispatcher should
    list all of them and the LLM should be able to query a specific one.

    Setup:
    - Two inner tools: ``alpha_worker`` and ``beta_worker``, each performing
      opaque internal work with distinct results.
    - The outer loop runs both, then uses ``ask_about_completed_tool`` to
      query specifically about alpha_worker's internal steps.
    """

    def alpha_compute() -> str:
        """Perform alpha computation."""
        return "alpha_secret_value=42"

    async def alpha_worker() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are alpha worker. Call `alpha_compute`, then reply with ONLY the word 'alpha_done'.",
        )
        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"alpha_compute": alpha_compute},
            max_steps=10,
            timeout=120,
        )

    alpha_worker.__name__ = "alpha_worker"
    alpha_worker.__qualname__ = "alpha_worker"

    def beta_compute() -> str:
        """Perform beta computation."""
        return "beta_secret_value=99"

    async def beta_worker() -> AsyncToolLoopHandle:
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(
            "You are beta worker. Call `beta_compute`, then reply with ONLY the word 'beta_done'.",
        )
        return start_async_tool_loop(
            client=inner_client,
            message="start",
            tools={"beta_compute": beta_compute},
            max_steps=10,
            timeout=120,
        )

    beta_worker.__name__ = "beta_worker"
    beta_worker.__qualname__ = "beta_worker"

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are an assistant. Follow these steps EXACTLY in order:\n"
        "1. Call `alpha_worker` with no arguments. Wait for it to complete.\n"
        "2. Call `beta_worker` with no arguments. Wait for it to complete.\n"
        "3. After both complete, call `ask_about_completed_tool` to ask about "
        "the alpha_worker: 'What value did alpha_compute return?'\n"
        '   Use the tool_id for the alpha_worker from its "[askable ...]" tail message.\n'
        "4. Reply with the value that alpha_compute returned.",
    )

    outer_handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={
            "alpha_worker": alpha_worker,
            "beta_worker": beta_worker,
        },
        max_steps=30,
        timeout=360,
    )

    final_result = await outer_handle.result()
    assert final_result is not None, "Outer loop should complete"

    # Verify ask_about_completed_tool was called
    msgs = client.messages or []
    ask_calls = [
        tc
        for m in msgs
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") == "ask_about_completed_tool"
    ]

    assert (
        len(ask_calls) >= 1
    ), "The outer LLM should have called ask_about_completed_tool at least once"

    # The final reply should contain alpha's secret value
    assert (
        "42" in final_result
    ), f"Final reply should mention alpha_secret_value=42, but got: {final_result!r}"


# as_shipped: deleted in step 5 (steerable handles)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_ask_about_completed_tool_unused_without_completed_steerable_tools(
    llm_config,
):
    """
    ``ask_about_completed_tool`` is now always present in the schema — its
    docstring is frozen and no longer regenerated with a live listing (see
    the "[askable ...]" tail-message mechanism), so removing it from the
    schema when nothing has completed would reintroduce the per-turn schema
    churn steer()'s design exists to kill. What's verified here instead:
    with nothing to ask about, the model correctly never calls it (the
    frozen docstring makes clear there's nothing listed to reference).
    """

    def simple_add(x: int, y: int) -> int:
        """Add two numbers and return the result.

        Parameters
        ----------
        x : int
            First number.
        y : int
            Second number.
        """
        return x + y

    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are a calculator. Call `simple_add` with x=3, y=4, then reply with the result.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"simple_add": simple_add},
        max_steps=10,
        timeout=120,
    )

    result = await handle.result()
    assert result is not None, "Loop should complete"

    # Verify ask_about_completed_tool was never offered / called
    msgs = client.messages or []
    ask_calls = [
        tc
        for m in msgs
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
        if tc.get("function", {}).get("name") == "ask_about_completed_tool"
    ]

    assert len(ask_calls) == 0, (
        "ask_about_completed_tool should not be called when there are no "
        f"completed steerable tools, but found {len(ask_calls)} calls"
    )


# ---------------------------------------------------------------------------
# Symbolic: pop_task retains handle reference in completed metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_pop_task_retains_handle_in_completed_askable_tools():
    """When a steerable task completes, pop_task should retain the handle
    reference in ``_completed_askable_tools`` alongside the ask_fn.

    This enables downstream consumers (e.g. the actor's storage loop)
    to inspect the handle's lifecycle state without reaching into
    closure internals.
    """
    from unify.common._async_tool.tools_data import ToolsData
    from unify.common._async_tool.tools_utils import ToolCallMetadata

    sentinel_handle = MagicMock()
    sentinel_handle.done.return_value = False

    async def dummy_tool():
        return "ok"

    # Minimal ToolsData with a no-op logger and client.
    logger = MagicMock()
    logger.log_steps = False
    td = ToolsData({"dummy": dummy_tool}, client=MagicMock(), logger=logger)

    # Simulate scheduling and completing a steerable task.
    task = asyncio.create_task(asyncio.sleep(0))
    await task

    call_id = "test_call_123"
    metadata = ToolCallMetadata(
        name="dummy",
        call_id=call_id,
        call_dict={
            "id": call_id,
            "type": "function",
            "function": {"name": "dummy", "arguments": "{}"},
        },
        call_idx=0,
        chat_context=None,
        assistant_msg={"role": "assistant", "tool_calls": []},
        is_interjectable=False,
        handle=sentinel_handle,
        tool_schema={},
        llm_arguments={},
        raw_arguments_json="{}",
    )
    td.save_task(task, metadata)

    # Simulate the live ask closure being registered for this task
    # (DynamicToolFactory.live_ask_fns, seeded via _live_ask_fns_ref).
    ask_key = "ask_dummy_test_call_123"
    td._task_ask_keys[task] = ask_key
    td._live_ask_fns_ref = {ask_key: MagicMock()}

    # pop_task should retain the handle in _completed_askable_tools.
    td.pop_task(task)

    assert (
        call_id in td._completed_askable_tools
    ), "pop_task should retain metadata for the completed steerable tool"
    entry = td._completed_askable_tools[call_id]
    assert (
        entry["handle"] is sentinel_handle
    ), "Retained metadata should include the original handle reference"
