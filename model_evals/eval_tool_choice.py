"""Evals of a live model (moved from tests/async_tool_loop/test_tool_loop_limits.py): each needs a real provider key and is run by hand
(model_evals/README.md). The test suite never collects or runs them."""


import asyncio
import pytest
import unillm
from typing import Dict, Callable
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.tool_spec import ToolSpec
from unify.common.llm_client import new_llm_client


@pytest.mark.asyncio
@pytest.mark.llm_call
async def test_policy_forces_single_tool_invocation(llm_config):
    """A custom ``tool_policy`` can replicate the old
    ``minimum_tool_turns=1`` semantics by forcing a *required* tool call on the
    first turn only.

    Live: the check is that the provider enforces ``tool_choice="required"``
    against a contrary instruction. That the policy's choice reaches the
    request is tested with a scripted model by the other policy tests."""

    flag = {"called": False}

    async def dummy_tool():
        flag["called"] = True
        return "ok"

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client,
        message="You are part of a test. Do *not* call any tools, just return to the user immediately",
        tools={"dummy_tool": dummy_tool},
        tool_policy=lambda i, tls: ("required", tls) if i < 1 else ("auto", tls),
    )
    await handle.result()

    # The loop had to wait for the tool to finish and therefore should return
    # the *final* assistant content.
    assert flag["called"] is True
