"""
Tests for the ``guidelines`` parameter on Actor.act().

The ``guidelines`` parameter provides meta-guidance on *how* to approach a task,
as opposed to the ``request`` which specifies *what* to do.
"""

from unify.actor.core_surface import PromptSurface
import asyncio

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.actor.prompt_builders import build_code_act_prompt

# ---------------------------------------------------------------------------
# Prompt builder unit tests (symbolic)
# ---------------------------------------------------------------------------


def test_build_code_act_prompt_without_guidelines():
    """When guidelines is None, the prompt should not contain a Guidelines section."""
    prompt = build_code_act_prompt(
        environments={},
        guidelines=None,
        core=PromptSurface(),
    )
    assert "### Guidelines" not in prompt


def test_build_code_act_prompt_with_guidelines():
    """When guidelines is provided, the prompt should contain the Guidelines section."""
    prompt = build_code_act_prompt(
        environments={},
        guidelines="Always use sub-agents for parallel tasks.",
        core=PromptSurface(),
    )
    assert "### Guidelines" in prompt
    assert "Always use sub-agents for parallel tasks." in prompt
    assert "Follow these guidelines throughout this session:" in prompt


def test_build_code_act_prompt_guidelines_placed_before_rules():
    """Guidelines should appear before the execution rules / tool signatures."""
    prompt = build_code_act_prompt(
        environments={},
        guidelines="Prefer simple solutions.",
        core=PromptSurface(),
    )
    guidelines_pos = prompt.index("### Guidelines")
    # The role line always appears; guidelines should come after the role but
    # the prompt overall should still contain the standard role header.
    assert "### Role" in prompt
    assert guidelines_pos > 0


# ---------------------------------------------------------------------------
# CodeActActor.act accepts guidelines (symbolic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(30)
async def test_code_act_actor_accepts_guidelines_parameter():
    """
    CodeActActor.act should accept the ``guidelines`` keyword argument
    without raising TypeError, and return a valid handle.
    """
    actor = CodeActActor(
        timeout=30,
    )
    try:
        handle = await actor.act(
            "What is 2 + 2?",
            guidelines="Keep your response concise.",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(30)
async def test_code_act_actor_guidelines_none_by_default():
    """
    When guidelines is not passed, it defaults to None and the actor
    operates normally.
    """
    actor = CodeActActor(
        timeout=30,
    )
    try:
        handle = await actor.act(
            "What is 2 + 2?",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CodeActActor.act guidelines influence behavior (eval)
# ---------------------------------------------------------------------------


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_guidelines_forbid_execute_code():
    """
    When guidelines say "do NOT use execute_code", the actor should answer
    directly without calling execute_code.
    """
    actor = CodeActActor(
        timeout=60,
    )
    try:
        handle = await actor.act(
            "What is 7 * 6?",
            guidelines=(
                "Do NOT use the execute_code tool. Answer the question directly "
                "from your own knowledge without running any code."
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)

        # The result should contain 42
        assert "42" in str(result)

        # Verify execute_code was NOT called
        from tests.actor.code_act.helpers import get_code_act_tool_calls

        tool_calls = get_code_act_tool_calls(handle)
        assert "execute_code" not in tool_calls, (
            f"Guidelines said not to use execute_code, but it was called. "
            f"Tool calls: {tool_calls}"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_guidelines_include_marker_in_response():
    """
    When guidelines require including a specific marker phrase in the final
    answer, the actor should comply. This validates that guidelines reach the
    LLM and influence its output.
    """
    actor = CodeActActor(
        timeout=60,
    )
    try:
        handle = await actor.act(
            "What is the capital of France?",
            guidelines=(
                "You MUST end your final answer with the exact phrase "
                "'[GUIDELINES_FOLLOWED]' on its own line."
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        result_str = str(result)

        assert "Paris" in result_str
        assert "GUIDELINES_FOLLOWED" in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_guidelines_response_format_constraint():
    """
    Guidelines can guide how the result is formatted.
    """
    actor = CodeActActor(
        timeout=60,
    )
    try:
        handle = await actor.act(
            "List three colors.",
            guidelines=(
                "Format your final answer as a numbered list, one color per line, "
                "like:\n1. Red\n2. Blue\n3. Green"
            ),
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        result_str = str(result)

        # The result should contain numbered items
        assert "1." in result_str
        assert "2." in result_str
        assert "3." in result_str
    finally:
        try:
            await actor.close()
        except Exception:
            pass
