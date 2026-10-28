"""
Tests for the ``guidelines`` parameter on Actor.act().

The ``guidelines`` parameter provides meta-guidance on *how* to approach a task,
as opposed to the ``request`` which specifies *what* to do.
"""

import asyncio

import pytest

from tests.scripted_model import ScriptedModel, reply, scripted
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
    )
    assert "### Guidelines" not in prompt


def test_build_code_act_prompt_with_guidelines():
    """When guidelines is provided, the prompt should contain the Guidelines section."""
    prompt = build_code_act_prompt(
        environments={},
        guidelines="Always use sub-agents for parallel tasks.",
    )
    assert "### Guidelines" in prompt
    assert "Always use sub-agents for parallel tasks." in prompt
    assert "Follow these guidelines throughout this session:" in prompt


def test_build_code_act_prompt_guidelines_placed_before_rules():
    """Guidelines should appear before the execution rules / tool signatures."""
    prompt = build_code_act_prompt(
        environments={},
        guidelines="Prefer simple solutions.",
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
@pytest.mark.timeout(30)
async def test_code_act_actor_accepts_guidelines_parameter():
    """
    CodeActActor.act should accept the ``guidelines`` keyword argument
    without raising TypeError, and return a valid handle.
    """
    actor = CodeActActor(
        timeout=30,
    )
    # The handle is stopped at once; a request that starts before the stop
    # is answered in process.
    with scripted(ScriptedModel(allow={"actor"})):
        handle = await actor.act(
            "What is 2 + 2?",
            guidelines="Keep your response concise.",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_code_act_actor_guidelines_none_by_default():
    """
    When guidelines is not passed, it defaults to None and the actor
    operates normally.
    """
    actor = CodeActActor(
        timeout=30,
    )
    # The handle is stopped at once; a request that starts before the stop
    # is answered in process.
    with scripted(ScriptedModel(allow={"actor"})):
        handle = await actor.act(
            "What is 2 + 2?",
            persist=False,
            clarification_enabled=False,
        )
        assert handle is not None
        await handle.stop()


# ---------------------------------------------------------------------------
# CodeActActor.act guidelines reach the model (scripted)
# ---------------------------------------------------------------------------
#
# The model is scripted (tests/scripted_model.py) to follow the guidelines;
# each test checks that the guidelines reached the model in the system
# prompt's Guidelines section and what the actor made of the reply that
# follows them. Whether a live model obeys them is the model's behaviour,
# not the harness's.


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


async def _act_with_guidelines(request: str, guidelines: str, answer: str):
    """Act once with *guidelines*, the scripted model answering *answer*
    directly. Returns the handle, the result and the model's one request."""
    model = ScriptedModel(actor=[reply(answer)])
    actor = CodeActActor(
        timeout=60,
    )
    with scripted(model):
        handle = await actor.act(
            request,
            guidelines=guidelines,
            persist=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)

    (call,) = model.of("actor")
    system = "\n".join(
        _text(m.get("content")) for m in call.messages if m.get("role") == "system"
    )
    assert "### Guidelines" in system
    assert "Follow these guidelines throughout this session:" in system
    assert guidelines in system
    assert any(
        request in _text(m.get("content"))
        for m in call.messages
        if m.get("role") == "user"
    )
    model.assert_used_up()
    return handle, result, call


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_guidelines_forbid_execute_code():
    """
    When guidelines say "do NOT use execute_code", they reach the model, and
    a model that answers directly ends the request with no execute_code call.
    """
    from tests.actor.code_act.helpers import get_code_act_tool_calls

    handle, result, call = await _act_with_guidelines(
        "What is 7 * 6?",
        guidelines=(
            "Do NOT use the execute_code tool. Answer the question directly "
            "from your own knowledge without running any code."
        ),
        answer="42",
    )

    # The result should contain 42
    assert "42" in str(result)

    # A guideline is advice to the model, not a tool filter: execute_code is
    # still offered, and was not called.
    assert "execute_code" in call.tool_names
    tool_calls = get_code_act_tool_calls(handle)
    assert "execute_code" not in tool_calls, (
        f"Guidelines said not to use execute_code, but it was called. "
        f"Tool calls: {tool_calls}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_guidelines_include_marker_in_response():
    """
    When guidelines require including a specific marker phrase in the final
    answer, they reach the model, and the marker line the model ends with
    survives into the result.
    """
    _handle, result, _call = await _act_with_guidelines(
        "What is the capital of France?",
        guidelines=(
            "You MUST end your final answer with the exact phrase "
            "'[GUIDELINES_FOLLOWED]' on its own line."
        ),
        answer="The capital of France is Paris.\n[GUIDELINES_FOLLOWED]",
    )
    result_str = str(result)

    assert "Paris" in result_str
    assert result_str.rstrip().endswith("\n[GUIDELINES_FOLLOWED]")


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_code_act_guidelines_response_format_constraint():
    """
    Guidelines can guide how the result is formatted: they reach the model,
    and the numbered list it answers with comes back line for line.
    """
    _handle, result, _call = await _act_with_guidelines(
        "List three colors.",
        guidelines=(
            "Format your final answer as a numbered list, one color per line, "
            "like:\n1. Red\n2. Blue\n3. Green"
        ),
        answer="1. Teal\n2. Amber\n3. Violet",
    )
    result_str = str(result)

    # The result keeps the numbered items, one per line
    assert result_str.splitlines() == ["1. Teal", "2. Amber", "3. Violet"]
