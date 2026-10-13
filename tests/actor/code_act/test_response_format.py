"""Symbolic: an act with ``response_format`` returns the parsed model.

The model is the scripted transport (``tests/scripted_model.py``). With a
response format the loop offers a ``final_response`` tool and requires a
tool call; the answer is that call's ``answer`` argument, validated against
the format. Arguments that are not valid JSON are refused back to the model,
which then answers again.
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import CodeActActor


class AnswerModel(BaseModel):
    answer: int = Field(description="The integer answer.")


async def _act(model: ScriptedModel):
    AnswerModel.model_rebuild()
    actor = CodeActActor(timeout=60)
    try:
        with scripted(model):
            handle = await actor.act(
                "Return {answer: 123}. Do not include any extra keys.",
                clarification_enabled=False,
                response_format=AnswerModel,
                persist=False,
                can_store=False,
            )
            return await asyncio.wait_for(handle.result(), timeout=90)
    finally:
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_code_act_response_format_roundtrip():
    """CodeActActor should return structured output when response_format is provided."""
    model = ScriptedModel(
        actor=[reply(calls=[("final_response", {"answer": {"answer": 123}})])],
    )
    res = await _act(model)
    assert isinstance(res, AnswerModel)
    assert res.answer == 123

    (only,) = model.of("actor")
    assert "final_response" in only.tool_names
    assert only.request["tool_choice"] == "required"
    model.assert_used_up()
