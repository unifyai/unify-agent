"""Symbolic: an act with ``response_format`` returns the parsed model.

The model is the scripted transport (``tests/scripted_model.py``). With a
response format the actor answers with ``tools.answer(answer)`` in a cell; the
answer is validated against the format. Cells run in the real sandboxed worker.
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import CodeActActor


class AnswerModel(BaseModel):
    answer: int = Field(description="The integer answer.")


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_code_act_response_format_roundtrip(core_world):  # noqa: F811
    """CodeActActor returns structured output when response_format is provided."""
    model = ScriptedModel(
        actor=[reply(calls=[("execute_code", {"code": "tools.answer({'answer': 123})"})])],
    )
    with scripted(model):
        handle = await CodeActActor(timeout=60).act(
            "Return {answer: 123}. Do not include any extra keys.",
            clarification_enabled=False,
            response_format=AnswerModel,
            persist=False,
        )
        res = await asyncio.wait_for(handle.result(), timeout=120)
    assert isinstance(res, AnswerModel)
    assert res.answer == 123
    (only,) = model.of("actor")
    assert only.tool_names == ["execute_code"]
    model.assert_used_up()
