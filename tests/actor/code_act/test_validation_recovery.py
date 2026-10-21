"""Symbolic: a refused ``execute_code`` call is answered with what to change,
and the loop goes on.

The model is the scripted transport (``tests/scripted_model.py``): it makes
the invalid call (a stateless cell that names a session), then the corrected
one, then answers. Whether a live model corrects itself is a behaviour
question; this checks the harness's half: the refusal reaches the model as a
tool result that says what to change, and the next call runs.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import CodeActActor


class ValidationRecoveryResult(BaseModel):
    saw_validation_error: bool
    final_stdout: str


def _execute(**params) -> tuple[str, dict]:
    return (
        "execute_code",
        {"thought": "Printing a greeting.", "code": 'print("hi")', **params},
    )


def _tool_results(call) -> list[str]:
    return [str(m.get("content")) for m in call.messages if m.get("role") == "tool"]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_validation_error_self_correction():
    """
    Real-world scenario: Actor makes an invalid call, is refused with a message
    explaining what to change, then self-corrects and succeeds.
    """
    ValidationRecoveryResult.model_rebuild()
    answer = {"saw_validation_error": True, "final_stdout": "hi"}
    model = ScriptedModel(
        actor=[
            reply(calls=[_execute(state_mode="stateless", session_name="oops")]),
            reply(calls=[_execute(state_mode="stateless")]),
            reply(calls=[("final_response", {"answer": answer})]),
        ],
    )

    actor = CodeActActor(timeout=60, tool_policy=None)
    try:
        with scripted(model):
            handle = await actor.act(
                "Print hi with a stateless cell, then report what you saw.",
                response_format=ValidationRecoveryResult,
                persist=False,
                clarification_enabled=False,
                can_store=False,
            )
            res = await asyncio.wait_for(handle.result(), timeout=170)
    finally:
        await actor.close()

    assert isinstance(res, ValidationRecoveryResult)
    assert res.saw_validation_error is True
    assert "hi" in (res.final_stdout or "")

    _, after_refusal, after_run = model.of("actor")
    (refusal,) = _tool_results(after_refusal)
    assert "Cannot use state_mode='stateless' with a session." in refusal
    assert "Remove session_id/session_name" in refusal
    # The corrected call ran: its output is the next tool result.
    assert "hi" in _tool_results(after_run)[-1]
    assert "Cannot use" not in _tool_results(after_run)[-1]
    model.assert_used_up()
