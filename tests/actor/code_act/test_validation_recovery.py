"""Symbolic: a refused ``execute_code`` call is answered with what to change,
and the loop goes on.

The model is the scripted transport (``tests/scripted_model.py``): it makes
the invalid call (a parameter execute_code does not have), then the corrected
one, then answers. Whether a live model corrects itself is a behaviour
question; this checks the harness's half: the refusal reaches the model as a
tool result that says what to change, and the next call runs.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import CodeActActor


class ValidationRecoveryResult(BaseModel):
    saw_validation_error: bool
    final_stdout: str


def _execute(code: str, **params) -> tuple[str, dict]:
    return ("execute_code", {"thought": "Printing a greeting.", "code": code, **params})


def _tool_results(call) -> list[str]:
    return [str(m.get("content")) for m in call.messages if m.get("role") == "tool"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_validation_error_self_correction(core_world):  # noqa: F811
    """
    Real-world scenario: Actor makes an invalid call, is refused with a message
    explaining what to change, then self-corrects and succeeds.
    """
    answer = {"saw_validation_error": True, "final_stdout": "hi"}
    greet = 'msg = "hi"\nprint(msg)'
    model = ScriptedModel(
        actor=[
            reply(calls=[_execute(greet, session_name="oops")]),
            reply(calls=[_execute(greet)]),
            reply(calls=[_execute(f"tools.answer({answer!r})")]),
        ],
    )

    actor = CodeActActor(timeout=60)
    with scripted(model):
        handle = await actor.act(
            "Print hi, then report what you saw.",
            response_format=ValidationRecoveryResult,
            persist=False,
            clarification_enabled=False,
        )
        res = await asyncio.wait_for(handle.result(), timeout=170)

    assert isinstance(res, ValidationRecoveryResult)
    assert res.saw_validation_error is True
    assert "hi" in (res.final_stdout or "")

    _, after_refusal, after_run = model.of("actor")
    (refusal,) = _tool_results(after_refusal)
    assert "execute_code was not run: it has no parameter named 'session_name'." in refusal
    assert "using only its own parameters: code, thought" in refusal
    # The corrected call ran: its output is the next tool result.
    assert "hi" in _tool_results(after_run)[-1]
    assert "was not run" not in _tool_results(after_run)[-1]
    model.assert_used_up()
