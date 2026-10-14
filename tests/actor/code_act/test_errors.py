import asyncio

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import parts_to_text


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_execute_code_can_run_without_bound_sandbox():
    """execute_code should work even if the sandbox ContextVar is missing."""
    actor = CodeActActor()

    tools = actor.get_tools("act")
    execute_code = tools["execute_code"]

    out = await execute_code(
        "run",
        "msg = 'x'\nprint(msg)",
        state_mode="stateful",
        session_id=0,
    )
    # execute_code returns a dict for non-rich outputs, and an ExecutionResult
    # (FormattedToolResult) for in-process Python rich outputs.
    err = out.get("error") if isinstance(out, dict) else getattr(out, "error", None)
    stdout = out.get("stdout") if isinstance(out, dict) else getattr(out, "stdout", "")
    stdout_text = (
        parts_to_text(stdout) if isinstance(stdout, list) else str(stdout or "")
    )

    assert err is None
    assert "x" in stdout_text

    await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_act_capacity_timeout_is_fast_when_configured():
    """act() should raise a clear error when semaphore acquisition times out."""
    actor = CodeActActor()

    # Exhaust capacity and ensure timeout is fast for the test.
    actor._act_semaphore = asyncio.Semaphore(0)
    actor._act_semaphore_timeout_s = 0.01

    with pytest.raises(RuntimeError) as e:
        await actor.act("hello")

    assert "at capacity" in str(e.value).lower()

    await actor.close()
