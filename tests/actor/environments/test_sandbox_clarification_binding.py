"""Deterministic contracts for sandbox clarification queue rebinding.

Locks the execute_code / execute_function bridge that points nested manager
clarifications at the outer tool call's per-call channel (mailbox A).
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.base import (
    _ClarificationQueueInjector,
    bind_sandbox_clarification_queues,
    restore_sandbox_clarification_queues,
)
from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX


class _AskTarget:
    async def ask(
        self,
        text: str,
        *,
        _clarification_up_q: asyncio.Queue[str] | None = None,
        _clarification_down_q: asyncio.Queue[str] | None = None,
    ) -> str:
        assert _clarification_up_q is not None
        assert _clarification_down_q is not None
        await _clarification_up_q.put(text)
        return await _clarification_down_q.get()


def test_bind_and_restore_sandbox_clarification_queues():
    orphan_up: asyncio.Queue[str] = asyncio.Queue()
    orphan_down: asyncio.Queue[str] = asyncio.Queue()
    call_up: asyncio.Queue[str] = asyncio.Queue()
    call_down: asyncio.Queue[str] = asyncio.Queue()
    injector = _ClarificationQueueInjector(
        target=SimpleNamespace(),
        clarification_up_q=orphan_up,
        clarification_down_q=orphan_down,
    )
    global_state = {"primitives": injector}

    token = bind_sandbox_clarification_queues(global_state, call_up, call_down)
    assert injector._clar_up_q is call_up
    assert injector._clar_down_q is call_down
    assert global_state["__clarification_up_q__"] is call_up
    assert global_state["__clarification_down_q__"] is call_down

    restore_sandbox_clarification_queues(global_state, token)
    assert injector._clar_up_q is orphan_up
    assert injector._clar_down_q is orphan_down
    assert "__clarification_up_q__" not in global_state
    assert "__clarification_down_q__" not in global_state


@pytest.mark.asyncio
async def test_bound_injector_routes_manager_clar_to_per_call_queues():
    """After bind, nested manager ask() uses the tool-call queues, not orphans."""
    orphan_up: asyncio.Queue[str] = asyncio.Queue()
    orphan_down: asyncio.Queue[str] = asyncio.Queue()
    call_up: asyncio.Queue[str] = asyncio.Queue()
    call_down: asyncio.Queue[str] = asyncio.Queue()

    hub = SimpleNamespace(contacts=_AskTarget())
    injector = _ClarificationQueueInjector(
        target=hub,
        clarification_up_q=orphan_up,
        clarification_down_q=orphan_down,
    )
    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = injector
    sb_token = _CURRENT_SANDBOX.set(sandbox)
    try:

        async def _answer() -> None:
            question = await asyncio.wait_for(call_up.get(), timeout=5)
            assert question == "Which owner?"
            assert orphan_up.empty()
            await call_down.put("acme/repo")

        answerer = asyncio.create_task(_answer())
        with CodeActActor._sandbox_call_binding(
            clarification_up_q=call_up,
            clarification_down_q=call_down,
        ):
            # Nested getattr creates a child injector that must see rebound queues.
            result = await sandbox.global_state["primitives"].contacts.ask(
                "Which owner?",
            )
        await answerer
    finally:
        _CURRENT_SANDBOX.reset(sb_token)

    assert result == "acme/repo"
    assert injector._clar_up_q is orphan_up
    assert injector._clar_down_q is orphan_down


def test_execute_code_accepts_clarification_kwargs():
    """Signature contract: ToolsData will allocate per-call clar channels."""
    src = inspect.getsource(CodeActActor._build_tools)
    assert "_clarification_up_q: asyncio.Queue[str] | None = None" in src
    assert "with self._sandbox_call_binding(" in src


# ---------------------------------------------------------------------------
# request_clarification, callable from inside generated code
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_request_clarification_blocks_until_answered():
    """The call site suspends, and resumes with the answer as its value.

    This is the property that makes asking from code meaningful. Without it a
    program can only *emit* a question and carry on, which is indistinguishable
    from not asking: the work proceeds on a guess that looks checked.
    """
    global_state: dict = {}
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    bind_sandbox_clarification_queues(global_state, up_q, down_q)

    ask = global_state["request_clarification"]
    task = asyncio.create_task(ask("Which Sarah — Chen or Okonkwo?"))

    question = await asyncio.wait_for(up_q.get(), timeout=2)
    assert question == "Which Sarah — Chen or Okonkwo?"

    # Still suspended: nothing has answered yet.
    await asyncio.sleep(0)
    assert not task.done()

    await down_q.put("Sarah Chen, in Finance.")
    assert await asyncio.wait_for(task, timeout=2) == "Sarah Chen, in Finance."


@pytest.mark.asyncio
async def test_sandbox_request_clarification_reads_queues_at_call_time():
    """Rebinding to a later tool call's queues redirects an existing handle.

    The function is installed once per bind but must not capture the queues,
    or code holding a reference from an earlier call would write into a
    channel nobody is watching.
    """
    global_state: dict = {}
    first_up: asyncio.Queue[str] = asyncio.Queue()
    first_down: asyncio.Queue[str] = asyncio.Queue()
    bind_sandbox_clarification_queues(global_state, first_up, first_down)
    ask = global_state["request_clarification"]

    second_up: asyncio.Queue[str] = asyncio.Queue()
    second_down: asyncio.Queue[str] = asyncio.Queue()
    bind_sandbox_clarification_queues(global_state, second_up, second_down)

    task = asyncio.create_task(ask("which one?"))
    assert await asyncio.wait_for(second_up.get(), timeout=2) == "which one?"
    assert first_up.empty()
    await second_down.put("the second")
    assert await asyncio.wait_for(task, timeout=2) == "the second"


@pytest.mark.asyncio
async def test_sandbox_request_clarification_absent_without_a_channel():
    """No channel, no tool — and a legible error if it is called anyway.

    A loop without clarification queues must not appear to offer asking. The
    contract is to proceed on a stated assumption instead.
    """
    global_state: dict = {}
    assert "request_clarification" not in global_state

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()
    token = bind_sandbox_clarification_queues(global_state, up_q, down_q)
    ask = global_state["request_clarification"]
    restore_sandbox_clarification_queues(global_state, token)

    # Restored away with the queues it belonged to.
    assert "request_clarification" not in global_state
    with pytest.raises(RuntimeError, match="No clarification channel"):
        await ask("anyone there?")
