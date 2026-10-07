import asyncio

import pytest
from pydantic import BaseModel, Field
from unittest.mock import MagicMock

from tests.actor.code_act.helpers import patch_actor_act
from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.actor import ActorEnvironment
from unify.actor.execution.session import PythonExecutionSession, _PARENT_CHAT_CONTEXT
from unify.actor.simulated import _StaticAnswerHandle
from tests.baked_defaults import as_shipped  # noqa: F401


class SecretModel(BaseModel):
    secret: int = Field(description="The secret number from context.")


# as_shipped: deleted in step 5 (parent chat context for primitives)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_code_act_initial_parent_chat_context_is_used():
    """CodeActActor should append _parent_chat_context before the first LLM turn."""
    SecretModel.model_rebuild()

    actor = CodeActActor(timeout=60)

    parent_ctx = [
        {"role": "user", "content": "The secret number is 456."},
        {"role": "assistant", "content": "Acknowledged."},
    ]

    handle = await actor.act(
        "What is the secret number? Return {secret: <int>} and do not guess.",
        clarification_enabled=False,
        response_format=SecretModel,
        _parent_chat_context=parent_ctx,
        persist=False,
    )
    try:
        res = await asyncio.wait_for(handle.result(), timeout=90)
        assert isinstance(res, SecretModel)
        assert res.secret == 456
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


_LUCY_ANSWER = "Lucy Baker: 555-0199"

_LUCY_PARENT_CTX = [
    {
        "role": "user",
        "content": ("Can you find Lucy's number? I think her surname is Baker."),
    },
    {"role": "assistant", "content": "Sure, let me look that up for you."},
]

_LUCY_REQUEST = (
    "Delegate exactly one sub-task with primitives.actor.act: "
    "request='Find Lucy's phone number'. Set "
    "include_parent_chat_context=true on the tool call so the "
    "conversation context is available inside the sandbox, then report "
    "the sub-actor's answer."
)


def _spy_actor_act(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Route ``primitives.actor.act`` to a spy that records the parent chat
    context visible at call time.

    The real ``_ActorRunner.act`` reads ``_PARENT_CHAT_CONTEXT`` rather than
    taking a kwarg, so the spy records what that ContextVar holds when the
    sandbox reaches the primitive.
    """
    calls: list[dict] = []

    async def _impl(request: str, **kwargs):
        calls.append(
            {
                "request": request,
                "parent_chat_context": _PARENT_CHAT_CONTEXT.get(None),
            },
        )
        return _StaticAnswerHandle(_LUCY_ANSWER)

    patch_actor_act(monkeypatch, _impl)
    return calls


class _SpyRunner:
    """Records calls to act() so the test can inspect received kwargs."""

    def __init__(self) -> None:
        self.act_calls: list[dict] = []

    async def act(
        self,
        request: str,
        _parent_chat_context: list[dict] | None = None,
        **kwargs,
    ):
        self.act_calls.append(
            {"request": request, "_parent_chat_context": _parent_chat_context},
        )
        return _StaticAnswerHandle(_LUCY_ANSWER)


# ────────────────────────────────────────────────────────────────────────────
# execute_function / execute_code context forwarding
# ────────────────────────────────────────────────────────────────────────────


# as_shipped: deleted in step 5 (parent chat context for primitives)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_execute_function_forwards_parent_chat_context(monkeypatch):
    """Parent chat context should flow from the outer act() loop through the
    execute_function tool into the sandbox via the _PARENT_CHAT_CONTEXT
    ContextVar, just like execute_code.

    Scenario: the parent conversation mentions "Baker" as the surname, but
    the act() description just says "Find Lucy's phone number."  A spy
    stands in for ``primitives.actor.act`` and records the context visible
    when it is reached.

    Context injection is opt-in, so the request explicitly instructs the
    model to set include_parent_chat_context=true on the tool call; the
    subject here is the forwarding plumbing, not the model's opt-in
    judgment.
    """
    calls = _spy_actor_act(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=60)

    try:
        handle = await actor.act(
            _LUCY_REQUEST,
            can_compose=False,
            persist=False,
            clarification_enabled=False,
            _parent_chat_context=_LUCY_PARENT_CTX,
        )
        await asyncio.wait_for(handle.result(), timeout=90)

        assert len(calls) > 0, "primitives.actor.act was never called"
        assert calls[0]["parent_chat_context"] is not None, (
            "primitives.actor.act ran without _PARENT_CHAT_CONTEXT set — "
            "execute_function needs to set it for the sandbox"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# as_shipped: deleted in step 5 (parent chat context for primitives)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.eval
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(300)
async def test_execute_code_forwards_parent_chat_context(monkeypatch):
    """Parent chat context should reach primitives called from within
    execute_code.

    Same Lucy Baker scenario as the execute_function test, but here the LLM
    generates code that calls ``primitives.actor.act(...)`` directly in the
    sandbox.  As there, the request instructs the explicit
    include_parent_chat_context=true opt-in because injection is opt-in and
    the subject is the forwarding plumbing.
    """
    calls = _spy_actor_act(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=60)

    try:
        handle = await actor.act(
            _LUCY_REQUEST,
            persist=False,
            clarification_enabled=False,
            _parent_chat_context=_LUCY_PARENT_CTX,
        )
        await asyncio.wait_for(handle.result(), timeout=90)

        assert len(calls) > 0, "primitives.actor.act was never called"
        assert calls[0]["parent_chat_context"] is not None, (
            "primitives.actor.act ran without _PARENT_CHAT_CONTEXT set — "
            "execute_code needs to set it for the sandbox"
        )
    finally:
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Symbolic: PythonExecutionSession wraps primitives via _PARENT_CHAT_CONTEXT
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_execute_wraps_primitives_via_contextvar():
    """PythonExecutionSession.execute() should wrap global_state['primitives']
    with ContextForwardingProxy when the _PARENT_CHAT_CONTEXT ContextVar is
    set.  This is the single wrapping site that covers ALL execution paths
    (stateless, stateful session 0, persistent sessions, read-only)."""
    spy = _SpyRunner()

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = MagicMock(actor=spy)

    ctx = [{"role": "user", "content": "Her surname is Baker"}]
    token = _PARENT_CHAT_CONTEXT.set(ctx)
    try:
        res = await sb.execute(
            'await primitives.actor.act(request="Lucy number?")',
        )
    finally:
        _PARENT_CHAT_CONTEXT.reset(token)

    assert res["error"] is None, f"sandbox execution failed: {res['error']}"
    assert len(spy.act_calls) == 1
    assert spy.act_calls[0]["_parent_chat_context"] is ctx


@pytest.mark.asyncio
async def test_sandbox_execute_no_wrap_when_contextvar_unset():
    """When _PARENT_CHAT_CONTEXT is not set (default None), the sandbox
    should NOT wrap primitives — act() receives None."""
    spy = _SpyRunner()

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = MagicMock(actor=spy)

    res = await sb.execute(
        'await primitives.actor.act(request="Lucy number?")',
    )

    assert res["error"] is None, f"sandbox execution failed: {res['error']}"
    assert len(spy.act_calls) == 1
    assert spy.act_calls[0]["_parent_chat_context"] is None


@pytest.mark.asyncio
async def test_sandbox_execute_restores_original_primitives():
    """After execution the original primitives object must be restored in
    global_state to avoid stacking proxies across calls."""
    spy = _SpyRunner()
    original_prims = MagicMock(actor=spy)

    sb = PythonExecutionSession()
    sb.global_state["primitives"] = original_prims

    ctx = [{"role": "user", "content": "context"}]
    token = _PARENT_CHAT_CONTEXT.set(ctx)
    try:
        await sb.execute('await primitives.actor.act(request="test")')
    finally:
        _PARENT_CHAT_CONTEXT.reset(token)

    assert (
        sb.global_state["primitives"] is original_prims
    ), "original primitives not restored after execution"
