"""
E2E tests for primitive handle steering through the CodeActActor.

Verifies that when the CodeActActor invokes ``primitives.actor.act`` via
``execute_function`` or ``execute_code``, the returned SteerableToolHandle(s)
are adopted by the outer tool loop and can be steered (interjected, paused,
resumed) from the outside.

``primitives.actor.act`` is routed to a ``SimulatedActor`` so each handle is
a real steerable handle (pause/resume/interject/stop all work) without
spawning a nested code-writing actor.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.actor.code_act.helpers import (
    extract_code_act_execute_code_snippets,
    patch_actor_act,
)
from tests.async_helpers import _wait_for_condition
from unify.actor.code_act_actor import CodeActActor
from unify.actor.environments.actor import ActorEnvironment
from unify.actor.simulated import SimulatedActor

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


# ────────────────────────────────────────────────────────────────────────────
# Budgets
# ────────────────────────────────────────────────────────────────────────────

# Every wait here derives from the actor's own budget, because the ordering
# between them decides what a slow run looks like. A wait shorter than the
# budget reports a run the actor was still entitled to finish as a bare
# TimeoutError from `wait_for`, which names neither the step that was slow nor
# the actor's verdict on it. Outliving the budget lets the actor time out
# first and say so.
ACTOR_TIMEOUT = 240.0

# The actor needs room to convert its own expiry into a result before the wait
# gives up on it.
RESULT_WAIT = ACTOR_TIMEOUT + 30.0

# A tool result appears mid-run, so it can never legitimately arrive later than
# the run itself is allowed to last.
TOOL_RESULT_WAIT = ACTOR_TIMEOUT

# Leaves headroom above the longest wait for fixture setup and teardown, so the
# in-test assertions are what fail rather than pytest killing the test first.
TEST_TIMEOUT = RESULT_WAIT + 90.0

# Wall-clock life of a simulated sub-actor that finishes on its own: long
# enough for the test's own steering calls to reach it mid-flight, short
# enough that the outer loop finishes promptly.
SUB_ACTOR_DURATION = 3.0


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _simulate_sub_actors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    steps: int | None = None,
    duration: float | None = SUB_ACTOR_DURATION,
) -> list[str]:
    """Back ``primitives.actor.act`` with a SimulatedActor per call.

    Each sub-actor finishes after ``steps`` steering calls (interject, ask,
    pause, resume) or ``duration`` seconds, whichever comes first.
    """
    requests: list[str] = []

    async def _impl(request: str, **kwargs):
        requests.append(request)
        return await SimulatedActor(steps=steps, duration=duration).act(
            request,
            clarification_enabled=False,
        )

    patch_actor_act(monkeypatch, _impl)
    return requests


def _restrict_to_execute_code(actor: CodeActActor) -> None:
    """Limit the actor tool surface so mode selection happens inside execute_code."""
    act_tools = actor.get_tools("act")
    actor.add_tools("act", {"execute_code": act_tools["execute_code"]})


async def _wait_for_tool_result_in_transcript(
    handle,
    tool_name: str,
    *,
    timeout: float = TOOL_RESULT_WAIT,
) -> None:
    """Wait until a tool result for *tool_name* appears in the handle's transcript.

    The transcript (``handle.get_history()``) is append-only, so a tool result
    message is a permanent, race-free signal that the tool ran and its return
    value was processed (including handle adoption when applicable).
    """

    async def _predicate():
        return any(
            m.get("role") == "tool" and m.get("name") == tool_name
            for m in handle.get_history()
        )

    await _wait_for_condition(_predicate, poll=0.1, timeout=timeout)


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_function path
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_function_primitive_steering(monkeypatch):
    """CodeActActor (can_compose=False) → execute_function → primitives.actor.act
    → handle adopted → interjection forwarded → result incorporates both turns.
    """
    requests = _simulate_sub_actors(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=ACTOR_TIMEOUT)

    try:
        # can_compose=False forces the LLM to use execute_function (no code sandbox).
        handle = await actor.act(
            "Step 1: Call FunctionManager_list_functions (required first step).\n"
            "Step 2: Call execute_function with function_name='primitives.actor.act' "
            "and call_kwargs={'request': 'Draft a short summary of the Berlin office'}. "
            "The function WILL be found even if the list appeared empty.",
            can_compose=False,
            clarification_enabled=False,
        )

        # Wait for the execute_function tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_function",
            timeout=TOOL_RESULT_WAIT,
        )

        # Steer: interject additional context mid-flight.
        await handle.interject(
            "Also cover the Munich office.",
        )

        # Steer: pause then resume to verify lifecycle methods propagate.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=RESULT_WAIT)
        assert result is not None, "Expected a non-None result from the actor"
        assert requests, "primitives.actor.act was never reached"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_code_mode_selection_realistic_steerable_intent(monkeypatch):
    """Natural request that implies mid-flight control should return a handle."""
    # The request promises a refinement while the sub-actor is underway, so
    # the sub-actor must still be running when it arrives, however long the
    # model takes to forward it: it finishes on the first steering it gets.
    # Anchoring completion to that tool call rather than to a clock racing
    # the model's turns also puts it in the same place on every replay.
    requests = _simulate_sub_actors(monkeypatch, steps=1, duration=None)
    actor = CodeActActor(
        environments=[ActorEnvironment()],
        timeout=ACTOR_TIMEOUT,
        tool_policy=None,
    )
    _restrict_to_execute_code(actor)
    handle = None

    try:
        handle = await actor.act(
            "Delegate drafting a summary of the Berlin office to a sub-actor now, "
            "but keep it running because I may refine the brief while it is underway.",
            clarification_enabled=False,
        )

        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=TOOL_RESULT_WAIT,
        )

        snippets = extract_code_act_execute_code_snippets(handle)
        assert snippets, "Expected CodeAct to use execute_code."
        assert any(
            "primitives.actor.act" in snippet and ".result(" not in snippet
            for snippet in snippets
        ), (
            "Expected at least one execute_code snippet to return a primitive handle "
            "without awaiting .result() for steerable user intent.\n"
            f"Snippets:\n{chr(10).join(snippets)}"
        )
        assert requests, (
            "Expected execute_code to launch the sub-actor, not only name it.\n"
            f"Snippets:\n{chr(10).join(snippets)}"
        )

        await handle.interject("Also cover the Munich office.")
        result = await asyncio.wait_for(handle.result(), timeout=RESULT_WAIT)
        assert result is not None, "Expected a non-None result from the actor"
        assert len(requests) == 1, (
            "Expected the refinement to reach the running sub-actor rather than "
            f"a relaunch; sub-actors launched: {requests}"
        )
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_code_mode_selection_realistic_inline_composition(monkeypatch):
    """Natural request that requires same-block processing should await result."""
    _simulate_sub_actors(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=ACTOR_TIMEOUT)
    _restrict_to_execute_code(actor)
    handle = None

    try:
        handle = await actor.act(
            "In one code step, delegate drafting a summary of the Berlin office to "
            "a sub-actor and immediately compute the word count of its answer "
            "before replying.",
            clarification_enabled=False,
        )

        result = await asyncio.wait_for(handle.result(), timeout=RESULT_WAIT)
        assert result is not None, "Expected a non-None result from the actor"

        snippets = extract_code_act_execute_code_snippets(handle)
        assert snippets, "Expected CodeAct to use execute_code."
        assert any(
            "primitives.actor.act" in snippet and ".result(" in snippet
            for snippet in snippets
        ), (
            "Expected at least one execute_code snippet to await .result() for inline "
            "composition intent.\n"
            f"Snippets:\n{chr(10).join(snippets)}"
        )
    finally:
        try:
            if handle is not None and not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_code path
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_code_primitive_steering(monkeypatch):
    """CodeActActor → execute_code calling primitives.actor.act(...)
    → handle returned as last expression → adopted → interjection forwarded.
    """
    requests = _simulate_sub_actors(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=ACTOR_TIMEOUT)

    try:
        # The system prompt already documents steerable handles. Give an
        # explicit instruction so the LLM returns the handle for steering
        # rather than awaiting it inline.
        handle = await actor.act(
            "Use execute_code to call `await primitives.actor.act(request='Draft a "
            "short summary of the Berlin office')` as the **last expression** so "
            "the handle is returned for steering. "
            "Do NOT await handle.result() inside the code.",
            clarification_enabled=False,
        )

        # Wait for the execute_code tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=TOOL_RESULT_WAIT,
        )

        # Steer: interject additional context mid-flight.
        await handle.interject(
            "Also cover the Munich office.",
        )

        # Steer: pause then resume.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=RESULT_WAIT)
        assert result is not None, "Expected a non-None result from the actor"
        assert requests, "primitives.actor.act was never reached"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Test: execute_code with two concurrent steerable handles
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_execute_code_dual_primitive_steering(monkeypatch):
    """CodeActActor → execute_code returning two steerable handles
    (two sub-actors) from a single code block.

    Both handles should be adopted via the multi-handle adoption path and
    each should be individually steerable from the outer loop.
    """
    requests = _simulate_sub_actors(monkeypatch)
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=ACTOR_TIMEOUT)

    try:
        handle = await actor.act(
            "Use a SINGLE execute_code call to launch two sub-actors and "
            "return both handles as the last expression (a dict). "
            "The code should be exactly:\n\n"
            "```python\n"
            "h1 = await primitives.actor.act(request='Draft a summary of the Berlin office')\n"
            "h2 = await primitives.actor.act(request='Draft a summary of the Munich office')\n"
            "{'berlin_handle': h1, 'munich_handle': h2}\n"
            "```\n\n"
            "Do NOT await .result() on either handle inside the code.",
            clarification_enabled=False,
        )

        # Wait for the execute_code tool result in the transcript.
        await _wait_for_tool_result_in_transcript(
            handle,
            "execute_code",
            timeout=TOOL_RESULT_WAIT,
        )

        # Steer the first handle via an interjection.
        await handle.interject("Also mention headcount in each summary.")

        # Steer the second handle via a pause/resume cycle.
        await handle.pause()
        await asyncio.sleep(0.5)
        await handle.resume()

        # Let the loop finish.
        result = await asyncio.wait_for(handle.result(), timeout=RESULT_WAIT)
        assert result is not None, "Expected a non-None result from the actor"
        assert len(requests) == 2, f"Expected two sub-actors, got: {requests}"
    finally:
        try:
            if not handle.done():
                await handle.stop("test cleanup")
        except Exception:
            pass
        try:
            await actor.close()
        except Exception:
            pass
