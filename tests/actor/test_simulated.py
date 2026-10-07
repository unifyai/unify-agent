import asyncio
import functools
import pytest

from unify.actor.simulated import SimulatedActor, SimulatedActorHandle
from pydantic import BaseModel, Field
from typing import List
from tests.helpers import (
    _handle_project,
    _assert_blocks_while_paused,
    DEFAULT_TIMEOUT,
    _unique_token,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (steerable handles)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic start-and-act                                                     #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_start_and_act():
    actor = SimulatedActor()
    handle = await actor.act("Perform a quick demo.")
    handle.trigger_completion()
    result = await handle.result()
    assert isinstance(result, str) and result.strip(), "Result should be non-empty"


class ActionResult(BaseModel):
    """Structured result from an actor action."""

    completed: bool = Field(..., description="Whether the action completed")
    steps_taken: List[str] = Field(
        default_factory=list,
        description="List of steps taken",
    )
    outcome: str = Field(..., description="Description of the outcome")


@pytest.mark.asyncio
@_handle_project
async def test_simulated_act_response_format():
    """Simulated Actor.act should return structured output when response_format is provided."""
    actor = SimulatedActor()

    handle = await actor.act(
        "Perform a quick demo task and report results",
        response_format=ActionResult,
    )
    handle.trigger_completion()
    result = await handle.result()

    assert isinstance(result, ActionResult)
    assert isinstance(result.completed, bool)
    assert isinstance(result.steps_taken, list)
    assert result.outcome.strip(), "Outcome should be non-empty"


# ────────────────────────────────────────────────────────────────────────────
# 2.  Stateful memory – serial asks (via handle.ask)                          #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stateful_memory_serial_asks():
    """
    Two consecutive activities should share the same stateful LLM context.
    We exercise this by asking questions via the handle's ask() method.

    Seed a unique token in the first ask and require the second ask to recall
    it, rather than asking the model to invent a codename.
    """
    actor = SimulatedActor()

    token = _unique_token("MARKER")
    h1 = await actor.act("Start some new research.")
    ask_handle1 = await h1.ask(
        "Acknowledge the research start in one short sentence. "
        f"Include this exact marker verbatim: {token}",
    )
    first_answer = (await ask_handle1.result()).strip()
    assert first_answer, "First answer should not be empty"

    h2 = await actor.act("Continue the research.")
    ask_handle2 = await h2.ask(
        "What exact marker did you include earlier? Quote it verbatim.",
    )
    answer2 = await ask_handle2.result()
    assert (
        isinstance(answer2, str) and answer2.strip()
    ), "Second answer should be non-empty"
    assert token in answer2, "LLM should recall the previously mentioned marker"

    # Explicitly complete both handles
    h1.trigger_completion()
    h2.trigger_completion()
    await h1.result()
    await h2.result()


# ────────────────────────────────────────────────────────────────────────────
# Steerable handle tests                                                      #
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# 3.  Interject                                                               #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_interject(monkeypatch):
    calls = {"interject": 0}
    original = SimulatedActorHandle.interject

    @functools.wraps(original)
    async def wrapped(self, instruction: str):  # type: ignore[override]
        calls["interject"] += 1
        return await original(self, instruction)

    monkeypatch.setattr(SimulatedActorHandle, "interject", wrapped, raising=True)

    actor = SimulatedActor()
    handle = await actor.act("Show me all steps performed so far.")
    await asyncio.sleep(0.05)
    await handle.interject("Also consider revenue trends.")
    handle.trigger_completion()
    await handle.result()
    assert calls["interject"] == 1, ".interject should be invoked exactly once"


# ────────────────────────────────────────────────────────────────────────────
# 4.  Stop                                                                    #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_stop(monkeypatch):
    actor = SimulatedActor()
    handle = await actor.act("Generate a long report.")
    await asyncio.sleep(0.05)
    await handle.stop("Not needed")
    result = await handle.result()
    assert isinstance(result, str) and result.strip()
    assert handle.done(), "Handle should report done after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 5.  Clarification handshake                                                 #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_requests_clarification():
    actor = SimulatedActor(_requests_clarification=True)

    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await actor.act(
        "Compile the quarterly report",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert "clarify" in question.lower()

    # Answering the clarification triggers completion
    await down_q.put("Yes, please compile the Q1 report now.")
    result = await handle.result()
    assert isinstance(result, str) and result.strip()
    assert "q1 report" in result.lower()


@pytest.mark.asyncio
@_handle_project
async def test_next_clarification_blocks_without_queue():
    """
    When no clarification queue is provided, next_clarification() should block
    until the action completes (similar to next_notification with emit_notifications=False).

    This prevents watcher loops from spinning when clarifications aren't enabled.
    """
    actor = SimulatedActor(steps=None)  # Action runs indefinitely
    handle = await actor.act("Do some work")

    # Start a task that waits on next_clarification
    clarification_task = asyncio.create_task(handle.next_clarification())

    # Give it a moment to start blocking
    await asyncio.sleep(0.1)

    # Task should NOT be done yet (it should be blocking)
    assert (
        not clarification_task.done()
    ), "next_clarification() should block when no queue"

    # Now complete the action
    handle.trigger_completion()

    # The clarification task should now complete
    result = await asyncio.wait_for(clarification_task, timeout=DEFAULT_TIMEOUT)
    assert result == {}, "Should return empty dict when no clarifications"


# ────────────────────────────────────────────────────────────────────────────
# 6.  Pause → Resume round-trip                                              #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_pause_and_resume(monkeypatch):
    counts = {"pause": 0, "resume": 0}

    orig_pause = SimulatedActorHandle.pause

    @functools.wraps(orig_pause)
    async def _patched_pause(self):  # type: ignore[override]
        counts["pause"] += 1
        return await orig_pause(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "pause",
        _patched_pause,
        raising=True,
    )

    orig_resume = SimulatedActorHandle.resume

    @functools.wraps(orig_resume)
    async def _patched_resume(self):  # type: ignore[override]
        counts["resume"] += 1
        return await orig_resume(self)

    monkeypatch.setattr(
        SimulatedActorHandle,
        "resume",
        _patched_resume,
        raising=True,
    )

    actor = SimulatedActor()
    handle = await actor.act("Summarise all open opportunities.")

    pause_reply = await handle.pause()
    assert "pause" in pause_reply.lower()

    res = await _assert_blocks_while_paused(handle.result())

    resume_reply = await handle.resume()
    assert "resume" in resume_reply.lower() or "running" in resume_reply.lower()

    # Trigger completion after resume
    handle.trigger_completion()

    answer = await asyncio.wait_for(res, timeout=DEFAULT_TIMEOUT)
    assert isinstance(answer, str) and answer.strip()
    assert counts == {"pause": 1, "resume": 1}


# ────────────────────────────────────────────────────────────────────────────
# 7.  Ask on handle                                                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_ask():
    actor = SimulatedActor()
    handle = await actor.act("Summarize all unread messages this week.")

    # Ask a follow-up while running
    await asyncio.sleep(0.05)
    ask_handle = await handle.ask("What is the key point to emphasize?")
    reply = await ask_handle.result()
    assert isinstance(reply, str) and reply.strip()

    # Trigger completion and get the result
    handle.trigger_completion()
    result = await handle.result()
    assert isinstance(result, str) and result.strip()


# ────────────────────────────────────────────────────────────────────────────
# 8.  Pause should freeze duration timer                                       #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_pause_freezes_duration():
    # This test specifically validates duration freezing behavior
    actor = SimulatedActor(duration=0.2)
    handle = await actor.act("Time-sensitive work.")

    # Give the worker thread a moment to start, then pause quickly
    await asyncio.sleep(0.05)
    await handle.pause()

    # While paused, wait longer than the total duration; it should NOT complete
    res = asyncio.create_task(handle.result())
    await asyncio.sleep(0.3)
    assert (
        not res.done()
    ), "result() must not complete while paused even if wall time exceeds duration"

    # Resume and ensure it doesn't complete immediately; some time should elapse
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await handle.resume()
    answer = await asyncio.wait_for(res, timeout=2)
    elapsed_after_resume = loop.time() - t0
    assert isinstance(answer, str) and answer.strip()
    assert (
        elapsed_after_resume >= 0.05
    ), "Should wait after resume; clock was frozen while paused"


# ────────────────────────────────────────────────────────────────────────────
# 9.  next_notification emits progress without consuming steps                #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_next_notification_emits_progress():
    """next_notification returns a progress event without consuming steps (observing isn't work)."""
    # Use steps=1 to verify that next_notification doesn't consume the step
    actor = SimulatedActor(steps=1)
    handle = await actor.act("Quick simulated task.")

    # Capture remaining steps before
    before = handle.get_remaining_steps()
    assert isinstance(before, int) and before == 1

    # next_notification should return a real event but NOT consume a step
    evt = await asyncio.wait_for(handle.next_notification(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(evt, dict)
    assert evt.get("type") == "notification"
    assert evt.get("tool_name") == "simulated_actor"
    assert isinstance(evt.get("message"), str) and evt.get("message").strip()

    # Steps should remain unchanged (observing progress isn't work)
    after = handle.get_remaining_steps()
    assert isinstance(after, int) and after == before

    # Trigger completion and ensure result is available
    handle.trigger_completion()
    res = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(res, str) and res.strip()


# ────────────────────────────────────────────────────────────────────────────
# 10.  Stop while paused should finish immediately                           #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_paused_finishes_immediately():
    actor = SimulatedActor()
    handle = await actor.act("A long-running simulated activity.")

    # Pause quickly to freeze the worker thread
    await asyncio.sleep(0.05)
    await handle.pause()

    # Give the worker enough time to enter the paused wait state before stopping
    await asyncio.sleep(0.2)

    # Stopping should unpause and complete immediately
    await handle.stop("cancelled by user")
    result = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(result, str) and "stopped" in result.lower()
    assert handle.done()
    # Verify the worker thread terminates promptly (would hang pre-fix)
    for _ in range(100):
        th = getattr(handle, "_action_thread", None)
        if th is None or not getattr(th, "is_alive", lambda: False)():
            break
        await asyncio.sleep(0.01)
    th = getattr(handle, "_action_thread", None)
    assert (
        th is None or not th.is_alive()
    ), "Action thread should terminate after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 11.  Stop while waiting for clarification should finish immediately         #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_stop_while_waiting_for_clarification_finishes_immediately():
    # Configure the actor to request a clarification at the start of work
    actor = SimulatedActor(_requests_clarification=True)
    up_q: asyncio.Queue[str] = asyncio.Queue()
    down_q: asyncio.Queue[str] = asyncio.Queue()

    handle = await actor.act(
        "Compile the annual report",
        _clarification_up_q=up_q,
        _clarification_down_q=down_q,
    )

    # Wait until the clarification question is asked
    question = await asyncio.wait_for(up_q.get(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(question, str) and "clarify" in question.lower()

    # Without answering, issue stop and ensure result returns promptly
    await handle.stop("no longer needed")
    result = await asyncio.wait_for(handle.result(), timeout=DEFAULT_TIMEOUT)
    assert isinstance(result, str) and "stopped" in result.lower()
    assert handle.done()
    # Verify the worker thread terminates promptly (would spin-wait pre-fix)
    for _ in range(100):
        th = getattr(handle, "_action_thread", None)
        if th is None or not getattr(th, "is_alive", lambda: False)():
            break
        await asyncio.sleep(0.01)
    th = getattr(handle, "_action_thread", None)
    assert (
        th is None or not th.is_alive()
    ), "Action thread should terminate after stop()"


# ────────────────────────────────────────────────────────────────────────────
# 12.  trigger_completion is idempotent                                       #
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_trigger_completion_idempotent():
    """Calling trigger_completion multiple times should be safe (no-op after first)."""
    actor = SimulatedActor()
    handle = await actor.act("Some task.")

    # First trigger completes the actor
    handle.trigger_completion("First completion")
    assert handle.done()
    result1 = await handle.result()
    assert "First completion" in result1

    # Second trigger should be a no-op (doesn't change result)
    handle.trigger_completion("Second completion")
    result2 = await handle.result()
    assert result1 == result2, "Result should not change after second trigger"


# ────────────────────────────────────────────────────────────────────────────
# 13.  Cancellation regression test for result/next_notification/next_clarification
# ────────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_handle_methods_are_cancellable():
    """
    Regression test: result(), next_notification(), and next_clarification()
    must be cancellable without blocking event loop shutdown.

    Previously these methods used asyncio.to_thread(_done_event.wait) which
    created executor threads that blocked indefinitely. Now they use polling,
    allowing clean cancellation.
    """
    actor = SimulatedActor(steps=None, emit_notifications=False)
    handle = await actor.act("Long running task")

    # Start tasks that would block indefinitely without completion
    result_task = asyncio.create_task(handle.result())
    notification_task = asyncio.create_task(handle.next_notification())
    clarification_task = asyncio.create_task(handle.next_clarification())

    # Give tasks time to start their polling loops
    await asyncio.sleep(0.2)

    # Verify tasks are still running (not completed)
    assert not result_task.done(), "result() should be waiting"
    assert not notification_task.done(), "next_notification() should be waiting"
    assert not clarification_task.done(), "next_clarification() should be waiting"

    # Cancel all tasks - this should succeed without hanging
    result_task.cancel()
    notification_task.cancel()
    clarification_task.cancel()

    # Wait for cancellation to complete (should be immediate)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(result_task, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(notification_task, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(clarification_task, timeout=1.0)

    # Cleanup: complete the handle to avoid any lingering state
    handle.trigger_completion()
    assert handle.done()
