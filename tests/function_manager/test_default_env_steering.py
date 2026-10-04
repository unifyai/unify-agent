"""Steering the direct function-execution path.

``FunctionManager._execute_python_function`` runs a stored function in-process
without going through the sandbox, so it needs its own probes. Without them the
same stored function is correctable or not depending purely on which route
reached it — via the actor's synthesised preamble, or called here directly.

The mechanism is asserted with the patch supplied directly; whether a real
model writes a usable one on this path is the eval test at the end.
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest

from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.steering import (
    InterruptionRequest,
    Patch,
    SteeringSession,
    current_session,
    use_session,
)

IMPLEMENTATION = (
    "async def notify_vendors(vendors):\n"
    "    sent = []\n"
    "    for v in vendors:\n"
    "        sent.append(await primitives.comms.send(v))\n"
    "    return sent\n"
)

VENDORS = ["eu-alpha", "us-beta", "us-gamma", "eu-delta"]

EU_ONLY_PATCH = (
    "async def notify_vendors(vendors):\n"
    "    sent = []\n"
    "    for v in vendors:\n"
    "        if v.startswith('eu-'):\n"
    "            sent.append(await primitives.comms.send(v))\n"
    "    return sent\n"
)


class _Comms:
    def __init__(self) -> None:
        self.sent: List[str] = []

    async def send(self, to: str) -> str:
        self.sent.append(to)
        await asyncio.sleep(0)
        return f"sent:{to}"


class _Prims:
    def __init__(self, comms: _Comms) -> None:
        self.comms = comms


def _manager() -> FunctionManager:
    """A FunctionManager instance without touching its constructor's I/O.

    Only ``_execute_python_function`` is under test, and it reads nothing from
    the instance except the in-process session dict.
    """
    manager = FunctionManager.__new__(FunctionManager)
    manager._in_process_sessions = {}
    return manager


async def _execute(
    manager: FunctionManager,
    comms: _Comms,
    *,
    state_mode: str = "stateless",
    session_id: int = 0,
    parent_chat_context: list | None = None,
) -> dict:
    return await manager._execute_python_function(
        implementation=IMPLEMENTATION,
        call_kwargs={"vendors": list(VENDORS)},
        state_mode=state_mode,
        session_id=session_id,
        extra_namespaces={"primitives": _Prims(comms)},
        _parent_chat_context=parent_chat_context,
    )


def _patch_after(comms: _Comms, count: int, queue: asyncio.Queue, text: str):
    async def _run() -> None:
        while len(comms.sent) < count:
            await asyncio.sleep(0)
        await queue.put(text)

    return asyncio.create_task(_run())


async def _author(*, interjections, session):
    return InterruptionRequest(
        reason=interjections[0],
        patches=[Patch(function_name="notify_vendors", source=EU_ONLY_PATCH)],
    )


async def _stop_author(*, interjections, session):
    return InterruptionRequest(reason=interjections[0], stop=True)


# ── unsteered behaviour is unchanged ───────────────────────────────────────
@pytest.mark.asyncio
async def test_runs_normally_with_no_session():
    comms = _Comms()
    out = await _execute(_manager(), comms)
    assert out["error"] is None, out["error"]
    assert comms.sent == VENDORS


@pytest.mark.asyncio
async def test_sync_function_still_runs():
    """A synchronous function runs on the same path as an async one."""
    manager = _manager()
    out = await manager._execute_python_function(
        implementation="def add(a, b):\n    return a + b\n",
        call_kwargs={"a": 2, "b": 3},
        extra_namespaces={},
    )
    assert out["error"] is None, out["error"]
    assert out["result"] == 5


@pytest.mark.asyncio
async def test_missing_definition_still_errors():
    manager = _manager()
    out = await manager._execute_python_function(
        implementation="x = 1\n",
        call_kwargs={},
        extra_namespaces={},
    )
    assert "No function definition found" in (out["error"] or "")


# ── steering reaches this path ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_correction_reaches_the_function_body():
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        out = await _execute(_manager(), comms)
    await steerer

    assert out["error"] is None, out["error"]
    assert session.retries == 1
    # The correction landed before this one.
    assert "us-gamma" not in comms.sent
    # Already contacted, so replayed rather than contacted again.
    assert comms.sent.count("eu-alpha") == 1
    assert "eu-delta" in comms.sent


@pytest.mark.asyncio
async def test_stop_is_returned_as_a_clean_function_result():
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait("cancel the remaining sends")
    session = SteeringSession(interject_q=queue, patch_author=_stop_author)

    with use_session(session):
        out = await _execute(_manager(), comms)

    assert out["error"] is None
    assert out["result"] == {
        "status": "stopped",
        "reason": "cancel the remaining sends",
    }
    assert comms.sent == []


@pytest.mark.asyncio
async def test_pending_stop_reaches_a_sync_function():
    """A sync frame cannot await a checkpoint, but raising needs no await:
    a correction already pending when the call begins fires at entry."""
    from unify.function_manager.steering import InterruptionRequest

    manager = _manager()
    session = SteeringSession()
    session.interruption = InterruptionRequest(reason="task revoked", stop=True)

    with use_session(session):
        out = await manager._execute_python_function(
            implementation="def add(a, b):\n    return a + b\n",
            call_kwargs={"a": 2, "b": 3},
            extra_namespaces={},
        )

    assert out["error"] is None
    assert out["result"] == {"status": "stopped", "reason": "task revoked"}


@pytest.mark.asyncio
async def test_completed_calls_replay_rather_than_repeat():
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        await _execute(_manager(), comms)
    await steerer

    assert session.cache.hits >= 1, "nothing replayed; the retry redid the prefix"
    assert len(comms.sent) == len(set(comms.sent)), comms.sent


@pytest.mark.asyncio
async def test_steering_composes_with_context_forwarding():
    """The memoised wrapper must sit outside the context-forwarding one.

    Context forwarding returns a synchronous wrapper around an async method,
    so the two have to compose in a fixed order or dispatches take the path
    that cannot suspend.
    """
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        out = await _execute(
            _manager(),
            comms,
            parent_chat_context=[{"role": "user", "content": "hi"}],
        )
    await steerer

    assert out["error"] is None, out["error"]
    assert session.retries == 1
    assert "us-gamma" not in comms.sent


@pytest.mark.asyncio
async def test_the_author_sees_how_a_direct_call_was_made():
    """A bare definition does not show its arguments; the author must be told.

    The sandbox route steers a synthesised block that ends in the call, so its
    arguments are in the source. This route steers the definition alone, and
    without the call a correction about *which* vendors has to guess what a
    vendor is.
    """
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    seen: dict = {}

    async def _recording_author(*, interjections, session):
        seen["invocation"] = session.invocation
        seen["completed"] = session.cache.completed_calls()
        return await _author(interjections=interjections, session=session)

    session = SteeringSession(interject_q=queue, patch_author=_recording_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        out = await _execute(_manager(), comms)
    await steerer

    assert out["error"] is None, out["error"]
    assert seen["invocation"] == f"notify_vendors(vendors={VENDORS!r})"
    assert seen["completed"] == [
        "comms.send('eu-alpha')",
        "comms.send('us-beta')",
    ]
    # Scoped to the call: nothing describes a call that has returned.
    assert session.invocation == ""


# ── the globals must be left as they were found ────────────────────────────
@pytest.mark.asyncio
async def test_stateful_globals_are_restored_after_a_steered_call():
    """A stateful session dict outlives the call, so probes must not.

    Leaving `_cp`, `runtime`, or a memoised `primitives` behind would let a
    later call in the same session write into a finished call's cache.
    """
    manager = _manager()
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        await _execute(manager, comms, state_mode="stateful", session_id=7)
    await steerer

    globals_dict = manager._in_process_sessions[7]
    assert current_session(globals_dict) is None
    for probe in ("_cp", "_int", "_around_cp", "runtime"):
        assert probe not in globals_dict, probe

    # And a later call in the same session behaves normally.
    second = _Comms()
    out = await _execute(manager, second, state_mode="stateful", session_id=7)
    assert out["error"] is None, out["error"]
    assert second.sent == VENDORS


@pytest.mark.asyncio
async def test_primitives_are_restored_when_the_session_had_none():
    """Injected namespaces are per-call; a wrapper must not linger as one."""
    manager = _manager()
    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=_author)
    steerer = _patch_after(comms, 2, queue, "only the EU vendors")

    with use_session(session):
        await _execute(manager, comms, state_mode="stateful", session_id=3)
    await steerer

    stored = manager._in_process_sessions[3].get("primitives")
    from unify.function_manager.steering import MemoisedDispatch

    assert not isinstance(stored, MemoisedDispatch)


# ── a real model, on this path ─────────────────────────────────────────────
@pytest.mark.eval
@pytest.mark.asyncio
async def test_real_model_corrects_a_directly_executed_function():
    """Same judgement question as the sandbox path, different entry point."""
    from unify.function_manager.steering_patcher import build_patch_author

    comms = _Comms()
    queue: asyncio.Queue = asyncio.Queue()
    session = SteeringSession(interject_q=queue, patch_author=build_patch_author())
    steerer = _patch_after(
        comms,
        2,
        queue,
        "Hold on — only the EU vendors should be contacted, not the US ones.",
    )

    with use_session(session):
        out = await _execute(_manager(), comms)
    await steerer

    assert out["error"] is None, out["error"]
    assert session.retries >= 1, "the correction never reached the function"
    assert "us-gamma" not in comms.sent
    assert len(comms.sent) == len(set(comms.sent)), comms.sent
