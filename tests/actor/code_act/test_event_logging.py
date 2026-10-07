"""Event logging and lineage tests for CodeActActor.

This module is intentionally compact and covers the highest-signal behaviors:

- `execute_code` boundary emits ManagerMethod events and restores `TOOL_LOOP_LINEAGE`.
- `execute_function` boundary emits ManagerMethod events with `execute_function({name})`
  in the lineage and restores `TOOL_LOOP_LINEAGE`.
- FunctionManager boundary (`_LineageTrackedFunction`) composes with `execute_code`
  so that nested manager calls carry the full hierarchy.
- Concurrency does not cause lineage crosstalk between sibling function calls.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from tests.helpers import _handle_project, capture_events
from unify.actor.code_act_actor import CodeActActor
from unify.actor.execution import (
    PythonExecutionSession,
    _CURRENT_SANDBOX,
    parts_to_text,
)
from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE
from unify.events.event_bus import EVENT_BUS
from unify.events.manager_event_logging import log_manager_call
from unify.function_manager.function_manager import _LineageTrackedFunction


def _get(out: Any, key: str, default: Any = None) -> Any:
    """Get a field from either a dict or Pydantic model."""
    if isinstance(out, dict):
        return out.get(key, default)
    return getattr(out, key, default)


# ---------------------------------------------------------------------------
# execute_code boundary unit tests
# ---------------------------------------------------------------------------


def _result_error(res: Any) -> Any:
    """Return the error field from an execute_code result (dict or ExecutionResult)."""
    if isinstance(res, dict):
        return res.get("error")
    return getattr(res, "error", None)


def _result_stdout_text(res: Any) -> str:
    """Return stdout as plain text from an execute_code result (dict or ExecutionResult)."""
    if isinstance(res, dict):
        stdout = res.get("stdout") or ""
    else:
        stdout = getattr(res, "stdout", "") or ""
    return parts_to_text(stdout) if isinstance(stdout, list) else str(stdout)


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_boundary_publishes_events_and_cleans_lineage(monkeypatch):
    actor = CodeActActor(
        environments=[],  # avoid default state-manager envs in unit test
    )

    async def _fake_execute(**_kwargs):
        return {
            "stdout": "ok",
            "stderr": "",
            "result": 1,
            "error": None,
            "state_mode": "stateless",
            "session_id": 0,
            "session_created": False,
            "duration_ms": 1,
        }

    monkeypatch.setattr(actor._session_executor, "execute", _fake_execute, raising=True)

    execute_code = actor.get_tools("act")["execute_code"]
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        async with capture_events("ManagerMethod") as events:
            out = await execute_code(
                thought="run",
                code="print('hi')",
                state_mode="stateless",
                session_id=None,
                session_name=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()
        assert out.get("error") is None
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]

        mm = [
            e
            for e in events
            if e.payload.get("manager") == "CodeActActor"
            and e.payload.get("method") == "execute_code"
        ]
        assert sorted([e.payload.get("phase") for e in mm]) == ["incoming", "outgoing"]
        _h = mm[0].payload.get("hierarchy")
        assert (
            len(_h) == 2
            and _h[0] == "CodeActActor.act"
            and re.match(
                r"execute_code\([0-9a-f]{4}\)$",
                _h[1],
            )
        )
        assert "execute_code(" in str(mm[0].payload.get("hierarchy_label"))
    finally:
        TOOL_LOOP_LINEAGE.reset(token)


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_boundary_marks_error_when_executor_raises(monkeypatch):
    actor = CodeActActor(
        environments=[],
    )

    async def _boom(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(actor._session_executor, "execute", _boom, raising=True)

    execute_code = actor.get_tools("act")["execute_code"]
    async with capture_events("ManagerMethod") as events:
        out = await execute_code(
            thought="run",
            code="print('hi')",
            state_mode="stateless",
            session_id=None,
            session_name=None,
            _notification_up_q=None,
        )
    EVENT_BUS.join_published()

    mm = [
        e
        for e in events
        if e.payload.get("manager") == "CodeActActor"
        and e.payload.get("method") == "execute_code"
        and e.payload.get("phase") == "outgoing"
    ]
    assert len(mm) == 1
    assert mm[0].payload.get("status") == "error"
    assert "RuntimeError" in str(mm[0].payload.get("error_type"))
    assert out.get("error")


# ---------------------------------------------------------------------------
# Integration: execute_code + FunctionManager boundary + manager
# ---------------------------------------------------------------------------


@dataclass
class _ResultHandle:
    """Tiny handle with the minimum API used by these integration tests."""

    value: Any

    async def result(self) -> Any:
        return self.value


class UnitStateManager:
    """Minimal manager-like object that publishes ManagerMethod events via decorator."""

    @log_manager_call("UnitStateManager", "ask", payload_key="question")
    async def ask(self, question: str, *, _call_id: str | None = None):
        _ = _call_id
        return _ResultHandle(value=f"answer:{question}")


def _make_primitives() -> Any:
    return SimpleNamespace(unit=UnitStateManager())


@pytest.mark.asyncio
@_handle_project
async def test_execute_code_function_boundary_to_manager_includes_full_hierarchy():
    """Full hierarchy list across execute_code + FM boundary + manager."""
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = _make_primitives()

    async def send_meeting_invite():
        h = await sandbox.global_state["primitives"].unit.ask("invite")
        return await h.result()

    sandbox.global_state["send_meeting_invite"] = _LineageTrackedFunction(
        send_meeting_invite,
        "send_meeting_invite",
    )

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        code = "out = await send_meeting_invite()\nprint(out)\n"
        async with capture_events("ManagerMethod") as events:
            res = await execute_code(
                thought="run",
                code=code,
                state_mode="stateful",
                session_id=0,
                session_name=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()

        assert _result_error(res) is None

        ask_events = [
            e
            for e in events
            if e.payload.get("manager") == "UnitStateManager"
            and e.payload.get("method") == "ask"
            and e.payload.get("phase") in ("incoming", "outgoing")
        ]
        assert {e.payload.get("phase") for e in ask_events} == {"incoming", "outgoing"}
        _h = ask_events[0].payload.get("hierarchy")
        assert len(_h) == 4
        assert _h[0] == "CodeActActor.act"
        assert re.match(r"execute_code\([0-9a-f]{4}\)$", _h[1])
        assert re.match(r"send_meeting_invite\([0-9a-f]{4}\)$", _h[2])
        assert re.match(r"UnitStateManager\.ask\([0-9a-f]{4}\)$", _h[3])
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_concurrent_function_boundaries_do_not_cross_talk_lineage_or_calling_ids():
    """Concurrent sibling boundaries must not nest under each other.

    _LineageTrackedFunction manages TOOL_LOOP_LINEAGE (ContextVar), not
    event-bus events.  The observable proof that lineage didn't cross-talk
    is that the inner manager calls (UnitStateManager.ask) each carry a
    hierarchy containing only their own function boundary, not the sibling's.
    """
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = _make_primitives()

    async def f1():
        h = await sandbox.global_state["primitives"].unit.ask("one")
        await asyncio.sleep(0)  # encourage interleaving
        return await h.result()

    async def f2():
        h = await sandbox.global_state["primitives"].unit.ask("two")
        await asyncio.sleep(0)
        return await h.result()

    sandbox.global_state["f1"] = _LineageTrackedFunction(f1, "f1")
    sandbox.global_state["f2"] = _LineageTrackedFunction(f2, "f2")

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        code = "import asyncio\nres = await asyncio.gather(f1(), f2())\nprint(res)\n"
        async with capture_events("ManagerMethod") as events:
            res = await execute_code(
                thought="run",
                code=code,
                state_mode="stateful",
                session_id=0,
                session_name=None,
                _notification_up_q=None,
            )
        EVENT_BUS.join_published()
        assert _result_error(res) is None

        # Inner manager calls (UnitStateManager.ask) are the observable events.
        # Each should carry a hierarchy that includes its own function boundary
        # but NOT the sibling's.
        ask_events = [
            e
            for e in events
            if e.payload.get("manager") == "UnitStateManager"
            and e.payload.get("method") == "ask"
            and e.payload.get("phase") == "incoming"
        ]
        assert len(ask_events) == 2

        # Match the boundary segment itself: a call id is four hex digits and
        # can spell "f1" or "f2" by chance.
        def _has_boundary(hierarchy: list[str], name: str) -> bool:
            return any(seg.startswith(f"{name}(") for seg in hierarchy)

        hierarchies = [e.payload.get("hierarchy", []) for e in ask_events]
        f1_hierarchy = [h for h in hierarchies if _has_boundary(h, "f1")]
        f2_hierarchy = [h for h in hierarchies if _has_boundary(h, "f2")]
        assert len(f1_hierarchy) == 1, f"Expected one f1 hierarchy, got {f1_hierarchy}"
        assert len(f2_hierarchy) == 1, f"Expected one f2 hierarchy, got {f2_hierarchy}"

        # f1's hierarchy must NOT contain f2, and vice versa.
        assert not _has_boundary(
            f1_hierarchy[0],
            "f2",
        ), f"f1 hierarchy leaked f2: {f1_hierarchy[0]}"
        assert not _has_boundary(
            f2_hierarchy[0],
            "f1",
        ), f"f2 hierarchy leaked f1: {f2_hierarchy[0]}"
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)


@pytest.mark.asyncio
@_handle_project
async def test_function_boundary_error_restores_lineage_and_surfaces_error():
    """_LineageTrackedFunction must restore TOOL_LOOP_LINEAGE when the
    wrapped function raises, and the error must surface in execute_code's result.

    _LineageTrackedFunction manages lineage (ContextVar) only — it does not
    publish ManagerMethod events to the event bus.  The execute_code boundary
    captures the exception and reports it in the result dict.
    """
    actor = CodeActActor(environments=[])
    execute_code = actor.get_tools("act")["execute_code"]

    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = _make_primitives()

    async def boom():
        raise RuntimeError("boom")

    sandbox.global_state["boom"] = _LineageTrackedFunction(boom, "boom")

    sb_token = _CURRENT_SANDBOX.set(sandbox)
    lineage_token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act"])
    try:
        res = await execute_code(
            thought="run",
            code="await boom()\n",
            state_mode="stateful",
            session_id=0,
            session_name=None,
            _notification_up_q=None,
        )

        # The error should be captured in the result.
        assert _result_error(res)
        assert "RuntimeError" in str(_result_error(res))

        # Lineage must be restored to the pre-call state.
        assert TOOL_LOOP_LINEAGE.get([]) == ["CodeActActor.act"]
    finally:
        TOOL_LOOP_LINEAGE.reset(lineage_token)
        _CURRENT_SANDBOX.reset(sb_token)
