"""
tests/async_tool_loop/test_ask_inspection_effort.py
===================================================

``handle.ask()`` inspects a loop with a client built for the inspected loop's
model. A client named for a model gets ``reasoning_effort="high"`` unless one
is set, so inspections of a loop configured at ``low`` ran at ``high``. The
inspection now runs at the effort of the loop it inspects.
"""

from __future__ import annotations

import asyncio

import pytest

from unify.common import async_tool_loop as atl
from unify.common.llm_client import new_llm_client


class _DummyLoopClient:
    endpoint = "openai/gpt-4o-mini@openrouter"

    def __init__(self, effort) -> None:
        self.reasoning_effort = effort
        self.messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "working"},
        ]


def _make_handle(client) -> atl.AsyncToolLoopHandle:
    async def _outer_done() -> str:
        return "outer-complete"

    task = asyncio.create_task(_outer_done(), name="InspectionEffortDummyTask")
    setattr(task, "get_ask_tools", lambda: {})
    return atl.AsyncToolLoopHandle(
        task=task,
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
        client=client,
        loop_id="InspectionEffortLoop",
    )


class _DummyInspectionHandle:
    async def result(self):
        return "inspection-complete"


async def _inspection_client(monkeypatch, effort):
    captured: dict = {}

    def _fake_start(*args, **kwargs):
        captured["client"] = args[0]
        return _DummyInspectionHandle()

    monkeypatch.setattr(atl, "start_async_tool_loop", _fake_start)
    handle = _make_handle(_DummyLoopClient(effort))
    helper = await handle.ask("What is it doing?")
    await helper.result()
    return captured["client"]


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("effort", ["low", "medium", "none"])
async def test_inspection_runs_at_the_inspected_loops_effort(monkeypatch, effort):
    client = await _inspection_client(monkeypatch, effort)
    assert client.reasoning_effort == effort


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("effort", [None, ""])
async def test_inspection_of_a_loop_without_an_effort_keeps_the_default(
    monkeypatch,
    effort,
):
    client = await _inspection_client(monkeypatch, effort)
    default = new_llm_client("openai/gpt-4o-mini@openrouter").reasoning_effort
    assert client.reasoning_effort == default
