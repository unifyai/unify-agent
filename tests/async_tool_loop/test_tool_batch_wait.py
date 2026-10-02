"""Symbolic: ``UNIFY_TOOL_BATCH_WAIT`` holds the model's turn for the rest of a batch.

When a turn calls two tools at once and the first returns, the loop starts
the model's next turn straight away; when the second lands during it, that
turn is cancelled and asked again. The provider has already billed the
cancelled step: charged cancelled calls were 16.8% of known AppWorld cost
and 24.8% of ScienceWorld's (2 Oct analysis), mostly a fast library search
returning a moment before its sibling. With the switch the loop waits up to
N seconds for the siblings first. The transport is scripted and slowed, so
the race is deterministic and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import ProductionSettings, SETTINGS

LLM_SECONDS = 0.6


async def fast_tool() -> str:
    """Return quickly."""
    await asyncio.sleep(0.05)
    return "FAST_RESULT"


async def medium_tool() -> str:
    """Return after the fast tool, while a model turn started then is in flight."""
    await asyncio.sleep(0.4)
    return "MEDIUM_RESULT"


async def slow_tool() -> str:
    """Return long after any batch window."""
    await asyncio.sleep(3.0)
    return "SLOW_RESULT"


def _replies(second: str):
    return [
        lambda: h.completion(calls=[("fast_tool", {}), (second, {})]),
        *[lambda: h.completion(content="done")] * 6,
    ]


async def _run(monkeypatch, wait_s: float, second: str) -> list[dict]:
    import unillm.clients.uni_llm as uni_llm

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_BATCH_WAIT", wait_s)
    tools = {
        "fast_tool": fast_tool,
        second: {"medium_tool": medium_tool, "slow_tool": slow_tool}[second],
    }
    with h.scripted(_replies(second)) as provider:
        scripted = uni_llm._acompletion_with_transient_retry

        async def slowed(**kw):
            # The first request answers at once; later ones take a while, like
            # a model thinking, so a sibling can land during them.
            if provider.requests:
                await asyncio.sleep(LLM_SECONDS)
            return await scripted(**kw)

        uni_llm._acompletion_with_transient_retry = slowed
        result = await h._run(h.new_client(), tools, "Run both tools.")
    assert result == "done"
    return provider.requests


def _results_seen(request: dict) -> set[str]:
    return {
        str(m.get("content"))
        for m in request["messages"]
        if m.get("role") == "tool" and "RESULT" in str(m.get("content"))
    }


@pytest.mark.asyncio
async def test_off_a_turn_starts_on_the_first_result_and_is_asked_again(monkeypatch):
    requests = await _run(monkeypatch, 0.0, "medium_tool")
    seen = [_results_seen(r) for r in requests[1:]]
    # A turn was sent with only the fast result, then again with both.
    assert any(len(s) == 1 for s in seen)
    assert any(len(s) == 2 for s in seen)
    assert len(requests) >= 3


@pytest.mark.asyncio
async def test_on_the_turn_waits_for_the_batch(monkeypatch):
    requests = await _run(monkeypatch, 2.0, "medium_tool")
    assert len(requests) == 2
    seen = _results_seen(requests[1])
    assert len(seen) == 2, json.dumps(requests[1]["messages"], default=str)[:2000]


@pytest.mark.asyncio
async def test_on_a_sibling_slower_than_the_window_is_raced_as_shipped(monkeypatch):
    requests = await _run(monkeypatch, 0.3, "slow_tool")
    seen = [_results_seen(r) for r in requests[1:]]
    # The window closed with the slow tool still running: a turn went out
    # with only the fast result, as without the switch.
    assert seen and len(seen[0]) == 1


def test_the_setting_is_bounded():
    assert ProductionSettings().UNIFY_TOOL_BATCH_WAIT == 0.0
    assert ProductionSettings(UNIFY_TOOL_BATCH_WAIT="2").UNIFY_TOOL_BATCH_WAIT == 2.0
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_TOOL_BATCH_WAIT="-1")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_TOOL_BATCH_WAIT="31")
