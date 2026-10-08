"""Symbolic: the loop never edits a message once it was sent (the cache discipline).

Every actor conversation on the pre-rebase build broke its cached prefix on
its last call: every earlier assistant message lost its
``provider_specific_fields`` and reasoning details, shed when a persistent
session parked, and a storage review's compaction note shortened the tool
results it covered. Both are gone, so each request is the previous request
plus what came after it.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h


def _as_bytes(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", h.ONE_SESSION)
async def test_on_each_request_extends_the_previous_one(monkeypatch, scenario):
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    if scenario == "compress":
        # The session restarts from its summary after the third request; the
        # restart is checked in test_cache_discipline_compression.py.
        requests = requests[:3]
    for before, after in zip(requests, requests[1:]):
        sent = _as_bytes(before["messages"])
        assert _as_bytes(after["messages"])[: len(sent)] == sent


@pytest.mark.asyncio
async def test_on_a_parked_session_keeps_reasoning_and_reviewed_results(monkeypatch):
    result, _counter, requests = await h.scenario_persist()
    assert result == "first done|second done"
    last = requests[-1]["messages"]
    assistants = [m for m in last if m["role"] == "assistant"]
    assert assistants[0]["reasoning_details"][0]["type"] == "reasoning.encrypted"
    assert assistants[0]["provider_specific_fields"] == {
        "reasoning_signature": "thinking about the first request",
    }
    assert assistants[1]["provider_specific_fields"] == {
        "reasoning_signature": "wrapping up",
    }
    tool = next(m for m in last if m["role"] == "tool")
    assert tool["content"] == "ran " + "x" * 1200
