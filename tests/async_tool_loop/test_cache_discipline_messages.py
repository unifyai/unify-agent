"""Symbolic: ``UNIFY_CACHE_DISCIPLINE`` never edits a message once it was sent.

Every actor conversation on the pre-rebase build broke its cached prefix on
its last call: every earlier assistant message lost its
``provider_specific_fields`` and reasoning details, shed when a persistent
session parked, and a storage review's compaction note shortened the tool
results it covered. With the switch on both are skipped, so each request is
the previous request plus what came after it.

With the switch off the same scripts are compared to the upstream bytes by
``test_cache_discipline_tools.py``.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS


def _as_bytes(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(h.SCENARIOS))
async def test_on_each_request_extends_the_previous_one(monkeypatch, scenario):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    for before, after in zip(requests, requests[1:]):
        sent = _as_bytes(before["messages"])
        assert _as_bytes(after["messages"])[: len(sent)] == sent


@pytest.mark.asyncio
async def test_on_a_parked_session_keeps_reasoning_and_reviewed_results(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
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


@pytest.mark.asyncio
async def test_off_the_parked_session_is_shed_and_compacted_as_upstream(monkeypatch):
    """The behaviour the switch removes, pinned so the contrast stays visible."""
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False)
    _result, _counter, requests = await h.scenario_persist()
    last = requests[-1]["messages"]
    first_assistant = next(m for m in last if m["role"] == "assistant")
    assert "provider_specific_fields" not in first_assistant
    assert "reasoning_details" not in first_assistant
    tool = next(m for m in last if m["role"] == "tool")
    assert "[compacted after skill review:" in tool["content"]
    sent = _as_bytes(requests[1]["messages"])
    assert _as_bytes(last)[: len(sent)] != sent
