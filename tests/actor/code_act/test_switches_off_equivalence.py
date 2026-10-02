"""Symbolic: with every switch the merged lanes add off, requests are upstream's.

Each lane checked its own switch-off path; this checks them together. The
actor's first request (the discovery gate's tools, its system prompt and
the task) and the full tool list it advertises after the gate were recorded
from a scripted ``act()`` on the upstream commit
(``tests/actor_switches_off_golden.json``); the cache-discipline scenarios
were recorded there too (``tests/cache_discipline_golden.json``). Requests
are captured at unillm's transport (``tests/cache_discipline_helpers.py``),
so nothing leaves the process.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

# Every switch the lanes add, at its off value.
NEW_SWITCHES = {
    "UNIFY_FUNCTION_PATCH": False,
    "UNIFY_STORE_DEDUPE": "",
    "UNIFY_STORE_TRUST": "",
    "UNIFY_CACHE_DISCIPLINE": False,
    "UNIFY_REVIEW_FORK": False,
    "UNIFY_BUILTIN_GUIDANCE": True,
    "UNIFY_TRANSCRIPTS": False,
    "UNIFY_WORKSPACE": "",
    "UNIFY_WORKSPACE_NETWORK": "",
}


@pytest.fixture
def all_off(monkeypatch):
    for name, value in NEW_SWITCHES.items():
        assert hasattr(SETTINGS, name), name
        monkeypatch.setattr(SETTINGS, name, value)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_actors_first_request_and_tools_are_upstreams(all_off):
    golden = json.loads(h.ACTOR_GOLDEN.read_text())
    _result, _, requests = await h.scenario_actor()
    recorded = h.actor_recording(requests)
    assert recorded["advertised_tools"] == golden["advertised_tools"]
    assert recorded["first_request"]["tools"] == golden["first_request"]["tools"]
    assert recorded["first_request"]["messages"] == golden["first_request"]["messages"]
    assert recorded == golden


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(h.SCENARIOS))
async def test_every_scenario_request_is_upstreams(all_off, scenario):
    golden = json.loads(h.GOLDEN.read_text())[scenario]
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    assert [h.request_bytes(r) for r in requests] == golden
