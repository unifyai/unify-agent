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
    "UNIFY_FUNCTION_CASES": False,
    "UNIFY_INLINE_CURATION": "",
    "UNIFY_STORE_DEDUPE": "",
    "UNIFY_STORE_TRUST": "",
    "UNIFY_REPEAT_GUARD": False,
    "UNIFY_TRY_FIRST": False,
    "UNIFY_CACHE_DISCIPLINE": False,
    "UNIFY_LIBRARY_SNAPSHOT": False,
    "UNIFY_PROMPT_CLOCK": "",
    "UNIFY_REVIEW_FORK": False,
    "UNIFY_BUILTIN_GUIDANCE": True,
    "UNIFY_REVIEW_FRAMING": "",
    "UNIFY_CURATION_DOCTRINE": "",
    "UNIFY_REPLY_PROTOCOL_NOTE": False,
    "UNIFY_CODE_FIRST": False,
    "UNIFY_STORE_INSTANCE_LINT": False,
    "UNIFY_WAIT_FOR_BATCH": False,
    "UNIFY_WAIT_CEILING_SECONDS": 15.0,
    "UNIFY_TRANSCRIPTS": False,
    "UNIFY_REVIEW_REASONING_EFFORT": "",
    "UNIFY_REVIEW_MODEL": "",
    "UNIFY_WORKSPACE": "",
    "UNIFY_WORKSPACE_NETWORK": "",
    "UNIFY_WORKSPACE_PYTHON": "",
    "UNIFY_WORKSPACE_PROXY_PORT": 0,
    "UNIFY_CACHE_AFFINITY_SCOPE": "prefix",
    "UNIFY_OUTCOME": False,
    "UNIFY_REVIEW_FAILED": "",
}

# The UNIFY_ settings of the commit the actor golden was recorded on
# (35c8633c7); every other one is a lane's and belongs in NEW_SWITCHES.
UPSTREAM_SETTINGS = frozenset(
    {
        "UNIFY_BUILTINS_PROJECT",
        "UNIFY_EMBED_URL",
        "UNIFY_ENV_NAMESPACES",
        "UNIFY_GUIDANCE_EMPTY_QUERY",
        "UNIFY_LOCAL_EMBEDDINGS",
        "UNIFY_LOCAL_ROOT",
        "UNIFY_LOG_DIR",
        "UNIFY_MAX_OUTPUT_TOKENS",
        "UNIFY_MAX_TOOL_LOOP_STEPS",
        "UNIFY_MODEL",
        "UNIFY_REASONING_EFFORT",
        "UNIFY_SEARCH_SKIP_UNLOADABLE",
        "UNIFY_STORE_ADMISSION",
        "UNIFY_STORE_CHECK",
        "UNIFY_STORE_VERIFY",
        "UNIFY_TERMINAL_LOG",
        "UNIFY_TERMINAL_LOG_LEVEL",
        "UNIFY_TOOL_CHOICE_FALLBACK",
        "UNIFY_TURN_STORAGE_REVIEWS",
        "UNIFY_VALIDATE_LLM_PROVIDERS",
    },
)


def test_every_lane_switch_is_here_at_its_default():
    from unify.settings import ProductionSettings

    fields = ProductionSettings.model_fields
    added = {name for name in fields if name.startswith("UNIFY_")} - UPSTREAM_SETTINGS
    assert sorted(added - set(NEW_SWITCHES)) == []
    assert {
        name: (fields[name].default, value)
        for name, value in NEW_SWITCHES.items()
        if fields[name].default != value
    } == {}


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
