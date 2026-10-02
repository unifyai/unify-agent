"""Symbolic: under ``UNIFY_CACHE_DISCIPLINE`` a session keeps one cache and measures it.

A provider prefix is only reused when the next request reaches the replica
holding it, so a session asks for one with a cache affinity key -- where the
installed unillm offers the key (the ``harness-cache`` branch does; ``main``
does not, and the client is then left alone). Each call logs how much of its
input came from the cache, which is how the 0% of the storage review's first
call and the losses at the second and last actor calls were found.

The key is shared by every session with the same model, system prompt and
tool list (``UNIFY_CACHE_AFFINITY_SCOPE=prefix``, the default), so a new
session reaches the replica holding the prefix an earlier one cached.
"""

from __future__ import annotations

import pytest

from tests import cache_discipline_helpers as h
from unify.common._async_tool import cache_discipline as cd
from unify.settings import SETTINGS


@pytest.fixture
def affinity_client_class(monkeypatch):
    """Record the ``cache_affinity`` keys set, with or without unillm's own API."""
    return h.install_affinity_api(monkeypatch)


@pytest.fixture
def scope(monkeypatch):
    def set_(value: str) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_AFFINITY_SCOPE", value)

    return set_


async def _session(client, replies=h.INTERRUPT_REPLIES, tools=None):
    counter: dict = {}
    made = h.make_tools(counter)
    with h.scripted(replies) as provider:
        result = await h._run(
            client,
            tools or {"execute_code": made["execute_code"]},
            "Do the task.",
        )
    return result, provider.requests


# ── cache affinity ───────────────────────────────────────────────────────
#
# With a key per session, the first call of every session in four measured
# cells read nothing from the cache (12/12, 12/12, 56/56 and 57/57), even
# where 54 of 57 sessions sent the same system prompt and tools: each new
# key sent the session to a replica that had never seen the prefix.


@pytest.mark.asyncio
async def test_on_sessions_with_the_same_prefix_share_one_key(
    monkeypatch,
    affinity_client_class,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    assert SETTINGS.UNIFY_CACHE_AFFINITY_SCOPE == "prefix"  # the default
    first, second = h.new_client(), h.new_client()
    _, first_requests = await _session(first)
    _, second_requests = await _session(second)
    assert isinstance(first.cache_affinity, str) and len(first.cache_affinity) == 32
    assert second.cache_affinity == first.cache_affinity
    # It is the hash of what both sessions' requests start with.
    assert first.cache_affinity == cd.prefix_affinity_key(
        h.MODEL,
        "You are a scripted test agent.",
        first_requests[0]["tools"],
    )
    assert h.request_bytes(first_requests[0])["tools"] == (
        h.request_bytes(second_requests[0])["tools"]
    )
    # Set once per session, before its first request.
    assert affinity_client_class == [
        (first.cache_affinity, 0),
        (first.cache_affinity, 0),
    ]


@pytest.mark.asyncio
async def test_on_a_different_system_prompt_or_tool_list_gets_another_key(
    monkeypatch,
    affinity_client_class,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    base, other_prompt, other_tools = (
        h.new_client(),
        h.new_client("You are another scripted agent."),
        h.new_client(),
    )
    await _session(base)
    await _session(other_prompt)
    tools = h.make_tools({})
    await _session(
        other_tools,
        tools={
            "execute_code": tools["execute_code"],
            "GuidanceManager_search": tools["GuidanceManager_search"],
        },
    )
    keys = {
        base.cache_affinity,
        other_prompt.cache_affinity,
        other_tools.cache_affinity,
    }
    assert len(keys) == 3 and None not in keys


def test_the_prefix_key_is_canonical_and_covers_model_prompt_and_tools():
    tools = [{"type": "function", "function": {"name": "t", "parameters": {}}}]
    key = cd.prefix_affinity_key("m@p", "sys", tools)
    assert len(key) == 32 and int(key, 16) >= 0
    # Dict key order does not matter; list order and every field do.
    reordered = [{"function": {"parameters": {}, "name": "t"}, "type": "function"}]
    assert cd.prefix_affinity_key("m@p", "sys", reordered) == key
    assert cd.prefix_affinity_key("n@p", "sys", tools) != key
    assert cd.prefix_affinity_key("m@p", "sys.", tools) != key
    assert cd.prefix_affinity_key("m@p", "sys", tools * 2) != key
    assert cd.prefix_affinity_key("m@p", "sys", None) == (
        cd.prefix_affinity_key("m@p", "sys", [])
    )


@pytest.mark.asyncio
async def test_on_the_session_scope_gives_each_session_its_own_key(
    monkeypatch,
    affinity_client_class,
    scope,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    scope("session")
    first, second = h.new_client(), h.new_client()
    await _session(first)
    await _session(second)
    assert isinstance(first.cache_affinity, str) and len(first.cache_affinity) == 32
    assert second.cache_affinity != first.cache_affinity


@pytest.mark.asyncio
async def test_on_the_run_scope_every_session_shares_the_process_key(
    monkeypatch,
    affinity_client_class,
    scope,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    scope("run")
    first, second = h.new_client(), h.new_client("A different prompt.")
    await _session(first)
    await _session(second)
    assert first.cache_affinity == second.cache_affinity == cd.run_affinity_key()
    assert len(first.cache_affinity) == 32


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [("", "prefix"), (None, "prefix"), (" Session ", "session"), ("RUN", "run")],
)
def test_the_scope_setting_parses(raw, parsed):
    from unify.settings import ProductionSettings

    assert ProductionSettings.parse_cache_affinity_scope(raw) == parsed


def test_the_scope_setting_refuses_anything_else():
    from unify.settings import ProductionSettings

    with pytest.raises(ValueError, match="UNIFY_CACHE_AFFINITY_SCOPE"):
        ProductionSettings.parse_cache_affinity_scope("task")


@pytest.mark.asyncio
async def test_on_a_key_already_set_is_kept_and_a_fork_shares_it(
    monkeypatch,
    affinity_client_class,
):
    from unify.common.llm_client import fork_llm_client

    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    client = h.new_client()
    client.set_cache_affinity("parent-key")
    await _session(client)
    assert client.cache_affinity == "parent-key"
    assert fork_llm_client(client, origin="StorageCheck").cache_affinity == (
        "parent-key"
    )


@pytest.mark.asyncio
async def test_off_no_key_is_set_even_where_unillm_takes_one(
    monkeypatch,
    affinity_client_class,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False)
    client = h.new_client()
    await _session(client)
    assert client.cache_affinity is None
    assert affinity_client_class == []


@pytest.mark.asyncio
async def test_on_a_unillm_without_the_key_is_left_alone(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    h.hide_affinity_api(monkeypatch)  # as on unillm main, whichever is installed
    client = h.new_client()
    assert not hasattr(client, "set_cache_affinity")
    result, _requests = await _session(client)
    assert result == "done"
    assert cd.ensure_cache_affinity(client) is None
    assert not hasattr(client, "cache_affinity")


# ── the cache-hit metric ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_each_call_logs_its_cache_share_and_the_sessions(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    lines: list[str] = []
    monkeypatch.setattr(cd.LOGGER, "info", lambda msg, *a, **k: lines.append(msg))
    replies = (
        lambda: h.completion(
            calls=[("execute_code", {"code": "one"})],
            prompt_tokens=1000,
            cached_tokens=0,
        ),
        lambda: h.completion(content="done", prompt_tokens=1200, cached_tokens=960),
    )
    client = h.new_client()
    await _session(client, replies)
    assert cd.cache_stats(client) == {
        "calls": 2,
        "input_tokens": 2200,
        "cached_tokens": 960,
        "unknown": 0,
    }
    cache_lines = [line for line in lines if "cache:" in line]
    assert len(cache_lines) == 2
    assert "0/1000 input tokens cached (0.0%)" in cache_lines[0]
    assert "960/1200 input tokens cached (80.0%)" in cache_lines[1]
    assert "session 960/2200 (43.6%) over 2 call(s)" in cache_lines[1]


@pytest.mark.asyncio
async def test_on_an_unreported_cache_count_stays_unknown(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    lines: list[str] = []
    monkeypatch.setattr(cd.LOGGER, "info", lambda msg, *a, **k: lines.append(msg))
    replies = (
        lambda: h.completion(
            calls=[("execute_code", {"code": "one"})],
            prompt_tokens=1000,
            cached_tokens=900,
        ),
        # no prompt_tokens_details at all
        lambda: h.completion(content="done", prompt_tokens=1100),
    )
    client = h.new_client()
    await _session(client, replies)
    assert cd.cache_stats(client) == {
        "calls": 2,
        "input_tokens": 1000,
        "cached_tokens": 900,
        "unknown": 1,
    }
    last = [line for line in lines if "cache:" in line][-1]
    assert "cached tokens not reported" in last
    assert "over 1 call(s), 1 unreported" in last


@pytest.mark.asyncio
async def test_off_nothing_is_measured(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", False)
    client = h.new_client()
    await _session(client)
    assert cd.cache_stats(client) is None


def test_cache_usage_reads_objects_and_dicts():
    assert cd.cache_usage(h.completion(content="x", prompt_tokens=7)) == (7, None)
    assert cd.cache_usage(
        h.completion(content="x", prompt_tokens=7, cached_tokens=3),
    ) == (7, 3)
    assert cd.cache_usage(
        {"usage": {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 5}}},
    ) == (5, 5)
    assert cd.cache_usage(None) == (None, None)
