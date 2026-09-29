"""Symbolic: under ``UNIFY_CACHE_DISCIPLINE`` a session keeps one cache and measures it.

A provider prefix is only reused when the next request reaches the replica
holding it, so a session asks for one with a cache affinity key -- where the
installed unillm offers the key (the ``harness-cache`` branch does; ``main``
does not, and the client is then left alone). Each call logs how much of its
input came from the cache, which is how the 0% of the storage review's first
call and the losses at the second and last actor calls were found.
"""

from __future__ import annotations

import pytest
import unillm

from tests import cache_discipline_helpers as h
from unify.common._async_tool import cache_discipline as cd
from unify.settings import SETTINGS


@pytest.fixture
def affinity_client_class(monkeypatch):
    """Give unillm's async client the ``cache_affinity`` API of harness-cache."""

    def set_cache_affinity(self, value):
        self._cache_affinity_key = value
        return self

    monkeypatch.setattr(
        unillm.AsyncUnify,
        "set_cache_affinity",
        set_cache_affinity,
        raising=False,
    )
    monkeypatch.setattr(
        unillm.AsyncUnify,
        "cache_affinity",
        property(lambda self: getattr(self, "_cache_affinity_key", None)),
        raising=False,
    )


async def _session(client, replies=h.INTERRUPT_REPLIES):
    counter: dict = {}
    tools = h.make_tools(counter)
    with h.scripted(replies) as provider:
        result = await h._run(
            client,
            {"execute_code": tools["execute_code"]},
            "Do the task.",
        )
    return result, provider.requests


# ── cache affinity ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_each_session_gets_its_own_affinity_key(
    monkeypatch,
    affinity_client_class,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    first, second = h.new_client(), h.new_client()
    await _session(first)
    await _session(second)
    assert isinstance(first.cache_affinity, str) and len(first.cache_affinity) == 32
    assert second.cache_affinity != first.cache_affinity


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


@pytest.mark.asyncio
async def test_on_a_unillm_without_the_key_is_left_alone(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
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
