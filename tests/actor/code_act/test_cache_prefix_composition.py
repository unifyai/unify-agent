"""Symbolic: the cache switches together keep every request a reusable prefix.

On the 4-5 Oct ARC LOW screen the first call of 20-24 of each run's 24 later
episodes read 0 cached tokens: the actor's system prompt carries the clock
(minute resolution), so each new minute was a new system message and, under
the ``prefix`` affinity scope, a new cache key. The review gate and, under
the core surface, the review were new conversations. This checks that
``UNIFY_PROMPT_CLOCK=message``, ``UNIFY_CACHE_AFFINITY_SCOPE=static``,
``UNIFY_REVIEW_FORK``, ``UNIFY_REVIEW_GATE_FORK`` and (core)
``UNIFY_REVIEW_FORK_CORE`` compose with the lean prompt profile and the core
tool surface: the system prompt, tools and affinity key are the same for
sessions within one minute and across minutes, the clock opens the first user
message, and the gate's and the review's first requests start with the
session's last request byte for byte.

Requests are captured at unillm's transport (tests/cache_discipline_helpers.py);
core-surface cells run in the sandboxed worker (skipped without bubblewrap).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import review_gate
from unify.settings import SETTINGS

CLOCK = "### Current Time"
TASK = "Double four."
LEAN_CACHE = {
    # the lean screen arm's prompt switches
    "UNIFY_PROMPT_PROFILE": "lean",
    "UNIFY_PROMPT_ACCURACY": True,
    "UNIFY_DELEGATION": "off",
    "UNIFY_DISCOVERY_GATE": False,
    "UNIFY_CURATION_DOCTRINE": "minimal",
    "UNIFY_REVIEW_FRAMING": "unified",
    "UNIFY_REVIEW_GATE": True,
    # the cache switches
    "UNIFY_CACHE_DISCIPLINE": True,
    "UNIFY_REVIEW_FORK": True,
    "UNIFY_PROMPT_CLOCK": "message",
    "UNIFY_CACHE_AFFINITY_SCOPE": "static",
    "UNIFY_REVIEW_GATE_FORK": True,
    "UNIFY_REVIEW_FORK_CORE": True,
}


@pytest.fixture
def lean_cache(monkeypatch):
    for name, value in LEAN_CACHE.items():
        monkeypatch.setattr(SETTINGS, name, value)
    # A library that is not empty, so the gate is asked.
    monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))


def _at_minute(monkeypatch, minute: int) -> None:
    from unify.common import prompt_helpers

    stamp = f"Friday, June 13, 2025 at 12:{minute:02d} PM UTC"
    monkeypatch.setattr(prompt_helpers, "now", lambda *a, **k: stamp)


def _last_user(request: dict) -> str:
    content = request["messages"][-1].get("content")
    return content if isinstance(content, str) else ""


def _kind(request: dict) -> str:
    if _last_user(request).startswith("## Library Review Gate"):
        return "gate"
    if any(
        m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(("## Curating The Library", "## Storage Review"))
        for m in request["messages"]
    ):
        return "review"
    system = request["messages"][0]["content"]
    if system == review_gate.GATE_SYSTEM_PROMPT or system.startswith(
        ("You are a skill librarian", "You are the agent that just completed"),
    ):
        return "standalone"
    return "session"


async def _episode(actor) -> tuple[list[dict], list[tuple[str, int]]]:
    """One session through ``act``, its gate and its review."""
    session = [
        lambda: h.completion(
            calls=[("execute_code", {"thought": "Compute.", "code": "print(4 * 2)"})],
        ),
        lambda: h.completion(content="8"),
    ]
    with h.scripted([]) as provider:

        def _reply():
            kind = _kind(provider.requests[-1])
            if kind == "gate":
                return h.completion(content='{"review": true, "reason": "code ran"}')
            if kind == "review":
                return h.completion(content="Nothing worth storing.")
            return session.pop(0)()

        provider.replies = [_reply] * 20
        handle = await actor.act(TASK, persist=False)
        await asyncio.wait_for(handle.result(), 120)
        await asyncio.wait_for(handle._completion_event.wait(), 120)
    return provider.requests


def _dumps(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


def _check_episode(requests: list[dict]) -> dict:
    kinds = [_kind(r) for r in requests]
    assert kinds == ["session", "session", "gate", "review"], kinds
    first, last, gate, review = requests
    system = first["messages"][0]
    assert system["role"] == "system" and CLOCK not in system["content"]
    first_user = next(m["content"] for m in first["messages"] if m["role"] == "user")
    assert first_user.startswith(CLOCK)
    n = len(last["messages"])
    for fork in (gate, review):
        assert _dumps(fork["messages"])[:n] == _dumps(last["messages"])
        assert h.request_bytes(fork)["tools"] == h.request_bytes(last)["tools"]
        assert fork["tool_choice"] == last["tool_choice"]
    # The review forks the session, not the gate's exchange.
    assert _dumps(review["messages"])[: n + 1] == _dumps(gate["messages"])[: n + 1]
    return {"system": system, "tools": h.request_bytes(first)["tools"]}


async def _three_episodes(actor_factory, monkeypatch) -> list[dict]:
    sets = h.install_affinity_api(monkeypatch)
    out = []
    for minute in (1, 1, 2):
        _at_minute(monkeypatch, minute)
        start = len(sets)
        actor = actor_factory()
        try:
            requests = await _episode(actor)
        finally:
            await actor.close()
        row = _check_episode(requests)
        key, sent = sets[start]
        assert sent == 0
        row["key"] = key
        out.append(row)
    return out


def _assert_shared(rows: list[dict]) -> None:
    # Within one minute (episodes 1, 2) and across a minute (episode 3).
    assert rows[0]["system"] == rows[1]["system"] == rows[2]["system"]
    assert rows[0]["tools"] == rows[1]["tools"] == rows[2]["tools"]
    assert rows[0]["key"] == rows[1]["key"] == rows[2]["key"]


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_lean_sessions_share_one_prefix_and_their_gate_and_review_fork(
    lean_cache,
    monkeypatch,
):
    from unify.actor.code_act_actor import CodeActActor

    rows = await _three_episodes(CodeActActor, monkeypatch)
    _assert_shared(rows)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
@_handle_project
async def test_core_sessions_share_one_prefix_and_their_gate_and_review_fork(
    core_world,
    lean_cache,
    monkeypatch,
):
    rows = await _three_episodes(new_actor, monkeypatch)
    _assert_shared(rows)
    assert json.loads(rows[0]["tools"])[0]["function"]["name"] == "execute_code"


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_with_the_clock_in_the_system_prompt_minutes_differ(
    lean_cache,
    monkeypatch,
):
    """The control: the clock in the system prompt splits the prefix by minute."""
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_CLOCK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_AFFINITY_SCOPE", "prefix")
    sets = h.install_affinity_api(monkeypatch)
    systems, keys = [], []
    for minute in (1, 2):
        _at_minute(monkeypatch, minute)
        start = len(sets)
        actor = CodeActActor()
        try:
            requests = await _episode(actor)
        finally:
            await actor.close()
        systems.append(requests[0]["messages"][0]["content"])
        keys.append(sets[start][0])
    assert CLOCK in systems[0] and systems[0] != systems[1]
    assert keys[0] != keys[1]
