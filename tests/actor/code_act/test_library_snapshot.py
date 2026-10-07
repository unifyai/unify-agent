"""The library snapshot: the session's first user message gives the library's size.

The line counts the stored functions (primitives excluded) and the guidance
entries at task start, once, in the message that opens the session. The
discovery-first gate reads the same counts each time it is evaluated and
treats an empty library as already searched. Requests are captured at
unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

SNAPSHOT = "Library at task start:"
TASK = "List the files in the workspace."


@pytest.fixture
def switches(monkeypatch):
    def set_(*, discipline: bool = False):
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)

    return set_


async def _act(*, seed=None) -> tuple[list[dict], "caa.CodeActActor"]:
    """One scripted ``act()`` on a fresh actor; *seed* fills its libraries first."""
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act(TASK, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests), actor


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


# ── the gate ─────────────────────────────────────────────────────────────


def test_the_gate_reads_the_counts_each_time_it_is_evaluated():
    counts = [(0, 0)]
    tools = {
        "FunctionManager_search_functions": object(),
        "GuidanceManager_search": object(),
        "execute_code": object(),
    }
    policy = caa._default_tool_policy(
        True,
        True,
        dict,
        library_counts=lambda: counts[0],
    )
    assert policy(0, tools, []) == ("auto", tools)
    counts[0] = (2, 0)
    mode, gated, _opts = policy(1, tools, [])
    assert mode == "required" and list(gated) == ["FunctionManager_search_functions"]
    # An unknown count keeps the gate as shipped.
    counts[0] = (None, None)
    mode, gated, _opts = policy(2, tools, [])
    assert mode == "required" and set(gated) == set(tools) - {"execute_code"}


def test_builtin_guidance_is_not_counted():
    from unify.guidance_manager.guidance_manager import GuidanceManager

    gm = GuidanceManager()
    assert caa._library_counts(None, gm) == (None, 0)


# ── the snapshot line ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_snapshot_is_in_the_first_user_message_only(switches):
    switches(discipline=True)
    on, _ = await _act()
    line = f"{SNAPSHOT} 0 stored functions, 0 guidance entries."
    first = _first_user(on[0])
    assert first.startswith(f"{line}\n\n")
    assert first.endswith(f"\n\n---\n\n{TASK}")
    for request in on:
        holders = [
            m
            for m in request["messages"]
            if SNAPSHOT in json.dumps(m.get("content"), default=str)
        ]
        # Once, in the message that opens the session, never edited.
        assert [m["role"] for m in holders] == ["user"]
        assert holders[0]["content"] == _first_user(on[0])
