"""Symbolic: ``UNIFY_LIBRARY_SNAPSHOT``: an empty library is not searched first.

The discovery-first gate forces ``tool_choice="required"`` library searches
on the first turn whatever the libraries hold: on ScienceWorld 224 of 1,318
function searches that returned nothing were forced by it, and the model
was never told how large the libraries were. With the switch on the gate
reads the stored function count (primitives excluded) and the guidance
count each time it is evaluated, and treats a family whose library is
empty as already searched; the session's first user message says how large
both were at task start. Only the tool choice changes: under
``UNIFY_CACHE_DISCIPLINE`` the tool list sent is byte for byte the same.
Requests are captured at unillm's transport, so nothing leaves the process.
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
    def set_(*, snapshot: bool, discipline: bool = False, builtins: bool = False):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", snapshot)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", builtins)

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


def _add_function(actor) -> None:
    actor.function_manager.add_functions(
        implementations=[
            'def list_names(path):\n    """List the names in a directory."""\n'
            "    import os\n    return sorted(os.listdir(path))",
        ],
    )


def _add_guidance(actor) -> None:
    actor.guidance_manager.add_guidance(
        title="Listing files",
        content="List a directory with os.listdir and sort the names.",
    )


# ── the gate ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_empty_libraries_the_first_turn_is_auto_with_the_same_tools(
    switches,
):
    switches(snapshot=False, discipline=True)
    off, _ = await _act()
    switches(snapshot=True, discipline=True)
    on, actor = await _act()
    assert caa._library_counts(actor.function_manager, actor.guidance_manager) == (
        0,
        0,
    )
    assert off[0]["tool_choice"] == "required"
    assert on[0]["tool_choice"] == "auto"
    # The fixed tool list, the system prompt and every byte but the first
    # user message's snapshot line are what the gate would have sent.
    assert h.request_bytes(on[0])["tools"] == h.request_bytes(off[0])["tools"]
    assert on[0]["messages"][0] == off[0]["messages"][0]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_empty_libraries_without_cache_discipline_the_full_toolkit_is_sent(
    switches,
):
    switches(snapshot=False)
    off, _ = await _act()
    switches(snapshot=True)
    on, _ = await _act()
    assert off[0]["tool_choice"] == "required"
    assert on[0]["tool_choice"] == "auto"
    # What the gate sends once it is satisfied, from the first turn.
    assert h.request_bytes(on[0])["tools"] == h.request_bytes(off[1])["tools"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("discipline", [False, True])
async def test_on_a_stored_entry_keeps_the_forced_search_as_shipped(
    switches,
    discipline,
):
    def seed(actor):
        _add_function(actor)
        _add_guidance(actor)

    switches(snapshot=False, discipline=discipline)
    off, _ = await _act(seed=seed)
    switches(snapshot=True, discipline=discipline)
    on, _ = await _act()  # the same store
    assert on[0]["tool_choice"] == off[0]["tool_choice"] == "required"
    assert h.request_bytes(on[0])["tools"] == h.request_bytes(off[0])["tools"]
    assert _first_user(on[0]) == (
        f"{SNAPSHOT} 1 stored function, 1 guidance entry.\n\n---\n\n{TASK}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_only_the_nonempty_library_is_searched_first(switches):
    switches(snapshot=True)
    on, _ = await _act(seed=_add_guidance)
    assert on[0]["tool_choice"] == "required"
    gated = [t["function"]["name"] for t in on[0]["tools"]]
    assert "GuidanceManager_search" in gated
    assert not any(name.startswith("FunctionManager_") for name in gated)
    assert "execute_code" not in gated
    assert _first_user(on[0]).startswith(
        f"{SNAPSHOT} 0 stored functions, 1 guidance entry. "
        "An empty library is not searched first.",
    )


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


def test_builtin_guidance_counts_only_where_it_is_shown(monkeypatch):
    from unify.guidance_manager.guidance_manager import GuidanceManager

    gm = GuidanceManager()
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    assert caa._library_counts(None, gm) == (None, 0)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", True)
    shown = caa._library_counts(None, gm)[1]
    assert shown == len(gm.filter())


# ── the snapshot line ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_snapshot_is_in_the_first_user_message_only(switches):
    switches(snapshot=True, discipline=True)
    on, _ = await _act()
    line = (
        f"{SNAPSHOT} 0 stored functions, 0 guidance entries. "
        "An empty library is not searched first."
    )
    assert _first_user(on[0]) == f"{line}\n\n---\n\n{TASK}"
    for request in on:
        holders = [
            m
            for m in request["messages"]
            if SNAPSHOT in json.dumps(m.get("content"), default=str)
        ]
        # Once, in the message that opens the session, never edited.
        assert [m["role"] for m in holders] == ["user"]
        assert holders[0]["content"] == _first_user(on[0])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_no_snapshot_and_the_gate_is_as_shipped(switches):
    switches(snapshot=False)
    off, _ = await _act()
    assert _first_user(off[0]) == TASK
    assert off[0]["tool_choice"] == "required"
    assert not any(SNAPSHOT in json.dumps(r["messages"], default=str) for r in off)


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_LIBRARY_SNAPSHOT=value).UNIFY_LIBRARY_SNAPSHOT
        is expected
    )
