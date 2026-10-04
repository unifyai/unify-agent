"""Symbolic: ``UNIFY_DISCOVERY_GATE`` off leaves the library searches to the model.

As shipped the actor's default tool policy opens every task with a
discovery-first gate: until each present library family has been searched
the model is offered only the FunctionManager and GuidanceManager search
tools with ``tool_choice="required"``, and the system prompt tells it to
search both first ("Discovery-First Policy (Active) -- HARD REQUIREMENT",
and the library section's "Always search ... A no-hit is **not**
permission ..."). With the switch off no turn is gated or forced: every turn
carries the actor's full tool list with ``tool_choice="auto"``, and the
prompt says once that the library exists and can be searched with the
listed tools when useful. The library tools and their schemas, stored
functions and ``UNIFY_LIBRARY_SNAPSHOT``'s counts are unchanged. Requests
are captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import prompt_builders as pb
from unify.settings import ProductionSettings, SETTINGS

TASK = "List the files in the workspace."
FM = "FunctionManager_search_functions"
GM = "GuidanceManager_search"
NEUTRAL = (
    "A library of reusable functions and guidance exists and can be "
    "searched with the listed library tools when useful."
)
# Every wording that tells the model to search the libraries first.
SEARCH_FIRST = (
    "Discovery-First Policy",
    "HARD REQUIREMENT",
    "Always search",
    "always search",
    "before deciding how to execute",
    "no-hit is **not** permission",
    "No-hit is not permission",
    "no-hit is not permission",
    "search first",
    "searched first",
    "first tool-calling assistant message",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _search_first_phrases(text: str) -> list[str]:
    flat = _flat(text)
    return [p for p in SEARCH_FIRST if p in text or _flat(p) in flat]


@pytest.fixture
def switches(monkeypatch):
    def set_(
        *,
        gate: bool,
        snapshot: bool = False,
        discipline: bool = False,
    ) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", snapshot)
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", discipline)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)

    return set_


async def _act(replies, *, seed=None) -> list[dict]:
    """One scripted ``act()`` on a fresh actor; its session's requests."""
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(TASK, persist=False)
            result = await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    assert result == "done"
    return h.session_requests(provider.requests)


def _answer_at_once():
    """The model answers on its first turn; the review after it stores nothing."""
    return [lambda: h.completion(content="done")] * 8


def _names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"] or []]


def _called(requests: list[dict]) -> list[str]:
    """Every tool the model called in the session, from the last request."""
    return [
        call["function"]["name"]
        for m in requests[-1]["messages"]
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
    ]


def _system(request: dict) -> str:
    return request["messages"][0]["content"]


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


# ── the request: tool_choice and tools ───────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_first_turn_is_a_forced_search_as_shipped(switches):
    switches(gate=True)
    on = await _act(h.ACTOR_REPLIES)
    assert on[0]["tool_choice"] == "required"
    assert set(_names(on[0])) == {FM, GM, "wait", "steer", "ask_about_completed_tool"}
    assert "execute_code" not in _names(on[0])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_first_turn_is_not_forced_to_search(switches):
    switches(gate=False)
    off = await _act(_answer_at_once())
    # The model answered on its first turn: one session request, no search.
    assert len(off) == 1
    assert off[0]["tool_choice"] == "auto"
    assert _called(off) == []
    assert all(m["role"] != "tool" for m in off[0]["messages"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("discipline", [False, True])
async def test_off_every_turn_offers_the_tools_the_gate_unlocks(switches, discipline):
    switches(gate=True, discipline=discipline)
    on = await _act(h.ACTOR_REPLIES)
    switches(gate=False, discipline=discipline)
    off = await _act(_answer_at_once())
    # The library tools are still exposed, with the schemas the gate's
    # satisfied turn sends, from the first turn.
    assert {FM, GM} <= set(_names(off[0]))
    assert "execute_code" in _names(off[0])
    assert "execute_function" in _names(off[0])
    assert h.request_bytes(off[0])["tools"] == h.request_bytes(on[1])["tools"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_a_search_the_model_chooses_still_runs(switches):
    switches(gate=False)
    off = await _act(
        [
            lambda: h.completion(content=None, calls=[(FM, {"query": "list files"})]),
            *_answer_at_once(),
        ],
    )
    assert off[0]["tool_choice"] == "auto"
    assert _called(off) == [FM]
    results = [m for m in off[-1]["messages"] if m.get("role") == "tool"]
    assert len(results) == 1 and "error" not in str(results[0]["content"]).lower()


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_a_stored_function_is_callable_without_a_search(switches):
    switches(gate=False)
    off = await _act(
        [
            lambda: h.completion(
                calls=[
                    (
                        "execute_function",
                        {"function_name": "list_names", "call_kwargs": {"path": "."}},
                    ),
                ],
            ),
            *_answer_at_once(),
        ],
        seed=_add_function,
    )
    assert _called(off) == ["execute_function"]
    result = next(m for m in off[-1]["messages"] if m.get("role") == "tool")
    content = json.dumps(result["content"], default=str)
    assert "Traceback" not in content and "not found" not in content.lower()


# ── the prompt ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_prompt_is_as_shipped(switches):
    switches(gate=True)
    on = await _act(h.ACTOR_REPLIES)
    system = _system(on[0])
    assert pb._DISCOVERY_FIRST_POLICY in system
    assert pb._ALWAYS_SEARCH_FIRST in system
    assert NEUTRAL not in _flat(system)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("discipline", [False, True])
async def test_off_no_request_tells_the_model_to_search_first(switches, discipline):
    switches(gate=False, snapshot=True, discipline=discipline)
    off = await _act(
        [
            lambda: h.completion(calls=[("execute_code", {"code": "print(1)"})]),
            *_answer_at_once(),
        ],
    )
    system = _system(off[0])
    assert _flat(system).count(NEUTRAL) == 1
    assert pb._SEARCH_WHEN_USEFUL in system
    for request in off:
        for message in request["messages"]:
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            assert _search_first_phrases(text) == [], message["role"]
    # Under UNIFY_CACHE_DISCIPLINE the gate's refusal rule would have
    # refused a call to anything but a search ("the libraries are searched
    # first"); off, the model's first call runs.
    assert _called(off) == ["execute_code"]
    assert off[0]["tool_choice"] == "auto"


SWITCH_VARIANTS = {
    "shipped": {},
    "unified": {"UNIFY_REVIEW_FRAMING": "unified"},
    "try first": {"UNIFY_TRY_FIRST": True},
    "unified try first": {"UNIFY_REVIEW_FRAMING": "unified", "UNIFY_TRY_FIRST": True},
}
PROMPT_VARIANTS = {
    "act": {"can_store": True},
    "persist": {"can_store": True, "persist": True},
    "read only": {"can_store": True, "library_read_only": True},
    "inline on": {"can_store": True, "inline_curation": "on"},
    "inline only": {"inline_curation": "only"},
    "no execute_code": {"no_code": True},
}


def _prompt(variant: dict, *, gate: bool) -> str:
    variant = dict(variant)
    actor = caa.CodeActActor()
    tools = dict(actor.get_tools("act"))
    if variant.pop("no_code", False):
        tools.pop("execute_code", None)
    return pb.build_code_act_prompt(
        environments={},
        tools=tools,
        discovery_first_policy=gate,
        **({} if gate else {"search_when_useful": True}),
        **variant,
    )


@pytest.mark.parametrize("switch_set", sorted(SWITCH_VARIANTS))
@pytest.mark.parametrize("prompt_variant", sorted(PROMPT_VARIANTS))
def test_off_every_prompt_variant_has_the_one_sentence_and_no_search_first(
    monkeypatch,
    switch_set,
    prompt_variant,
):
    for name, value in SWITCH_VARIANTS[switch_set].items():
        monkeypatch.setattr(SETTINGS, name, value)
    on = _prompt(PROMPT_VARIANTS[prompt_variant], gate=True)
    off = _prompt(PROMPT_VARIANTS[prompt_variant], gate=False)
    assert "Always search" in on and pb._DISCOVERY_FIRST_POLICY in on
    assert _search_first_phrases(off) == []
    assert _flat(off).count(NEUTRAL) == 1
    # Only the search-first text changed.
    assert off == on.replace("\n\n" + pb._DISCOVERY_FIRST_POLICY, "").replace(
        pb._ALWAYS_SEARCH_FIRST,
        pb._SEARCH_WHEN_USEFUL,
    )


def test_the_sentence_has_no_frequency_rule_or_benchmark_wording():
    sentence = NEUTRAL.lower()
    for word in ("always", "first", "every", "must", "before", "each", "never"):
        assert word not in sentence.split()
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    assert not BENCHMARK_WORDS.search(pb._SEARCH_WHEN_USEFUL)


# ── UNIFY_LIBRARY_SNAPSHOT ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "seed, line",
    [
        (None, "Library at task start: 0 stored functions, 0 guidance entries."),
        (
            _add_guidance,
            "Library at task start: 0 stored functions, 1 guidance entry.",
        ),
    ],
)
async def test_off_the_snapshot_still_states_the_counts(switches, seed, line):
    switches(gate=False, snapshot=True)
    off = await _act(_answer_at_once(), seed=seed)
    # The counts, without the gate's note that an empty library is not
    # searched first: there is no gate, and nothing says to search.
    assert _first_user(off[0]) == f"{line}\n\n---\n\n{TASK}"
    assert off[0]["tool_choice"] == "auto"


def test_the_snapshot_line_names_the_gate_only_when_there_is_one():
    counts = (0, 1)
    gated = caa._library_snapshot_line(
        counts,
        has_fm_tools=True,
        has_gm_tools=True,
        discovery_gate=True,
    )
    free = caa._library_snapshot_line(
        counts,
        has_fm_tools=True,
        has_gm_tools=True,
        discovery_gate=False,
    )
    assert gated.endswith("An empty library is not searched first.")
    assert free == "Library at task start: 0 stored functions, 1 guidance entry."


# ── the setting ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, expected",
    [("1", True), ("true", True), ("0", False), ("false", False)],
)
def test_the_setting_parses_booleans(value, expected):
    assert (
        ProductionSettings(UNIFY_DISCOVERY_GATE=value).UNIFY_DISCOVERY_GATE is expected
    )


def test_the_setting_is_on_by_default():
    assert ProductionSettings.model_fields["UNIFY_DISCOVERY_GATE"].default is True


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_callers_own_policy_is_unaffected(switches):
    switches(gate=False)
    seen = []

    def policy(step, tools):
        seen.append(step)
        return ("required", {FM: tools[FM]}) if step == 0 else ("auto", tools)

    actor = caa.CodeActActor(tool_policy=policy)
    try:
        with h.scripted(
            [lambda: h.completion(calls=[(FM, {"query": "q"})]), *_answer_at_once()],
        ) as provider:
            handle = await actor.act(TASK, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    first = h.session_requests(provider.requests)[0]
    assert seen and first["tool_choice"] == "required"
    assert FM in _names(first) and "execute_code" not in _names(first)
