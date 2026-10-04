"""Symbolic: ``UNIFY_DISCOVERY_SPECULATIVE_TURN`` off drops the gate's speculative turn.

While the actor's discovery gate is open, its policy returns ``eager=True``:
as soon as a turn's library searches are scheduled the loop asks the model
again, without their results. A model that searched only one family is
asked for the other at once; when the first search lands during that turn,
the turn is cancelled and asked again, and the provider bills it anyway. In
the 4 Oct ARC LOW run (arc-ufix3-h-low-s0, 3 episodes) 31 of the 37 billed
cancelled requests were that turn. With the switch off the gate is the same
(``tool_choice="required"``, only the discovery tools, parallel calls asked
for, ``compress_context`` withheld) but grants no turn while its searches
run: a model that searched one family is asked for the other once that
search has returned. The transport is scripted and slowed, so the race is
deterministic and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.code_act_actor import _default_tool_policy
from unify.common._async_tool.loop import _policy_gates_turn
from unify.settings import ProductionSettings, SETTINGS

# Every model turn after the first takes this long, like a model thinking, so
# a search can land during it.
LLM_SECONDS = 0.6
# The function search returns after this long, the guidance search at once.
FM_SECONDS = 0.4

FM = "FunctionManager_search_functions"
GM = "GuidanceManager_search"
BASE_TOOLS = {FM, GM}


def _gate_tools() -> dict:
    async def FunctionManager_search_functions(query: str) -> str:
        """Search stored functions by meaning.

        Args:
            query: What the function should do.
        """
        await asyncio.sleep(FM_SECONDS)
        return "FM_RESULT"

    async def GuidanceManager_search(k: int) -> str:
        """Search stored guidance.

        Args:
            k: How many entries to return.
        """
        return "GM_RESULT"

    return {FM: FunctionManager_search_functions, GM: GuidanceManager_search}


def _policy():
    return _default_tool_policy(
        has_fm_tools=True,
        has_gm_tools=True,
        filter_tools=lambda tools: tools,
    )


def _done():
    return [lambda: h.completion(content="done")] * 6


def _partial_replies():
    """The model searches functions only, then guidance when asked."""
    return [
        lambda: h.completion(calls=[(FM, {"query": "q"})]),
        lambda: h.completion(calls=[(GM, {"k": 3})]),
        *_done(),
    ]


def _both_replies():
    return [
        lambda: h.completion(calls=[(FM, {"query": "q"}), (GM, {"k": 3})]),
        *_done(),
    ]


async def _run(monkeypatch, replies, *, speculative: bool):
    """Run one gated loop; return its requests (with parallel_tool_calls), send times, handle."""
    import unillm.clients.uni_llm as uni_llm
    from unify.common.async_tool_loop import start_async_tool_loop

    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", speculative)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_FOR_BATCH", False)
    sent: list[float] = []
    parallel: list = []
    with h.scripted(replies) as provider:
        scripted = uni_llm._acompletion_with_transient_retry

        async def slowed(**kw):
            sent.append(time.monotonic())
            parallel.append(kw.get("parallel_tool_calls"))
            if provider.requests:
                await asyncio.sleep(LLM_SECONDS)
            return await scripted(**kw)

        uni_llm._acompletion_with_transient_retry = slowed
        handle = start_async_tool_loop(
            h.new_client(),
            "Run the tools.",
            _gate_tools(),
            log_steps=False,
            timeout=60,
            max_steps=30,
            tool_policy=_policy(),
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
        # A cancelled turn is still answered in the background; give its
        # charge time to be reported before the transport is restored.
        await asyncio.sleep(LLM_SECONDS + 0.3)
    assert result == "done"
    requests = [
        dict(r, parallel_tool_calls=p) for r, p in zip(provider.requests, parallel)
    ]
    return requests, sent, handle


def _results_seen(request: dict) -> set[str]:
    return {
        str(m.get("content"))
        for m in request["messages"]
        if m.get("role") == "tool" and "RESULT" in str(m.get("content"))
    }


def _pending_placeholder(request: dict) -> bool:
    return any(
        m.get("role") == "tool" and "_placeholder" in str(m.get("content"))
        for m in request["messages"]
    )


def _tool_names(request: dict) -> set[str]:
    return {t["function"]["name"] for t in request["tools"]}


def _shape(request: dict) -> tuple:
    """What a request asks of the model, without ids or timings."""
    return (
        request["tool_choice"],
        tuple(sorted(_tool_names(request))),
        tuple(sorted(_results_seen(request))),
        _pending_placeholder(request),
        request["parallel_tool_calls"],
    )


# ── the policy ──────────────────────────────────────────────────────────────


def test_the_policy_gates_without_the_eager_turn_when_off(monkeypatch):
    tools = {FM: object(), GM: object(), "execute_code": object()}
    policy = _policy()

    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", True)
    assert policy(0, tools, []) == (
        "required",
        {FM: tools[FM], GM: tools[GM]},
        {"eager": True},
    )

    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", False)
    mode, gated, opts = policy(0, tools, [])
    assert (mode, set(gated), opts) == (
        "required",
        {FM, GM},
        {"eager": False, "gated": True},
    )
    # The missing family is still required once the first has been called.
    mode, gated, opts = policy(1, tools, [FM])
    assert (mode, set(gated), opts) == (
        "required",
        {GM},
        {"eager": False, "gated": True},
    )
    # Both called: the full set, as shipped.
    assert policy(2, tools, [FM, GM]) == ("auto", tools)


def test_a_gated_result_gates_its_turn():
    assert _policy_gates_turn(("required", {}, {"eager": True}))
    assert _policy_gates_turn(("required", {}, {"eager": False, "gated": True}))
    assert _policy_gates_turn(("required", {}, True))
    assert not _policy_gates_turn(("required", {}, {"eager": False}))
    assert not _policy_gates_turn(("auto", {}))


def test_the_setting_defaults_to_as_shipped():
    assert ProductionSettings().UNIFY_DISCOVERY_SPECULATIVE_TURN is True
    assert (
        ProductionSettings(
            UNIFY_DISCOVERY_SPECULATIVE_TURN="false",
        ).UNIFY_DISCOVERY_SPECULATIVE_TURN
        is False
    )


# ── the race ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_as_shipped_a_partial_search_gets_a_speculative_turn_that_is_cancelled(
    monkeypatch,
):
    requests, sent, handle = await _run(
        monkeypatch,
        _partial_replies(),
        speculative=True,
    )
    # Asked for guidance at once, while the function search still runs ...
    assert _pending_placeholder(requests[1])
    assert _results_seen(requests[1]) == set()
    assert sent[1] - sent[0] < FM_SECONDS - 0.1
    # ... and that turn is cancelled when the search lands, then asked again.
    assert handle._runtime_state.cancelled_turns >= 1
    assert handle._runtime_state.cancelled_turns_by_cause.get("tool_result", 0) >= 1
    assert _results_seen(requests[2]) == {"FM_RESULT"}


@pytest.mark.asyncio
async def test_off_a_partial_search_is_followed_by_the_missing_family_once_it_returns(
    monkeypatch,
):
    requests, sent, handle = await _run(
        monkeypatch,
        _partial_replies(),
        speculative=False,
    )
    # The gated opening turn is the shipped one.
    assert requests[0]["tool_choice"] == "required"
    assert _tool_names(requests[0]) & BASE_TOOLS == {FM, GM}
    assert "compress_context" not in _tool_names(requests[0])
    assert requests[0]["parallel_tool_calls"] is True
    # No speculative turn: the next waits for the function search, and the
    # gate, still open, requires only the guidance search on it.
    assert not _pending_placeholder(requests[1])
    assert _results_seen(requests[1]) == {"FM_RESULT"}
    assert sent[1] - sent[0] >= FM_SECONDS - 0.05
    assert requests[1]["tool_choice"] == "required"
    assert _tool_names(requests[1]) & BASE_TOOLS == {GM}
    assert "compress_context" not in _tool_names(requests[1])
    assert requests[1]["parallel_tool_calls"] is True
    # Then the full set, with both results, and nothing cancelled.
    assert _results_seen(requests[2]) == {"FM_RESULT", "GM_RESULT"}
    assert requests[2]["tool_choice"] == "auto"
    assert len(requests) == 3
    assert handle._runtime_state.cancelled_turns == 0


@pytest.mark.asyncio
async def test_a_turn_that_searches_both_families_is_unchanged(monkeypatch):
    """No eager turn follows a turn that satisfies the gate, so off changes nothing."""
    shipped, _, shipped_handle = await _run(
        monkeypatch,
        _both_replies(),
        speculative=True,
    )
    off, _, off_handle = await _run(monkeypatch, _both_replies(), speculative=False)
    assert [_shape(r) for r in off] == [_shape(r) for r in shipped]
    assert (
        off_handle._runtime_state.cancelled_turns
        == shipped_handle._runtime_state.cancelled_turns
    )


# ── the actor ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_actors_first_request_and_tools_are_unchanged(monkeypatch):
    from tests.actor.code_act.test_switches_off_equivalence import NEW_SWITCHES

    for name, value in NEW_SWITCHES.items():
        monkeypatch.setattr(SETTINGS, name, value)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", False)
    golden = json.loads(h.ACTOR_GOLDEN.read_text())
    _result, _, requests = await h.scenario_actor()
    assert h.actor_recording(requests) == golden
