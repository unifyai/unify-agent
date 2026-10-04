"""Symbolic: ``UNIFY_DISCOVERY_SPECULATIVE_TURN`` off answers the gate's searches as one unit.

While the actor's discovery gate is open, its policy returns ``eager=True``:
as soon as a turn's library searches are scheduled the loop asks the model
again, without their results. A model that searched only one family is
asked for the other at once; when the first search lands during that turn,
the turn is cancelled and asked again, and the provider bills it anyway. In
the 4 Oct ARC LOW run (arc-ufix3-h-low-s0, 3 episodes) 31 of the 37 billed
cancelled requests were that turn. With the switch off the gate is the same
(``tool_choice="required"``, only the discovery tools, parallel calls asked
for, ``compress_context`` withheld) but grants no turn while its searches
run, and the searches it forces are one unit: when a turn makes both, the
model is woken once, when both have returned (at most
``UNIFY_WAIT_CEILING_SECONDS``; a new message still wakes it at once),
instead of on the first and again, cancelling, on the second. The mutator
that adds the missing family to a turn that searched one recognises the
actor's gate request, which as shipped it never does: that request also
lists the loop's ``wait``, ``steer`` and ``ask_about_completed_tool``. A
loop without the mutator that searched one family is asked for the other
once that search has returned. The transport is scripted and slowed, so the
race is deterministic and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.code_act_actor import (
    _build_discovery_parallel_mutator,
    _default_tool_policy,
    _is_discovery_gate_schema,
)
from unify.common._async_tool import batch_wait
from unify.common._async_tool.loop import _policy_gates_turn, _policy_requires_unit
from unify.settings import ProductionSettings, SETTINGS

# Every model turn after the first takes this long, like a model thinking, so
# a search can land during it.
LLM_SECONDS = 0.6
# The function search returns after this long, the guidance search at once
# unless a test says otherwise.
FM_SECONDS = 0.4

FM = "FunctionManager_search_functions"
GM = "GuidanceManager_search"
BASE_TOOLS = {FM, GM}
# The tools the loop adds to every request, as in the actor's gate request.
LOOP_TOOLS = {"wait", "steer", "ask_about_completed_tool"}


def _gate_tools(fm_seconds: float = FM_SECONDS, gm_seconds: float = 0.0) -> dict:
    # The parameters the gate mutator's appended calls use, so they run.
    async def FunctionManager_search_functions(query: str, n: int = 5) -> str:
        """Search stored functions by meaning.

        Args:
            query: What the function should do.
            n: How many functions to return.
        """
        await asyncio.sleep(fm_seconds)
        return "FM_RESULT"

    async def GuidanceManager_search(
        k: int = 5,
        references: dict | None = None,
    ) -> str:
        """Search stored guidance.

        Args:
            k: How many entries to return.
            references: What the guidance should be about.
        """
        if gm_seconds:
            await asyncio.sleep(gm_seconds)
        return "GM_RESULT"

    return {FM: FunctionManager_search_functions, GM: GuidanceManager_search}


def _with_gate_mutator(client):
    """Install the actor's discovery mutator on *client*, as ``act`` does."""
    mutator = _build_discovery_parallel_mutator()
    generate = client.generate

    def _generate(*args, **kwargs):
        kwargs.setdefault("completion_mutator", mutator)
        return generate(*args, **kwargs)

    client.generate = _generate
    return client


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


def _function_only_replies():
    """The model searches functions only, then answers."""
    return [lambda: h.completion(calls=[(FM, {"query": "q"})]), *_done()]


async def _run(
    monkeypatch,
    replies,
    *,
    speculative: bool,
    tools=None,
    mutator: bool = False,
    ceiling: float = 15.0,
    during=None,
):
    """Run one gated loop; return its requests (with parallel_tool_calls), send times, handle."""
    import unillm.clients.uni_llm as uni_llm
    from unify.common.async_tool_loop import start_async_tool_loop

    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", speculative)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_FOR_BATCH", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_CEILING_SECONDS", ceiling)
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
        client = h.new_client()
        handle = start_async_tool_loop(
            _with_gate_mutator(client) if mutator else client,
            "Run the tools.",
            tools or _gate_tools(),
            log_steps=False,
            timeout=60,
            max_steps=30,
            tool_policy=_policy(),
        )
        if during is not None:
            await during(handle)
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


def _opening_calls(request: dict) -> list[str]:
    """The tools the first assistant turn in *request* called."""
    first = next(m for m in request["messages"] if m.get("role") == "assistant")
    return [c["function"]["name"] for c in first.get("tool_calls") or []]


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
    off = {"eager": False, "gated": True, "required_unit": True}
    mode, gated, opts = policy(0, tools, [])
    assert (mode, set(gated), opts) == ("required", {FM, GM}, off)
    # The missing family is still required once the first has been called.
    mode, gated, opts = policy(1, tools, [FM])
    assert (mode, set(gated), opts) == ("required", {GM}, off)
    # Both called: the full set, as shipped.
    assert policy(2, tools, [FM, GM]) == ("auto", tools)


def test_a_gated_result_gates_its_turn():
    assert _policy_gates_turn(("required", {}, {"eager": True}))
    assert _policy_gates_turn(("required", {}, {"eager": False, "gated": True}))
    assert _policy_gates_turn(("required", {}, True))
    assert not _policy_gates_turn(("required", {}, {"eager": False}))
    assert not _policy_gates_turn(("auto", {}))


def test_a_required_unit_result_holds_its_required_calls():
    assert _policy_requires_unit(("required", {}, {"required_unit": True}))
    assert not _policy_requires_unit(("auto", {}, {"required_unit": True}))
    assert not _policy_requires_unit(("required", {}, {"eager": True}))
    assert not _policy_requires_unit(("required", {}, True))
    assert not _policy_requires_unit(("required", {}))


@pytest.mark.asyncio
async def test_a_policy_hold_holds_only_its_own_calls():
    async def _idle():
        await asyncio.sleep(0)

    forced = {asyncio.create_task(_idle()) for _ in range(2)}
    other = asyncio.create_task(_idle())
    await asyncio.gather(*forced, other)

    declared = batch_wait.BatchHold()
    declared.install(forced, 5.0)
    assert declared.holds({other}) and declared.holds(set(forced))

    own = batch_wait.BatchHold()
    own.install(forced, 5.0, own_only=True)
    assert own.holds(set(forced))
    assert not own.holds({other}) and not own.holds({other, *forced})
    own.release()
    assert not own.own_only and not own.holds(set(forced))


def test_off_the_mutator_recognises_the_actors_gate_request(monkeypatch):
    """As shipped the loop's own tools hide the gate from the mutator."""
    request = [FM, GM, *sorted(LOOP_TOOLS)]
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", True)
    assert not _is_discovery_gate_schema(request)
    assert _is_discovery_gate_schema([FM, GM, "compress_context"])
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", False)
    assert _is_discovery_gate_schema(request)
    assert _is_discovery_gate_schema([*request, "final_response"])
    assert not _is_discovery_gate_schema([*request, "execute_code"])
    assert not _is_discovery_gate_schema([FM, *sorted(LOOP_TOOLS)])


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
async def test_as_shipped_a_turn_that_searches_both_families_is_woken_twice(
    monkeypatch,
):
    requests, _, handle = await _run(
        monkeypatch,
        _both_replies(),
        speculative=True,
        tools=_gate_tools(gm_seconds=0.05),
    )
    # Woken on the guidance result; the function result cancels that turn.
    assert {"GM_RESULT"} in [_results_seen(r) for r in requests[1:]]
    assert handle._runtime_state.cancelled_turns_by_cause.get("tool_result", 0) >= 1


@pytest.mark.asyncio
async def test_off_a_turn_that_searches_both_families_is_woken_once_with_both(
    monkeypatch,
):
    requests, sent, handle = await _run(
        monkeypatch,
        _both_replies(),
        speculative=False,
        tools=_gate_tools(gm_seconds=0.05),
    )
    # One turn after the searches, sent once the slower one has returned.
    assert len(requests) == 2, [_results_seen(r) for r in requests]
    assert _results_seen(requests[1]) == {"FM_RESULT", "GM_RESULT"}
    assert not _pending_placeholder(requests[1])
    assert sent[1] - sent[0] >= FM_SECONDS - 0.05
    assert requests[1]["tool_choice"] == "auto"
    assert handle._runtime_state.cancelled_turns == 0


@pytest.mark.asyncio
async def test_off_the_mutator_completes_a_one_family_turn_and_both_are_one_unit(
    monkeypatch,
):
    requests, sent, handle = await _run(
        monkeypatch,
        _function_only_replies(),
        speculative=False,
        mutator=True,
        tools=_gate_tools(gm_seconds=0.05),
    )
    # The loop's own tools are in the gate request, as in the actor's.
    assert LOOP_TOOLS <= _tool_names(requests[0])
    # The mutator added the guidance search to the model's function search ...
    assert _opening_calls(requests[1]) == [FM, GM]
    # ... and the model is woken once, with both results.
    assert len(requests) == 2, [_results_seen(r) for r in requests]
    assert _results_seen(requests[1]) == {"FM_RESULT", "GM_RESULT"}
    assert sent[1] - sent[0] >= FM_SECONDS - 0.05
    assert handle._runtime_state.cancelled_turns == 0


@pytest.mark.asyncio
async def test_as_shipped_the_mutator_leaves_a_one_family_turn_alone(monkeypatch):
    requests, _, handle = await _run(
        monkeypatch,
        _partial_replies(),
        speculative=True,
        mutator=True,
    )
    assert _opening_calls(requests[1]) == [FM]
    assert _pending_placeholder(requests[1])
    assert handle._runtime_state.cancelled_turns >= 1


@pytest.mark.asyncio
async def test_off_the_unit_wait_ends_at_the_ceiling(monkeypatch):
    requests, sent, _ = await _run(
        monkeypatch,
        _both_replies(),
        speculative=False,
        tools=_gate_tools(fm_seconds=2.5),
        ceiling=1.0,
    )
    # Woken at the ceiling with the guidance result; the function search runs on.
    assert _results_seen(requests[1]) == {"GM_RESULT"}
    assert _pending_placeholder(requests[1])
    assert 0.9 <= sent[1] - sent[0] < 2.0


@pytest.mark.asyncio
async def test_off_an_interjection_during_the_unit_wait_wakes_the_model_at_once(
    monkeypatch,
):
    async def interject(handle):
        await asyncio.sleep(0.3)
        await handle.interject("STOP_AND_LISTEN")

    requests, sent, _ = await _run(
        monkeypatch,
        _both_replies(),
        speculative=False,
        tools=_gate_tools(fm_seconds=3.0),
        during=interject,
    )
    assert "STOP_AND_LISTEN" in json.dumps(requests[1]["messages"])
    assert sent[1] - sent[0] < 1.0


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


ACTOR_FUNCTION_ONLY_REPLIES = (
    lambda: h.completion(calls=[(FM, {"query": "list files"})]),
    *([lambda: h.completion(content="done")] * 8),
)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("speculative", [True, False], ids=["shipped", "off"])
async def test_the_actors_one_family_opening_turn_is_completed_only_when_off(
    monkeypatch,
    speculative,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_SPECULATIVE_TURN", speculative)
    _result, _, requests = await h.scenario_actor(ACTOR_FUNCTION_ONLY_REPLIES)
    session = h.session_requests(requests)
    assert LOOP_TOOLS <= _tool_names(session[0])
    if speculative:
        # As shipped the mutator does not recognise the gate request.
        assert _opening_calls(session[1]) == [FM]
    else:
        # The mutator added the guidance search, and the model is woken once
        # both searches have returned, on a turn past the gate.
        assert _opening_calls(session[1]) == [FM, GM]
        assert not _pending_placeholder(session[1])
        assert session[1]["tool_choice"] == "auto"
