"""Symbolic: ``UNIFY_WAIT_FOR_BATCH`` lets the model ask to be woken once per batch.

When a turn calls several tools and the first returns, the loop starts the
model's next turn straight away; when a sibling lands during it, that turn is
cancelled and asked again. The provider has already billed the cancelled step:
charged cancelled calls were 16.8% of known AppWorld cost and 24.8% of
ScienceWorld's in the 2 Oct research-build cells, almost all of them a turn
started on the first result of a two-search batch. With the switch the
``wait`` tool takes ``until="all"``: a turn that adds it to its own calls is
woken once, when they have all finished (or at ``max_seconds``, clamped to
``UNIFY_WAIT_CEILING_SECONDS``); a new message, a clarification or a stop still
wakes it at once, and a turn without it is woken as shipped. Every cancelled
turn is counted and published as a ``ToolLoopCancelledTurn`` event either way.
The transport is scripted and slowed, so the race is deterministic and nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
from decimal import Decimal

import pytest

from tests import cache_discipline_helpers as h
from unify.common._async_tool import batch_wait
from unify.settings import ProductionSettings, SETTINGS

# Every model turn after the first takes this long, like a model thinking, so
# a sibling can land during it.
LLM_SECONDS = 0.6


async def fast_tool() -> str:
    """Return quickly."""
    await asyncio.sleep(0.05)
    return "FAST_RESULT"


async def medium_tool() -> str:
    """Return after the fast tool, while a model turn started then is in flight."""
    await asyncio.sleep(0.4)
    return "MEDIUM_RESULT"


async def slow_tool() -> str:
    """Return long after any hold in these tests."""
    await asyncio.sleep(3.0)
    return "SLOW_RESULT"


TOOLS = {"fast_tool": fast_tool, "medium_tool": medium_tool, "slow_tool": slow_tool}


def _done():
    return [lambda: h.completion(content="done")] * 6


def _batch(*names: str, wait: dict | None = None, wait_first: bool = False):
    calls = [(name, {}) for name in names]
    if wait is not None:
        calls = [("wait", wait), *calls] if wait_first else [*calls, ("wait", wait)]
    return lambda: h.completion(calls=calls)


async def _run(
    monkeypatch,
    replies,
    *,
    on: bool = True,
    ceiling: float = 15.0,
    tools=None,
    during=None,
    **loop_kwargs,
):
    """Run one loop; return its requests, their send times and the handle."""
    import unillm.clients.uni_llm as uni_llm
    from unify.common.async_tool_loop import start_async_tool_loop

    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_FOR_BATCH", on)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_CEILING_SECONDS", ceiling)
    sent: list[float] = []
    started = dt.datetime.now(dt.UTC)
    with h.scripted(replies) as provider:
        scripted = uni_llm._acompletion_with_transient_retry

        async def slowed(**kw):
            sent.append(time.monotonic())
            if provider.requests:
                await asyncio.sleep(LLM_SECONDS)
            return await scripted(**kw)

        uni_llm._acompletion_with_transient_retry = slowed
        handle = start_async_tool_loop(
            h.new_client(),
            "Run the tools.",
            tools or TOOLS,
            log_steps=False,
            timeout=60,
            max_steps=30,
            **loop_kwargs,
        )
        handle._test_started = started
        if during is not None:
            await during(handle)
        result = await asyncio.wait_for(handle.result(), timeout=60)
        # A cancelled turn is still answered in the background; give its
        # charge time to be reported before the transport is restored.
        await asyncio.sleep(LLM_SECONDS + 0.3)
    assert result == "done"
    return provider.requests, sent, handle


def _results_seen(request: dict) -> set[str]:
    return {
        str(m.get("content"))
        for m in request["messages"]
        if m.get("role") == "tool" and "RESULT" in str(m.get("content"))
    }


def _wait_calls(request: dict) -> list[dict]:
    return [
        call
        for m in request["messages"]
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
        if call["function"]["name"] == "wait"
    ]


def _cancelled_events(handle) -> list[dict]:
    """This run's cancelled-turn events, oldest first."""
    from unify.events.event_bus import EVENT_BUS

    return [
        evt.payload
        for evt in reversed(
            EVENT_BUS.search(
                filter=f"type == '{batch_wait.CANCELLED_TURN_EVENT}'",
                limit=1000,
            ),
        )
        if evt.timestamp >= handle._test_started
    ]


# ── the declaration ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_first", [False, True])
async def test_a_declared_batch_wakes_the_model_once_with_every_result(
    monkeypatch,
    wait_first,
):
    requests, _, handle = await _run(
        monkeypatch,
        [
            _batch(
                "fast_tool",
                "medium_tool",
                wait={"until": "all"},
                wait_first=wait_first,
            ),
            *_done(),
        ],
    )
    # One turn after the batch, sent once both results were in.
    assert len(requests) == 2, [_results_seen(r) for r in requests]
    assert _results_seen(requests[1]) == {"FAST_RESULT", "MEDIUM_RESULT"}
    # The declaration leaves no trace in the transcript.
    assert _wait_calls(requests[1]) == []
    state = handle._runtime_state
    assert state.cancelled_turns == 0
    assert _cancelled_events(handle) == []


@pytest.mark.asyncio
async def test_without_the_declaration_the_shipped_race_is_reproduced_and_counted(
    monkeypatch,
):
    requests, _, handle = await _run(
        monkeypatch,
        [_batch("fast_tool", "medium_tool"), *_done()],
    )
    seen = [_results_seen(r) for r in requests[1:]]
    # A turn was sent with only the fast result, then again with both.
    assert {"FAST_RESULT"} in seen
    assert {"FAST_RESULT", "MEDIUM_RESULT"} in seen
    assert len(requests) >= 3
    state = handle._runtime_state
    assert state.cancelled_turns >= 1
    assert state.cancelled_turns_by_cause.get("tool_result", 0) >= 1
    events = _cancelled_events(handle)
    cancelled = [e for e in events if e["phase"] == "cancelled"]
    assert len(cancelled) == state.cancelled_turns
    assert cancelled[0]["cause"] == "tool_result"
    assert cancelled[0]["pending_tools"] >= 1
    # The provider answered the cancelled request in the background; its
    # charge, where unillm priced it, is reported under the same turn id.
    billed = [e for e in events if e["phase"] == "billed"]
    assert billed, events
    assert {e["turn_id"] for e in billed} <= {e["turn_id"] for e in cancelled}
    for e in billed:
        assert e["prompt_tokens"] == 100 and e["completion_tokens"] == 5
        assert Decimal(e["provider_cost_usd"]) > 0
    assert state.cancelled_turns_priced == len(billed)
    assert Decimal(state.cancelled_turns_usd) == sum(
        Decimal(e["provider_cost_usd"]) for e in billed
    )


@pytest.mark.asyncio
async def test_the_switch_off_wait_takes_no_arguments_and_holds_nothing(monkeypatch):
    requests, _, handle = await _run(
        monkeypatch,
        [_batch("fast_tool", "medium_tool", wait={"until": "all"}), *_done()],
        on=False,
    )
    wait_schema = next(
        t for t in requests[0]["tools"] if t["function"]["name"] == "wait"
    )
    assert wait_schema["function"]["parameters"]["properties"] == {}
    assert {"FAST_RESULT"} in [_results_seen(r) for r in requests[1:]]
    assert handle._runtime_state.cancelled_turns >= 1


@pytest.mark.asyncio
async def test_the_switch_on_wait_schema_and_prompt_line(monkeypatch):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "medium_tool", wait={"until": "all"}), *_done()],
    )
    wait_schema = next(
        t for t in requests[0]["tools"] if t["function"]["name"] == "wait"
    )
    props = wait_schema["function"]["parameters"]["properties"]
    assert props["until"]["enum"] == ["next", "all"]
    assert "max_seconds" in props
    assert 'wait(until=\\"all\\")' in json.dumps(wait_schema)

    from unify.actor import prompt_builders

    assert prompt_builders._WAIT_FOR_BATCH_LINE in prompt_builders._tools_section()
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_FOR_BATCH", False)
    assert prompt_builders._tools_section() == prompt_builders._TOOLS_SECTION


# ── the ceiling and what still wakes the model ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wait,ceiling",
    [({"until": "all"}, 1.0), ({"until": "all", "max_seconds": 1}, 15.0)],
    ids=["ceiling", "max_seconds"],
)
async def test_a_slow_call_holds_landed_results_only_until_the_limit(
    monkeypatch,
    wait,
    ceiling,
):
    requests, sent, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool", wait=wait), *_done()],
        ceiling=ceiling,
    )
    # Woken at the limit with the fast result; the slow call is still running.
    assert _results_seen(requests[1]) == {"FAST_RESULT"}
    assert 0.9 <= sent[1] - sent[0] < 2.0


def test_max_seconds_is_clamped_to_the_ceiling(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_CEILING_SECONDS", 15.0)
    assert batch_wait.hold_seconds({"until": "all"}) == 15.0
    assert batch_wait.hold_seconds({"until": "all", "max_seconds": 4}) == 4.0
    assert batch_wait.hold_seconds({"until": "all", "max_seconds": 500}) == 15.0
    assert batch_wait.hold_seconds({"until": "all", "max_seconds": -3}) == 0.0
    assert batch_wait.hold_seconds({"until": "all", "max_seconds": "x"}) == 15.0
    assert batch_wait.declares_batch({"until": "ALL"})
    assert not batch_wait.declares_batch({"until": "next"})
    assert not batch_wait.declares_batch({})


@pytest.mark.asyncio
async def test_an_interjection_during_a_declared_wait_wakes_the_model_at_once(
    monkeypatch,
):
    async def interject(handle):
        await asyncio.sleep(0.3)
        await handle.interject("STOP_AND_LISTEN")

    requests, sent, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool", wait={"until": "all"}), *_done()],
        during=interject,
    )
    assert "STOP_AND_LISTEN" in json.dumps(requests[1]["messages"])
    assert sent[1] - sent[0] < 1.0


# ── the eager turn a discovery gate grants ──────────────────────────────────


def _gate_tools() -> dict:
    async def FunctionManager_search_functions(query: str) -> str:
        """Search stored functions by meaning.

        Args:
            query: What the function should do.
        """
        await asyncio.sleep(0.4)
        return "FM_RESULT"

    async def GuidanceManager_search(k: int) -> str:
        """Search stored guidance.

        Args:
            k: How many entries to return.
        """
        return "GM_RESULT"

    return {
        "FunctionManager_search_functions": FunctionManager_search_functions,
        "GuidanceManager_search": GuidanceManager_search,
    }


def _gate_replies(wait: dict | None):
    first = [("FunctionManager_search_functions", {"query": "q"})]
    if wait is not None:
        first.append(("wait", wait))
    return [
        lambda: h.completion(calls=first),
        lambda: h.completion(calls=[("GuidanceManager_search", {"k": 3})]),
        *_done(),
    ]


def _pending_placeholder(request: dict) -> bool:
    return any(
        m.get("role") == "tool" and "_placeholder" in str(m.get("content"))
        for m in request["messages"]
    )


@pytest.mark.asyncio
async def test_an_eager_gate_turn_is_speculative_without_the_declaration(monkeypatch):
    requests, _, handle = await _run(
        monkeypatch,
        _gate_replies(None),
        tools=_gate_tools(),
        tool_policy=h.gate_policy,
    )
    # The gate grants a turn while the function search is still running.
    assert _pending_placeholder(requests[1])
    assert "FM_RESULT" not in json.dumps(requests[1]["messages"])


@pytest.mark.asyncio
async def test_a_declared_wait_skips_the_eager_gate_turn(monkeypatch):
    requests, sent, handle = await _run(
        monkeypatch,
        _gate_replies({"until": "all"}),
        tools=_gate_tools(),
        tool_policy=h.gate_policy,
    )
    # No speculative turn: the next one waits for the search, and the gate
    # still requires the guidance search on it.
    assert not _pending_placeholder(requests[1])
    assert "FM_RESULT" in json.dumps(requests[1]["messages"])
    assert requests[1]["tool_choice"] == "required"
    gated = {t["function"]["name"] for t in requests[1]["tools"]}
    assert "GuidanceManager_search" in gated
    assert "FunctionManager_search_functions" not in gated
    assert sent[1] - sent[0] >= 0.35
    assert handle._runtime_state.cancelled_turns == 0


# ── settings ────────────────────────────────────────────────────────────────


def test_the_settings_are_bounded():
    assert ProductionSettings().UNIFY_WAIT_FOR_BATCH is False
    assert ProductionSettings().UNIFY_WAIT_CEILING_SECONDS == 15.0
    assert (
        ProductionSettings(UNIFY_WAIT_CEILING_SECONDS="120").UNIFY_WAIT_CEILING_SECONDS
        == 120.0
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WAIT_CEILING_SECONDS="0.5")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_WAIT_CEILING_SECONDS="121")
