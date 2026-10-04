"""Symbolic: the lean-loop switches, each alone and together.

``UNIFY_BATCH_WAKE``: the model is woken once per tool batch.
 No model turn starts while a call is still running, and a result or a progress
notification never cancels a turn already sent. Only tool results wait for the
batch: a message from the user, the environment or another agent (an
interjection, a clarification or a notification) wakes the model at once. In
the fixed-build ARC LOW cell 158 of the 161 cancelled turns were a model turn
started on the first of two library searches the model itself chose (once the
discovery gate is satisfied the policy returns ``auto`` with no hold), the
empty function search landing in about 10 ms and the guidance search 40-600 ms
later; the model never declared ``wait(until="all")``.

``UNIFY_PENDING_REQUIRED=0``: a turn sent while calls run keeps its policy's
``tool_choice`` instead of being forced to ``required`` (337 of 1,446 main
calls in that cell).

The transport is scripted and slowed (``LLM_SECONDS`` per turn after the
first), so the races are deterministic and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import ProductionSettings, SETTINGS

LLM_SECONDS = 0.6


async def fast_tool() -> str:
    """Return quickly."""
    await asyncio.sleep(0.05)
    return "FAST_RESULT"


async def medium_tool() -> str:
    """Return after the fast tool, while a turn started then is in flight."""
    await asyncio.sleep(0.4)
    return "MEDIUM_RESULT"


async def late_tool() -> str:
    """Return while a turn woken at a 1 s ceiling is in flight."""
    await asyncio.sleep(1.3)
    return "LATE_RESULT"


async def slow_tool() -> str:
    """Return long after any ceiling in these tests."""
    await asyncio.sleep(2.5)
    return "SLOW_RESULT"


async def FunctionManager_search_functions(query: str) -> str:
    """Search stored functions by meaning (an empty library: at once).

    Args:
        query: What the function should do.
    """
    await asyncio.sleep(0.01)
    return "FM_EMPTY_RESULT"


async def GuidanceManager_search(query: str) -> str:
    """Search stored guidance.

    Args:
        query: What the guidance should cover.
    """
    await asyncio.sleep(0.45)
    return "GM_RESULT"


TOOLS = {
    "fast_tool": fast_tool,
    "medium_tool": medium_tool,
    "late_tool": late_tool,
    "slow_tool": slow_tool,
    "FunctionManager_search_functions": FunctionManager_search_functions,
    "GuidanceManager_search": GuidanceManager_search,
}

SHIPPED = {"UNIFY_BATCH_WAKE": False, "UNIFY_PENDING_REQUIRED": True}


def _done(n: int = 6):
    return [lambda: h.completion(content="done")] * n


def _batch(*calls):
    """One turn calling *calls*: names, or ``(name, args)`` pairs."""
    calls = [c if isinstance(c, tuple) else (c, {}) for c in calls]
    return lambda: h.completion(calls=calls)


async def _run(
    monkeypatch,
    replies,
    *,
    switches: dict,
    ceiling: float = 15.0,
    tools=None,
    during=None,
    after_first=None,
    expect: str = "done",
    **loop_kwargs,
):
    """Run one loop; return its requests, their send times and the handle."""
    import unillm.clients.uni_llm as uni_llm
    from unify.common.async_tool_loop import start_async_tool_loop

    for name, value in switches.items():
        monkeypatch.setattr(SETTINGS, name, value)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_FOR_BATCH", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_WAIT_CEILING_SECONDS", ceiling)
    sent: list[float] = []
    started = dt.datetime.now(dt.UTC)
    with h.scripted(replies) as provider:
        scripted = uni_llm._acompletion_with_transient_retry

        async def slowed(**kw):
            sent.append(time.monotonic())
            if provider.requests:
                await asyncio.sleep(LLM_SECONDS)
            elif after_first is not None:
                after_first()
            return await scripted(**kw)

        uni_llm._acompletion_with_transient_retry = slowed
        handle = start_async_tool_loop(
            h.new_client(),
            "Run the tools.",
            tools or TOOLS,
            log_steps=False,
            timeout=60,
            max_steps=40,
            **loop_kwargs,
        )
        handle._test_started = started
        if during is not None:
            await during(handle)
        result = await asyncio.wait_for(handle.result(), timeout=60)
        # A cancelled turn is still answered in the background.
        await asyncio.sleep(LLM_SECONDS + 0.3)
    assert result == expect
    return provider.requests, sent, handle


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


def _sent_while(request: dict, running: str) -> bool:
    """Whether *request* went out before the call returning *running* had."""
    return running not in json.dumps(request["messages"])


def _notices(request: dict) -> list[str]:
    """The lifecycle announcements and visibility message a request carries."""
    out = []
    for m in request["messages"]:
        text = str(m.get("content") or "")
        if text.startswith(("[steerable ", "[askable ")) or (
            "User Visibility Context" in text
        ):
            out.append(text)
    return out


def _append_only(requests: list[dict]) -> bool:
    """Each request's messages extend the previous request's, byte for byte."""
    for before, after in zip(requests, requests[1:]):
        prev = [json.dumps(m, sort_keys=True, default=str) for m in before["messages"]]
        nxt = [json.dumps(m, sort_keys=True, default=str) for m in after["messages"]]
        if nxt[: len(prev)] != prev:
            return False
    return True


# ── UNIFY_BATCH_WAKE ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_model_chosen_search_pair_wakes_the_model_once(monkeypatch):
    """The ARC LOW pattern: after the gate, the model calls both searches
    itself; the empty function search lands first, the guidance search later."""
    replies = [
        _batch(
            ("FunctionManager_search_functions", {"query": "q"}),
            ("GuidanceManager_search", {"query": "q"}),
        ),
        *_done(),
    ]
    requests, sent, handle = await _run(
        monkeypatch,
        replies,
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
    )
    assert len(requests) == 2, [_results_seen(r) for r in requests]
    assert _results_seen(requests[1]) == {"FM_EMPTY_RESULT", "GM_RESULT"}
    assert not _pending_placeholder(requests[1])
    assert sent[1] - sent[0] >= 0.4
    assert handle._runtime_state.cancelled_turns == 0

    # As shipped the same script starts a turn on the function search alone
    # and cancels it when the guidance search lands.
    requests, _, handle = await _run(monkeypatch, replies, switches=SHIPPED)
    assert {"FM_EMPTY_RESULT"} in [_results_seen(r) for r in requests[1:]]
    assert handle._runtime_state.cancelled_turns >= 1
    assert handle._runtime_state.cancelled_turns_by_cause.get("tool_result", 0) >= 1


@pytest.mark.asyncio
async def test_no_turn_starts_while_any_sibling_is_pending(monkeypatch):
    requests, sent, handle = await _run(
        monkeypatch,
        [
            _batch(
                "fast_tool",
                "medium_tool",
                ("FunctionManager_search_functions", {"query": "q"}),
            ),
            *_done(),
        ],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
    )
    # One turn after the batch, with no placeholder: nothing was pending.
    assert len(requests) == 2
    assert _results_seen(requests[1]) == {
        "FAST_RESULT",
        "MEDIUM_RESULT",
        "FM_EMPTY_RESULT",
    }
    assert not _pending_placeholder(requests[1])
    assert sent[1] - sent[0] >= 0.35
    assert handle._runtime_state.cancelled_turns == 0


@pytest.mark.asyncio
async def test_a_slow_batch_with_nothing_landed_is_not_cut_by_the_ceiling(
    monkeypatch,
):
    requests, sent, _ = await _run(
        monkeypatch,
        [_batch("slow_tool"), *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        ceiling=1.0,
    )
    # The ceiling bounds how long a landed result waits; with none landed
    # the model is woken once, by the result.
    assert len(requests) == 2
    assert _results_seen(requests[1]) == {"SLOW_RESULT"}
    assert sent[1] - sent[0] >= 2.4


@pytest.mark.asyncio
async def test_the_ceiling_wakes_the_model_with_the_results_so_far(monkeypatch):
    requests, sent, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), _batch("wait"), *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        ceiling=1.0,
    )
    # Woken about 1 s after the fast result, while the slow call still runs.
    assert _results_seen(requests[1]) == {"FAST_RESULT"}
    assert 0.9 <= sent[1] - sent[0] < 1.8
    # Then once more, when the slow call has finished.
    assert len(requests) == 3
    assert _results_seen(requests[2]) == {"FAST_RESULT", "SLOW_RESULT"}


@pytest.mark.asyncio
async def test_a_result_landing_during_a_sent_turn_does_not_cancel_it(monkeypatch):
    """Woken at the ceiling with the late call still running; it lands while
    that turn is in flight. The turn is kept, not cancelled and asked again."""
    requests, sent, handle = await _run(
        monkeypatch,
        [_batch("fast_tool", "late_tool", "slow_tool"), _batch("wait"), *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        ceiling=1.0,
    )
    assert _results_seen(requests[1]) == {"FAST_RESULT"}
    state = handle._runtime_state
    assert state.cancelled_turns == 0
    assert state.cancelled_turns_by_cause == {}
    # The kept turn's `wait` is answered by the next turn, which has every
    # result: the late one ingested after the kept turn, the slow one held.
    assert len(requests) == 3
    assert _results_seen(requests[2]) == {"FAST_RESULT", "LATE_RESULT", "SLOW_RESULT"}


@pytest.mark.asyncio
async def test_an_interjection_still_wakes_the_model_at_once(monkeypatch):
    async def interject(handle):
        await asyncio.sleep(0.3)
        await handle.interject("STOP_AND_LISTEN")

    requests, sent, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), _batch("wait"), *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        during=interject,
    )
    assert "STOP_AND_LISTEN" in json.dumps(requests[1]["messages"])
    assert sent[1] - sent[0] < 1.0


@pytest.mark.asyncio
async def test_a_clarification_still_wakes_the_model_at_once(monkeypatch):
    async def asker(
        _clarification_up_q: asyncio.Queue | None = None,
        _clarification_down_q: asyncio.Queue | None = None,
    ) -> str:
        """Ask a question, then return the answer."""
        await _clarification_up_q.put("WHICH_COLOUR?")
        answer = await _clarification_down_q.get()
        return f"ASKED_RESULT {answer}"

    tools = {"asker": asker, "slow_tool": slow_tool}
    replies = [
        lambda: h.completion(
            calls=[("asker", {}), ("slow_tool", {})],
            call_ids=["call_ask", "call_slow"],
        ),
        lambda: h.completion(
            calls=[
                (
                    "steer",
                    {"call_id": "call_ask", "action": "clarify", "payload": "blue"},
                ),
            ],
        ),
        *_done(),
    ]
    requests, sent, _ = await _run(
        monkeypatch,
        replies,
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        tools=tools,
    )
    assert "WHICH_COLOUR?" in json.dumps(requests[1]["messages"])
    assert sent[1] - sent[0] < 1.5
    # After the answer the model is woken once both calls have finished.
    assert _results_seen(requests[-1]) >= {"ASKED_RESULT blue", "SLOW_RESULT"}


def _notifier(at: float, total: float, text: str = "HALFWAY"):
    async def notifier(_notification_up_q: asyncio.Queue | None = None) -> str:
        """Report progress, then return."""
        await asyncio.sleep(at)
        await _notification_up_q.put({"message": text})
        await asyncio.sleep(total - at)
        return "NOTIFIER_RESULT"

    return notifier


@pytest.mark.asyncio
async def test_a_message_from_another_agent_wakes_the_model_at_once(monkeypatch):
    """A running agent's message (a progress notification) wakes the model
    while the batch still runs; the partial batch of results did not."""
    tools = {
        "fast_tool": fast_tool,
        "slow_tool": slow_tool,
        "notifier": _notifier(0.5, 2.5),
    }
    requests, sent, handle = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool", "notifier"), _batch("wait"), *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        tools=tools,
    )
    # Not woken by the fast result alone; woken by the message, promptly.
    assert "HALFWAY" in json.dumps(requests[1]["messages"])
    assert _results_seen(requests[1]) == {"FAST_RESULT"}
    assert 0.4 <= sent[1] - sent[0] < 1.2
    # Then once more, when the whole batch has finished.
    assert len(requests) == 3
    assert _results_seen(requests[2]) == {
        "FAST_RESULT",
        "SLOW_RESULT",
        "NOTIFIER_RESULT",
    }
    assert handle._runtime_state.cancelled_turns == 0


@pytest.mark.asyncio
async def test_a_notification_during_a_sent_turn_does_not_cancel_it(monkeypatch):
    """Woken at the ceiling; a message lands while that turn is in flight. The
    turn is kept, and the message wakes the model as soon as it has landed."""
    tools = {"fast_tool": fast_tool, "notifier": _notifier(1.3, 2.5)}
    requests, _, handle = await _run(
        monkeypatch,
        [_batch("fast_tool", "notifier"), *[_batch("wait")] * 2, *_done()],
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        tools=tools,
        ceiling=1.0,
    )
    assert handle._runtime_state.cancelled_turns == 0
    assert _results_seen(requests[1]) == {"FAST_RESULT"}
    assert "HALFWAY" not in json.dumps(requests[1]["messages"])
    assert "HALFWAY" in json.dumps(requests[2]["messages"])
    assert "NOTIFIER_RESULT" not in _results_seen(requests[2])
    assert len(requests) == 4
    assert "NOTIFIER_RESULT" in _results_seen(requests[3])


@pytest.mark.asyncio
async def test_batch_wake_grants_no_eager_gate_turn(monkeypatch):
    async def FunctionManager_search_functions(query: str) -> str:
        """Search stored functions.

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

    tools = {
        "FunctionManager_search_functions": FunctionManager_search_functions,
        "GuidanceManager_search": GuidanceManager_search,
    }
    replies = [
        _batch(("FunctionManager_search_functions", {"query": "q"})),
        _batch(("GuidanceManager_search", {"k": 3})),
        *_done(),
    ]
    requests, _, handle = await _run(
        monkeypatch,
        replies,
        switches={**SHIPPED, "UNIFY_BATCH_WAKE": True},
        tools=tools,
        tool_policy=h.gate_policy,
    )
    assert not _pending_placeholder(requests[1])
    assert "FM_RESULT" in json.dumps(requests[1]["messages"])
    # The gate still requires the other search on the turn it is woken for.
    assert requests[1]["tool_choice"] == "required"
    assert handle._runtime_state.cancelled_turns == 0


# ── UNIFY_PENDING_REQUIRED ──────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("required", [True, False], ids=["shipped", "off"])
async def test_a_turn_sent_while_calls_run_keeps_its_tool_choice(
    monkeypatch,
    required,
):
    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), *[_batch("wait")] * 4, *_done()],
        switches={**SHIPPED, "UNIFY_PENDING_REQUIRED": required},
    )
    # As shipped the fast result wakes the model while the slow call runs.
    while_pending = [r for r in requests[1:] if _sent_while(r, "SLOW_RESULT")]
    assert while_pending
    expected = "required" if required else "auto"
    assert {r["tool_choice"] for r in while_pending} == {expected}
    assert requests[0]["tool_choice"] == "auto"
    assert requests[-1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_a_gates_required_choice_is_kept_with_the_switch_off(monkeypatch):
    replies = [
        _batch(("FunctionManager_search_functions", {"query": "q"})),
        _batch(("GuidanceManager_search", {"query": "q"})),
        *_done(),
    ]
    requests, _, _ = await _run(
        monkeypatch,
        replies,
        switches={**SHIPPED, "UNIFY_PENDING_REQUIRED": False},
        tool_policy=h.gate_policy,
    )
    assert requests[0]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_pending_required_is_read_once_per_loop(monkeypatch):
    def flip():
        monkeypatch.setattr(SETTINGS, "UNIFY_PENDING_REQUIRED", True)

    requests, _, _ = await _run(
        monkeypatch,
        [_batch("fast_tool", "slow_tool"), *[_batch("wait")] * 4, *_done()],
        switches={**SHIPPED, "UNIFY_PENDING_REQUIRED": False},
        after_first=flip,
    )
    while_pending = [r for r in requests[1:] if _sent_while(r, "SLOW_RESULT")]
    assert while_pending
    assert {r["tool_choice"] for r in while_pending} == {"auto"}


def test_the_settings_default_to_as_shipped():
    settings = ProductionSettings()
    assert settings.UNIFY_BATCH_WAKE is False
    assert settings.UNIFY_PENDING_REQUIRED is True
    lean = ProductionSettings(UNIFY_BATCH_WAKE="1", UNIFY_PENDING_REQUIRED="0")
    assert lean.UNIFY_BATCH_WAKE is True
    assert lean.UNIFY_PENDING_REQUIRED is False
