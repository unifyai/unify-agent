"""Per-call cost rows by purpose (integration Task 22): decimal strings, unknown when unpriced."""

import asyncio
from types import SimpleNamespace

import pytest

from unify.memory_v2.episodes import CostRow
from unify.memory_v2.integration import cost
from unify.memory_v2.integration.cost import CostListener, money, recording_turn


def _event(
    provider_cost,
    origin="CodeActActor.act",
    model="openai/gpt-6-luna",
    usage=None,
):
    return SimpleNamespace(
        request={"model": model, "messages": []},
        response={"usage": usage or {"prompt_tokens": 10, "completion_tokens": 2}},
        provider_cost=provider_cost,
        origin=origin,
    )


def test_listener_rows_are_decimal_strings_or_unknown():
    lis = CostListener()
    lis.activate()
    for c in (0.00012, None, float("nan"), float("inf"), True, 1e-07):
        lis(_event(c))
    assert [r.usd for r in lis.rows] == [
        "0.00012",
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "0.0000001",
    ]
    assert lis.rows[0] == CostRow("actor", "openai/gpt-6-luna", 10, 2, "0.00012")
    assert all("e" not in r.usd.lower() or r.usd == "unknown" for r in lis.rows)


def test_nothing_is_recorded_while_inactive():
    lis = CostListener()
    lis(_event(0.1))
    assert lis.rows == []
    lis.activate()
    lis(_event(0.1))
    lis.deactivate()
    lis(_event(0.2))
    assert [r.usd for r in lis.rows] == ["0.1"]


def test_sol_origin_is_skipped_and_embedding_is_its_own_purpose():
    lis = CostListener()
    lis.activate()
    lis(_event(0.5, origin="memory_v2.sol"))
    lis(_event(0.5, origin="memory_v2.sol.pass"))
    lis(_event(0.25, origin="SemanticSearch.embed_query"))
    lis(_event(0.125, origin=None))
    assert [(r.purpose, r.usd) for r in lis.rows] == [
        ("embedding", "0.25"),
        ("actor", "0.125"),
    ]


def test_unreported_tokens_and_model_stay_unknown_and_bad_events_never_raise():
    lis = CostListener()
    lis.activate()
    lis(SimpleNamespace(request=None, response=None, provider_cost=0.01, origin="x"))
    lis(object())  # no attributes at all: still a call, of unknown cost; nothing raised
    assert lis.rows == [
        CostRow("actor", "unknown", None, None, "0.01"),
        CostRow("actor", "unknown", None, None, "unknown"),
    ]


@pytest.mark.parametrize(
    "value, want",
    [
        ("0.01", "0.01"),
        ("1E-7", "0.0000001"),
        ("0.3+unknown", "unknown"),
        ("unknown", "unknown"),
        ("-0.01", "unknown"),
        ("NaN", "unknown"),
        (None, "unknown"),
    ],
)
def test_money_is_a_plain_decimal_string_or_unknown(value, want):
    assert money(value) == want


def test_recording_turn_appends_one_sol_row_per_turn_and_unknown_on_error():
    calls = {"n": 0}

    async def turn(messages, tools):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"role": "assistant"}, "0.01"
        raise RuntimeError("provider down")

    rows: list[CostRow] = []
    wrapped = recording_turn(turn, rows, "openai/gpt-6-sol")
    msg, usd = asyncio.run(wrapped([], []))
    assert (msg, usd) == ({"role": "assistant"}, "0.01")
    with pytest.raises(RuntimeError):
        asyncio.run(wrapped([], []))
    assert rows == [
        CostRow("sol", "openai/gpt-6-sol", None, None, "0.01"),
        CostRow("sol", "openai/gpt-6-sol", None, None, "unknown"),
    ]


def test_recording_turn_never_copies_the_r22_form_into_a_row():
    async def turn(messages, tools):
        return {"role": "assistant"}, "0.2+unknown"

    rows: list[CostRow] = []
    _, usd = asyncio.run(recording_turn(turn, rows, "m")([], []))
    assert usd == "0.2+unknown"  # passed through unchanged to the pass
    assert rows == [CostRow("sol", "m", None, None, "unknown")]


def test_install_is_idempotent(monkeypatch):
    import unillm

    added = []
    monkeypatch.setattr(unillm, "add_llm_event_listener", lambda cb: added.append(cb))
    monkeypatch.setattr(cost, "_LISTENER", None)
    first = cost.install()
    assert cost.install() is first
    assert added == [first]
