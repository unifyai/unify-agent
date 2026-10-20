"""A pass's writer calls are priced from the proxy journal by lane and window; what
cannot be priced stays unknown, never zero, and is booked at the worst case."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from unify.memory.call_bounds import call_worst_case_usd
from unify.memory.reconcile import (
    UNKNOWN,
    estimate,
    journal_offset,
    lane_calls,
    money,
    reconcile_window,
)

WRITER_MODEL, ACTOR_MODEL = "openai/gpt-6-sol", "openai/gpt-6-luna"
#: one writer request's worst case: the pinned catalogue's bound for the writer's model (no default exists)
WRITER_CALL_WORST_CASE_USD = call_worst_case_usd(WRITER_MODEL)


def _write(path, rows):
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _started(rid, model=WRITER_MODEL):
    return {
        "origin": "request_started",
        "request_attempt_id": rid,
        "requested_model": model,
        "status": "started",
    }


def _done(rid, usd, model=WRITER_MODEL):
    return {
        "origin": "response",
        "request_attempt_id": rid,
        "requested_model": model,
        "status": "completed",
        "account_charge": usd,
    }


@pytest.mark.asyncio
async def test_a_cancelled_pass_prices_two_and_books_the_unpriced_one_at_the_worst_case(
    tmp_path,
):
    j = tmp_path / "costs.jsonl"
    _write(j, [_started("before")])  # an earlier pass's call: outside the window
    start = journal_offset(str(j))
    _write(
        j,
        [
            _started("a"),
            _started("actor", ACTOR_MODEL),
            _done("a", "0.10"),
            _started("b"),
            _started("c"),
        ],
    )
    end = journal_offset(str(j))
    # terminal rows written after the supervisor recorded the end still count for calls started inside
    _write(
        j,
        [
            _done("b", "0.25"),
            {
                "origin": "transport_error",
                "request_attempt_id": "c",
                "requested_model": WRITER_MODEL,
                "status": "cancelled",
                "error_type": "client disconnected",
            },
        ],
    )
    j.open("a").write('{"half a line')  # a line still being written is never read
    out = await reconcile_window(str(j), start, end, models=[WRITER_MODEL],
        worst_case_usd=WRITER_CALL_WORST_CASE_USD, budget_s=0,
    )
    assert out == {
        "journal": "read",
        "calls": 3,
        "priced": 2,
        "estimated": 0,
        "estimated_usd": "0",
        "unknown": 1,
        "usd": "0.35",
        "worst_case_usd": money(WRITER_CALL_WORST_CASE_USD),
        "booked_usd": money(Decimal("0.35") + WRITER_CALL_WORST_CASE_USD),
    }


def test_only_writer_lane_requests_started_in_the_window_belong_to_the_pass(tmp_path):
    j = tmp_path / "costs.jsonl"
    _write(j, [_started("x")])
    start = journal_offset(str(j))
    _write(
        j,
        [_started("actor", ACTOR_MODEL), _done("x", "1.00")],
    )  # x started before the window
    end = journal_offset(str(j))
    assert lane_calls(str(j), start, end, [WRITER_MODEL]) == {}


@pytest.mark.asyncio
async def test_reconcile_waits_for_a_late_terminal_row_within_its_budget(tmp_path):
    j = tmp_path / "costs.jsonl"
    start = journal_offset(str(j)) or 0
    _write(j, [_started("a")])
    end = journal_offset(str(j))
    clock = [0.0]

    async def sleep(s):
        clock[0] += s
        _write(
            j,
            [_done("a", "0.02")],
        )  # the proxy writes the row while reconcile waits

    out = await reconcile_window(
        str(j),
        start,
        end,
        models=[WRITER_MODEL],
        worst_case_usd=WRITER_CALL_WORST_CASE_USD,
        budget_s=30,
        clock=lambda: clock[0],
        sleep=sleep,
    )
    assert out["priced"] == 1 and out["unknown"] == 0 and out["usd"] == "0.02"


@pytest.mark.asyncio
async def test_no_journal_prices_nothing_and_never_says_zero(tmp_path):
    for path, start, end in (
        (None, 0, 1),
        (str(tmp_path / "missing.jsonl"), 0, 1),
        (str(tmp_path / "x"), None, None),
    ):
        out = await reconcile_window(
            path,
            start,
            end,
            models=[WRITER_MODEL],
            worst_case_usd=WRITER_CALL_WORST_CASE_USD,
            budget_s=0,
        )
        assert (
            out["journal"] == "unavailable"
            and out["unknown"] is None
            and out["booked_usd"] == UNKNOWN
        )


def test_money_is_a_decimal_string_or_unknown():
    assert money("0.10") == "0.10" and money(0) == "0" and money("-1") == UNKNOWN
    assert money(None) == UNKNOWN and money(True) == UNKNOWN and money("nan") == UNKNOWN


PRICES = {
    "input": Decimal("2.0"),
    "output": Decimal("10.0"),
    "cache_write": Decimal("2.5"),
}


@pytest.mark.asyncio
async def test_a_call_with_tokens_but_no_charge_is_priced_from_the_pinned_prices(
    tmp_path,
):
    j = tmp_path / "costs.jsonl"
    start = journal_offset(str(j)) or 0
    tokens = {
        "input": 100000,
        "output": 2000,
        "cache_write": 4000,
        "cache_read": None,
        "reasoning": None,
    }
    _write(
        j,
        [
            _started("a"),
            _done("a", "0.10"),
            _started("t"),
            {
                "origin": "response",
                "request_attempt_id": "t",
                "requested_model": WRITER_MODEL,
                "status": "completed",
                "account_charge": None,
                "tokens": tokens,
            },
            _started("n"),  # no charge and no tokens: still the worst case
        ],
    )
    end = journal_offset(str(j))
    out = await reconcile_window(
        str(j),
        start,
        end,
        models=[WRITER_MODEL],
        worst_case_usd=WRITER_CALL_WORST_CASE_USD,
        budget_s=0,
        prices=PRICES,
    )
    # 100000 x 2.0 + 2000 x 10.0 + 4000 x 2.5 = 230000 per million
    assert out["priced"] == 1 and out["estimated"] == 1 and out["unknown"] == 1
    assert out["estimated_usd"] == "0.23" and out["usd"] == "0.33"
    assert out["booked_usd"] == money(Decimal("0.33") + WRITER_CALL_WORST_CASE_USD)
    # without pinned prices the same call stays unknown
    bare = await reconcile_window(str(j), start, end, models=[WRITER_MODEL],
        worst_case_usd=WRITER_CALL_WORST_CASE_USD, budget_s=0,
    )
    assert bare["estimated"] == 0 and bare["unknown"] == 2


def test_the_estimate_needs_prices_and_input_or_output_counts():
    assert estimate({"input": 10, "output": None}, PRICES) == Decimal("0.00002")
    assert estimate({"input": None, "output": None, "cache_write": 5}, PRICES) is None
    assert estimate({"input": "x"}, PRICES) is None
    assert estimate({"input": -1, "output": 1}, PRICES) is None
    assert estimate({"input": 10}, None) is None
    no_cache = {"input": Decimal("2.0"), "output": Decimal("10.0")}
    assert estimate({"input": 0, "output": 0, "cache_write": 1000000}, no_cache) == 0


def test_the_prices_setting_is_parsed_strictly():
    from unify.memory.integration import switch

    assert switch.parse_writer_prices("") == ""
    got = switch.parse_writer_prices(
        '{"output": 10.0, "input": "2.0", "cache_write": 2.5}',
    )
    assert got == '{"cache_write": "2.5", "input": "2.0", "output": "10.0"}'
    for bad in (
        '{"input": 2}',
        '{"input": 2, "output": 1, "other": 1}',
        "[1]",
        "x",
        '{"input": -1, "output": 1}',
        '{"input": true, "output": 1}',
    ):
        with pytest.raises(ValueError):
            switch.parse_writer_prices(bad)
    assert switch.PARSERS[switch.WRITER_PRICES] is switch.parse_writer_prices


@pytest.mark.asyncio
async def test_there_is_no_default_worst_case(tmp_path):
    """Review: an unpriced call is booked at the worst case the caller gives for the writer's model; there is no
    module default to fall back to."""
    import unify.memory.reconcile as R

    assert not hasattr(R, "WRITER_CALL_WORST_CASE_USD")
    with pytest.raises(TypeError):
        await reconcile_window(str(tmp_path / "j.jsonl"), 0, 0, models=[WRITER_MODEL], budget_s=0)
    with pytest.raises(TypeError):
        R.unavailable()  # noqa: the worst case is required
