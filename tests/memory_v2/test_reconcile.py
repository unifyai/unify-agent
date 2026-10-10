"""Spec v2.1 §6, P7 Amendment D: a pass's Sol calls are priced from the proxy journal by lane and window; what
cannot be priced stays unknown, never zero, and is booked at the worst case."""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from unify.memory_v2.reconcile import (
    SOL_CALL_WORST_CASE_USD,
    UNKNOWN,
    journal_offset,
    lane_calls,
    money,
    reconcile_window,
)

SOL, LUNA = "openai/gpt-6-sol", "openai/gpt-6-luna"


def _write(path, rows):
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _started(rid, model=SOL):
    return {
        "origin": "request_started",
        "request_attempt_id": rid,
        "requested_model": model,
        "status": "started",
    }


def _done(rid, usd, model=SOL):
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
            _started("actor", LUNA),
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
                "requested_model": SOL,
                "status": "cancelled",
                "error_type": "client disconnected",
            },
        ],
    )
    j.open("a").write('{"half a line')  # a line still being written is never read
    out = await reconcile_window(str(j), start, end, models=[SOL], budget_s=0)
    assert out == {
        "journal": "read",
        "calls": 3,
        "priced": 2,
        "unknown": 1,
        "usd": "0.35",
        "worst_case_usd": money(SOL_CALL_WORST_CASE_USD),
        "booked_usd": money(Decimal("0.35") + SOL_CALL_WORST_CASE_USD),
    }


def test_only_sol_lane_requests_started_in_the_window_belong_to_the_pass(tmp_path):
    j = tmp_path / "costs.jsonl"
    _write(j, [_started("x")])
    start = journal_offset(str(j))
    _write(
        j, [_started("luna", LUNA), _done("x", "1.00")]
    )  # x started before the window
    end = journal_offset(str(j))
    assert lane_calls(str(j), start, end, [SOL]) == {}


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
            j, [_done("a", "0.02")]
        )  # the proxy writes the row while reconcile waits

    out = await reconcile_window(
        str(j),
        start,
        end,
        models=[SOL],
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
        out = await reconcile_window(path, start, end, models=[SOL], budget_s=0)
        assert (
            out["journal"] == "unavailable"
            and out["unknown"] is None
            and out["booked_usd"] == UNKNOWN
        )


def test_money_is_a_decimal_string_or_unknown():
    assert money("0.10") == "0.10" and money(0) == "0" and money("-1") == UNKNOWN
    assert money(None) == UNKNOWN and money(True) == UNKNOWN and money("nan") == UNKNOWN
