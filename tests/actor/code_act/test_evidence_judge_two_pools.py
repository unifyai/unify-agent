"""Symbolic: ``UNIFY_EVIDENCE_LIST_MATCHER=judge2``, the evidence list's judge with two pools.

In MEMORY's offline replay the one-pool judge listed a generic note on 71% of
the ARC task starts where nothing fit, and a generic note at confidence 99 beat
the entries sharing the request's identifier. Two pools fix both: the entries
sharing an *evidence* identifier (not a number with a unit, a date, a time,
nor a token of the request's paths or of most requests) are judged alone and
their pick is listed from confidence 50; the closest other cards are judged
apart and listed only from 90. These tests pin that policy; the model is a
scripted coroutine and nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.actor.code_act import test_evidence_judge as tej
from unify.actor import evidence_judge as ej
from unify.actor import evidence_list as ev
from unify.settings import SETTINGS

INVOICE = tej._function(
    5,
    "reconcile_invoice",
    "Reconcile one invoice against the ledger.",
    "Reconcile invoice INV-20417 with ledger.csv",
)
REQUEST = "Email the vendor about invoice INV-20417 being late."
LOGGED = [
    tej.SPEND,
    "Reconcile invoice INV-20417 with ledger.csv",
    "Plan the offsite.",
    REQUEST,
]


@pytest.fixture(autouse=True)
def two_pools(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LIST_MATCHER", "judge2")


def _replies(keyed: str, rest: str):
    """A judge that answers the identifier pool with *keyed* and the other pool with *rest*."""
    prompts = []

    async def generate(text):
        prompts.append(text)
        return keyed if "its request also named" in text else rest

    return generate, prompts


def _aselect(lib, request, generate, logged):
    return asyncio.run(
        ev.aselect(
            lib,
            request,
            request,
            None,
            list(logged),
            {},
            threshold=0.175,
            embed=tej._embed,
            generate=generate,
        ),
    )


def _listed(listing):
    return [card.key() for card, _ in listing.seen]


def test_numbers_with_units_dates_and_times_are_not_evidence():
    for word in ("30-minute", "3rd", "Q3", "12kg", "2024-sep"):
        assert ej.unit_token(word), word
    for word in ("INV-20417", "task-89bde01c", "E1338", "sku-1049"):
        assert not ej.unit_token(word), word


def test_tokens_of_paths_and_of_most_requests_are_environment():
    request = "Summarise /home/alice99/reports/q3.csv for team-tx42 by Friday."
    earlier = [f"Request {i} for team-tx42." for i in range(4)]
    env = ej.environment_tokens(request, earlier)
    assert "alice99" in env and "team-tx42" in env
    assert "team-tx42" not in ej.environment_tokens(request, earlier[:3])


def test_the_identifier_pool_is_judged_alone_and_listed_from_fifty():
    lib = ev.build_library([tej.TOTAL, INVOICE], [tej.NOTE], [])
    generate, prompts = _replies(
        '{"choice": "E1", "confidence": 60}',
        '{"choice": "none", "confidence": 95}',
    )
    listing = _aselect(lib, REQUEST, generate, LOGGED)
    assert _listed(listing) == [("function", "reconcile_invoice")]
    keyed_prompt = next(p for p in prompts if "its request also named" in p)
    assert (
        "E1: reconcile_invoice" in keyed_prompt and "total_extreme" not in keyed_prompt
    )
    other = [p for p in prompts if p is not keyed_prompt]
    assert all("reconcile_invoice" not in p for p in other)


def test_an_identifier_pick_below_fifty_is_not_listed():
    lib = ev.build_library([INVOICE], [], [])
    generate, _ = _replies('{"choice": "E1", "confidence": 40}', '{"choice": "none"}')
    assert _listed(_aselect(lib, REQUEST, generate, LOGGED)) == []


@pytest.mark.parametrize("confidence, listed", [(85, False), (90, True), (99, True)])
def test_a_pick_without_identifier_evidence_needs_ninety(confidence, listed):
    generate, prompts = _replies(
        '{"choice": "none"}',
        f'{{"choice": "E1", "confidence": {confidence}}}',
    )
    listing = _aselect(tej.LIB, tej.SPEND_AGAIN, generate, [tej.SPEND, tej.SPEND_AGAIN])
    assert len(prompts) == 1
    assert bool(listing.seen) is listed


def test_a_shared_number_with_a_unit_does_not_make_an_entry_keyed():
    timed = tej._function(
        6,
        "book_slot",
        "Book a meeting slot.",
        "Book a 30-minute slot with Dana.",
    )
    lib = ev.build_library([timed], [], [])
    request = "Find a 30-minute window to review the budget."
    generate, prompts = _replies(
        '{"choice": "E1", "confidence": 99}',
        '{"choice": "none"}',
    )
    listing = _aselect(
        lib,
        request,
        generate,
        ["Book a 30-minute slot with Dana.", "Plan the offsite.", request],
    )
    assert all("its request also named" not in p for p in prompts)
    assert listing.seen == []


def test_an_exact_repeat_is_listed_and_the_pools_judge_the_rest():
    generate, prompts = _replies('{"choice": "none"}', '{"choice": "none"}')
    listing = _aselect(tej.LIB, tej.SPEND, generate, [tej.SPEND])
    assert _listed(listing) == [("function", "total_extreme_payments")]
    assert listing.seen[0][1].rank == 3
    assert all("E1: total_extreme_payments" not in p for p in prompts)


def test_the_setting_accepts_judge2_and_the_list_counts_as_judged(monkeypatch):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(
            UNIFY_EVIDENCE_LIST_MATCHER="judge2",
        ).UNIFY_EVIDENCE_LIST_MATCHER
        == "judge2"
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LIST", "on")
    assert ev.judged()
