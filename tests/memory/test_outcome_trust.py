"""Outcome-driven trust, one rule set per signal regime, with the renamed statuses.

An item's status moves ``on-trial`` -> ``adopted`` -> ``retired`` on outcomes alone, and an outcome counts for a
procedure only for a use whose result was handed back (``answered``). What counts depends on the episode's regime:

- ``dense``: the visible verdict after the answered action. A rejected result counts against the procedure for that
  use, even when the episode later succeeded another way.
- ``sparse``: the episode-end outcome, plus intermediate signals (errors, refusals, ``holds``).
- ``implicit`` and ``none``: implicit signals only (it ran without error, its conditions held, nothing corrected it).
  The record never claims correctness: it says "used without problems in N episodes", on an ``implicit`` basis.

A refusal is never a failure. A failed use becomes a ``failure`` pool entry that carries the request as a fixture
source, so a repair (or a known-failure test alone) turns the failure into a test.

Every name this file assumes for code that does not exist yet is in the block of constants below.
"""

from __future__ import annotations

import hashlib
import importlib
import re

import pytest

from unify.memory import lifecycle as lc
from unify.memory.fixtures import canonical

# ---------------------------------------------------------------------------------------------------------------
# Assumed names (the spec leaves these open; adapt them here)
# ---------------------------------------------------------------------------------------------------------------
ON_TRIAL, ADOPTED, RETIRED = "on-trial", "adopted", "retired"
OLD_STATUSES = ("experimental", "stable", "suspect")
LEVELS = ("unchecked", "reproduces-its-case", "confirmed-by-outcomes")
LEVELS_HOME = ("unify.memory.trust", "LEVELS")
#: ItemUse gains these fields (keyword arguments with defaults).
ACCEPTED, REJECTED = "accepted", "rejected"
POSITIVE, CONTRARY = "positive", "contrary"
HELD, FAILED_CONDITIONS = "held", "failed"
#: The record's basis for an adopted status.
BASIS_VERDICTS, BASIS_IMPLICIT = "verdicts", "implicit"
#: The implicit-regime wording of a clean record.
CLEAN_IMPLICIT_TEXT = "used without problems in {n} episodes"
#: The track record's groups and counts.
TRACK_GROUPS = {
    "verdicts": ("accepted", "rejected"),
    "end_outcomes": ("positive", "contrary"),
    "intermediate": ("errors", "refusals", "holds_true", "holds_false"),
    "implicit": ("ran_ok", "agreed", "disagreed", "corrected", "reused_episodes"),
}
#: Which signal failed, as a failed-use row names it, for a rejected verdict.
FAILED_BY_VERDICT = "rejected"
#: The pool kind for a failed use.
POOL_FAILURE_KIND = "failure"


def _track(item: str, facts: list) -> dict:
    """``lifecycle.aggregate_track(facts)[item]``, unwrapped from a per-commit mapping when it is keyed so."""
    out = lc.aggregate_track(facts)[item]
    if set(out) <= {f.memory_main for f in facts}:
        (out,) = out.values()
    return out


def _basis(item: str, facts: list) -> str | None:
    return lc.status_basis(item, facts)


def _failed_uses(facts: list) -> list[dict]:
    return lc.failed_uses(facts)


def _result_digest(value) -> str | None:
    from unify.memory.analysis.use import result_digest

    return result_digest(value)


def _failure_entry(**kw):
    from unify.memory import pool

    return pool.failure(**kw)


# ---------------------------------------------------------------------------------------------------------------

ITEM = "memory.lists.order:sort_numbers"
COMMIT = "a" * 40
CORRECTNESS = re.compile(r"(?i)\b(?:correct|correctly|right|verified|accurate)\b")


def use(**kw) -> lc.ItemUse:
    """A use whose call returned, on the runtime basis; *kw* sets the outcome fields."""
    base = dict(
        used=True,
        uses=1,
        refused=False,
        errored=False,
        unknown=False,
        basis=lc.RUNTIME,
        calls=(1, 0, 0),
    )
    base.update(kw)
    return lc.ItemUse(**base)


def ep(eid: str, u: lc.ItemUse, regime: str, *, pos=False, neg=False) -> lc.EpisodeUse:
    return lc.EpisodeUse(eid, COMMIT, regime, pos, neg, {ITEM: u})


def status(facts, prior=None) -> str:
    return lc.decide_status(ITEM, prior, facts)[0]


def _clean_of_regime(regime: str, n: int) -> list:
    return [
        ep(f"e{i}", use(answered=True, conditions=HELD, corrected=False), regime)
        for i in range(n)
    ]


# --- the renamed statuses and levels ---------------------------------------------------------------------------


def test_statuses_and_levels_carry_the_new_names():
    from unify.memory import library_index as li

    assert {ON_TRIAL, ADOPTED, RETIRED} <= set(li.STATUSES)
    assert not set(OLD_STATUSES) & set(li.STATUSES)
    assert RETIRED in li.HIDDEN and ADOPTED not in li.HIDDEN and ON_TRIAL not in li.HIDDEN
    mod, name = LEVELS_HOME
    assert tuple(getattr(importlib.import_module(mod), name)) == LEVELS
    # a level and a status never share a name
    assert not set(LEVELS) & set(li.STATUSES)


def test_a_new_item_is_on_trial_and_one_episode_never_decides():
    assert status([]) == ON_TRIAL
    one = [ep("e1", use(answered=True, verdict=REJECTED), "dense", neg=True)]
    assert status(one) == ON_TRIAL


# --- dense: verdicts update the record -------------------------------------------------------------------------


def test_dense_two_rejected_answers_retire_and_three_accepted_adopt():
    rejected = [
        ep(f"e{i}", use(answered=True, verdict=REJECTED), "dense", neg=True)
        for i in range(2)
    ]
    assert status(rejected) == RETIRED
    t = _track(ITEM, rejected)
    assert (t["verdicts"]["accepted"], t["verdicts"]["rejected"]) == (0, 2)

    accepted = [
        ep(f"e{i}", use(answered=True, verdict=ACCEPTED), "dense", pos=True)
        for i in range(3)
    ]
    assert status(accepted) == ADOPTED
    assert _basis(ITEM, accepted) == BASIS_VERDICTS
    t = _track(ITEM, accepted)
    assert (t["verdicts"]["accepted"], t["verdicts"]["rejected"]) == (3, 0)


def test_dense_a_rejection_counts_even_when_the_episode_later_succeeded():
    # the procedure's answer was rejected; the actor then succeeded another way, so the episode ends positive
    later_won = [
        ep(f"e{i}", use(answered=True, verdict=REJECTED), "dense", pos=True, neg=False)
        for i in range(2)
    ]
    assert status(later_won) == RETIRED
    assert _track(ITEM, later_won)["verdicts"]["rejected"] == 2
    rows = [r for r in _failed_uses(later_won) if r["item"] == ITEM]
    assert sorted(r["episode"] for r in rows) == ["e0", "e1"]
    assert all(r["signal"] == FAILED_BY_VERDICT for r in rows)


def test_dense_one_rejection_blocks_adoption():
    facts = [
        ep(f"e{i}", use(answered=True, verdict=ACCEPTED), "dense", pos=True)
        for i in range(3)
    ] + [ep("e3", use(answered=True, verdict=REJECTED), "dense", pos=True)]
    assert status(facts) == ON_TRIAL


def test_dense_uses_that_were_not_handed_back_carry_no_verdict():
    # returned, but the result handed back was something else: neither the episode's win nor its loss is theirs
    facts = [
        ep(f"e{i}", use(answered=False), "dense", pos=True) for i in range(3)
    ]
    assert status(facts) != ADOPTED
    t = _track(ITEM, facts)
    assert (t["verdicts"]["accepted"], t["verdicts"]["rejected"]) == (0, 0)
    # a verdict attached to an unanswered use is never read
    stray = [
        ep(f"e{i}", use(answered=False, verdict=REJECTED), "dense", neg=True)
        for i in range(2)
    ]
    assert _track(ITEM, stray)["verdicts"]["rejected"] == 0


def test_dense_adopted_item_is_retired_by_later_rejections():
    accepted = [
        ep(f"e{i}", use(answered=True, verdict=ACCEPTED), "dense", pos=True)
        for i in range(3)
    ]
    rejected = [
        ep(f"e{3 + i}", use(answered=True, verdict=REJECTED), "dense", neg=True)
        for i in range(2)
    ]
    assert status(accepted + rejected[:1], prior=ADOPTED) == ADOPTED  # one episode never decides
    assert status(accepted + rejected, prior=ADOPTED) == RETIRED


# --- sparse: an episode-end outcome plus intermediate signals --------------------------------------------------


def test_sparse_answers_in_episodes_that_ended_contrary_retire():
    facts = [
        ep(f"e{i}", use(answered=True, end_outcome=CONTRARY), "sparse", neg=True)
        for i in range(2)
    ]
    assert status(facts) == RETIRED
    assert _track(ITEM, facts)["end_outcomes"]["contrary"] == 2


def test_sparse_intermediate_errors_and_failed_conditions_retire():
    errors = [
        ep(f"e{i}", use(errored=True, calls=(0, 0, 1), used=False), "sparse")
        for i in range(2)
    ]
    assert status(errors) == RETIRED
    assert _track(ITEM, errors)["intermediate"]["errors"] == 2

    holds_false = [
        ep(f"e{i}", use(answered=True, conditions=FAILED_CONDITIONS), "sparse")
        for i in range(2)
    ]
    assert status(holds_false) == RETIRED
    assert _track(ITEM, holds_false)["intermediate"]["holds_false"] == 2


def test_sparse_needs_a_positive_end_before_adoption():
    no_end = [
        ep(f"e{i}", use(answered=True, conditions=HELD), "sparse") for i in range(3)
    ]
    assert status(no_end) == ON_TRIAL
    ended_well = [
        ep(
            f"e{i}",
            use(answered=True, conditions=HELD, end_outcome=POSITIVE if i == 0 else None),
            "sparse",
            pos=i == 0,
        )
        for i in range(3)
    ]
    assert status(ended_well) == ADOPTED
    assert _basis(ITEM, ended_well) != BASIS_IMPLICIT
    t = _track(ITEM, ended_well)
    assert t["end_outcomes"]["positive"] == 1
    assert t["intermediate"]["holds_true"] == 3


# --- implicit and none: implicit signals only, and no claim of correctness ------------------------------------


@pytest.mark.parametrize("regime", ["implicit", "none"])
def test_no_reward_three_clean_runs_adopt_on_an_implicit_basis(regime):
    facts = _clean_of_regime(regime, 3)
    st, _rule, reason, _eps = lc.decide_status(ITEM, None, facts)
    assert st == ADOPTED
    assert _basis(ITEM, facts) == BASIS_IMPLICIT
    assert CLEAN_IMPLICIT_TEXT.format(n=3) in (reason or "")
    assert not CORRECTNESS.search(reason or "")
    t = _track(ITEM, facts)
    assert t["implicit"]["ran_ok"] == 3
    assert (t["verdicts"]["accepted"], t["verdicts"]["rejected"]) == (0, 0)


@pytest.mark.parametrize("regime", ["implicit", "none"])
def test_no_reward_never_reads_a_verdict(regime):
    # a checker verdict is not observable in these regimes: neither an accepted nor a rejected one counts
    accepted = [
        ep(f"e{i}", use(answered=True, verdict=ACCEPTED, conditions=HELD), regime)
        for i in range(3)
    ]
    assert _track(ITEM, accepted)["verdicts"]["accepted"] == 0
    assert _basis(ITEM, accepted) != BASIS_VERDICTS
    rejected = [
        ep(f"e{i}", use(answered=True, verdict=REJECTED, conditions=HELD), regime)
        for i in range(2)
    ]
    assert status(rejected) != RETIRED
    assert _track(ITEM, rejected)["verdicts"]["rejected"] == 0


def test_implicit_a_correction_after_an_answer_twice_retires():
    facts = [
        ep(f"e{i}", use(answered=True, corrected=True), "implicit", neg=True)
        for i in range(2)
    ]
    assert status(facts) == RETIRED
    assert _track(ITEM, facts)["implicit"]["corrected"] == 2


@pytest.mark.parametrize("regime", ["implicit", "none"])
def test_no_reward_failed_conditions_or_errors_twice_retire(regime):
    holds_false = [
        ep(f"e{i}", use(answered=True, conditions=FAILED_CONDITIONS), regime)
        for i in range(2)
    ]
    assert status(holds_false) == RETIRED
    errors = [
        ep(f"e{i}", use(errored=True, used=False, calls=(0, 0, 1)), regime)
        for i in range(2)
    ]
    assert status(errors) == RETIRED


@pytest.mark.parametrize("regime", ["implicit", "none"])
def test_no_reward_records_never_claim_correctness(regime):
    facts = _clean_of_regime(regime, 3) + [
        ep("e9", use(answered=True, conditions=FAILED_CONDITIONS), regime),
    ]
    for f in (facts[:3], facts, facts[:1]):
        _st, _rule, reason, _eps = lc.decide_status(ITEM, None, f)
        assert not CORRECTNESS.search(reason or ""), reason
    # the track's own keys and values say nothing about correctness either
    assert not CORRECTNESS.search(repr(_track(ITEM, facts)))


# --- the track record --------------------------------------------------------------------------------------------


def test_the_track_has_every_group_and_count():
    t = _track(ITEM, _clean_of_regime("implicit", 1))
    for group, counts in TRACK_GROUPS.items():
        assert group in t, group
        for c in counts:
            assert isinstance(t[group][c], int) and not isinstance(t[group][c], bool), (group, c)


def test_a_refusal_changes_nothing():
    refused = [
        ep(
            f"r{i}",
            use(used=False, refused=True, calls=(0, 1, 0)),
            regime,
        )
        for i, regime in enumerate(["dense", "sparse", "implicit"])
    ]
    assert status(refused) == ON_TRIAL
    assert status(_clean_of_regime("implicit", 3) + refused) == ADOPTED
    t = _track(ITEM, refused)
    assert t["verdicts"] == {"accepted": 0, "rejected": 0}
    assert t["implicit"]["ran_ok"] == 0
    assert _failed_uses(refused) == []


def test_the_result_digest_is_the_canonical_json_digest():
    value = {"b": [3, 1], "a": "x"}
    want = hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:16]
    assert _result_digest(value) == want
    assert _result_digest({"a": "x", "b": [3, 1]}) == want  # key order never matters
    assert _result_digest([1, 3]) != _result_digest([3, 1])


# --- repair and re-save: the failure becomes a test ----------------------------------------------------------


REQUEST = "Sort the numbers below in ascending order and reply with the list.\n5 3 9 1\n"


def test_a_failed_use_becomes_a_failure_entry_carrying_the_request_as_a_fixture_source():
    from unify.memory import pool

    assert POOL_FAILURE_KIND in pool.KINDS
    digest = _result_digest([9, 5, 3, 1])
    e = _failure_entry(
        item=ITEM,
        episode="e7",
        signal=FAILED_BY_VERDICT,
        result_digest=digest,
        request=REQUEST,
    )
    assert e.kind == POOL_FAILURE_KIND
    assert e.status == "open"
    assert e.provenance["episodes"] == ["e7"]
    assert e.data["item"] == ITEM
    assert e.data["signal"] == FAILED_BY_VERDICT
    assert e.data["result_digest"] == digest
    assert REQUEST in e.files.values()  # the input a repair's red test, or a known-failure test, is built on
    again = _failure_entry(
        item=ITEM,
        episode="e7",
        signal=FAILED_BY_VERDICT,
        result_digest=digest,
        request=REQUEST,
    )
    assert again.id == e.id
    other = _failure_entry(
        item=ITEM,
        episode="e8",
        signal=FAILED_BY_VERDICT,
        result_digest=digest,
        request=REQUEST,
    )
    assert other.id != e.id


def test_every_failed_use_is_listed_for_the_writer_with_the_signal_that_failed():
    facts = [
        ep("e1", use(answered=True, verdict=REJECTED), "dense", pos=True),
        ep("e2", use(answered=True, end_outcome=CONTRARY), "sparse", neg=True),
        ep("e3", use(answered=True, conditions=FAILED_CONDITIONS), "implicit"),
        ep("e4", use(answered=True, corrected=True), "implicit", neg=True),
        ep("e5", use(answered=True, verdict=ACCEPTED), "dense", pos=True),
        ep("e6", use(answered=False), "dense", neg=True),
    ]
    rows = {r["episode"]: r for r in _failed_uses(facts) if r["item"] == ITEM}
    assert sorted(rows) == ["e1", "e2", "e3", "e4"]
    assert len({r["signal"] for r in rows.values()}) == 4  # each names its own signal


def test_a_repaired_item_starts_again_on_trial():
    # its facts are those of the episodes that ran its new version only
    old = [
        ep(f"e{i}", use(answered=True, verdict=REJECTED), "dense", neg=True)
        for i in range(2)
    ]
    assert status(old) == RETIRED
    new_main = "b" * 40
    new = [
        lc.EpisodeUse("e9", new_main, "dense", True, False, {ITEM: use(answered=True, verdict=ACCEPTED)}),
    ]
    assert status(new) == ON_TRIAL


def test_the_writer_brief_offers_a_known_failure_test():
    from unify.memory import library_prompts as lp

    brief = lp.write_brief_now()
    assert "a test marked as a known failure" in brief
