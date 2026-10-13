"""Outcome-driven trust per signal regime: what each use counts as, promotion and retirement from outcomes, the level
outcomes give, and the lifecycle delegating to the one decider."""

from __future__ import annotations

import re

from unify.memory import lifecycle as lc
from unify.memory import trust as tr
from unify.memory.analysis.use import attribute_outcomes, result_digest
from unify.memory.signals import Signal

ITEM = "memory.text.tables:totals"


def _use(eid, regime, **kw):
    return tr.Use(eid, regime, **kw)


_CLAIM = re.compile(r"\b(correct|right|verified)\b", re.I)


def _no_correctness_claim(*texts):
    for t in texts:
        assert t is None or not _CLAIM.search(t), t


# --- dense: verdicts update the record --------------------------------------------------------------------------


def test_dense_rejected_answers_twice_retire_and_accepted_answers_adopt_with_verdicts():
    rejected = [_use(f"e{i}", "dense", answered=True, verdict="rejected") for i in range(2)]
    out = tr.apply_outcome(tr.REPRODUCES_ITS_CASE, None, rejected)
    assert (out.status, out.rule) == (tr.RETIRED, "rejected")
    assert out.episodes == ("e0", "e1")
    accepted = [_use(f"e{i}", "dense", answered=True, verdict="accepted") for i in range(3)]
    out = tr.apply_outcome(tr.REPRODUCES_ITS_CASE, None, accepted)
    assert (out.status, out.basis, out.level) == (tr.ADOPTED, tr.BASIS_VERDICTS, tr.CONFIRMED_BY_OUTCOMES)
    t = tr.track(accepted)
    assert t["verdicts"] == {"accepted": 3, "rejected": 0}
    assert tr.record_line(t) == "its record has verdicts: 3 accepted, 0 rejected"


def test_dense_clean_uses_without_any_accepted_verdict_stay_on_trial():
    quiet = [_use(f"e{i}", "dense", answered=False) for i in range(3)]
    out = tr.apply_outcome(tr.REPRODUCES_ITS_CASE, None, quiet)
    assert out.status == tr.ON_TRIAL and "accepted verdict or a positive end" in out.reason
    assert out.level == tr.REPRODUCES_ITS_CASE


def test_a_rejection_counts_against_the_procedure_even_when_the_episode_later_succeeded():
    # the answered use was rejected; the episode then ended positive another way
    u = _use("e1", "dense", answered=True, verdict="rejected", end_outcome="positive")
    assert tr.judge(u)[:2] == (tr.FAILED, "rejected")
    out = tr.apply_outcome(None, None, [u, _use("e2", "dense", answered=True, verdict="rejected")])
    assert out.status == tr.RETIRED


def test_a_use_whose_result_was_not_handed_back_gets_no_verdict():
    record = {"items": {}}
    results = {ITEM: [{"outcome": "ok", "result": result_digest([1, 2])}]}
    attribute_outcomes(record, results, [[3]], verdict="rejected", end_outcome="contrary", corrected=True)
    [row] = record["items"][ITEM]["outcomes"]
    assert row["answered"] is False and row["verdict"] is None and row["end_outcome"] is None
    assert row["corrected"] is False and row["conditions"] == "none"
    record = {"items": {}}
    attribute_outcomes(record, results, [[1, 2]], verdict="rejected")
    [row] = record["items"][ITEM]["outcomes"]
    assert row["answered"] is True and row["verdict"] == "rejected"


# --- sparse: the end outcome plus intermediate signals ------------------------------------------------------------


def test_sparse_answer_in_a_contrary_episode_and_an_error_mid_episode_each_count_as_failed():
    assert tr.judge(_use("e1", "sparse", answered=True, end_outcome="contrary"))[:2] == (
        tr.FAILED,
        "contrary_end",
    )
    assert tr.judge(_use("e2", "sparse", outcome="error"))[:2] == (tr.FAILED, "error")
    assert tr.judge(_use("e3", "sparse", holds=False))[:2] == (tr.FAILED, "holds_false")
    mixed = [
        _use("e1", "sparse", answered=True, end_outcome="contrary"),
        _use("e2", "sparse", outcome="error"),
    ]
    out = tr.apply_outcome(None, None, mixed)
    assert (out.status, out.rule) == (tr.RETIRED, "failed_uses")
    good = [_use(f"p{i}", "sparse", answered=True, end_outcome="positive") for i in range(3)]
    out = tr.apply_outcome(tr.UNCHECKED, None, good)
    assert (out.status, out.basis, out.level) == (tr.ADOPTED, tr.BASIS_VERDICTS, tr.CONFIRMED_BY_OUTCOMES)


# --- implicit and none: implicit signals only, never a claim of correctness ---------------------------------------


def test_implicit_and_none_adopt_on_clean_runs_and_never_claim_correctness():
    for regime in ("implicit", "none"):
        clean = [_use(f"e{i}", regime, answered=True, agreed=True) for i in range(3)]
        out = tr.apply_outcome(tr.REPRODUCES_ITS_CASE, None, clean)
        assert (out.status, out.basis) == (tr.ADOPTED, tr.BASIS_IMPLICIT)
        assert out.level == tr.REPRODUCES_ITS_CASE  # implicit outcomes never confirm
        assert tr.status_text(out.status, out.basis) == "adopted (implicit)"
        line = tr.record_line(tr.track(clean))
        assert line == "its record has no verdicts; it ran without errors in 3 episodes"
        _no_correctness_claim(line, out.reason)


def test_implicit_corrections_after_answers_retire_and_none_cannot_see_them():
    corrected = [_use(f"e{i}", "implicit", answered=True, corrected=True) for i in range(2)]
    out = tr.apply_outcome(None, None, corrected)
    assert (out.status, out.rule) == (tr.RETIRED, "corrected")
    _no_correctness_claim(out.reason)
    unseen = [_use(f"e{i}", "none", answered=True, corrected=True) for i in range(3)]
    assert tr.apply_outcome(None, None, unseen).status == tr.ADOPTED
    disagreed = [_use(f"e{i}", "none", agreed=False) for i in range(2)]
    assert tr.apply_outcome(None, None, disagreed).rule == "disagreed"


def test_a_refusal_changes_nothing():
    refused = [_use(f"e{i}", r, outcome="refused") for i, r in enumerate(tr.REGIMES)]
    out = tr.apply_outcome(tr.UNCHECKED, None, refused)
    assert (out.status, out.level, out.episodes) == (tr.ON_TRIAL, tr.UNCHECKED, ())
    assert tr.track(refused)["intermediate"]["refusals"] == 4
    assert tr.record_line(tr.track(refused)) == "it has not been used since it was stored"


def test_an_adopted_item_stays_until_retirement_fires_and_then_loses_its_confirmation():
    one_bad = [_use("e9", "dense", answered=True, verdict="rejected")]
    kept = tr.apply_outcome(tr.CONFIRMED_BY_OUTCOMES, tr.ADOPTED, one_bad)
    assert kept.status == tr.ADOPTED  # one episode never decides
    two_bad = one_bad + [_use("e10", "dense", answered=True, verdict="rejected")]
    out = tr.apply_outcome(tr.CONFIRMED_BY_OUTCOMES, tr.ADOPTED, two_bad)
    assert (out.status, out.level) == (tr.RETIRED, tr.REPRODUCES_ITS_CASE)


def test_signal_outcome_reads_only_what_the_regime_and_the_actor_could_see():
    def sig(source, label, visible=True, regime="dense"):
        return Signal("s", "e1", source, label, "t", regime=regime, visible_to_actor=visible)

    assert tr.signal_outcome([sig("checker", "fail")], "dense", checker_visible=True)["verdict"] == "rejected"
    assert tr.signal_outcome([sig("checker", "fail", False)], "dense", checker_visible=True)["verdict"] is None
    assert tr.signal_outcome([sig("checker", "pass")], "dense", checker_visible=False)["verdict"] is None
    assert tr.signal_outcome([sig("checker", "pass")], "sparse", checker_visible=True)["end_outcome"] == "positive"
    assert tr.signal_outcome([sig("reader", "re_ask")], "implicit", checker_visible=False)["corrected"]
    assert tr.signal_outcome([sig("reader", "re_ask")], "none", checker_visible=False) == {
        "verdict": None,
        "end_outcome": None,
        "corrected": False,
    }


# --- the lifecycle delegates to the one decider -------------------------------------------------------------------


def test_the_lifecycle_counts_attributed_outcomes_from_the_use_record():
    def record(verdict_digest):
        rec = {"items": {ITEM: {"called": 1}}, "runtime_calls": {"cells": 1}, "outcomes_known": True}
        rec["items"][ITEM]["runtime"] = {"ok": 1, "refused": 0, "error": 0}
        return attribute_outcomes(
            rec,
            {ITEM: [{"outcome": "ok", "result": verdict_digest}]},
            [[1]],
            verdict="rejected",
        )

    facts = [
        lc.facts_from_use(f"e{i}", "a" * 40, "dense", record(result_digest([1])), [], checker_visible=True)
        for i in range(2)
    ]
    assert facts[0].items[ITEM].outcomes[0].verdict == "rejected"
    assert lc.decide_status(ITEM, None, facts)[:2] == ("retired", "rejected")
    # the same calls whose results were never handed back carry no verdict: nothing counts against the item
    unanswered = [
        lc.facts_from_use(f"e{i}", "a" * 40, "dense", record(result_digest([2])), [], checker_visible=True)
        for i in range(2)
    ]
    assert lc.decide_status(ITEM, None, unanswered)[0] == "on-trial"


def test_lifecycle_names_map_onto_the_trust_statuses():
    assert {tr.FROM_LIFECYCLE[k] for k in ("experimental", "stable", "suspect")} == set(tr.STATUSES)
    assert set(tr.LEVELS) == {"unchecked", "reproduces-its-case", "confirmed-by-outcomes"}
