"""``UNIFY_MEMORY_REGIME=inferred``: each episode's regime inferred from the signals that arrive, never defaulted;
signal timing; late verdicts reopening what they bear on; and the status a card states by what it rests on.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.memory.test_episodes import _ep
from unify.memory import captures, lifecycle, trust
from unify.memory.episodes import Action, EpisodeWriter
from unify.memory.evidence import EvidenceStore
from unify.memory.integration import outcomes
from unify.memory.integration import request as request_mod
from unify.memory.integration.consolidate import open_stores
from unify.memory.integration.paths import Paths
from unify.memory.redact import Redactor
from unify.memory.signals import (
    AFTER_EPISODE,
    IN_EPISODE,
    Signal,
    infer_regime,
    note_text,
    run_prior,
    timing_of,
    valid_about,
)
from unify.settings import SETTINGS

ITEM = "memory.reports.totals:sum_rows"
ENDED = "2026-10-08T01:01:00Z"


def _sig(
    eid, n, label="pass", *, visible=True, refers=None, timing="", source="checker"
):
    return Signal(
        f"{eid}.s{n}",
        eid,
        source,
        label,
        f"2026-10-08T01:00:{n:02d}Z",
        refers_to=refers if refers is not None else eid,
        visible_to_actor=visible,
        timing=timing,
    )


# --- 1. four episode shapes give the four regimes -----------------------------------------------------------------


def test_four_shapes_infer_the_four_regimes():
    per_action = [
        _sig("e1", 1, "fail", refers="e1/actions/0"),
        _sig("e1", 2, "pass", refers="e1/actions/1"),
    ]
    assert infer_regime(per_action) == ("dense", "signals")
    two_unpaired = [_sig("e1", 1, "fail"), _sig("e1", 2, "pass")]
    assert infer_regime(two_unpaired) == (
        "dense",
        "signals",
    )  # verdicts after several answered actions
    assert infer_regime([_sig("e2", 1, "pass")]) == (
        "sparse",
        "signals",
    )  # one outcome for the whole episode
    assert infer_regime([_sig("e2", 1, "fail", timing=AFTER_EPISODE)]) == (
        "sparse",
        "signals",
    )
    assert infer_regime(
        [_sig("e2", 1, "support", source="provenance", visible=False)]
    ) == ("sparse", "signals")
    assert infer_regime([], implicit_evidence=True) == (
        "implicit",
        "signals",
    )  # only an error or a holds result
    assert infer_regime([_sig("e3", 1, "correct", source="reader", visible=False)]) == (
        "implicit",
        "signals",
    )
    assert infer_regime([]) == ("none", "nothing")


def test_a_hidden_graders_verdict_decides_nothing():
    hidden = [
        _sig("e1", 1, "pass", visible=False),
        _sig("e1", 2, "fail", visible=False),
    ]
    assert infer_regime(hidden) == ("none", "nothing")
    assert infer_regime(hidden, prior="sparse") == ("sparse", "prior")


def test_a_late_verdict_never_makes_a_regime_dense():
    late = [_sig("e1", 1, "fail", refers="e1/actions/0", timing=AFTER_EPISODE)]
    assert infer_regime(late) == ("sparse", "signals")


def test_the_prior_is_the_runs_own_signal_given_regimes_and_never_feeds_itself():
    assert run_prior([]) is None
    assert (
        run_prior([("sparse", "signals"), ("sparse", "signals"), ("dense", "signals")])
        == "sparse"
    )
    assert (
        run_prior([("sparse", "prior"), ("sparse", "prior"), ("dense", "signals")])
        == "dense"
    )
    assert (
        run_prior([("dense", "signals"), ("sparse", "signals")]) == "dense"
    )  # a tie: the one that observes more
    assert run_prior([("none", "nothing")]) is None
    assert infer_regime([], prior="implicit") == ("implicit", "prior")
    assert infer_regime([], prior="none") == ("none", "nothing")


def test_timing_and_about():
    assert timing_of("2026-10-08T01:00:30Z", ENDED) == IN_EPISODE
    assert timing_of("2026-10-08T01:01:00+00:00", ENDED) == IN_EPISODE
    assert timing_of("2026-10-08T01:02:00Z", ENDED) == AFTER_EPISODE
    with pytest.raises(ValueError):
        timing_of("2026-10-08T01:02:00", ENDED)  # no zone
    for ok in (
        "",
        "action:3",
        "item:memory.reports.totals:sum_rows",
        "item:notes/reports/summary.md",
        "capture:0123456789abcdef",
    ):
        assert valid_about(ok), ok
    for bad in ("task:1", "action:x", "capture:12", "item:", "item: spaced", 7):
        assert not valid_about(bad), bad


def test_a_signal_without_timing_is_noted_as_before_and_one_with_it_carries_it():
    plain = Signal("e.checker", "e", "checker", "pass", "t", refers_to="e")
    assert set(json.loads(note_text(plain))) == {
        "signal_id",
        "episode_id",
        "source",
        "label",
        "ts",
        "refers_to",
        "regime",
        "revealed",
        "reveal_p",
    }
    timed = outcomes.timed(
        Signal("e.c.1", "e", "checker", "fail", "t"),
        "2026-10-08T01:05:00Z",
        ENDED,
        "action:2",
    )
    row = json.loads(note_text(timed))
    assert (row["timing"], row["about"], row["episode_ended_at"]) == (
        AFTER_EPISODE,
        "action:2",
        ENDED,
    )


# --- the evidence store keeps timing, the regime with its basis, and every change ------------------------------


def _schema(path) -> list:
    con = sqlite3.connect(path)
    try:
        return sorted(
            r[0]
            for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        )
    finally:
        con.close()


def test_the_store_keeps_its_schema_until_a_regime_is_inferred(tmp_path):
    path = tmp_path / "ev.sqlite"
    ev = EvidenceStore(path)
    ev.index_episode(_ep(episode_id="e0"), "0" * 40)
    before = _schema(path)
    ev.add_signal(Signal("e0.checker", "e0", "checker", "pass", "t"))
    assert (
        _schema(path) == before
        and ev.inferred_regimes() == []
        and ev.regime_changes() == []
    )
    ev.add_signal(
        outcomes.timed(
            Signal("e0.c.1", "e0", "checker", "fail", "t"),
            "2026-10-08T01:00:30Z",
            ENDED,
        )
    )
    assert "signal_timing" in _schema(path)
    got = {s.signal_id: (s.timing, s.arrived_at) for s in ev.signals_for("e0")}
    assert got == {
        "e0.checker": ("", ""),
        "e0.c.1": (IN_EPISODE, "2026-10-08T01:00:30Z"),
    }
    assert (
        ev.record_regime("e0", "sparse", "signals", ENDED) is None
    )  # its first record
    assert (
        ev.record_regime("e0", "sparse", "signals", ENDED) == "sparse"
    )  # unchanged: not logged again
    assert (
        ev.record_regime(
            "e0", "dense", "signals", "2026-10-08T02:00:00Z", signal_id="e0.late.x.1"
        )
        == "sparse"
    )
    assert ev.regime_of("e0") == "dense" and ev.regime_record("e0") == (
        "dense",
        "signals",
    )
    assert [
        (c["before"], c["after"], c["signal_id"]) for c in ev.regime_changes("e0")
    ] == [
        (None, "sparse", None),
        ("sparse", "dense", "e0.late.x.1"),
    ]
    with pytest.raises(KeyError):
        ev.record_regime("nope", "none", "nothing", ENDED)


# --- a run whose episodes differ: each episode is judged in its own regime ------------------------------------------


def _record(
    stores, eid, *, regime, minute, used=True, actions=None, ended=None, errored=0
):
    ep = _ep(
        episode_id=eid,
        started_at=f"2026-10-08T01:{minute:02d}:00Z",
        ended_at=ended or f"2026-10-08T01:{minute:02d}:30Z",
        regime=regime,
        actions=actions or [Action(0, "files", "read", [], {}, {"n": 1}, "ok", "read")],
    )
    if used:
        ep.memory_use = {
            "items": {
                ITEM: {"imported": 1, "called": 1, "refused": 0, "errored": errored}
            },
            "outcomes_known": True,
        }
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    return sha, ep


def _post(stores, sha, sig, regime):
    from unify.memory.signals import post_signal

    sig.regime = regime
    post_signal(sig, stores.episodes, sha, stores.evidence)


def test_episodes_with_different_regimes_in_one_run_are_each_judged_in_their_own(
    tmp_path,
):
    stores = open_stores(Paths.under(tmp_path / "home"))
    shapes = {
        "dense0": [
            _sig("dense0", 1, "fail", refers="dense0/actions/0"),
            _sig("dense0", 2, "fail", refers="dense0/actions/0"),
        ],
        "sparse0": [_sig("sparse0", 1, "fail")],
        "implicit0": [_sig("implicit0", 1, "correct", source="reader", visible=False)],
        "none0": [],
    }
    expected = {
        "dense0": "dense",
        "sparse0": "sparse",
        "implicit0": "implicit",
        "none0": "none",
    }
    for minute, (eid, sigs) in enumerate(shapes.items()):
        # each shape on its own signals (the run's prior is tested apart)
        regime, basis = infer_regime(sigs)
        assert regime == expected[eid], eid
        sha, _ = _record(stores, eid, regime=regime, minute=minute)
        stores.evidence.record_regime(eid, regime, basis, ENDED)
        for s in sigs:
            _post(stores, sha, s, regime)
    facts = {
        f.episode_id: f
        for f in lifecycle.facts_from_evidence(stores.evidence)
    }
    assert {e: f.regime for e, f in facts.items()} == expected
    judged = {
        e: [trust.judge(u)[:2] for u in lifecycle.trust_uses(f, f.items[ITEM])]
        for e, f in facts.items()
    }
    # a verdict judges the use where verdicts are given, an end outcome where outcomes are, a correction where the
    # regime observes corrections; with nothing, the use is clean and no outcome claims anything
    assert judged["dense0"] == [(trust.FAILED, "negative")]
    assert judged["sparse0"] == [(trust.FAILED, "negative")]
    assert judged["implicit0"] == [(trust.FAILED, "negative")]
    assert judged["none0"] == [(trust.CLEAN, None)]
    by = {e: lifecycle.evidence_basis(ITEM, [f])[0] for e, f in facts.items()}
    assert by == {
        "dense0": trust.BASIS_VERDICTS,
        "sparse0": trust.BASIS_END_OUTCOMES,
        "implicit0": trust.BASIS_IMPLICIT,
        "none0": trust.BASIS_NO_OUTCOME_SIGNALS,
    }


# --- 2. a late contrary verdict reopens what it bears on and its uses are recomputed ------------------------------


def _capture(store, eid):
    """A stored capture judged on episode *eid* with no outcome yet (as the capture step leaves it)."""
    source = '"""Sum a column."""\n\ndef run(value):\n    return sum(value)\n'
    cid = captures.capture_id(source)
    d = store / cid
    d.mkdir(parents=True)
    rec = {
        "episodes": [eid],
        "cases": [{"episode": eid, "reproduces": "reproduces", "outcome": "none"}],
        "uses": [trust.Use(eid, "none", "returned", answered=True).row()],
        "outcome": "none",
        "level": "unchecked",
        "status": "on-trial",
        "track": {},
    }
    (d / captures.PROCEDURE_FILE).write_text(source)
    (d / captures.RECORD_FILE).write_text(json.dumps(rec))
    return cid


def test_a_late_verdict_is_attached_by_episode_id_only(tmp_path):
    stores = open_stores(Paths.under(tmp_path / "home"))
    _record(stores, "e1", regime="none", minute=0)
    exists = stores.evidence.episode_exists
    for episode, why in (
        ("20261020T010203-0a1b2c3d", "no recorded episode has that id"),
        ("e1x", "no recorded episode has that id"),
        ("../e1", "episode must be a recorded episode's id"),
        ("", "episode must be a recorded episode's id"),
        (3, "episode must be a recorded episode's id"),
    ):
        assert (
            outcomes.late_entry("fail", episode, None, current="e9", exists=exists)[1]
            == why
        )
    assert outcomes.late_entry("maybe", "e1", None, current="e9", exists=exists)[
        1
    ].startswith("a checker line is")
    assert outcomes.late_entry("fail", "e9", None, current="e9", exists=exists)[
        1
    ].startswith("that is this request's")
    assert outcomes.late_entry("fail", "e1", "task:1", current="e9", exists=exists)[
        1
    ].startswith("about must be")
    # an id as a request names its episode (the one the CLI announces) is taken
    (tmp_path / "transcripts").mkdir()
    real = request_mod.new_episode_id(tmp_path / "transcripts")
    _record(stores, real, regime="none", minute=1)
    entry, why = outcomes.late_entry("pass", real, None, current="e9", exists=exists)
    assert why == "" and entry["episode"] == real and entry["label"] == "pass"


# --- the request: inferred before the episode is written, late verdicts posted when it is recorded ----------------


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_REGIME", "inferred")
    paths = Paths.under(tmp_path / "home")
    run = request_mod.RequestRun("Sum the rows.", paths)
    run.stores = open_stores(paths)
    run.regime_inferred = request_mod.regime_inferred_on()
    run.episode_id = "f" * 16
    return run


# --- 4. with no signals the card says so; status text for every basis --------------------------------------------


def test_the_status_text_states_what_it_rests_on():
    counts = {"clean": 3, "failed": 1, "neutral": 0}
    assert trust.basis_text("on-trial", trust.BASIS_NO_OUTCOME_SIGNALS, {}) == (
        "on-trial: no outcome signal has arrived for any use of this item (it has not been used)"
    )
    assert trust.basis_text(
        "on-trial", trust.BASIS_NO_OUTCOME_SIGNALS, {"clean": 2}
    ) == (
        "on-trial: no outcome signal has arrived for any use of this item (2 uses without one)"
    )
    assert trust.basis_text("adopted", trust.BASIS_END_OUTCOMES, counts) == (
        "adopted: 3 clean uses, 1 failed, all judged at episode end"
    )
    assert trust.basis_text("adopted", trust.BASIS_VERDICTS, counts).endswith(
        "judged by the verdict after each answered call"
    )
    assert "never by an outcome" in trust.basis_text(
        "on-trial", trust.BASIS_IMPLICIT, counts
    )
    assert trust.status_basis([]) == trust.BASIS_NO_OUTCOME_SIGNALS
    assert (
        trust.status_basis([trust.Use("e", "none", "returned")])
        == trust.BASIS_NO_OUTCOME_SIGNALS
    )
    assert (
        trust.status_basis([trust.Use("e", "none", "error")])
        == trust.BASIS_NO_OUTCOME_SIGNALS
    )
    assert (
        trust.status_basis([trust.Use("e", "implicit", "returned", holds=True)])
        == trust.BASIS_IMPLICIT
    )
    assert trust.status_basis(
        [trust.Use("e", "sparse", "returned", answered=True, end_outcome="positive")]
    ) == (trust.BASIS_END_OUTCOMES)
    assert trust.status_basis(
        [trust.Use("e", "dense", "returned", answered=True, verdict="accepted")]
    ) == (trust.BASIS_VERDICTS)
    assert trust.status_basis(
        [trust.Use("e", "dense", "returned", answered=None, episode_negative=True)]
    ) == (trust.BASIS_VERDICTS)


# --- 5. dense-shaped episodes behave as built ---------------------------------------------------------------------


# --- the CLI ------------------------------------------------------------------------------------------------------


def test_a_recovered_episodes_regime_is_inferred_too(tmp_path, monkeypatch):
    from unify.memory.integration import journal

    stores = open_stores(Paths.under(tmp_path / "home"))
    for minute, eid in enumerate(("s1", "s2")):
        _record(stores, eid, regime="sparse", minute=minute)
        stores.evidence.record_regime(eid, "sparse", "signals", ENDED)
    ep = _ep(episode_id="r1", regime="dense")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_REGIME", "off")
    assert journal._inferred_regime(ep, stores) is None and ep.regime == "dense"
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_REGIME", "inferred")
    assert (
        journal._inferred_regime(ep, stores) == "prior" and ep.regime == "sparse"
    )  # nothing of its own arrived


def test_a_hidden_graders_verdict_never_judges_a_capture_under_the_switch():
    """The capture step's outcome (captures.judged_outcome): under the switch (signals carry their timing) the last
    verdict the actor saw stands for the episode, a later hidden one is left out; without it, as built, the last
    verdict given stands, whichever it is."""
    ep = _ep(episode_id="e1", regime="sparse")
    shown = _sig("e1", 1, "pass", visible=True, timing=IN_EPISODE)
    hidden = _sig("e1", 2, "fail", visible=False, timing=IN_EPISODE)
    got = captures.judged_outcome([shown, hidden], ep, "sparse")
    assert got["positive"] and not got["contrary"]
    alone = captures.judged_outcome([hidden], ep, "sparse")
    assert not alone["positive"] and not alone["contrary"]
    built = captures.judged_outcome(
        [_sig("e1", 1, "pass", visible=True), _sig("e1", 2, "fail", visible=False)], ep, "dense"
    )
    assert built == captures.judged_outcome(
        [_sig("e1", 2, "fail", visible=False)], ep, "dense"
    )
