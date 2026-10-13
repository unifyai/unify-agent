"""Use records per library commit and the signals the lifecycle may count."""

from __future__ import annotations

from types import SimpleNamespace

from tests.memory.test_episodes import _ep
from unify.memory import lifecycle as lc
from unify.memory.evidence import EvidenceStore
from unify.memory.signals import Signal

PD, WEEK, TOK = (
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
)
A, B = "a" * 40, "b" * 40


def _use_rec(rows, unknown=()):
    return {
        "version": 6,
        "items": rows,
        "items_outcome_unknown": list(unknown),
        "outcomes_known": not unknown,
        "items_at_pin": [PD, WEEK, TOK],
        "layout": "library_mode",
    }


def _visible(source, label, regime):
    """A checker signal marked agent-visible (``Signal.visible_to_actor``); counted_signals reads only
    ``source``, ``label`` and that flag, so a stand-in carries just those.
    """
    return SimpleNamespace(
        source=source,
        label=label,
        regime=regime,
        visible_to_actor=True,
    )


def test_hidden_checker_and_no_signal_regime_count_nothing():
    fail = [_visible("checker", "fail", "dense")]
    assert lc.counted_signals(fail, "dense", checker_visible=False) == (False, False)
    assert lc.counted_signals(fail, "dense", checker_visible=True) == (False, True)
    ok = [_visible("checker", "pass", "sparse")]
    assert lc.counted_signals(ok, "sparse", checker_visible=True) == (True, False)
    masked = [
        _visible("checker", "fail", "implicit"),
    ]  # the regime cannot observe a checker
    assert lc.counted_signals(masked, "implicit", checker_visible=True) == (
        False,
        False,
    )
    re_ask = [Signal("s", "e1", "recurrence", "re_ask", "t", regime="implicit")]
    assert lc.counted_signals(re_ask, "implicit", checker_visible=False) == (
        False,
        True,
    )
    env = [Signal("s", "e1", "environment", "error", "t", regime="none")]
    assert lc.counted_signals(env, "none", checker_visible=True) == (False, False)


def test_reading_the_use_record_and_the_evidence_index_agree(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    rec = _use_rec(
        {
            PD: {"imported": 1, "called": 2},
            WEEK: {"called": 1, "refused": 1},
            TOK: {"referenced": 1},
        },
        unknown=[WEEK],
    )
    ev.index_episode(
        _ep(episode_id="e1", regime="implicit", memory_main=A, memory_use=rec),
        "1" * 40,
    )
    sig = Signal("s1", "e1", "reader", "correct", "t", regime="implicit")
    ev.add_signal(sig)
    direct = lc.facts_from_use("e1", A, "implicit", rec, [sig], checker_visible=False)
    (indexed,) = lc.facts_from_evidence(ev, checker_visible=False)
    assert direct == indexed
    assert direct.items == {
        PD: lc.ItemUse(True, 2, False, False, False),
        # its one call refused: exposure, not use, and the refusal is no error
        WEEK: lc.ItemUse(False, 1, True, False, True, seen=True),
    }
    assert direct.negative and not direct.positive


def test_use_records_are_kept_per_library_commit():
    f1 = lc.EpisodeUse(
        "e1",
        A,
        "dense",
        True,
        False,
        {PD: lc.ItemUse(True, 2, False, False, False)},
    )
    f2 = lc.EpisodeUse(
        "e2",
        A,
        "dense",
        False,
        True,
        {PD: lc.ItemUse(True, 1, False, True, False)},
    )
    f3 = lc.EpisodeUse(
        "e3",
        B,
        "none",
        False,
        False,
        {
            PD: lc.ItemUse(True, 1, True, False, True),
            WEEK: lc.ItemUse(False, 0, False, True, False),
        },
    )
    assert lc.aggregate_use([f1, f2, f3]) == {
        PD: {
            A: {
                "episodes": 2,
                "uses": 3,
                "seen": 0,
                "errors": 1,
                "refused": 0,
                "negative_signals": 1,
                "positive_signals": 1,
                "unknown": 0,
            },
            B: {
                "episodes": 1,
                "uses": 1,
                "seen": 0,
                "errors": 0,  # a refusal is counted apart, never as an error
                "refused": 1,
                "negative_signals": 0,
                "positive_signals": 0,
                "unknown": 1,
            },
        },
        WEEK: {
            B: {
                "episodes": 0,
                "uses": 0,
                "seen": 0,
                "errors": 1,
                "refused": 0,
                "negative_signals": 0,
                "positive_signals": 0,
                "unknown": 0,
            },
        },
    }


def test_a_lookup_alone_is_not_a_use():
    f = lc.facts_from_use(
        "e1",
        A,
        "dense",
        _use_rec({PD: {"referenced": 2}}),
        [],
        checker_visible=False,
    )
    assert f.items == {} and lc.aggregate_use([f]) == {}


def test_a_checker_verdict_counts_only_when_the_signal_is_marked_visible():
    """A grader ``checker pass`` without ``visible_to_actor`` is ignored even when the run allows
    visible checker signals; a flagged pass counts."""
    unflagged = [Signal("s", "e1", "checker", "pass", "t", regime="dense")]
    assert lc.counted_signals(unflagged, "dense", checker_visible=True) == (
        False,
        False,
    )
    assert lc.counted_signals(
        [_visible("checker", "pass", "dense")],
        "dense",
        checker_visible=True,
    ) == (True, False)


def test_a_call_that_returned_is_a_use_and_an_import_alone_is_only_seen():
    rec = _use_rec(
        {
            PD: {"imported": 1},  # imported, never called
            WEEK: {
                "imported": 1,
                "called": 2,
                "refused": 1,
            },  # one call refused, the other returned
            TOK: {"called": 1, "errored": 1},  # its only call raised
        },
    )
    f = lc.facts_from_use("e1", A, "dense", rec, [], checker_visible=False)
    assert f.items == {
        PD: lc.ItemUse(False, 0, False, False, False, seen=True),
        WEEK: lc.ItemUse(True, 2, True, False, False),
        TOK: lc.ItemUse(False, 1, False, True, False, seen=True),
    }
    use = lc.aggregate_use([f])
    assert (use[PD][A]["episodes"], use[PD][A]["seen"]) == (0, 1)
    assert (
        use[WEEK][A]["episodes"],
        use[WEEK][A]["refused"],
        use[WEEK][A]["errors"],
    ) == (1, 1, 0)
    assert (use[TOK][A]["episodes"], use[TOK][A]["errors"]) == (0, 1)


def test_returned_compares_call_sites_with_the_exceptions_that_left_the_item():
    assert lc.returned(1, 0, 0) and lc.returned(2, 1, 0) and lc.returned(3, 1, 1)
    assert (
        not lc.returned(0, 0, 0)
        and not lc.returned(1, 1, 0)
        and not lc.returned(1, 0, 1)
    )


def test_a_signal_after_an_import_alone_is_not_counted_for_the_item():
    f = lc.EpisodeUse(
        "e1",
        A,
        "dense",
        True,
        True,
        {PD: lc.ItemUse(False, 0, False, False, False, seen=True)},
    )
    row = lc.aggregate_use([f])[PD][A]
    assert (row["negative_signals"], row["positive_signals"], row["seen"]) == (0, 0, 1)
