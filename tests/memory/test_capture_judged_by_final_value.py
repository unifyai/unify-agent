"""Symbolic: a capture is judged by the verdicts on the value its episode handed back, never by a verdict on another
value.

An episode that was rejected and then accepted handed back its accepted value; its working procedure is captured as
positive. An episode whose final value was rejected is captured as contrary (negative knowledge), whatever an earlier
verdict said. Rev 2: the procedure is never run; its outcome comes from the signals only. Checker verdicts name the action they judged where the pairing is certain
(``<episode>/actions/<i>``); when none does, the last verdict stands for the final value
(``captures.final_value_signals``).
"""

from __future__ import annotations

import json

from tests.memory.test_capture_store import (
    PROCEDURE,
    _capture,
    _cid,
    _record,
    _request,
    _stage,
    _store,
    _summary,
    needs_bwrap,
)
from unify.memory import captures
from unify.memory.episodes import Action, Episode
from unify.memory.signals import Signal

EID = "ep-0001"
#: Three submissions; the last is the sorted numbers the procedure computes.
SUBMITTED = ([9, 5, 3, 1], [3, 1, 5, 9], [1, 3, 5, 9])


def _episode(submitted=SUBMITTED) -> Episode:
    request = _request()
    return Episode(
        episode_id=EID,
        started_at="2026-10-13T01:00:00Z",
        ended_at="2026-10-13T01:05:00Z",
        build="0" * 9,
        model="test/model",
        effort="low",
        regime="dense",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[
            Action(i, "user", "act", [value], {}, "received", "ok", "write", kind="dialogue")
            for i, value in enumerate(submitted)
        ],
        replies=[json.dumps(submitted[-1])],
    )


def _verdict(n: int, label: str, *, action: int | None) -> Signal:
    return Signal(
        signal_id=f"{EID}.checker.{n}",
        episode_id=EID,
        source="checker",
        label=label,
        ts=f"2026-10-13T01:0{n}:00Z",
        refers_to=f"{EID}/actions/{action}" if action is not None else EID,
        regime="dense",
        visible_to_actor=True,
    )


def _captured(tmp_path, signals):
    home = tmp_path / "home"
    staging = _stage(home, EID, PROCEDURE, summary=_summary())
    store = _store(home)
    _capture(staging, _episode(), store, signals)
    return _record(store, _cid(PROCEDURE))


def test_only_the_verdicts_on_the_final_value_judge_it():
    ep = _episode()
    rejected = [_verdict(1, "fail", action=0), _verdict(2, "fail", action=1)]
    accepted = _verdict(3, "pass", action=2)
    assert captures.final_value_signals([*rejected, accepted], ep) == [accepted]
    # verdicts that name actions, none of them the final one: none judges the final value
    assert captures.final_value_signals(rejected, ep) == []
    # no verdict names an action: the last one given stands for the final value
    unnamed = [_verdict(1, "fail", action=None), _verdict(3, "pass", action=None), _verdict(2, "fail", action=None)]
    assert captures.final_value_signals(unnamed, ep) == [unnamed[1]]
    # signals about the episode itself are kept
    provenance = Signal(f"{EID}.provenance", EID, "provenance", "support", "2026-10-13T01:06:00Z", regime="dense")
    assert captures.final_value_signals([provenance, *rejected, accepted], ep) == [provenance, accepted]


@needs_bwrap
def test_rejected_twice_then_accepted_is_captured_as_positive(tmp_path):
    rec = _captured(
        tmp_path,
        [_verdict(1, "fail", action=0), _verdict(2, "fail", action=1), _verdict(3, "pass", action=2)],
    )
    assert rec["own_check"]["ran"] is False
    assert rec["outcome"] == "positive"


@needs_bwrap
def test_rejected_twice_then_accepted_without_named_actions_is_positive_too(tmp_path):
    rec = _captured(
        tmp_path,
        [_verdict(1, "fail", action=None), _verdict(2, "fail", action=None), _verdict(3, "pass", action=None)],
    )
    assert rec["own_check"]["ran"] is False
    assert rec["outcome"] == "positive"


@needs_bwrap
def test_accepted_then_a_rejected_final_value_is_captured_as_contrary(tmp_path):
    rec = _captured(
        tmp_path,
        [_verdict(1, "pass", action=0), _verdict(2, "pass", action=1), _verdict(3, "fail", action=2)],
    )
    assert rec["outcome"] == captures.CONTRARY and captures.negative(rec)


def _correction(source: str, label: str, ts: str) -> Signal:
    return Signal(f"{EID}.{source}.{label}.{ts}", EID, source, label, ts, refers_to=EID, regime="dense")


def test_an_implicit_correction_judges_the_final_value_only_once_it_was_handed_back():
    """The episode ended at 01:05: a correction before then was about an earlier value, one after about the answer."""
    ep = _episode()
    before = _correction("reader", "correct", "2026-10-13T01:04:00Z")
    after = [
        _correction("reader", "re_ask", "2026-10-13T02:00:00Z"),
        _correction("recurrence", "re_ask", "2026-10-14T00:00:00+00:00"),
    ]
    unknown = _correction("reader", "correct", "not a time")
    provenance = Signal(f"{EID}.provenance", EID, "provenance", "correct", "2026-10-13T01:01:00Z", regime="dense")
    assert captures.final_value_signals([before, *after, unknown, provenance], ep) == [*after, provenance]


@needs_bwrap
def test_a_correction_after_the_answer_captures_it_as_contrary(tmp_path):
    accepted = _verdict(3, "pass", action=2)
    rec = _captured(tmp_path, [accepted, _correction("reader", "re_ask", "2026-10-13T02:00:00Z")])
    assert rec["own_check"]["ran"] is False
    assert rec["outcome"] == captures.CONTRARY and captures.negative(rec)


@needs_bwrap
def test_a_correction_before_the_final_answer_leaves_it_positive(tmp_path):
    accepted = _verdict(3, "pass", action=2)
    rec = _captured(tmp_path, [_correction("reader", "correct", "2026-10-13T01:04:00Z"), accepted])
    assert rec["own_check"]["ran"] is False
    assert rec["outcome"] == "positive"
