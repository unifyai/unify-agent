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


def test_without_an_answered_value_a_verdict_on_any_action_judges_the_work():
    """Rev 2.1: in tool and action worlds the work is the actions and nothing is handed back through a
    dialogue action, so the last verdict on any of the episode's actions judges it (rev 2 judged nothing then)."""
    import dataclasses

    acting = dataclasses.replace(
        _episode(),
        actions=[Action(i, "board", "step", [d], {}, "moved", "ok", "write", kind="tool") for i, d in enumerate("lrl")],
        replies=[],
        worktree_before="a",
        worktree_after="b",
    )
    assert captures.answered_action(acting) is None
    first, last = _verdict(1, "fail", action=0), _verdict(3, "pass", action=2)
    assert captures.final_value_signals([first, last], acting) == [last]
    judged = captures.judged_outcome([first, last], acting, "dense")
    assert judged["positive"] is True and judged["contrary"] is False
