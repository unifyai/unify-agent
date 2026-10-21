"""Rev 2, item 10(a): a segment capture gets its outcome from the signals about the segment, once its episode is
recorded.

Before: a segment was captured while its episode ran, when no signal of it can be recorded yet (a signal is posted
only for a recorded episode), and nothing judged it again, so no segment capture ever carried an outcome. Now it is
captured unjudged and labelled once its episode is recorded, by the verdicts on its own answered action, the
corrections after it, and what was said while its cells ran; never by the verdict on the episode's final value.
No model, network or key; no world vocabulary.
"""

from __future__ import annotations

import json
from dataclasses import replace

from tests.memory.test_capture_trace import _ep, _staging
from unify.memory import captures as cp
from unify.memory.episodes import Action
from unify.memory.signals import Signal

PROC = 'def run(state):\n    """The first value."""\n    return state.data[0]\n'


def _submit(i, value):
    return Action(i, "env", "submit", [value], {}, "ok", "ok", "unknown", kind="dialogue")


def _episode():
    """A recorded episode: a wrong submission (action 0) mid-episode, then a right one (action 1) at its end."""
    return replace(_ep("e1", actions=[_submit(0, 1), _submit(1, 2)], regime="dense"),
                   started_at="2026-10-20T00:00:00Z", ended_at="2026-10-20T00:01:00Z")  # fmt: skip


def _segment():
    """The episode as it stood at its first checkpoint (00:00:20): one submission so far."""
    return replace(_ep("e1.s1", actions=[_submit(0, 1)], regime="none"), ended_at="2026-10-20T00:00:20Z")


def _verdict(n, label, at, action=None):
    refers = f"e1/actions/{action}" if action is not None else "e1"
    return Signal(f"e1.checker.{n}", "e1", "checker", label, at, refers_to=refers, regime="dense",
                  visible_to_actor=True)  # fmt: skip


def _outcome(signals):
    return cp.segment_outcome(signals, _segment(), _episode(), regime="dense")


def test_a_verdict_on_the_segments_own_action_labels_it():
    assert _outcome([_verdict(0, "fail", "2026-10-20T00:00:10Z", action=0)]) == "contrary"
    assert _outcome([_verdict(0, "pass", "2026-10-20T00:00:10Z", action=0)]) == "positive"


def test_the_verdict_on_the_episodes_final_value_never_labels_a_segment():
    final = _verdict(1, "pass", "2026-10-20T00:00:59Z", action=1)
    assert _outcome([final]) == "none"
    # and an unnamed verdict after the checkpoint is about later work
    assert _outcome([_verdict(2, "pass", "2026-10-20T00:00:50Z")]) == "none"
    # one said while the segment's cells ran judges it
    assert _outcome([_verdict(3, "fail", "2026-10-20T00:00:15Z")]) == "contrary"
    assert _outcome([_verdict(0, "fail", "2026-10-20T00:00:10Z", action=0), final]) == "contrary"


def test_a_segment_is_captured_unjudged_and_labelled_once_its_episode_is_recorded(tmp_path):
    store = cp.store_dir(tmp_path)
    _staging(tmp_path, "e1.s1", {"procedure.py": PROC})
    recorded: dict = {}
    sigs = [_verdict(0, "fail", "2026-10-20T00:00:10Z", action=0), _verdict(1, "pass", "2026-10-20T00:00:59Z", 1)]

    def episode(eid):
        return recorded[eid]  # KeyError while the episode runs, as the index lookup raises

    def signals(eid):
        if eid not in recorded:
            raise KeyError(eid)
        return sigs

    def call():
        return cp.capture_settled(tmp_path, ["e1.s1"], episode=episode, signals=signals, 
                                  segment_episode=lambda name: _segment() if name == "e1.s1" else None)  # fmt: skip

    [row] = call()
    assert row["segment"] == 1 and row["outcome"] == "none" and "judged" not in row
    assert call() == []  # the episode still runs: nothing to judge yet
    recorded["e1"] = _episode()
    [judged] = call()
    assert judged["staging"] == "e1.s1" and judged["outcome"] == "contrary" and judged["judged"] == cp.SEGMENT_JUDGED
    rec = json.loads((store / row["capture_id"] / cp.RECORD_FILE).read_text())
    [case] = rec["cases"]
    assert case["outcome"] == "contrary" and rec["outcome"] == "contrary" and cp.negative(rec)
    assert call() == []  # judged once
    assert cp.episode_rows(store)["e1.s1"]["outcome"] == "contrary"  # what the ledger reads


def test_a_late_rejudge_of_the_episode_leaves_its_segments_alone(tmp_path):
    store = cp.store_dir(tmp_path)
    _staging(tmp_path, "e1.s1", {"procedure.py": PROC})
    cp.capture_settled(tmp_path, ["e1.s1"], episode=lambda e: _episode(), signals=lambda e: [], 
                       segment_episode=lambda name: _segment())  # fmt: skip
    final = _verdict(1, "pass", "2026-10-20T00:00:59Z", action=1)
    assert cp.rejudge_episode(store, "e1", [final], _episode(), "dense") == []


# --- item 10(b): re-pins only ------------------------------------------------------------------------------------------


def test_re_pins_only_starts_no_segment_fork_at_a_pause(tmp_path, monkeypatch):
    from tests.memory.integration import test_in_episode as TIE
    from unify.memory.integration import fork as fork_mod
    from unify.memory.integration.in_episode import InEpisode

    forks = TIE._Forks()
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run = TIE._Run(tmp_path / "home")
    run.paths.state_dir.mkdir(parents=True)
    ie = InEpisode(run, TIE._exporter({TIE.C1: TIE._tree(TIE.LIB)}, None), segment_forks=False)
    ie.open(TIE.C1)
    run.code.append("x = 1")
    ie.cell_done()
    ie.mark("actor_call")
    assert ie.at_boundary(object(), new_message=True) is None
    assert forks.calls == [] and ie.pending == [] and ie.segments == 0
