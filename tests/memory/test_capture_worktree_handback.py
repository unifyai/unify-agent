"""A work-tree change is the outcome: its closing reply is not a handed-back value.

An episode that changed the work tree and answered no dialogue action handed back no value: its last reply is often a
prose summary of what was done, and comparing a captured procedure's result with that sentence would mark a procedure
that did the job as failed. Such an episode is still a completed attempt; its capture's reproduction is unknown and
its outcome comes from the signals alone. An answered dialogue action, and a final reply in an episode whose work
tree did not change, hand back their value as before.

No model and no box runs: the confined box is a stand-in that writes the result a box would.
"""

from __future__ import annotations

from tests.memory.test_captures import PROC, VALUE, _box, _checker, _ep, _staging
from unify.memory import captures as cp


def _worktree_episode(
    eid, reply="Done: I updated the quarterly sheet and sent the summary."
):
    ep = _ep(eid)
    ep.actions = []
    ep.replies = [reply]
    ep.worktree_before, ep.worktree_after = "a" * 40, "b" * 40
    return ep


def test_a_work_tree_change_with_no_answered_action_hands_back_nothing():
    ep = _worktree_episode("e1")
    assert cp.handed_back(ep) == (False, None)
    assert cp.completed(ep) is True  # still a completed attempt: the work tree changed
    assert cp.answered_action(ep) is None


def test_its_capture_is_unknown_not_failed_and_its_outcome_comes_from_signals(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": PROC})
    store = cp.store_dir(tmp_path)
    runner, calls = _box(VALUE, [6, 9])
    row = cp.capture_trace_step(staging, _worktree_episode("e1"), store, runner=runner)
    assert row["safety"] == "ok" and len(calls) == 1
    assert (row["reproduces"], row["outcome"]) == ("unknown", "none")
    rec = cp.load(store)[row["capture_id"]]["record"]
    assert rec["outcome"] == "none" and rec["reproduces"] == "unknown"
    # a positive signal makes it positive; nothing about the prose reply makes it failed
    staging2 = _staging(
        tmp_path, "e2", {"procedure.py": PROC.replace("Total", "Add up")}
    )
    runner, _ = _box(VALUE, [6, 9])
    row2 = cp.capture_trace_step(
        staging2,
        _worktree_episode("e2"),
        store,
        signals=[_checker("e2", "pass")],

        runner=runner,
    )
    assert (row2["reproduces"], row2["outcome"]) == ("unknown", "positive")
    # and a contrary signal still makes it failed
    staging3 = _staging(
        tmp_path, "e3", {"procedure.py": PROC.replace("Total", "Count")}
    )
    runner, _ = _box(VALUE, [6, 9])
    row3 = cp.capture_trace_step(
        staging3,
        _worktree_episode("e3"),
        store,
        signals=[_checker("e3", "fail")],

        runner=runner,
    )
    assert row3["outcome"] == "failed"


def test_an_answered_dialogue_action_hands_back_its_value_with_or_without_a_work_tree_change():
    ep = _ep("e1")  # ARC-shaped: the answer is a dialogue action
    found, value = cp.handed_back(ep)
    assert found is True and value is not None and cp.answered_action(ep) == 0
    ep.worktree_before, ep.worktree_after = "a" * 40, "b" * 40
    assert cp.handed_back(ep) == (found, value)


def test_a_final_reply_with_no_work_tree_change_hands_back_its_value_as_before():
    ep = _ep("e1")
    ep.actions = []
    ep.replies = ['{"rows": [6, 9]}']
    found, value = cp.handed_back(ep)
    assert found is True and value is not None
    ep.worktree_before = ep.worktree_after = (
        "c" * 40
    )  # recorded but unchanged: still the reply's value
    assert cp.handed_back(ep) == (found, value)
    ep.replies = ["the answer is 42"]
    ep.worktree_before = ep.worktree_after = None
    found, _ = cp.handed_back(ep)
    assert found is True
