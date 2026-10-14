"""A writer pass at the verdict: an admitted pending episode whose capture is a verdict (its outcome positive and its
capture reproducing the value it handed back) makes a pass due at once, before the batch reaches E. The pass covers
what a batched pass would cover now and moves the same cursor; the batched trigger is otherwise unchanged."""

from __future__ import annotations

from unify.memory import captures
from unify.memory.evidence import EvidenceStore
from unify.memory.trigger import Trigger
from tests.memory.test_trigger import _big


def _indexed(tmp_path, *eids):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for i, eid in enumerate(eids):
        ev.index_episode(_big(eid, 400), f"{i + 1}" * 40)
    return ev


def test_a_verdict_capture_is_positive_and_reproducing_only():
    assert captures.verdict({"outcome": "positive", "reproduces": "reproduces"})
    for row in (
        {"outcome": "failed", "reproduces": "reproduces"},
        {"outcome": "failed", "reproduces": "differs"},
        {"outcome": "positive", "reproduces": "unknown"},
        {"outcome": "none", "reproduces": "reproduces"},
        {"outcome": "positive", "reproduces": None},
        None,
        "positive",
    ):
        assert not captures.verdict(row), row


def test_the_verdict_episodes_come_from_each_episodes_last_capture_row(tmp_path):
    store = tmp_path / "store"
    captures._event(store, {"episode": "a", "outcome": "positive", "reproduces": "reproduces"})
    captures._event(store, {"episode": "b", "outcome": "positive", "reproduces": "differs"})
    captures._event(store, {"episode": "c", "outcome": "failed", "reproduces": "reproduces"})
    captures._event(store, {"episode": "d", "outcome": "positive", "reproduces": "unknown"})
    assert captures.verdict_episodes(store) == {"a"}


def test_a_verdict_makes_a_pass_due_at_once_covering_what_a_batch_would(tmp_path):
    ev = _indexed(tmp_path, "a", "b", "c")
    plain = Trigger(ev, experience_budget=10**9)
    assert plain.after_episode("c") == []  # far from E: the batched trigger waits
    tr = Trigger(ev, experience_budget=10**9, verdict=lambda e: e == "b")
    [r] = tr.after_episode("c")
    assert (r.kind, r.episodes, r.verdicts, r.held) == ("verdict", ["a", "b", "c"], ["b"], [])
    assert r.experience_tokens == sum(ev.experience_of(e)[0] for e in "abc") < tr.budget
    # the same budget as a batched pass: E x the rate
    assert tr.pass_budget_usd() == plain.pass_budget_usd()
    tr.mark_done(r)  # the batched cursor moves past everything it covered
    assert tr.pending() == [] and tr.after_episode("c") == []
    ev.index_episode(_big("d", 400), "4" * 40)
    assert tr.after_episode("d") == [] and [e for e, _ in tr.pending()] == ["d"]


def test_no_verdict_makes_nothing_due_and_a_batch_at_e_stays_batched(tmp_path):
    ev = _indexed(tmp_path, "a", "b")
    assert Trigger(ev, experience_budget=10**9, verdict=lambda e: False).after_episode("b") == []
    full = sum(ev.experience_of(e)[0] for e in "ab")
    [r] = Trigger(ev, experience_budget=full, verdict=lambda e: e == "a").after_episode("b")
    assert (r.kind, r.episodes, r.verdicts) == ("batched", ["a", "b"], ["a"])
    [r] = Trigger(ev, experience_budget=full).after_episode("b")
    assert (r.kind, r.verdicts) == ("batched", [])  # no verdict reader: exactly as before


class _Admission:
    def __init__(self, settled):
        self.settled_ids, self.rolled_ids = set(settled), []

    def settled(self, eid):
        return eid in self.settled_ids

    def rolled(self):
        return list(self.rolled_ids)

    def set_rolled(self, eids):
        self.rolled_ids = list(eids)


def test_an_episode_whose_fork_has_not_settled_is_held_not_a_verdict(tmp_path):
    ev = _indexed(tmp_path, "a", "b")
    adm = _Admission({"a"})
    tr = Trigger(ev, experience_budget=10**9, admission=adm, verdict=lambda e: e == "b")
    assert tr.after_episode("b") == []  # b's verdict waits for its fork to settle
    adm.settled_ids.add("b")
    [r] = tr.after_episode("b")
    assert (r.kind, r.episodes, r.verdicts, r.held) == ("verdict", ["a", "b"], ["b"], [])
    held = Trigger(ev, experience_budget=10**9, admission=_Admission({"b"}), verdict=lambda e: e == "b")
    [r] = held.after_episode("b")
    assert (r.episodes, r.held) == (["b"], ["a"])  # the unsettled episode rolls to a later pass
    held.mark_done(r)
    assert held.admission.rolled() == ["a"]


def test_every_verdict_in_a_row_gets_its_pass_and_nothing_caps_them(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    tr = Trigger(ev, experience_budget=10**9, verdict=lambda e: True)
    covered = []
    for i in range(12):
        eid = f"e{i}"
        ev.index_episode(_big(eid, 400), f"{i:040x}")
        [r] = tr.after_episode(eid)
        assert (r.kind, r.episodes, r.verdicts) == ("verdict", [eid], [eid])
        tr.mark_done(r)
        covered.append(eid)
    assert covered == [f"e{i}" for i in range(12)]
