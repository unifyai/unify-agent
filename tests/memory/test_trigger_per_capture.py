"""A writer pass per stored capture (``UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on``): every stored capture makes a
pass due at once, whatever its outcome and whether it reproduced its value; the experience budget alone makes nothing
due; one pass covers at most ``per_pass`` capture episodes and the rest stay pending for the passes after it. Off, the
trigger is as built (the batched and verdict tests cover it)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from unify.memory import captures
from unify.memory.evidence import EvidenceStore
from unify.memory.integration import async_pass, consolidate, switch
from unify.memory.trigger import Trigger
from tests.memory.test_trigger import _big
from tests.memory.test_trigger_verdict import _Admission

CID = "0123456789abcdef"


def _indexed(tmp_path, *eids, chars=400):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for i, eid in enumerate(eids):
        ev.index_episode(_big(eid, chars), f"{i + 1:040x}")
    return ev


# --- what counts as a stored capture -------------------------------------------------------------------------------


def test_a_stored_capture_is_any_row_that_names_one_whatever_its_outcome():
    for row in (
        {"capture_id": CID, "outcome": "positive", "reproduces": "reproduces"},
        {"capture_id": CID, "outcome": "failed", "reproduces": "differs"},
        {"capture_id": CID, "outcome": "none", "reproduces": "unknown"},
    ):
        assert captures.stored(row), row
    for row in (
        {"capture_id": None, "safety": "no procedure"},
        {"capture_id": "not-an-id"},
        {"outcome": "positive", "reproduces": "reproduces"},
        None,
        "x",
    ):
        assert not captures.stored(row), row


def test_the_stored_episodes_come_from_each_episodes_last_capture_row(tmp_path):
    store = tmp_path / "store"
    captures._event(store, {"episode": "a", "capture_id": CID, "outcome": "failed", "reproduces": "differs"})
    captures._event(store, {"episode": "b", "capture_id": None, "safety": "no procedure"})
    captures._event(store, {"episode": "c", "capture_id": CID, "outcome": "none", "reproduces": "unknown"})
    assert captures.stored_episodes(store) == {"a", "c"}
    # the verdict rule, by contrast, needs a reproduced value and no contrary signal
    assert captures.verdict_episodes(store) == set()


# --- the trigger -----------------------------------------------------------------------------------------------------


def test_each_capture_makes_a_pass_due_at_once_one_per_pass(tmp_path):
    ev = _indexed(tmp_path, "a", "b", "c", "d")
    tr = Trigger(ev, experience_budget=10**9, verdict={"b", "d"}.__contains__, per_pass=1)
    [r] = tr.after_episode("d")
    assert (r.kind, r.episodes, r.verdicts, r.held) == ("capture", ["a", "b"], ["b"], [])
    assert r.experience_tokens == sum(ev.experience_of(e)[0] for e in "ab")
    tr.mark_done(r)
    # c and d stayed pending: the next pass is due at once
    assert [e for e, _ in tr.pending()] == ["c", "d"]
    [r] = tr.after_episode("d")
    assert (r.kind, r.episodes, r.verdicts) == ("capture", ["c", "d"], ["d"])
    tr.mark_done(r)
    assert tr.pending() == [] and tr.after_episode("d") == []


def test_a_larger_bound_covers_more_captures_in_one_pass(tmp_path):
    ev = _indexed(tmp_path, "a", "b", "c", "d", "e")
    tr = Trigger(ev, experience_budget=10**9, verdict={"a", "c", "e"}.__contains__, per_pass=2)
    [r] = tr.after_episode("e")
    assert (r.episodes, r.verdicts) == (["a", "b", "c"], ["a", "c"])
    tr.mark_done(r)
    [r] = tr.after_episode("e")
    assert (r.episodes, r.verdicts) == (["d", "e"], ["e"])


def test_the_experience_budget_alone_makes_no_pass_due(tmp_path):
    ev = _indexed(tmp_path, "a", "b", chars=4000)
    full = sum(ev.experience_of(e)[0] for e in "ab")
    assert Trigger(ev, experience_budget=full).after_episode("b")[0].kind == "batched"  # as built
    assert Trigger(ev, experience_budget=full, verdict=lambda e: False, per_pass=1).after_episode("b") == []
    assert Trigger(ev, experience_budget=full, per_pass=1).after_episode("b") == []  # no capture reader: none due
    [r] = Trigger(ev, experience_budget=full, verdict={"b"}.__contains__, per_pass=1).after_episode("b")
    assert (r.kind, r.episodes) == ("capture", ["a", "b"])  # the capture, not the budget, makes it due


def test_an_unsettled_episode_before_the_capture_rolls_and_one_after_it_stays_pending(tmp_path):
    ev = _indexed(tmp_path, "a", "b", "c", "d")
    adm = _Admission({"b", "d"})
    tr = Trigger(ev, experience_budget=10**9, admission=adm, verdict={"b", "d"}.__contains__, per_pass=1)
    [r] = tr.after_episode("d")
    assert (r.episodes, r.held, r.verdicts) == (["b"], ["a"], ["b"])
    tr.mark_done(r)
    assert adm.rolled() == ["a"]  # a rides with the next pass; c (after the cut) is still pending as recorded
    assert [e for e, _ in tr.pending()] == ["a", "c", "d"]
    adm.settled_ids.add("a")
    [r] = tr.after_episode("d")
    assert (r.episodes, r.held, r.verdicts) == (["a", "d"], ["c"], ["d"])


def test_episodes_rolled_over_always_ride_with_the_next_pass(tmp_path):
    ev = _indexed(tmp_path, "a", "b", "c")
    adm = _Admission({"a", "b", "c"})
    adm.rolled_ids = ["a"]
    tr = Trigger(ev, experience_budget=10**9, admission=adm, verdict={"a", "b", "c"}.__contains__, per_pass=1)
    tr.ev.set_cursor("@batched", tr.ev.seq_of("a"))  # a was covered before, then rolled over
    [r] = tr.after_episode("c")
    assert (r.episodes, r.verdicts) == (["a"], ["a"])  # the carried capture fills the bound
    tr.mark_done(r)
    [r] = tr.after_episode("c")
    assert (r.episodes, r.verdicts) == (["b"], ["b"])


def test_the_bound_is_a_positive_number(tmp_path):
    ev = _indexed(tmp_path, "a")
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            Trigger(ev, per_pass=bad)


# --- the switches ------------------------------------------------------------------------------------------------------


def _s(**kw):
    base = dict(UNIFY_MEMORY_LIBRARY="on", UNIFY_MEMORY_LIBRARY_FORK="on", UNIFY_MEMORY_LIBRARY_POOL="on")
    return SimpleNamespace(**{**base, **kw})


def test_the_switches_are_off_by_default_and_parse_their_values():
    for parse in (
        switch.parse_library_withdraw_closed_captures,
        switch.parse_library_fixed_answers_as_examples,
        switch.parse_library_review_each_capture,
    ):
        assert parse("ON") == "on" and parse("off") == "off"
        with pytest.raises(ValueError):
            parse("maybe")
    assert switch.parse_library_fixed_answers_as_examples("") == "off"
    assert switch.parse_library_review_each_capture("") == "off"
    # empty stays empty, so the redesign profile can give it (off without the profile)
    assert switch.parse_library_withdraw_closed_captures("") == ""
    assert switch.parse_library_captures_per_pass("") == 1
    assert switch.parse_library_captures_per_pass("3") == 3
    for bad in ("0", "-2", "x", True):
        with pytest.raises(ValueError):
            switch.parse_library_captures_per_pass(bad)
    blank = SimpleNamespace()
    assert not switch.library_withdraw_closed_captures(blank)
    assert not switch.library_fixed_answers_as_examples(blank)
    assert switch.captures_per_pass(blank) is None
    assert switch.captures_per_pass(_s(UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE="on")) == 1
    on = _s(UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE="on", UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS="2")
    assert switch.captures_per_pass(on) == 2
    for name in (
        switch.LIBRARY_WITHDRAW_CLOSED_CAPTURES,
        switch.LIBRARY_FIXED_ANSWERS_AS_EXAMPLES,
        switch.LIBRARY_REVIEW_EACH_CAPTURE,
        switch.LIBRARY_CAPTURES_PER_PASS,
    ):
        assert name in switch.PARSERS


@pytest.mark.parametrize(
    "name, needs",
    [
        ("UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES", "UNIFY_MEMORY_LIBRARY_POOL"),
        ("UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES", "UNIFY_MEMORY_LIBRARY_FORK"),
        ("UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE", "UNIFY_MEMORY_LIBRARY_FORK"),
    ],
)
def test_a_switch_without_what_it_needs_is_refused_before_anything_runs(name, needs):
    assert switch.profile_refusal(_s(**{name: "on"})) is None
    refused = switch.profile_refusal(_s(**{name: "on", needs: "off"}))
    assert refused == f"{name}=on needs {needs}=on; nothing runs"
    refused = switch.profile_refusal(_s(**{name: "on", "UNIFY_MEMORY_LIBRARY": "off"}))
    assert refused.startswith(f"{name}=on needs UNIFY_MEMORY_LIBRARY=on")
    assert switch.profile_refusal(_s(**{name: "off", needs: "off"})) is None  # off: nothing is needed


def test_the_capture_reader_follows_the_switch(tmp_path):
    paths = SimpleNamespace(state_dir=tmp_path)
    store = captures.store_dir(tmp_path)
    captures._event(store, {"episode": "a", "capture_id": CID, "outcome": "failed", "reproduces": "differs"})
    captures._event(store, {"episode": "b", "capture_id": CID, "outcome": "positive", "reproduces": "reproduces"})
    built = consolidate.verdict_check(paths, _s())
    assert (built("a"), built("b")) == (False, True)  # as built: the verdict rule
    each = consolidate.verdict_check(paths, _s(UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE="on"))
    assert (each("a"), each("b")) == (True, True)
    assert consolidate.verdict_check(paths, _s(UNIFY_MEMORY_LIBRARY="off")) is None


def test_both_places_that_ask_whether_a_pass_is_due_pass_the_bound():
    for fn in (consolidate.run_due_passes, async_pass.maybe_spawn):
        src = inspect.getsource(fn)
        assert "per_pass=captures_per_pass(settings) if library_enabled(settings) else None" in src
