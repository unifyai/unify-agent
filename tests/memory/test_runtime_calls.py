"""Runtime call evidence: the calls the actor's cells actually made, from the worker's audit records, carried in
the request's use record, indexed by the evidence store and preferred by the lifecycle over static call sites.
"""

from __future__ import annotations

from tests.memory.test_episodes import _ep
from unify.memory import lifecycle as lc
from unify.memory.evidence import EvidenceStore, runtime_counts
from unify.memory.item_states import ADOPTED, ON_TRIAL, RETIRED
from unify.memory.integration import hooks
from unify.memory.integration.request import (
    MAX_RUNTIME_ITEMS,
    RequestRun,
    add_runtime_calls,
    runtime_calls_empty,
    with_runtime_calls,
)

PD, WEEK, TOK = (
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
)
A = "a" * 40


def _drained(counts, **extra):
    return {
        "records": [],
        "dropped": 0,
        "failed": 0,
        "bytes": 0,
        "calls": {
            "counts": counts,
            "records": [],
            "dropped": 0,
            "records_dropped": 0,
            "malformed": 0,
            **extra,
        },
    }


def _use_rec(rows):
    return {
        "version": 6,
        "items": rows,
        "items_outcome_unknown": [],
        "outcomes_known": True,
        "items_at_pin": [PD, WEEK, TOK],
        "layout": "library_mode",
    }


# --- the request's accumulation ------------------------------------------------------------------------------


def test_cells_add_their_counts_and_a_cell_without_calls_is_still_observed():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, _drained({PD: {"ok": 2, "refused": 1, "error": 0}}))
    add_runtime_calls(acc, {"records": [], "dropped": 0, "failed": 0, "bytes": 0})
    add_runtime_calls(
        acc,
        _drained({PD: {"ok": 1, "error": 1}, WEEK: {"refused": 3}}, malformed=2),
    )
    add_runtime_calls(acc, None)  # a cell whose records were lost is not observed
    assert acc["cells"] == 3
    assert acc["counts"] == {
        PD: {"ok": 3, "refused": 1, "error": 1},
        WEEK: {"ok": 0, "refused": 3, "error": 0},
    }
    assert acc["malformed"] == 2


def test_malformed_counts_are_skipped_and_items_over_the_bound_are_dropped():
    acc = runtime_calls_empty()
    add_runtime_calls(
        acc,
        _drained({PD: {"ok": True, "error": -4, "refused": "9"}, 5: {"ok": 1}}),
    )
    assert (
        acc["counts"] == {PD: {"ok": 0, "refused": 0, "error": 0}}
        and acc["malformed"] == 1
    )
    acc = runtime_calls_empty()
    many = {f"memory.p.m:f{i}": {"ok": 1} for i in range(MAX_RUNTIME_ITEMS + 3)}
    add_runtime_calls(acc, _drained(many))
    assert len(acc["counts"]) == MAX_RUNTIME_ITEMS and acc["dropped"] == 3


def test_the_worker_cell_hook_hands_calls_to_the_request(monkeypatch):
    run = RequestRun("r", None)
    monkeypatch.setattr(hooks, "_run", lambda: run)
    hooks.worker_cell_done(_drained({TOK: {"ok": 1}}))
    hooks.worker_cell_done(None)
    assert run.runtime_calls["cells"] == 1
    assert run.runtime_calls["counts"] == {TOK: {"ok": 1, "refused": 0, "error": 0}}


# --- the use record ------------------------------------------------------------------------------------------


def test_the_use_record_carries_per_item_runtime_counts():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, _drained({PD: {"ok": 1}, TOK: {"error": 2}}))
    rec = with_runtime_calls(_use_rec({PD: {"called": 1}, WEEK: {"called": 1}}), acc)
    assert rec["items"][PD] == {
        "called": 1,
        "runtime": {"ok": 1, "refused": 0, "error": 0},
    }
    assert rec["items"][TOK] == {
        "runtime": {"ok": 0, "refused": 0, "error": 2},
    }  # called only at runtime
    assert "runtime" not in rec["items"][WEEK]
    assert rec["runtime_calls"] == {
        "cells": 1,
        "items": 2,
        "dropped": 0,
        "records_dropped": 0,
        "malformed": 0,
    }
    assert runtime_counts(rec, PD) == (1, 0, 0)
    assert runtime_counts(rec, WEEK) == (
        0,
        0,
        0,
    )  # observed: a call site that never ran


def test_without_an_observed_cell_the_record_is_unchanged():
    rec = _use_rec({PD: {"called": 1}})
    assert with_runtime_calls(dict(rec), runtime_calls_empty()) == rec
    assert with_runtime_calls(dict(rec), None) == rec
    assert runtime_counts(rec, PD) is None


def test_dropped_items_leave_uncounted_items_without_runtime_evidence():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, _drained({PD: {"ok": 1}}, dropped=4))
    rec = with_runtime_calls(_use_rec({WEEK: {"called": 1}}), acc)
    assert runtime_counts(rec, PD) == (1, 0, 0)
    assert runtime_counts(rec, WEEK) is None


def test_memory_use_attaches_the_runtime_counts():
    from unify.memory.integration.request import memory_use

    acc = runtime_calls_empty()
    add_runtime_calls(acc, _drained({TOK: {"ok": 2}}))
    rec = memory_use(_ep(transcript=[], actions=[]), [TOK], runtime_calls=acc)
    assert rec["items"][TOK]["runtime"] == {"ok": 2, "refused": 0, "error": 0}
    assert rec["runtime_calls"]["cells"] == 1
    assert "runtime_calls" not in memory_use(_ep(transcript=[], actions=[]), [TOK])


# --- the lifecycle -------------------------------------------------------------------------------------------


def _runtime_rec(rows, counts):
    acc = runtime_calls_empty()
    add_runtime_calls(acc, _drained(counts))
    return with_runtime_calls(_use_rec(rows), acc)


def test_lifecycle_prefers_runtime_counts_and_marks_the_basis():
    rec = _runtime_rec(
        {
            PD: {"imported": 1, "called": 1},  # static: used; runtime: never ran
            WEEK: {
                "called": 1,
                "refused": 1,
            },  # static: not used; runtime: returned once, refused twice
        },
        {WEEK: {"ok": 1, "refused": 2}, TOK: {"error": 1}},
    )
    f = lc.facts_from_use("e1", A, "implicit", rec, [])
    assert f.items == {
        PD: lc.ItemUse(
            False,
            1,
            False,
            False,
            False,
            seen=True,
            basis="runtime",
            calls=(0, 0, 0),
        ),
        WEEK: lc.ItemUse(True, 1, True, False, False, basis="runtime", calls=(1, 2, 0)),
        TOK: lc.ItemUse(
            False,
            0,
            False,
            True,
            False,
            seen=True,
            basis="runtime",
            calls=(0, 0, 1),
        ),
    }
    use = lc.aggregate_use([f])
    assert use[WEEK][A]["episodes"] == 1 and use[WEEK][A]["refused"] == 1
    assert use[PD][A]["episodes"] == 0 and use[PD][A]["seen"] == 1
    assert use[TOK][A]["errors"] == 1 and use[TOK][A]["runtime_basis"] == 1


def test_lifecycle_falls_back_to_static_counts_for_records_without_runtime_evidence():
    f = lc.facts_from_use(
        "e1",
        A,
        "implicit",
        _use_rec({PD: {"called": 2}}),
        [],
    )
    assert f.items == {PD: lc.ItemUse(True, 2, False, False, False)}
    assert f.items[PD].basis == "static" and f.items[PD].calls is None
    assert "runtime_basis" not in lc.aggregate_use([f])[PD][A]


def test_refusals_alone_never_demote_and_runtime_errors_do():
    refused = [
        lc.facts_from_use(
            f"e{i}",
            A,
            "implicit",
            _runtime_rec({}, {PD: {"refused": 3}}),
            [],
        )
        for i in range(3)
    ]
    status, rule, reason, _ = lc.decide_status(PD, ON_TRIAL, refused)
    assert (status, rule) == (ON_TRIAL, None) and "refused" in reason
    erred = [
        lc.facts_from_use(
            f"e{i}",
            A,
            "implicit",
            _runtime_rec({PD: {"called": 5}}, {PD: {"ok": 4, "error": 1}}),
            [],
        )
        for i in range(2)
    ]
    assert lc.decide_status(PD, ON_TRIAL, erred)[:2] == (RETIRED, "errors")
    clean = [
        lc.facts_from_use(
            f"e{i}",
            A,
            "implicit",
            _runtime_rec({}, {PD: {"ok": 1}}),
            [],
        )
        for i in range(3)
    ]
    assert lc.decide_status(PD, ON_TRIAL, clean)[0] == ADOPTED


def test_the_evidence_index_and_the_record_agree_on_runtime_facts(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    rec = _runtime_rec(
        {PD: {"imported": 1, "called": 1}, WEEK: {"called": 1}},
        {WEEK: {"ok": 1, "refused": 1}, TOK: {"error": 1}},
    )
    ev.index_episode(
        _ep(episode_id="e1", regime="implicit", memory_main=A, memory_use=rec),
        "1" * 40,
    )
    old = _use_rec({PD: {"called": 2}})
    ev.index_episode(
        _ep(episode_id="e2", regime="implicit", memory_main=A, memory_use=old),
        "2" * 40,
    )
    indexed = {
        f.episode_id: f for f in lc.facts_from_evidence(ev)
    }
    direct = lc.facts_from_use("e1", A, "implicit", rec, [])
    assert indexed["e1"].items == direct.items
    assert indexed["e2"].items == {PD: lc.ItemUse(True, 2, False, False, False)}
    row = ev.item_use(["e1"])[WEEK]
    assert (row["runtime_basis"], row["runtime_ok"], row["runtime_refused"]) == (
        1,
        1,
        1,
    )


def test_library_mode_installs_the_call_hook_without_a_work_tree_capture(monkeypatch):
    """The runtime calls "used" rests on are recorded even when no work tree is captured (the library copy is the
    only file root then). The library layout is always on, so a run always has the hook; no run, none."""
    from unify.memory.integration import hooks, worktree_capture

    monkeypatch.setattr(hooks, "_run", lambda: object())
    monkeypatch.setattr(worktree_capture, "active", lambda: None)
    got = hooks.worker_audit()
    from unify.sandbox import LIBRARY_MOUNT

    # no file roots but the library copy's: its reads count as lookups, and no work-tree event is recorded
    assert got is not None and got["roots"] == [LIBRARY_MOUNT] and got["path"].endswith("audit.py")
    monkeypatch.setattr(hooks, "_run", lambda: None)
    assert hooks.worker_audit() is None


# --- the library package's own helpers ---------------------------------------------------------------------


def test_the_package_helpers_calls_are_counted_by_helper_and_outcome():
    from unify.memory.integration.request import library_failures, library_helper_calls

    acc = runtime_calls_empty()
    assert library_helper_calls(acc) == {
        name: {"ok": 0, "refused": 0, "error": 0}
        for name in ("ask", "get", "items", "match", "run")  # memory.ask counts too
    }
    first = _drained({})
    first["calls"]["helpers"] = {"memory:get": {"ok": 2}, "memory:items": {"ok": 1, "error": 1}}
    second = _drained({PD: {"ok": 1}})
    second["calls"]["helpers"] = {
        "memory:get": {"ok": 1},
        "memory:match": {"refused": 1},
        "other": {"ok": 5},
        "memory:show": {"ok": 1},  # an earlier call's id: no longer one of the package's helpers
    }
    add_runtime_calls(acc, first)
    add_runtime_calls(acc, second)
    assert library_helper_calls(acc) == {
        "ask": {"ok": 0, "refused": 0, "error": 0},
        "get": {"ok": 3, "refused": 0, "error": 0},
        "items": {"ok": 1, "refused": 0, "error": 1},
        "match": {"ok": 0, "refused": 1, "error": 0},
        "run": {"ok": 0, "refused": 0, "error": 0},
    }
    assert acc["malformed"] == 2  # the ids that are not one of the package's helpers
    # the failures still count the helpers' refused and error calls once
    assert library_failures(acc, {})["library_calls_failed"] == 2
