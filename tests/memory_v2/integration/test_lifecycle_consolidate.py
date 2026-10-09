"""Item records at consolidation (spec §4.4, §10; D38, D40): one map per commit, status commits, bisect, copies."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory_v2.integration.test_consolidate import (
    FakeSol,
    _record,
    _settings,
    _stores,
)
from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_layout import LIB
from tests.memory_v2.test_library_helper import _commit
from unify.memory_v2 import lifecycle
from unify.memory_v2.history_probe import ProbeRow
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration.checkout import export_actor_v21
from unify.memory_v2.integration.state import State
from unify.memory_v2.integration.switch import checker_visible
from unify.memory_v2.item_records import read_records, records_at, status_of

PD, WEEK, SUM = (
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.report:summary",
)
DATES_TEST = "memory/text/tests/test_dates.py"


class _Probe:
    def __init__(self):
        self.calls = []

    def __call__(self, mem, versions, later, tests, *, work):
        self.calls.append((list(versions), list(tests)))
        return [
            ProbeRow(label, sha, t, failed=["test_dates.py::test_week"])
            for label, sha in versions
            for t in tests
        ]


def _episode(stores, eid, main, rows, regime="none"):
    rec = {
        "version": 6,
        "items": rows,
        "items_outcome_unknown": [],
        "outcomes_known": True,
        "layout": "v21",
    }
    stores.evidence.index_episode(
        _ep(episode_id=eid, memory_main=main, regime=regime, memory_use=rec),
        "1" * 40,
    )


def _consolidate(stores, tmp_path, probe=None):
    return lifecycle.consolidate_records(
        stores,
        checker_visible=False,
        work=tmp_path / "w",
        probe=probe or _Probe(),
        refused=lambda ev, m, i: [],
    )


def _copy(stores, sha, dest):
    recs = records_at(stores.memory, sha)[0]
    return export_actor_v21(
        stores.memory.git_dir,
        sha,
        dest,
        status_of=status_of(recs),
        records=recs,
    )


@pytest.fixture
def lib(tmp_path):
    stores = _stores(tmp_path)
    head = _commit(stores.memory, LIB, "library")
    verification = {WEEK: {"tests": [DATES_TEST]}, PD: {"tests": [DATES_TEST]}}
    stores.evidence.record_pass(
        {
            "pass_id": "p1",
            "kind": "write",
            "passed": 1,
            "items_merged": json.dumps([PD, WEEK]),
            "reasons": json.dumps(
                ["v21-verification " + json.dumps(verification, sort_keys=True)],
            ),
        },
    )
    return stores, head


def test_a_consolidation_records_the_map_on_main_and_hides_a_suspect(lib, tmp_path):
    stores, head = lib
    _episode(stores, "e1", head, {WEEK: {"called": 1, "errored": 1}})
    _episode(stores, "e2", head, {WEEK: {"called": 1, "errored": 1}, PD: {"called": 1}})
    probe = _Probe()
    out = _consolidate(stores, tmp_path, probe)
    assert out["noted"] == head and out["changes"] == {WEEK: "suspect", SUM: "suspect"}
    recs = read_records(stores.memory, head)
    assert (recs[WEEK]["status_rule"], recs[WEEK]["status_evidence"]) == (
        "errors",
        ["e1", "e2"],
    )
    assert (recs[SUM]["status_rule"], recs[SUM]["status_evidence"]) == ("taint", [WEEK])
    assert recs[WEEK]["use"] == {
        head: {
            "episodes": 2,
            "uses": 2,
            "errors": 2,
            "refused": 0,
            "negative_signals": 0,
            "positive_signals": 0,
            "unknown": 0,
        },
    }
    assert (
        recs[WEEK]["verification"]["tests"] == [DATES_TEST]
        and recs[WEEK]["provenance"]["pass"] == "p1"
    )
    assert out["bisected"] == [WEEK] and probe.calls == [([("v0", head)], [DATES_TEST])]
    # RUNTIME S1: the map was published before the bisect; the result is stored and reaches the next map
    assert recs[WEEK]["bisect"] is None
    assert (
        out["bisects"][WEEK]["bisect"]["introduced"] == head
        and out["bisects"][WEEK]["rollback"] is None
    )
    assert lifecycle.bisect_results(stores.evidence)[WEEK][0] == head
    files = _copy(stores, head, tmp_path / "co")
    index = files["INDEX.md"].decode()
    assert "week(" not in index and "summary(" not in index and "parse_date(" in index
    items = json.loads(files[".memory/items.json"])["items"]
    assert (
        items[WEEK]["status"] == "suspect"
        and "raised in 2 episodes" in items[WEEK]["status_reason"]
    )
    assert items[WEEK]["use"]["errors"] == 2


def test_status_change_lands_as_a_commit_and_notes_are_never_rewritten(lib, tmp_path):
    stores, head = lib
    for i in range(2):
        _episode(stores, f"e{i}", head, {WEEK: {"called": 1, "errored": 1}})
    _consolidate(stores, tmp_path)
    note = stores.memory.run("notes", "--ref=items", "show", head)
    copy = _copy(stores, head, tmp_path / "a")
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    out = _consolidate(stores, tmp_path)
    status = stores.memory.head()
    assert out["noted"] == status != head and out["changes"] == {PD: "stable"}
    assert stores.memory.run("rev-parse", f"{status}^").strip() == head
    assert stores.memory.run("rev-parse", f"{status}^{{tree}}") == stores.memory.run(
        "rev-parse",
        f"{head}^{{tree}}",
    )
    assert f"Status: {PD} stable" in stores.memory.run(
        "log",
        "-1",
        "--format=%B",
        status,
    )
    assert stores.memory.run("notes", "--ref=items", "show", head) == note
    assert (
        _copy(stores, head, tmp_path / "b") == copy
    )  # the copy of a commit never changes
    assert read_records(stores.memory, status)[PD]["status"] == "stable"
    again = _consolidate(stores, tmp_path)
    assert (
        again["noted"] is None
        and again["changes"] == {}
        and stores.memory.head() == status
    )


def test_a_new_version_lands_with_its_pass_commit_and_restarts_its_count(lib, tmp_path):
    stores, head = lib
    for i in range(2):
        _episode(stores, f"e{i}", head, {WEEK: {"called": 1, "errored": 1}})
    _consolidate(stores, tmp_path)
    dates = (
        stores.memory.show(head, "memory/text/dates.py")
        .decode()
        .replace("    return 1\n", "    return 41\n")
    )
    new = _commit(stores.memory, {"memory/text/dates.py": dates}, "fix week")
    assert (
        read_records(stores.memory, new) is None
        and records_at(stores.memory, new)[1] == head
    )
    out = _consolidate(stores, tmp_path)
    recs = read_records(stores.memory, new)
    assert out["noted"] == new and out["changes"] == {
        WEEK: "experimental",
        SUM: "experimental",
    }
    assert recs[WEEK]["changed_at"] == new and recs[WEEK]["use"][head]["errors"] == 2
    assert recs[WEEK]["bisect"] is None


def test_after_passes_runs_only_under_v21_and_after_a_recorded_pass(lib, monkeypatch):
    stores, _ = lib
    seen = []

    def spy(stores, **kw):
        seen.append(kw)
        return {"head": "h", "noted": None, "changes": {}, "bisected": []}

    monkeypatch.setattr(lifecycle, "consolidate_records", spy)
    consolidate.after_passes(
        stores,
        SimpleNamespace(UNIFY_MEMORY_V21="off"),
        ["p1"],
        None,
    )
    consolidate.after_passes(stores, SimpleNamespace(UNIFY_MEMORY_V21="on"), [], None)
    assert seen == []
    consolidate.after_passes(
        stores,
        SimpleNamespace(UNIFY_MEMORY_V21="on"),
        ["p1"],
        None,
    )
    assert [kw["checker_visible"] for kw in seen] == [False]


def test_a_v21_consolidation_records_the_map_of_main(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    settings = _settings()
    settings.UNIFY_MEMORY_V21 = "on"
    asyncio.run(
        consolidate.run_due_passes(
            stores,
            "e1",
            sha,
            State(stores.paths.state),
            effort="low",
            settings=settings,
            emit=None,
        ),
    )
    assert stores.evidence.pass_exists("e1.p0")
    assert read_records(stores.memory, stores.memory.head()) == {}


def test_checker_visibility_is_a_declared_switch():
    assert checker_visible(SimpleNamespace()) is False
    assert (
        checker_visible(SimpleNamespace(UNIFY_MEMORY_V21_CHECKER_VISIBLE="on")) is True
    )
    with pytest.raises(ValueError):
        checker_visible(SimpleNamespace(UNIFY_MEMORY_V21_CHECKER_VISIBLE="yes"))


# --- the bisect bound (MAIN, 9 Oct; AGENTS.md: bounded runtime) ----------------------------------------------


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_the_bisect_budget_is_shared_across_items_and_the_records_are_written(
    lib,
    tmp_path,
):
    stores, head = lib
    for eid in ("e1", "e2"):
        _episode(
            stores,
            eid,
            head,
            {WEEK: {"called": 1, "errored": 1}, PD: {"called": 1, "errored": 1}},
        )
    clock, calls = _Clock(), []

    def slow(mem, versions, later, tests, *, work, deadline, stop=None):
        calls.append(deadline)
        clock.t += 700.0  # one bisect spends the whole budget
        return [
            ProbeRow(label, sha, t, failed=["x"])
            for label, sha in versions
            for t in tests
        ]

    out = lifecycle.consolidate_records(
        stores,
        checker_visible=False,
        work=tmp_path / "w",
        probe=slow,
        refused=lambda ev, m, i: [],
        clock=clock,
    )
    recs = {
        i: r["bisect"] for i, r in out["bisects"].items()
    }  # published before the bisect (RUNTIME S1)
    assert read_records(stores.memory, out["noted"])[WEEK]["status"] == "suspect"
    assert calls == [
        600.0,
    ]  # PD first (sorted), with the call's whole budget as its deadline
    assert "introduced" in recs[PD] and recs[WEEK] == {
        "skipped": "budget",
        "at": head,
    }


def test_a_stop_between_probe_runs_ends_the_bisect_and_the_records_are_written(
    lib,
    tmp_path,
):
    from functools import partial

    from unify.memory_v2 import history_probe
    from unify.memory_v2.sandbox_run import PytestOutcome

    stores, head = lib
    for eid in ("e1", "e2"):
        _episode(
            stores,
            eid,
            head,
            {WEEK: {"called": 1, "errored": 1}, PD: {"called": 1, "errored": 1}},
        )
    stopped, runs = [], []

    def runner(target, **kw):  # SIGTERM arrives while the first run is in flight
        runs.append(target)
        stopped.append(True)
        return PytestOutcome(failed={"test_dates.py::test_week"}, returncode=1)

    out = lifecycle.consolidate_records(
        stores,
        checker_visible=False,
        work=tmp_path / "w",
        probe=partial(history_probe.probe, pytest_runner=runner),
        refused=lambda ev, m, i: [],
        stop=lambda: bool(stopped),
    )
    recs = {
        i: r["bisect"] for i, r in out["bisects"].items()
    }  # published before the bisect (RUNTIME S1)
    assert read_records(stores.memory, out["noted"])[WEEK]["status"] == "suspect"
    assert runs == [DATES_TEST]  # no run starts after the stop
    assert "introduced" in recs[PD] and recs[WEEK] == {
        "skipped": "stopped",
        "at": head,
    }


def test_an_item_is_not_bisected_again_at_the_same_suspect_version(lib, tmp_path):
    from unify.memory_v2.item_records import empty_record, write_records

    stores, head = lib
    rec = {
        **empty_record(WEEK, "function"),
        "changed_at": head,
        "bisect": {"skipped": "budget", "at": head},
    }
    write_records(
        stores.memory,
        head,
        {WEEK: rec},
    )  # an earlier consolidation's map at this version
    for eid in ("e1", "e2"):
        _episode(stores, eid, head, {WEEK: {"called": 1, "errored": 1}})
    probe = _Probe()
    out = _consolidate(stores, tmp_path, probe)
    recs = read_records(stores.memory, out["noted"])
    assert recs[WEEK]["status"] == "suspect" and probe.calls == []
    assert recs[WEEK]["bisect"] == {"skipped": "budget", "at": head}


def test_the_records_are_published_before_the_bisect_and_a_killed_bisect_runs_again(
    lib,
    tmp_path,
):
    """RUNTIME S1 (1): a kill during the bisect never holds back the map; the item is bisected next time, and its
    result reaches the next map written (a commit's map is never rewritten)."""
    stores, head = lib
    for eid in ("e1", "e2"):
        _episode(stores, eid, head, {WEEK: {"called": 1, "errored": 1}})

    def killed(*a, **k):
        raise KeyboardInterrupt  # SIGINT/SIGKILL mid-bisect

    with pytest.raises(KeyboardInterrupt):
        _consolidate(stores, tmp_path / "a", killed)
    recs = read_records(stores.memory, head)
    assert recs[WEEK]["status"] == "suspect" and recs[WEEK]["bisect"] is None
    probe = _Probe()
    out = _consolidate(stores, tmp_path / "b", probe)
    assert out["noted"] is None and out["bisected"] == [WEEK] and len(probe.calls) == 1
    again = _Probe()
    for i in range(3):
        _episode(
            stores,
            f"c{i}",
            head,
            {PD: {"called": 1}},
        )  # a status change: the next map is written
    later = _consolidate(stores, tmp_path / "c", again)
    assert again.calls == [] and later["noted"] not in (None, head)
    assert (
        read_records(stores.memory, later["noted"])[WEEK]["bisect"]["introduced"]
        == head
    )


def test_an_item_whose_last_change_is_beyond_the_scan_is_not_promoted(
    lib,
    tmp_path,
    monkeypatch,
):
    """RUNTIME S1 (2): with its version unknown, three clean uses do not make an item stable; with the history
    found in its own file it is promoted as usual."""
    from unify.memory_v2 import item_bisect, library_export

    stores, head = lib
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    monkeypatch.setattr(
        library_export,
        "item_history",
        lambda mem, sha, **k: ({}, False),
    )
    monkeypatch.setattr(item_bisect, "versions_of", lambda *a, **k: [])
    out = _consolidate(stores, tmp_path / "a")
    rec = read_records(stores.memory, out["noted"])[PD]
    assert (
        rec["status"] == "experimental"
        and "beyond the scanned history" in rec["status_reason"]
    )
    monkeypatch.undo()
    assert (
        lifecycle.next_records(
            items={PD: "function"},
            prev={},
            facts=[],
            since={PD: set()},
            changed_at={PD: None},
            functions={PD: "memory.text.dates"},
            graph={},
            note_uses={},
            unknown_version=[PD],
        )[0][PD]["status"]
        == "experimental"
    )


def test_the_last_change_comes_from_the_items_own_history_when_the_scan_misses_it(
    lib,
    tmp_path,
    monkeypatch,
):
    from unify.memory_v2 import library_export

    stores, head = lib
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    monkeypatch.setattr(
        library_export,
        "item_history",
        lambda mem, sha, **k: ({}, False),
    )
    out = _consolidate(stores, tmp_path / "a")
    rec = read_records(stores.memory, out["noted"])[PD]
    assert rec["changed_at"] == head and rec["status"] == "stable"
