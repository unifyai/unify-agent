"""Item records at consolidation: one map per commit, status commits, copies. Rev 2 bisects no retired item (a bisect
would replay recorded task data)."""

from __future__ import annotations

import json

import pytest

from tests.memory.integration.test_consolidate import (
    _stores,
)
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import LIB
from tests.memory.test_library_helper import _commit
from unify.memory import lifecycle
from unify.memory.integration import consolidate
from unify.memory.integration.checkout import export_actor_library_mode
from unify.memory.item_records import read_records, records_at, status_of

PD, WEEK, SUM = (
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.report:summary",
)
DATES_TEST = "memory/text/tests/test_dates.py"


def _episode(stores, eid, main, rows, regime="none"):
    rec = {
        "version": 6,
        "items": rows,
        "items_outcome_unknown": [],
        "outcomes_known": True,
        "layout": "library_mode",
    }
    stores.evidence.index_episode(
        _ep(episode_id=eid, memory_main=main, regime=regime, memory_use=rec),
        "1" * 40,
    )


def _consolidate(stores):
    return lifecycle.consolidate_records(stores)


def _copy(stores, sha, dest):
    recs = records_at(stores.memory, sha)[0]
    return export_actor_library_mode(
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
                [
                    "library_mode-verification "
                    + json.dumps(verification, sort_keys=True),
                ],
            ),
        },
    )
    return stores, head


def test_status_change_lands_as_a_commit_and_notes_are_never_rewritten(lib, tmp_path):
    stores, head = lib
    for i in range(2):
        _episode(stores, f"e{i}", head, {WEEK: {"called": 1, "errored": 1}})
    _consolidate(stores)
    note = stores.memory.run("notes", "--ref=items", "show", head)
    copy = _copy(stores, head, tmp_path / "a")
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    out = _consolidate(stores)
    status = stores.memory.head()
    assert out["noted"] == status != head and out["changes"] == {PD: "adopted"}
    assert stores.memory.run("rev-parse", f"{status}^").strip() == head
    assert stores.memory.run("rev-parse", f"{status}^{{tree}}") == stores.memory.run(
        "rev-parse",
        f"{head}^{{tree}}",
    )
    assert f"Status: {PD} adopted" in stores.memory.run(
        "log",
        "-1",
        "--format=%B",
        status,
    )
    assert stores.memory.run("notes", "--ref=items", "show", head) == note
    assert (
        _copy(stores, head, tmp_path / "b") == copy
    )  # the copy of a commit never changes
    assert read_records(stores.memory, status)[PD]["status"] == "adopted"
    again = _consolidate(stores)
    assert (
        again["noted"] is None
        and again["changes"] == {}
        and stores.memory.head() == status
    )


def test_a_new_version_lands_with_its_pass_commit_and_restarts_its_count(lib, tmp_path):
    stores, head = lib
    for i in range(2):
        _episode(stores, f"e{i}", head, {WEEK: {"called": 1, "errored": 1}})
    _consolidate(stores)
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
    out = _consolidate(stores)
    recs = read_records(stores.memory, new)
    assert out["noted"] == new and out["changes"] == {
        WEEK: "on-trial",
        SUM: "on-trial",
    }
    assert recs[WEEK]["changed_at"] == new and recs[WEEK]["use"][head]["errors"] == 2
    assert recs[WEEK]["bisect"] is None


def test_after_passes_runs_only_after_a_recorded_pass(
    lib,
    monkeypatch,
):
    from unify.memory.captures import store_dir
    stores, _ = lib
    seen = []

    def spy(stores, **kw):
        seen.append(kw)
        return {"head": "h", "noted": None, "changes": {}}

    monkeypatch.setattr(lifecycle, "consolidate_records", spy)
    consolidate.after_passes(
        stores,
        [],
        None,
    )
    assert seen == []
    consolidate.after_passes(
        stores,
        ["p1"],
        None,
    )
    # the library is always on in rev 2: a recorded pass computes the records, with the episodes' actions and the
    # capture store
    assert [sorted(kw) for kw in seen] == [["action_lookup", "captures"]]
    assert seen[0]["captures"] == store_dir(stores.paths.state_dir) and callable(seen[0]["action_lookup"])


def test_an_item_whose_last_change_is_beyond_the_scan_is_not_promoted(
    lib,
    tmp_path,
    monkeypatch,
):
    """With its version unknown, three clean uses do not make an item adopted; with the history
    found in its own file it is promoted as usual."""
    from unify.memory import item_versions, library_export

    stores, head = lib
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    monkeypatch.setattr(
        library_export,
        "item_history",
        lambda mem, sha, **k: ({}, False),
    )
    monkeypatch.setattr(item_versions, "versions_of", lambda *a, **k: [])
    out = _consolidate(stores)
    rec = read_records(stores.memory, out["noted"])[PD]
    assert (
        rec["status"] == "on-trial"
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
        == "on-trial"
    )


def test_the_last_change_comes_from_the_items_own_history_when_the_scan_misses_it(
    lib,
    tmp_path,
    monkeypatch,
):
    from unify.memory import library_export

    stores, head = lib
    for i in range(3):
        _episode(stores, f"c{i}", head, {PD: {"called": 1}})
    monkeypatch.setattr(
        library_export,
        "item_history",
        lambda mem, sha, **k: ({}, False),
    )
    out = _consolidate(stores)
    rec = read_records(stores.memory, out["noted"])[PD]
    assert rec["changed_at"] == head and rec["status"] == "adopted"
