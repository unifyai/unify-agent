"""Item records: one map per library commit on refs/notes/items, written once, never rewritten."""

from __future__ import annotations

import json

import pytest

from tests.memory.test_library_helper import _commit
from unify.memory import item_records as ir
from unify.memory.item_states import ADOPTED, ON_TRIAL, RETIRED
from unify.memory.gitio import Repo
from unify.memory.memory_repo import MemoryRepo

ITEM = "memory.text.dates:week"
ZERO = {
    "commits": 0,
    "episodes": 0,
    "uses": 0,
    "seen": 0,
    "errors": 0,
    "refused": 0,
    "negative_signals": 0,
    "positive_signals": 0,
}


def _rec(item=ITEM, status=ON_TRIAL, **kw):
    r = ir.empty_record(item, "function")
    r.update(status=status, **kw)
    return r


@pytest.fixture
def mem(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    _commit(repo, {"memory/text/dates.py": "def week(s):\n    return 41\n"}, "library")
    return repo


def test_a_map_is_written_once_and_read_back(mem):
    head = mem.head()
    recs = {
        ITEM: _rec(
            status=RETIRED,
            status_rule="errors",
            status_reason="its input check failed or it raised in 2 episodes",
        ),
    }
    assert ir.write_records(mem, head, recs) is True
    assert ir.read_records(mem, head) == recs
    before = mem.run("notes", "--ref=items", "show", head)
    assert ir.write_records(mem, head, {ITEM: _rec(status=ADOPTED)}) is False
    assert mem.run("notes", "--ref=items", "show", head) == before
    assert json.loads(before.splitlines()[0]) == {
        "items_map": 1,
        "commit": head,
        "count": 1,
    }


def test_an_unrecorded_commit_reads_its_nearest_recorded_ancestor(mem):
    first = mem.head()
    assert ir.records_at(mem, first) == ({}, None)
    ir.write_records(mem, first, {ITEM: _rec(status=RETIRED)})
    second = _commit(
        mem,
        {"notes/text/dates.md": "---\ntitle: D\ndescription: d\n---\nx\n"},
        "note",
    )
    recs, source = ir.records_at(mem, second)
    assert source == first and recs[ITEM]["status"] == RETIRED
    assert ir.status_of(recs)(ITEM) == RETIRED
    assert ir.status_of(recs)("memory.x.y:z") == ON_TRIAL


def test_unknown_statuses_and_mismatched_ids_are_refused(mem):
    with pytest.raises(ValueError, match="unknown status"):
        ir.write_records(mem, mem.head(), {ITEM: _rec(status="great")})
    with pytest.raises(ValueError, match="names"):
        ir.write_records(mem, mem.head(), {"memory.a.b:c": _rec()})
    assert ir.read_records(mem, mem.head()) is None
    assert ir.status_of({ITEM: {"status": "great"}})(ITEM) == ON_TRIAL
    # the old status names are no longer written
    with pytest.raises(ValueError, match="unknown status"):
        ir.write_records(mem, mem.head(), {ITEM: _rec(status="experimental")})


def test_a_line_separator_in_a_value_survives(mem):
    ir.write_records(mem, mem.head(), {ITEM: _rec(status_reason="a b")})
    assert ir.read_records(mem, mem.head())[ITEM]["status_reason"] == "a b"


def test_the_status_commit_changes_no_file_and_carries_trailers(mem):
    parent = mem.head()
    sha = MemoryRepo(mem).status_commit({ITEM: RETIRED}, ["e2", "e1", "e2"])
    assert mem.head() == sha and mem.run("rev-parse", f"{sha}^").strip() == parent
    assert mem.run("rev-parse", f"{sha}^{{tree}}") == mem.run(
        "rev-parse",
        f"{parent}^{{tree}}",
    )
    body = mem.run("log", "-1", "--format=%B", sha)
    assert f"Status: {ITEM} retired" in body and "Evidence: e1\nEvidence: e2" in body
    assert (
        "def week" in mem.show("main", "memory/text/dates.py").decode()
    )  # the code stays (show reaches it)
    with pytest.raises(ValueError):
        MemoryRepo(mem).status_commit({}, [])


def test_summaries_for_show():
    rec = _rec(
        verification={
            "tests": ["t.py"],
            "drawn_inputs_read": 8,
            "cross_episode": {"ran": 3, "ok": 3},
            "mutation": {"killed": 4, "total": 5, "equivalent": 0},
            "guard": None,
        },
        use={
            "a"
            * 40: {
                "episodes": 2,
                "uses": 3,
                "errors": 0,
                "refused": 0,
                "negative_signals": 0,
                "positive_signals": 1,
                "unknown": 0,
            },
            "b"
            * 40: {
                "episodes": 1,
                "uses": 1,
                "errors": 1,
                "refused": 1,
                "negative_signals": 0,
                "positive_signals": 0,
                "unknown": 0,
            },
        },
    )
    # no summary of a verification record (its tests, examples, mutation) is served: no test result is evidence
    assert not hasattr(ir, "verification_summary") and not hasattr(ir, "effect_of")
    assert ir.use_totals(rec) == {
        "commits": 2,
        "episodes": 3,
        "uses": 4,
        "seen": 0,  # a record written before "seen" existed reads as 0
        "errors": 1,
        "refused": 1,  # refusals are reported beside errors
        "negative_signals": 0,
        "positive_signals": 1,
    }
    assert ir.use_totals(_rec()) == ZERO


def test_the_effect_stays_a_record_field_for_the_writer():
    assert ir.empty_record(ITEM, "function")["effect"] is None
