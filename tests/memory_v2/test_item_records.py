"""Item records (spec §4.4, D40): one map per library commit on refs/notes/items, written once, never rewritten."""

from __future__ import annotations

import json

import pytest

from tests.memory_v2.test_library_helper import _commit
from unify.memory_v2 import item_records as ir
from unify.memory_v2.gitio import Repo
from unify.memory_v2.memory_repo import MemoryRepo

ITEM = "memory.text.dates:week"
ZERO = {
    "commits": 0,
    "episodes": 0,
    "uses": 0,
    "errors": 0,
    "negative_signals": 0,
    "positive_signals": 0,
}


def _rec(item=ITEM, status="experimental", **kw):
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
            status="suspect",
            status_reason="its input check failed or it raised in 2 episodes",
        ),
    }
    assert ir.write_records(mem, head, recs) is True
    assert ir.read_records(mem, head) == recs
    before = mem.run("notes", "--ref=items", "show", head)
    assert ir.write_records(mem, head, {ITEM: _rec(status="stable")}) is False
    assert mem.run("notes", "--ref=items", "show", head) == before
    assert json.loads(before.splitlines()[0]) == {
        "items_map": 1,
        "commit": head,
        "count": 1,
    }


def test_an_unrecorded_commit_reads_its_nearest_recorded_ancestor(mem):
    first = mem.head()
    assert ir.records_at(mem, first) == ({}, None)
    ir.write_records(mem, first, {ITEM: _rec(status="suspect")})
    second = _commit(
        mem,
        {"notes/text/dates.md": "---\ntitle: D\ndescription: d\n---\nx\n"},
        "note",
    )
    recs, source = ir.records_at(mem, second)
    assert source == first and recs[ITEM]["status"] == "suspect"
    assert ir.status_of(recs)(ITEM) == "suspect"
    assert ir.status_of(recs)("memory.x.y:z") == "experimental"


def test_unknown_statuses_and_mismatched_ids_are_refused(mem):
    with pytest.raises(ValueError, match="unknown status"):
        ir.write_records(mem, mem.head(), {ITEM: _rec(status="great")})
    with pytest.raises(ValueError, match="names"):
        ir.write_records(mem, mem.head(), {"memory.a.b:c": _rec()})
    assert ir.read_records(mem, mem.head()) is None
    assert ir.status_of({ITEM: {"status": "great"}})(ITEM) == "experimental"


def test_a_line_separator_in_a_value_survives(mem):
    ir.write_records(mem, mem.head(), {ITEM: _rec(status_reason="a b")})
    assert ir.read_records(mem, mem.head())[ITEM]["status_reason"] == "a b"


def test_the_status_commit_changes_no_file_and_carries_trailers(mem):
    parent = mem.head()
    sha = MemoryRepo(mem).status_commit({ITEM: "suspect"}, ["e2", "e1", "e2"])
    assert mem.head() == sha and mem.run("rev-parse", f"{sha}^").strip() == parent
    assert mem.run("rev-parse", f"{sha}^{{tree}}") == mem.run(
        "rev-parse",
        f"{parent}^{{tree}}",
    )
    body = mem.run("log", "-1", "--format=%B", sha)
    assert f"Status: {ITEM} suspect" in body and "Evidence: e1\nEvidence: e2" in body
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
    assert ir.verification_summary(rec) == {
        "tests": 1,
        "drawn_inputs_read": 8,
        "cross_episode": "3/3 ok",
        "mutation": "4/5 killed",
    }
    assert ir.use_totals(rec) == {
        "commits": 2,
        "episodes": 3,
        "uses": 4,
        "errors": 1,
        "negative_signals": 0,
        "positive_signals": 1,
    }
    assert ir.verification_summary(_rec()) is None and ir.use_totals(_rec()) == ZERO
