"""The status and evidence-level names: the mapping of stored names, and a library written in the old names."""

from __future__ import annotations

import json

import pytest

from tests.memory.test_library_helper import DATE, _commit, _library, _py
from unify.memory import item_records as ir
from unify.memory import lifecycle as lc
from unify.memory.gitio import Repo
from unify.memory.item_records import empty_record
from unify.memory.item_states import (
    ADOPTED,
    CURATE_RULE,
    ON_TRIAL,
    REPRODUCES_ITS_CASE,
    RETIRED,
    UNCHECKED,
    level_from_stored,
    level_of_record,
    status_from_stored,
)

WEEK, PD = "memory.text.dates:week", "memory.text.dates:parse_date"
TOK, NOTE = "memory.text.parse:tokens", "notes/text/dates.md"
A, B = "a" * 40, "b" * 40


def test_stored_statuses_read_in_todays_names():
    assert [status_from_stored(s) for s in ("experimental", "stable", "suspect", "deprecated")] == [
        ON_TRIAL,
        ADOPTED,
        RETIRED,
        RETIRED,
    ]
    assert [status_from_stored(s) for s in (ON_TRIAL, ADOPTED, RETIRED)] == [ON_TRIAL, ADOPTED, RETIRED]
    assert status_from_stored(None) == ON_TRIAL and status_from_stored("gold") == ON_TRIAL
    assert status_from_stored("gold", default=None) is None


def test_stored_levels_read_in_todays_names():
    assert level_from_stored("verified") == REPRODUCES_ITS_CASE
    assert level_from_stored("unverified") == UNCHECKED and level_from_stored("experimental") == UNCHECKED
    assert level_from_stored(REPRODUCES_ITS_CASE) == REPRODUCES_ITS_CASE
    assert level_from_stored(None) == UNCHECKED  # a missing level fails closed
    assert level_from_stored(None, default=None) is None


def test_the_level_of_a_verification_record():
    # no level and no examples, or no record at all: unchecked
    assert level_of_record({"tests": ["t.py"]}) == UNCHECKED
    assert level_of_record(None) == UNCHECKED
    # an older record whose example calls ran on recorded values
    assert level_of_record({"examples": {"calls": 3, "grounded_calls": 2}}) == REPRODUCES_ITS_CASE
    assert level_of_record({"examples": {"calls": 3, "grounded_calls": 0}}) == UNCHECKED
    # a stored level wins, old names mapped
    assert level_of_record({"level": UNCHECKED, "examples": {"grounded_calls": 2}}) == UNCHECKED
    assert level_of_record({"level": "verified"}) == REPRODUCES_ITS_CASE


def _old(item: str, kind: str, status: str, rule: str | None, **kw) -> dict:
    """A record as a library written before the rename stores it: the old status name, no evidence fields."""
    rec = empty_record(item, kind)
    for key in ("level", "level_reasons", "role", "reader", "reads_for", "uid"):
        rec.pop(key)
    rec.update(status=status, status_rule=rule, changed_at=A, **kw)
    return rec


OLD = {
    # an alias the curator made: the old code returned rule None for a kept curator's retirement
    WEEK: _old(WEEK, "function", "deprecated", None, verification={"tests": ["t.py"]}),
    PD: _old(
        PD,
        "function",
        "suspect",
        "errors",
        status_reason="it raised in 2 episodes",
        verification={"tests": ["t.py"], "examples": {"calls": 2, "grounded_calls": 2, "episodes": ["e1"]}},
    ),
    TOK: _old(TOK, "function", "stable", None, verification={"tests": ["t.py"]}),
    NOTE: _old(NOTE, "note", "experimental", None),
}


@pytest.fixture
def mem(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    _commit(repo, {"memory/text/dates.py": "def week(s):\n    return 41\n"}, "library")
    return repo


def _write_old_map(mem: Repo, sha: str) -> None:
    """The stored bytes of an old map, written as note lines directly (write_records refuses the old names)."""

    def line(obj: dict) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))

    lines = [line({"items_map": ir.RECORD_VERSION, "commit": sha, "count": len(OLD)})]
    lines += [line(OLD[i]) for i in sorted(OLD)]
    mem.append_note_lines(sha, lines, ir.NOTES_REF)


def test_the_next_map_keeps_a_curators_retirement_and_decides_an_outcome_retirement_again(mem):
    head = mem.head()
    _write_old_map(mem, head)
    prev = ir.read_records(mem, head)
    base = dict(
        items={WEEK: "function", PD: "function", TOK: "function", NOTE: "note"},
        facts=[],
        since={WEEK: {A, B}, PD: {A, B}, TOK: {A, B}, NOTE: {A, B}},
        # WEEK has a new version: a curator's retirement carries across versions
        changed_at={WEEK: B, PD: A, TOK: A, NOTE: A},
        functions={WEEK: "memory.text.dates", PD: "memory.text.dates", TOK: "memory.text.parse"},
        graph={"memory.text.dates": {"memory.text.parse"}, "memory.text.parse": set()},
        note_uses={NOTE: [PD]},
    )
    recs, changes = lc.next_records(**base, prev=prev)
    assert (recs[WEEK]["status"], recs[WEEK]["status_rule"]) == (RETIRED, CURATE_RULE)
    # no use of PD's version went wrong this time: the retirement by outcomes is decided again
    assert (recs[PD]["status"], recs[PD]["status_rule"]) == (ON_TRIAL, None)
    assert recs[TOK]["status"] == ADOPTED and recs[NOTE]["status"] == ON_TRIAL
    assert changes == {PD: ON_TRIAL}
    # the curator's retirement taints nothing (TOK's importers, WEEK's module mates and NOTE stay as they are)
    assert all(r["status_rule"] != "taint" for r in recs.values())
    # the evidence fields: WEEK's new version has no verification (unchecked); PD keeps its own (its examples ran)
    assert (recs[WEEK]["level"], recs[PD]["level"], recs[TOK]["level"], recs[NOTE]["level"]) == (
        UNCHECKED,
        REPRODUCES_ITS_CASE,
        UNCHECKED,
        None,
    )
    # the map after it, written in today's names, keeps the curator's retirement sticky too
    again, _ = lc.next_records(**{**base, "changed_at": {WEEK: A, PD: A, TOK: A, NOTE: A}}, prev=recs)
    assert (again[WEEK]["status"], again[WEEK]["status_rule"]) == (RETIRED, CURATE_RULE)
    # the old records given directly give the same map
    assert lc.next_records(**base, prev=OLD) == (recs, changes)


def test_the_gates_level_fields_reach_the_record():
    recs, _ = lc.next_records(
        items={WEEK: "function", PD: "function"},
        prev={},
        facts=[],
        since={WEEK: {A}, PD: {A}},
        changed_at={WEEK: A, PD: A},
        functions={WEEK: "memory.text.dates", PD: "memory.text.dates"},
        graph={"memory.text.dates": set()},
        note_uses={},
        verification={
            PD: {"level": UNCHECKED, "level_reasons": ["exact assertion: none"], "reads_for": ["b.json", "a.json"]},
            WEEK: {"level": REPRODUCES_ITS_CASE, "readers": [PD]},
        },
    )
    assert (recs[PD]["level"], recs[PD]["level_reasons"]) == (UNCHECKED, ["exact assertion: none"])
    assert (recs[PD]["role"], recs[PD]["reads_for"], recs[PD]["reader"]) == ("reader", ["a.json", "b.json"], [])
    assert (recs[WEEK]["level"], recs[WEEK]["role"], recs[WEEK]["reader"]) == (REPRODUCES_ITS_CASE, None, [PD])


def test_a_record_missing_its_level_shows_its_support_in_the_index_and_show(tmp_path):
    """Rev 2.1: the index and show give what an item rests on (its kind and episodes), never an evidence level."""
    rec = {k: v for k, v in empty_record(DATE, "function").items() if k != "level"}
    root = _library(tmp_path, records={DATE: rec})
    index = (root / "INDEX.md").read_text()
    assert "Parse a date written as YYYY-MM-DD. [on-trial, curated item from 0 episodes]; " in index
    out = json.loads(_py(root, f"import json, memory\nprint(json.dumps(memory.show({DATE!r})))"))
    head = "memory.text.dates:parse_date(s: str) -> str  [on-trial, curated item from 0 episodes]"
    assert out.splitlines()[0] == head
    assert "Evidence level" not in out


def test_the_index_and_show_give_the_support_and_the_gates_findings(tmp_path):
    reasons = ["examples: one call returns on a recorded value", "exact assertion: none"]
    rec = {**empty_record(DATE, "function"), "level": UNCHECKED, "level_reasons": reasons,
           "provenance": {"episodes": ["e1", "e2"]}}  # fmt: skip
    root = _library(tmp_path, records={DATE: rec})
    assert "YYYY-MM-DD. [on-trial, curated item from 2 episodes]; " in (root / "INDEX.md").read_text()
    out = json.loads(_py(root, f"import json, memory\nprint(json.dumps(memory.show({DATE!r})))"))
    assert out.splitlines()[1:4] == ["Gate findings:", "  " + reasons[0], "  " + reasons[1]]
