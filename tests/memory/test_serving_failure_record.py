"""``UNIFY_MEMORY_SERVING_FAILURE_RECORD``: serving on the situation plus each item's failure record, evidence instead
of a claim of fit, status stated by what it rests on, and library items withdrawn as captures are.
"""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from tests.memory.test_capture_examples import PRELUDE, _cell, _put
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import _tree
from unify.memory import captures, librarian, serving, trust
from unify.memory import library_export as lx
from unify.memory import library_search
from unify.memory.catalogue import write_files
from unify.memory.episodes import EpisodeWriter
from unify.memory.integration.consolidate import open_stores
from unify.memory.integration.paths import Paths
from unify.memory.integration.request import RequestRun
from unify.memory.item_records import empty_record
from unify.memory.redact import Redactor
from unify.memory.signals import Signal, post_signal

assert PRELUDE  # the cells below run with the package imported as ``memory``

PARSE_DATE = "memory.text.dates:parse_date"
WEEK = "memory.text.dates:week"
WORDS = "memory.text.parse:words"
SUMMARY = "memory.text.report:summary"
TOKENS = "memory.text.parse:tokens"
TWO_FAILURES = [
    {
        "episode": "a1b2c3d4e5f60718",
        "pin": "1a2b3c4d5e6f",
        "why": "error",
        "ref": "episode:a1b2c3d4e5f60718",
    },
    {
        "episode": "0f1e2d3c4b5a6978",
        "pin": "9a8b7c6d5e4f",
        "why": "contrary_signal",
        "ref": "signal:x.checker.1",
    },
]


def _records() -> dict[str, dict]:
    recs = {
        i: empty_record(i, "function")
        for i in (PARSE_DATE, WEEK, WORDS, SUMMARY, TOKENS)
    }
    recs[WEEK].update(
        status="retired",
        status_rule="curate",
        alias_of=PARSE_DATE,
        status_reason="an alias",
    )
    recs[WORDS].update(
        status="retired",
        status_reason="its stated result condition was false in 2 episodes",
    )
    recs[SUMMARY].update(
        superseded_by=TOKENS
    )  # superseded by the writer, its status still on trial
    return recs


def _serving(
    records: dict, evidence: dict | None = None, stated: dict | None = None
) -> dict:
    statuses = {i: r.get("status") or "on-trial" for i, r in records.items()}
    found = (
        evidence
        if evidence is not None
        else {
            PARSE_DATE: {
                "failure_record": TWO_FAILURES,
                "status_basis": trust.BASIS_END_OUTCOMES,
                "counts": {"clean": 3, "failed": 2, "neutral": 0},
                "contradictions": ["1 of its stated tests did not go as stated"],
            },
        }
    )
    return {
        "evidence": serving.evidence_doc(
            found,
            stated if stated is not None else {PARSE_DATE: "the text is a date"},
            statuses,
        ),
        "withdrawn_items": serving.withdrawn_items(records),
    }


def _copy(
    tmp_path: Path, served: dict | None, librarian_on: bool = True
) -> tuple[Path, dict]:
    tree = _tree(tmp_path / "lib")
    records = _records()
    status_of = (
        lambda i: (records.get(i) or {}).get("status") or "on-trial"
    )  # noqa: E731
    files = lx.generated_library_mode(
        tree,
        status_of=status_of,
        records=records,
        librarian=librarian_on,
        **({"serving": served} if served is not None else {}),
    )
    write_files(tree, files)
    return tree, files


# --- 1. an item with failures shows them on its card and in the librarian's input --------------------------------


def test_an_item_with_two_failures_shows_them_on_its_card_and_in_the_librarians_input(
    tmp_path,
):
    root, files = _copy(tmp_path, _serving(_records()))
    got = _cell(root, f"print(json.dumps(str(memory.get({PARSE_DATE!r}))))\n")
    assert (
        "Recorded failures: 2: error in episode a1b2c3d4 (version 1a2b3c4); contrary_signal in episode 0f1e2d3c"
        in got
    )
    assert (
        "Status by evidence: on-trial: 3 clean uses, 2 failed, all judged at episode end"
        in got
    )
    assert "Contradicted: 1 of its stated tests did not go as stated" in got
    assert "States that it applies when: the text is a date" in got
    catalogue = files[".memory/catalogue.md"].decode()
    card = catalogue.split(f"### {PARSE_DATE}\n", 1)[1].split("###", 1)[0]
    assert (
        "Recorded failures: 2: error in episode a1b2c3d4" in card
        and "Status by evidence: on-trial: 3 clean" in card
    )
    rows = {r["id"]: r for r in json.loads(files[".memory/search.json"])["items"]}
    # the search card is embedded and matched on, so its identifiers are masked (rev 2: nothing is found by an id)
    assert "recorded failures: 2: error in episode <id> (version <id>)" in rows[PARSE_DATE]["card"]
    assert "a1b2c3d4" not in rows[PARSE_DATE]["card"]
    # rev 2: no reader runs, so the message carries the situation and names the catalogue; the failures are on the
    # cards (above), never reader facts
    sent = librarian.sent_to_librarian(
        "Turn this into an ISO date.",
        {"commit": "c", "dispatch": "d"},
        evidence=librarian.read_evidence(root),
        last_message="The date is 9 October.",
        rows=3,
    )
    message = sent["message"]
    assert "The latest message the agent received after its request:\nThe date is 9 October." in message
    assert "dispatch.json, 3 row(s)" in message and "reader" not in message


def test_reader_and_applies_outcomes_are_evidence_lines_never_a_claim_of_fit(tmp_path):
    brief = librarian.system_prompt(evidence=True)
    assert librarian.BRIEF_EVIDENCE in brief and librarian.BRIEF not in brief
    assert "is evidence that the item's input fits" not in brief
    assert (
        "evidence only of what an item's code does" in brief
        and "recorded failures" in brief
    )
    assert librarian.system_prompt().endswith(librarian.BRIEF)  # off: as built
    root, _ = _copy(tmp_path, _serving(_records()))
    got = _cell(
        root,
        f"f = memory.get({PARSE_DATE!r}).fits({{'date': '2026-10-09'}})\n"
        "print(json.dumps([str(f), bool(f), type(f).__name__, f.failures, isinstance(f, memory.Fit)]))\n",
    )
    text, ok, kind, failures, is_fit = got
    assert kind == "FitEvidence" and is_fit and failures == 2 and ok is False
    assert text.startswith("not structurally similar: ") and text.endswith(
        "; 2 recorded failures"
    )
    assert "fits:" not in text


# --- 2 and 3. a retired or superseded library item is withdrawn, and its replacement offered -----------------------


def test_withdrawn_library_items_leave_every_listing_and_their_id_gives_the_replacement(
    tmp_path,
):
    root, files = _copy(tmp_path, _serving(_records()))
    got = _cell(
        root,
        "listed = [i.id for i in memory.items()]\n"
        f"moved = memory.get({WEEK!r})\n"
        f"superseded = memory.get({SUMMARY!r})\n"
        f"print(json.dumps({{'listed': listed, 'moved': moved.id, 'card': str(moved), 'superseded': superseded.id, "
        f"'retired': err(memory.get, {WORDS!r}), 'run': memory.run({SUMMARY!r}, 'x').detail, "
        f"'fit': str(memory.get({PARSE_DATE!r}).fits({{'a': 1}}))}}))\n",
    )
    assert (
        WEEK not in got["listed"]
        and WORDS not in got["listed"]
        and SUMMARY not in got["listed"]
    )
    assert PARSE_DATE in got["listed"] and TOKENS in got["listed"]
    assert got["moved"] == PARSE_DATE
    assert got["card"].startswith(
        f"{WEEK} is no longer offered: the library's writer superseded it"
    )
    assert f"{PARSE_DATE} is offered in its place." in got["card"]
    assert (
        got["superseded"] == TOKENS
    )  # a library item superseded by the writer, not only a capture
    assert (
        got["retired"][0] == "NotFound"
        and "no longer offered: the library's writer retired it" in got["retired"][1]
    )
    assert "no longer offered" in got["run"] and f"memory.get({TOKENS!r})" in got["run"]
    # the librarian's files never offer a withdrawn item, and the replacement carries what it replaced
    dispatch = json.loads(files[".memory/dispatch.json"])
    assert not {r["id"] for r in dispatch["procedures"]} & {WEEK, WORDS, SUMMARY}
    catalogue = files[".memory/catalogue.md"].decode()
    for gone in (WEEK, WORDS, SUMMARY):
        assert f"### {gone}\n" not in catalogue
    card = catalogue.split(f"### {PARSE_DATE}\n", 1)[1].split("###", 1)[0]
    assert f"- In place of {WEEK} (superseded): The ISO week of a date." in card
    ids = {r["id"] for r in json.loads(files[".memory/search.json"])["items"]}
    assert not ids & {WEEK, WORDS, SUMMARY}
    assert {WEEK, WORDS, SUMMARY} <= set(
        json.loads(files[".memory/withdrawn_items.json"])["items"]
    )
    find = json.loads(files[".memory/find.json"])
    assert not {f["id"] for f in find.get("functions", [])} & {WEEK, WORDS, SUMMARY}


def test_the_withdrawn_items_follow_chains_and_stop_at_loops_and_dead_ends():
    recs = {
        "a": {"status": "on-trial", "superseded_by": "b"},
        "b": {"status": "retired", "alias_of": "c"},
        "c": {"status": "on-trial"},
        "x": {"status": "on-trial", "superseded_by": "y"},
        "y": {"status": "on-trial", "superseded_by": "x"},
        "r": {"status": "retired"},
        "s": {"status": "suspect"},  # an old name of retired
        "self": {"status": "on-trial", "superseded_by": "self"},
    }
    got = serving.withdrawn_items(recs)
    assert got["a"]["by"] == "c" and got["a"]["status"] == "superseded"
    assert got["b"]["by"] == "c"
    assert (
        got["x"]["by"] is None and got["y"]["by"] is None
    )  # a loop offers nothing in its place
    assert got["r"] == {"status": "retired", "by": None, "reason": "retired"}
    assert got["s"]["status"] == "retired"
    assert "c" not in got and "self" not in got


# --- 4. status text for each basis ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basis, counts, text",
    [
        (
            trust.BASIS_NO_OUTCOME_SIGNALS,
            {},
            "on-trial: no outcome signal has arrived for any use of this item",
        ),
        (
            trust.BASIS_IMPLICIT,
            {"clean": 1},
            "on-trial: 1 clean use, 0 failed, judged by implicit signals only",
        ),
        (
            trust.BASIS_END_OUTCOMES,
            {"clean": 3, "failed": 1},
            "on-trial: 3 clean uses, 1 failed, all judged at episode end",
        ),
        (
            trust.BASIS_VERDICTS,
            {"clean": 2},
            "on-trial: 2 clean uses, 0 failed, judged by the verdict after each",
        ),
    ],
)
def test_the_status_text_for_each_basis_reaches_the_card(tmp_path, basis, counts, text):
    evidence = {
        PARSE_DATE: {"failure_record": [], "status_basis": basis, "counts": counts}
    }
    root, files = _copy(tmp_path, _serving(_records(), evidence=evidence, stated={}))
    got = _cell(root, f"print(json.dumps(str(memory.get({PARSE_DATE!r}))))\n")
    assert f"Status by evidence: {text}" in got and "Recorded failures: none" in got
    assert f"Status by evidence: {text}" in files[".memory/catalogue.md"].decode()


def test_an_item_without_evidence_says_no_outcome_signal_has_arrived(tmp_path):
    root, _ = _copy(tmp_path, _serving(_records(), evidence={}, stated={}))
    got = _cell(root, f"print(json.dumps(str(memory.get({TOKENS!r}))))\n")
    assert (
        "Status by evidence: no outcome signal has arrived for any use of this item"
        in got
    )


# --- off: every byte as built -------------------------------------------------------------------------------------


def test_off_the_copy_is_byte_for_byte_as_built(tmp_path):
    _, off = _copy(tmp_path / "off", None)
    tree = _tree(tmp_path / "plain" / "lib")
    records = _records()
    status_of = (
        lambda i: (records.get(i) or {}).get("status") or "on-trial"
    )  # noqa: E731
    built = lx.generated_library_mode(
        tree, status_of=status_of, records=records
    )
    assert off == built
    assert serving.EVIDENCE_FILE not in off and serving.WITHDRAWN_ITEMS_FILE not in off
    assert library_search.rows(
        tree, status_of=status_of, records=records
    ) == library_search.rows(tree, status_of=status_of, records=records, serving=None)
    root, _ = _copy(tmp_path / "cells", None)
    got = _cell(
        root,
        f"f = memory.get({PARSE_DATE!r}).fits({{'a': 1}})\n"
        f"print(json.dumps([type(f).__name__, str(f), 'Recorded failures' in str(memory.get({PARSE_DATE!r}))]))\n",
    )
    assert got[0] == "Fit" and got[1].startswith("does not fit: ") and got[2] is False


# --- the evidence, from the stores ----------------------------------------------------------------------------------


ITEM = "memory.reports.totals:sum_rows"


def _record(
    stores, eid, minute, *, errored=0, regime="sparse", main="c0ffee" * 6 + "abcd"
):
    ep = _ep(
        episode_id=eid,
        started_at=f"2026-10-08T01:{minute:02d}:00Z",
        ended_at=f"2026-10-08T01:{minute:02d}:30Z",
        regime=regime,
        memory_main=main,
    )
    ep.memory_use = {
        "items": {ITEM: {"imported": 1, "called": 2, "refused": 0, "errored": errored}},
        "outcomes_known": True,
    }
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    return sha


def test_the_failure_record_comes_from_the_evidence_with_its_pins_and_late_contrary_told_apart(
    tmp_path,
):
    stores = open_stores(Paths.under(tmp_path / "home"))
    _record(stores, "e1", 0, errored=1)
    sha2 = _record(stores, "e2", 1)
    sha3 = _record(stores, "e3", 2)
    _record(stores, "e4", 3)
    within = Signal(
        "e2.checker.1",
        "e2",
        "checker",
        "fail",
        "2026-10-08T01:01:10Z",
        refers_to="e2",
        regime="sparse",
        visible_to_actor=True,
    )
    post_signal(within, stores.episodes, sha2, stores.evidence)
    late = Signal(
        "e3.late.e4.1",
        "e3",
        "checker",
        "fail",
        "2026-10-08T01:05:00Z",
        refers_to="e3",
        regime="sparse",
        visible_to_actor=True,
        arrived_at="2026-10-08T01:05:00Z",
        episode_ended_at="2026-10-08T01:02:30Z",
        timing="after_episode",
    )
    post_signal(late, stores.episodes, sha3, stores.evidence)
    found = serving.item_evidence(
        stores, captures={}, home=None, labels=lambda _i: None
    )
    rows = {(r["episode"], r["why"]): r for r in found[ITEM]["failure_record"]}
    assert set(rows) == {
        ("e1", "error"),
        ("e2", "contrary_signal"),
        ("e3", "late_contrary"),
    }
    assert (
        rows[("e1", "error")]["pin"].startswith("c0ffee")
        and rows[("e1", "error")]["ref"] == "episode:e1"
    )
    assert rows[("e2", "contrary_signal")]["ref"] == "signal:e2.checker.1"
    assert rows[("e3", "late_contrary")]["ref"] == "signal:e3.late.e4.1"
    assert found[ITEM]["status_basis"] == trust.BASIS_END_OUTCOMES
    assert (
        found[ITEM]["counts"][trust.FAILED] >= 3 and found[ITEM]["contradictions"] == []
    )
    # the item a hidden grader judged has no failure from it
    assert not [r for r in found[ITEM]["failure_record"] if r["episode"] == "e4"]


def test_an_actors_report_on_the_run_record_is_a_failure(tmp_path, monkeypatch):
    stores = open_stores(Paths.under(tmp_path / "home"))
    _record(stores, "e1", 0)
    fake = types.ModuleType("unify.agents.run_record")
    fake.read_entries = lambda home, since=0: [
        {
            "seq": 7,
            "role": "actor",
            "episode": "e1",
            "refs": [f"item:{ITEM}", "e1/action:3"],
        },
        {
            "seq": 8,
            "role": "writer",
            "episode": None,
            "refs": [f"item:{ITEM}"],
        },  # not an actor's report
        {
            "seq": 9,
            "role": "actor",
            "episode": "e1",
            "refs": ["capture:0123456789abcdef"],
        },
    ]
    monkeypatch.setitem(sys.modules, "unify.agents.run_record", fake)
    if "unify.agents" in sys.modules:
        monkeypatch.setattr(
            sys.modules["unify.agents"], "run_record", fake, raising=False
        )
    found = serving.item_evidence(
        stores, captures={}, home=tmp_path, labels=lambda _i: None
    )
    assert {
        "episode": "e1",
        "pin": "c0ffee" * 6 + "abcd",
        "why": "actor_reported",
        "ref": "entry:7",
    } in (found[ITEM]["failure_record"])
    assert (
        found["capture:0123456789abcdef"]["failure_record"][0]["why"]
        == "actor_reported"
    )
    assert serving.actor_reports(None) == []


def test_contradictions_come_from_the_evidence_label():
    assert serving.contradictions(None) == []
    label = {
        "tests": {"as_stated": 3, "not_as_stated": 2},
        "effects": {"declares": "none", "seen_in_tests": {"read": 1, "write": 2}},
    }
    assert serving.contradictions(label) == [
        "2 of its stated tests did not go as stated",
        "it declares no effects, but its tests saw 1 read(s) and 2 write(s)",
    ]
    assert serving.contradictions(
        {"effects": {"declares": "reads", "seen_in_tests": {"write": 1}}}
    ) == [
        "it declares reads only, but its tests saw 1 write(s)",
    ]


# --- 5. the NW E4 replay: an instance capture served as fitting now reaches the librarian with its failures --------

#: The office LOW capture of NW E4 (a recorded writer capture), cut to
#: its reader, its applies and the hard-coded year that made it fail on the repeats.
E4 = '''"""Investigate the requested alert date and file its report."""

import re


def read_input(request):
    """Return the full request text; this request supplied no JSON data."""
    return request


def applies(value):
    """Return whether the request asks to investigate a checkout alert."""
    text = str(value).lower()
    return "checkout alert" in text and "investigate" in text


def run(value):
    """Investigate the requested alert date and file its report."""
    match = re.search(r"(\\d{1,2})\\s+([A-Za-z]+)\\s+(2026)", str(value))
    if not match:
        raise ValueError("Could not determine alert date from request")
    return " ".join(match.groups())
'''


def test_the_e4_replay_the_librarian_input_now_carries_the_failure_record(
    tmp_path, monkeypatch
):
    stores = open_stores(Paths.under(tmp_path / "home"))
    state = stores.paths.state_dir
    store = captures.store_dir(state)
    cid = _put(
        store,
        E4,
        asked="Investigate the checkout alert of 14 September and file the report.",
    )
    rec_path = store / cid / captures.RECORD_FILE
    rec = json.loads(rec_path.read_text())
    # the first repeat: the call raised; the second: the actor reported on the run record that its answer was wrong
    rec["uses"] = [trust.Use("e15oct", "none", "error", answered=None).row()]
    rec_path.write_text(json.dumps(rec))
    _record(stores, "e15oct", 0)
    _record(stores, "e19oct", 1)
    fake = types.ModuleType("unify.agents.run_record")
    fake.read_entries = lambda home, since=0: [
        {"seq": 41, "role": "actor", "episode": "e19oct", "refs": [f"capture:{cid}"]},
    ]
    monkeypatch.setitem(sys.modules, "unify.agents.run_record", fake)
    if "unify.agents" in sys.modules:
        monkeypatch.setattr(
            sys.modules["unify.agents"], "run_record", fake, raising=False
        )
    caps = captures.load(store)
    found = serving.item_evidence(
        stores,
        captures=caps,
        home=tmp_path,
        labels=lambda _i: None,
    )
    doc = serving.evidence_doc(
        found, serving.conditions(tmp_path, {}, caps), {f"capture:{cid}": "on-trial"}
    )
    row = doc["items"][f"capture:{cid}"]
    assert [r["why"] for r in row["failure_record"]] == ["error", "actor_reported"]
    assert (
        row["conditions"]
        == "Return whether the request asks to investigate a checkout alert."
    )
    # what the librarian was told on the repeats was "the stored reader parsed this request and its applies check
    # held"; rev 2 runs no reader, so the message tells it nothing of the kind, and the failures travel on the card
    message = librarian.sent_to_librarian(
        "Investigate the checkout alert of 19 October and file the report.",
        {"commit": "c", "dispatch": "d"},
        evidence=doc["items"],
    )["message"]
    assert f"capture:{cid}" not in message and "reader" not in message and "applies" not in message
    assert "No message has come after the agent's request." in message


# --- the switch and the request's wiring ----------------------------------------------------------------------------


def test_the_request_serves_evidence_only_under_its_switch(tmp_path, monkeypatch):
    # the copy is written by _export_copy (at the request's open, and at a re-pin under UNIFY_MEMORY_IN_EPISODE)
    opening = inspect.getsource(RequestRun._export_copy)
    assert "if serving_failure_record(SETTINGS)" in opening
    stores = open_stores(Paths.under(tmp_path / "home"))
    _record(stores, "e1", 0, errored=1)
    run = RequestRun("x", stores.paths)
    run.stores = stores
    from unify.settings import SETTINGS

    records = {ITEM: dict(empty_record(ITEM, "function"), status="retired")}
    computed = run._serving(
        records, lambda i: records.get(i, {}).get("status") or "on-trial"
    )(tmp_path)
    assert computed["evidence"]["items"][ITEM]["failure_record"][0]["why"] == "error"
    assert computed["evidence"]["items"][ITEM]["status_text"].startswith("retired: ")
    assert computed["withdrawn_items"] == {
        ITEM: {"status": "retired", "by": None, "reason": "retired"}
    }
    monkeypatch.setattr(
        SETTINGS, "UNIFY_MEMORY_LIBRARY_WITHDRAW_CLOSED_CAPTURES", "off"
    )
    assert (
        run._serving(records, lambda i: "on-trial")(tmp_path)["withdrawn_items"] == {}
    )
