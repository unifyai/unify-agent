"""The librarian's evidence is a structural catalogue, and nothing runs to recognise or rank.

``.memory/dispatch.json`` holds one row per stored procedure from its syntax and records; ``memory.ask`` sends no
reader facts and an item's check (``item.fits``) compares structure only, so no process starts on either. Identifiers never reach
the librarian: not in a row's ``applies_text``, not in a searched card, not in the question it is sent. No model,
network or key; no world vocabulary.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from unify.memory import librarian as lib_mod
from unify.memory import library_export as le
from unify.memory.layout import Function, Library

#: A synthetic identifier of the request's form (not a word of any world).
KEY = "order-7f3a91c2"

PROCS = {
    "a" * 16: ("value", '''def read_input(request):
    return request.data[0]


def applies(value):
    return "rows" in value


def run(value):
    """Sum each row."""
    return [sum(r) for r in value["rows"]]
'''),
    "b" * 16: ("state", '''def run(state):
    """The latest date."""
    return state.data[-1]["date"]
'''),
    "c" * 16: ("state_env", '''def run(state, env):
    """Write the report, then list the folder."""
    env.files.write("out/report.txt", "done")
    return env.files.list("out")
'''),
    "d" * 16: ("state", '''def run(state):
    """The answer."""
    return [6, 9]
'''),
    "e" * 16: ("value", f'''def read_input(request):
    return request.text


def applies(value):
    return "{KEY}" in value


def run(value):
    """Reply to the request."""
    return "ok"
'''),
}


def _captures() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for cid, (form, src) in PROCS.items():
        rec = {"v": 2, "entry_point": {"form": form}, "outcome": "none", "occurrences": 2,
               "episodes": ["e1", "e2"],
               "cases": [{"episode": "e1", "outcome": "positive"},
                         {"episode": "e2", "outcome": "none"}]}  # fmt: skip
        out[f".memory/captures/{cid}/record.json"] = json.dumps(rec).encode()
        out[f".memory/captures/{cid}/procedure.py"] = src.encode()
        out[f".memory/captures/{cid}/signature.json"] = json.dumps({"input": None}).encode()
    return out


def _rows():
    data = le.dispatch_data(Library(), captures=_captures())
    assert data["version"] == 2
    return {r["id"]: r for r in data["procedures"]}


def test_every_capture_form_gets_a_structural_row():
    rows = _rows()
    assert {r["form"] for r in rows.values()} == {"value", "state", "state_env"}
    for r in rows.values():
        assert set(r) == {"id", "kind", "form", "signature", "applies_text", "calls", "occurrences", "outcomes",
                          "provenance"}  # fmt: skip
        assert r["kind"] == "capture-shape" and r["occurrences"] == 2
        assert r["outcomes"] == {"positive": 1, "contrary": 0, "none": 1}
        assert r["provenance"] == {"episodes": 2}  # counts, never an id
    env = rows["capture:" + "c" * 16]
    assert env["calls"] == ["env.files.list", "env.files.write"]
    assert rows["capture:" + "a" * 16]["applies_text"].startswith("def applies(value):")


def test_a_capture_that_returns_a_constant_gets_the_same_row_as_any_other():
    """Nothing judges whether code gives a fixed answer or reads its input: no row says so."""
    const, other = _rows()["capture:" + "d" * 16], _rows()["capture:" + "b" * 16]
    assert set(const) == set(other) and "constant" not in const and "reads_input" not in const


def test_an_identifier_keyed_applies_keeps_its_identifier():
    """The analysis found a capture whose applies tested a task's identifier in the request; with the identifier
    replaced by a placeholder on both sides the librarian served it for another request. Now the catalogue row
    shows the applies as written, so the identifier it keys on is visible and is never a wildcard."""
    row = _rows()["capture:" + "e" * 16]
    assert KEY in row["applies_text"] and "<id>" not in json.dumps(row)
    assert "<id>" not in json.dumps(le.dispatch_data(Library(), captures=_captures()))


def test_the_question_the_librarian_is_sent_keeps_its_identifiers():
    """The asker's question (and its situation) reach the librarian as written, so it sees the same
    identifier on both sides and can tell an item keyed on another request's identifier from a fit."""
    for evidence in (None, {}):
        sent = lib_mod.sent_to_librarian(f"Please handle {KEY} as before", {"commit": "c", "dispatch": "d"},
                                         evidence=evidence, last_message=f"About {KEY}: done?", rows=5)  # fmt: skip
        assert f"Please handle {KEY} as before" in sent["message"] and "<id>" not in sent["message"]
        if evidence is not None:
            assert f"About {KEY}: done?" in sent["message"]
        assert "dispatch.json, 5 row(s)" in sent["message"] and "reader" not in sent["message"]


def test_searched_cards_keep_their_identifiers_as_written(tmp_path):
    """A card shows what its procedure says, identifiers included, never a placeholder."""
    from unify.memory import library_search

    caps = _captures()
    cid = "e" * 16
    caps[f".memory/captures/{cid}/procedure.py"] = f'"""Reply to {KEY}."""\n\n\ndef run(state):\n    return 1\n'.encode()
    rows = library_search.rows(tmp_path, Library(), captures=caps)
    assert rows and KEY in json.dumps(rows) and "<id>" not in json.dumps(rows)


def test_a_function_row_reads_its_code_and_its_named_applies():
    fn = Function("memory.t.m:total", "memory.t.m", "total", "memory/t/m.py", 1, "total(state, env)", "Total.", [])
    check = Function("memory.t.m:fits", "memory.t.m", "fits", "memory/t/m.py", 9, "fits(value)", "Fits.", [])
    bodies = {
        fn.item_id: "def total(state, env):\n    env.store.put(state.last)\n    return 1\n",
        check.item_id: f"def fits(value):\n    return '{KEY}' in value\n",
    }
    records = {fn.item_id: {"applies": check.item_id}, check.item_id: {}}
    data = le.dispatch_data(Library(functions=[fn, check]), records=records, bodies=bodies)
    row = {r["id"]: r for r in data["procedures"]}[fn.item_id]
    assert row["kind"] == "function" and row["form"] == "function" and row["calls"] == ["env.store.put"]
    # the named applies as written, the identifier it tests included (never masked)
    assert row["applies_text"] == bodies[check.item_id]


# --- nothing runs on ask or a check -------------------------------------------------------------------------------------


@pytest.fixture
def no_process(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "fork", refuse)
    return refuse


def test_the_recogniser_is_gone_and_no_reader_loads():
    from unify.memory import library_helper as lh

    for gone in ("_RECOGNISER_RUNNER", "_recognise", "_facts", "_match_recognised", "_dispatch"):
        assert not hasattr(lh, gone), gone
    src = (le._HERE / "library_helper.py").read_text() + le.ASK_SOURCE.read_text()
    # no loader of a stored reader remains (an earlier LookupError path): read_input is read only by a tracked
    # call's child; an earlier _spec(target, name) loaded a stored reader; importlib's module_from_spec and find_spec are not it
    assert not re.search(r"(?<![\w.])_spec\(", src) and "_RECOGNISER_RUNNER" not in src
    assert src.count('ns.get("read_input")') == 1


def test_a_check_compares_structure_only_and_starts_no_process(monkeypatch, no_process):
    from unify.memory import library_helper as lh
    from unify.memory import memory_helper

    data = {"rows": [[1, 2]]}
    recorded = lh.Candidate(
        id="capture:" + "a" * 16, source="capture", signature={"input": memory_helper.value_shape(data)},
        docstring="Total these rows.", asked="Total these: rows", approach="", level="unchecked",
        status="on-trial", failed=False, broke=(),
    )  # fmt: skip
    monkeypatch.setattr(lh, "_candidates_all", lambda: ([recorded], 0))
    monkeypatch.setattr(lh, "_matcher", lambda: memory_helper)  # the shipped matcher is this module's code
    call = f"memory.get({recorded.id!r}).fits(...)"
    fit = lh._fit_check(recorded.id, data, call)
    assert fit.ok is True and "(exact)" in fit.why
    # the words it was saved against have no structure: no fit, and nothing ran to tell
    words = lh._fit_check(recorded.id, "Total these: rows", call)
    assert words.ok is False and "no structure to compare" in words.why
