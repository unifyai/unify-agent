"""A captured procedure whose code looks like a stored answer (its ``run`` never reads its input, or returns only
literals) is served exactly as every other captured procedure: nothing judges whether code gives a fixed answer,
reads its input or acts, since none of that can be told reliably. Its record carries no mark, the dispatch file no
``constant`` or ``reads_input`` field, the catalogue and the search index name it a recorded procedure, and the
package calls it like any other. The package tests run the generated package in a fresh isolated interpreter, as a
cell imports it. No model is called.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.memory.test_layout import _tree
from unify.memory import captures, library_search, memory_helper
from unify.memory import library_export as lx
from unify.memory.catalogue import write_files
from unify.memory.integration.request import RequestRun
from unify.memory.layout import discover

ROWS = [{"item": "pen", "amount": 3}, {"item": "ink", "amount": 4}]
GRID = [[1, 2, 3], [4, 5, 6]]
TABLE = '''"""Sum the amount column of an order table."""

import json


def read_input(text):
    return json.loads(text.split("TABLE\\n", 1)[1].splitlines()[0])


def run(value):
    """Sum the amount column of an order table of items."""
    return sum(r["amount"] for r in value)
'''
TRANSPOSE = '''"""Turn a grid's rows into its columns."""

import json


def read_input(text):
    return json.loads(text.split("GRID\\n", 1)[1].splitlines()[0])


def run(value):
    """Turn a grid's rows into its columns."""
    return [list(c) for c in zip(*value)]
'''
FIXED = '''"""The answer of one attempt."""


def read_input(text):
    return text


def run(value):
    """Hand back the grid."""
    return [[1, 4], [2, 5], [3, 6]]
'''
RETIRED = '''"""Fold every hand."""


def run(value):
    if value:
        return "fold"
    return "fold"
'''


def _signature(value, result) -> dict:
    return {
        "input": memory_helper.value_shape(value),
        "output": memory_helper.value_shape(result) if isinstance(result, (list, dict)) else {"kind": "scalar"},
        "relations": [],
        "constraints": {},
        "goal": "Asked once.",
    }


def _put(store: Path, source: str, *, outcome: str = "none", signature: dict | None = None, asked: str = "") -> str:
    """One capture in *store* as the capture step leaves it (the files :func:`captures.load` reads)."""
    cid = captures.capture_id(source)
    d = store / cid
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "episodes": ["ep-1"],
        "cases": [{"episode": "ep-1", "reproduces": "reproduces", "outcome": outcome}],
        "outcome": outcome,
        "level": "unchecked",
        "status": "on-trial",
        "track": {},
    }
    (d / captures.PROCEDURE_FILE).write_text(source)
    (d / captures.RECORD_FILE).write_text(json.dumps(rec))
    if signature is not None:
        (d / captures.SIGNATURE_FILE).write_text(json.dumps(signature))
    (d / captures.SAVED_FILE).write_text(json.dumps({"docstring": "", "asked": asked, "approach": ""}))
    return cid


def _store(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    store = captures.store_dir(tmp_path / "state")
    ids = {
        "table": _put(store, TABLE, signature=_signature(ROWS, 7), asked="Add up an order table."),
        "grid": _put(store, TRANSPOSE, signature=_signature(GRID, [[1, 4]]), asked="Turn grid rows into columns."),
        "fixed": _put(store, FIXED, signature=_signature("x", [[1, 4]]), asked="Answer one grid."),
        "retired": _put(store, RETIRED, signature=_signature("x", "fold"), asked="Play one hand."),
    }
    return store, ids


# --- no judgement of the code ------------------------------------------------------------------------------------------


def test_no_fixed_answer_classifier_exists():
    from unify.memory import fixed_answer

    for name in ("fixed_answer", "reads_input", "NEVER_READS", "ONLY_LITERALS"):
        assert not hasattr(fixed_answer, name), name
    assert not hasattr(captures, "EXAMPLE_KEY")
    assert "examples" not in inspect.signature(captures.export_files).parameters


def test_the_dispatch_file_names_the_calls_a_procedure_makes_and_judges_nothing():
    from unify.memory.library_export import _code_facts

    acting = "def run(state, env):\n    env.mail.send(state['to'], 'hi')\n    return None\n"
    assert _code_facts(acting, None) == {"calls": ["env.mail.send"], "applies_text": ""}
    assert _code_facts(FIXED, None) == {"calls": [], "applies_text": ""}


# --- the actor's snapshot ---------------------------------------------------------------------------------------------


def test_every_capture_is_exported_alike_whatever_its_code_looks_like(tmp_path):
    store, ids = _store(tmp_path)
    snap = captures.export_files(store)
    assert {p.split("/")[2] for p in snap} == set(ids.values())
    for cid in ids.values():
        rec = json.loads(snap[f"{captures.EXPORT_DIR}/{cid}/{captures.RECORD_FILE}"])
        assert "worked_example" not in rec and "constant" not in rec
        assert f"{captures.EXPORT_DIR}/{cid}/{captures.PROCEDURE_FILE}" in snap


# --- the librarian's files ---------------------------------------------------------------------------------------------


def _snapshot(tmp_path):
    """The fixture store's snapshot beside a fixture library tree."""
    store, ids = _store(tmp_path)
    return _tree(tmp_path / "lib"), captures.export_files(store), ids


def test_the_catalogue_and_the_search_index_name_every_capture_a_recorded_procedure(tmp_path):
    tree, snap, ids = _snapshot(tmp_path)
    rows = {r["id"]: r for r in lx.dispatch_data(discover(tree), captures=snap)["procedures"]}
    for key in ("table", "grid", "fixed", "retired"):
        row = rows[f"capture:{ids[key]}"]
        assert "constant" not in row and "reads_input" not in row and "effects" not in row and "calls" in row
    text = lx.catalogue_text(tree, captures=snap)
    assert "worked example" not in text and "fixed answer" not in text
    card = text.split(f"### capture:{ids['fixed']}\n", 1)[1].split("###", 1)[0]
    assert "Evidence: recorded procedure" in card
    rows = {r["id"]: r for r in library_search.rows(tree, captures=snap)}
    assert {rows[f"capture:{ids[k]}"]["kind"] for k in ids} == {"captured procedure"}


# --- the package, as a cell runs it -----------------------------------------------------------------------------------

PRELUDE = '''
import json, sys
import memory


def err(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as exc:
        return [type(exc).__name__, str(exc)]
    return None
'''


def _copy(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A library copy as the request's export writes it, with the capture snapshot."""
    tree, snap, ids = _snapshot(tmp_path)
    write_files(tree, lx.generated_library_mode(tree))
    for rel, data in snap.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    return tree, ids


def _cell(root: Path, code: str) -> object:
    out = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys\nsys.path.insert(0, {str(root)!r})\n{PRELUDE}\n{code}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_a_capture_that_looks_like_a_stored_answer_is_called_like_any_other(tmp_path):
    root, ids = _copy(tmp_path)
    fixed = f"capture:{ids['fixed']}"
    got = _cell(
        root,
        f"ex = memory.get({fixed!r})\n"
        f"print(json.dumps({{'kind': ex.kind, 'card': str(ex), 'call': ex('x'), "
        f"'notes': [i.id for i in memory.items(kind='note')]}}))\n",
    )
    assert got["kind"] != "note" and fixed not in got["notes"]
    assert got["call"] == [[1, 4], [2, 5], [3, 6]]
    assert "worked example" not in got["card"] and "fixed answer" not in got["card"]


# --- the request ------------------------------------------------------------------------------------------------------


def test_the_request_exports_every_capture_alike():
    opening = inspect.getsource(RequestRun._export_copy)
    assert "export_files(store_dir(paths.state_dir), withdrawn=withdrawn)" in opening
    assert "worked_examples" not in opening and "EXAMPLE_KEY" not in opening
