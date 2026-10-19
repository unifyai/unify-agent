"""Captures that give a fixed answer, offered as worked examples (``UNIFY_MEMORY_LIBRARY_FIXED_ANSWERS_AS_EXAMPLES``).

Off (the default) the actor's snapshot is exactly as built. On, a captured procedure whose ``run`` never reads its input
or only returns literals (a rule on its code's syntax alone: no task, position or word) is offered as a worked example
to read: its record carries the mark, the dispatch file gives it no reader, the catalogue and the search index say what
it is, and the package lists it as a note-kind item whose text is the example, refusing to call it or check its fit. A
failed attempt is never offered anyway. The package tests run the generated package in a fresh isolated interpreter, as
a cell imports it. No model is called.
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
from unify.memory.fixed_answer import NEVER_READS, ONLY_LITERALS, fixed_answer
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


# --- the fixed-answer rule ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source, why, returns",
    [
        (FIXED, NEVER_READS, ["[[1, 4], [2, 5], [3, 6]]"]),
        (RETIRED, ONLY_LITERALS, ["'fold'"]),
        ("import json\ndef run(value):\n    return [a * 2 for a in json.loads('[1]')]\n", NEVER_READS, []),
        ("def run():\n    return compute()\n", NEVER_READS, []),
        ("def main(v):\n    return {'a': -1}\n", NEVER_READS, ["{'a': -1}"]),
        ('def run(v):\n    """Doc."""\n    return f"fixed"\n', NEVER_READS, ["f'fixed'"]),
    ],
)
def test_a_run_that_never_reads_its_input_or_returns_only_literals_gives_a_fixed_answer(source, why, returns):
    assert fixed_answer(source) == {"why": why, "returns": returns}


@pytest.mark.parametrize(
    "source",
    [
        TABLE,
        TRANSPOSE,
        "def run(value):\n    if value:\n        return 1\n    return len(value)\n",
        "def run(v):\n    def inner():\n        return v\n    return inner()\n",
        "def run(*, v):\n    return v\n",
        "def solve(grid):\n    return [row[::-1] for row in grid]\n",
        "def run(v:\n",  # does not parse: not classified
        "x = 1\n",  # no function
        None,
    ],
)
def test_a_procedure_that_reads_its_input_is_no_fixed_answer(source):
    assert fixed_answer(source) is None


# --- the actor's snapshot ---------------------------------------------------------------------------------------------


def test_off_the_snapshot_is_exactly_as_built(tmp_path):
    store, ids = _store(tmp_path)
    snap = captures.export_files(store)
    assert snap == captures.export_files(store, examples=False)
    assert all(captures.EXAMPLE_KEY not in json.loads(v) for p, v in snap.items() if p.endswith("/record.json"))
    assert {p.split("/")[2] for p in snap} == set(ids.values())


def test_a_fixed_answer_is_marked_a_worked_example_and_a_failed_one_is_not(tmp_path):
    store, ids = _store(tmp_path)
    failed = _put(store, "def run(v):\n    return 'no action'\n", outcome="failed", signature=_signature("x", "y"))
    snap = captures.export_files(store, examples=True)

    def rec(cid):
        return json.loads(snap[f"{captures.EXPORT_DIR}/{cid}/{captures.RECORD_FILE}"])

    assert rec(ids["fixed"])[captures.EXAMPLE_KEY] == {"why": NEVER_READS, "returns": ["[[1, 4], [2, 5], [3, 6]]"]}
    assert rec(ids["retired"])[captures.EXAMPLE_KEY]["why"] == ONLY_LITERALS
    assert captures.EXAMPLE_KEY not in rec(ids["table"]) and captures.EXAMPLE_KEY not in rec(ids["grid"])
    assert captures.EXAMPLE_KEY not in rec(failed)  # a failed attempt is never offered anyway
    assert f"{captures.EXPORT_DIR}/{ids['fixed']}/{captures.PROCEDURE_FILE}" in snap  # its source stays to read


# --- the librarian's files ---------------------------------------------------------------------------------------------


def _snapshot(tmp_path, **export):
    """The fixture store's snapshot, exported with *export*, beside a fixture library tree."""
    store, ids = _store(tmp_path)
    return _tree(tmp_path / "lib"), captures.export_files(store, **export), ids


def test_the_dispatch_file_gives_a_worked_example_no_reader(tmp_path):
    tree, snap, ids = _snapshot(tmp_path, examples=True)
    rows = lx.dispatch_data(discover(tree), captures=snap)["procedures"]
    assert sorted(r["id"] for r in rows if r["id"].startswith("capture:")) == sorted(
        f"capture:{ids[k]}" for k in ("table", "grid")
    )


def test_the_catalogue_and_the_search_index_say_a_worked_example_is_one(tmp_path):
    tree, snap, ids = _snapshot(tmp_path, examples=True)
    text = lx.catalogue_text(tree, captures=snap)
    card = text.split(f"### capture:{ids['fixed']}\n", 1)[1].split("###", 1)[0]
    assert "reader none (a worked example)" in card and "a worked example, never called: " + NEVER_READS in card
    assert "applies: no" in card
    plain = text.split(f"### capture:{ids['table']}\n", 1)[1].split("###", 1)[0]
    assert "reader read_input" in plain and "worked example" not in plain
    rows = {r["id"]: r for r in library_search.rows(tree, captures=snap)}
    assert rows[f"capture:{ids['fixed']}"]["kind"] == "worked example"
    assert rows[f"capture:{ids['table']}"]["kind"] == "captured procedure"


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


def _copy(tmp_path: Path, **export) -> tuple[Path, dict[str, str]]:
    """A library copy as the request's export writes it, with the capture snapshot exported with *export*."""
    tree, snap, ids = _snapshot(tmp_path, **export)
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


def test_a_worked_example_is_read_never_called_or_matched(tmp_path):
    root, ids = _copy(tmp_path, examples=True)
    fixed = f"capture:{ids['fixed']}"
    got = _cell(
        root,
        f"ex = memory.get({fixed!r})\n"
        f"print(json.dumps({{'kind': ex.kind, 'card': str(ex), 'call': err(ex, 'x'), 'fits': err(ex.fits, [[1]]), "
        f"'run': memory.run({fixed!r}, 'x').detail, 'notes': [i.id for i in memory.items(kind='note')], "
        f"'functions': [i.id for i in memory.items(kind='function')]}}))\n",
    )
    assert got["kind"] == "note"
    assert "A worked example, not a procedure: " + NEVER_READS in got["card"]
    assert "It handed back: [[1, 4], [2, 5], [3, 6]]" in got["card"] and "def run(value):" in got["card"]
    assert got["call"][0] == "WrongCall" and "a worked example cannot be called" in got["call"][1]
    assert got["fits"][0] == "WrongCall" and "a worked example has no inputs to check" in got["fits"][1]
    assert "worked example" in got["run"]
    assert fixed in got["notes"] and fixed not in got["functions"]


# --- the request ------------------------------------------------------------------------------------------------------


def test_the_request_marks_worked_examples_only_under_its_switch_and_counts_them():
    opening = inspect.getsource(RequestRun._open_library_mode)
    assert "examples=library_fixed_answers_as_examples(SETTINGS)" in opening
    assert '"worked_examples": examples' in opening
    assert '{"type": "capture_serving", "episode_id": self.episode_id, **self.capture_serving}' in inspect.getsource(
        RequestRun._open,
    )
