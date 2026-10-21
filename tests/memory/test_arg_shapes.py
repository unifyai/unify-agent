"""The shapes of the arguments a function's tests call it on: what ``memory.find`` matches, frozen per commit.

Each descriptor names the parameter its argument was passed as, so ``find`` says which parameter a value fits.
Rows recorded before arguments were shaped have no parameter; they are kept and still found.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.memory.test_layout import _tree
from tests.memory.test_library_helper import _py
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import body_digest, write_files
from unify.memory.evidence import EvidenceStore
from unify.memory.example_calls import ArgRecord, CallKey, ExampleCall
from unify.memory.layout import function_bodies
from unify.memory.library_helper import FIND_FILE
from unify.memory.shape_rows import arg_descriptors, lookup_from, snapshot_rows

FILL_ID = "memory.grid.fill:flood_fill"
TOTAL_ID = "memory.grid.ledger:total"
TEST_FILE = "memory/grid/tests/test_fill.py"
LEDGER_PATH = "memory/grid/tests/data/ledger.csv"
LEDGER = "vendor,amount\nV-1,3\nV-2,4\n"
GRID = [[0, 1], [1, 1]]
FILES = {
    "memory/grid/__init__.py": '"""Grid helpers."""\n',
    "memory/grid/fill.py": (
        '"""Fills."""\n\n\ndef flood_fill(grid, colour):\n    """Fill a region."""\n'
        "    return [[colour for _ in row] for row in grid]\n"
    ),
    "memory/grid/ledger.py": (
        '"""Ledgers."""\n\nimport csv\n\n\ndef total(path):\n    """Sum a ledger\'s amounts."""\n'
        '    with open(path, newline="") as fh:\n'
        '        return sum(int(r["amount"]) for r in csv.DictReader(fh))\n'
    ),
    TEST_FILE: (
        "from memory.grid.fill import flood_fill\n\n\n"
        "def test_fills():\n    assert flood_fill([[0, 1], [1, 1]], 3) == [[3, 3], [3, 3]]\n"
    ),
    LEDGER_PATH: LEDGER,
}


def _arg(name: str, value=None, *, path: str | None = None, cut: bool = False):
    return ArgRecord(
        name=name,
        sha="0" * 64,
        type=type(value).__name__,
        json=None if cut else value,
        cut=cut,
        path=path,
    )


def _call(item: str, *args: ArgRecord, outcome: str = "returned", ordinal: int = 0):
    return ExampleCall(
        item=item,
        key=CallKey(f"{TEST_FILE}::test_fills", (), ordinal),
        node=f"{TEST_FILE}::test_fills",
        file=TEST_FILE,
        line=5,
        args=tuple(args),
        outcome=outcome,
        raised="KeyError" if outcome == "raised" else None,
        result_sha=None,
        result_type=None,
        result_json=None,
        opened=(),
    )


def _rows(root: Path, new: dict, prev: dict | None = None) -> dict:
    bodies = function_bodies(root)
    return snapshot_rows(
        root,
        prev or {},
        {item: (body_digest(bodies[item]), descs) for item, descs in new.items()},
    )


def _find(root: Path, rows: dict, value) -> list[list]:
    write_files(root, lx.generated_library_mode(root, shapes=lookup_from(rows)))
    return json.loads(
        _py(
            root,
            f"import json, memory\nprint(json.dumps([list(f) for f in memory.find({value!r})]))",
        ),
    )


def test_a_grid_argument_is_found_by_a_grid_and_names_its_parameter(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    calls = [_call(FILL_ID, _arg("grid", GRID), _arg("colour", 3))]
    notes: list[str] = []
    descs = arg_descriptors(calls, root, note=notes.append)
    # the int has no shape and adds no descriptor
    assert descs == [{**memory_helper.value_shape(GRID), "param": "grid"}]
    assert notes == []
    found = _find(root, _rows(root, {FILL_ID: descs}), [[5, 5, 5], [5, 0, 5]])
    assert [(f[0], f[2], f[3], f[5]) for f in found] == [
        (FILL_ID, "grid", "weak", "matched the whole shape (it has no named keys)"),
    ]
    assert _find(root, _rows(root, {FILL_ID: descs}), [{"a": 1}]) == []


def test_only_returned_calls_are_shaped_and_drops_are_counted_in_a_note(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    key_shaped = {"AKIA" + "Q" * 16: 1, "rows": [1]}
    calls = [
        _call(FILL_ID, _arg("grid", [[1]]), outcome="refused", ordinal=1),
        _call(FILL_ID, _arg("grid", {"x": [1]}), outcome="raised", ordinal=2),
        _call(FILL_ID, _arg("grid", None, cut=True), ordinal=3),
        _call(FILL_ID, _arg("grid", key_shaped), ordinal=4),
        _call(FILL_ID, _arg("colour", 3), ordinal=5),
    ]
    notes: list[str] = []
    assert arg_descriptors(calls, root, note=notes.append) == []
    assert notes == [
        f"G2 shapes: 2 argument shapes of {FILL_ID} dropped (key-shaped or over the size bound)",
    ]
    # the same call twice gives one descriptor
    twice = [_call(FILL_ID, _arg("grid", GRID), ordinal=i) for i in range(2)]
    assert len(arg_descriptors(twice, root)) == 1


def test_a_path_argument_is_shaped_from_the_fixture_bytes(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    calls = [_call(TOTAL_ID, _arg("path", LEDGER_PATH, path=LEDGER_PATH))]
    descs = arg_descriptors(calls, root)
    assert descs == [
        {
            **memory_helper.file_shape(LEDGER_PATH, LEDGER.encode("utf-8")),
            "param": "path",
        },
    ]
    rows = _rows(root, {TOTAL_ID: descs})
    found = _find(root, rows, "vendor,amount\nV-9,1\n")
    assert [(f[0], f[2]) for f in found] == [(TOTAL_ID, "path")]
    other = tmp_path / "other.csv"
    other.write_text("vendor,amount\nV-3,8\n")
    assert [(f[0], f[2]) for f in _find(root, rows, str(other))] == [
        (TOTAL_ID, "path"),
    ]


def test_a_path_argument_outside_the_library_gives_no_shape_and_a_note(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    outside = tmp_path / "outside.csv"
    outside.write_text(LEDGER)
    link = "memory/grid/tests/data/link.csv"
    (root / link).symlink_to(outside)
    calls = [
        _call(TOTAL_ID, _arg("path", "../outside.csv", path="../outside.csv")),
        _call(TOTAL_ID, _arg("path", link, path=link), ordinal=1),
        _call(TOTAL_ID, _arg("path", "memory/grid", path="memory/grid"), ordinal=2),
    ]
    notes: list[str] = []
    assert arg_descriptors(calls, root, note=notes.append) == []
    assert notes == [
        f"G2 shapes: 3 path arguments of {TOTAL_ID} name no regular file of the library "
        f"(../outside.csv, memory/grid, {link}); no shape is taken from them",
    ]


def test_snapshot_rows_are_frozen_with_param_and_legacy_rows_are_still_found(
    tmp_path,
):
    root = _tree(tmp_path / "lib", FILES)
    bodies = function_bodies(root)
    # a row recorded from a function's covers, before arguments were shaped: no parameter
    legacy = memory_helper.value_shape([{"vendor": "V-1", "amount": 3}])
    prev = {
        TOTAL_ID: {
            "body": body_digest(bodies[TOTAL_ID]),
            "shapes": [legacy],
            "backfilled": True,
        },
    }
    descs = arg_descriptors([_call(FILL_ID, _arg("grid", GRID))], root)
    rows = _rows(root, {FILL_ID: descs}, prev)
    assert rows[FILL_ID]["shapes"] == descs and descs[0]["param"] == "grid"
    assert rows[TOTAL_ID]["shapes"] == [legacy] and "param" not in legacy
    ev = EvidenceStore(tmp_path / "evidence.sqlite")
    assert ev.write_commit_shapes("c1", rows)
    assert not ev.write_commit_shapes("c1", {})  # frozen: written once
    frozen = ev.commit_shapes("c1")
    assert frozen == rows
    first = lx.generated_library_mode(root, shapes=lookup_from(frozen))
    assert lx.generated_library_mode(root, shapes=lookup_from(frozen)) == first
    entries = {
        e["id"]: e for e in json.loads(first[FIND_FILE].decode("utf-8"))["functions"]
    }
    assert all("input" not in e for e in entries.values())
    assert entries[FILL_ID]["input_shapes"][0]["param"] == "grid"
    assert entries[TOTAL_ID]["input_shapes"] == [legacy]
    found = _find(root, frozen, [{"vendor": "V-7", "amount": 9}])
    assert [(f[0], f[2], f[3]) for f in found] == [(TOTAL_ID, "", "exact")]
