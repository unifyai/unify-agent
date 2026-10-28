"""Argument shapes as the library export records them, frozen per commit.

A row an earlier gate recorded from a function's example calls names the parameter its argument was passed as; rows
recorded from a function's covers have no parameter. Both are kept in the export. (The gate now runs no test, so it
records no argument shapes.)
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.memory.test_layout import _tree
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import body_digest
from unify.memory.evidence import EvidenceStore
from unify.memory.layout import function_bodies
from unify.memory.library_helper import FIND_FILE
from unify.memory.shape_rows import lookup_from, snapshot_rows

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


def _rows(root: Path, new: dict, prev: dict | None = None) -> dict:
    bodies = function_bodies(root)
    return snapshot_rows(
        root,
        prev or {},
        {item: (body_digest(bodies[item]), descs) for item, descs in new.items()},
    )


#: the row an earlier gate recorded for ``flood_fill(grid, colour)`` called on GRID and 3 (an int has no shape)
GRID_ROW = [{**memory_helper.value_shape(GRID), "param": "grid"}]


def test_snapshot_rows_are_frozen_with_param_and_legacy_rows_are_kept(
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
    descs = GRID_ROW
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
