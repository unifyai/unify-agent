"""Normalised code units: renaming, literal lifting, and names kept from each unit's own text."""

from unify.memory import mining as m
from unify.memory.mining import Cell

PARSE_A = (
    "rows = [list(map(int, line.split())) for line in grid.splitlines() if line.strip()]\n"
    "h, w = len(rows), len(rows[0])\n"
    "print(h, w)\n"
)
# the same code with other names and literals
PARSE_B = (
    "g = [list(map(int, ln.split())) for ln in text.splitlines() if ln.strip()]\n"
    "a, b = len(g), len(g[0])\n"
    "print(a, b)\n"
)
OTHER = "import json\nd = json.loads(raw)\nprint(sorted(d.keys())[:5], len(d), type(d).__name__)\n"


def test_failed_cells_and_tracebacks_are_not_successful():
    assert m.successful(Cell(0, PARSE_A, "out"))
    assert not m.successful(Cell(0, PARSE_A, "out", "boom"))
    assert not m.successful(
        Cell(0, PARSE_A, "Traceback (most recent call last):\n  ..."),
    )
    assert not m.successful(Cell(0, "ls", "", None, "bash"))


def test_code_that_does_not_parse_has_no_units():
    assert m.units(Cell(0, "def (:")) == []


def test_units_are_the_whole_cell_and_each_large_statement():
    us = m.units(Cell(3, PARSE_A + OTHER))
    assert us[0].kind == "cell" and us[0].lines == (1, 6) and us[0].cell == 3
    assert {u.kind for u in us[1:]} == {"stmt"} and all(
        u.size >= m.MIN_NODES for u in us
    )
    # renamed code gives the same keys, unit for unit
    assert [u.key for u in m.units(Cell(0, PARSE_B))] == [
        u.key for u in m.units(Cell(0, PARSE_A))
    ]


def test_a_stop_ends_the_walk_with_the_units_found_so_far():
    whole = m.units(Cell(0, PARSE_A + OTHER))
    cut = m.units(Cell(0, PARSE_A + OTHER), stop=lambda: True)
    assert [u.kind for u in cut] == ["cell"] and cut[0].key == whole[0].key
