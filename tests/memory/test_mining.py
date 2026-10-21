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


def test_renaming_and_literal_lifting_make_the_same_shape_hash_equal():
    ta, tb = m._parse(PARSE_A), m._parse(PARSE_B)
    ka = m.normalise(ta, m.keep_names([ta]))
    kb = m.normalise(tb, m.keep_names([tb]))
    assert ka[0] == kb[0] and ka[2] >= m.MIN_NODES
    assert "v0" in ka[1] and "rows" not in ka[1] and "for line" not in ka[1]
    # free names (builtins, called functions, attributes) are kept: they carry the meaning
    assert "map(int" in ka[1] and ".splitlines()" in ka[1]


def test_literals_become_parameters_with_their_values_kept():
    tree = m._parse(
        "x = [[1, 2], [3, 4]]\ny = x[0][1] + 7 * len(x)\nprint(y, 'done')\n",
    )
    key, src, size, params = m.normalise(tree, m.keep_names([tree]))
    assert params == [[[1, 2], [3, 4]], 0, 1, 7, "done"]
    assert "p0" in src and "7" not in src and "done" not in src


def test_a_different_call_is_a_different_shape():
    a = m._parse("v = sorted(xs, key=len)\nprint(v[0], v[-1], len(v))\n")
    b = m._parse("v = reversed(xs, key=len)\nprint(v[0], v[-1], len(v))\n")
    assert m.normalise(a, m.keep_names([a]))[0] != m.normalise(b, m.keep_names([b]))[0]


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


def test_the_same_code_hashes_the_same_whatever_other_cells_import():
    """The kept names come from the unit's own text: an import elsewhere in the episode changes nothing."""
    code = "path = os.path.join(base, name)\nrows = json.loads(open(path).read())\nprint(len(rows), rows[:3])\n"
    t = m._parse(code)
    # an episode-wide import set (`from os import path` in one episode, `import os` in another) splits the hash
    one = m.normalise(t, m.keep_names([m._parse("from os import path"), t]))[0]
    other = m.normalise(t, m.keep_names([m._parse("import os"), t]))[0]
    assert one != other
    # the unit's own text decides: the whole cell hashes the same in both episodes
    assert m.normalise(t, m.own_keep(t))[0] == m.units(Cell(5, code))[0].key
    assert "path" not in m.own_keep(m._parse(code)) and "len" in m.own_keep(
        m._parse(code),
    )
    with_import = m.own_keep(m._parse("import numpy as np\nx = np.zeros(3)\n"))
    assert "np" in with_import and "x" not in with_import
    # a builtin the code rebinds is a variable there
    assert "list" not in m.own_keep(m._parse("list = [1, 2]\nprint(list)\n"))


def test_a_stop_ends_the_walk_with_the_units_found_so_far():
    whole = m.units(Cell(0, PARSE_A + OTHER))
    cut = m.units(Cell(0, PARSE_A + OTHER), stop=lambda: True)
    assert [u.kind for u in cut] == ["cell"] and cut[0].key == whole[0].key
