"""Memory v2.1 r5 S2: a function is verified by replaying the recorded instances it replaces."""

import shutil
from pathlib import Path

import pytest

from unify.memory_v2 import replay_verify as rv
from unify.memory_v2.episodes import Cell
from tests.memory_v2.test_sol_v21_tools import _episode

GRID = "g = [[8,8,0],[0,0,8],[8,0,8]]\nR, C = len(g), len(g[0])\nseen = set(); comps = []\n"
FILL = (
    "for r in range(R):\n"
    " for c in range(C):\n"
    "  if g[r][c]==8 and (r,c) not in seen:\n"
    "   st=[(r,c)];seen.add((r,c));comp=[]\n"
    "   while st:\n"
    "    x,y=st.pop();comp.append((x,y))\n"
    "    for nx,ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):\n"
    "     if 0<=nx<R and 0<=ny<C and g[nx][ny]==8 and (nx,ny) not in seen:seen.add((nx,ny));st.append((nx,ny))\n"
    "   comps.append(comp)\n"
)
GOOD = '''def components(g, colour, seen, comps):
    """Append each 4-connected region of *colour* not yet in *seen* to *comps*."""
    R, C = len(g), len(g[0])
    for r in range(R):
        for c in range(C):
            if g[r][c] == colour and (r, c) not in seen:
                st = [(r, c)]
                seen.add((r, c))
                comp = []
                while st:
                    x, y = st.pop()
                    comp.append((x, y))
                    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                        if 0 <= nx < R and 0 <= ny < C and g[nx][ny] == colour and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            st.append((nx, ny))
                comps.append(comp)
'''
BAD = GOOD.replace("comps.append(comp)", "pass")
EP = _episode("e1", [Cell(0, GRID, ""), Cell(1, FILL, "")])
INST = [
    {
        "episode": "e1",
        "cell": 1,
        "lines": [1, 9],
        "call": "components(g, 8, seen, comps)",
    },
]


def _lib(tmp_path, source):
    pkg = tmp_path / "lib" / "memory" / "grids"
    pkg.mkdir(parents=True)
    (tmp_path / "lib" / "memory" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "regions.py").write_text(source)
    return tmp_path / "lib"


def test_call_names_and_rewrite():
    assert rv.call_names("out = f(g, 8, seen)") == (["f", "g", "seen"], ["out"])
    assert (
        rv.rewrite("a=1\n  b=2\n  c=3\nd=4", (2, 3), "x = f(a)")
        == "a=1\n  x = f(a)\nd=4\n"
    )
    with pytest.raises(ValueError):
        rv.call_names("a = 1; b = 2")


needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="needs bubblewrap",
)


@needs_bwrap
def test_a_function_that_reproduces_the_instance_is_verified_and_gets_a_test(tmp_path):
    lib = _lib(tmp_path, GOOD)
    got = rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        INST,
        lib,
        scratch=tmp_path,
    )
    assert [g["verdict"] for g in got] == ["verified"], got
    rel, src = rv.test_source(
        "memory.grids.regions:components",
        [{**got[0]["case"], "at": "e1:1"}],
    )
    assert (
        rel == "memory/grids/tests/test_components_replay.py"
        and "from memory.grids.regions import components" in src
    )
    test = lib / rel
    test.parent.mkdir(parents=True)
    test.write_text(src)
    ns = {}
    import sys

    sys.path.insert(0, str(lib))
    try:
        exec(compile(src, str(test), "exec"), ns)
        ns[
            "test_components_reproduces_its_recorded_instances"
        ]()  # the generated test passes on the library
    finally:
        sys.path.remove(str(lib))
        for m in [m for m in sys.modules if m == "memory" or m.startswith("memory.")]:
            del sys.modules[m]


@needs_bwrap
def test_a_wrong_function_differs(tmp_path):
    got = rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        INST,
        _lib(tmp_path, BAD),
        scratch=tmp_path,
    )
    assert [g["verdict"] for g in got] == ["differs"]


@needs_bwrap
def test_replay_nondeterministic_is_not_verified(tmp_path):
    ep = _episode(
        "e2",
        [
            Cell(
                0,
                "import random\nvals = [random.random() for _ in range(5)]\nout = sorted(vals)\n",
                "",
            ),
        ],
    )
    inst = [
        {"episode": "e2", "cell": 0, "lines": [3, 3], "call": "out = components(vals)"},
    ]
    got = rv.verify(
        {"e2": ep}.__getitem__,
        "memory.grids.regions:components",
        inst,
        _lib(tmp_path, "def components(v):\n    return sorted(v)\n"),
        scratch=tmp_path,
    )
    assert got[0]["verdict"] == "not replayable: nondeterministic"


def test_a_bad_entry_is_a_verdict_not_a_crash(tmp_path):
    got = rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        [{"episode": "e1", "cell": 7, "lines": [1, 2], "call": "x = f()"}],
        Path(tmp_path),
        scratch=tmp_path,
    )
    assert got[0]["verdict"].startswith("refused: ")
