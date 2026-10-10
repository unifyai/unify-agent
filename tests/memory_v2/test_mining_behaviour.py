"""Memory v2.1 S0 tier (b) (design r4 §4): behavioural equivalence. The replay and cross-run functions are called
in-process on test code here; the box is exercised only where bubblewrap exists."""

import json
import shutil

import pytest

from unify.memory_v2 import mining_behaviour as b
from unify.memory_v2.mining import Cell

GRID_A = "g = [[8,8,0,0],[8,0,0,8],[0,0,8,8],[8,0,0,0]]\nR, C = len(g), len(g[0])\nseen = set(); comps = []\n"
FILL_A = (  # a stack, direction tuples
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
GRID_B = "D = [[3,0,3],[3,3,0],[0,0,3]]\nh, w = len(D), len(D[0])\nvisited = set(); regions = []\n"
FILL_B = (  # a queue, a dr/dc loop, other names and colour
    "for i in range(h):\n"
    "    for j in range(w):\n"
    "        if D[i][j] == 3 and (i, j) not in visited:\n"
    "            q = [(i, j)]\n"
    "            visited.add((i, j))\n"
    "            region = []\n"
    "            while q:\n"
    "                a, c2 = q.pop(0)\n"
    "                region.append((a, c2))\n"
    "                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):\n"
    "                    nr, nc = a + dr, c2 + dc\n"
    "                    if 0 <= nr < h and 0 <= nc < w and D[nr][nc] == 3 and (nr, nc) not in visited:\n"
    "                        visited.add((nr, nc))\n"
    "                        q.append((nr, nc))\n"
    "            regions.append(region)\n"
)
# the same shape as FILL_B, but it collects the background (0) instead: a different behaviour
WRONG = FILL_B.replace("D[i][j] == 3", "D[i][j] != 3").replace(
    "D[nr][nc] == 3",
    "D[nr][nc] != 3",
)


def _episodes(*pairs):
    return [
        (f"e{i}", f"r{i}", [Cell(0, grid, "", None), Cell(1, fill, "", None)])
        for i, (grid, fill) in enumerate(pairs)
    ]


def _run(episodes):
    units, jobs, causes = b.collect_units(episodes)
    snaps = {}
    for job in jobs.values():
        r = b.child_replay(json.loads(json.dumps(job)))
        snaps.update(r["snaps"])
        causes.update(r["causes"])
    pairs = b.candidate_pairs(units, snaps)
    ids = {x for p in pairs for x in p}
    cross = b.child_cross(
        json.loads(
            json.dumps(
                {
                    "mode": "cross",
                    "units": {i: units[i] for i in ids},
                    "snaps": {i: snaps[i] for i in ids},
                    "pairs": pairs,
                },
            ),
        ),
    )
    keys = {u["akey"] for u in units.values()}
    return units, snaps, causes, cross, b.analyse(units, snaps, cross["results"], keys)


def test_plain_data_round_trips_and_other_values_are_refused():
    v = {
        "g": [[1, 2], [3, 4]],
        "s": {(1, 2), (3, 4)},
        "t": (1, "x"),
        "f": 1.5,
        "n": None,
        "ss": {frozenset({1, 2})},
    }
    assert b.dec(json.loads(json.dumps(b.enc(v)))) == v
    with pytest.raises(b.NotPlain):
        b.enc(lambda: 0)
    assert b.canonical([3, 1, 2], unordered=True) == b.canonical(
        [1, 2, 3],
        unordered=True,
    )
    assert b.canonical([3, 1, 2]) != b.canonical([1, 2, 3])


def test_free_names_are_the_inputs_read_before_bound():
    import ast

    names = b.free_names(ast.parse(FILL_A), {"range", "len"})
    assert names == ["R", "C", "g", "seen", "comps"]


def test_literal_parameters_are_lifted_but_structural_ones_stay():
    lifted, params = b.lift("if g[r][c]==8 and x+1<w: print('hit', -1)\n")
    assert (
        params == [8, "hit"]
        and "__q0" in lifted
        and "x + 1" in lifted
        and "-1" in lifted
    )


def test_two_flood_fills_written_differently_cluster_by_behaviour():
    units, snaps, causes, cross, report = _run(
        _episodes((GRID_A, FILL_A), (GRID_B, FILL_B)),
    )
    outer = [
        u
        for u in units.values()
        if u["src"].startswith(("for r in range", "for i in range"))
    ]
    assert len({u["akey"] for u in outer}) == 2  # tier (a) keeps them apart
    assert all(u["uid"] in snaps for u in outer), causes
    top = report["clusters"][0]
    assert top["episodes"] == 2 and len(top["akeys"]) >= 2 and top["new_vs_a"]
    assert {units[m]["episode"] for m in top["units"]} == {"e0", "e1"}
    verdicts = {(a, c): v for a, c, v in cross["results"]}
    a_out = next(u["uid"] for u in outer if u["episode"] == "e0")
    b_out = next(u["uid"] for u in outer if u["episode"] == "e1")
    assert verdicts[(a_out, b_out)] in (
        "exact",
        "unordered",
    )  # stack vs queue: the same components


def test_a_different_behaviour_does_not_cluster():
    units, snaps, causes, cross, report = _run(
        _episodes((GRID_A, FILL_A), (GRID_B, WRONG)),
    )
    outer = {
        u["uid"]
        for u in units.values()
        if u["src"].startswith(("for r in range", "for i in range"))
    }
    assert not any(outer <= set(c["units"]) for c in report["clusters"])


def test_units_that_cannot_run_are_reported_by_cause(monkeypatch):
    monkeypatch.setattr(b, "MIN_NODES", 12)
    eps = [
        (
            "e0",
            "r0",
            [
                Cell(
                    0,
                    "def f(xs):\n    for x in xs:\n        if x > 10 and x < 20 and x % 3 == 1:\n            return x * 2 + 1\n",
                    "",
                    None,
                ),
            ],
        ),
        (
            "e1",
            "r1",
            [
                Cell(
                    0,
                    "import json\nk = json.loads('[1,2]')\nraise SystemExit\nfor v in range(3):\n    k.append([v * 7, v * 9, len(k) + 11])\n",
                    "",
                    None,
                ),
            ],
        ),
    ]
    units, jobs, causes = b.collect_units(eps)
    assert any(
        w.startswith("not standalone") for w in causes.values()
    )  # the `return` outside a function
    for job in jobs.values():
        causes.update(b.child_replay(json.loads(json.dumps(job)))["causes"])
    assert "not reached: SystemExit" in causes.values()


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")
def test_the_box_has_no_network_and_a_private_tmp(tmp_path):
    code = (  # the cell raises if the box leaks: a connection opens, the root is writable, or /tmp is the host's
        "import socket, os\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\nexcept OSError:\n    pass\n"
        "else:\n    raise RuntimeError('network')\n"
        "try:\n    open('/usr/a1-box-probe', 'w')\nexcept OSError:\n    pass\n"
        "else:\n    raise RuntimeError('writable root')\n"
        "if os.listdir('/tmp'):\n    raise RuntimeError('shared tmp')\n"
    )
    job = {
        "mode": "replay",
        "cells": [{"index": 0, "code": code, "output": ""}],
        "targets": {},
    }
    r = b.in_box(job, 60)
    assert "box_error" not in r, r
    assert r["cell_errors"] == {}, r["cell_errors"]
