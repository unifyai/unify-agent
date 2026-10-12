"""A function is verified by replaying the recorded instances it replaces: values, stdout and file deltas, each run
in its own pristine working tree."""

import shutil
from pathlib import Path

import pytest

from unify.memory_v2 import replay_verify as rv
from unify.memory_v2.episodes import Cell
from tests.memory_v2.test_writer_library_tools import _episode

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
    assert got[0]["verdict"] == "unknown: oracle unsuitable: nondeterministic", got
    assert got[0]["oracle"]["recording"] == "reproduced"
    assert got[0]["oracle"]["deterministic"] is False and "case" not in got[0]


def test_a_bad_entry_is_a_verdict_not_a_crash(tmp_path):
    got = rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        [{"episode": "e1", "cell": 7, "lines": [1, 2], "call": "x = f()"}],
        Path(tmp_path),
        scratch=tmp_path,
    )
    assert got[0]["verdict"].startswith("refused: ")


def _one(tmp_path, cells, lines, call, source, trees=None, outputs=None):
    outputs = outputs or [""] * len(cells)
    ep = _episode("e9", [Cell(i, c, o) for i, (c, o) in enumerate(zip(cells, outputs))])
    inst = [{"episode": "e9", "cell": len(cells) - 1, "lines": lines, "call": call}]
    lib = tmp_path / f"lib-{abs(hash(source)) % 10**8}"
    pkg = lib / "memory" / "grids"
    pkg.mkdir(parents=True)
    (lib / "memory" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "regions.py").write_text(source)
    got = rv.verify(
        {"e9": ep}.__getitem__,
        "memory.grids.regions:fn",
        inst,
        lib,
        trees=trees,
        scratch=tmp_path / "scratch",
    )
    assert len(got) == 1 and got[0]["obs"] == rv._mb.OBS_VERSION, got
    return got[0]


SCALAR = "vals = [1, 2, 3]\nn = 10\nfor v in vals:\n    n += v\n"


@needs_bwrap
def test_a_scalar_result_is_compared(tmp_path):
    good = _one(
        tmp_path,
        [SCALAR],
        [3, 4],
        "n = fn(vals, n)",
        "def fn(v, n):\n    return n + sum(v)\n",
    )
    bad = _one(
        tmp_path,
        [SCALAR],
        [3, 4],
        "n = fn(vals, n)",
        "def fn(v, n):\n    return n + sum(v) + 1\n",
    )
    assert good["verdict"] == "verified" and good["case"]["expect"] == {"n": 16}, good
    assert bad["verdict"] == "differs" and bad["differs_in"] == "values", bad


DRAIN = "todo = [1, 2]\ndone = []\nwhile todo:\n    done.append(todo.pop())\n"


@needs_bwrap
def test_a_container_left_empty_is_compared(tmp_path):
    good = _one(
        tmp_path,
        [DRAIN],
        [3, 4],
        "fn(todo, done)",
        "def fn(t, d):\n    while t:\n        d.append(t.pop())\n",
    )
    bad = _one(
        tmp_path,
        [DRAIN],
        [3, 4],
        "fn(todo, done)",
        "def fn(t, d):\n    d.extend(reversed(t))\n",
    )
    assert good["verdict"] == "verified" and good["case"]["expect"]["todo"] == [], good
    assert bad["verdict"] == "differs", bad


PRINT = "rows = [3, 1]\nfor r in sorted(rows):\n    print(r)\n"


@needs_bwrap
def test_stdout_decides_when_printing_is_the_only_effect(tmp_path):
    rec = ["1\n3\n"]
    good = _one(
        tmp_path,
        [PRINT],
        [2, 3],
        "fn(rows)",
        "def fn(rows):\n    for r in sorted(rows):\n        print(r)\n",
        outputs=rec,
    )
    bad = _one(
        tmp_path,
        [PRINT],
        [2, 3],
        "fn(rows)",
        "def fn(rows):\n    for r in rows:\n        print(r)\n",
        outputs=rec,
    )
    silent = _one(
        tmp_path,
        [PRINT],
        [2, 3],
        "fn(rows)",
        "def fn(rows):\n    pass\n",
        outputs=rec,
    )
    assert good["verdict"] == "verified", good
    assert bad["verdict"] == "differs" and bad["differs_in"] == "stdout", bad
    assert silent["verdict"] == "differs" and silent["differs_in"] == "stdout", silent


@needs_bwrap
def test_a_stdout_difference_beside_another_effect_is_a_tag(tmp_path):
    cell = "vals = [1, 2]\ntotal = sum(vals)\nprint('debug', total)\n"
    got = _one(
        tmp_path,
        [cell],
        [2, 3],
        "total = fn(vals)",
        "def fn(v):\n    return sum(v)\n",
        outputs=["debug 3\n"],
    )
    assert got["verdict"] == "verified" and got.get("stdout_differs") is True, got


SAVE = "text = 'hello'\nwith open('out.txt', 'w') as f:\n    f.write(text)\n"


@needs_bwrap
def test_file_deltas_are_compared(tmp_path):
    good = _one(
        tmp_path,
        [SAVE],
        [2, 3],
        "fn(text)",
        "def fn(t):\n    with open('out.txt', 'w') as f:\n        f.write(t)\n",
    )
    bad = _one(
        tmp_path,
        [SAVE],
        [2, 3],
        "fn(text)",
        "def fn(t):\n    with open('out.txt', 'w') as f:\n        f.write(t + '!')\n",
    )
    assert good["verdict"] == "verified" and [p for p, _c, _s in good["files"]] == [
        "out.txt",
    ], good
    assert good["files"][0][1] == "created"
    assert bad["verdict"] == "differs" and bad["differs_in"] == "files", bad


@needs_bwrap
def test_each_run_starts_in_its_own_pristine_tree_and_tmp(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "log.txt").write_text("a")
    cell = (
        "import os\n"
        "first = not os.path.exists('/tmp/mark')\n"
        "open('/tmp/mark', 'w').close()\n"
        "with open('log.txt', 'a') as f:\n"
        "    f.write('x')\n"
        "n = len(open('log.txt').read()) + (0 if first else 100)\n"
    )
    got = _one(
        tmp_path,
        [cell],
        [4, 5],
        "fn('log.txt')",
        "def fn(p):\n    with open(p, 'a') as f:\n        f.write('x')\n",
        trees={"e9": tree},
    )
    # a shared tree or /tmp would make the second run differ (nondeterministic) and the third see two writes
    assert got["verdict"] == "verified", got
    assert (
        tree / "log.txt"
    ).read_text() == "a"  # the recorded tree itself is never written


@needs_bwrap
def test_aliases_are_recorded_as_unknown_and_not_compared(tmp_path):
    cell = "a = [1, 2]\nb = a\n"
    got = _one(
        tmp_path,
        [cell],
        [2, 2],
        "b = fn(a)",
        "def fn(x):\n    return list(x)\n",
    )
    assert got["verdict"] == "verified", got
    assert got["unknown"]["original"]["aliases"] == [["a", "b"]]
    assert got["unknown"]["rewritten"]["aliases"] == []


def test_a_box_failure_is_transient(tmp_path, monkeypatch):
    monkeypatch.setattr(rv._mb, "in_box", lambda *a, **k: {"box_error": "box timeout"})
    got = rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        INST,
        _lib(tmp_path, GOOD),
        scratch=tmp_path / "scratch",
    )
    assert (
        got[0]["verdict"] == "not replayable: box timeout"
        and got[0]["transient"] is True
    )


def test_the_three_runs_get_separate_writable_trees(tmp_path, monkeypatch):
    seen = {}

    def fake(job, timeout, binds):
        seen["binds"], seen["cwds"] = binds, job["cwds"]
        return {"box_error": "stop"}

    monkeypatch.setattr(rv._mb, "in_box", fake)
    rv.verify(
        {"e1": EP}.__getitem__,
        "memory.grids.regions:components",
        INST,
        _lib(tmp_path, GOOD),
        scratch=tmp_path / "scratch",
    )
    worlds = [(src, dst) for src, dst, rw in seen["binds"] if rw]
    assert [d for _s, d in worlds] == ["/w1", "/w2", "/w3"] == seen["cwds"]
    assert len({s for s, _d in worlds}) == 3
    assert not any(Path(s).exists() for s, _d in worlds)  # removed after the box


def test_a_result_of_another_observation_version_is_not_compared():
    job = {"targets": ["n"], "names": [], "rewritten_call": "n = f()"}
    run = {"after": {"n": 1}, "bound": [], "stdout": [], "files": [], "obs": 1}
    assert rv.verdict(job, {"first": run, "second": run, "third": run})["verdict"] == (
        "not replayable: observation version mismatch"
    )


# --- the recording is the oracle only when the original run reproduces it -------------------------------------


@needs_bwrap
def test_an_original_that_does_not_reproduce_the_recording_is_unknown(tmp_path):
    cell = "rows = [3, 1]\nprint(sorted(rows))\nn = len(rows)\n"
    src = "def fn(r):\n    return len(r)\n"
    got = _one(tmp_path, [cell], [3, 3], "n = fn(rows)", src, outputs=["[3, 1]\n"])
    assert (
        got["verdict"]
        == "unknown: oracle unsuitable: the original does not reproduce the recorded output"
    ), got
    assert got["oracle"] == {
        "recording": "not reproduced",
        "differs_at": [0],
        "unrecorded": [],
        "deterministic": None,
    }
    assert "case" not in got and rv.summary([got])["unknown"] == 1
    same = _one(
        tmp_path / "again",
        [cell],
        [3, 3],
        "n = fn(rows)",
        src,
        outputs=["[1, 3]\n"],
    )
    assert same["verdict"] == "verified", same
    assert same["oracle"] == {
        "recording": "reproduced",
        "differs_at": [],
        "unrecorded": [],
        "deterministic": True,
    }


@needs_bwrap
def test_a_deterministic_reproducing_original_gets_the_usual_verdicts(tmp_path):
    good = _one(
        tmp_path,
        [SCALAR],
        [3, 4],
        "n = fn(vals, n)",
        "def fn(v, n):\n    return n + sum(v)\n",
    )
    bad = _one(
        tmp_path,
        [SCALAR],
        [3, 4],
        "n = fn(vals, n)",
        "def fn(v, n):\n    return n - sum(v)\n",
    )
    for got in (good, bad):
        assert (
            got["oracle"]["recording"] == "reproduced"
            and got["oracle"]["deterministic"] is True
        ), got
    assert good["verdict"] == "verified" and bad["verdict"] == "differs"
    assert rv.summary([good, bad]) == {
        "verified": 1,
        "differs": 1,
        "unknown": 0,
        "not_replayable": 0,
        "refused": 0,
        "tested": True,
    }


JOB = {
    "cells": [{"index": 0, "code": "n = 1\nprint('hi')\n", "output": "hi\n"}],
    "targets": ["n"],
    "names": [],
    "rewritten_call": "n = f()",
}


def _run(printed="hi", n=1, stdout=(3, "a")):
    return {
        "after": {"n": n},
        "bound": [],
        "stdout": [list(stdout)],
        "printed": [rv._mb.printed_digest(printed)],
        "files": [],
        "obs": rv._mb.OBS_VERSION,
    }


def test_the_original_must_reproduce_the_recording_before_anything_is_compared():
    got = rv.verdict(
        JOB,
        {"first": _run("bye"), "second": _run("bye"), "third": _run("bye")},
    )
    assert got["verdict"].startswith("unknown: oracle unsuitable: ") and got["oracle"][
        "differs_at"
    ] == [0]
    # a wrong function does not turn an unsuitable oracle into a failure
    got = rv.verdict(
        JOB,
        {"first": _run("bye"), "second": _run("bye"), "third": _run("bye", n=2)},
    )
    assert got["verdict"].startswith("unknown: oracle unsuitable: "), got


def test_a_nondeterministic_original_is_unknown_whatever_differs():
    for second in (_run(n=2), _run(stdout=(3, "b")), {"error": "ValueError"}):
        got = rv.verdict(JOB, {"first": _run(), "second": second, "third": _run()})
        assert got["verdict"] == "unknown: oracle unsuitable: nondeterministic", (
            second,
            got,
        )
        assert (
            got["oracle"]["recording"] == "reproduced"
            and got["oracle"]["deterministic"] is False
        )


def test_an_original_that_raises_is_unknown():
    got = rv.verdict(
        JOB,
        {"first": {"error": "NameError"}, "second": _run(), "third": _run()},
    )
    assert (
        got["verdict"] == "unknown: oracle unsuitable: the original raised NameError"
    ), got
    assert got["oracle"]["recording"] == "not reproduced"


def test_a_reproducing_deterministic_original_is_compared_as_usual():
    ok = rv.verdict(JOB, {"first": _run(), "second": _run(), "third": _run()})
    bad = rv.verdict(JOB, {"first": _run(), "second": _run(), "third": _run(n=2)})
    assert ok["verdict"] == "verified" and ok["oracle"] == {
        "recording": "reproduced",
        "differs_at": [],
        "unrecorded": [],
        "deterministic": True,
    }
    assert bad["verdict"] == "differs" and bad["oracle"]["deterministic"] is True


def test_a_missing_recording_is_said_not_taken_as_reproduced():
    job = {**JOB, "cells": [{**JOB["cells"][0], "output": None}]}
    got = rv.verdict(job, {"first": _run(), "second": _run(), "third": _run()})
    assert got["verdict"] == "verified"
    assert got["oracle"]["recording"] == "not available" and got["oracle"][
        "unrecorded"
    ] == [0]


def test_recorded_output_readings_drop_what_the_recorder_adds():
    assert rv.recorded_readings(None) is None
    assert "" in rv.recorded_readings("(no output)")
    assert "" in rv.recorded_readings('{\n  "duration_ms": 12\n}\n[variables] n: int')
    assert "hi" in rv.recorded_readings("hi\n\n[variables] n: int")
    assert "" not in rv.recorded_readings("hi")


def test_the_summary_counts_unknowns_apart():
    got = rv.summary(
        [
            {"verdict": "unknown: oracle unsuitable: nondeterministic"},
            {"verdict": "not replayable: box timeout", "transient": True},
            {"verdict": "refused: KeyError: 'episode'"},
        ],
    )
    assert got == {
        "verified": 0,
        "differs": 0,
        "unknown": 1,
        "not_replayable": 1,
        "refused": 1,
        "tested": False,
    }
