"""Memory v2.1 S0 (design r3): the deterministic code miner."""

from unify.memory_v2 import mining as m
from unify.memory_v2.mining import Cell

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


def _ep(eid, req, *codes, errors=()):
    return (
        eid,
        req,
        [
            Cell(i, c, "out", "boom" if i in errors else None)
            for i, c in enumerate(codes)
        ],
    )


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


def test_a_cluster_needs_two_independent_requests():
    same = [
        _ep("e1", "r1", PARSE_A),
        _ep("e2", "r1", PARSE_B),
    ]  # a revisit: the same request bytes
    clusters, _ = m.mine(same)
    assert clusters == []
    clusters, per = m.mine([_ep("e1", "r1", PARSE_A), _ep("e2", "r2", PARSE_B)])
    assert clusters and clusters[0].episodes == 2
    top = clusters[0]
    assert {i.episode for i in top.instances} == {"e1", "e2"}
    assert per["e1"]["cells_in_clusters"] == 1 and per["e2"]["successful_cells"] == 1


def test_failed_cells_and_tracebacks_are_not_mined():
    eps = [
        _ep("e1", "r1", PARSE_A, errors=(0,)),
        ("e2", "r2", [Cell(0, PARSE_B, "Traceback (most recent call last):\n  ...")]),
    ]
    clusters, per = m.mine(eps)
    assert clusters == [] and per["e1"]["successful_cells"] == 0


def test_implied_sub_clusters_are_dropped_and_windows_are_found():
    eps = [
        _ep("e1", "r1", PARSE_A, OTHER),
        _ep("e2", "r2", PARSE_B, OTHER.replace("d = ", "e = ").replace("(d", "(e")),
    ]
    clusters, _ = m.mine(eps)
    kinds = [c.kind for c in clusters]
    # the two-cell run covers both whole cells and every statement in them: one maximal cluster remains
    assert kinds == ["window"], [(c.kind, c.size, c.template[:40]) for c in clusters]
    assert clusters[0].instances[0].lines == (1, 2)


def test_code_that_does_not_parse_is_skipped():
    clusters, per = m.mine([_ep("e1", "r1", "def (:"), _ep("e2", "r2", "def (:")])
    assert clusters == [] and per["e1"]["successful_cells"] == 1


def test_the_script_reads_a_git_copy(tmp_path):
    import json
    import subprocess

    repo = tmp_path / "eps.git"
    work = tmp_path / "w"
    subprocess.run(["git", "init", "-q", "-b", "main", str(work)], check=True)
    for eid, req, code in (("e1", ["a"], PARSE_A), ("e2", ["b"], PARSE_B)):
        d = work / "2026" / "10" / eid
        d.mkdir(parents=True)
        (d / "request.json").write_text(json.dumps(req))
        (d / "cells.jsonl").write_text(
            json.dumps({"index": 0, "code": code, "output": "", "error": None}) + "\n",
        )
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(work),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "-m",
            "x",
        ],
        check=True,
    )
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(repo)], check=True)
    out = tmp_path / "out"
    assert m.main(["--git", str(repo), "--out", str(out)]) == 0
    clusters = json.loads((out / "clusters.json").read_text())
    assert clusters and clusters[0]["episodes"] == 2
    assert {a["episode"] for a in clusters[0]["at"]} == {"e1", "e2"}
    assert set(json.loads((out / "episodes.json").read_text())) == {"e1", "e2"}
