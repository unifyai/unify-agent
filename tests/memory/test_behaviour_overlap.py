"""Functions judged by what they do (behaviour_overlap): recorded argument lists, the comparison in the box, the
verdicts, the candidates, and the lists the gate keeps for each landed function."""

from __future__ import annotations

import shutil
import pytest

from tests.memory import library_example_world as w
from tests.memory.library_example_world import (
    library_example_world,
)  # noqa: F401 (fixture)
from tests.memory.test_layout import _tree
from unify.memory import behaviour_overlap as bo
from unify.memory.catalogue import body_digest
from unify.memory.evidence import EvidenceStore
from unify.memory.example_calls import ArgRecord, CallKey, ExampleCall
from unify.memory.gate import GateResult
from unify.memory.layout import function_bodies

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

TEST_FILE = "memory/grid/tests/test_ops.py"
GRID_SHAPE = {"shape": {"kind": "list", "of": "list"}}

FILES = {
    "memory/grid/__init__.py": '"""Grids."""\n',
    "memory/grid/ops.py": (
        '"""Grid operations."""\n\n\n'
        "class MemoryInputError(ValueError):\n    pass\n\n\n"
        "def double(grid):\n    return [[2 * v for v in row] for row in grid]\n\n\n"
        "def twice(grid):\n    out = []\n    for row in grid:\n        out.append([v + v for v in row])\n"
        "    return out\n\n\n"
        "def triple(grid):\n    return [[3 * v for v in row] for row in grid]\n\n\n"
        "def double_small(grid):\n"
        "    if max(max(r) for r in grid) > 5:\n        raise MemoryInputError('values over 5')\n"
        "    return [[2 * v for v in row] for row in grid]\n\n\n"
        "def width(grid):\n    return len(grid[0])\n\n\n"
        "def slow(grid):\n    while True:\n        pass\n\n\n"
        "def scaled(grid, k):\n    return [[k * v for v in row] for row in grid]\n"
    ),
}


def _arg(name, value, cut=False):
    return ArgRecord(
        name=name,
        sha="0" * 64,
        type=type(value).__name__,
        json=None if cut else value,
        cut=cut,
        path=None,
    )


def _call(item, *args, outcome="returned", ordinal=0):
    return ExampleCall(
        item=item,
        key=CallKey(f"{TEST_FILE}::test_ops", (), ordinal),
        node=f"{TEST_FILE}::test_ops",
        file=TEST_FILE,
        line=3,
        args=tuple(args),
        outcome=outcome,
        raised=None,
        result_sha=None,
        result_type=None,
        result_json=None,
        opened=(),
    )


def _id(name):
    return f"memory.grid.ops:{name}"


def test_recorded_inputs_keep_whole_replayable_lists_of_calls_that_returned():
    a = _id("double")
    calls = [
        _call(a, _arg("grid", [[1]]), ordinal=3),
        _call(a, _arg("grid", [[2]]), ordinal=1),
        _call(a, _arg("grid", [[2]]), ordinal=2),  # the same list again
        _call(a, _arg("grid", [[9]]), outcome="refused", ordinal=4),
        _call(a, _arg("grid", None, cut=True), ordinal=5),
        _call(a, _arg("grid", {"__type__": "Grid", "vars": {}}), ordinal=6),
        _call(a, _arg("grid", {"__bytes__": "ab"}), ordinal=7),
    ]
    assert bo.recorded_inputs(calls) == [[["grid", [[2]]]], [["grid", [[1]]]]]
    many = [_call(a, _arg("grid", [[i]]), ordinal=i) for i in range(20)]
    assert len(bo.recorded_inputs(many)) == bo.MAX_INPUTS
    big = [
        _call(a, _arg("grid", str(i) * (bo.MAX_INPUT_BYTES // 2)), ordinal=i)
        for i in range(3)
    ]
    assert len(bo.recorded_inputs(big)) == 1
    assert (
        bo.input_id([["grid", [[1]]]])
        == bo.input_id([["grid", [[1]]]])
        != bo.input_id([["grid", [[2]]]])
    )


def _results(rows):
    return {(f, i): (o, d) for f, i, o, d in rows}


def test_compare_needs_a_shared_case_from_each_side_and_no_difference():
    own = {"a": ["1", "2"], "b": ["3"]}
    same = _results(
        [
            ("a", "1", "returned", "x"),
            ("b", "1", "returned", "x"),
            ("a", "3", "returned", "y"),
            ("b", "3", "returned", "y"),
            ("a", "2", "returned", "z"),
            ("b", "2", "refused", None),
        ],
    )
    row = bo.compare("a", "b", own, same)
    assert (
        row["verdict"] == bo.SAME
        and row["agree"] == 2
        and row["own_cases"] == [1, 1]
        and row["own_lists"] == [2, 1]
    )
    differ = dict(same)
    differ[("b", "1")] = ("returned", "q")
    assert bo.compare("a", "b", own, differ)["verdict"] == bo.DIFFER
    one_side = _results(
        [
            ("a", "1", "returned", "x"),
            ("b", "1", "returned", "x"),
            ("a", "3", "unbindable", None),
            ("b", "3", "returned", "y"),
        ],
    )
    assert bo.compare("a", "b", own, one_side)["verdict"] == bo.NO_CASE
    assert bo.compare("a", "b", own, {})["verdict"] == bo.NO_CASE


def test_shape_pairs_need_a_shared_shape():
    shapes = {
        "a": {"shapes": [{**GRID_SHAPE, "param": "grid"}]},
        "b": {"shapes": [{**GRID_SHAPE, "param": "g"}]},
        "c": {"shapes": [{"shape": "text", "param": "s"}]},
    }
    assert bo.shape_pairs(shapes, ["a", "b", "c"]) == [("a", "b")]
    assert bo.shape_pairs(shapes, ["a", "b", "c"], within=["c"]) == []


@needs_bwrap
def test_candidates_are_the_pairs_that_do_the_same_on_each_others_lists(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    names = ["double", "twice", "triple", "double_small", "width", "scaled"]
    inputs = {
        _id("double"): [[["grid", [[1, 2], [3, 4]]]]],
        _id("twice"): [[["grid", [[7]]]]],
        _id("triple"): [[["grid", [[1, 2], [3, 4]]]]],
        _id("double_small"): [[["grid", [[1, 1]]]]],
        _id("width"): [[["grid", [[1, 2, 3]]]]],
        _id("scaled"): [[["grid", [[1]]], ["k", 2]]],
    }
    shapes = {_id(n): {"shapes": [{**GRID_SHAPE, "param": "grid"}]} for n in names}
    found, cut = bo.candidates(root, shapes, inputs)
    assert cut == []
    pairs = {tuple(c["items"]) for c in found}
    # double and twice compute the same; the narrower double_small agrees with double where it returns, but refuses
    # twice's only list, so it shares no case of twice's own; triple and width differ; scaled binds no other list
    assert pairs == {
        (_id("double"), _id("twice")),
        (_id("double"), _id("double_small")),
    }
    row = next(c for c in found if c["items"] == [_id("double"), _id("twice")])
    assert (
        row["rule"] == bo.RULE
        and row["agree"] == 2
        and "1 of memory.grid.ops:double's 1" in row["reasons"][0]
    )
    # deterministic
    assert bo.candidates(root, shapes, inputs)[0] == found


@needs_bwrap
def test_the_box_reports_each_way_a_call_ends(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    functions = {
        _id(n): ("memory.grid.ops", n)
        for n in ("double", "double_small", "slow", "scaled")
    }
    pool = {"g": [["grid", [[9]]]]}
    runs = [(f, "g") for f in functions]
    results, problem = bo.run_box(root, functions, pool, runs, call_timeout_s=0.5)
    assert problem is None
    assert (
        results[(_id("double"), "g")][0] == "returned"
        and len(results[(_id("double"), "g")][1]) == 64
    )
    assert results[(_id("double_small"), "g")] == ("refused", None)
    assert results[(_id("slow"), "g")] == ("timeout", None)
    assert results[(_id("scaled"), "g")] == ("unbindable", None)


def test_a_box_without_a_readable_result_gives_no_results_and_says_why(tmp_path):
    root = _tree(tmp_path / "lib", FILES)

    class R:
        returncode, stdout, stderr, timed_out = 1, "", "", False

    def confined(argv, **kw):
        return R()

    got = bo.run_box(
        root,
        {"a": ("memory.grid.ops", "double")},
        {"g": []},
        [("a", "g")],
        confined=confined,
    )
    assert got == ({}, "the comparison box left no readable result")
    R.timed_out = True
    assert bo.run_box(
        root,
        {"a": ("memory.grid.ops", "double")},
        {"g": []},
        [("a", "g")],
        confined=confined,
    ) == (
        {},
        "the comparison box timed out",
    )


def test_the_pairs_past_the_bound_are_named(tmp_path):
    root = _tree(tmp_path / "lib", FILES)

    def confined(argv, **kw):
        raise AssertionError("no function has lists: nothing runs")

    rows, cut = bo.compare_pairs(
        root,
        [("a", "b"), ("a", "c"), ("b", "c")],
        {},
        {},
        max_pairs=2,
        confined=confined,
    )
    assert list(rows) == [("a", "b"), ("a", "c")] and cut == [
        "behaviour: 1 of 3 pairs not compared",
    ]


def test_the_evidence_store_keeps_lists_by_body(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    assert ev.function_inputs(["x"]) == {}
    ev.write_function_inputs("x", "b1", [[["grid", [[1]]]]])
    ev.write_function_inputs("x", "b2", [[["grid", [[2]]]]])
    ev.write_function_inputs("y", "b1", [])  # nothing to keep
    ev.write_function_inputs(
        "x",
        "b1",
        [[["grid", [[3]]]]],
    )  # the same body again: the newer lists
    assert ev.function_inputs(["x", "y"]) == {
        "x": {"b1": [[["grid", [[3]]]]], "b2": [[["grid", [[2]]]]]},
    }
    root = _tree(tmp_path / "lib", FILES)
    digests = bo.body_digests(root)
    assert digests[_id("double")] == body_digest(function_bodies(root)[_id("double")])
    ev.write_function_inputs(_id("double"), digests[_id("double")], [[["grid", [[5]]]]])
    ev.write_function_inputs(_id("twice"), "an older body", [[["grid", [[5]]]]])
    assert bo.inputs_at(ev, digests) == {_id("double"): [[["grid", [[5]]]]]}
    assert GateResult(True).recorded_inputs == {}


@needs_bwrap
def test_the_gate_keeps_a_landed_functions_recorded_lists(library_example_world):
    mem, ev, blobs, gate = library_example_world
    files, fixtures = w.files_and_fixtures(blobs.get)
    res = gate().merge(
        mem.head(),
        w.candidate(mem, files),
        w.manifest(fixtures),
        "p1",
        "write",
        None,
        "0.01",
    )
    assert res.passed, res.reasons
    digest, lists = res.recorded_inputs[w.ITEM_ID]
    kept = ev.function_inputs([w.ITEM_ID])[w.ITEM_ID]
    assert kept == {digest: lists} and 1 <= len(lists) <= bo.MAX_INPUTS
    assert all(
        isinstance(rec, list) and all(len(arg) == 2 for arg in rec) for rec in lists
    )
