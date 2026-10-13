"""Replay-verified cases as a data file: the test reads its inputs from it, the gate replays the instances its lines
name again, and the inputs it holds ground the calls they reach."""

import hashlib
import json
import sys

from unify.memory import example_evidence as xe
from unify.memory import fixtures as fx
from unify.memory import replay_verify as rv
from unify.memory.example_calls import ArgRecord, CallKey, ExampleCall

ITEM = "memory.grids.cells:double"
SOURCE = "def double(xs, pair):\n    return [2 * x for x in xs] + list(pair)\n"
CASES = [
    {
        "inputs": {"xs": [1, 2], "pair": {"t": "t", "v": [7, 8]}},
        "call": "out = double(xs, pair)",
        "expect": {"out": [2, 4, 7, 8]},
        "at": "e1:3:[2, 4]",
        "episode": "e1",
        "cell": 3,
        "lines": [2, 4],
    },
    {
        "inputs": {"xs": [5], "pair": {"t": "t", "v": [0, 1]}},
        "call": "out = double(xs, pair)",
        "expect": {"out": [10, 0, 1]},
        "at": "e2:0:[1, 1]",
        "episode": "e2",
        "cell": 0,
        "lines": [1, 1],
    },
]
REPLACES = [
    {"episode": "e1", "cell": 3, "lines": [2, 4], "call": "out = double(xs, pair)"},
    {"episode": "e2", "cell": 0, "lines": [1, 1], "call": "out = double(xs, pair)"},
]
REL = "memory/grids/tests/data/double.replay.jsonl"


def _run_test(tmp_path, data: bytes):
    lib = tmp_path / "lib"
    (lib / "memory" / "grids" / "tests" / "data").mkdir(parents=True)
    (lib / "memory" / "__init__.py").write_text("")
    (lib / "memory" / "grids" / "__init__.py").write_text("")
    (lib / "memory" / "grids" / "cells.py").write_text(SOURCE)
    rel, src = rv.test_source(ITEM, CASES)
    (lib / rel).write_text(src)
    (lib / REL).write_bytes(data)
    ns = {"__file__": str(lib / rel)}
    sys.path.insert(0, str(lib))
    try:
        exec(compile(src, str(lib / rel), "exec"), ns)
        ns["test_double_reproduces_its_recorded_instances"]()
    finally:
        sys.path.remove(str(lib))
        for m in [m for m in sys.modules if m == "memory" or m.startswith("memory.")]:
            del sys.modules[m]


def test_the_case_file_names_each_instance_and_the_test_reads_its_inputs_from_it(tmp_path):
    rel, data = rv.case_file(ITEM, CASES)
    assert rel == REL and fx.DATA_PATH.match(rel)
    lines = [json.loads(x) for x in data.decode().splitlines()]
    assert lines[0] == {
        "input": CASES[0]["inputs"],
        "source": {"episode": "e1", "source": "replay:3:2-4", "slice": None},
    }
    _, src = rv.test_source(ITEM, CASES)
    assert '"inputs"' not in src and "double.replay.jsonl" in src
    _run_test(tmp_path, data)
    one_line = data.splitlines(keepends=True)[0]
    try:
        _run_test(tmp_path / "short", one_line)
    except AssertionError as exc:
        assert "different instances" in str(exc)
    else:
        raise AssertionError("a case file with fewer lines than the test's cases passed")


def test_a_case_file_has_replay_provenance_that_bytes_alone_do_not_verify(tmp_path):
    _, data = rv.case_file(ITEM, CASES)
    entry = fx.assembled_entry(data)
    assert entry["kind"] == "replay"
    assert [(r["episode"], r["source"], r["trust"]) for r in entry["lines"]] == [
        ("e1", "replay:3:2-4", 2),
        ("e2", "replay:0:1-1", 2),
    ]
    assert fx.verify(REL, data, entry, lambda e: None, lambda s: b"") == (
        f"{REL} is a replay case file; it is verified by replaying its cases, not by its bytes"
    )
    (tmp_path / REL).parent.mkdir(parents=True)
    (tmp_path / REL).write_bytes(data)
    assert fx.derive_ledger(tmp_path)[REL] == entry
    ledger, notes = fx.ledger_for({REL: data}, {}, lambda p, s: None, lambda e: None, lambda s: b"", ["e1"])
    assert ledger == {} and notes == [
        f"G2 examples: replay case file {REL} was not re-verified by replay in this pass; it grounds no argument",
    ]


def test_case_inputs_ground_the_calls_they_reach_as_the_call_receives_them():
    _, data = rv.case_file(ITEM, CASES)
    views = {REL: fx.value_views(REL, data, fx.assembled_entry(data))}
    pair = [v for v in views[REL][fx.value_digest([7, 8])] if v.name == "line:1:input.pair"]
    assert pair == [fx.View("line:1:input.pair", "e1", "replay:3:2-4", 2, 1)]

    def call(ordinal, xs, pair):
        args = (
            ArgRecord("xs", fx.value_digest(fx.normalise(xs)), "list", xs, False, None),
            ArgRecord("pair", fx.value_digest(fx.normalise(pair)), "tuple", list(pair), False, None),
        )
        return ExampleCall(ITEM, CallKey("memory/grids/tests/test_double_replay.py::t", (), ordinal), "n", "f", 1,
                           args, "returned", None, None, None, None, ())

    calls = [call(0, [1, 2], (7, 8)), call(1, [5], (0, 1))]
    ground = xe.groundings(calls, views, {})
    assert ground[calls[0].key]["pair"] == xe.Grounding("copied", (REL,), ("e1",))
    assert xe.evidence_check(ITEM, calls, ground, set()) == ([], {"episodes": ["e1", "e2"], "fixtures": [REL]})


def test_normalise_matches_the_argument_digest_rules():
    import datetime
    import decimal

    class Point:
        def __init__(self):
            self.x, self.y = 1, (2, 3)

    assert fx.normalise((1, [2, (3,)])) == [1, [2, [3]]]
    assert fx.normalise({3, 1, 2}) == [1, 2, 3]
    assert fx.normalise(frozenset({"b", "a"})) == ["a", "b"]
    assert fx.normalise(b"ab") == {"__bytes__": hashlib.sha256(b"ab").hexdigest()}
    assert fx.normalise(decimal.Decimal("1.50")) == "1.50"
    assert fx.normalise(datetime.date(2026, 10, 12)) == "2026-10-12"
    assert fx.normalise(Point()) == {"__type__": "Point", "vars": {"x": 1, "y": [2, 3]}}


def _verified(insts, inputs):
    return [{"verdict": "verified", "case": {"inputs": i}} for i in inputs][: len(insts)]


def test_reverify_replays_the_named_instances_and_keeps_only_files_that_verify(tmp_path, monkeypatch):
    _, data = rv.case_file(ITEM, CASES)
    seen = []

    def fake(load, item, insts, tree, *, trees=None, scratch):
        seen.append((item, [i["episode"] for i in insts]))
        return _verified(insts, [c["inputs"] for c in CASES])

    monkeypatch.setattr(rv, "verify", fake)
    entries, reasons = rv.reverify(lambda e: None, {REL: data}, {ITEM: REPLACES}, tmp_path, scratch=tmp_path)
    assert reasons == [] and entries == {REL: fx.assembled_entry(data)}
    assert seen == [(ITEM, ["e1", "e2"])]

    monkeypatch.setattr(
        rv,
        "verify",
        lambda load, item, insts, tree, *, trees=None, scratch: [
            {"verdict": "verified", "case": {"inputs": CASES[0]["inputs"]}},
            {"verdict": "differs"},
        ],
    )
    entries, reasons = rv.reverify(lambda e: None, {REL: data}, {ITEM: REPLACES}, tmp_path, scratch=tmp_path)
    assert entries == {} and reasons == [
        f"[examples:replay] {REL} line 2: its instance does not verify by replay now (differs)",
    ]

    monkeypatch.setattr(rv, "verify", fake)
    edited = data.replace(b"[5]", b"[6]")
    entries, reasons = rv.reverify(lambda e: None, {REL: edited}, {ITEM: REPLACES}, tmp_path, scratch=tmp_path)
    assert entries == {} and reasons == [
        f"[examples:replay] {REL} line 2: its inputs differ from the inputs the replay records",
    ]


def test_reverify_refuses_unlisted_instances_orphan_files_and_too_many_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "verify", lambda *a, **k: [{"verdict": "verified", "case": {"inputs": {}}}])
    _, data = rv.case_file(ITEM, CASES)
    _, reasons = rv.reverify(lambda e: None, {REL: data}, {ITEM: REPLACES[:1]}, tmp_path, scratch=tmp_path)
    assert reasons == [
        f"[examples:replay] {REL} line 2 names an instance (e2 replay:0:1-1) that {ITEM}'s replaces does not list",
    ]
    _, reasons = rv.reverify(lambda e: None, {REL: data}, {}, tmp_path, scratch=tmp_path)
    assert reasons == [
        f"[examples:replay] {REL}: no function item lists replaces for this case file; replay writes it from an "
        "item's replaces",
    ]
    many = [dict(CASES[0], episode=f"e{i}") for i in range(rv.MAX_REPLAY_REVERIFY + 1)]
    _, big = rv.case_file(ITEM, many)
    insts = [dict(REPLACES[0], episode=f"e{i}") for i in range(rv.MAX_REPLAY_REVERIFY + 1)]
    entries, reasons = rv.reverify(lambda e: None, {REL: big}, {ITEM: insts}, tmp_path, scratch=tmp_path)
    assert entries == {} and reasons == [
        f"[examples:replay] {rv.MAX_REPLAY_REVERIFY + 1} recorded instances are named by new or changed replay "
        f"case files ({REL}); at most {rv.MAX_REPLAY_REVERIFY} are replayed again in one pass, so they are not "
        "judged",
    ]
