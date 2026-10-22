"""Replay-verified cases as a data file: the test reads its inputs from it, and its provenance says it is verified by
replay, never by its bytes. (The gate does not replay it: it runs no code, rev 2.1, and labels the file not run.)"""

import json
import sys

from unify.memory import fixtures as fx
from unify.memory import replay_verify as rv

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


