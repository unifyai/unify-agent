"""The host side of the call record: every line of the plugin's file is bounded and checked, and a record is
complete only when the plugin loaded, an end line arrived and nothing was dropped or unreadable.
"""

import json
import os

from unify.memory import example_calls as ec


def _call(**over):
    row = {
        "event": "call",
        "item": "memory.grid.fill:flood_fill",
        "node": "memory/grid/tests/test_fill.py::test_fills[0]",
        "test": "memory/grid/tests/test_fill.py::test_fills",
        "indices": {"case": 0},
        "ordinal": 0,
        "file": "memory/grid/tests/test_fill.py",
        "line": 14,
        "args": [
            {
                "name": "grid",
                "sha": "ab",
                "type": "list",
                "json": [[0, 1]],
                "cut": False,
                "path": None,
            },
        ],
        "outcome": "returned",
        "raised": None,
        "result_sha": "cd",
        "result_type": "list",
        "result_json": [[3, 1]],
        "opened": ["memory/grid/tests/data/grid1.json"],
    }
    row.update(over)
    return row


def _write(path, rows, extra=b""):
    path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows) + extra)


def test_a_full_record_parses_and_is_complete(tmp_path):
    p = tmp_path / "calls.jsonl"
    _write(
        p,
        [
            {"event": "start", "targets": ["x"], "unobservable": None},
            _call(),
            {"event": "end", "calls": 1, "dropped": 0, "unresolved": []},
        ],
    )
    rec = ec.read_calls(p)
    assert rec.complete and len(rec.calls) == 1
    c = rec.calls[0]
    assert c.key == ec.CallKey(
        "memory/grid/tests/test_fill.py::test_fills",
        (("case", 0),),
        0,
    )
    assert (
        c.args[0].name == "grid"
        and c.args[0].json == [[0, 1]]
        and c.opened == ("memory/grid/tests/data/grid1.json",)
    )


def test_incomplete_records_are_never_complete(tmp_path):
    p = tmp_path / "calls.jsonl"
    _write(p, [{"event": "start"}, _call()])  # no end line
    assert not ec.read_calls(p).complete
    _write(
        p,
        [{"event": "start"}, _call(outcome="maybe"), {"event": "end", "dropped": 0}],
    )
    rec = ec.read_calls(p)
    assert rec.unreadable == 1 and not rec.complete
    _write(p, [{"event": "start"}, {"event": "end", "dropped": 3}])
    assert ec.read_calls(p).dropped == 3 and not ec.read_calls(p).complete
    _write(p, [{"event": "start"}, {"event": "end", "dropped": 0}])
    assert not ec.read_calls(p, loaded_property=False).complete
    _write(
        p,
        [{"event": "start"}, {"event": "end", "dropped": 0}],
        extra=b"x" * (ec.MAX_LINE_BYTES + 1) + b"\n",
    )
    assert ec.read_calls(p).unreadable == 1


def test_a_missing_or_linked_record_is_not_loaded(tmp_path):
    assert not ec.read_calls(tmp_path / "none.jsonl").loaded
    real = tmp_path / "real.jsonl"
    _write(real, [{"event": "start"}, {"event": "end", "dropped": 0}])
    os.symlink(real, tmp_path / "link.jsonl")
    assert not ec.read_calls(tmp_path / "link.jsonl").loaded


def test_the_plugin_is_refused_until_it_is_built(tmp_path):
    import pytest

    if ec.CALLS_SOURCE:
        pytest.skip("the plugin is built")
    with pytest.raises(NotImplementedError):
        ec.plugin_files(tmp_path)
    assert json.loads(ec.calls_env(["b", "a", "a"])[ec.TARGETS_ENV]) == ["a", "b"]
