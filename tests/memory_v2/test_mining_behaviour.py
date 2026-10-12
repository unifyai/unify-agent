"""The replay box: plain values cross it as tagged JSON, it runs only verification jobs, and it has no network, a
read-only root and a private /tmp. The in-process functions run on test code here; the box itself only where
bubblewrap exists."""

import json
import shutil

import pytest

from unify.memory_v2 import mining_behaviour as b


def test_plain_data_round_trips_and_other_values_are_refused():
    v = {
        "g": [[1, 2], [3, 4]],
        "s": {(1, 2), (3, 4)},
        "t": (1, "x"),
        "f": 1.5,
        "n": None,
        "ss": {frozenset({1, 2})},
        "dec": __import__("decimal").Decimal("1.10"),
        "d": __import__("datetime").date(2026, 10, 1),
        "dt": __import__("datetime").datetime(2026, 10, 1, 9, 30),
    }
    assert b.dec(json.loads(json.dumps(b.enc(v)))) == v
    with pytest.raises(b.NotPlain):
        b.enc(lambda: 0)


def test_the_child_runs_verification_jobs_only(monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps({"mode": "replay"})))
    with pytest.raises(KeyError):
        b.child_main()


def test_a_cell_that_reads_its_working_directory_runs_in_the_recorded_tree(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)  # restored after the test: the job changes the working directory
    (tmp_path / "claims.csv").write_text("a,1\nb,2\n")
    code = "rows = [line.split(',') for line in open('claims.csv').read().splitlines()]\ntotal = {}\nfor k, v in rows:\n    total[k] = total.get(k, 0) + int(v)\n"
    job = {
        "mode": "verify",
        "cwd": str(tmp_path),
        "cells": [{"index": 0, "code": code}],
        "cell": 0,
        "line": 3,
        "names": ["rows"],
        "targets": ["total"],
        "rewritten": code,
        "imports": [],
    }
    got = b.child_verify(json.loads(json.dumps(job)))
    first = got["first"]
    assert "error" not in first, first
    assert first["inputs"]["rows"] == [["a", "1"], ["b", "2"]]
    assert b.dec(first["after"]["total"]) == {"a": 3, "b": 2}
    assert got["first"]["after"] == got["second"]["after"] == got["third"]["after"]


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")
def test_the_box_has_no_network_and_a_private_tmp():
    code = (  # the cell raises if the box leaks: a connection opens, the root is writable, or /tmp is the host's
        "import socket, os\n"
        "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\nexcept OSError:\n    pass\n"
        "else:\n    raise RuntimeError('network')\n"
        "try:\n    open('/usr/box-probe', 'w')\nexcept OSError:\n    pass\n"
        "else:\n    raise RuntimeError('writable root')\n"
        "if os.listdir('/tmp'):\n    raise RuntimeError('shared tmp')\n"
        "ok = [1]\n"
    )
    job = {
        "mode": "verify",
        "cells": [{"index": 0, "code": code}],
        "cell": 0,
        "line": 1,
        "names": [],
        "targets": [],
        "rewritten": code,
        "imports": [],
    }
    r = b.in_box(job, 60)
    assert "box_error" not in r, r
    assert "error" not in r["first"], r["first"]
    assert r["first"]["after"]["ok"] == [1]
