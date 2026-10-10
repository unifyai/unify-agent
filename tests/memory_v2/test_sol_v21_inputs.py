import json
from pathlib import Path

from unify.memory_v2.episodes import Action, Cell, Episode
from unify.memory_v2.sol_pass import export_for_sol, export_blobs


def _ep():
    return Episode(
        episode_id="e1",
        started_at="2026-10-09T00:00:00Z",
        ended_at="2026-10-09T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="implicit",
        memory_main="abc",
        worktree_before=None,
        worktree_after=None,
        request=["req"],
        transcript=[],
        cells=[
            Cell(0, "def f():\n    return 1\n", "", error=None),
            Cell(1, "f()", "", error="Boom"),
        ],
        actions=[
            Action(
                cell=1,
                channel="svc",
                method="get",
                args=[],
                kwargs={},
                status="error",
                error="x",
            ),
        ],
        worktree_diff="",
        memory_use={"imported": []},
    )


def test_v21_export_carries_signals_and_actor_code(tmp_path: Path):
    export_for_sol(lambda e: _ep(), ["e1"], tmp_path, v21=True)
    row = json.loads((tmp_path / "e1.json").read_text())
    assert row["regime"] == "implicit" and row["memory_main"] == "abc"
    assert (
        row["cells"][1]["error"] == "Boom" and row["cells"][0]["language"] == "python"
    )
    assert {"kind": "action_error", "cell": 1, "action": 0} in row["signals"]
    assert row["memory_use"] == {"imported": []}


def test_v2_export_unchanged_when_off(tmp_path: Path):
    export_for_sol(lambda e: _ep(), ["e1"], tmp_path)
    row = json.loads((tmp_path / "e1.json").read_text())
    assert set(row) == {"episode_id", "request", "cells", "actions", "memory_channels"}
    assert set(row["cells"][0]) == {"index", "code", "output"}


class _Blobs:
    def __init__(self, data):
        self.data = data

    def has(self, s):
        return s in self.data

    def size(self, s):
        return len(self.data[s])

    def get(self, s):
        return self.data[s]


def test_export_blobs_uncapped_when_none(tmp_path: Path):
    big = b"x" * (40 * 1024**2)
    sha = "a" * 64
    ep = _ep()
    ep.actions = [
        Action(
            cell=0,
            channel="ws",
            method="write",
            args=[],
            kwargs={},
            kind="worktree",
            response={"blob_before": sha},
        ),
    ]
    out = export_blobs(
        lambda e: ep,
        ["e1"],
        _Blobs({sha: big}),
        tmp_path,
        per_blob_bytes=None,
        total_bytes=None,
    )
    assert out["exported"] == [sha] and out["skipped"] == {}


def _staged(tmp_path: Path, **cfg) -> Path:
    from tests.memory_v2.test_sol_pass import _sol
    from unify.memory_v2.trigger import PassRequest

    _, _, sol = _sol(tmp_path, None, **cfg)
    sol.load = lambda e: _ep()
    inputs = tmp_path / "inputs"
    sol._stage_inputs(PassRequest("incremental", "svc", ["e1"], False), inputs)
    return inputs


def test_stage_inputs_writes_the_batch_map_only_under_v21(tmp_path: Path):
    m = json.loads((_staged(tmp_path / "on", v21=True) / "batch_map.json").read_text())
    assert m["version"] == 1 and m["episodes"][0]["episode_id"] == "e1"
    assert m["episodes"][0]["required_parts"] == [
        "request",
        "cell:0",
        "cell:1",
        "action:0",
    ]
    assert not (_staged(tmp_path / "off") / "batch_map.json").exists()


def test_export_blobs_records_the_bytes_exported(tmp_path: Path):
    sha = "b" * 64
    ep = _ep()
    ep.actions = [
        Action(
            cell=0,
            channel="ws",
            method="write",
            args=[],
            kwargs={},
            kind="worktree",
            response={"blob_after": sha},
        ),
    ]
    out = export_blobs(
        lambda e: ep,
        ["e1"],
        _Blobs({sha: b"12345"}),
        tmp_path / "on",
        per_blob_bytes=None,
        total_bytes=None,
        record_bytes=True,
    )
    assert out["exported_bytes"] == 5
    assert (
        json.loads((tmp_path / "on" / "index.json").read_text())["exported_bytes"] == 5
    )
    off = export_blobs(lambda e: ep, ["e1"], _Blobs({sha: b"12345"}), tmp_path / "off")
    assert "exported_bytes" not in off
    assert "exported_bytes" not in json.loads(
        (tmp_path / "off" / "index.json").read_text(),
    )
