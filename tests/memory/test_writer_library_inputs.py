import json
from pathlib import Path

from unify.memory.episodes import Action, Cell, Episode
from unify.memory.writer_pass import export_for_writer, export_blobs


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


def test_library_export_carries_structural_events_and_actor_code(tmp_path: Path):
    export_for_writer(
        lambda e: _ep(),
        ["e1"],
        tmp_path / "episodes",
        library_mode=True,
        usage_dest=tmp_path / "usage",
    )
    row = json.loads((tmp_path / "episodes" / "e1.json").read_text())
    assert row["regime"] == "implicit" and row["memory_main"] == "abc"
    assert (
        row["cells"][1]["error"] == "Boom" and row["cells"][0]["language"] == "python"
    )
    assert {"kind": "action_error", "cell": 1, "action": 0} in row["structural_events"]
    # the use record is telemetry: its own file, never in the episode
    assert "memory_use" not in row and "signals" not in row
    assert json.loads((tmp_path / "usage" / "e1.json").read_text()) == {"imported": []}


def test_v2_export_unchanged_when_off(tmp_path: Path):
    export_for_writer(lambda e: _ep(), ["e1"], tmp_path)
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
    from tests.memory.test_writer_pass import _writer
    from unify.memory.trigger import PassRequest

    _, _, writer = _writer(tmp_path, None, **cfg)
    writer.load = lambda e: _ep()
    inputs = tmp_path / "inputs"
    writer._stage_inputs(PassRequest("incremental", "svc", ["e1"], False), inputs)
    return inputs


def test_stage_inputs_writes_the_batch_map(tmp_path: Path):
    m = json.loads(
        (_staged(tmp_path / "on") / "batch_map.json").read_text(),
    )
    assert m["version"] == 1 and m["episodes"][0]["episode_id"] == "e1"
    assert "required_parts" not in m["episodes"][0]
    assert m["episodes"][0]["end_parts"] == ["cell:0", "action:0"]
    on = tmp_path / "on" / "inputs"
    assert json.loads((on / "usage" / "e1.json").read_text()) == {"imported": []}


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
