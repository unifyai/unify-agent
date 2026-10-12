"""Memory v2.1 r5 (r4 §4): S0 at the start of a WRITE pass; it never stops a pass."""

import asyncio
import json
import shutil

import pytest

from unify.memory_v2 import mining_behaviour, s0_pass
from unify.memory_v2.episodes import Cell
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_mining_behaviour import FILL_A, FILL_B, GRID_A, GRID_B
from tests.memory_v2.test_writer_pass import Turns, _call, _sol
from tests.memory_v2.test_writer_library_tools import _episode

EPS = {
    "e0": _episode(
        "e0",
        [Cell(0, GRID_A, ""), Cell(1, FILL_A, "")],
        request=("first request",),
    ),
    "e1": _episode(
        "e1",
        [Cell(0, GRID_B, ""), Cell(1, FILL_B, "")],
        request=("second request",),
    ),
}


def test_tier_a_clusters_and_seen_shapes_without_a_box(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mining_behaviour,
        "in_box",
        lambda *a, **k: {"box_error": "box exit 1"},
    )
    same = {
        **EPS,
        "e2": _episode(
            "e2",
            [Cell(0, GRID_A, ""), Cell(1, FILL_A, "")],
            request=("third",),
        ),
    }
    got = s0_pass.run(
        same.__getitem__,
        ["e2"],
        ["e0", "e1"],
        tmp_path / "s0",
        tmp_path / "cache",
    )
    assert (
        "error" not in got
        and got["tier_a_clusters"] >= 1
        and got["clustered"] == ["e2"]
    )
    assert got[
        "seen"
    ]  # the earlier episodes' code shapes, for the analysts' novelty flag
    assert json.loads((tmp_path / "s0" / "clusters.json").read_text())


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="needs bubblewrap")
def test_tier_b_finds_the_two_flood_fills_and_caches_replays(tmp_path):
    got = s0_pass.run(
        EPS.__getitem__,
        ["e1"],
        ["e0"],
        tmp_path / "s0",
        tmp_path / "cache",
        jobs=2,
    )
    assert got["tier_b"]["behavioural_clusters"] >= 1 and "e1" in got["clustered"]
    assert sorted(p.name for p in (tmp_path / "cache" / "replays").iterdir()) == [
        "e0.json",
        "e1.json",
    ]
    assert not (
        tmp_path / "s0" / "work"
    ).exists()  # tier (b)'s work stays out of /inputs


def test_s0_failure_does_not_stop_the_pass(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bubblewrap and prlimit are required")

    monkeypatch.setattr(s0_pass, "_as_mining", boom)
    model = Turns([_call("c", "read", {"path": "/inputs/batch_map.json"})])
    _, _, writer = _sol(
        tmp_path,
        model,
        library_mode=True,
        max_calls=4,
        s0_cache=str(tmp_path / "cache"),
    )
    writer.load = EPS.__getitem__
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"),
    )
    assert out.s0 and "error" in out.s0 and out.coverage is not None


def test_s0_runs_on_every_pass_when_the_loader_is_bound_to_the_calling_thread(
    tmp_path,
    monkeypatch,
):
    """M3c/M4 (11 Oct): S0 died on every pass after the first. The pass's loader reaches a SQLite connection that
    belongs to the calling thread, and S0 runs in a worker thread; with history to load it raised. RED on
    f31f504a6."""
    import sqlite3

    monkeypatch.setattr(
        mining_behaviour,
        "in_box",
        lambda *a, **k: {"box_error": "no box in this test"},
    )
    conn = sqlite3.connect(":memory:")  # created here: other threads may not use it

    def load(eid):
        conn.execute("SELECT 1").fetchone()
        return EPS[eid]

    outs = []
    for pass_id, batch in (("p1", ["e0"]), ("p2", ["e1"])):
        model = Turns([_call("c", "read", {"path": "/inputs/batch_map.json"})])
        _, _, writer = _sol(
            tmp_path / pass_id,
            model,
            library_mode=True,
            max_calls=4,
            s0_cache=str(tmp_path / "cache"),
        )
        writer.load = load
        seen = ["e0"] if pass_id == "p2" else []
        monkeypatch.setattr(
            writer.ev,
            "experience_since",
            lambda after, _s=seen: [(e, 0) for e in _s],
        )
        outs.append(
            asyncio.run(
                writer.run(PassRequest("incremental", "svc", batch, False), pass_id),
            ),
        )
    for out in outs:
        assert out.s0 is not None and "error" not in out.s0, out.s0


def test_an_unreadable_history_episode_is_s0s_error_and_the_pass_goes_on(
    tmp_path,
    monkeypatch,
):
    """RUNTIME (11 Oct): loading S0's episodes on the pass's thread must keep "S0 never stops a pass"."""
    monkeypatch.setattr(
        mining_behaviour,
        "in_box",
        lambda *a, **k: {"box_error": "no box in this test"},
    )

    def load(eid):
        if eid == "e0":
            raise OSError("history episode unreadable")
        return EPS[eid]

    model = Turns([_call("c", "read", {"path": "/inputs/batch_map.json"})])
    _, _, writer = _sol(
        tmp_path,
        model,
        library_mode=True,
        max_calls=4,
        s0_cache=str(tmp_path / "cache"),
    )
    writer.load = load
    monkeypatch.setattr(writer.ev, "experience_since", lambda after: [("e0", 0)])
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"),
    )
    assert out.s0 and "OSError" in out.s0["error"] and out.coverage is not None
