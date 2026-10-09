"""Spec v2.1 §6 (P7 Task 8): requests never wait for a pass, see only their pinned commit, and learn of a landed
pass once."""

from __future__ import annotations

import os
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from tests.memory_v2.test_batch_map import _ep
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import async_pass as ap
from unify.memory_v2.integration.consolidate import open_stores
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State
from unify.memory_v2.memory_writer import land, publish, served_head

SLEEPER = "import time; time.sleep(30)"
SETTINGS = SimpleNamespace(
    UNIFY_MEMORY_V21="on",
    UNIFY_MEMORY_V21_E=1,
    UNIFY_MEMORY_V21_PASS_WALL_S=60,
)


def _snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_request_pins_the_served_head_and_never_sees_an_unrecorded_pass(tmp_path):
    """Needs P3's actor export and library fixtures and P5's records: runs once they are integrated."""
    checkout = pytest.importorskip("unify.memory_v2.integration.checkout")
    if not hasattr(
        checkout,
        "export_actor_v21",
    ):  # importorskip skips only a missing module
        pytest.skip("needs P3's checkout.export_actor_v21")
    export_actor_v21 = checkout.export_actor_v21
    write_records = pytest.importorskip("unify.memory_v2.item_records").write_records
    LIB = pytest.importorskip("tests.memory_v2.test_layout").LIB
    _commit = pytest.importorskip("tests.memory_v2.test_library_helper")._commit

    mem = Repo.init_bare(tmp_path / "memory")
    old = _commit(mem, LIB, "seed")
    export_actor_v21(
        mem.git_dir,
        served_head(mem),
        tmp_path / "a",
    )  # a request in progress
    before = _snapshot(tmp_path / "a")
    with mem.temp_checkout() as wt:  # a pass's candidate: a commit that no ref names
        (wt / "memory/text/extra.py").write_text(
            'def more(x):\n    """More."""\n    return x\n',
        )
        candidate = mem.commit_all(wt, "candidate", {})
    assert served_head(mem) == old  # a request starting now pins the old commit
    assert land(mem, candidate, old) == candidate  # one writer, one compare-and-swap
    # no records map yet: still not servable
    assert mem.head() == candidate and served_head(mem) == old
    export_actor_v21(mem.git_dir, served_head(mem), tmp_path / "b")
    assert _snapshot(tmp_path / "b") == before
    write_records(mem, candidate, {})  # P5's after_passes writes the map
    # the last step: the pin-able head advances
    assert publish(mem) == candidate == served_head(mem)
    assert (
        _snapshot(tmp_path / "a") == before
    )  # the earlier request's copy is untouched
    export_actor_v21(mem.git_dir, served_head(mem), tmp_path / "c")
    assert "memory/text/extra.py" in _snapshot(tmp_path / "c")


def test_finish_spawns_and_returns_without_waiting_and_one_pass_runs_at_a_time(
    tmp_path,
):
    stores = open_stores(Paths.under(tmp_path))
    stores.evidence.index_episode(_ep(eid="e1"), "1" * 40)
    started = time.monotonic()
    ev = ap.maybe_spawn(
        stores,
        "e1",
        "1" * 40,
        effort="low",
        settings=SETTINGS,
        emit=None,
        argv_prefix=[sys.executable, "-c", SLEEPER],
    )
    rec = ap.read_inflight(stores.paths)
    try:
        assert time.monotonic() - started < 2.0 and ev["phase"] == "spawned"
        assert rec is not None and ap.owns(rec)
        assert rec.pass_id == "e1.p0" and rec.after_episode == "e1"
        assert rec.deadline_at - rec.started_at == 60
        calls = []
        busy = ap.maybe_spawn(
            stores,
            "e1",
            "1" * 40,
            effort="low",
            settings=SETTINGS,
            emit=None,
            popen=lambda *a, **k: calls.append(a),
        )
        assert busy["phase"] == "busy" and busy["pass_id"] == "e1.p0" and calls == []
        # the worker holds the pass lock
        assert ap.try_lock(ap.lock_path(stores.paths)) is None
    finally:
        os.killpg(rec.pgid, signal.SIGKILL)


def test_the_worker_argv_carries_no_credentials(tmp_path):
    stores = open_stores(Paths.under(tmp_path))
    stores.evidence.index_episode(_ep(eid="e1"), "1" * 40)
    seen = {}

    class _Proc:
        pid = os.getpid()  # a live process, so the record has a start time

    def popen(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return _Proc()

    ev = ap.maybe_spawn(
        stores,
        "e1",
        "1" * 40,
        effort="low",
        settings=SETTINGS,
        emit=None,
        popen=popen,
    )
    assert ev["phase"] == "spawned"
    argv = seen["argv"]
    assert argv[1:3] == ["-m", ap.WORKER_MODULE]
    assert argv[3:] == [
        "--home",
        str(tmp_path),
        "--episode",
        "e1",
        "--sha",
        "1" * 40,
        "--effort",
        "low",
        "--lock-fd",
        argv[-1],
    ]
    assert seen["kw"]["start_new_session"] and seen["kw"]["close_fds"]
    assert seen["kw"]["pass_fds"] == (int(argv[-1]),)


def test_not_due_spawns_nothing(tmp_path):
    stores = open_stores(Paths.under(tmp_path))
    stores.evidence.index_episode(_ep(eid="e1"), "1" * 40)
    calls = []
    ev = ap.maybe_spawn(
        stores,
        "e1",
        "1" * 40,
        effort="low",
        settings=SimpleNamespace(
            UNIFY_MEMORY_V21="on",
        ),  # E = 100k: one episode is not enough
        emit=None,
        popen=lambda *a, **k: calls.append(a),
    )
    assert ev["phase"] == "not_due" and calls == []
    assert ap.read_inflight(stores.paths) is None


def test_results_apply_once_and_name_the_first_request(tmp_path):
    paths = Paths.under(tmp_path)
    state = State(paths.state, drift={"x"}, suspect={"x", "y"})
    ap.record_result(
        paths,
        {
            "pass_id": "e1.p0",
            "ended": "landed",
            "commit": "c" * 40,
            "drift_cleared": ["x"],
            "suspect_cleared": ["x"],
        },
    )
    rows = ap.apply_results(state, paths)
    assert [r["pass_id"] for r in rows] == ["e1.p0"]
    assert state.drift == set() and state.suspect == {"y"}
    assert State.load(paths.state).suspect == {"y"}
    assert ap.apply_results(state, paths) == [] and ap.unapplied(paths) == []
    assert ap.landed_event(rows[0], "e7", "c" * 40) == {
        "type": "consolidation",
        "phase": "landed",
        "pass_id": "e1.p0",
        "ended": "landed",
        "commit": "c" * 40,
        "first_request": "e7",
        "pinned": "c" * 40,
    }


def test_a_half_written_results_line_waits_for_its_end(tmp_path):
    paths = Paths.under(tmp_path)
    paths.state_dir.mkdir(parents=True)
    ap.results_path(paths).write_text(
        '{"pass_id": "e1.p0", "ended": "landed"}\n{"pass_id": "e2',
    )
    state = State.load(paths.state)
    assert [r["pass_id"] for r in ap.apply_results(state, paths)] == ["e1.p0"]
    with open(ap.results_path(paths), "a") as fh:
        fh.write('.p0", "ended": "refused"}\n')
    assert [r["pass_id"] for r in ap.apply_results(state, paths)] == ["e2.p0"]


def test_a_record_without_a_start_time_owns_nothing():
    rec = ap.InFlight("e1.p0", os.getpid(), os.getpgid(0), "", 0.0, 1.0, "e1")
    assert not ap.owns(rec)  # never a signal on a pid alone
    live = ap.InFlight(
        "e1.p0",
        os.getpid(),
        os.getpgid(0),
        ap.proc_start(os.getpid()),
        0.0,
        1.0,
        "e1",
    )
    assert ap.owns(live)
