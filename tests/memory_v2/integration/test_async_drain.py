"""Spec v2.1 §6 shutdown (P7 Task 9): wait at most the pass's bound, then cancel through the abort path; kill and
verify a worker that does not end; never start a pass; report every pass that ended after the last request,
uncounted."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from unify.memory_v2.integration import async_pass as ap
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State

STUB = r"""
import json, os, signal, sys, time
home, mode = sys.argv[1], sys.argv[2]
results = os.path.join(home, "memory-v2", "pass-results.jsonl")
def done(ended):
    with open(results, "a") as fh:
        fh.write(json.dumps({"pass_id": "e1.p0", "ended": ended, "commit": None}) + "\n")
    os._exit(0)
if mode == "straggler":  # a process left in the worker's group that ignores SIGTERM
    if os.fork() == 0:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True:
            time.sleep(0.05)
    mode = "graceful"
if mode == "graceful":
    signal.signal(signal.SIGTERM, lambda *a: done("cancelled"))
elif mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
elif mode == "quick":
    done("landed")
while True:
    time.sleep(0.05)
"""


def _launch(paths, mode, deadline_in):
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", STUB, str(paths.home), mode],
        start_new_session=True,
    )
    time.sleep(0.5)  # the stub installs its handler
    now = time.time()
    ap.write_inflight(
        paths,
        ap.InFlight(
            "e1.p0",
            proc.pid,
            proc.pid,
            ap.proc_start(proc.pid) or "",
            now,
            now + deadline_in,
            "e1",
        ),
    )
    return proc


def test_drain_waits_at_most_the_bound_then_cancels(tmp_path):
    paths = Paths.under(tmp_path)
    proc = _launch(paths, "graceful", deadline_in=0.5)
    started = time.time()
    report = ap.drain(paths, grace_s=5)
    assert proc.wait(timeout=5) == 0
    assert report["in_flight"] is True and report["pass_id"] == "e1.p0"
    assert report["ended"] == "cancelled" and report["sigkill"] is False
    assert 0.3 <= report["waited_s"] < 1.5 and time.time() - started < 7
    assert report["terminated"] is True and report["counted"] is False
    assert report["after_last_request"] == [
        {"pass_id": "e1.p0", "ended": "cancelled", "commit": None},
    ]
    assert ap.read_inflight(paths) is None


def test_drain_kills_a_worker_that_ignores_sigterm(tmp_path):
    paths = Paths.under(tmp_path)
    proc = _launch(paths, "stubborn", deadline_in=0.2)
    report = ap.drain(paths, grace_s=0.5, settle_s=3)
    assert report["ended"] == "killed" and report["terminated"] is True
    assert report["sigkill"] is True
    assert proc.wait(timeout=3) == -9 and ap.read_inflight(paths) is None


def test_a_process_left_in_the_workers_group_is_killed_before_termination_is_claimed(
    tmp_path,
):
    paths = Paths.under(tmp_path)
    proc = _launch(paths, "straggler", deadline_in=0.2)
    report = ap.drain(paths, grace_s=1, settle_s=3)
    proc.wait(timeout=3)
    assert report["ended"] == "cancelled" and report["sigkill"] is True
    assert report["terminated"] is True and ap._group_alive(proc.pid) is False


def test_a_stale_record_gets_no_signal(tmp_path):
    paths = Paths.under(tmp_path)
    proc = _launch(paths, "quick", deadline_in=60)
    proc.wait(timeout=5)
    sent = []
    report = ap.drain(paths, kill=lambda *a: sent.append(a))
    assert sent == [] and report["in_flight"] is False and report["ended"] == "landed"


def test_wait_s_shortens_the_wait_and_a_finished_worker_is_reported(tmp_path):
    paths = Paths.under(tmp_path)
    proc = _launch(paths, "graceful", deadline_in=60)
    report = ap.drain(paths, wait_s=0.2, grace_s=5)
    proc.wait(timeout=5)
    assert report["ended"] == "cancelled" and report["waited_s"] < 1.0
    paths2 = Paths.under(tmp_path / "two")
    proc2 = _launch(paths2, "quick", deadline_in=60)
    proc2.wait(timeout=5)
    report2 = ap.drain(paths2)
    assert report2["in_flight"] is False and report2["ended"] == "landed"
    assert report2["waited_s"] == 0


def test_a_pass_landing_after_the_last_request_is_reported_and_not_counted(tmp_path):
    paths = Paths.under(tmp_path)
    state = State.load(paths.state)
    ap.record_result(
        paths,
        {
            "pass_id": "e1.p0",
            "ended": "landed",
            "commit": "a" * 40,
            "drift_cleared": [],
            "suspect_cleared": [],
        },
    )
    # the last request saw it
    assert [r["pass_id"] for r in ap.apply_results(state, paths)] == ["e1.p0"]
    ap.record_result(
        paths,
        {
            "pass_id": "e9.p0",
            "ended": "landed",
            "commit": "b" * 40,
            "drift_cleared": [],
            "suspect_cleared": [],
        },
    )  # after the last request
    report = ap.drain(paths)
    assert report["in_flight"] is False and report["counted"] is False
    assert report["after_last_request"] == [
        {"pass_id": "e9.p0", "ended": "landed", "commit": "b" * 40},
    ]
    events = [
        json.loads(x)
        for x in (paths.state_dir / "events.jsonl").read_text().splitlines()
    ]
    assert not any(
        e.get("phase") == "landed" and e.get("pass_id") == "e9.p0" for e in events
    )
    assert events[-1] == report
    assert ap.unapplied(paths) == [
        report["after_last_request"][0] | {"drift_cleared": [], "suspect_cleared": []},
    ]


def test_drain_never_starts_a_pass(tmp_path):
    paths = Paths.under(tmp_path)
    report = ap.drain(paths)
    assert report["in_flight"] is False and report["ended"] == "none"
    assert ap.read_inflight(paths) is None and not ap.results_path(paths).exists()
    assert not ap.lock_path(paths).exists()


def test_drain_archives_the_repo_with_its_notes_once_no_worker_writes(tmp_path):
    from unify.memory_v2.gitio import Repo

    paths = Paths.under(tmp_path)
    mem = Repo.init_bare(paths.memory)
    mem.add_note(mem.head(), "{}", ref="items")  # P5's records live on refs/notes/items
    report = ap.drain(paths, archive_to=tmp_path / "out" / "memory.bundle")
    refs = report["archive"]["refs"]
    assert "refs/notes/items" in refs and "refs/heads/main" in refs


@pytest.mark.timeout(180)
def test_the_cli_prints_the_report(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "unify.memory_v2.integration.async_pass",
            "drain",
            "--home",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0 and json.loads(out.stdout)["phase"] == "shutdown"
