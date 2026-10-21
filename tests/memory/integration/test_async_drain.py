"""Shutdown: wait at most the pass's bound, then cancel through the abort path; kill and
verify a worker that does not end; never start a pass; report every pass that ended after the last request,
uncounted."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from unify.memory.integration import async_pass as ap
from unify.memory.integration.paths import Paths

STUB = r"""
import json, os, signal, sys, time
home, mode = sys.argv[1], sys.argv[2]
results = os.path.join(home, "memory-state", "pass-results.jsonl")
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


def test_drain_archives_the_repo_with_its_notes_once_no_worker_writes(tmp_path):
    from unify.memory.gitio import Repo

    paths = Paths.under(tmp_path)
    mem = Repo.init_bare(paths.memory)
    mem.add_note(
        mem.head(),
        "{}",
        ref="items",
    )  # the item records live on refs/notes/items
    report = ap.drain(paths, archive_to=tmp_path / "out" / "memory.bundle")
    refs = report["archive"]["refs"]
    assert "refs/notes/items" in refs and "refs/heads/main" in refs


@pytest.mark.timeout(180)
def test_the_cli_prints_the_report(tmp_path):
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "unify.memory.integration.async_pass",
            "drain",
            "--home",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0 and json.loads(out.stdout)["phase"] == "shutdown"


UNSEEN = r"""
import fcntl, json, os, sys, time
lock, results, hold_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", flush=True)
time.sleep(hold_s)
with open(results, "a") as fh:
    fh.write(json.dumps({"pass_id": "e1.p0", "ended": "landed", "commit": None}) + "\n")
"""


def _unseen_worker(tmp_path, hold_s):
    """A pass worker the drain cannot see (its record names a pid of another pid namespace) that holds the pass lock
    for *hold_s*, then writes its result and ends."""
    paths = Paths.under(tmp_path / "home")
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", UNSEEN, str(ap.lock_path(paths)), str(ap.results_path(paths)), str(hold_s)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "held"
    now = time.time()
    ap.write_inflight(paths, ap.InFlight("e1.p0", 4_000_000, 4_000_000, "123", now, now + 120, "e1"))
    return paths, proc


