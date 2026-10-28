"""S14: a drain counts as verified only when nothing of the writer service is left.

What the live runner's "drain verified" rests on: ``writer_service.stop_service`` returns ``verified`` only when no
process of the service's group is alive. Two ways it can return verified while a child of the service still lives:

1. the service wrote its last state (``stopped``) and released its lock, but a child it started (a pass's box, a
   helper) is still running: the early "nothing is left to stop" return never looks at the group;
2. the service died abruptly (no ``stopped``), and a child it started after its last heartbeat and member record is
   still running: the members it did not record count only up to when it was last known alive.

The service here is a stand-in leader in its own session that records itself in ``service.json`` exactly as the
service does (pid, group, start time, heartbeat, members, state) and starts one child in its group; the unit under test,
``stop_service`` with its ``/proc`` checks, is the real one. Found as a one-off teardown failure of a system test (a
process left in the service's group after a verified drain); 60 stress runs did not repeat it, so these tests make the
two windows deterministic.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEADER = r"""
import json, os, subprocess, sys, time
from pathlib import Path
from unify.memory import writer_service as ws

home, state, beat_age = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
me = os.getpid()
child = subprocess.Popen(["sleep", "120"])  # in the leader's group, like a pass's box or a helper
members = [[me, ws.proc_start(me)]] + ([[child.pid, ws.proc_start(child.pid)]] if state == "stopped" else [])
at = ws.stamp(ws._now() - __import__("datetime").timedelta(seconds=beat_age))
doc = {"v": 1, "pid": me, "pgid": os.getpgid(0), "pgid_start": ws.proc_start(os.getpgid(0)) or "",
       "proc_start": ws.proc_start(me) or "", "heartbeat_at": at, "members": members, "state": state,
       "pass_id": None, "pass_deadline": None}
d = ws.service_dir(home)
d.mkdir(parents=True, exist_ok=True)
ws._atomic_json(d / ws.STATE_FILE, doc)
print(child.pid, flush=True)
if state == "stopped":
    os._exit(0)  # the service ends on its own: its last state written, its lock released
time.sleep(120)  # else it is killed by the test (no last state)
"""


def _leader(home: Path, state: str, beat_age: float) -> tuple[subprocess.Popen, int]:
    p = subprocess.Popen([sys.executable, "-c", LEADER, str(home), state, str(beat_age)], cwd=ROOT,
                         start_new_session=True, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT)})
    child = int(p.stdout.readline())
    return p, child


def _alive(pid: int) -> bool:
    from unify.memory import writer_service as ws

    return ws.proc_start(pid) is not None


def _stop(home: Path):
    from unify.memory import writer_service as ws

    return ws.stop_service(home, "drain", "the run ended", grace_s=0.5, term_s=1.0, settle_s=0.5)


def _cleanup(child: int) -> None:
    try:
        os.kill(child, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_a_drain_is_not_verified_while_a_child_of_a_stopped_service_lives(tmp_path):
    from unify.memory import writer_service as ws

    home = tmp_path / "home"
    leader, child = _leader(home, "stopped", 0.0)
    try:
        leader.wait(timeout=30)
        assert _alive(child) and json.loads((ws.service_dir(home) / ws.STATE_FILE).read_text())["state"] == "stopped"
        done = _stop(home)
        assert not (done.verified and _alive(child)), \
            ("the drain was verified while a child of the service is alive", done, child)
    finally:
        _cleanup(child)


def test_a_drain_is_not_verified_while_a_child_started_after_the_last_heartbeat_lives(tmp_path):
    home = tmp_path / "home"
    leader, child = _leader(home, "running", 5.0)  # last known alive 5 s before the child started
    try:
        time.sleep(0.5)
        os.kill(leader.pid, signal.SIGKILL)  # the service dies abruptly: no last state
        leader.wait(timeout=30)
        assert _alive(child)
        done = _stop(home)
        assert not (done.verified and _alive(child)), \
            ("the drain was verified while a child of the service is alive", done, child)
    finally:
        _cleanup(child)
