"""The writer service (UNIFY_MEMORY_WRITER_SERVICE): exactly one per memory home, a stale holder taken over, a reused pid
never taken for it, the idle exit, the stop marker with verified termination, its own cap, resumable passes, the PID
namespace case, the host's lease, and shutdown that returns only once the service is gone (the redesign's A2 tests
1-10).

The service's process tests run :mod:`tests.memory.service_driver`: the real service loop with a scripted pass, started
through ``ensure_service`` exactly as the real service is. No model, network or key.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.agents import run_record as rr
from unify.memory import attention
from unify.memory import writer_service as ws

DRIVER = str(Path(__file__).with_name("service_driver.py"))
SWITCHES = {
    "UNIFY_MEMORY": "on",
    "UNIFY_MEMORY_LIBRARY": "on",
    "UNIFY_MEMORY_LIBRARY_FORK": "on",
    "UNIFY_MEMORY_CAPTURE_TRACE": "on",
    "UNIFY_TOOLS_IN_CODE": "on",
    "UNIFY_AGENTS_RUN_RECORD": "on",
    "UNIFY_MEMORY_ATTENTION_LEDGER": "on",
    "UNIFY_MEMORY_WRITER_SERVICE": "on",
    # a pass needs room for one writer call at the worst case (13.125 USD) to start
    "UNIFY_MEMORY_WRITER_SERVICE_CAP_USD": "100",
    "UNIFY_MEMORY_LIBRARY_PASS_WALL_S": "60",
    "UNIFY_HOST_LEASE_END": "2099-01-01T00:00:00Z",  # required with the service; far off unless a test sets its own
}


def _settings(**over) -> SimpleNamespace:
    return SimpleNamespace(**{**SWITCHES, **over})


@pytest.fixture
def svc(tmp_path, monkeypatch):
    """A memory home, the switches in the environment (the service's process reads them), the driver's output folder,
    and every service started here stopped (verified) at the end."""
    home = tmp_path / "home"
    out = tmp_path / "out"
    for k, v in SWITCHES.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DRIVER_OUT", str(out))
    # the home's stores made here, as a request makes them before any service runs: the service's first look at the
    # home then writes nothing (no git process of its own to find in its group)
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    open_stores(Paths.under(home))
    started: list[Path] = [home]
    yield SimpleNamespace(home=home, out=out, homes=started)
    for h in started:
        doc = ws.read_state(h)
        if doc and ws.owns(doc):
            ws.stop_service(h, "operator", "the test ended", grace_s=0.5, term_s=2)


def _ensure(home: Path, **kw) -> ws.ServiceState:
    return ws.ensure_service(
        home,
        settings=_settings(**kw.pop("over", {})),
        argv_prefix=[sys.executable, DRIVER],
        wait_s=60,
        **kw,
    )


def _until(check, timeout=60.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = check()
        if got:
            return got
        time.sleep(step)
    raise AssertionError("timed out")


def _open_entry(home: Path, n: int = 1, first_seq: int = 1000) -> list[str]:
    """*n* open message entries, for record seqs from *first_seq* (the ledger makes one entry per seq)."""
    ledger = attention.Ledger(home)
    return [
        ledger.create(
            "message", attention.Refs(record_seq=first_seq + i), first_seq + i
        )
        for i in range(n)
    ]


def _events(home: Path, kind: str) -> list[dict]:
    return [r for r in ws.events(home) if r.get("event") == kind]


# --- 1: exactly one ----------------------------------------------------------------------------------------------------


def test_two_ensure_service_calls_at_once_leave_exactly_one_service(svc):
    got: list[ws.ServiceState] = []
    threads = [
        threading.Thread(target=lambda: got.append(_ensure(svc.home))) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    # the second call found the first's service, or started one that found the lock held and left no trace (both
    # calls then name the one service)
    assert sorted(s.status for s in got) in (
        ["busy", "started"],
        ["running", "started"],
        ["started", "started"],
    ), got
    if [s.status for s in got] == ["started", "started"]:
        assert got[0].pid == got[1].pid, got
    doc = _until(
        lambda: ws.read_state(svc.home) if ws.owns(ws.read_state(svc.home)) else None
    )
    holders = ws.lock_holders(ws.service_dir(svc.home) / ws.LOCK_FILE)
    assert holders == [doc["pid"]]
    # the stores exist (the fixture made them), so the service's first look at the home starts no git process
    assert ws.group_pids(doc["pgid"], since=doc["proc_start"]) == [doc["pid"]]
    # a third call finds it running
    assert _ensure(svc.home).status == "running"
    _until(lambda: _events(svc.home, "start"))  # written once the service has joined the record
    assert len(_events(svc.home, "start")) == 1


# --- 2 and 3: a stale holder is taken over; a reused pid is never taken for the service ----------------------------------

HOLDER = r"""
import fcntl, json, os, sys, time
home, lock = sys.argv[1], sys.argv[2]
os.setsid()
fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
stat = open(f"/proc/{os.getpid()}/stat").read().rsplit(")", 1)[-1].split()
child = os.fork()
if child == 0:  # a leftover child that kept the lock's descriptor
    while True:
        time.sleep(0.1)
cstat = open(f"/proc/{child}/stat").read().rsplit(")", 1)[-1].split()
doc = {"v": 1, "pid": os.getpid(), "pgid": os.getpgid(0), "proc_start": stat[19], "state": "running",
       "heartbeat_at": sys.argv[3], "started_at": sys.argv[3],
       "members": [[os.getpid(), stat[19]], [child, cstat[19]]]}
path = os.path.join(home, "memory-state", "memory-service", "service.json")
open(path + ".tmp", "w").write(json.dumps(doc)); os.replace(path + ".tmp", path)
print(child, flush=True)
while True:
    time.sleep(0.1)
"""


def test_a_stale_holder_whose_service_died_is_taken_over_and_a_new_service_starts(svc, monkeypatch):
    """A2 test 2: the service is killed -9 but a leftover child it recorded still holds its lock; its heartbeat goes
    stale (the stale age shortened to 1 s here, so the child can have started before the last heartbeat), so the next
    ensure_service kills the recorded leftover (verified), writes takeover and starts a new service.
    """
    monkeypatch.setattr(ws, "STALE_S", 1.0)
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    beat = ws.stamp(datetime.now(timezone.utc))
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(svc.home), str(folder / ws.LOCK_FILE), beat],
                              stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(holder.stdout.readline())
    time.sleep(1.5)  # the heartbeat is stale now
    leader_pgid = os.getpgid(holder.pid)
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait()
    assert ws.lock_holders(folder / ws.LOCK_FILE) == [child]
    got = _ensure(svc.home)
    assert got.status == "started" and got.took_over
    assert _events(svc.home, "takeover")
    assert not ws.group_pids(leader_pgid)  # the leftover was killed
    doc = ws.read_state(svc.home)
    assert ws.owns(doc) and doc["pid"] not in (holder.pid, child)


def test_a_fresh_heartbeat_is_left_alone_even_if_the_owner_check_fails(svc):
    """Under 90 s the holder is not stale: nothing is killed and no second service starts."""
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    now = ws.stamp(datetime.now(timezone.utc))
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(svc.home), str(folder / ws.LOCK_FILE), now],
                              stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(holder.stdout.readline())
    try:
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait()
        assert _ensure(svc.home).status == "running"
        assert ws.lock_holders(folder / ws.LOCK_FILE) == [child]
    finally:
        os.kill(child, signal.SIGKILL)


def test_a_service_killed_with_its_lock_is_taken_over_by_the_next_start(svc):
    """kill -9 frees the lock at once: the next ensure_service starts a service that writes takeover."""
    assert _ensure(svc.home).status == "started"
    doc = _until(
        lambda: ws.read_state(svc.home) if ws.owns(ws.read_state(svc.home)) else None
    )
    os.killpg(doc["pgid"], signal.SIGKILL)
    _until(lambda: not ws.owns(doc))
    got = _ensure(svc.home)
    assert got.status == "started"
    take = _events(svc.home, "takeover")
    assert take and take[-1]["previous_pid"] == doc["pid"]


def test_a_reused_pid_is_never_taken_for_the_service(svc, tmp_path):
    """A2 test 3: service.json names a live process of someone else (this test's own pid) with another start time:
    it is not the service, so nothing is signalled and the stop is verified at once."""
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"v": 1, "pid": os.getpid(), "pgid": os.getpgid(0), "proc_start": "1", "state": "running",
           "heartbeat_at": ws.stamp(datetime.now(timezone.utc)), "started_at": ws.stamp(datetime.now(timezone.utc))}  # fmt: skip
    (folder / ws.STATE_FILE).write_text(json.dumps(doc))
    assert not ws.owns(doc)
    sent = []
    done = ws.stop_service(
        svc.home,
        "operator",
        "test",
        kill=lambda *a: sent.append(a),
        kill_pid=lambda *a: sent.append(a),
        grace_s=0.1,
        term_s=0.1,
    )
    assert sent == [] and done == ws.Terminated(True, "clean", ())
    assert os.getpid() in ws.group_pids(os.getpgid(0))  # alive, untouched


STRANGER = r"""
import os, sys, time
os.setsid()
child = os.fork()
if child == 0:
    while True:
        time.sleep(0.1)
print(child, flush=True)
while True:
    time.sleep(0.1)
"""


def test_a_group_number_given_to_someone_elses_group_is_never_signalled(svc):
    """A pid-reuse mis-kill: the recorded service led its group and is gone; the number now leads a stranger's group
    (a leader and its child, both started after the service). Neither the stop nor a takeover touches it."""
    stranger = subprocess.Popen([sys.executable, "-c", STRANGER], stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(stranger.stdout.readline())
    try:
        doc = {"v": 1, "pid": stranger.pid, "pgid": stranger.pid, "proc_start": "1", "state": "running",
               "heartbeat_at": ws.stamp(datetime.now(timezone.utc) - timedelta(seconds=ws.STALE_S + 5)),
               "started_at": ws.stamp(datetime.now(timezone.utc)), "members": [[stranger.pid, "1"]]}  # fmt: skip
        folder = ws.service_dir(svc.home)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ws.STATE_FILE).write_text(json.dumps(doc))
        assert ws.group_pids(stranger.pid, since="1") == sorted([stranger.pid, child])
        assert ws.service_group(doc) == []
        assert ws._take_over(svc.home, doc) is False
        sent = []
        done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: sent.append(a),
                               kill_pid=lambda *a: sent.append(a), grace_s=0.1, term_s=0.1)  # fmt: skip
        assert sent == [] and done == ws.Terminated(True, "clean", ())
        assert stranger.poll() is None and ws.proc_start(child) is not None
    finally:
        os.killpg(stranger.pid, signal.SIGKILL)
        stranger.wait()


ORPHANED = r"""
import os, sys, time
os.setsid()
child = os.fork()
if child == 0:
    while True:
        time.sleep(0.1)
print(child, flush=True)
"""


def test_a_reused_group_whose_new_leader_exited_leaving_members_is_never_signalled(svc):
    """H2: the recorded service led group N and died long ago; N now names a stranger's group whose leader exited,
    leaving a member (so no leader shows a start time to compare). Nothing recorded for the service is that member
    (it started after the service's last heartbeat), so neither a stop nor a takeover signals it; no group signal is
    ever sent for a service that is not alive."""
    stranger = subprocess.Popen([sys.executable, "-c", ORPHANED], stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(stranger.stdout.readline())
    stranger.wait()  # the new leader exited; its member lives on in group N
    try:
        n = stranger.pid
        assert ws.proc_start(n) is None and ws.group_pids(n) == [child]
        child_start = ws.proc_start(child)
        doc = {"v": 1, "pid": n, "pgid": n, "proc_start": "1", "pgid_start": "1", "state": "running",
               "heartbeat_at": ws.stamp(datetime.now(timezone.utc) - timedelta(seconds=600)),
               "started_at": ws.stamp(datetime.now(timezone.utc) - timedelta(seconds=900)),
               # even a member row naming the stranger's pid and true start time: it started after the last heartbeat
               "members": [[n, "1"], [child, child_start]]}  # fmt: skip
        folder = ws.service_dir(svc.home)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ws.STATE_FILE).write_text(json.dumps(doc))
        assert ws.service_group(doc) == [] and ws.recorded_members(doc) == []
        assert ws._take_over(svc.home, doc) is False
        sent = []
        done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: sent.append(("group",) + a),
                               kill_pid=lambda *a: sent.append(("pid",) + a), grace_s=0.1, term_s=0.1)  # fmt: skip
        assert sent == [] and done == ws.Terminated(True, "clean", ())
        assert ws.proc_start(child) == child_start
    finally:
        os.kill(child, signal.SIGKILL)


def test_a_dead_services_recorded_leftover_is_signalled_by_pid_and_verified(svc):
    """H2: once the service is not alive, its recorded processes (started before its last heartbeat) are signalled
    one by one and checked gone; its group is never signalled."""
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    beat = ws.stamp(datetime.now(timezone.utc))
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(svc.home), str(folder / ws.LOCK_FILE), beat],
                              stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(holder.stdout.readline())
    os.kill(holder.pid, signal.SIGKILL)
    holder.wait()
    doc = ws.read_state(svc.home)
    assert not ws.owns(doc) and ws.recorded_members(doc) == [child]
    groups, pids = [], []

    def by_pid(pid, sig):
        pids.append((pid, sig))
        os.kill(pid, sig)

    done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: groups.append(a), kill_pid=by_pid,
                           grace_s=0.1, term_s=2)  # fmt: skip
    assert groups == [] and pids and {p for p, _ in pids} == {child}
    assert done.verified and done.how in ("term", "kill") and ws.proc_start(child) is None


def test_a_stopped_service_doc_naming_a_live_caller_is_never_signalled(svc):
    """A step (or a test) ran the loop in its own process and left service.json ``stopped`` naming it: a later stop
    has nothing to stop and signals nothing."""
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    me = {"v": 1, "pid": os.getpid(), "pgid": os.getpgid(0), "proc_start": ws.proc_start(os.getpid()),
          "pgid_start": ws.proc_start(os.getpgid(0)) or "", "state": "stopped"}  # fmt: skip
    (folder / ws.STATE_FILE).write_text(json.dumps(me))
    assert ws.owns(me)
    sent = []
    done = ws.stop_service(svc.home, "drain", "test", kill=lambda *a: sent.append(a), grace_s=0.1, term_s=0.1)
    assert sent == [] and done == ws.Terminated(True, "clean", ())


# --- 4: idle exit -------------------------------------------------------------------------------------------------------


def _lock(home: Path) -> int:
    path = ws.service_dir(home) / ws.LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


async def _no_pass(*a, **k):  # pragma: no cover - a pass would fail the test
    raise AssertionError("a pass ran")


def test_the_service_exits_after_its_idle_window_with_nothing_open(svc, monkeypatch):
    """A2 test 4, the idle clock injected: every look advances it a second; the window is 30 looks."""
    fd = _lock(svc.home)
    try:
        why = asyncio.run(
            ws.service_main(
                svc.home,
                fd,
                runner=_no_pass,
                settings=_settings(),
                poll_s=0.0,
                idle_exit_s=30,
                monotonic=Clock(),
            ),
        )
    finally:
        os.close(fd)
    assert why == "idle-exit"
    assert (
        _events(svc.home, "idle-exit")
        and _events(svc.home, "stop")[-1]["why"] == "idle-exit"
    )
    assert ws.read_state(svc.home)["state"] == "stopped"
    roster = {r["name"]: r for r in rr.read_roster(svc.home)["agents"]}
    assert roster["writer"]["state"] == "ended"


# --- 5 and 10: the stop marker during a pass; shutdown returns only once the service is gone ------------------------------


def test_the_stop_marker_during_a_pass_lets_it_run_to_its_deadline_then_kills_verified(
    svc, monkeypatch
):
    """A2 test 5: a pass in flight runs to its deadline (plus the grace); then SIGTERM, and the end is verified."""
    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("UNIFY_MEMORY_LIBRARY_PASS_WALL_S", "3")
    _open_entry(svc.home)
    assert _ensure(svc.home).status == "started"
    doc = _until(
        lambda: (d := ws.read_state(svc.home)) and d.get("state") == "running" and d
    )
    deadline = ws.parse_stamp(doc["pass_deadline"])
    started = datetime.now(timezone.utc)
    done = ws.stop_service(svc.home, "operator", "test", grace_s=1.0, term_s=10)
    assert done.verified and done.leftover_pids == ()
    assert datetime.now(timezone.utc) >= deadline  # the pass had its deadline
    assert (
        done.how in ("term", "kill")
        and (datetime.now(timezone.utc) - started).total_seconds() < 30
    )
    assert not ws.owns(doc) and not ws.group_pids(doc["pgid"], since=doc["proc_start"])
    # the pass was cancelled through its abort path: its row says so
    pass_doc = json.loads(
        (
            ws.service_dir(svc.home) / ws.TURNS_DIR / doc["pass_id"] / ws.PASS_FILE
        ).read_text()
    )
    assert pass_doc["state"] == "ended"


def test_drain_returns_only_once_the_service_is_verified_gone(svc, monkeypatch):
    """A2 test 10, through the shutdown the kit runs before it stops the key proxy."""
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_SERVICE", "on")
    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("UNIFY_MEMORY_LIBRARY_PASS_WALL_S", "2")
    _open_entry(svc.home, 2)
    assert _ensure(svc.home).status == "started"
    doc = _until(
        lambda: (d := ws.read_state(svc.home)) and d.get("state") == "running" and d
    )
    report = async_pass.drain(Paths.under(svc.home))
    assert report["service"]["verified"] is True and report["terminated"] is True
    assert not ws.owns(doc) and not ws.group_pids(doc["pgid"], since=doc["proc_start"])
    # nothing is left owed: what the run could not reach is marked so
    states = {e.state for e in attention.Ledger(svc.home).entries().values()}
    assert states <= {"not_reached", "closed", "unreadable"}
    # a request that comes after the drain does not start it again
    assert _ensure(svc.home).status == "refused"


# --- 6: the cap ---------------------------------------------------------------------------------------------------------


def test_a_cap_below_one_pass_starts_no_pass_and_reports_the_entries_not_reached(
    svc, monkeypatch
):
    """A2 test 6: a cap whose room holds not even one writer call at its worst case (13.125 USD) starts no pass."""
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    ids = _open_entry(svc.home, 3)
    fd = _lock(svc.home)
    try:
        why = asyncio.run(
            ws.service_main(
                svc.home,
                fd,
                runner=_no_pass,
                settings=_settings(UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="13.12"),
                poll_s=0.0,
                idle_exit_s=5,
                monotonic=Clock(),
            ),
        )
    finally:
        os.close(fd)
    assert why == "idle-exit"
    entries = attention.Ledger(svc.home).entries()
    assert {entries[i].state for i in ids} == {"not_reached"} and {
        entries[i].why for i in ids
    } == {"spend_cap"}
    capped = _events(svc.home, "cap-reached")
    assert (
        len(capped) == 1
        and capped[0]["by"] == "service"
        and capped[0]["room_usd"] == "13.12"
        and capped[0]["call_worst_case_usd"] == "13.125"
    )
    assert ws.spend_totals(svc.home) == (
        Decimal(0),
        Decimal(0),
    )  # nothing reserved, nothing spent


def test_the_cap_counts_every_pass_reserved_and_settled(svc):
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    rows = [
        {"op": "reserve", "pass_id": "a", "usd": "1.5"},
        {"op": "settle", "pass_id": "a", "usd": "0.25"},
        {
            "op": "reserve",
            "pass_id": "b",
            "usd": "1.5",
        },  # killed: its whole reservation stays committed
    ]
    (folder / ws.SPEND_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ws.spend_totals(svc.home) == (Decimal("1.75"), Decimal("0.25"))


# --- 7 and 8: resume, and the PID namespace -------------------------------------------------------------------------------


def _result(out: Path) -> list:
    (path,) = list(out.glob("result-*.json"))
    return json.loads(path.read_text())


def _calls(out: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out / "model-calls.jsonl").read_text().splitlines()
    ]


def _uninterrupted(tmp_path: Path, monkeypatch) -> list:
    home, out = tmp_path / "plain-home", tmp_path / "plain-out"
    monkeypatch.setenv("DRIVER_OUT", str(out))
    monkeypatch.setenv("DRIVER_BLOCK_AT", "-1")
    _open_entry(home)
    assert _ensure(home).status == "started"
    _until(lambda: list(out.glob("result-*.json")))
    ws.stop_service(home, "operator", "done", grace_s=0.5, term_s=2)
    return _result(out)


def test_a_pass_killed_at_turn_3_resumes_at_turn_3_with_one_killed_turn_and_the_same_result(
    svc, tmp_path, monkeypatch
):
    """A2 test 7."""
    want = _uninterrupted(tmp_path, monkeypatch)
    monkeypatch.setenv("DRIVER_OUT", str(svc.out))
    monkeypatch.setenv("DRIVER_BLOCK_AT", "3")
    _open_entry(svc.home)
    assert _ensure(svc.home).status == "started"
    _until(lambda: (svc.out / "blocked").exists())
    doc = ws.read_state(svc.home)
    pass_id = doc["pass_id"]
    os.killpg(doc["pgid"], signal.SIGKILL)
    _until(lambda: not ws.owns(doc))
    assert ws.pending_passes(svc.home) == [pass_id]
    assert (
        json.loads(
            (ws.service_dir(svc.home) / ws.TURNS_DIR / pass_id / "3.json").read_text()
        )["state"]
        == "pending"
    )
    assert _ensure(svc.home).status == "started"
    _until(lambda: list(svc.out.glob("result-*.json")))
    assert _result(svc.out) == want
    calls = _calls(svc.out)
    first, second = [c for c in calls if not c["resume"]], [
        c for c in calls if c["resume"]
    ]
    assert [c["turn"] for c in first] == [0, 1, 2, 3] and [
        c["turn"] for c in second
    ] == [3, 4]
    killed = _until(lambda: _events(svc.home, "killed-turn"))
    assert [k["turn_id"] for k in killed] == [f"{pass_id}/3"]
    assert _events(svc.home, "resume")[0]["pass_id"] == pass_id
    # the pass row is written once the pass has returned (after its result file): wait for it
    results = svc.home / "memory-state" / "pass-results.jsonl"
    row = _until(
        lambda: next((json.loads(x) for x in results.read_text().splitlines()), None) if results.exists() else None
    )
    assert row["pass_id"] == pass_id and row["ended"] == "landed"


@pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap makes the PID namespace"
)
def test_a_service_started_in_a_pid_namespace_that_dies_is_resumed_by_the_next_request(
    svc, monkeypatch
):
    """A2 test 8, the paper adapter's case: a request in its own PID namespace starts the service; killing the
    namespace takes the service with it (its init ends every process in it), and the next request's ensure_service
    starts one that takes over and resumes the pass, losing at most one turn."""
    monkeypatch.setenv("DRIVER_BLOCK_AT", "2")
    _open_entry(svc.home)
    request = (
        "import sys; from types import SimpleNamespace; from unify.memory import writer_service as ws; "
        f"s = SimpleNamespace(**{SWITCHES!r}); "
        f"print(ws.ensure_service({str(svc.home)!r}, settings=s, argv_prefix=[sys.executable, {DRIVER!r}], wait_s=60).status, flush=True); "
        "import time; time.sleep(3600)"
    )
    ns = subprocess.Popen(
        ["bwrap", "--unshare-pid", "--die-with-parent", "--dev-bind", "/", "/", "--proc", "/proc", "--",
         sys.executable, "-c", request],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        assert ns.stdout.readline().strip() == "started"
        _until(lambda: (svc.out / "blocked").exists())
        ns.kill()  # the namespace's init ends, and every process in the namespace with it
        ns.wait()
    finally:
        if ns.poll() is None:
            ns.kill()
    _until(
        lambda: not ws.lock_holders(ws.service_dir(svc.home) / ws.LOCK_FILE), timeout=30
    )
    got = _ensure(svc.home)
    assert got.status == "started"
    _until(lambda: list(svc.out.glob("result-*.json")))
    assert len(_events(svc.home, "killed-turn")) <= 1
    assert _events(svc.home, "takeover")


# --- 9: the lease --------------------------------------------------------------------------------------------------------


def test_with_the_lease_31_minutes_away_the_service_refuses_a_new_pass_and_stops(
    svc, monkeypatch
):
    """A2 test 9: the service never runs past the lease less 30 minutes, so a pass (60 s wall clock) that would is
    not started; the entries are not reached (lifetime)."""
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    monkeypatch.setattr(
        ws, "pass_reservation", lambda settings, effort: Decimal("0.01")
    )
    lease = ws.stamp(datetime.now(timezone.utc) + timedelta(minutes=31))
    lease = lease[:-5] + "Z"  # whole seconds
    ids = _open_entry(svc.home)
    fd = _lock(svc.home)
    try:
        why = asyncio.run(
            ws.service_main(
                svc.home,
                fd,
                runner=_no_pass,
                settings=_settings(UNIFY_HOST_LEASE_END=lease),
                poll_s=0.0,
            ),
        )
    finally:
        os.close(fd)
    assert why == "lease"
    assert _events(svc.home, "lifetime")[-1]["why"] == "lease"
    entries = attention.Ledger(svc.home).entries()
    assert entries[ids[0]].state == "not_reached" and entries[ids[0]].why == "lifetime"
    # 29 minutes away, no service starts at all
    near = ws.stamp(datetime.now(timezone.utc) + timedelta(minutes=29))[:-5] + "Z"
    got = ws.ensure_service(svc.home, settings=_settings(UNIFY_HOST_LEASE_END=near))
    assert got.status == "refused" and "lease" in got.detail


def test_the_lifetime_is_36_hours_and_never_past_the_lease_less_30_minutes():
    start = datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc)
    end, why = ws.lifetime_end(start, _settings())
    assert end - start == timedelta(hours=36) and why == "lifetime"
    end, why = ws.lifetime_end(
        start, _settings(UNIFY_HOST_LEASE_END="2026-10-20T10:00:00Z")
    )
    assert end == datetime(2026, 10, 20, 9, 30, tzinfo=timezone.utc) and why == "lease"


# --- resumable turns, directly ---------------------------------------------------------------------------------------------


def test_resumable_turns_replay_what_was_done_and_count_a_lost_turn_once(tmp_path):
    made = []

    async def model(messages, tools):
        made.append(len(messages))
        return {"role": "assistant", "content": f"r{len(messages)}"}, "0.002"

    async def run(turn, n):
        msgs = [{"role": "user", "content": "go"}]
        for _ in range(n):
            reply, _ = await turn(msgs, [])
            msgs += [reply, {"role": "user", "content": "more"}]
        return msgs

    folder = tmp_path / "turns" / "p1"
    first = ws.ResumableTurn(model, folder)
    full = asyncio.run(run(first, 3))
    # the third turn's file left pending, as a kill during its call leaves it
    doc = json.loads((folder / "2.json").read_text())
    doc["state"] = "pending"
    (folder / "2.json").write_text(json.dumps(doc))
    made.clear()
    again = ws.ResumableTurn(model, folder)
    assert asyncio.run(run(again, 3)) == full
    assert made == [5] and again.replayed == 2 and again.killed == ["p1/2"]
    # a conversation that differs from the recorded one is made again from there, never answered from a file
    made.clear()
    other = ws.ResumableTurn(model, folder)

    async def differ():
        reply, _ = await other([{"role": "user", "content": "something else"}], [])
        return reply

    asyncio.run(differ())
    assert made == [1] and other.diverged and other.killed == []


# --- the proxy's own cap, and entries a crash left handed -----------------------------------------------


class Recording:
    """A runner that records each pass's entries and returns its report (*reasons* as the pass's outcome's)."""

    def __init__(self, reasons=(), decide=False):
        self.passes: list[tuple[str, list[str], bool]] = []
        self.reasons = list(reasons)
        #: deliver and decide every handed entry (a pass that reads what it is given); else read nothing
        self.decide = decide

    async def __call__(
        self,
        home,
        pass_id,
        handed,
        *,
        resume,
        folder,
        journal_start,
        effort,
        settings,
        stop=None,
    ):
        from unify.memory.integration.consolidate import AttentionPassReport

        self.passes.append((pass_id, [e.entry for e in handed], resume))
        seqs = [e.refs.record_seq for e in handed] if self.decide else []
        if self.decide:
            attention.Ledger(home).decide(
                pass_id,
                [e.entry for e in handed],
                "nothing_to_keep",
                reason="the recording pass keeps nothing",
                reads=attention.PassReads(),
                delivered=seqs,
                decisions_path=Path(folder) / "decisions.jsonl",
            )
        return AttentionPassReport(
            pass_id,
            [e.entry for e in handed],
            "error" if self.reasons else "landed",
            outcome=SimpleNamespace(
                reasons=list(self.reasons), items_merged=[], commit=None, pool=None
            ),
            reads=attention.PassReads(),
            delivered=seqs,
            usd_known="0.001",
            settle_usd="0.001",
        )


def _in_process(home, runner, monkeypatch, **over):
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    monkeypatch.setattr(
        ws, "pass_reservation", lambda settings, effort: Decimal("0.01")
    )
    fd = _lock(home)
    try:
        return asyncio.run(
            ws.service_main(
                home,
                fd,
                runner=runner,
                settings=_settings(**over),
                poll_s=0.0,
                idle_exit_s=3,
                monotonic=Clock(),
            ),
        )
    finally:
        os.close(fd)


#: The key proxy's refusal bodies in the shape it sends them (HTTP 403, one writer-lane call at a time; its lane's cap), as a
#: model client shows them in its error's text
_BUSY_BODY = (
    '{"error": {"code": 403, "type": "benchmark_lane_budget_exhausted", '
    '"message": "benchmark budget: another writer call is in flight (one at a time)"}}'
)
_CAP_BODY = (
    '{"error": {"code": 403, "type": "benchmark_lane_budget_exhausted", "message": "benchmark budget: the spend plus '
    "one call's reserve reached this cell's ceiling of USD 210.00\"}}"
)


def _refused_reason(status: int, body: str) -> str:
    """The pass reason a writer call the route refused leaves, made as the writer's route makes it."""
    from unify.memory import writer_pass

    class PermissionDeniedError(Exception):
        status_code = status

    try:
        raise PermissionDeniedError(f"litellm.PermissionDeniedError: OpenrouterException - {body}")
    except PermissionDeniedError as exc:
        name, category, detail = writer_pass._call_failure(exc)
    return f"model call failed: WriterCallError: writer call failed: {name} ({category}){detail}"


class FirstRefused(Recording):
    """The first pass ends on *reasons*; every later pass reads and decides what it is handed."""

    async def __call__(self, home, pass_id, handed, **kw):
        if self.passes:
            self.reasons, self.decide = [], True
        return await super().__call__(home, pass_id, handed, **kw)


@pytest.mark.parametrize("status, body", [(403, _CAP_BODY), (402, "{}")], ids=["refusal-body-cap", "http-402"])
def test_a_writer_call_the_proxy_refuses_for_its_cap_is_the_cap_reached(
    svc, monkeypatch, status, body
):
    """The two caps disagree: the service reserved under its own, the proxy refused for its lane's cap (its refusal
    body says so, or HTTP 402). The service reads that as the cap reached: the entries still owed are not reached
    (spend_cap) and no pass follows.
    """
    ids = _open_entry(svc.home, 6)
    runner = Recording(reasons=[_refused_reason(status, body)])
    assert _in_process(svc.home, runner, monkeypatch) == "idle-exit"
    assert len(runner.passes) == 1
    capped = _events(svc.home, "cap-reached")
    assert [c["by"] for c in capped] == ["proxy"]
    entries = attention.Ledger(svc.home).entries()
    left = [entries[i] for i in ids if i not in runner.passes[0][1]]
    assert (
        left
        and {e.state for e in left} == {"not_reached"}
        and {e.why for e in left} == {"spend_cap"}
    )


@pytest.mark.parametrize(
    "status, body",
    [
        (403, _BUSY_BODY),  # the proxy's one-call-at-a-time refusal: busy for now, not its cap
        (403, '{"error": {"code": 403, "message": "forbidden"}}'),  # a 403 whose body names no refusal
        (403, "forbidden"),  # a 403 with no body to read: never the cap from the status code alone
        (429, "{}"),
        (503, "{}"),
    ],
    ids=["call-in-flight", "403-other-body", "403-no-body", "429", "503"],
)
def test_a_pass_ended_by_a_refusal_that_is_not_the_cap_never_stops_the_service(svc, monkeypatch, status, body):
    """One writer call refused because another call was in flight (or a 403 that does not say it is the cap, a
    rate limit, a server error) ends that pass only: no cap is reached, its entries go back, and the next pass reads
    them. (Before, any 403 was read as the proxy's cap: the service stopped every pass for the rest of the run.)"""
    ids = _open_entry(svc.home, 6)
    runner = FirstRefused(reasons=[_refused_reason(status, body)])
    assert _in_process(svc.home, runner, monkeypatch) == "idle-exit"
    assert _events(svc.home, "cap-reached") == []
    later = {e for _pid, handed, _resume in runner.passes[1:] for e in handed}
    assert len(runner.passes) >= 2 and set(runner.passes[0][1]) <= later
    entries = attention.Ledger(svc.home).entries()
    assert {entries[i].state for i in ids} == {"closed"}, {entries[i].state for i in ids}


def test_entries_a_crash_left_handed_go_back_and_are_handed_again(svc, monkeypatch):
    """On start, every entry handed to a pass that is not live and has no resumable turns goes back
    (``service_restart``) and is handed to the next pass."""
    (eid,) = _open_entry(svc.home)
    attention.Ledger(svc.home).hand("dead" * 4, 4)
    runner = Recording()
    assert _in_process(svc.home, runner, monkeypatch) == "idle-exit"
    rows = attention.Ledger(svc.home).rows()
    assert any(
        r.get("op") == "return"
        and r.get("why") == "service_restart"
        and r.get("entry") == eid
        for r in rows
    )
    assert (
        runner.passes and runner.passes[0][1] == [eid] and runner.passes[0][2] is False
    )


def test_an_agent_waiting_on_the_writer_sees_the_service_stopping(svc):
    """The stop marker also marks the writer's roster row ``stopping``, so ``who`` says the service is going
    away, and once it has ended a wait on it returns at once (nobody who could answer is running)."""
    from unify.agents import run_record_tools as T

    rec = rr.join_run_record(svc.home)
    rec.join("writer", "writer", state="idle")
    rec.join("actor-3f9a", "actor")
    assert ws.stop_service(svc.home, "operator", "test", grace_s=0.1).verified
    assert "writer (writer, idle): " in T.who_text(rr.RunRecordView(rec, "actor-3f9a", "actor"))
    assert "; stopping" in T.who_text(rr.RunRecordView(rec, "actor-3f9a", "actor"))
    rec.set_state("writer", "ended")
    found, why = asyncio.run(rec.wait_outcome("actor-3f9a", timeout=30))
    assert found == [] and why == "nothing_can_arrive"


# --- deterministic step mode (the offline replay drives the loop in-process) ---------------------------------------------


def _step(home, runner, monkeypatch, **kw):
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    monkeypatch.setattr(
        ws, "pass_reservation", lambda settings, effort: Decimal("0.01")
    )
    return ws.step(home, settings=_settings(**kw.pop("over", {})), runner=runner, **kw)


def test_a_step_runs_one_pass_over_the_open_entries_and_then_finds_nothing_open(
    svc, monkeypatch
):
    ids = _open_entry(svc.home, 2)
    runner = Recording()
    got = _step(svc.home, runner, monkeypatch)
    assert len(runner.passes) == 1 and runner.passes[0][2] is False
    assert got["pass_id"] == runner.passes[0][0] and sorted(got["handed"]) == sorted(ids)
    assert got["why_idle"] is None and got["stop"] is None and got["ended"] == "landed"
    assert got["usd"] == "0.001" and got["usd_known"] == "0.001"
    # nothing was read (the runner reported no reads): both went back, as the service sends them back
    assert sorted(got["returned"]) == sorted(ids) and got["read"] == [] and got["closed"] == []
    # the step took the home's lock and released it; no process, no heartbeat thread; the writer's row is idle
    fd = ws.try_lock(ws.service_dir(svc.home) / ws.LOCK_FILE)
    assert fd is not None
    os.close(fd)
    assert ws.read_state(svc.home)["state"] == "stopped"
    roster = {r["name"]: r["state"] for r in rr.read_roster(svc.home)["agents"]}
    assert roster["writer"] == "idle"
    # the spend is reserved and settled under the service's cap exactly as the service books it
    assert ws.spend_totals(svc.home) == (Decimal("0.001"), Decimal("0.001"))
    # the entries that went back are open again: the next step hands them again
    again = _step(svc.home, runner, monkeypatch)
    assert sorted(again["handed"]) == sorted(ids) and len(runner.passes) == 2


def test_a_step_with_nothing_open_runs_no_pass(svc, monkeypatch):
    got = _step(svc.home, _no_pass, monkeypatch)
    assert got["pass_id"] is None and got["why_idle"] == "nothing_open" and got["stop"] is None


def test_a_step_does_nothing_while_a_service_holds_the_home(svc, monkeypatch):
    _open_entry(svc.home)
    fd = _lock(svc.home)
    try:
        got = _step(svc.home, _no_pass, monkeypatch)
    finally:
        os.close(fd)
    assert got["why_idle"] == "busy" and got["pass_id"] is None


def test_a_step_at_the_stop_marker_does_what_the_service_does_when_it_stops(
    svc, monkeypatch
):
    ids = _open_entry(svc.home, 2)
    assert ws.stop_service(svc.home, "operator", "test", grace_s=0.1).verified
    got = _step(svc.home, _no_pass, monkeypatch)
    assert got["stop"] == "operator" and got["why_idle"] == "stopping"
    entries = attention.Ledger(svc.home).entries()
    assert {entries[i].state for i in ids} == {"not_reached"}
    assert {entries[i].why for i in ids} == {ws.NOT_REACHED_BY["operator"]}
    roster = {r["name"]: r["state"] for r in rr.read_roster(svc.home)["agents"]}
    assert roster["writer"] == "ended"


def test_a_step_past_the_lease_margin_starts_no_pass(svc, monkeypatch):
    _open_entry(svc.home)
    now = datetime(2026, 10, 21, 5, 45, tzinfo=timezone.utc)
    got = _step(
        svc.home,
        _no_pass,
        monkeypatch,
        over={"UNIFY_HOST_LEASE_END": "2026-10-21T06:00:00Z"},
        now=now,
    )
    assert got["stop"] == "lease" and got["pass_id"] is None


def test_the_service_loop_and_a_step_run_the_same_turn(svc, monkeypatch):
    """service_main's loop body is the step's: both go through ws._iteration."""
    seen: list[str] = []
    real = ws._iteration

    async def counted(service, *a, **k):
        seen.append("turn")
        return await real(service, *a, **k)

    monkeypatch.setattr(ws, "_iteration", counted)
    _open_entry(svc.home)
    runner = Recording(decide=True)
    assert _in_process(svc.home, runner, monkeypatch) == "idle-exit"
    looped = len(seen)
    assert looped >= 2 and len(runner.passes) == 1
    _open_entry(svc.home, first_seq=2000)
    _step(svc.home, runner, monkeypatch)
    assert len(seen) == looped + 1 and len(runner.passes) == 2


def test_with_the_run_record_off_the_service_runs_its_passes_without_it(
    svc, monkeypatch
):
    """The mesh-off arm (the mesh alone off): the service still hands the ledger's entries to its passes;
    it holds no row on a run record and writes none."""
    ids = _open_entry(svc.home, 2)
    runner = Recording()
    got = _step(svc.home, runner, monkeypatch, over={"UNIFY_AGENTS_RUN_RECORD": "off"})
    assert got["pass_id"] is not None and sorted(got["handed"]) == sorted(ids)
    assert not rr.roster_path(svc.home).exists() and not rr.record_path(svc.home).exists()
    # and a stop leaves no record behind either
    assert ws.stop_service(svc.home, "operator", "test", grace_s=0.1).verified
    stopped = _step(svc.home, _no_pass, monkeypatch, over={"UNIFY_AGENTS_RUN_RECORD": "off"})
    assert stopped["stop"] == "operator" and not rr.roster_path(svc.home).exists()


def _pipe_holders(inodes: set[int]) -> list[tuple[int, int]]:
    """(pid, inode) for every process of this user holding a descriptor on one of these pipes."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{name}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                st = os.stat(f"/proc/{name}/fd/{fd}")
            except OSError:
                continue
            if st.st_ino in inodes and (os.readlink(f"/proc/{name}/fd/{fd}") or "").startswith("pipe:"):
                out.append((int(name), st.st_ino))
    return out


def test_no_process_of_the_service_holds_a_route_token_pipe(svc, monkeypatch):
    """Review (C's live audit): the writer's and the helpers' route tokens reach the service on pipes; it reads both
    and closes them before it forks, so once it is up no process holds either pipe, and it still has both routes."""
    from pydantic import SecretStr

    from unify.memory.integration import async_pass, switch

    monkeypatch.setattr(async_pass, "_route_token", lambda settings: "writer-route-token-for-a-test")
    monkeypatch.setattr(
        switch,
        "helper_lane_route",
        lambda: (("http://127.0.0.1:9/helper/v1", SecretStr("helper-route-token-for-a-test")), None),
    )
    inodes: set[int] = set()
    real = ws._token_pipes

    def spy():
        env, fds = real()
        inodes.update(os.fstat(fd).st_ino for fd in fds)
        return env, fds

    monkeypatch.setattr(ws, "_token_pipes", spy)
    assert _ensure(svc.home).status == "started"
    doc = _until(lambda: ws.read_state(svc.home) if ws.owns(ws.read_state(svc.home)) else None)
    assert len(inodes) == 2
    _until(lambda: not _pipe_holders(inodes), timeout=30)
    assert _pipe_holders(inodes) == [] and ws.owns(doc)
    (routes,) = [json.loads(p.read_text()) for p in svc.out.glob("routes-*.json")]
    assert routes == {"writer": True, "helper": True}



def _in_ns(ns: str) -> list[int]:
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                if os.readlink(f"/proc/{name}/ns/pid") == ns:
                    out.append(int(name))
            except OSError:
                pass
    return out


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
def test_a_service_killed_mid_pass_leaves_nothing_once_stopped(svc, monkeypatch):
    """Review: a pass with a confined box and a git call in flight; the service is killed with SIGKILL (its own pid
    only). The box (bubblewrap in a session of its own) dies with it (die-with-parent); the git call is left an
    orphan in the service's group, and the stop kills it by pid (the service recorded it). Then nothing is left in
    the box's pid namespace or the service's group, and the stop is verified."""
    monkeypatch.setenv("DRIVER_CHILDREN", "1")
    _open_entry(svc.home)
    assert _ensure(svc.home).status == "started"
    kids = _until(lambda: json.loads((svc.out / "children.json").read_text()) if (svc.out / "children.json").exists() else None)
    doc = ws.read_state(svc.home)
    assert kids["service"] == doc["pid"] == doc["pgid"] and kids["box_ns"]
    group = ws.group_pids(doc["pgid"], since=doc["proc_start"])
    assert {kids["service"], kids["git"]} <= set(group), group
    assert ws.proc_start(kids["box"]) is not None and _in_ns(kids["box_ns"])  # the box runs, in its own namespace
    os.kill(doc["pid"], signal.SIGKILL)
    _until(lambda: not ws.owns(doc))
    # the stop signals only what the service recorded or what joined its group while it lived, never the group
    groups = []
    done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: groups.append(a), grace_s=0.1, term_s=5)
    assert groups == [] and done.verified, done
    _until(lambda: not _in_ns(kids["box_ns"]), timeout=15)
    assert ws.group_pids(doc["pgid"]) == [] and ws.proc_start(kids["git"]) is None
    assert ws.proc_start(kids["box"]) is None and ws.proc_start(kids["inside"]) is None



def test_the_starting_process_lets_go_of_the_token_pipes_before_it_waits(svc, monkeypatch):
    """Review (C's live audit): the process that starts the service closes its copies of the token pipes as soon as
    the child has them, not after its wait for the child: while it waits, no process but the child holds either, and
    the child lets go of both before it forks (no process holds them once it is up)."""
    from pydantic import SecretStr

    from unify.memory.integration import async_pass, switch

    monkeypatch.setattr(async_pass, "_route_token", lambda settings: "writer-route-token-for-a-test")
    monkeypatch.setattr(
        switch,
        "helper_lane_route",
        lambda: (("http://127.0.0.1:9/helper/v1", SecretStr("helper-route-token-for-a-test")), None),
    )
    inodes: set[int] = set()
    real_pipes = ws._token_pipes

    def spy():
        env, fds = real_pipes()
        inodes.update(os.fstat(fd).st_ino for fd in fds)
        return env, fds

    monkeypatch.setattr(ws, "_token_pipes", spy)
    held_while_waiting: list = []

    class Waiting:
        def __init__(self, proc):
            self.proc = proc

        def wait(self, timeout=None):
            # right after the spawn: no process but the started child holds either pipe (this one never)
            held_while_waiting.append(sorted({p for p, _ in _pipe_holders(inodes)} - {self.proc.pid}))
            return self.proc.wait(timeout=timeout)

    got = ws.ensure_service(
        svc.home,
        settings=_settings(),
        argv_prefix=[sys.executable, DRIVER],
        wait_s=60,
        popen=lambda *a, **k: Waiting(subprocess.Popen(*a, **k)),
    )
    assert got.status == "started" and len(inodes) == 2
    assert held_while_waiting == [[]], held_while_waiting
    _until(lambda: not _pipe_holders(inodes), timeout=30)



ORPHAN_GROUP = r"""
import os, sys, time
os.setsid()
child = os.fork()
if child == 0:
    while True:
        time.sleep(0.1)
print(child, flush=True)
while True:
    time.sleep(0.1)
"""


def test_a_member_the_dead_service_never_recorded_is_still_stopped(svc):
    """Review (a): the service led its group and was killed before it could record a child (no member rows at all);
    the child started while the service was last known alive, so the stop finds it in the service's group and kills
    it by pid, verified."""
    leader = subprocess.Popen([sys.executable, "-c", ORPHAN_GROUP], stdout=subprocess.PIPE, text=True)  # fmt: skip
    child = int(leader.stdout.readline())
    start = ws.proc_start(leader.pid)
    time.sleep(0.05)
    beat = ws.stamp(datetime.now(timezone.utc))  # last known alive: after the child started
    doc = {"v": 1, "pid": leader.pid, "pgid": leader.pid, "proc_start": start, "pgid_start": start,
           "state": "running", "heartbeat_at": beat, "started_at": beat, "members": []}  # fmt: skip
    folder = ws.service_dir(svc.home)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ws.STATE_FILE).write_text(json.dumps(doc))
    os.kill(leader.pid, signal.SIGKILL)
    leader.wait()
    try:
        assert ws.recorded_members(doc) == [] and ws.orphaned_members(doc) == [child]
        groups = []
        done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: groups.append(a), grace_s=0.1, term_s=5)
        assert groups == [] and done.verified and ws.proc_start(child) is None
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_a_members_file_from_another_service_is_ignored(svc):
    """Review (b): service-members.json written by another service (another pid or start time) adds nothing: a live
    process it lists is never taken for this service's."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        now = datetime.now(timezone.utc)
        doc = {"v": 1, "pid": 4_000_000, "pgid": 4_000_000, "proc_start": "1", "pgid_start": "1", "state": "running",
               "heartbeat_at": ws.stamp(now), "started_at": ws.stamp(now), "members": []}  # fmt: skip
        other = {"pid": 4_000_001, "proc_start": "2", "at": ws.stamp(now),
                 "members": [[bystander.pid, ws.proc_start(bystander.pid)]]}  # fmt: skip
        folder = ws.service_dir(svc.home)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ws.STATE_FILE).write_text(json.dumps(doc))
        (folder / ws.MEMBERS_FILE).write_text(json.dumps(other))
        assert ws.recorded_members(doc, ws.read_members(svc.home)) == []
        assert ws.service_group(doc, ws.read_members(svc.home)) == []
        # the same file, had it been this service's, would have named the bystander
        mine = dict(other, pid=doc["pid"], proc_start=doc["proc_start"])
        assert ws.recorded_members(doc, mine) == [bystander.pid]
        sent = []
        done = ws.stop_service(svc.home, "operator", "test", kill=lambda *a: sent.append(a),
                               kill_pid=lambda *a: sent.append(a), grace_s=0.1, term_s=0.1)  # fmt: skip
        assert sent == [] and done.verified and bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_a_service_on_the_writers_lane_gets_no_provider_key(monkeypatch):
    """With the writer's route the service makes every model call through its lanes, so its environment holds no
    provider key: each name gets a fixed non-key (missing, the model client would read a .env or a team's Secret
    Manager). An embedding route still sends the key, so with one the environment is left as it is."""
    from unify.memory.integration import async_pass, switch

    placeholder = "sk-or-v1-" + "0" * 64  # pragma: allowlist secret
    monkeypatch.setenv("OPENROUTER_API_KEY", placeholder)
    monkeypatch.setattr(async_pass, "_route_token", lambda settings: "writer-route-token-for-a-test")
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    monkeypatch.setattr(ws, "_settings", lambda: SimpleNamespace(UNIFY_EMBED_URL=""))
    env, fds = ws._token_pipes()
    for fd in fds:
        os.close(fd)
    assert {name: env[name] for name in ws.PROVIDER_KEY_NAMES} == dict.fromkeys(ws.PROVIDER_KEY_NAMES, ws.NO_PROVIDER_KEY)
    assert placeholder not in env.values()
    kept = ws.without_provider_keys({"OPENROUTER_API_KEY": placeholder}, SimpleNamespace(UNIFY_EMBED_URL="http://127.0.0.1:9/e"))
    assert kept == {"OPENROUTER_API_KEY": placeholder}


def test_without_the_writers_route_the_service_environment_is_unchanged(monkeypatch):
    from unify.memory.integration import async_pass, switch

    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-not-a-key")
    monkeypatch.setattr(async_pass, "_route_token", lambda settings: None)
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    env, fds = ws._token_pipes()
    assert fds == [] and env["OPENROUTER_API_KEY"] == "placeholder-not-a-key"
