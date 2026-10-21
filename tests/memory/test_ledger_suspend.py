"""Rev 2, item 4: the end of one stage of a run that goes on (``suspend``) is not the run's end.

A ``final`` drain marks every attention entry still owed ``not_reached: run_ended`` (every entry ends closed or listed as not reached, at the run's end);
a ``suspend`` drain stops the service the same verified way and marks nothing: the entries stay owed, the suspend's stop
marker is set aside, and the next start hands them again (those a stopped pass held going back with ``service_restart``,
no miss). The default stays ``final``. The process tests run :mod:`tests.memory.service_driver` (the real service loop,
a scripted pass); no model, network or key.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import attention
from unify.memory import writer_service as ws

DRIVER = str(Path(__file__).with_name("service_driver.py"))
SWITCHES = {
    "UNIFY_MEMORY": "on",
    "UNIFY_MEMORY_CAP_USD": "100",
    "DRIVER_PASS_WALL_S": "60",
}


@pytest.fixture
def svc(tmp_path, monkeypatch):
    home = tmp_path / "home"
    out = tmp_path / "out"
    for k, v in SWITCHES.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DRIVER_OUT", str(out))
    from unify.memory.integration.consolidate import open_stores
    from unify.memory.integration.paths import Paths

    open_stores(Paths.under(home))
    yield SimpleNamespace(home=home, out=out)
    doc = ws.read_state(home)
    if doc and ws.owns(doc):
        ws.stop_service(home, "operator", "the test ended", grace_s=0.5, term_s=2)


def _ensure(home: Path) -> ws.ServiceState:
    return ws.ensure_service(home, settings=SimpleNamespace(**SWITCHES), argv_prefix=[sys.executable, DRIVER], wait_s=60)


def _until(check, timeout=60.0, step=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = check()
        if got:
            return got
        time.sleep(step)
    raise AssertionError("timed out")


def _open_entries(home: Path, n: int, first_seq: int = 1000) -> list[str]:
    ledger = attention.Ledger(home)
    return [ledger.create("message", attention.Refs(record_seq=first_seq + i), first_seq + i) for i in range(n)]


def _ops(home: Path, op: str) -> list[dict]:
    return [r for r in attention.Ledger(home).rows() if r.get("op") == op]


# --- the stop kinds ------------------------------------------------------------------------------------------------------


def test_suspend_is_a_stop_kind_that_marks_nothing():
    assert "suspend" in ws.STOP_BY and ws.NOT_REACHED_BY.get("suspend") is None
    assert ws.NOT_REACHED_BY["drain"] == "run_ended"


def test_the_drain_refuses_an_unknown_end(tmp_path):
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths

    with pytest.raises(ValueError):
        async_pass.drain(Paths.under(tmp_path / "home"), end="pause")


def test_the_drain_cli_defaults_to_final_and_takes_suspend(monkeypatch, tmp_path, capsys):
    from unify.memory.integration import async_pass

    seen = []
    monkeypatch.setattr(async_pass, "drain", lambda paths, **kw: seen.append(kw["end"]) or {"terminated": True})
    assert async_pass.main(["drain", "--home", str(tmp_path)]) == 0
    assert async_pass.main(["drain", "--home", str(tmp_path), "--end", "suspend"]) == 0
    with pytest.raises(SystemExit):
        async_pass.main(["drain", "--home", str(tmp_path), "--end", "pause"])
    assert seen == ["final", "suspend"]


# --- the service: suspend keeps the entries owed; the next start hands them again --------------------------------------


def test_a_suspend_drain_stops_the_service_verified_and_keeps_every_entry_owed(svc, monkeypatch):
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths

    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("DRIVER_PASS_WALL_S", "2")
    ids = _open_entries(svc.home, 6)
    assert _ensure(svc.home).status == "started"
    doc = _until(lambda: (d := ws.read_state(svc.home)) and d.get("state") == "running" and d)
    report = async_pass.drain(Paths.under(svc.home), end="suspend")
    assert report["end"] == "suspend"
    assert report["service"]["verified"] is True and report["terminated"] is True
    assert not ws.owns(doc) and not ws.group_pids(doc["pgid"], since=doc["proc_start"])
    # nothing is marked not reached, and every entry is still owed (open, or held by the stopped pass)
    assert _ops(svc.home, "not_reached") == []
    entries = attention.Ledger(svc.home).entries()
    assert all(entries[i].active for i in ids)
    assert report["service"]["kept_owed"] == len(ids)
    # the suspend's stop marker is set aside, recorded; a request may start the next stage's service
    assert ws.stopping_marker(svc.home) is None and report["service"]["marker_cleared"] is True
    assert any(r.get("event") == "stop-cleared" and r.get("previous_by") == "suspend" for r in ws.events(svc.home))


def test_after_a_suspend_the_next_start_hands_the_same_entries_again_without_a_miss(svc, monkeypatch):
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths

    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("DRIVER_PASS_WALL_S", "2")
    ids = _open_entries(svc.home, 3)
    assert _ensure(svc.home).status == "started"
    _until(lambda: _ops(svc.home, "hand"))
    async_pass.drain(Paths.under(svc.home), end="suspend")
    held = {r["entry"] for r in _ops(svc.home, "hand")}
    # the next stage: the turns run at once now, so the pass decides what it is handed
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "0")
    assert _ensure(svc.home).status == "started"
    _until(lambda: {r["entry"] for r in _ops(svc.home, "close")} >= set(ids), timeout=90)
    entries = attention.Ledger(svc.home).entries()
    assert all(entries[i].state == "closed" for i in ids)
    # an entry the stopped pass held went back without a miss
    assert all(entries[i].misses == 0 for i in held)
    assert {r.get("why") for r in _ops(svc.home, "return") if r["entry"] in held} <= {"service_restart"}
    assert _ops(svc.home, "not_reached") == []


def test_a_final_drain_still_marks_what_is_owed_not_reached_and_refuses_a_later_start(svc, monkeypatch):
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths

    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("DRIVER_PASS_WALL_S", "2")
    _open_entries(svc.home, 2)
    assert _ensure(svc.home).status == "started"
    _until(lambda: (d := ws.read_state(svc.home)) and d.get("state") == "running" and d)
    report = async_pass.drain(Paths.under(svc.home))   # the default: final
    assert report["end"] == "final" and report["service"]["verified"] is True
    states = {e.state for e in attention.Ledger(svc.home).entries().values()}
    assert states <= {"not_reached", "closed", "unreadable"}
    assert {r.get("why") for r in _ops(svc.home, "not_reached")} == {"run_ended"}
    assert _ensure(svc.home).status == "refused"


def test_suspend_then_final_marks_only_at_the_final_end(svc, monkeypatch):
    """The qualify row's sequence (q_ledger_suspend): entries survive a suspend, are re-handed, and only the final end
    marks what is still owed."""
    from unify.memory.integration import async_pass
    from unify.memory.integration.paths import Paths

    monkeypatch.setenv("DRIVER_TURNS", "1")
    monkeypatch.setenv("DRIVER_TURN_SLEEP", "3600")
    monkeypatch.setenv("DRIVER_PASS_WALL_S", "2")
    ids = _open_entries(svc.home, 2)
    assert _ensure(svc.home).status == "started"
    _until(lambda: _ops(svc.home, "hand"))
    async_pass.drain(Paths.under(svc.home), end="suspend")
    assert _ops(svc.home, "not_reached") == []
    assert _ensure(svc.home).status == "started"
    _until(lambda: len(_ops(svc.home, "hand")) >= 2)
    async_pass.drain(Paths.under(svc.home), end="final")
    nr = {r["entry"] for r in _ops(svc.home, "not_reached")}
    entries = attention.Ledger(svc.home).entries()
    assert all(entries[i].state in ("closed", "not_reached") for i in ids) and nr
