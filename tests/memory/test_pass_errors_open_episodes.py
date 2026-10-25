"""Rev 2, item 5: a writer pass never asks for an episode that is still running, and no failed pass is silent.

An entry names episodes (a message posted mid-episode names its sender's running episode). It is handed only once each
can be read: a recorded (indexed) episode. Until then it waits ``open`` (one ``wait`` row, no hand, so no miss). A pass that does fail says why on its pass row: the class of
what it raised and where (``file:line:function``), never the message. No model, network or key.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import attention, metering
from unify.memory import writer_service as ws
from unify.memory.integration import async_pass as ap
from tests.memory import store_reads

HERE = Path(__file__).name


def _resolver(recorded):
    return attention.episode_resolver(lambda e: e in recorded)


def _ready(resolve):
    return lambda e: attention.entry_ready(e, resolve)


def _ops(led, op):
    return [r for r in led.rows() if r.get("op") == op]


# --- the ledger: an entry waits for its episode, without a miss --------------------------------------------------------


def test_a_message_naming_a_running_episode_waits_without_a_miss_and_is_handed_once_recorded(tmp_path):
    led = attention.Ledger(tmp_path / "home")
    eid = led.create("message", attention.Refs(episodes=("ep-run",), record_seq=7), 7)
    recorded: set[str] = set()
    for pid in ("p1", "p2", "p3"):  # the service asks each turn, with a fresh resolver
        assert led.hand(pid, 4, ready=_ready(_resolver(recorded))) == []
    e = led.get(eid)
    assert e.state == "open" and e.misses == 0 and e.hand_count == 0
    assert _ops(led, "hand") == [] and _ops(led, "return") == []
    (wait,) = _ops(led, "wait")  # one wait row, however many turns asked
    assert wait["entry"] == eid and wait["episodes"] == ["ep-run"] and wait["why"] == "episode_not_recorded"
    recorded.add("ep-run")
    resolve = _resolver(recorded)
    handed = led.hand("p4", 4, ready=_ready(resolve))
    assert [x.entry for x in handed] == [eid]
    assert attention.pass_episodes(handed, resolve=resolve) == ["ep-run"]


def test_ready_entries_are_handed_past_a_waiting_one(tmp_path):
    led = attention.Ledger(tmp_path / "home")
    waiting = led.create("message", attention.Refs(episodes=("ep-run",), record_seq=1), 1)
    rest = [led.create("message", attention.Refs(record_seq=s), s) for s in (2, 3)]
    rest.append(led.create("message", attention.Refs(episodes=("ep-done",), record_seq=4), 4))
    handed = led.hand("p1", 4, ready=_ready(_resolver({"ep-done"})))
    # every ready entry, in the ledger's own hand order (entries made in the same second are ordered by their ids)
    assert sorted(x.entry for x in handed) == sorted(rest)
    assert led.get(waiting).state == "open" and led.get(waiting).misses == 0


def test_without_a_readiness_check_the_hand_is_as_before(tmp_path):
    led = attention.Ledger(tmp_path / "home")
    eid = led.create("message", attention.Refs(episodes=("ep-run",), record_seq=1), 1)
    assert [x.entry for x in led.hand("p1", 4)] == [eid] and _ops(led, "wait") == []
    assert attention.pass_episodes(led.in_pass("p1")) == ["ep-run"]


def test_what_still_waits_at_the_run_end_is_not_reached(tmp_path):
    led = attention.Ledger(tmp_path / "home")
    eid = led.create("message", attention.Refs(episodes=("ep-run",), record_seq=1), 1)
    led.hand("p1", 4, ready=_ready(_resolver(set())))
    assert led.mark_not_reached("run_ended") == [eid]
    assert led.get(eid).state == "not_reached" and store_reads.guaranteed(led.report())


def test_a_failed_index_read_says_not_recorded_yet():
    def broken(_eid):
        raise RuntimeError("database is locked")

    assert attention.indexed_by(SimpleNamespace(episode_exists=broken))("ep") is False
    assert attention.indexed_by(SimpleNamespace(episode_exists=lambda e: e == "ep"))("ep") is True


# --- the attention pass: no episode that cannot be read is asked for ---------------------------------------------------


def _pass(tmp_path, monkeypatch, refs, *, writer_run=None):
    """One run_attention_pass over a message entry with *refs*, handed without a readiness check (as a pass resumed
    from before it was handed so); the recorded episode ``e1`` exists. Returns (the episodes the writer was asked
    for, the report)."""
    from tests.memory.integration import test_consolidate as tc
    from tests.memory.test_service_spend import W
    from unify.memory.integration import consolidate
    from unify.memory.integration.async_pass import Supervisor
    from unify.memory.reconcile import unavailable

    stores = tc._stores(tmp_path)
    tc._record(stores, "e1")
    home = stores.paths.home
    ledger = attention.Ledger(home)
    ledger.create("message", refs, 1)
    pid = "ef" * 8
    handed = ledger.hand(pid, 4)
    asked: list[list[str]] = []

    class Spy(consolidate.WriterPass):
        async def run(self, req, pass_id):
            asked.append(list(req.episodes))
            if writer_run is not None:
                return await writer_run(self, req, pass_id)
            return await super().run(req, pass_id)

    monkeypatch.setattr(consolidate, "WriterPass", Spy)

    async def recon(self, **kw):
        return unavailable(W)

    monkeypatch.setattr(Supervisor, "reconcile", recon)
    folder = ws.service_dir(home) / ws.TURNS_DIR / pid
    folder.mkdir(parents=True, exist_ok=True)
    settings = tc._settings()
    settings.UNIFY_MEMORY_CAP_USD = "30"
    report = asyncio.run(
        consolidate.run_attention_pass(
            home, pid, handed, resume=False, folder=folder, journal_start=None, effort="low",
            settings=settings, turn_factory=lambda: tc.FakeWriter()("m", "low"),
        ),
    )  # fmt: skip
    return asked, report


def test_a_pass_never_asks_for_a_running_episode(tmp_path, monkeypatch):
    """Rev 1's failure (o rdnoep: 36 of 36 error passes): the message named its sender's running episode, the
    writer asked the index for it and the pass died on KeyError before its first call."""
    asked, report = _pass(tmp_path, monkeypatch, attention.Refs(episodes=("e-running", "e1"), record_seq=1))
    assert asked and all(a == ["e1"] for a in asked)
    assert report.ended != "error" and report.error_class is None


# --- no failed pass is silent ------------------------------------------------------------------------------------------


def test_error_of_names_the_class_and_innermost_frame_never_the_message():
    def deep():
        raise KeyError("ep-secret recorded text")

    try:
        deep()
    except KeyError as exc:
        got = ap.error_of(exc)
    assert got["class"] == "KeyError" and got["where"].startswith(f"{HERE}:") and got["where"].endswith(":deep")
    assert "secret" not in json.dumps(got)
    assert ap.error_of(ValueError("never raised")) == {"class": "ValueError", "where": "unknown"}


def test_the_supervisor_keeps_what_the_pass_raised():
    async def broken():
        raise KeyError("ep-secret")

    async def go():
        return await ap.Supervisor(60, asyncio.Event())(broken)

    got = asyncio.run(go())
    assert got.ended == "error" and got.error["class"] == "KeyError" and got.error["where"].endswith(":broken")
    assert "secret" not in json.dumps(got.error)


def test_a_writer_that_raises_puts_its_class_on_the_report(tmp_path, monkeypatch):
    async def raising(self, req, pass_id):
        raise KeyError("ep-secret")

    _asked, report = _pass(tmp_path, monkeypatch, attention.Refs(record_seq=1), writer_run=raising)
    assert report.ended == "error" and report.error_class == "KeyError" and report.error_where.endswith(":raising")
    row = metering.pass_row(tmp_path, report)
    assert row["error_class"] == "KeyError" and "secret" not in json.dumps(row)


@pytest.mark.parametrize("ended", ap._ENDED)
@pytest.mark.parametrize("cls", [None, "", "KeyError"])
@pytest.mark.parametrize("where", [None, "", "x.py:1:f"])
def test_every_error_row_names_a_cause_and_no_other_row_has_one(tmp_path, ended, cls, where):
    report = SimpleNamespace(
        pass_id="p", ended=ended, outcome=None, seconds=0.0, reconciled=None, error_class=cls, error_where=where,
        usd_known="0", unknown_calls=0, worst_case_booked="0", killed_turns=[],
    )
    row = metering.pass_row(tmp_path, report)
    if ended == "error":
        assert row["error_class"] == (cls or "unrecorded") and row["error_where"] == (where or "unrecorded")
    else:
        assert "error_class" not in row and "error_where" not in row


def test_a_service_pass_that_raises_writes_its_row_with_the_cause(tmp_path, monkeypatch):
    """The service's raise path wrote no pass row before: the request reports and the stage checks saw nothing."""
    from tests.memory import test_writer_service as T
    from tests.memory.test_service_spend import _step

    home = tmp_path / "home"
    T._open_entry(home)

    async def broken(home_, pass_id, handed, *, resume, folder, journal_start, effort, settings, stop=None):
        raise RuntimeError("the pass broke on recorded text")

    got = _step(home, broken, monkeypatch)
    assert got["ended"] == "error"
    rows = [json.loads(x) for x in metering.results_path(home).read_text().splitlines() if x.strip()]
    (row,) = [r for r in rows if r["pass_id"] == got["pass_id"]]
    assert row["ended"] == "error" and row["error_class"] == "RuntimeError" and row["error_where"].endswith(":broken")
    assert "recorded text" not in json.dumps(row)


def test_the_service_hands_nothing_while_its_only_entry_waits(tmp_path, monkeypatch):
    from tests.memory import test_writer_service as T
    from tests.memory.test_service_spend import _step

    home = tmp_path / "home"
    led = attention.Ledger(home)
    eid = led.create("message", attention.Refs(episodes=("ep-run",), record_seq=1), 1)
    for _ in range(2):
        got = _step(home, T._no_pass, monkeypatch)
        assert got["pass_id"] is None and got["why_idle"] == "nothing_handed"
    assert [r["entry"] for r in _ops(led, "wait")] == [eid] and led.get(eid).misses == 0
