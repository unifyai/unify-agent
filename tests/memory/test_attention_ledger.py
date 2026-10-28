"""The attention ledger (``UNIFY_MEMORY_ATTENTION_LEDGER``): every entry that needs the writer is read at least once.

Each kind fires from a harness record and from nothing else; duplicates append; reads are verified from the pass's own
read evidence, never from what the writer says; one decision may close several entries, each read; an unread entry
comes back, goes alone, then is unreadable; the age backstop keeps every kind moving; at stop, cap and run end every
entry is closed, unreadable or not reached; late evidence reopens what it bears on. No model is called.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from unify.memory import attention as at
from unify.memory.attention import (
    EpisodeCalls,
    GateRefusal,
    LateSignal,
    Ledger,
    LedgerError,
    PassReads,
    Refs,
    SettledFork,
    UseFailure,
)
from tests.memory import store_reads


class Clock:
    def __init__(self, start: str = "2026-10-20T00:00:00.000Z") -> None:
        self.t = at.parse_time(start)

    def __call__(self) -> dt.datetime:
        return self.t

    def tick(self, seconds: float = 1.0) -> None:
        self.t = self.t + dt.timedelta(seconds=seconds)


@dataclass
class Fake:
    """Synthetic harness records (each list is one kind of record)."""

    forks: list = field(default_factory=list)
    captures: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    late: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    items_of: dict = field(default_factory=dict)
    refusals: list = field(default_factory=list)
    record: list = field(default_factory=list)
    deliveries: dict = field(default_factory=dict)

    def settled_forks(self):
        return list(self.forks)

    def capture_rows(self):
        return dict(self.captures)

    def use_failures(self):
        return list(self.failures)

    def late_signals(self):
        return list(self.late)

    def episode_calls(self):
        return list(self.calls)

    def episode_items(self, episode):
        return tuple(self.items_of.get(episode, ()))

    def gate_refusals(self):
        return list(self.refusals)

    def record_entries(self):
        return list(self.record)

    def record_deliveries(self):
        return dict(self.deliveries)


def _ledger(tmp_path, clock=None):
    return Ledger(tmp_path / "home", clock=clock or Clock())


def _fork(name, *, output=True, procedure=True):
    return SettledFork(name, name, output, procedure, None)


def _cap(name, cid, **own):
    check = {"ran": True, "error": None, "refused": False, "deterministic": True, "value_match": "not_applicable",
             "calls": {"mode": "replay", "matched": 2, "differed": 0, "missing": 0, "unrecorded": 0}}  # fmt: skip
    check.update(own)
    return {"v": 2, "capture_id": cid, "episode": name, "own_check": check, "outcome": "none"}


def _kinds(entries):
    return sorted(e.kind for e in entries)


def _reads(paths=(), episodes=None):
    return PassReads(frozenset(paths), {k: frozenset(v) for k, v in (episodes or {}).items()})


# --- 1. each kind fires from its record, and not without one --------------------------------------------------------


def test_each_kind_fires_from_its_record_and_not_without_one(tmp_path):
    empty = _ledger(tmp_path / "a")
    assert empty.sync(Fake()) == []
    assert empty.entries() == {}

    src = Fake(
        forks=[_fork("e1"), _fork("e2", procedure=False), _fork("e3", output=False)],
        captures={"e1": _cap("e1", "c" * 16, error="ValueError")},
        failures=[UseFailure("e4", "memory.pkg.mod:total", "error")],
        late=[LateSignal("s1", "e5", "2026-10-20T00:00:05.000Z")],
        calls=[EpisodeCalls("e6", "2026-10-19T00:00:00Z", ("notes/a/b.md",)),
               EpisodeCalls("e7", "2026-10-19T00:01:00Z", ("notes/a/b.md",))],  # fmt: skip
        refusals=[GateRefusal("p1", ("e1",), ("memory.pkg.mod:f",), False)],
        record=[{"seq": 3, "author": "actor-3f9a", "role": "actor", "kind": "post", "mentions": ["writer"],
                 "episode": "e8", "refs": [], "ts": "2026-10-20T00:00:01.000Z"}],  # fmt: skip
    )
    led = _ledger(tmp_path / "b")
    made = led.sync(src)
    assert _kinds(made) == sorted(
        [
            "new_experience",  # e1, its capture row present
            "new_experience",  # e2: no procedure, so no capture to wait for
            # rev 2: e1's capture row says its own check errored, and that makes no entry (none runs now)
            "use_went_badly",
            "late_evidence",
            "repeated_use",  # two new episodes called the note
            "deferred",  # the gate refused a draft item
            "message",  # an entry mentioning the writer
        ],
    )
    # e3 left no output: no entry; a second sync over the same records makes nothing new
    assert not any("e3" in e.refs.episodes for e in made)
    assert led.sync(src) == []
    # a fork with a procedure waits for its capture row
    led2 = _ledger(tmp_path / "c")
    assert led2.sync(Fake(forks=[_fork("e9")])) == []
    [n] = led2.sync(Fake(forks=[_fork("e9")], captures={"e9": _cap("e9", "d" * 16)}))
    assert n.kind == "new_experience" and n.refs.captures == ("d" * 16,)
    # a message by the writer itself, or one addressed to nobody it reads, makes nothing
    led3 = _ledger(tmp_path / "d")
    assert led3.sync(Fake(record=[{"seq": 1, "author": "writer", "role": "writer", "mentions": ["writer"]},
                                  {"seq": 2, "author": "fork-3f9a", "role": "fork", "mentions": ["librarian"]}])) == []  # fmt: skip
    # a reply to a writer entry is a message to it
    [m] = led3.sync(Fake(record=[{"seq": 1, "author": "writer", "role": "writer", "mentions": ["fork-3f9a"]},
                                 {"seq": 2, "author": "fork-3f9a", "role": "fork", "mentions": [], "reply_to": 1}]))  # fmt: skip
    assert m.kind == "message" and m.refs.record_seq == 2


def test_rows_hold_ids_and_kinds_only_and_are_appended(tmp_path):
    led = _ledger(tmp_path)
    led.sync(Fake(forks=[_fork("e1", procedure=False)]))
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert rows and all(r["v"] == 1 for r in rows)
    [create] = rows
    assert set(create) == {"v", "op", "entry", "kind", "at", "refs"}
    assert create["at"].endswith("Z") and len(create["entry"]) == 16
    # a line cut by a crash is skipped on load and the next append goes on
    with open(led.path, "a") as fh:
        fh.write('{"op": "create", "entry": "abc')
    assert len(led.entries()) == 1


# --- 2. duplicates append ----------------------------------------------------------------------------------------------


def test_a_byte_identical_capture_appends_to_the_open_entry(tmp_path):
    led = _ledger(tmp_path)
    cid = "a" * 16
    led.sync(Fake(forks=[_fork("e1")], captures={"e1": _cap("e1", cid)}))
    made = led.sync(Fake(forks=[_fork("e1"), _fork("e2")], captures={"e1": _cap("e1", cid), "e2": _cap("e2", cid)}))
    assert made == []
    [e] = led.entries().values()
    assert e.refs.episodes == ("e1", "e2") and e.refs.captures == (cid,)
    # idempotent: syncing again appends nothing more
    led.sync(Fake(forks=[_fork("e1"), _fork("e2")], captures={"e1": _cap("e1", cid), "e2": _cap("e2", cid)}))
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert [r["op"] for r in rows] == ["create", "append_ref"]
    # once closed, the same capture from a new episode is counted on the closed entry (an occurrence row), never a new
    # entry: its outcome is not contrary
    led.hand("p1", 4)
    reads = _reads(["/inputs/staging/e1/summary.md", f"/inputs/captures/{cid}/procedure.py"])
    led.decide("p1", e.entry, "nothing_to_keep", reason="one fixed answer", reads=reads,
               decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    assert led.sync(Fake(forks=[_fork("e3")], captures={"e3": _cap("e3", cid)})) == []
    assert len(led.entries()) == 1
    rows = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert rows[-1]["op"] == "occurrence" and rows[-1]["entry"] == e.entry


# --- 3. read verification uses the logs only ------------------------------------------------------------------------


def test_reads_are_verified_from_the_logs_never_from_the_writers_words(tmp_path):
    led = _ledger(tmp_path)
    cid = "b" * 16
    led.sync(Fake(forks=[_fork("e1")], captures={"e1": _cap("e1", cid)}))
    [e] = led.hand("p1", 4)
    # the writer saying it read the entry, anywhere, is no read: only the pass's own read evidence counts
    claim = _reads(["/memory/notes/x.md"])
    assert led.verify_reads("p1", claim, []) == set()
    with pytest.raises(LedgerError, match="not read yet"):
        led.decide("p1", e.entry, "nothing_to_keep", reason=f"I read {e.entry} fully", reads=claim,
                   decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    # the summary alone is not enough: the procedure is owed too
    assert led.verify_reads("p1", _reads(["/inputs/staging/e1/summary.md"]), []) == set()
    # both opened (a tool read row or a box open row): verified, with its parts and share
    both = PassReads.collect(
        tool_events=[{"status": "ok", "paths": [{"path": "/inputs/staging/e1/summary.md", "mode": "read"}]}],
        box_opens=[{"path": f"/inputs/captures/{cid}/procedure.py", "mode": "r"}],
    )
    assert led.verify_reads("p1", both, []) == {e.entry}
    got = led.get(e.entry)
    assert got.state == "read" and set(got.parts_read) == {"summary", "procedure.py"} and got.share == "0.67"
    # a failed tool call, a write and a listing are never reads
    none = PassReads.collect(
        tool_events=[{"status": "refused", "paths": [{"path": "/inputs/staging/e1/summary.md", "mode": "read"}]},
                     {"status": "ok", "paths": [{"path": "/inputs/staging/e1/summary.md", "mode": "list"}]}],
        box_opens=[{"path": f"/inputs/captures/{cid}/procedure.py", "mode": "w"}],
    )  # fmt: skip
    assert none.paths == frozenset()


def test_a_message_is_read_when_its_entry_was_delivered_to_the_pass(tmp_path):
    led = _ledger(tmp_path)
    led.sync(Fake(record=[{"seq": 7, "author": "actor-1a2b", "role": "actor", "mentions": ["writer"],
                           "episode": "e1", "refs": ["item:notes/a/b.md"]}]))  # fmt: skip
    [m] = led.hand("p1", 4)
    assert m.kind == "message"
    assert led.verify_reads("p1", _reads(), [6, 8]) == set()
    assert led.verify_reads("p1", _reads(), [7]) == {m.entry}


def test_delivered_seqs_read_the_record_delivery_log(tmp_path):
    home = tmp_path / "home"
    d = home / "memory-state" / "agents"
    d.mkdir(parents=True)
    rows = [
        {"v": 1, "agent": "writer", "role": "writer", "seqs": [3, 4], "via": "block", "pass_id": "p1"},
        {"v": 1, "agent": "writer", "role": "writer", "seqs": [9], "via": "messages", "pass_id": "p2"},
        {"v": 1, "agent": "actor-1a2b", "role": "actor", "seqs": [5], "via": "block", "pass_id": "p1"},
    ]
    (d / "run-record.cursors.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert at.delivered_seqs(home, "p1") == {3, 4}
    assert at.delivered_seqs(home, "p2") == {9}


# --- 4. one decision closes several entries, each read ---------------------------------------------------------------


def test_one_decision_closes_several_read_entries_and_refuses_an_unread_one(tmp_path):
    led = _ledger(tmp_path)
    src = Fake(forks=[_fork(f"e{i}", procedure=False) for i in range(3)])
    led.sync(src)
    handed = led.hand("p1", 4)
    assert len(handed) == 3
    ids = [e.entry for e in handed]
    two = _reads(["/inputs/staging/e0/summary.md", "/inputs/staging/e1/summary.md"])
    dpath = tmp_path / "passes" / "p1" / "decisions.jsonl"
    # one of the three unread: the whole call is refused, saying which and where its material is; nothing changes
    with pytest.raises(LedgerError) as err:
        led.decide("p1", ids, "new_item", items=["memory.pkg.mod:f"], reason="one rule", reads=two, decisions_path=dpath)
    text = str(err.value)
    assert ids[2] in text and "/inputs/staging/e2/" in text and "nothing was decided" in text
    assert all(led.get(i).state == "handed" for i in ids) and not dpath.exists()
    # the two read ones close together
    res = led.decide("p1", ids[:2], "new_item", items=["memory.pkg.mod:f"], reason="one rule covers both", reads=two,
                     decisions_path=dpath)  # fmt: skip
    assert res.reason_ref == "pass:p1/decisions.jsonl#0" and res.items == ("memory.pkg.mod:f",)
    assert [led.get(i).state for i in ids] == ["closed", "closed", "handed"]
    # the reason text is in the pass's decisions file, never in the ledger
    assert "one rule covers both" in dpath.read_text() and "one rule covers both" not in led.path.read_text()
    # instructive refusals: unknown id, an entry of another pass, a decision needing an item, an unknown decision
    with pytest.raises(LedgerError, match="no entry 'ffff"):
        led.decide("p1", "ffff", "note", reason="x", reads=two, decisions_path=dpath)
    with pytest.raises(LedgerError, match="names the item it rests on"):
        led.decide("p1", ids[2], "merged_into", reason="x", reads=two, decisions_path=dpath)
    with pytest.raises(LedgerError, match="is not one of"):
        led.decide("p1", ids[2], "keep", reason="x", reads=two, decisions_path=dpath)
    with pytest.raises(LedgerError, match="is not in this pass"):
        led.decide("p2", ids[2], "note", reason="x", reads=_reads(["/inputs/staging/e2/"]), decisions_path=dpath)
    with pytest.raises(LedgerError, match="reason says why"):
        led.decide("p1", ids[2], "note", reason=" ", reads=two, decisions_path=dpath)
    # a defer closes the entry and creates a deferred entry, owed its own read later
    res = led.decide("p1", ids[2], "defer", reason="needs a second episode", reads=_reads(["/inputs/staging/e2/x.md"]),
                     decisions_path=dpath)  # fmt: skip
    d = led.get(res.deferred)
    assert d.kind == "deferred" and d.state == "open" and d.refs.passes == ("p1",)


# --- 5. return, alone, unreadable ---------------------------------------------------------------------------------------


def test_an_unread_entry_comes_back_then_goes_alone_then_is_unreadable(tmp_path):
    led = _ledger(tmp_path)
    led.sync(Fake(forks=[_fork(f"e{i}", procedure=False) for i in range(3)]))
    target = led.hand("p1", 4)[0].entry
    assert set(led.return_unread("p1")) == {e.entry for e in led.entries().values()}
    assert led.get(target).misses == 1 and led.get(target).state == "open"
    led.hand("p2", 4)
    led.return_unread("p2")
    assert led.get(target).misses == 2
    # after two misses it goes alone
    [alone] = led.hand("p3", 4)
    assert alone.misses >= 2 and alone.hand_count == 3
    led.return_unread("p3")
    gone = led.get(alone.entry)
    assert gone.state == "unreadable" and gone.why == "pass:p3/read-log.json"
    # read but undecided: back without a miss
    led2 = _ledger(tmp_path / "x")
    led2.sync(Fake(forks=[_fork("e1", procedure=False)]))
    [e] = led2.hand("p1", 4)
    led2.verify_reads("p1", _reads(["/inputs/staging/e1/summary.md"]), [])
    led2.return_unread("p1")
    back = led2.get(e.entry)
    assert back.state == "open" and back.misses == 0 and back.why == "read_not_decided"
    # a restart returns what was handed to a pass that is not live, without a miss
    led2.hand("p2", 4)
    assert led2.return_orphans(live_pass_ids=["p9"]) == [e.entry]
    assert led2.get(e.entry).why == "service_restart" and led2.get(e.entry).misses == 0
    # a pass the service was asked to stop during (a suspend or a drain) returns its entries as "suspend", no miss
    led2.hand("p3", 4)
    assert led2.return_unread("p3", stopped=True) == [e.entry]
    assert led2.get(e.entry).why == "suspend" and led2.get(e.entry).misses == 0


def test_hand_order_is_bad_news_first_oldest_first_and_k_entries(tmp_path):
    clock = Clock()
    led = _ledger(tmp_path, clock)
    kinds = [
        Fake(refusals=[GateRefusal("p0", (), ("memory.a.b:c",), False)]),
        Fake(forks=[_fork("e1", procedure=False)]),
        Fake(calls=[EpisodeCalls("e2", None, ("notes/x/y.md",)), EpisodeCalls("e3", None, ("notes/x/y.md",))]),
        Fake(record=[{"seq": 1, "author": "actor-1", "role": "actor", "mentions": ["writer"]}]),
        Fake(failures=[UseFailure("e4", "memory.a.b:c", "error")]),
        Fake(late=[LateSignal("s1", "e5", None)]),
        Fake(forks=[_fork("e6")], captures={"e6": _cap("e6", "f" * 16, error="TypeError")}),
    ]
    for src in kinds:
        led.sync(src)
        clock.tick()
    first = led.hand("p1", 4)
    # rev 2: e6's failing own check makes no entry; its fork is a new experience
    assert [e.kind for e in first] == ["late_evidence", "use_went_badly", "message", "new_experience"]
    rest = led.hand("p2", 4)
    # the oldest open entry (the deferred one) is older than every age at hand so far: the backstop hands it first
    assert [e.kind for e in rest] == ["deferred", "new_experience", "repeated_use"]
    with pytest.raises(LedgerError, match="positive whole number"):
        led.hand("p3", 0)


def test_take_pulls_an_open_entry_into_the_pass(tmp_path):
    led = _ledger(tmp_path)
    led.sync(Fake(forks=[_fork(f"e{i}", procedure=False) for i in range(6)]))
    handed = led.hand("p1", 4)
    others = [e for e in led.entries().values() if e.state == "open"]
    assert len(others) == 2
    got = led.take("p1", others[0].entry)
    assert got.taken and got.material and led.get(others[0].entry).pass_id == "p1"
    assert led.take("p1", others[0].entry).detail.endswith("already in this pass")
    nope = led.take("p2", handed[0].entry)
    assert not nope.taken and "not open" in nope.detail
    assert not led.take("p1", "0" * 16).taken
    listing = led.open_entries()
    assert [o.entry for o in listing] == [others[1].entry] and listing[0].goal


# --- 6. starvation ---------------------------------------------------------------------------------------------------


def test_new_experience_is_still_read_under_constant_bad_news(tmp_path):
    clock = Clock()
    led = _ledger(tmp_path, clock)
    led.sync(Fake(forks=[_fork("old", procedure=False)]))
    old = next(iter(led.entries()))
    read_in: str | None = None
    for p in range(40):
        clock.tick(60)
        # four new failed uses arrive every pass: without the backstop they would fill every pass
        led.sync(Fake(failures=[UseFailure(f"e{p}-{i}", "memory.a.b:c", "error") for i in range(4)]))
        handed = led.hand(f"p{p}", 4)
        if any(e.entry == old for e in handed):
            read_in = f"p{p}"
            break
        reads = _reads([f"/inputs/attention/{e.entry}/use.json" for e in handed] + ["/memory/memory/a/b.py"])
        led.decide(f"p{p}", [e.entry for e in handed], "nothing_to_keep", reason="seen", reads=reads,
                   decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    assert read_in is not None and int(read_in[1:]) <= 2


# --- 7. the guarantee ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("why", ["spend_cap", "run_ended", "service_stopped", "lifetime"])
def test_every_entry_ends_closed_unreadable_or_not_reached(tmp_path, why):
    led = _ledger(tmp_path)
    led.sync(Fake(forks=[_fork(f"e{i}", procedure=False) for i in range(6)]))
    [a, b, c, d] = led.hand("p1", 4)
    led.decide("p1", a.entry, "note", reason="kept as a note",
               reads=_reads([f"/inputs/staging/{a.refs.episodes[0]}/summary.md"]),
               decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    led.return_unread("p1")
    led.hand("p2", 4)
    report = led.report()
    assert not store_reads.guaranteed(report)
    led.return_unread("p2")
    left = led.mark_not_reached(why)
    report = led.report()
    assert store_reads.guaranteed(report) and left
    ne = report.kinds["new_experience"]
    assert ne.created == 6 and ne.closed + ne.unreadable + ne.not_reached == 6 and ne.open == 0
    assert report.not_reached_why == {why: ne.not_reached}
    assert any("new_experience: 6 /" in line for line in report.lines())
    with pytest.raises(LedgerError):
        led.mark_not_reached("bored")


def test_the_report_gives_the_creation_to_read_delay(tmp_path):
    clock = Clock()
    led = _ledger(tmp_path, clock)
    led.sync(Fake(forks=[_fork("e1", procedure=False), _fork("e2", procedure=False)]))
    led.hand("p1", 4)
    clock.tick(30)
    led.verify_reads("p1", _reads(["/inputs/staging/e1/summary.md"]), [])
    clock.tick(30)
    led.verify_reads("p1", _reads(["/inputs/staging/e1/summary.md", "/inputs/staging/e2/summary.md"]), [])
    k = led.report().kinds["new_experience"]
    assert (k.read_delay_median_s, k.read_delay_p90_s) == ("45.0", "60.0")


# --- 9. reopening -------------------------------------------------------------------------------------------------------


def test_late_evidence_reopens_the_closed_entries_of_the_items_it_bears_on(tmp_path):
    clock = Clock()
    led = _ledger(tmp_path, clock)
    item = "memory.pkg.mod:total"
    led.sync(Fake(failures=[UseFailure("e1", item, "error")]))
    [u] = led.hand("p1", 4)
    led.decide("p1", u.entry, "changed_item", items=[item], reason="fixed the header case",
               reads=_reads([f"/inputs/attention/{u.entry}/use.json", "/memory/memory/pkg/mod.py"]),
               decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    clock.tick(10)
    # a late signal older than the close reopens nothing
    stale = Fake(late=[LateSignal("s0", "e2", "2026-10-19T00:00:00.000Z")], items_of={"e2": (item,)})
    [l0] = led.sync(stale)
    assert l0.kind == "late_evidence" and led.get(u.entry).state == "closed"
    # a newer one reopens it, naming the cause; the same signal never reopens it twice
    fresh = Fake(late=[LateSignal("s1", "e3", at.now_iso(clock()))], items_of={"e3": (item,)})
    [l1] = led.sync(fresh)
    back = led.get(u.entry)
    assert back.state == "open" and back.reopen_count == 1 and back.why == f"late_evidence:{l1.entry}"
    assert l1.refs.items == (item,)
    led.sync(fresh)
    assert led.get(u.entry).reopen_count == 1
    assert led.report().reopens_by_item == {item: 1}


def test_an_actor_report_on_the_record_is_a_message_and_a_failed_use(tmp_path):
    home = tmp_path / "home"
    led = Ledger(home, clock=Clock())
    entry = {"v": 1, "seq": 12, "ts": "2026-10-20T21:03:11.402Z", "author": "actor-3f9a", "kind": "post",
             "text": "@writer the served item gave a wrong sum here", "mentions": ["writer"], "role": "actor",
             "episode": "3f9a0c1d2e4b5a69", "pass_id": None, "reply_to": None,
             "refs": ["3f9a0c1d2e4b5a69/action:7", "item:notes/invoices/totals.md"]}  # fmt: skip
    src = Fake(record=[entry], failures=[UseFailure("3f9a0c1d2e4b5a69", "notes/invoices/totals.md", "actor_reported",
                                                    (), 12)])  # fmt: skip
    made = led.sync(src)
    assert _kinds(made) == ["message", "use_went_badly"]
    msg = next(e for e in made if e.kind == "message")
    assert msg.refs.record_seq == 12 and msg.refs.items == ("notes/invoices/totals.md",)
    assert msg.refs.episodes == ("3f9a0c1d2e4b5a69",)
    # the home's sources read an actor's entry to the writer (the role and the writer mention); which items it is about
    # comes from its item refs, else from the library names it gives (test_harmful_items_are_revisited.py)
    assert at._actor_to_writer(entry)
    assert at._actor_to_writer({**entry, "refs": ["3f9a0c1d2e4b5a69/action:7"]})
    assert not at._actor_to_writer({**entry, "role": "fork", "author": "fork-3f9a"})
    assert not at._actor_to_writer({**entry, "mentions": []})


def test_material_names_what_each_kind_owes(tmp_path):
    def e(kind, **refs):
        return at.AttentionEntry("a" * 16, kind, "handed", "2026-10-20T00:00:00.000Z", Refs(**refs))

    m = at.material(e("new_experience", episodes=("e1",), captures=("c" * 16,)))
    assert [(p.name, p.required) for p in m] == [("summary", True), ("procedure.py", True), ("episode", False)]
    assert "/inputs/captures/" + "c" * 16 + "/procedure.py" in m[1].paths
    bare = at.material(e("new_experience", episodes=("e1",)))
    assert bare[0].paths == ("/inputs/staging/e1/summary.md",)
    assert at.item_box_paths("memory.pkg.mod:total") == ("/memory/memory/pkg/mod.py",)
    assert at.item_box_paths("notes/a/b.md") == ("/memory/notes/a/b.md",)
    assert at.item_box_paths("capture:" + "c" * 16) == ("/inputs/captures/" + "c" * 16 + "/procedure.py",)
    assert [p.name for p in at.material(e("message", record_seq=4))] == ["entry"]
    assert at.material(e("deferred", passes=("p1",)))[0].paths == (
        "/inputs/attention/" + "a" * 16 + "/note.json",
        "/inputs/drafts/p1/gate.md",
    )
    assert at.pass_episodes([e("new_experience", episodes=("e1",)), e("late_evidence", episodes=("e2",))]) == [
        "e1",
        "e2",
    ]


def test_export_material_writes_ids_and_harness_facts(tmp_path):
    led = _ledger(tmp_path)
    src = Fake(failures=[UseFailure("e1", "notes/a/b.md", "holds_false")], late=[LateSignal("s1", "e2", None)])
    led.sync(src)
    handed = led.hand("p1", 4)
    out = tmp_path / "inputs" / "attention"
    index = at.export_material(handed, out, ledger=led, sources=src, open_listing=led.open_entries())
    assert {r["kind"] for r in index["entries"]} == {"use_went_badly", "late_evidence"}
    u = next(e for e in handed if e.kind == "use_went_badly")
    use = json.loads((out / u.entry / "use.json").read_text())
    assert use["uses"] == [{"episode": "e1", "item": "notes/a/b.md", "why": "holds_false", "signals": [],
                            "record_seq": None}]  # fmt: skip
    assert (out / "index.json").is_file()
    # the pass's read evidence round-trips through its record
    reads = _reads(["/inputs/x"], {"e1": ["request"]})
    at.write_pass_reads(tmp_path / "state" / "passes" / "p1" / at.READ_LOG_FILE, reads)
    assert at.pass_reads(tmp_path / "state", "p1") == reads
    assert at.pass_reads(tmp_path / "state", "p9") == PassReads()


def test_rows_never_carry_content(tmp_path):
    """The ledger holds ids and kinds: no request, summary, record text or reason ever reaches it."""
    led = _ledger(tmp_path)
    secret_text = "the customer's account 1234-5678 wanted a refund"
    src = Fake(record=[{"seq": 1, "author": "actor-1", "role": "actor", "mentions": ["writer"], "text": secret_text,
                        "refs": ["item:notes/a/b.md"], "episode": "e1"}])  # fmt: skip
    led.sync(src)
    [m] = led.hand("p1", 4)
    led.decide("p1", m.entry, "nothing_to_keep", reason=secret_text, reads=_reads(), delivered=[1],
               decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    assert secret_text not in led.path.read_text()


# --- 8. the Phase 0 fixture -----------------------------------------------------------------------------------------------

#: The harness records behind Phase 0's offline ledger of closed cells, per cell, as counts and re-hashed ids
#: (continual-harness-research: PHASE0-ATTENTION-LEDGER-20261020, summary.json sha256/12 a998cedb68cd; built and
#: checked against it by REDESIGN-BUILD-20261020/b2-phase0-fixture/build_fixture.py). Phase 0 observed no late signal.
PHASE0 = Path(__file__).resolve().parent / "ledger_fixtures" / "phase0_ledger.json"


def _read_all(entries) -> PassReads:
    """A pass that read every part of *entries*' material."""
    paths: set = set()
    episodes: dict = {}
    for e in entries:
        for p in at.material(e):
            paths.update(at._norm(x) for x in p.paths)
            for x in p.episodes:
                episodes.setdefault(x, frozenset({"request"}))
    return PassReads(frozenset(paths), episodes)


def _phase0_counts(tmp_path, cell: dict) -> dict:
    """``sync`` over one cell's records: every kind but repeated use from one sync; repeated use call by call in time
    order, each entry read and decided at once (Phase 0 assumed prompt reads)."""
    led = _ledger(tmp_path / "all")
    src = Fake(
        forks=[_fork(f"n{i:05d}", procedure=False) for i in range(cell["N"])],
        captures={f"s{i:05d}": _cap(f"s{i:05d}", f"{i:016x}", error="ValueError") for i in range(cell["S"])},
        failures=[UseFailure(ep, f"memory.u.m{item}:f", "error") for ep, item in cell["U"]],
        refusals=[
            GateRefusal(f"p{j}", (), tuple(f"memory.d{j}.m:f{x}" for x in range(n)), whole)
            for j, (whole, n) in enumerate(cell["D"])
        ],
    )
    counts = {k: 0 for k in at.KINDS}
    for e in led.sync(src):
        counts[e.kind] += 1
    clock = Clock()
    rled = _ledger(tmp_path / "repeated", clock)
    calls = Fake()
    for k, (item, episodes) in enumerate(sorted(cell["R"].items())):
        name = f"memory.r.m{k}:f{item[:6]}"
        for j, ep in enumerate(episodes):
            clock.tick(60)
            calls.calls.append(EpisodeCalls(ep, at.now_iso(clock()), (name,)))
            clock.tick(1)
            made = rled.sync(calls)
            counts["repeated_use"] += len(made)
            if made:
                pid = f"r{k}-{j}"
                handed = rled.hand(pid, len(made))
                assert {e.entry for e in handed} == {e.entry for e in made}
                assert rled.verify_reads(pid, _read_all(handed)) == {e.entry for e in handed}
                rled.close(pid, [e.entry for e in handed], "nothing_to_keep", (), f"pass:{pid}/decisions.jsonl#0")
    return counts


def test_sync_over_the_phase0_cells_reproduces_its_per_kind_counts(tmp_path):
    doc = json.loads(PHASE0.read_text())
    assert doc["from"]["summary.json"] == "a998cedb68cd" and len(doc["cells"]) == 30
    kind_of = {"N": "new_experience", "S": "own_check_failed", "U": "use_went_badly", "R": "repeated_use",
               "D": "deferred", "L": "late_evidence"}  # fmt: skip
    for name, cell in sorted(doc["cells"].items()):
        got = _phase0_counts(tmp_path / name, cell)
        # rev 2: no capture is run, so no own check fails and Phase 0's own-check entries are not made
        want = {kind_of[k]: (0 if k == "S" else n) for k, n in cell["expected"].items()}
        assert {k: got[k] for k in want} == want, name
        assert got["message"] == 0  # Phase 0's builds had no run record


# --- the conformance findings: decisions reach the records, refused landings end, delivered messages are read --------


def _decide(led, tmp_path, pid, entries, decision, items):
    return led.decide(pid, [e.entry for e in entries], decision, items, "the material shows it",
                      reads=_read_all(entries), decisions_path=tmp_path / "passes" / pid / "decisions.jsonl")  # fmt: skip


def test_the_writers_retire_and_merge_decisions_reach_the_item_records(tmp_path):
    from unify.memory import lifecycle as lc
    from unify.memory import serving

    note, old, new = "notes/files/sort-rows.md", "memory.files.sort:rows", "memory.files.sort:rows_by_key"
    clock = Clock()
    led = _ledger(tmp_path, clock)
    src = Fake(failures=[UseFailure("e1", note, "error"), UseFailure("e2", old, "error")])
    made = {e.refs.items[0]: e for e in led.sync(src)}
    clock.tick()
    handed = led.hand("p1", 4)
    assert {e.entry for e in handed} == {made[note].entry, made[old].entry}
    _decide(led, tmp_path, "p1", [made[note]], "retire", [note])
    _decide(led, tmp_path, "p1", [made[old]], "merged_into", [new])
    decided = at.item_decisions(tmp_path / "home")
    assert {i: (d["status"], d["by"]) for i, d in decided.items()} == {
        note: ("retired", None),
        old: ("superseded", new),
        new: ("kept", None),
    }
    assert decided[note]["ref"] == "pass:p1/decisions.jsonl#0"
    base = dict(
        items={note: "note", old: "function", new: "function"},
        facts=[],
        since={note: {"A"}, old: {"A"}, new: {"A"}},
        changed_at={note: "A", old: "A", new: "A"},
        functions={old: "memory.files.sort", new: "memory.files.sort"},
        graph={"memory.files.sort": set()},
        note_uses={note: []},
    )
    recs, _ = lc.next_records(**base, prev={}, writer_decisions=decided)
    assert (recs[note]["status"], recs[note]["status_rule"]) == ("retired", lc.WRITER_RULE)
    assert (recs[old]["status"], recs[old].get("superseded_by")) == ("retired", new)
    assert recs[new]["status"] == "on-trial" and recs[new].get("superseded_by") is None
    # serving withdraws both: the retired note with nothing in its place, the merged function for its replacement
    assert serving.withdrawn_items(recs) == {
        note: {"status": "retired", "by": None, "reason": recs[note]["status_reason"][:200]},
        old: {"status": "superseded", "by": new, "reason": recs[old]["status_reason"][:200]},
    }
    # off (no decisions given), the records are as built
    plain, _ = lc.next_records(**base, prev={})
    assert plain[note]["status"] == "on-trial" and plain[old]["status"] == "on-trial"
    # a later decision that changes the note un-retires it: the latest decision naming an item wins
    src.failures.append(UseFailure("e3", note, "error"))
    [again] = led.sync(src)
    clock.tick()
    led.hand("p2", 4)
    _decide(led, tmp_path, "p2", [again], "changed_item", [note])
    later, _ = lc.next_records(**base, prev=recs, writer_decisions=at.item_decisions(tmp_path / "home"))
    assert later[note]["status"] == "on-trial" and later[old].get("superseded_by") == new


def test_a_landing_the_gate_keeps_refusing_ends_unreadable_after_three_passes(tmp_path):
    item = "notes/files/how.md"
    led = _ledger(tmp_path)
    src = Fake()
    rounds = []
    for k in (1, 2, 3):
        src.refusals.append(GateRefusal(f"p{k}", ("e1",), (item,), k == 3))
        rounds.append(led.sync(src))
    assert [len(r) for r in rounds] == [1, 1, 2]  # the third pass was refused whole: its own entry and the item's
    assert all(e.state == "open" for r in rounds[:2] for e in r)
    states = {e.entry: e for e in led.entries().values()}
    third = [states[e.entry] for e in rounds[2]]
    assert all(e.state == "unreadable" for e in third)
    assert all("refused this landing in 3 passes: p1, p2, p3" in (e.why or "") for e in third)
    # nothing more is handed for it, and the guarantee holds
    assert {e.entry for e in led.hand("p4", 10)} == {e.entry for r in rounds[:2] for e in r}


def test_a_message_the_writer_was_already_delivered_is_read_in_that_pass(tmp_path):
    led = _ledger(tmp_path)
    # a fork's post that names an item (a correction): it carries something of its own, so it is handed at once
    post = {"seq": 7, "author": "fork-3f9a", "role": "fork", "kind": "post", "mentions": ["writer"], "episode": "e1",
            "refs": ["item:memory.text.words:upper"], "ts": "2026-10-20T00:00:01.000Z"}  # fmt: skip
    later = {**post, "seq": 9}
    [first, second] = sorted(led.sync(Fake(record=[post, later], deliveries={7: "p0"})), key=lambda e: e.refs.record_seq)
    assert first.state == "closed" and first.reason_ref == "record:delivered#7" and first.pass_id == "p0"
    assert first.decision == "nothing_to_keep" and first.parts_read == ("entry",)
    assert second.state == "open"  # never delivered: handed as before
    assert [e.entry for e in led.hand("p1", 4)] == [second.entry]


def test_an_entry_read_but_never_decided_goes_alone_then_ends_unreadable(tmp_path):
    clock = Clock()
    led = _ledger(tmp_path, clock)
    src = Fake(forks=[_fork("e1", procedure=False)])
    [one] = led.sync(src)
    for k in (1, 2, 3):
        clock.tick(60)
        if k == 3:
            src.forks.append(_fork("e2", procedure=False))
            [other] = led.sync(src)
        handed = led.hand(f"p{k}", 4)
        # read and not decided in two passes: handed alone in the third, whatever else is open
        assert [e.entry for e in handed] == [one.entry]
        assert led.verify_reads(f"p{k}", _read_all(handed)) == {one.entry}
        assert led.return_unread(f"p{k}") == [one.entry]
    e = led.get(one.entry)
    assert e.state == "unreadable" and e.undecided == 2 and "read and not decided in 3 passes, the last pass:p3" in e.why
    assert [x.entry for x in led.hand("p4", 4)] == [other.entry]
    # a reopen starts the count again
    with led.locked():
        led._write([{"op": "reopen", "entry": one.entry, "by": "late_evidence:x", "at": led.now()}])
    assert led.get(one.entry).undecided == 0


def test_a_grouped_decision_refused_names_every_unread_entry(tmp_path):
    led = _ledger(tmp_path)
    src = Fake(forks=[_fork(n, procedure=False) for n in ("e1", "e2", "e3")])
    made = sorted(led.sync(src), key=lambda e: e.refs.episodes)
    led.hand("p1", 4)
    reads = _read_all([made[0]])
    with pytest.raises(LedgerError) as err:
        led.decide("p1", [e.entry for e in made], "nothing_to_keep", [], "why", reads=reads,
                   decisions_path=tmp_path / "d.jsonl")  # fmt: skip
    text = str(err.value)
    assert f"2 of the 3 entries are unread: {made[1].entry}, {made[2].entry}" in text
    assert f"the read ones, {made[0].entry}, may be decided on their own" in text and "nothing was decided" in text
