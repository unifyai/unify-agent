"""Each writer pass reads a capture group together with the library state it touches.

Membership comes from the harness's records only (the same ordered sequence of tool calls or a shared run of calls,
shared library items, the same input shape of the same form, a judge's ``kin`` rows), never from an id, a word, a
request's text or a stream position, and nothing runs. A new capture is handed only with another related capture (a
new one, or its history); a capture with no related capture is never handed alone, an entry with no capture is closed
by the harness, and every other kind of entry is handed at once.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path
from types import SimpleNamespace

from unify.memory import attention as at
from unify.memory import attention_tools as tools
from unify.memory import capture_groups as cg
from unify.memory.attention import Ledger, LateSignal, SettledFork

from tests.memory.test_attention_ledger import Clock, Fake


def _p(entry, *, form="state_env", effects=(), shape=(), items=(), outcome="none", sequence=()):
    return cg.Profile(entry, (entry[:16],), form, frozenset(effects or sequence), frozenset(shape), frozenset(items),
                      outcome, tuple(sequence))  # fmt: skip


STEP = ("env.board.read", "env.board.step")


def _e(eid, state="open", at_="2026-10-20T00:00:00.000Z"):
    return at.AttentionEntry(eid, "new_experience", state, at_, at.Refs(episodes=(eid,), captures=(eid[:16],)))


NOW = at.parse_time("2026-10-20T00:01:00.000Z")


# --- relations: records only ---------------------------------------------------------------------------------------------


def test_captures_are_related_by_a_shared_call_sequence_items_or_input_shape_and_nothing_else():
    a = _p("a", sequence=("env.mail.read", "env.mail.send"), items={"memory.mail.reply:draft"})
    b = _p("b", sequence=("env.mail.read", "env.mail.send"))
    b2 = _p("b2", sequence=("env.mail.send", "env.mail.read"))  # the same calls in another order: other steps
    c = _p("c", items={"memory.mail.reply:draft"})
    d = _p("d", form="value", shape={"[].amount:int"})
    e = _p("e", form="value", shape={"[].amount:int"})
    f = _p("f", form="state", shape={"[].amount:int"})  # the same shape in another form: the entry form is no relation
    s1, s2 = _p("s1", form="state", shape={"[].amount:int"}), _p("s2", form="state", shape={"[].amount:int"})
    assert cg.related(a, b) == ["sequence: env.mail.read > env.mail.send"]
    assert cg.related(a, b2) == [] and cg.related(b, b2) == []
    assert cg.related(a, c) == ["items: memory.mail.reply:draft"]
    assert cg.related(d, e) == ["input"] and cg.related(s1, s2) == ["input"]  # a state's structured data too
    assert cg.related(d, f) == ["input"] and cg.related(b, c) == []
    # one shared call is no relation: in a world with one action channel every capture that calls it makes it
    one = _p("o1", sequence=("env.game.step",)), _p("o2", sequence=("env.game.step",))
    assert cg.related(*one) == []
    # the same outcome alone, or nothing at all in common, never relates two captures
    assert cg.related(_p("x", outcome="positive"), _p("y", outcome="positive")) == []


def test_an_episode_that_called_a_capture_is_related_to_that_capture_both_ways():
    x = _p("capturexxxxxxxxxx1", sequence=("env.mail.read",))
    user = _p("user", items={"capture:capturexxxxxxxxx"})  # its episode called x's capture (x's id is its first 16)
    assert x.captures == ("capturexxxxxxxxx",)
    assert cg.related(x, user) == ["items: capture:capturexxxxxxxxx"]
    assert cg.related(user, x) == ["items: capture:capturexxxxxxxxx"]
    # served, shown or at the pin only (not in items): no relation
    assert cg.related(x, _p("other")) == []


# --- when a group is due ------------------------------------------------------------------------------------------------


def _groups(entries, profiles, now=NOW):
    return cg.groups({e.entry: e for e in entries}, lambda e: profiles[e.entry], now=now)


def test_two_related_open_captures_form_a_due_group_and_an_unrelated_one_waits():
    es = [_e("a" * 16), _e("b" * 16), _e("c" * 16)]
    profiles = {"a" * 16: _p("a" * 16, sequence=STEP), "b" * 16: _p("b" * 16, sequence=STEP),
                "c" * 16: _p("c" * 16, sequence=("env.files.read", "env.files.write"))}  # fmt: skip
    got = _groups(es, profiles)
    formed = [g for g in got if g.due == "formed"]
    assert len(formed) == 1 and [e.entry for e in formed[0].open] == ["a" * 16, "b" * 16]
    alone = [g for g in got if [e.entry for e in g.open] == ["c" * 16]]
    assert alone and alone[0].due is None  # no related capture: it waits, never written up alone


def test_a_new_capture_related_to_a_decided_one_joins_its_group():
    es = [_e("a" * 16, state="closed"), _e("b" * 16, at_="2026-10-20T00:00:30.000Z")]
    profiles = {"a" * 16: _p("a" * 16, items={"memory.t.m:f"}), "b" * 16: _p("b" * 16, items={"memory.t.m:f"})}
    (g,) = _groups(es, profiles)
    assert g.due == "history" and [e.entry for e in g.context] == ["a" * 16] and g.relations


def test_a_capture_with_no_related_capture_is_never_due_whatever_comes_or_however_long():
    """No count of later, unrelated captures and no idle time hands a capture alone: transfer needs two instances."""
    first = _e("0" * 16)
    later = [_e(f"{i:x}" * 16, at_=f"2026-10-20T00:00:{10 + i:02d}.000Z") for i in range(1, 12)]
    profiles = {first.entry: _p(first.entry, sequence=("env.a.b", "env.a.c"))}
    profiles.update({e.entry: _p(e.entry, sequence=(f"env.other{i}.x", f"env.other{i}.y")) for i, e in enumerate(later)})
    got = {tuple(e.entry for e in g.open): g.due for g in _groups([first, *later], profiles)}
    assert got[(first.entry,)] is None
    a_year_later = NOW + dt.timedelta(days=365)
    assert _groups([first], {first.entry: _p(first.entry)}, now=a_year_later)[0].due is None


def test_the_history_is_the_most_related_closed_captures_at_most_history_max():
    new = _e("f" * 16, at_="2026-10-21T00:00:00.000Z")
    old = [_e(f"{i:x}" * 16, state="closed", at_=f"2026-10-20T00:00:{10 + i:02d}.000Z") for i in range(1, 12)]
    profiles = {new.entry: _p(new.entry, items={"memory.t.m:f"}, sequence=STEP)}
    # one old capture shares two relations, the rest one
    profiles.update({e.entry: _p(e.entry, items={"memory.t.m:f"}) for e in old})
    profiles[old[0].entry] = _p(old[0].entry, items={"memory.t.m:f"}, sequence=STEP)
    (g,) = _groups([new, *old], profiles)
    assert g.due == "history" and len(g.context) == cg.HISTORY_MAX
    assert g.context[0].entry == old[0].entry  # the strongest first
    assert [e.entry for e in g.context[1:]] == [e.entry for e in reversed(old[1:])][: cg.HISTORY_MAX - 1]  # then the newest


def test_a_call_feature_most_captures_make_relates_nothing_and_a_rarer_run_of_calls_does():
    """A world's own envelope of calls (made by more than half of the captures that make calls) distinguishes none of
    them; a shared run of two calls that few make relates two captures. A capture that makes no call has no call
    feature: it neither adds to nor dilutes the envelope."""
    env = ("env.chat.ask", "env.chat.reply")
    ps = [_p(f"{i:x}" * 16, sequence=env) for i in range(4)]
    rare_a = _p("a" * 16, sequence=("env.files.read", "env.files.write", *env))
    rare_b = _p("b" * 16, sequence=("env.files.read", "env.files.write", "env.chat.reply"))
    common = cg.common_features([*ps, rare_a, rare_b])
    assert env in common and ("env.files.read", "env.files.write") not in common
    assert cg.related(ps[0], ps[1], common) == []
    assert cg.related(rare_a, rare_b, common) == ["subsequence: env.files.read > env.files.write"]
    assert cg.related(ps[0], ps[1]) == ["sequence: env.chat.ask > env.chat.reply"]  # with no envelope known
    quiet = [_p(f"q{i}" * 8, sequence=()) for i in range(8)]
    assert cg.common_features([*ps, rare_a, rare_b, *quiet]) == common


def test_a_judges_kin_row_relates_two_captures_and_unknown_ids_are_left_out(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    names = ("e1", "e2")
    cids = {"e1": "1" * 16, "e2": "2" * 16}
    led.sync(Fake(forks=[_fork(n) for n in names], captures={n: _cap(n, cids[n]) for n in names}))
    ids = {e.refs.captures[0]: e.entry for e in led.entries().values()}

    def grouping(entries):
        return cg.groups(entries, cg.HomeProfiles(led.state_dir), now=clock())

    assert led.hand("p0", 4, grouping=grouping) == []  # nothing relates them yet
    assert led.record_kin(ids["2" * 16], ["f" * 16], "unknown") is False
    assert led.record_kin(ids["2" * 16], [ids["1" * 16]], "both sort rows by a column") is True
    assert led.get(ids["2" * 16]).kin == (ids["1" * 16],)
    handed = led.hand("p1", 4, grouping=grouping)
    assert sorted(e.entry for e in handed) == sorted(ids.values())
    assert led.group_of("p1")["reasons"]["judged"] == 1


def test_an_entry_whose_episode_left_no_capture_is_closed_by_the_harness_and_runs_no_pass(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    led.sync(Fake(forks=[SettledFork("e1", "e1", None, True, False, None)]))
    (e,) = led.entries().values()
    assert e.refs.captures == ()
    assert led.hand("p1", 4, grouping=lambda entries: cg.groups(entries, lambda x: _p(x.entry), now=clock())) == []
    closed = led.get(e.entry)
    assert closed.state == "closed" and closed.decision == "nothing_to_keep" and closed.reason_ref == cg.NO_CAPTURE
    assert led.group_of("p1") is None


def test_a_group_with_a_member_in_a_running_pass_is_not_due():
    es = [_e("a" * 16, state="handed"), _e("b" * 16), _e("c" * 16)]
    profiles = {e.entry: _p(e.entry, items={"memory.x.y:z"}) for e in es}
    (g,) = _groups(es, profiles)
    assert g.pending and g.due is None


# --- the ledger hands one due group, everything else at once -------------------------------------------------------------


def _fork(name):
    return SettledFork(name, name, None, True, True, None)


def _cap(name, cid):
    return {"v": 2, "capture_id": cid, "episode": name, "segment": None, "own_check": {"ran": False}, "outcome": "none"}


def test_the_ledger_hands_other_kinds_first_then_one_due_group_with_its_row(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    names = ("e1", "e2", "e3")
    cids = {"e1": "1" * 16, "e2": "2" * 16, "e3": "3" * 16}
    led.sync(Fake(forks=[_fork(n) for n in names], captures={n: _cap(n, cids[n]) for n in names}))
    clock.tick()
    led.sync(Fake(late=[LateSignal("s1", "e9", None)]))
    seqs = {"1" * 16: STEP, "2" * 16: STEP, "3" * 16: ("env.files.read", "env.files.write")}

    def grouping(entries):
        return cg.groups(entries, lambda e: _p(e.entry, sequence=seqs[e.refs.captures[0]]), now=clock())

    first = led.hand("p1", 4, grouping=grouping)
    assert [e.kind for e in first] == ["late_evidence"] and led.group_of("p1") is None
    second = led.hand("p2", 4, grouping=grouping)
    assert sorted(e.refs.captures[0] for e in second) == ["1" * 16, "2" * 16]
    row = led.group_of("p2")
    assert row["due"] == "formed" and sorted(row["members"]) == sorted(e.entry for e in second)
    # e3 relates to nothing: it waits, however long, and is never handed alone
    assert led.hand("p3", 4, grouping=grouping) == []
    clock.tick(10**6)
    assert led.hand("p4", 4, grouping=grouping) == [] and led.get(next(
        e.entry for e in led.entries().values() if e.refs.captures == ("3" * 16,))).state == "open"
    # without a grouping the ledger hands as before (every kind at once)
    assert Ledger(tmp_path / "home2", clock=clock).hand("q", 4) == []


# --- what a group pass is given ----------------------------------------------------------------------------------------


def test_a_group_no_pass_finishes_is_given_up_never_unreadable_and_never_holds_others_up(tmp_path):
    """A group whose members come back unread pass after pass yields to another due group, and at the bound the harness
    closes its captures (never unreadable): they stay served and are the history of a later related capture."""
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    names = ("e1", "e2", "e3", "e4")
    cids = {n: str(i + 1) * 16 for i, n in enumerate(names)}
    led.sync(Fake(forks=[_fork(n) for n in names], captures={n: _cap(n, cids[n]) for n in names}))
    other = ("env.files.read", "env.files.write")
    seqs = {cids["e1"]: STEP, cids["e2"]: STEP, cids["e3"]: other, cids["e4"]: other}

    def grouping(entries):
        return cg.groups(entries, lambda e: _p(e.entry, sequence=seqs[e.refs.captures[0]]), now=clock())

    stuck = {e.entry for e in led.entries().values() if e.refs.captures[0] in (cids["e1"], cids["e2"])}
    first = led.hand("p1", 4, grouping=grouping)
    assert {e.entry for e in first} == stuck  # the oldest group first
    led.return_unread("p1")  # nothing read: a miss each
    second = led.hand("p2", 4, grouping=grouping)
    assert not {e.entry for e in second} & stuck  # the group that came back yields to the other due group
    led.return_unread("p2")
    for n in range(3, 6):
        handed = led.hand(f"p{n}", 4, grouping=grouping)
        led.return_unread(f"p{n}")
    states = {e.entry: e for e in led.entries().values()}
    assert all(states[i].state == "closed" and states[i].reason_ref == at.GIVEN_UP for i in stuck)
    assert not any(e.state == "unreadable" for e in states.values())
    assert at.GIVEN_UP == cg.GIVEN_UP and "group passes" in states[next(iter(stuck))].why
    # a closed, given-up capture is history: the group document says the harness closed it
    doc = cg.group_doc(cg.Group(id="x", open=[], context=[states[next(iter(stuck))]]), lambda e: _p(e.entry))
    assert doc["related_closed"][0]["closed_by"].startswith("the harness")


def test_the_group_document_names_members_related_decisions_and_library_items():
    members = [_e("a" * 16), _e("b" * 16)]
    closed = at.AttentionEntry("c" * 16, "new_experience", "closed", "2026-10-19T00:00:00.000Z",
                               at.Refs(episodes=("c" * 16,), captures=("c" * 16,)), decision="new_item",
                               decided_items=("memory.board.moves:step_left",))  # fmt: skip
    profiles = {e.entry: _p(e.entry, sequence=STEP, items={"memory.board.read:grid"})
                for e in [*members, closed]}  # fmt: skip
    g = cg.Group(id="a" * 16, open=members, context=[closed], due="formed",
                 relations=[{"a": "a" * 16, "b": "b" * 16, "why": ["sequence: env.board.read > env.board.step"]}])  # fmt: skip
    doc = cg.group_doc(g, lambda e: profiles[e.entry], landed={"a" * 16: "memory.board.moves:step_up"})
    assert [m["entry"] for m in doc["members"]] == ["a" * 16, "b" * 16] and doc["due"] == "formed"
    (rel,) = doc["related_closed"]
    assert rel["decision"] == "new_item" and rel["decided_items"] == ["memory.board.moves:step_left"]
    assert doc["library_items"] == ["memory.board.moves:step_left", "memory.board.moves:step_up", "memory.board.read:grid"]
    assert doc["relations"][0]["why"] == ["sequence: env.board.read > env.board.step"]
    assert doc["members"][0]["sequence"] == list(STEP)


def test_the_related_closed_entries_come_most_related_first_then_most_recent():
    members = [_e("a" * 16)]
    old_strong = _e("b" * 16, state="closed", at_="2026-10-19T00:00:00.000Z")
    new_weak = _e("c" * 16, state="closed", at_="2026-10-19T12:00:00.000Z")
    newer_weak = _e("d" * 16, state="closed", at_="2026-10-19T18:00:00.000Z")
    rel = [{"a": "a" * 16, "b": "b" * 16, "why": ["sequence: x > y", "items: memory.p.m:f"]},
           {"a": "c" * 16, "b": "a" * 16, "why": ["items: memory.p.m:f"]},
           {"a": "a" * 16, "b": "d" * 16, "why": ["input"]}]  # fmt: skip
    g = cg.Group(id="a" * 16, open=members, context=[old_strong, new_weak, newer_weak], relations=rel)
    # two relations beat one, whatever the age; between equals the more recent first
    assert [e.entry for e in cg.by_strength(g)] == ["b" * 16, "d" * 16, "c" * 16]


def test_the_call_sequence_is_read_in_written_order_with_repeats_kept_once():
    from unify.memory.capture_shape import call_sequence

    src = (
        "def run(state, env):\n"
        "    grid = env.board.read()\n"
        "    for _ in range(3):\n"
        "        env.board.step('left')\n"
        "    env.board.step('up')\n"
        "    return env.mail.send(grid)\n"
    )
    assert call_sequence(src) == ["env.board.read", "env.board.step", "env.mail.send"]
    assert call_sequence("def run(state):\n    return state\n") == [] and call_sequence("def run(:") == []


def _home(tmp_path):
    home = tmp_path / "home"
    staging = home / "memory-state" / "memory-staging"
    for name in ("e1", "e2"):
        d = staging / name
        d.mkdir(parents=True)
        (d / "summary.md").write_text("## Asked\nMove the piece.\n")
        (d / "fork.json").write_text(json.dumps({"episode": name, "status": "ok", "files": ["summary.md"]}))
    return home


def test_a_closed_entry_is_taken_read_only_and_stays_closed(tmp_path):
    home = _home(tmp_path)
    led = Ledger(home)
    led.sync(Fake(forks=[SettledFork(n, n, None, True, False, None) for n in ("e1", "e2")]))
    (e1,) = led.hand("p0", 1)
    reads = at.PassReads(frozenset({"/inputs/staging/e1/summary.md"}))
    led.decide("p0", [e1.entry], "nothing_to_keep", [], "nothing transfers", reads=reads,
               decisions_path=tmp_path / "decisions.jsonl")  # fmt: skip
    assert led.get(e1.entry).state == "closed"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    att = tools.AttentionPass(ledger=led, pass_id="p1", record_dir=home / "memory-state" / "passes" / "p1",
                              inputs=inputs, home=home, handed=[])  # fmt: skip
    writer = SimpleNamespace(_attention=att, _read_logs={}, tool_events=[], box_opens=[], _tests=None, _live=None)
    ctx = SimpleNamespace(state=SimpleNamespace(writer=writer, memory_lock=threading.Lock()))
    got = json.loads(tools.take(ctx, {"entry_id": e1.entry}).text)
    assert got["taken"] == "read-only" and got["decision"] == "nothing_to_keep"
    assert (inputs / "staging" / "e1" / "summary.md").is_file() and "/inputs/staging/e1/" in got["material"]
    assert led.get(e1.entry).state == "closed"  # nothing owed, nothing changed


def test_the_first_message_names_the_group():
    att = SimpleNamespace(pass_id="p9", group={"related_closed": [{}], "library_items": ["memory.a.b:c", "notes/x/y.md"]})
    es = [_e("a" * 16, state="handed"), _e("b" * 16, state="handed")]
    text = tools.first_message(att, SimpleNamespace(attention=es), "900")
    assert text.startswith("Pass p9: a group of 2 related captures needs your attention")
    assert "with 1 related capture closed earlier (by a pass, or by the harness)" in text and "2 library items" in text
    assert "each related pair with its relations" in text and "from its records only" not in text
    assert "/inputs/attention/group.json" in text and "900 s" in text


def test_grouping_reads_no_id_word_or_request_text():
    """The grouping module reads profiles built from code, signatures and use rows: it never reads a goal, a summary,
    a request or an episode id to relate captures."""
    src = Path(cg.__file__).read_text(encoding="utf-8")
    for word in ("summary.md", "asked(", ".request", "goal", "episode_id ==", "task"):
        assert word not in src.split('"""', 2)[2], word


# --- rev 2.1 grouping fix: used items, the shared envelope, the logged reasons ------------------------------------------


def _use_store(rows):
    """A stand-in evidence store: an ``item_use`` table with the call columns (stdlib sqlite, in memory)."""
    import sqlite3

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE item_use(item TEXT, episode_id TEXT, called INTEGER, refused INTEGER, errored INTEGER, "
               "shown INTEGER, PRIMARY KEY(item, episode_id))")  # fmt: skip
    db.executemany("INSERT INTO item_use VALUES(?,?,?,?,?,?)", rows)
    return SimpleNamespace(db=db)


def _entry(eid, captures=()):
    return SimpleNamespace(entry=eid, refs=SimpleNamespace(episodes=(eid,), captures=tuple(captures)))


def test_items_served_to_every_episode_relate_nothing_and_only_used_ones_relate(tmp_path):
    """A few general items are served to every episode (at the pin, shown): they must not join everything. Two
    episodes that called the same item, and one whose call of it was refused, are related by it; nothing else is."""
    episodes = [f"ep{i:02d}" for i in range(10)]
    rows = []
    for e in episodes:
        rows += [("notes/general/checklist.md", e, 0, 0, 0, 1), ("memory.common.parse:read", e, 0, 0, 0, 1)]
    rows += [("memory.ledger.post:book", "ep03", 2, 0, 0, 1), ("memory.ledger.post:book", "ep07", 1, 0, 0, 1),
             ("memory.ledger.post:book", "ep08", 0, 1, 0, 1)]  # fmt: skip
    profiles = cg.HomeProfiles(tmp_path / "state", _use_store(rows))
    got = {e: profiles(_entry(e)) for e in episodes}
    assert got["ep03"].items == frozenset({"memory.ledger.post:book"})
    assert got["ep00"].items == frozenset()
    # served only: neither a relation nor shown to the writer as the group's
    doc = cg.group_doc(cg.Group(id="ep00", open=[_e("ep00" * 4)]), lambda e: got["ep00"])
    assert doc["library_items"] == [] and doc["members"][0]["used_items"] == []
    related = {(a, b) for i, a in enumerate(episodes) for b in episodes[i + 1:] if cg.related(got[a], got[b])}
    assert related == {("ep03", "ep07"), ("ep03", "ep08"), ("ep07", "ep08")}


def _capture(store, cid, tree):
    d = store / cid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text("def run(value):\n    return value\n", encoding="utf-8")
    (d / "signature.json").write_text(json.dumps({"input": {"kind": "value", "tree": tree}}), encoding="utf-8")


def test_an_input_shape_every_capture_shares_relates_nothing_and_what_lies_below_it_does(tmp_path):
    """The key paths every capture of the home has (a world's shared envelope) distinguish none of them: only the
    structure below it relates captures, and with fewer than ENVELOPE_MIN shaped captures nothing is stripped."""
    from unify.memory.captures import store_dir

    state = tmp_path / "state"
    store = store_dir(state)
    envelope = {"head": {"from": "str", "sent": "str"}, "body": "str"}
    trees = {
        "a" * 16: {**envelope, "rows": [{"amount": "int", "ref": "str"}]},
        "b" * 16: {**envelope, "rows": [{"amount": "int", "ref": "str"}]},
        "c" * 16: {**envelope, "grid": [["int"]]},
        "d" * 16: dict(envelope),
    }
    _capture(store, "a" * 16, trees["a" * 16])
    _capture(store, "b" * 16, trees["b" * 16])
    profiles = cg.HomeProfiles(state)
    early = [profiles(_entry(c, (c,))) for c in ("a" * 16, "b" * 16)]
    assert cg.related(*early) == ["input"] and profiles.envelope() == frozenset()   # 2 shaped captures: no envelope
    for cid in ("c" * 16, "d" * 16):
        _capture(store, cid, trees[cid])
    got = {c: profiles(_entry(c, (c,))) for c in trees}
    assert profiles.envelope() and all(not (p.input & profiles.envelope()) for p in got.values())
    assert cg.related(got["a" * 16], got["b" * 16]) == ["input"]
    assert not cg.related(got["a" * 16], got["c" * 16]) and not cg.related(got["c" * 16], got["d" * 16])
    assert got["d" * 16].input == frozenset()   # the envelope alone: no shape left, so nothing relates it


def test_a_group_row_logs_why_the_group_formed(tmp_path):
    clock = Clock()
    led = Ledger(tmp_path / "home", clock=clock)
    names = ("e1", "e2", "e3")
    cids = {"e1": "1" * 16, "e2": "2" * 16, "e3": "3" * 16}
    led.sync(Fake(forks=[_fork(n) for n in names], captures={n: _cap(n, cids[n]) for n in names}))
    profs = {"1" * 16: _p("x", sequence=STEP, items={"memory.t.m:f"}), "2" * 16: _p("y", sequence=STEP),
             "3" * 16: _p("z", items={"memory.t.m:f"})}  # fmt: skip

    def grouping(entries):
        return cg.groups(entries, lambda e: profs[e.refs.captures[0]], now=clock())

    led.hand("p1", 4, grouping=grouping)
    row = led.group_of("p1")
    assert row["reasons"] == {"pairs": 2, "sequence": 1, "subsequence": 0, "items": 1, "input": 0, "judged": 0}
    assert cg.reasons([{"why": ["sequence: a > b", "input"]}, {"why": ["items: x", "judged"]}]) == {
        "pairs": 2, "sequence": 1, "subsequence": 0, "items": 1, "input": 1, "judged": 1}
