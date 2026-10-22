"""Rev 2.1, fix 2: each writer pass reads a capture group together with the library state it touches.

Membership comes from the harness's records only (shared ``env`` chains, shared library items, the same input shape
of the same form), never from an id, a word, a request's text or a stream position, and nothing runs. A new capture
waits until its group is due (formed, joined, waited out by counters); every other kind of entry is handed at once.
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


def _p(entry, *, form="state_env", effects=(), shape=(), items=(), outcome="none"):
    return cg.Profile(entry, (entry[:16],), form, frozenset(effects), frozenset(shape), frozenset(items), outcome)


def _e(eid, state="open", at_="2026-10-20T00:00:00.000Z"):
    return at.AttentionEntry(eid, "new_experience", state, at_, at.Refs(episodes=(eid,), captures=(eid[:16],)))


NOW = at.parse_time("2026-10-20T00:01:00.000Z")


# --- relations: records only ---------------------------------------------------------------------------------------------


def test_captures_are_related_by_shared_effects_items_or_input_shape_and_nothing_else():
    a = _p("a", effects={"env.mail.send", "env.mail.read"}, items={"memory.mail.reply:draft"})
    b = _p("b", effects={"env.mail.send"})
    c = _p("c", items={"memory.mail.reply:draft"})
    d = _p("d", form="value", shape={"[].amount:int"})
    e = _p("e", form="value", shape={"[].amount:int"})
    f = _p("f", form="state", shape={"[].amount:int"})  # the same shape in another form: not the same input
    assert cg.related(a, b) == ["effects: env.mail.send"]
    assert cg.related(a, c) == ["items: memory.mail.reply:draft"]
    assert cg.related(d, e) == ["input"]
    assert cg.related(d, f) == [] and cg.related(b, c) == []
    # the same outcome alone, or nothing at all in common, never relates two captures
    assert cg.related(_p("x", outcome="positive"), _p("y", outcome="positive")) == []


# --- when a group is due ------------------------------------------------------------------------------------------------


def _groups(entries, profiles, now=NOW):
    return cg.groups({e.entry: e for e in entries}, lambda e: profiles[e.entry], now=now)


def test_two_related_open_captures_form_a_due_group_and_an_unrelated_one_waits():
    es = [_e("a" * 16), _e("b" * 16), _e("c" * 16)]
    profiles = {"a" * 16: _p("a" * 16, effects={"env.board.step"}), "b" * 16: _p("b" * 16, effects={"env.board.step"}),
                "c" * 16: _p("c" * 16, effects={"env.files.write"})}  # fmt: skip
    got = _groups(es, profiles)
    formed = [g for g in got if g.due == "formed"]
    assert len(formed) == 1 and [e.entry for e in formed[0].open] == ["a" * 16, "b" * 16]
    alone = [g for g in got if [e.entry for e in g.open] == ["c" * 16]]
    assert alone and alone[0].due is None  # alone, with fewer than W later captures and no idle time yet


def test_a_new_capture_related_to_a_decided_one_joins_its_group():
    es = [_e("a" * 16, state="closed"), _e("b" * 16, at_="2026-10-20T00:00:30.000Z")]
    profiles = {"a" * 16: _p("a" * 16, items={"memory.t.m:f"}), "b" * 16: _p("b" * 16, items={"memory.t.m:f"})}
    (g,) = _groups(es, profiles)
    assert g.due == "joined" and [e.entry for e in g.context] == ["a" * 16] and g.relations


def test_a_capture_alone_is_due_after_w_later_captures_or_idle_time():
    first = _e("0" * 16)
    later = [_e(f"{i:x}" * 16, at_=f"2026-10-20T00:00:{10 + i:02d}.000Z") for i in range(1, cg.W + 1)]
    profiles = {first.entry: _p(first.entry, effects={"env.a.b"})}
    profiles.update({e.entry: _p(e.entry, effects={f"env.other{i}.x"}) for i, e in enumerate(later)})
    got = {tuple(e.entry for e in g.open): g.due for g in _groups([first, *later], profiles)}
    assert got[(first.entry,)] == "waited"
    quiet = {first.entry: _p(first.entry)}
    idle_now = at.parse_time("2026-10-20T00:00:00.000Z") + dt.timedelta(seconds=cg.IDLE_S)
    assert _groups([first], quiet, now=idle_now)[0].due == "idle"
    assert _groups([first], quiet, now=idle_now - dt.timedelta(seconds=1))[0].due is None


def test_a_group_with_a_member_in_a_running_pass_is_not_due():
    es = [_e("a" * 16, state="handed"), _e("b" * 16), _e("c" * 16)]
    profiles = {e.entry: _p(e.entry, effects={"env.x.y"}) for e in es}
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
    effects = {"1" * 16: {"env.board.step"}, "2" * 16: {"env.board.step"}, "3" * 16: {"env.files.write"}}

    def grouping(entries):
        return cg.groups(entries, lambda e: _p(e.entry, effects=effects[e.refs.captures[0]]), now=clock())

    first = led.hand("p1", 4, grouping=grouping)
    assert [e.kind for e in first] == ["late_evidence"] and led.group_of("p1") is None
    second = led.hand("p2", 4, grouping=grouping)
    assert sorted(e.refs.captures[0] for e in second) == ["1" * 16, "2" * 16]
    row = led.group_of("p2")
    assert row["due"] == "formed" and sorted(row["members"]) == sorted(e.entry for e in second)
    # e3 relates to nothing and nothing new has come: it waits (no pass), until W later captures or idle time
    assert led.hand("p3", 4, grouping=grouping) == []
    clock.tick(cg.IDLE_S)
    (alone,) = led.hand("p4", 4, grouping=grouping)
    assert alone.refs.captures == ("3" * 16,) and led.group_of("p4")["due"] == "idle"
    # without a grouping the ledger hands as before (every kind at once)
    assert Ledger(tmp_path / "home2", clock=clock).hand("q", 4) == []


# --- what a group pass is given ----------------------------------------------------------------------------------------


def test_the_group_document_names_members_related_decisions_and_library_items():
    members = [_e("a" * 16), _e("b" * 16)]
    closed = at.AttentionEntry("c" * 16, "new_experience", "closed", "2026-10-19T00:00:00.000Z",
                               at.Refs(episodes=("c" * 16,), captures=("c" * 16,)), decision="new_item",
                               decided_items=("memory.board.moves:step_left",))  # fmt: skip
    profiles = {e.entry: _p(e.entry, effects={"env.board.step"}, items={"memory.board.read:grid"})
                for e in [*members, closed]}  # fmt: skip
    g = cg.Group(id="a" * 16, open=members, context=[closed], due="formed",
                 relations=[{"a": "a" * 16, "b": "b" * 16, "why": ["effects: env.board.step"]}])  # fmt: skip
    doc = cg.group_doc(g, lambda e: profiles[e.entry], landed={"a" * 16: "memory.board.moves:step_up"})
    assert [m["entry"] for m in doc["members"]] == ["a" * 16, "b" * 16] and doc["due"] == "formed"
    (rel,) = doc["related_closed"]
    assert rel["decision"] == "new_item" and rel["decided_items"] == ["memory.board.moves:step_left"]
    assert doc["library_items"] == ["memory.board.moves:step_left", "memory.board.moves:step_up", "memory.board.read:grid"]
    assert doc["relations"][0]["why"] == ["effects: env.board.step"]


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
    assert "with 1 related capture an earlier pass decided" in text and "2 library items" in text
    assert "/inputs/attention/group.json" in text and "900 s" in text


def test_grouping_reads_no_id_word_or_request_text():
    """The grouping module reads profiles built from code, signatures and use rows: it never reads a goal, a summary,
    a request or an episode id to relate captures."""
    src = Path(cg.__file__).read_text(encoding="utf-8")
    for word in ("summary.md", "asked(", ".request", "goal", "episode_id ==", "task"):
        assert word not in src.split('"""', 2)[2], word
