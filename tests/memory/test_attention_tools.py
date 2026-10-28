"""The writer's attention-ledger tools in code: ``tools.open_entries``, ``tools.take``, ``tools.decide``,
``tools.find`` and ``tools.test``, on an attention pass's own state.

Each exists only with its flag; a take exports the entry's material under /inputs; a decision is refused until the
pass's own read evidence shows the material opened, and then closes the entries together; find ranks without
proposing groups. No model is called and no box runs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from unify.common import tool_registry as reg
from unify.memory import attention as at
from unify.memory import attention_tools as tools
from unify.memory.attention import Ledger, SettledFork


class _Src:
    def __init__(self, forks):
        self.forks = forks

    def settled_forks(self):
        return self.forks

    def capture_rows(self):
        return {}

    def use_failures(self):
        return []

    def late_signals(self):
        return []

    def episode_calls(self):
        return []

    def episode_items(self, episode):
        return ()

    def gate_refusals(self):
        return []

    def record_entries(self):
        return []


def _home(tmp_path) -> Path:
    home = tmp_path / "home"
    staging = home / "memory-state" / "memory-staging"
    for name, asked in (("e1", "Total the rows of a table"), ("e2", "Total the rows of a sheet"), ("e3", "Book a room")):
        d = staging / name
        d.mkdir(parents=True)
        (d / "summary.md").write_text(f"## Asked\n{asked}.\n## Approach\n- summed (`cell:0` \"sum\")\n")
        (d / "fork.json").write_text(json.dumps({"episode": name, "status": "ok", "files": ["summary.md"]}))
    return home


def _ctx(tmp_path, home, entries):
    led = Ledger(home)
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    att = tools.AttentionPass(
        ledger=led,
        pass_id="p1",
        record_dir=home / "memory-state" / "passes" / "p1",
        inputs=inputs,
        home=home,
        handed=entries,
    )
    box = tmp_path / "box"
    box.mkdir(exist_ok=True)
    writer = SimpleNamespace(_attention=att, _read_logs={}, tool_events=[], box_opens=[], _tests=None, _live=(box, None))
    return SimpleNamespace(state=SimpleNamespace(writer=writer, memory_lock=threading.Lock())), att, writer


def _forks():
    return [SettledFork(n, n, True, False, None) for n in ("e1", "e2", "e3")]


def test_the_writer_always_has_the_attention_and_test_tools():
    """Rev 2: every writer pass is an attention pass with stated tests, so the writer's allow-list has these tools with
    no flag (the attention and tests switches are gone, and so is the curator)."""
    plain = set(reg.allow_list("writer", frozenset()))
    assert {"open_entries", "take", "decide", "find", "test"} <= plain
    for name in ("open_entries", "take", "decide", "find", "test"):
        assert reg.help_text(name, "writer")


def test_open_entries_take_and_decide(tmp_path):
    home = _home(tmp_path)
    led = Ledger(home)
    led.sync(_Src(_forks()))
    handed = led.hand("p1", 1)
    ctx, att, writer = _ctx(tmp_path, home, handed)
    tools.export_entries(writer, att, handed)
    listing = json.loads(tools.open_entries(ctx, {}).text)
    assert listing["count"] == 2 and all(r["goal"] for r in listing["open"])
    other = listing["open"][0]["entry"]
    got = json.loads(tools.take(ctx, {"entry_id": other}).text)
    assert got["taken"] and got["material"]
    taken_name = led.get(other).refs.episodes[0]
    assert (att.inputs / "staging" / taken_name / "summary.md").is_file()
    assert (att.inputs / "attention" / other / "entry.json").is_file()
    index = json.loads((att.inputs / "attention" / "index.json").read_text())
    assert {r["entry"] for r in index["entries"]} == {handed[0].entry, other}
    both = [handed[0].entry, other]
    refused = tools.decide(ctx, {"entry_ids": both, "decision": "new_item", "items": ["memory.a.b:c"], "reason": "one"})
    assert refused.status == "refused" and "decide in a later cell" in refused.text
    # the summaries opened, as the harness records it (a read tool's row, a box's open row)
    names = [led.get(e).refs.episodes[0] for e in both]
    writer.tool_events.append({"status": "ok", "paths": [{"path": f"/inputs/staging/{names[0]}/summary.md", "mode": "read"}]})
    writer.box_opens.append({"path": f"/inputs/staging/{names[1]}/summary.md", "mode": "r"})
    done = json.loads(tools.decide(ctx, {"entry_ids": both, "decision": "new_item", "items": ["memory.a.b:c"],
                                         "reason": "one rule"}).text)  # fmt: skip
    assert done["closed"] == both and done["reason_ref"] == "pass:p1/decisions.jsonl#0"
    assert all(led.get(e).state == "closed" for e in both)
    assert json.loads((att.record_dir / "decisions.jsonl").read_text().splitlines()[0])["reason"] == "one rule"
    # the read log is kept for the service
    tools.write_read_log(writer)
    reads = at.pass_reads(home / "memory-state", "p1")
    assert f"/inputs/staging/{names[0]}/summary.md" in reads.paths
    assert tools.take(ctx, {"entry_id": ""}).status == "refused"
    # rev 2.1: a closed entry is taken read-only: its material and what was decided, and it stays closed
    again = json.loads(tools.take(ctx, {"entry_id": other}).text)
    assert again["taken"] == "read-only" and again["decision"] == "new_item" and led.get(other).state == "closed"


def test_find_lists_summaries_entries_and_items_unranked_without_an_embedding_route(tmp_path):
    """Rev 2.1, fix 8: no word matching ranks anything; with no embedding route (none here) find lists in order."""
    home = _home(tmp_path)
    led = Ledger(home)
    led.sync(_Src(_forks()))
    ctx, att, writer = _ctx(tmp_path, home, [])
    hits = json.loads(tools.find(ctx, {"query": "total the rows", "scope": "summaries", "k": 2}).text)
    assert hits["scope"] == "summaries" and len(hits["hits"]) == 2
    assert [h["id"] for h in hits["hits"]] == ["e1", "e2"] and hits["ranked"].startswith("not ranked, because")
    assert all(h["score"] is None for h in hits["hits"])
    entries = json.loads(tools.find(ctx, {"query": "book a room", "scope": "open_entries", "k": 3}).text)
    # the open entries in the ledger's own order, whatever the query's words
    assert [h["id"] for h in entries["hits"]] == [o.entry for o in led.open_entries()]
    items = json.loads(tools.find(ctx, {"query": "anything"}).text)
    assert items["scope"] == "items" and items["hits"] == []
    for bad, words in (({"query": ""}, "query is"), ({"query": "x", "scope": "all"}, "scope is one of"),
                       ({"query": "x", "k": 0}, "k is a positive")):  # fmt: skip
        r = tools.find(ctx, bad)
        assert r.status == "refused" and words in r.text


def test_without_a_ledger_the_tools_say_so(tmp_path):
    ctx = SimpleNamespace(state=SimpleNamespace(writer=SimpleNamespace(_attention=None, _tests=None)))
    for fn in (tools.open_entries, tools.take, tools.decide, tools.find):
        r = fn(ctx, {"entry_id": "x", "query": "x"})
        assert r.status == "refused" and "not an attention pass" in r.text


def test_the_first_message_names_the_entries():
    att = SimpleNamespace(pass_id="p9")
    e = at.AttentionEntry("a" * 16, "message", "handed", "2026-10-20T00:00:00.000Z", at.Refs(record_seq=3))
    text = tools.first_message(att, SimpleNamespace(attention=[e]), "900")
    assert text.startswith("Pass p9: 1 entry needs your attention") and f"{'a' * 16} (message)" in text and "900 s" in text


def test_a_retire_takes_effect_on_what_it_names_or_is_refused_with_why(tmp_path):
    """A retire naming an item the library does not hold (no function or note of the pass's copy, no capture of the
    store) is refused naming that item, and nothing is decided; a retire naming a capture of the store closes the
    entry, and the capture is then one the request's copy leaves out, with the writer's reason."""
    home = _home(tmp_path)
    led = Ledger(home)
    led.sync(_Src(_forks()))
    handed = led.hand("p1", 1)
    ctx, att, writer = _ctx(tmp_path, home, handed)
    tools.export_entries(writer, att, handed)
    name = led.get(handed[0].entry).refs.episodes[0]
    writer.tool_events.append({"status": "ok", "paths": [{"path": f"/inputs/staging/{name}/summary.md", "mode": "read"}]})
    box = writer._live[0]
    (box / "notes" / "text").mkdir(parents=True)
    (box / "notes" / "text" / "rows.md").write_text("---\ntitle: rows\ndescription: rows\nuses: []\n---\n## Rule\n- x\n")
    held, other = "a" * 16, "b" * 16
    store = home / "memory-state" / "memory-captures" / held
    store.mkdir(parents=True)
    (store / "record.json").write_text(json.dumps({"outcome": "positive"}))
    entry = handed[0].entry
    for missing in ("notes/none/nothing.md", f"capture:{other}", "memory.none.mod:fn"):
        got = tools.decide(ctx, {"entry_ids": [entry], "decision": "retire", "items": [missing], "reason": "broke"})
        assert got.status == "refused" and missing in got.text and "nothing was decided" in got.text, got.text
    assert led.get(entry).state != "closed" and not (att.record_dir / "decisions.jsonl").exists()
    # an item the copy holds may be retired; other decisions are not checked against the copy
    ok = json.loads(tools.decide(ctx, {"entry_ids": [entry], "decision": "retire",
                                       "items": [f"capture:{held}", "notes/text/rows.md"],
                                       "reason": "its use failed in two requests"}).text)  # fmt: skip
    assert ok["closed"] == [entry] and led.get(entry).decision == "retire"
    assert at.retired_captures(home / "memory-state") == {
        held: {"status": "retired", "by": None, "reason": "its use failed in two requests"}}
    assert at.item_decisions(home)["notes/text/rows.md"]["status"] == "retired"
