"""S2: a related second capture forms a group with real history; a writer pass reads both; the item it lands links to
the capture it generalises and that capture links back, and both captures stay served.

Design: change 0.2 (a pass is due for a new capture with related history; a capture with no related capture is written
up once its request has ended; the group's pass reads both instances),
0.6 (the ``group`` row carries the members, the history and the reasons), 0R.5 and S4.3 (``generalises`` /
``generalised by`` links; the writer never silently hides a capture), P (history bounded at 8); spec R1.1, R1.2,
R1.5, R3.5, R4.1, R4.3, R6.6, R6.7.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
import json

from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_related_capture_forms_a_group_the_pass_lands_an_item_linked_both_ways(system, mode):
    s = system
    s.writer.plan = "keep"
    r1 = await s.accepted_words(["stone", "river"], "reverse")
    a = s.capture_of(r1.episode_id)
    # alone, it is written up once its request has ended (in a pass of its own, or with the pass that runs then
    # anyway); that pass keeps nothing here
    first = [e.entry for e in s.entries() if a in (e.refs.captures or ())]
    assert any(r.get("entry") in first for r in s.ledger_rows("hand")), s.ledger_rows("hand")
    s.writer.plan = "land"
    # the second episode used the first capture: a used item relates them (the same input shape alone never
    # does: every request of this world reads the same shape)
    words = ["glass", "cloud"]
    use_a = f"memory.get({'capture:' + a!r})({words!r})\n"
    r2 = await s.request(SW.words_request(words, op="upper"), ScriptedModel(
        actor=[reply(calls=[cell(use_a + SW.words_cell(words, op="upper"))]),
               reply(json.dumps(SW.words_answer(words, op="upper"))), reply("Good.")],
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))), SW.accepted_host())
    s.settle()
    b = s.capture_of(r2.episode_id)
    assert a != b

    # ── the group: both captures, with the reasons that relate them ──
    entry = {cid: [e.entry for e in s.entries() if cid in (e.refs.captures or ())] for cid in (a, b)}
    assert entry[a] and entry[b], entry
    groups = [g for g in s.ledger_rows("group")
              if set(entry[a]) & set(g.get("members", []) + g.get("context", []))
              and set(entry[b]) & set(g.get("members", []) + g.get("context", []))]
    assert groups, s.ledger_rows("group")
    g = groups[0]
    assert g.get("due") in ("formed", "history"), g
    assert g.get("reasons") and any(v for v in g["reasons"].values() if isinstance(v, int) and v > 0), g
    assert len(g.get("context") or []) <= 8, g

    # ── the pass read both instances and landed one item citing both episodes ──
    assert s.writer.pass_bodies, "no writer pass ran for the group"
    handed = set(SW.handed(s.writer.pass_bodies[-1]))
    first = SW._first_user(s.writer.pass_bodies[-1])
    assert set(entry[b]) & handed, (handed, first[:1500])
    # the first capture is read with it: a member, or the history its group row names
    assert set(entry[a]) & set((g.get("members") or []) + (g.get("context") or [])), g
    landed = [p for p in s.pass_results() if SW.NOTE_ID in ((p.get("items") or {}).get("landed") or [])]
    assert landed, s.pass_results()
    assert s.library_main_has(SW.NOTE_ID)

    # ── the next request: the item links to both captures, each capture links back, both stay listed ──
    sink: dict = {}
    ids = {"a": f"capture:{a}", "b": f"capture:{b}", "note": SW.NOTE_ID}
    code = SW.probe_json(
        "_try('items', lambda: [i.id for i in memory.items()])\n"
        + "".join(f"_try({k!r}, lambda: str(memory.get({v!r})))\n" for k, v in ids.items())
    )
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink),
        librarian=Always(lambda c: SW.librarian_answer([SW.fit(ids["a"])]))))
    out = sink["out"]
    assert ids["a"] in out["items"] and ids["b"] in out["items"] and SW.NOTE_ID in out["items"], out["items"]
    note_card = out["note"].lower()
    # the note is built from the new capture of its pass; the first capture, written up in its own pass (which kept
    # nothing), was read as history
    assert ids["b"] in note_card and "generalis" in note_card, out["note"][:2000]
    assert SW.NOTE_ID in out["b"] and "generalised by" in out["b"].lower(), out["b"][:2000]

