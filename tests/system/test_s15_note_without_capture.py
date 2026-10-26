"""S15: an episode that left no capture but left a note candidate reaches the writer, which lands a note with its scope;
the scope is shown wherever the note is served.

Design (rev 2.4, P1): an episode whose fork staged lessons (``notes.md``) or a note candidate (``notes/<name>.md``) is
not closed unseen for want of a procedure: its entry waits for its group as a capture does, with the same standing, and
its lessons reach the writer as its material. The fork's brief lets it keep a note candidate (a stated preference or
rule with who stated it and what it applies to, a correction, an outcome, a caution) with no ``procedure.py``. The
writer can land a note with no capture behind it; a note's front matter takes ``stated_by`` and ``applies_to``, shown
wherever it is served. Only the model and the world are scripted.
"""

from __future__ import annotations

import json

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_note_candidate_with_no_capture_reaches_the_writer_and_lands_a_scoped_note(system, mode):
    s = system
    s.writer.plan = "land_scoped"
    # a request with no code: the actor answers in text, so its fork writes no procedure; it keeps a note candidate
    r1 = await s.request(SW.PREF_REQUEST, ScriptedModel(
        actor=[reply(json.dumps(SW.PREF_ANSWER))],
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    s.settle()
    assert not [cid for cid, d in s.captures().items()
                if r1.episode_id in ((d.get("record") or {}).get("episodes") or [])], "no capture was expected"
    mine = [e for e in s.entries() if e.kind == "new_experience" and r1.episode_id in e.refs.episodes]
    assert len(mine) == 1, [(e.kind, e.refs) for e in s.entries()]
    (entry,) = mine
    # its note candidate is its material; the harness did not close it unseen
    assert entry.refs.captures == () and any(n.endswith("/notes/order.md") for n in entry.refs.notes), entry.refs
    assert entry.reason_ref != "no_capture", entry
    assert any(r.get("entry") == entry.entry for r in s.ledger_rows("hand")), "the entry was never handed to a pass"
    # the writer read the candidate in its pass and landed a note with no capture behind it
    assert entry.state == "closed" and entry.decision == "new_item" and SW.NOTE_ID in entry.decided_items, entry
    assert entry.parts_read and "notes" in entry.parts_read, entry.parts_read
    assert s.library_main_has(SW.NOTE_ID)
    # the pass's first message named it an entry, not a capture
    firsts = [SW._first_user(b) for b in s.writer.pass_bodies]
    assert any(entry.entry in f for f in firsts), firsts
    assert not any(entry.entry in f and "related capture" in f.split("needs your attention")[0] for f in firsts)
    # the next request is served the note with its scope: memory.get, memory.items() and the library's index
    sink: dict = {}
    code = SW.probe_json(f"_try('get', lambda: str(memory.get({SW.NOTE_ID!r})))\n"
                         "_try('items', lambda: str(memory.items(kind='note')))\n")
    await s.request(SW.words_request(["elm"]), ScriptedModel(
        actor=SW.probe_turns(code, sink), librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    out = sink["out"]
    assert f"applies_to: {SW.PREF_SCOPE}" in out["get"] and f"stated_by: {SW.PREF_WHO}" in out["get"], out["get"]
    assert f"applies to: {SW.PREF_SCOPE[:20]}" in out["items"], out["items"]
    index = (s.copy() / "INDEX.md").read_text()  # the index of the copy that request mounted
    assert f"applies to: {SW.PREF_SCOPE}; stated by: {SW.PREF_WHO}" in index, index[-2000:]
