"""S3: a lone capture is written up once its request has ended, in one pass, and stays servable.

Design: a pass is due for a new capture with related history; a capture with no related capture is due once its own
request has ended, all such at one moment in one pass (or with a pass that runs then anyway), whatever outcome the
world gave; nothing else (no count, no idle time) makes a pass due.
"""

from __future__ import annotations

import json

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_lone_capture_is_written_up_once_its_request_ends_and_stays_servable(system, mode):
    s = system
    words = ["stone", "river"]
    keep = "open('/workspace/kept-from-request-1.txt', 'w').write('kept')\n"  # "it persists across tasks"
    r1 = await s.request(SW.words_request(words), ScriptedModel(actor=[
        reply(calls=[cell(SW.words_cell(words) + keep)]), reply(json.dumps(SW.words_answer(words))), reply("Good.")]),
        SW.accepted_host())
    s.settle()
    cid = s.capture_of(r1.episode_id)
    # its request has ended: the lone capture is handed, in a pass of its own or with the pass that runs then anyway
    # (the fork's message's); no judge (nothing to compare it with)
    assert s.writer.pass_bodies and not s.writer.judge_bodies
    led = H.ledger(s.home)
    mine = [e for e in s.entries() if cid in (e.refs.captures or ())]
    (hand,) = [r for r in led.rows() if r.get("op") == "hand" and r.get("entry") in {e.entry for e in mine}]
    row = led.group_of(hand["pass_id"])
    assert hand.get("by") == "with_pass" or (row["group"] is None and row["due"] == "ended"), (hand, row)
    assert all(e.state != "open" for e in mine), [(e.kind, e.state) for e in mine]
    # still servable on the next request
    sink: dict = {}
    item = f"capture:{cid}"
    code = SW.probe_json("_try('kept', lambda: open('/workspace/kept-from-request-1.txt').read())\n"
                         f"_try('items', lambda: [i.id for i in memory.items()])\n"
                         f"_try('call', lambda: memory.get({item!r})())\n")
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink), librarian=Always(lambda c: SW.librarian_answer([SW.fit(item)]))))
    assert item in sink["out"]["items"] and sink["out"]["call"] == ["KAO"], sink["out"]
    assert sink["out"]["kept"] == "kept", ("the working directory did not persist across requests", sink["out"]["kept"])
    # an entry with no capture (a summary-only fork's) runs no pass: the harness closes it
    for e in s.entries():
        if e.kind == "new_experience" and not e.refs.captures:
            assert e.state == "closed" and e.reason_ref == "no_capture", (e.kind, e.state, e.reason_ref)
    s.settle()
    # the run's end: nothing is left that was never reached
    report = H.drain(s.home)
    H.check_terminated(report, s.home)
    rows = H.ledger(s.home).never_grouped()
    assert not any(cid in json.dumps(r) for r in rows), rows
    assert (s.captures().get(cid) or {}).get("procedure"), "the capture's files stay on record"

@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_help_on_memory_as_the_first_call_shows_the_library_documentation(system, mode):
    """Guard J11 (audit A2): ``help(memory)`` as a request's first touch of the library shows the package's own
    documentation (how to list, open and call), not the lazy loader's."""
    s = system
    words = ["stone"]
    r1 = await s.request(SW.words_request(words), ScriptedModel(actor=[
        reply(calls=[cell(SW.words_cell(words))]), reply(json.dumps(SW.words_answer(words)))]))
    s.settle()
    item = f"capture:{s.capture_of(r1.episode_id)}"
    sink: dict = {}
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json("import pydoc\n_try('help', lambda: pydoc.plain(pydoc.render_doc(memory)))\n"),
                             sink),
        librarian=Always(lambda c: SW.librarian_answer([SW.fit(item)]))))
    text = sink["out"]["help"]
    assert "memory.get" in text and "memory.items" in text, text[:800]
