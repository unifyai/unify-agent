"""S3: a lone capture is never written up alone; it waits, stays servable, and the run's end counts it as never grouped.

Design: change 0.2 (a pass is due only for a new capture with related history; a lone capture waits and stays
servable; the end of run records it as ``never_grouped``), 0R.1; spec R1.1, R1.6, R1.7.
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
async def test_a_lone_capture_waits_stays_servable_and_is_counted_never_grouped(system, mode):
    s = system
    words = ["stone", "river"]
    r1 = await s.request(SW.words_request(words), ScriptedModel(actor=[
        reply(calls=[cell(SW.words_cell(words))]), reply(json.dumps(SW.words_answer(words))), reply("Good.")]),
        SW.accepted_host())
    s.settle()
    cid = s.capture_of(r1.episode_id)
    # never written up alone: no pass, no judge (nothing to compare it with)
    assert not s.writer.pass_bodies and not s.writer.judge_bodies
    mine = [e for e in s.entries() if cid in (e.refs.captures or ())]
    assert mine and all(e.state == "open" for e in mine), [(e.kind, e.state) for e in mine]
    # still servable on the next request, with no pass
    sink: dict = {}
    item = f"capture:{cid}"
    code = SW.probe_json(f"_try('items', lambda: [i.id for i in memory.items()])\n"
                         f"_try('call', lambda: memory.get({item!r})())\n")
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink), librarian=Always(lambda c: SW.librarian_answer([SW.fit(item)]))))
    assert item in sink["out"]["items"] and sink["out"]["call"] == ["KAO"], sink["out"]
    # an entry with no capture (a segment's, a summary-only fork's) runs no pass: the harness closes it (R1.4, 0.5)
    for e in s.entries():
        if e.kind == "new_experience" and not e.refs.captures:
            assert e.state == "closed" and e.reason_ref == "no_capture", (e.kind, e.state, e.reason_ref)
    s.settle()
    # the run's end: the drain hands no lone capture and the coverage report counts it never grouped
    report = H.drain(s.home)
    H.check_terminated(report, s.home)
    assert not s.writer.pass_bodies, "the drain handed a lone capture to a pass"
    rows = H.ledger(s.home).never_grouped()
    assert any(cid in json.dumps(r) for r in rows), rows
    assert H.ledger(s.home).report().never_grouped >= 1
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
