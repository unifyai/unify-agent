"""S11: no prompt flooding, and one library with two labelled kinds, as an actor and a librarian see them.

Design: P (a note is served as its Rule, ≤ 2,000 characters; a card ≤ 2,000 characters with code only on explicit
request; the librarian gets a fresh context per question in a request), S4/R4.1 (one listing, two labelled kinds);
spec R4.1, R6.3, R6.4, R6.5 (R6.4 and R6.5 are the builder's declared deviations 4d and 4c: these tests fail until
they are built).
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES


async def _library(s):
    """Two related captures and the note a pass lands from them (with its long evidence appendix)."""
    s.writer.plan = "land"
    a = s.capture_of((await s.accepted_words(["stone", "river"], "reverse")).episode_id)
    b = s.capture_of((await s.accepted_words(["glass", "cloud"], "upper")).episode_id)
    assert s.library_main_has(SW.NOTE_ID), [p.get("items") for p in s.pass_results()]
    return f"capture:{a}", f"capture:{b}"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_one_listing_two_labelled_kinds_and_a_note_served_as_its_rule(system, mode):
    s = system
    a, b = await _library(s)
    sink: dict = {}
    code = SW.probe_json("_try('listing', lambda: str(memory.items()))\n"
                         f"_try('note', lambda: str(memory.get({SW.NOTE_ID!r})))\n")
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink), librarian=Always(lambda c: SW.librarian_answer([SW.fit(a)]))))
    out = sink["out"]
    rows = {i: " ".join(ln for ln in out["listing"].splitlines() if i in ln).lower() for i in (a, b, SW.NOTE_ID)}
    assert all(rows.values()), (rows, out["listing"][:2000])
    assert "recorded procedure" in rows[a] and "recorded procedure" in rows[b], rows
    assert "recorded procedure" not in rows[SW.NOTE_ID], rows
    assert "the list follows the line WORDS" in out["note"], out["note"][:1500]
    assert "case 119" not in out["note"] and len(out["note"]) <= 2000, (len(out["note"]), out["note"][-600:])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_card_is_bounded_and_shows_code_only_on_request(system, mode):
    s = system
    a, _ = await _library(s)
    sink: dict = {}
    code = SW.probe_json(f"_try('card', lambda: str(memory.get({a!r})))\n"
                         f"_try('source', lambda: memory.get({a!r}).text)\n")
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink), librarian=Always(lambda c: SW.librarian_answer([SW.fit(a)]))))
    out = sink["out"]
    assert "def read_input" in str(out["source"]), "the code is reachable on explicit request"
    assert len(out["card"]) <= 2000, len(out["card"])
    assert "def read_input" not in out["card"], "the card prints the code the actor did not ask for"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_the_librarian_gets_a_fresh_context_for_each_question(system, mode):
    s = system
    a, b = await _library(s)
    calls: list = []

    def librarian(call):
        calls.append(call)
        return SW.librarian_answer([SW.fit(a)])

    q1, q2 = "Which entry reverses words?", "Which entry writes words in capitals?"
    sink: dict = {}
    code = SW.probe_json(f"_try('q1', lambda: str(memory.ask({q1!r})))\n_try('q2', lambda: str(memory.ask({q2!r})))\n")
    await s.request(SW.words_request(["oak"]), ScriptedModel(actor=SW.probe_turns(code, sink),
                                                             librarian=Always(librarian)))
    import json

    second = [c for c in calls if q2 in json.dumps(c.messages)]
    assert second, "the second question never reached the librarian"
    assert q1 not in json.dumps(second[0].messages), "the second question's context carries the first question's"
