"""S2b: real relations only, and a bounded, ranked history.

* an observed call of a capture relates the calling episode's capture to it (R1.3, design 0R.4), with no other
  relation and a judge that names no kin;
* a capture that was only served (named in the request-start block, never called) relates nothing (R1.2: used items,
  not served ones);
* the history handed with a new capture holds at most 8 earlier captures (R1.5, R6.6, P: H = 8).
"""

from __future__ import annotations

import json

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES


def _groups_with(s, entries_a, entries_b) -> list[dict]:
    return [g for g in s.ledger_rows("group")
            if set(entries_a) & set(g.get("members", []) + g.get("context", []))
            and set(entries_b) & set(g.get("members", []) + g.get("context", []))]


def _entries(s, cid) -> list[str]:
    return [e.entry for e in s.entries() if cid in (e.refs.captures or ())]


async def _tally_episode(s, *, call: str | None, librarian):
    """An accepted tally episode whose actor first calls *call* (a capture handle) with no arguments, or not. A call
    needs a request the capture can read, so its request also carries a list of words; with no call it carries none
    (an equal input shape would relate the two captures by itself)."""
    SW.seed_tally(s.workspace)
    first = SW.TALLY_CELL + (f"try:\n    print(memory.get({call!r})())\nexcept Exception as e:\n    print(e)\n"
                             if call else "")
    model = ScriptedModel(actor=[reply(calls=[cell(first)]), reply("The tally is updated."), reply("Good.")],
                          librarian=librarian)
    text = SW.TALLY_REQUEST + ("\n" + SW.WORDS_MARK + json.dumps(["x"]) + "\nEND" if call else "")
    r = await s.request(text, model, SW.accepted_host())
    s.settle()
    return r


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_call_of_a_capture_relates_the_calling_episodes_capture_to_it(system, mode):
    s = system
    s.writer.kin = False  # the judge names no kin: only the harness's own relations can relate them
    s.writer.plan = "keep"
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    item = f"capture:{a}"
    c = s.capture_of((await _tally_episode(s, call=item, librarian=Always(
        lambda call: SW.librarian_answer([SW.fit(item)])))).episode_id)
    assert s.item_use(item).get("runtime_ok", 0) + s.item_use(item).get("runtime_errored", 0) >= 1
    groups = _groups_with(s, _entries(s, a), _entries(s, c))
    assert groups and groups[0]["reasons"].get("items", 0) >= 1, s.ledger_rows("group")


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_an_item_only_served_relates_nothing(system, mode):
    """Two captures of unrelated work, from two requests that were both served the same library item and never
    called it: no relation, no group, no pass (R1.2: used items, never served ones; the pre-ff01d6158 root cause)."""
    s = system
    s.writer.plan = "land"
    s.writer.kin = False
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")  # a pass lands the note: from here on every request is served it
    assert s.library_main_has(SW.NOTE_ID)
    passes = len(s.writer.passes)
    served = Always(lambda call: SW.librarian_answer([SW.fit(SW.NOTE_ID, call=f"print(memory.get({SW.NOTE_ID!r}))")]))
    c = s.capture_of((await _tally_episode(s, call=None, librarian=served)).episode_id)
    r = await s.request(SW.SUM_REQUEST, ScriptedModel(
        actor=[reply(calls=[cell(SW.SUM_CELL)]), reply("12"), reply("Good.")], librarian=served), SW.accepted_host())
    s.settle()
    n = s.capture_of(r.episode_id)
    assert not _groups_with(s, _entries(s, c), _entries(s, n)), [g.get("reasons") for g in s.ledger_rows("group")]
    assert len(s.writer.passes) == passes, "a pass ran for captures that no real relation ties"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", ["on"])
async def test_the_history_handed_with_a_new_capture_is_at_most_eight(system, mode):
    s = system
    s.writer.plan = "keep"
    s.writer.kin = True
    for k in range(1, 12):
        await s.accepted_words([f"w{k}"], f"mark{k}")
    groups = s.ledger_rows("group")
    assert groups and len(s.captures()) >= 11, (len(groups), len(s.captures()))
    assert all(len(g.get("context") or []) <= 8 for g in groups), [len(g.get("context") or []) for g in groups]
    assert max(len(g.get("members") or []) + len(g.get("context") or []) for g in groups) >= 2, groups
