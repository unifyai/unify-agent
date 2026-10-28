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
from tests.memory.conformance import harness as H
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


async def _tally_episode_unsettled(s, *, librarian):
    SW.seed_tally(s.workspace)
    model = ScriptedModel(actor=[reply(calls=[cell(SW.TALLY_CELL)]), reply("The tally is updated."), reply("Good.")],
                          librarian=librarian)
    return await s.request(SW.TALLY_REQUEST, model, SW.accepted_host())


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
    called it: no relation, no group, and no pass hands them together; each is handed alone once its own request
    has ended (R1.2: used items, never served ones; the pre-ff01d6158 root cause)."""
    s = system
    s.writer.plan = "land"
    s.writer.kin = False
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    # a pass lands the note (the second episode used the first capture): from here on every request is served it
    await s.accepted_words(["glass"], "upper", uses=a)
    assert s.library_main_has(SW.NOTE_ID)
    passes = len(s.writer.passes)
    served = Always(lambda call: SW.librarian_answer([SW.fit(SW.NOTE_ID, call=f"print(memory.get({SW.NOTE_ID!r}))")]))
    # this build writes no use row for an item that was only served, so the fixture writes the row a build that
    # records served items would write (shown, never called) for both requests, before their captures are grouped:
    # the relation must still not count it
    rc = await _tally_episode_unsettled(s, librarian=served)
    s.seed_served_row(rc.episode_id, SW.NOTE_ID)
    s.settle()
    c = s.capture_of(rc.episode_id)
    r = await s.request(SW.SUM_REQUEST, ScriptedModel(
        actor=[reply(calls=[cell(SW.SUM_CELL)]), reply("12"), reply("Good.")], librarian=served), SW.accepted_host())
    s.seed_served_row(r.episode_id, SW.NOTE_ID)
    s.settle()
    n = s.capture_of(r.episode_id)
    assert not _groups_with(s, _entries(s, c), _entries(s, n)), [g.get("reasons") for g in s.ledger_rows("group")]
    assert len(s.writer.passes) - passes <= 2, "more than one pass for each request that left a capture"
    led = H.ledger(s.home)
    ec, en = set(_entries(s, c)), set(_entries(s, n))
    handed: dict = {}
    for row in led.rows():
        if row.get("op") == "hand" and row.get("entry") in ec | en:
            handed.setdefault(row["pass_id"], []).append(row)
    assert ec <= {r["entry"] for rows in handed.values() for r in rows}, "a capture was never handed"
    assert en <= {r["entry"] for rows in handed.values() for r in rows}, "a capture was never handed"
    for pass_id, rows in handed.items():
        got = {r["entry"] for r in rows}
        assert not (got & ec and got & en), "a pass ran for captures that no real relation ties"
        row = led.group_of(pass_id)
        assert all(r.get("by") == "with_pass" for r in rows) or (row["group"] is None and row["due"] == "ended"), \
            (rows, row)


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


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_the_pass_is_shown_only_the_items_its_captures_used_or_are_linked_to(system, mode):
    """R6.7 (P): ``group.json``'s library items are those the group's captures called or are linked to; an item that
    was only served (a row the fixture writes, as for the served-item test) is not among them."""
    s = system
    s.writer.plan = "keep"
    s.writer.kin = "same_world"
    x = "capture:" + s.capture_of((await s.accepted_tally()).episode_id)
    a = "capture:" + s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    s.writer.probe = lambda body: (
        "import json, os\n"
        "docs = [json.load(open(os.path.join(r, 'group.json'))) for r, d, f in os.walk('/inputs') if 'group.json' in f]\n"
        "print('SYSJSON:' + json.dumps({'docs': docs}))\n")
    model = ScriptedModel(
        actor=[reply(calls=[cell(SW.words_cell(["glass"], op="mark1") + f"print(memory.get({a!r})())\n")]),
               reply(json.dumps(SW.words_answer(["glass"], op="mark1"))), reply("Good.")],
        librarian=Always(lambda c: SW.librarian_answer([SW.fit(x, call=f"print(memory.get({x!r}))")])))
    r = await s.request(SW.words_request(["glass"], op="mark1"), model, SW.accepted_host())
    s.seed_served_row(r.episode_id, x)  # served to this request, never called
    s.settle()
    assert s.writer.probes, "no pass ran for the group"
    docs = SW.marked(s.writer.probes[-1])["docs"]
    assert docs, "the pass was handed no group.json"
    items = set(docs[-1].get("library_items") or [])
    assert a in items, (sorted(items), "the item a member's episode called is the group's")
    assert x not in items, (sorted(items), "an item only served to a member's episode is not the group's")
