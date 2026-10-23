"""S4: a failed use of a capture is recorded at once, is not offered again in the same request, opens a writer
attention entry, and reaches the card and the librarian on the next request.

Design: Q2.1-Q2.3 (cards show their evidence; a failed or refused use is recorded at once; within the request it is
never the primary again; it opens a ``use_went_badly`` entry), change f (runtime calls counted), 0R.4; spec R3.4,
R5.1, R5.2, R5.3; audit H5-1 (capture uses never reach the card) and guard J6.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

#: A request of the same kind that carries no list, so the recorded procedure's reader fails on it.
NO_LIST = "Change the words to capitals; this time no list is given."


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_failed_use_is_recorded_not_reoffered_opens_an_entry_and_reaches_the_card(system, mode):
    s = system
    s.writer.plan = "keep"
    r1 = await s.accepted_words(["stone", "river"], "reverse")
    cid = s.capture_of(r1.episode_id)
    item = f"capture:{cid}"

    # ── request 2: the call fails; asked again in the same request, the librarian may not offer it again ──
    lib_turns: list = []

    def librarian(call):
        lib_turns.append(SW.last_tools(call))
        if "tried and failed" in (lib_turns[-1] or "") or len(lib_turns) > 2:
            return SW.librarian_answer([], nearest=[{"id": item, "differs": "its reader found no list"}])
        return SW.librarian_answer([SW.fit(item)])

    sink: dict = {}
    code = SW.probe_json(
        f"_try('call', lambda: memory.get({item!r})())\n"
        "_try('ask', lambda: [i.id for i in memory.ask('which entry can do this request now?')])\n"
    )
    r2 = await s.request(NO_LIST, ScriptedModel(actor=SW.probe_turns(code, sink), librarian=Always(librarian)))
    out = sink["out"]
    assert str(out["call"]).startswith("RAISED"), out
    s.settle()

    # ── recorded at once in the evidence store, and a writer entry about it ──
    use = s.item_use(item)
    assert (use.get("runtime_errored", 0) + use.get("runtime_refused", 0)) >= 1, use
    bad = [e for e in s.entries() if e.kind == "use_went_badly" and item in (e.refs.items or ())]
    assert bad and r2.episode_id in bad[0].refs.episodes, [(e.kind, e.refs.row()) for e in s.entries()]

    # ── request 3: the card and the librarian show the failure ──
    sink3: dict = {}
    lib3: dict = {}

    def lib_probe(call):
        return reply(calls=[cell(f"print(memory.get({item!r}))\n")])

    def lib_answer(call):
        lib3["card"] = SW.last_tools(call)
        return SW.librarian_answer([], nearest=[{"id": item, "differs": "it failed before"}])

    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json(f"_try('card', lambda: str(memory.get({item!r})))\n"), sink3,
                             on_first=lambda call: s.item_evidence()),
        librarian=[lib_probe, lib_answer]))
    ev = sink3["seen"].get(item) or {}
    assert (ev.get("counts") or {}).get("failed", 0) >= 1 and ev.get("failure_record"), ev
    card = sink3["out"]["card"]
    assert "Recorded failures: none" not in card and "Recorded failures: 1" in card, card[:1500]
    assert "Recorded failures: 1" in lib3.get("card", ""), lib3.get("card", "")[:1500]
    # within the request that failed, the failed item was never offered again (Q2.2: the same-request rule)
    assert item not in (out["ask"] if isinstance(out["ask"], list) else []), \
        f"the failed item was offered again in the same request: {out['ask']}"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_retire_is_reachable_and_withdraws_without_deleting(system, mode):
    """R9.6 (change f, 0R.5): the writer, handed the failed use, retires the capture by a decision; it leaves every
    listing at once, its files and record stay, and the ledger keeps the decision with its reason."""
    s = system
    s.writer.plan = "retire_bad"
    cid = s.capture_of((await s.accepted_words(["stone", "river"], "reverse")).episode_id)
    item = f"capture:{cid}"
    sink: dict = {}
    await s.request(NO_LIST, ScriptedModel(
        actor=SW.probe_turns(SW.probe_json(f"_try('call', lambda: memory.get({item!r})())\n"), sink),
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    assert str(sink["out"]["call"]).startswith("RAISED"), sink["out"]
    s.settle()
    bad = [e for e in s.entries() if e.kind == "use_went_badly" and item in (e.refs.items or ())]
    assert bad and bad[0].state == "closed" and bad[0].decision == "retire", [(e.kind, e.state, e.decision) for e in bad]
    sink2: dict = {}
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json("_try('items', lambda: [i.id for i in memory.items()])\n"), sink2),
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    assert item not in sink2["out"]["items"], sink2["out"]["items"]
    assert (s.captures().get(cid) or {}).get("procedure"), "a retired capture's files stay on record"
