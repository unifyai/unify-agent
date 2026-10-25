"""S1: an accepted episode is captured, served at once with a neutral card, and reused on a new request; its use is
recorded where the card, the librarian and trust read it.

Design: REV2.2 change 0R (served at once, neutral card, uses feed back at once), change a2 (a no-argument call re-runs
the work on the new request), b1 (the request-start block names it), f (uses counted from runtime calls), S4.2;
spec R3.1, R3.2, R3.3, R3.4, R4.2, R11.2; audit H5-1 and guard J6 (record contract: every field read was written).
"""

from __future__ import annotations

import json
import re

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

#: Wording that marks a stored entry as second-class or tells an agent to wait for curated items (0R.3).
DISCOURAGING = ("not yet reviewed", "unreviewed", "pending", "may hold only for that case", "has not been used",
                "on-trial", "on trial", "unchecked")
USE_LINE = re.compile(r"\b1\b[^\n]{0,60}\buses?\b|\buses?\b[^\n]{0,60}\b1\b", re.I)


def card_head(card: str) -> str:
    """The card's lines above the procedure's own source."""
    return card.split("\n\n", 1)[0]


async def _accepted_words_episode(s, words, op="reverse"):
    text = SW.words_request(words, op=op)
    model = ScriptedModel(actor=[reply(calls=[cell(SW.words_cell(words, op=op))]),
                                 reply(json.dumps(SW.words_answer(words, op=op))), reply("Good.")])
    r = await s.request(text, model, SW.accepted_host())
    s.settle()
    return r


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_capture_is_served_at_once_reused_with_no_arguments_and_its_use_is_recorded(system, mode):
    s = system
    s.writer.plan = "keep"  # the writer keeps nothing from its pass: what is served is the capture itself
    # ── request 1: accepted work; the fork captures it ──
    r1 = await _accepted_words_episode(s, ["stone", "river"])
    cid = s.capture_of(r1.episode_id)
    item = f"capture:{cid}"
    rec = s.captures()[cid]["record"]
    cases = [c for c in rec.get("cases") or [] if c.get("episode") == r1.episode_id]
    assert cases and cases[0].get("outcome") == "positive", rec
    # it keeps the code the fork wrote (R11.1), and nothing ran it to classify it (H7)
    assert "return [w[::-1].upper() for w in value]" in (s.captures()[cid].get("procedure") or ""), cid
    assert (cases[0].get("own_check") or {}).get("ran") is False, cases[0]
    # its request has ended, so it was handed to a pass, which kept nothing: what follows is served with nothing a
    # pass wrote (0R.1)
    assert s.writer.pass_bodies and not s.library_main_has(SW.NOTE_ID)

    # ── request 2: the start block names it; the actor calls it with no arguments on the NEW request ──
    w2 = ["glass", "cloud", "ember"]
    sink: dict = {}
    code = SW.probe_json(
        f"_try('items', lambda: [i.id for i in memory.items()])\n"
        f"_try('card', lambda: str(memory.get({item!r})))\n"
        f"_try('call', lambda: memory.get({item!r})())\n"
    )
    asked: list = []

    def librarian(call):
        asked.append(call)
        return SW.librarian_answer([SW.fit(item)])

    r2 = await s.request(SW.words_request(w2), ScriptedModel(actor=SW.probe_turns(code, sink),
                                                             librarian=Always(librarian)))
    from unify.memory.librarian import START_HEAD

    first = SW.first_user(sink["first"])
    assert START_HEAD in first and item in first, first[:2000]
    assert f"memory.get({item!r})()" in first, "the block gives the one call that runs it"
    out = sink["out"]
    assert item in out["items"], out["items"]
    head = card_head(out["card"])
    assert not [w for w in DISCOURAGING if w in head.lower()], head
    assert "accepted" in head.lower(), f"its episode was accepted and the card does not say so: {head}"
    assert "no arguments" in head.lower(), f"the card must say a no-argument call runs it on the request: {head}"
    assert out["call"] == SW.words_answer(w2), out["call"]  # it ran on request 2's words, not request 1's
    s.settle()

    # ── the use is recorded at once: the evidence store's runtime columns ──
    use = s.item_use(item)
    assert use.get("runtime_ok", 0) >= 1 and use.get("used_requests", 0) >= 1, use

    # ── request 3: the card, the librarian and trust show the use ──
    sink3: dict = {}
    code3 = SW.probe_json(f"_try('card', lambda: str(memory.get({item!r})))\n")
    lib_seen: dict = {}

    def lib_probe(call):
        return reply(calls=[cell(f"print(memory.get({item!r}))\n")])

    def lib_answer(call):
        lib_seen["card"] = SW.last_tools(call)
        return SW.librarian_answer([SW.fit(item)])

    await s.request(SW.words_request(["moss"]), ScriptedModel(
        actor=SW.probe_turns(code3, sink3, on_first=lambda call: s.item_evidence()),
        librarian=[lib_probe, lib_answer, Always(lambda c: SW.librarian_answer([SW.fit(item)]))]))
    ev = sink3["seen"].get(item) or {}
    counts = ev.get("counts") or {}
    assert sum(v for v in counts.values() if isinstance(v, int)) >= 1, f"served evidence shows no use: {ev}"
    assert not counts.get("failed"), ev
    head3 = card_head(sink3["out"]["card"])
    assert USE_LINE.search(head3) and not USE_LINE.search(head), f"the actor's card does not show the use:\n{head3}"
    assert USE_LINE.search(card_head(lib_seen.get("card", ""))), "the librarian's card does not show the use"
    uses = (s.captures()[cid]["record"] or {}).get("uses") or []
    assert any(u.get("episode") == r2.episode_id for u in uses if isinstance(u, dict)), \
        f"the capture's record (what trust reads) has no row for the use: {uses}"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_capture_whose_episode_has_no_verdict_never_claims_acceptance(system, mode):
    """R3.3: the card says only what the record shows; with no verdict on record it never says "accepted"."""
    s = system
    words = ["stone"]
    r1 = await s.request(SW.words_request(words), ScriptedModel(actor=[
        reply(calls=[cell(SW.words_cell(words))]), reply(json.dumps(SW.words_answer(words)))]))
    s.settle()
    item = f"capture:{s.capture_of(r1.episode_id)}"
    sink: dict = {}
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json(f"_try('card', lambda: str(memory.get({item!r})))\n"), sink),
        librarian=Always(lambda c: SW.librarian_answer([SW.fit(item)]))))
    head = card_head(sink["out"]["card"])
    assert "accepted" not in head.lower(), head
    assert "no arguments" in head.lower(), head
