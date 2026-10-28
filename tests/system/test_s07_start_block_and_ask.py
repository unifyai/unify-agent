"""S7: the request-start block and every ``memory.ask`` answer are one primary and at most one fallback labelled for
when the first fails, never a menu, and bounded.

Design: Q1 (one primary, never a menu; amendment 4: the block is the same answer), b1 (block ≤ 2 entries, ≤ 1,500
characters, in the first user message, the system prompt unchanged), P (an ask answer ≤ 2,000 characters); spec
R4.5, R6.1, R6.2; audit A4/A14/A21 (``memory.ask()`` renders "n item(s) apply:").
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

LONG = "it fits because " + "the request carries a list of words to transform " * 60


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_the_block_and_the_ask_answer_are_one_primary_and_one_fallback_bounded(system, mode):
    from unify.memory.librarian import START_HEAD

    s = system
    s.writer.plan = "keep"
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    b = s.capture_of((await s.accepted_words(["glass"], "upper")).episode_id)
    c = s.capture_of((await s.accepted_words(["moss"], "title")).episode_id)
    ids = [f"capture:{x}" for x in (a, b, c)]
    seen: list = []

    def librarian(call):
        seen.append(SW.last_tools(call))
        if len(seen) == 1:  # a call by path, not through memory: the harness must refuse it
            return SW.librarian_answer([SW.fit(ids[0], "it fits", call="tools.read('/memory/notes/x.md')")])
        if len(seen) == 2:  # a menu of three: the harness must refuse it
            return SW.librarian_answer([SW.fit(i, LONG) for i in ids])
        return SW.librarian_answer([SW.fit(ids[0], LONG), SW.fit(ids[1], LONG)])

    sink: dict = {}
    code = SW.probe_json("_try('ask', lambda: str(memory.ask()))\n"
                         "_try('ask_q', lambda: str(memory.ask('which entry transforms the words?')))\n")
    await s.request(SW.words_request(["oak", "elm"]), ScriptedModel(actor=SW.probe_turns(code, sink),
                                                                     librarian=Always(librarian)))
    # the call by path, then the menu of three, were refused back to the librarian
    assert len(seen) >= 3 and "through memory" in seen[1] and "at most" in seen[2], seen[1:3]
    # ── the block: in the first user message, one primary + one fallback labelled, ≤ 1,500 characters ──
    first = SW.first_user(sink["first"])
    assert START_HEAD in first, first[:1500]
    assert START_HEAD not in SW.system_text(sink["first"]), "the block must not enter the cached system prompt"
    block = first[first.index(START_HEAD):]
    named = [i for i in ids if i in block]
    assert named == ids[:2], (named, block)
    assert "only if the first fails" in block and block.index(ids[0]) < block.index("only if the first fails") \
        < block.index(ids[1]), block
    assert len(block) <= 1500, len(block)
    # ── the ask answers: the same form, never a menu, ≤ 2,000 characters ──
    for key in ("ask", "ask_q"):
        text = sink["out"][key]
        assert not text.startswith("RAISED"), text
        assert "item(s) apply" not in text and "items apply" not in text, f"{key} renders a menu:\n{text}"
        assert ids[0] in text and ids[1] in text and ids[2] not in text, text
        assert "only if the first fails" in text and text.index(ids[0]) < text.index(ids[1]), text
        assert len(text) <= 2000, (key, len(text))
