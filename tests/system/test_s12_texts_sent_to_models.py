"""S12: what the harness sends every model in a memory run (the actor, its librarian, the fork, the writer, the
kinship judge) steers neither towards nor against storing specifics, and names no world.

Design: the steering audit (lead's hard rule, 23 Oct: no text pushes what is stored, either way), H5 (no single-world
bias); spec R12, H3, H5; audit A9, F2, W4, W9, W28, H6-5/H6-6 and guard J8 (steering and world guards on rendered
text, with paraphrases). Only the harness's own texts are scanned: the scripted models' turns and the test's own
request texts are left out.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

#: Steering towards storing or reusing specifics, and against using what is stored (both directions), with the
#: paraphrases the harness-truth audit found live.
STEERING = re.compile(
    r"could be reused|looks reusable|about to generalis|is knowledge too|every value the job needs"
    r"|every parameter read from it|derive every value|may hold only for that case|not yet reviewed|unreviewed"
    r"|has not been used|prefer (?:the )?curated|wait for the writer",
    re.I,
)
#: World words beyond the conformance guard's list (the audit's H6-6 additions).
MORE_WORLD = re.compile(r"\b(payment|payments|currency|money|tax|demonstration|demonstrations|submit|submission)\b",
                        re.I)


def _harness_texts(s) -> list[tuple[str, str]]:
    out = []
    for call in s.lanes.calls:
        msgs = [m for m in call.body.get("messages") or [] if not (isinstance(m, dict) and m.get("role") == "assistant")]
        out.append((call.lane, json.dumps(msgs, ensure_ascii=False) + json.dumps(call.body.get("tools") or [])))
    for model in s.models:
        for c in model.calls:
            msgs = [m for m in c.messages if m.get("role") != "assistant"]
            out.append((c.kind, json.dumps(msgs, ensure_ascii=False) + json.dumps(c.request.get("tools") or [])))
    return out


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_no_text_sent_to_any_model_steers_storage_or_names_a_world(system, mode):
    s = system
    s.writer.plan = "land"
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    await s.accepted_words(["glass"], "upper")  # a pass
    await s.accepted_tally()  # the judge
    item = f"capture:{a}"
    sink: dict = {}
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json(f"_try('card', lambda: str(memory.get({item!r})))\n"
                                           "_try('ask', lambda: str(memory.ask()))\n"), sink),
        librarian=Always(lambda c: SW.librarian_answer([SW.fit(item)]))))
    texts = _harness_texts(s)
    assert {k for k, _ in texts} >= {"fork", "writer", "actor", "librarian"}, {k for k, _ in texts}
    own = [m.group(0) for _, t in texts for m in re.finditer(r"WORDS|moves\.json|tally\.json", t)]
    steering = sorted({(k, m.group(0).lower()) for k, t in texts for m in STEERING.finditer(t)})
    assert not steering, steering
    world = H.world_words([t for _, t in texts], own=[SW.TALLY_REQUEST, SW.words_request(["x"]), *own])
    world += sorted({m.group(0).lower() for _, t in texts for m in MORE_WORLD.finditer(t)})
    assert not world, world[:20]
