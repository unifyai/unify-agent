"""S8: a teammate the actor starts works on the actor's request: its ``tools.request()`` and ``memory.request()``, a
recorded procedure called with no arguments, and its ``memory.ask`` all use the request, never the delegation text.

Design: change d (``tools.request()`` in the actor's and its helpers' cells), e (helpers reach the librarian, the same
way as the actor), F3 (one way to do each thing, the same calls for every allowed role); spec R7.5, R8.1, R9.3;
audit T1 (teammates run stored procedures and ``memory.ask()`` on their delegation text).
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

TASK = "Check the words you are given and say what you found."
WAIT = (
    "got = ''\n"
    "for _ in range(8):\n"
    "    got += str(tools.wait(timeout=120))\n"
    "    if '(done)' in got or '(reply)' in got:\n"
    "        break\n"
    "print(got[-1500:])\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_a_teammate_uses_the_actors_request(system, mode):
    s = system
    cid = s.capture_of((await s.accepted_words(["stone", "river"], "reverse")).episode_id)
    item = f"capture:{cid}"
    words = ["glass", "cloud", "ember"]
    request = SW.words_request(words)
    questions: list = []

    def librarian(call):
        questions.append(SW.first_user(call) + "\n" + "\n".join(SW.flat(m.get("content")) for m in call.messages
                                                                 if m.get("role") == "user"))
        return SW.librarian_answer([SW.fit(item)])

    helper: dict = {}
    probe = SW.probe_json(
        "_try('tools_request', lambda: str(tools.request()))\n"
        "_try('memory_request', lambda: str(memory.request()))\n"
        f"_try('call', lambda: memory.get({item!r})())\n"
        "_try('ask', lambda: str(memory.ask()))\n"
    )

    def h_first(call):
        helper["first"] = call
        return reply(calls=[cell(probe)])

    def h_second(call):
        helper["raw"] = SW.last_tools(call)
        try:
            helper["out"] = SW.marked(helper["raw"])
        except AssertionError:
            pass
        return reply(calls=[cell("tools.answer('checked')\n")])

    from tests.scripted_model import LIBRARIAN_FRAGMENT

    steps = [h_first, h_second]

    def helper_kind(call):
        # the scripted model classes a request by its text: a teammate's librarian session quotes the teammate's
        # "a helper in a team" text, so it arrives here; it is answered as the librarian
        if LIBRARIAN_FRAGMENT in SW.system_text(call):
            return librarian(call)
        return steps.pop(0)(call) if steps else reply("done")

    model = ScriptedModel(
        actor=[reply(calls=[cell(f"print(tools.delegate({TASK!r}))\n")]), reply(calls=[cell(WAIT)]), reply("Done.")],
        helper=Always(helper_kind),
        librarian=Always(librarian),
    )
    await s.request(request, model)
    out = helper.get("out")
    assert out, ("the teammate's probe gave no output", helper.get("raw", "")[-3000:],
                 [(c.kind, [(m.get("role"), str(m.get("content"))[:80]) for m in c.messages][-3:]) for c in model.calls])
    # the teammate reaches the request's librarian the same way as the actor (R8.1, change e)
    assert not str(out["ask"]).startswith("RAISED") and item in str(out["ask"]), out["ask"]
    assert questions, "the teammate's ask never reached the librarian"
    # ... and everything it does is on the actor's request, never on its delegation text (audit T1)
    asked_on = questions[-1]
    assert all(w in asked_on for w in words) and TASK not in asked_on, asked_on[:1500]
    for key in ("tools_request", "memory_request"):
        assert SW.WORDS_MARK.strip() in out[key] and all(w in out[key] for w in words), (key, out[key][:600])
        assert TASK not in out[key], (key, out[key][:600])
    assert out["call"] == SW.words_answer(words), out["call"]
