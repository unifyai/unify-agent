"""S5: every path or handle shown to the actor, the librarian and the writer opens for that agent in its own real box,
and every refusal about the library names a call that works when the agent runs it as written.

Design: F (harness-first navigation: handles; one call per action; refusals name the next step; confinement unchanged),
F5 ``test_every_printed_path_opens`` / ``test_refusals_name_a_next_step``; spec R7.1-R7.6, R10.2; audit A7/A8/A20/H6-1
(library reads refused in every mode while the test mocks ``sandbox.host_path``), A19 (the run-by-hand recipe calls
``exec``) and guard J4 (unmocked path-openability).
"""

from __future__ import annotations

import pydoc  # noqa: F401 - named in the probe's text only

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

ACTOR_GATHER = (
    SW.GATHER_REFUSALS
    + "import pydoc\n"
    "_ids = [i.id for i in memory.items()]\n"
    "TEXTS = {'items': str(memory.items()), 'help': pydoc.plain(pydoc.render_doc(memory)), 'ask': str(memory.ask())}\n"
    "for _i in _ids:\n"
    "    TEXTS['card ' + _i] = str(memory.get(_i))\n"
    "REFUSALS = {\n"
    "    'read a missing library file': _err(lambda: tools.read('/memory-library/notes/none.md')),\n"
    "    'get an unknown handle': _err(lambda: memory.get('notes/none.md')),\n"
    "    'call an acting procedure': _err(lambda: memory.get(ACTING)()),\n"
    "}\n"
)


def _actor_prep(acting: str):
    def first_cell(call) -> str:
        host = {"system prompt": SW.system_text(call), "first message": SW.first_user(call)}
        return SW.probe_json(f"ACTING = {acting!r}\n" + ACTOR_GATHER + f"TEXTS.update({host!r})\n"
                             f"MISSING = {SW.PROBED_MISSING!r}\n" + SW.OPEN_PREP)
    return first_cell


def _writer_probe(body: dict) -> list[str]:
    texts = {"writer first message": SW._first_user(body)}
    gather = (
        SW.GATHER_REFUSALS
        + f"TEXTS = {texts!r}\n"
        "for _p in ('/inputs/README.md', '/inputs/attention/index.json'):\n"
        "    try:\n        TEXTS[_p] = str(tools.read(_p))\n    except BaseException as _e:\n"
        "        TEXTS[_p] = 'RAISED ' + str(_e)\n"
        "REFUSALS = {\n"
        "    'read a missing input': _err(lambda: tools.read('/inputs/attention/none/use.json')),\n"
        "    'read a missing library file': _err(lambda: tools.read('/memory/notes/none.md')),\n"
        "    'get an unknown capture': _err(lambda: memory.get('capture:0000000000000000')),\n"
        "}\n"
        f"MISSING = {SW.PROBED_MISSING!r}\n"
    )
    return [SW.probe_json(gather + SW.OPEN_PREP)] + [SW.OPEN_STEP_CELL] * 10


def _librarian_prep(call) -> str:
    texts = {"librarian brief": SW.system_text(call), "librarian question": SW.first_user(call)}
    gather = (SW.GATHER_REFUSALS + f"TEXTS = {texts!r}\n"
              "REFUSALS = {'read a missing library file': _err(lambda: tools.read('/memory/notes/none.md'))}\n"
              f"MISSING = {SW.PROBED_MISSING!r}\n")
    return SW.probe_json(gather + SW.OPEN_PREP)


def _problems(role: str, out: dict) -> list:
    """Everything wrong for one role: a path or handle shown to it that does not open for it, a refusal that names
    no call, a named call that fails when run as written."""
    got: list = []
    if out.get("left", 1) != 0 or out.get("opened", 0) <= 0:
        got.append((role, "not everything was opened", out.get("left"), out.get("opened")))
    got += [(role, "does not open", f["x"], "shown in " + str(f["from"]), f["err"][:200]) for f in out.get("fails") or []]
    for src, r in (out.get("refusals") or {}).items():
        if not r["text"]:
            got.append((role, src, "no refusal at all"))
        elif not r["calls"]:
            got.append((role, src, "the refusal names no call", r["text"][:300]))
        got += [(role, src, "named call fails", c, e[:200]) for c, e in r["calls"].items() if e != "ok"]
    return got


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_every_path_and_handle_shown_opens_and_refusals_name_a_working_call(system, mode):
    s = system
    s.writer.plan = "land"
    s.writer.probe = _writer_probe
    await s.accepted_words(["stone", "river"], "reverse")
    await s.accepted_words(["glass", "cloud"], "upper")  # the group's pass: the writer probes its own box
    acting = "capture:" + s.capture_of((await s.accepted_tally()).episode_id)
    assert s.writer.probes, "no writer pass ran"
    problems = _problems("writer", SW.marked(s.writer.probes[0]))

    actor: dict = {}
    lib: dict = {}
    item = f"capture:{s.capture_of(s.sessions[0].episode_id)}"
    answer = lambda call: SW.librarian_answer([SW.fit(item)])  # noqa: E731
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=Always(SW.opening_turns(_actor_prep(acting), actor)),
        librarian=Always(SW.opening_turns(_librarian_prep, lib, answer=answer))))
    assert "out" in actor and "out" in lib, (sorted(actor), sorted(lib))
    problems += _problems("actor", actor["out"]) + _problems("librarian", lib["out"])
    assert not problems, "\n".join(map(str, problems))
