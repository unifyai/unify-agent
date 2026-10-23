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


#: Paths the writer's brief names for what exists only in some passes: a refused earlier pass's drafts and gate
#: results, and what a launched helper staged (the brief says when each exists).
WRITER_CONDITIONAL = ("/inputs/gate/", "/inputs/drafts/", "/inputs/helpers/")


def _writer_prep(body: dict) -> str:
    texts = {"writer first message": SW._first_user(body), "writer brief": SW._system(body)}
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
        f"CONDITIONAL = {WRITER_CONDITIONAL!r}\n"
    )
    return SW.probe_json(gather + SW.OPEN_PREP)


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
    s.writer.probe = SW.SteppedProbe(_writer_prep)  # step cells only while something is left to open
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


#: Secrets seeded into the library copy: a credential file by name, and a key-shaped string in an ordinary note.
ENV_SECRET = "seeded-env-value-7f3a9c"
KEY_SECRET = "sk-or-v1-" + "0123456789abcdef" * 4


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_library_reads_apply_the_sandbox_masks(system, mode):
    """R7.6 (the mask half): whatever path an actor reads the library by (``tools.read``, ``tools.grep``, the box's
    own files), the sandbox's masks apply: a credential file and a secret-shaped string in a seeded library never
    reach the actor."""
    s = system
    s.writer.plan = "land"
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")
    # the export copies only the library's listed items, so masked material cannot be seeded through main: it is
    # placed in the request's own copy, host side, once the box has mounted it (the mount shows it), as material a
    # landed item could carry
    seeded = {"notes/text/seed-control.md": "---\ntitle: control\n---\n## Rule\n- seed-control-visible\n",
              "notes/text/.env": f"TOKEN={ENV_SECRET}\n",
              "notes/text/keys.md": f"---\ntitle: keys\n---\n## Rule\n- the key is {KEY_SECRET}\n"}

    def seed(call):
        import os
        import stat

        for rel, text in seeded.items():
            p = s.copy() / rel
            mode = p.parent.stat().st_mode
            os.chmod(p.parent, mode | stat.S_IWUSR)  # the copy is read-only on disk too; the fixture opens it
            try:
                p.write_text(text)
            finally:
                os.chmod(p.parent, mode)
        return sorted(seeded)

    sink: dict = {}
    code = SW.probe_json(
        "_try('read_env', lambda: str(tools.read('/memory-library/notes/text/.env')))\n"
        "_try('read_keys', lambda: str(tools.read('/memory-library/notes/text/keys.md')))\n"
        f"_try('grep', lambda: str(tools.grep({ENV_SECRET[:12]!r}, '/memory-library')))\n"
        "_try('grep_key', lambda: str(tools.grep('sk-or-v1', '/memory-library')))\n"
        "import os\n"
        "def _walk():\n"
        "    got = []\n"
        "    for r, d, f in os.walk('/memory-library'):\n"
        "        for n in f:\n"
        "            try:\n"
        "                got.append(open(os.path.join(r, n), errors='replace').read())\n"
        "            except OSError as e:\n"
        "                got.append('unreadable ' + n)\n"
        "    return '\\n'.join(got)\n"
        "_try('box_files', _walk)\n"
        "_try('get_keys', lambda: str(memory.get('notes/text/keys.md')))\n"
        f"out['leaks'] = sorted(k for k, v in out.items() if {ENV_SECRET!r} in str(v) or {KEY_SECRET!r} in str(v))\n"
        "out['control'] = 'seed-control-visible' in str(out.get('box_files'))\n"
        "out['files'] = sorted(os.path.relpath(os.path.join(r, n), '/memory-library') for r, d, f in os.walk('/memory-library')"
        " for n in f if 'notes' in r)\n"
    )
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(code, sink, on_first=seed), librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    # the control: the seeded library is the one the box mounts (an ordinary seeded file is readable there)
    assert sink["out"]["control"], ("the seeded library is not the one the box mounts", sink["out"]["files"])
    assert not sink["out"]["leaks"], {k: str(sink["out"].get(k))[:300] for k in sink["out"]["leaks"]}
