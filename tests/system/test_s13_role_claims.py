"""S13: the checkers of the claims the writer's helpers, the writer's capture tools, the shared refusal templates and
the librarian's notes and search make (guard J1's rows that had no checker), each on the real harness, and the lead's
ruling on retire (R9.6: a decision takes effect, or is refused with its reason).

Each test runs a real writer pass (the writer, its helpers and the gate, from the service), or a real request (the
actor and its librarian), with only the models and the worlds scripted, and asserts what the sentence says.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

ROOTS = re.compile(r"(?<![\w.:/-])/(inputs|task|memory-library|memory|workspace|work|outputs)\b")
#: In-cell: every root a refusal text names, read with the agent's own tools.read (it must open for that agent).
ROOT_CHECK = (
    "import re as _re\n"
    "def _roots_open(texts):\n"
    "    bad = {}\n"
    "    for src, t in texts.items():\n"
    "        for r in sorted(set(_re.findall(r'(?<![\\w.:/-])(/(?:inputs|task|memory-library|memory|workspace|work|outputs))\\b', t))):\n"
    "            try:\n"
    "                tools.read(r)\n"
    "            except BaseException as e:\n"
    "                bad.setdefault(src, []).append(r + ': ' + str(e)[:120])\n"
    "    return bad\n"
)

HELPER_PROBE = SW.probe_json(
    SW.GATHER_REFUSALS + ROOT_CHECK
    + "import os, socket\n"
    "def _write(root):\n"
    "    try:\n"
    "        open(root + '/probe-write.txt', 'w').write('x')\n"
    "        return 'ok'\n"
    "    except OSError as e:\n"
    "        return 'refused ' + type(e).__name__\n"
    "out['writes'] = {r: _write(r) for r in ('/inputs', '/task', '/memory', '/work')}\n"
    "_try('task_eps', lambda: sorted(os.listdir('/task/episodes')))\n"
    "_try('read_task', lambda: str(tools.read('/task/episodes/' + sorted(os.listdir('/task/episodes'))[0]))[:300])\n"
    "_try('read_inputs', lambda: str(tools.read('/inputs'))[:300])\n"
    "_try('grep_inputs', lambda: str(tools.grep('summary', '/inputs'))[:300])\n"
    "_try('read_memory', lambda: str(tools.read('/memory'))[:300])\n"
    "_try('memory_bound', lambda: str(memory.items())[:300])\n"
    "out['dirs'] = {d: os.path.isdir(d) for d in ('/inputs/staging', '/inputs/pool')}\n"
    "def _net():\n"
    "    c = socket.socket(); c.settimeout(3); c.connect(('203.0.113.7', 443)); return 'connected'\n"
    "_try('network', _net)\n"
    "REF = {'outside': _err(lambda: tools.read('/etc/hostname')), 'missing': _err(lambda: tools.read('/work/none.md'))}\n"
    "out['refusals'] = REF\n"
    "out['roots_not_open'] = _roots_open(REF)\n"
)


def _failed_host():
    return H.replies([{"checker": {"label": "fail"}}, {"message": "The check failed."}])


async def _failed_words(s, words, op):
    model = ScriptedModel(actor=[reply(calls=[cell(SW.words_cell(words, op=op))]),
                                 reply(json.dumps(SW.words_answer(words, op=op))), reply("I see.")],
                          librarian=Always(lambda c: SW.librarian_answer([], nearest=[])))
    r = await s.request(SW.words_request(words, op=op), model, _failed_host())
    s.settle()
    return r


# ── the writer's helpers: their box and roots ─────────────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_a_writers_helper_has_the_box_and_roots_its_texts_state(system, mode):
    """The helper's brief: "Your box: /inputs (read-only); /task (read-only); /memory (read-only); /work (writable)";
    "They also read /task, the records of the episodes your task names (/task/episodes/<id>.json), and /memory";
    "`memory` (the library under /memory) is already bound"; "`execute_code` runs Python in your own box: no network";
    "`tools.read` and `tools.grep` read /inputs"; "/inputs/staging and /inputs/pool hold ... drafts"."""
    s = system
    s.writer.plan = "keep"
    s.writer.probe = [SW.DELEGATE, SW.WAIT_HELPER]
    s.helper = SW.HelperLane(HELPER_PROBE)
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")
    assert s.helper.out, ("no helper ran", [SW.tool_texts(b)[-1:] for b in s.writer.pass_bodies][:3])
    out = SW.marked(s.helper.out[0])
    assert out["writes"] == {"/inputs": out["writes"]["/inputs"], "/task": out["writes"]["/task"],
                             "/memory": out["writes"]["/memory"], "/work": "ok"}, out["writes"]
    assert all(out["writes"][r].startswith("refused") for r in ("/inputs", "/task", "/memory")), out["writes"]
    assert out["task_eps"] and not str(out["read_task"]).startswith("RAISED"), (out["task_eps"], out["read_task"])
    for k in ("read_inputs", "grep_inputs", "read_memory", "memory_bound"):
        assert not str(out[k]).startswith("RAISED"), (k, out[k])
    assert str(out["network"]).startswith("RAISED"), out["network"]
    assert all(out["dirs"].values()), out["dirs"]


# ── the writer's capture tools ────────────────────────────────────────────────────────────────────────────────────

PASS1 = SW.probe_json(
    SW.GATHER_REFUSALS
    + "import json, os, shutil\n"
    "idx = json.load(open('/inputs/attention/index.json'))\n"
    "caps = sorted({c for r in idx['entries'] for c in ((r.get('refs') or {}).get('captures') or [])})\n"
    "def _outcome(c):\n"
    "    try:\n"
    "        return json.load(open('/inputs/captures/' + c + '/record.json')).get('outcome')\n"
    "    except OSError:\n"
    "        return None\n"
    "outs = {c: _outcome(c) for c in caps}\n"
    "out['outcomes'] = outs\n"
    "good = [c for c in caps if outs[c] == 'positive']\n"
    "bad = [c for c in caps if outs[c] in ('negative', 'contrary')]\n"
    "_try('land', lambda: str(tools.land_capture(good[0], 'words', 'reverse_words')))\n"
    "out['landed_files'] = sorted(os.path.relpath(os.path.join(r, n), '/memory') for r, d, f in os.walk('/memory/memory/words')"
    " for n in f)\n"
    "out['manifest'] = open('/memory/.pass/manifest.json').read() if os.path.exists('/memory/.pass/manifest.json') else ''\n"
    "shutil.rmtree('/memory/memory/words', ignore_errors=True)\n"
    "out['unknown'] = _err(lambda: tools.land_capture('0000000000000000', 'w', 'x'))\n"
    "out['unknown_wrote'] = os.path.exists('/memory/memory/w')\n"
    "out['failed'] = _err(lambda: tools.land_capture(bad[0], 'w2', 'y')) if bad else 'NO FAILED CAPTURE'\n"
    "out['failed_wrote'] = os.path.exists('/memory/memory/w2')\n"
    "_try('find', lambda: json.loads(str(tools.find('transform the words of a list', scope='summaries'))))\n"
    "out['batch_episodes'] = sorted({e for r in idx['entries'] for e in ((r.get('refs') or {}).get('episodes') or [])})\n"
    "_try('overview', lambda: str(tools.overview()))\n"
    "if os.path.exists('/memory/.pass/manifest.json'):\n"
    "    os.remove('/memory/.pass/manifest.json')\n"
)
PASS2 = SW.probe_json(
    "import json\n"
    "g = json.load(open('/inputs/attention/group.json'))\n"
    "closed = [m['entry'] for m in g.get('related_closed') or []]\n"
    "out['closed'] = closed\n"
    "_try('take', lambda: json.loads(str(tools.take(closed[0]))))\n"
)


def _capture_tools_probe(body: dict) -> str:
    return PASS2 if "an earlier pass decided" in SW._first_user(body) else PASS1


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mode", ["on"])
async def test_the_writers_capture_tools_do_what_their_texts_state(system, mode):
    """``land_capture`` "turns one into library items you can change before you finish" (a reader, a procedure, its
    tests and the manifest entries); an unknown id "is not a capture of the store; nothing was written"; a failed
    capture "is a failed or retired attempt; it is not landed"; ``find`` returns ``{"scope", "ranked", "hits": [{"id",
    "kind", "path", "score", "card"}], "of"}``; taking a closed entry "exports its material read-only and returns what
    was decided; nothing about it is owed"."""
    s = system
    s.writer.plan = "keep"
    s.writer.probe = _capture_tools_probe
    s.writer.kin = "same_world"
    await _failed_words(s, ["moss"], "title")
    await s.accepted_words(["stone"], "reverse")  # pass 1: the failed and the accepted capture together
    await s.accepted_words(["glass"], "upper")  # pass 2: a new capture with the decided ones as history
    assert len(s.writer.probes) >= 2, (len(s.writer.probes), s.writer.passes)
    one, two = SW.marked(s.writer.probes[0]), SW.marked(s.writer.probes[1])
    assert "positive" in one["outcomes"].values(), one["outcomes"]
    assert not one["land"].startswith("RAISED"), one["land"]
    files = one["landed_files"]
    assert any(f.endswith("readers.py") for f in files) and any(f.endswith("procedures.py") for f in files), files
    assert any("test_reverse_words" in f for f in files), files
    assert "reverse_words" in one["manifest"], one["manifest"][:600]
    assert "not a capture of the store" in one["unknown"] and not one["unknown_wrote"], one["unknown"]
    assert "failed or retired attempt" in one["failed"] and not one["failed_wrote"], one["failed"]
    found = one["find"]
    assert isinstance(found, dict) and {"scope", "ranked", "hits", "of"} <= set(found), found
    assert found["hits"] and all({"id", "kind", "path", "score", "card"} <= set(h) for h in found["hits"]), found
    # overview with no episode: one line per batch episode, from /inputs/batch_map.json
    assert all(any(e in ln for ln in one["overview"].splitlines()) for e in one["batch_episodes"]), \
        (one["batch_episodes"], one["overview"][:800])
    take = two["take"]
    assert isinstance(take, dict) and take.get("taken") == "read-only" and take.get("decision"), take
    entry = H.ledger(s.home).entries().get(two["closed"][0])
    assert entry is not None and entry.state == "closed" and entry.decision == take["decision"], entry
    assert not [e for e in s.entries() if e.active and e.entry == two["closed"][0]], "a taken closed entry is owed"


# ── the shared refusal templates: each role is refused in its own roots ──────────────────────────────────────────

READ_FIRST = "import json\n"
WRITER_REFUSALS = SW.probe_json(
    SW.GATHER_REFUSALS + ROOT_CHECK
    + "REF = {'outside': _err(lambda: tools.read('/etc/hostname')), 'missing': _err(lambda: tools.read('/inputs/none.md'))}\n"
    "out['refusals'] = REF\n"
    "out['roots_not_open'] = _roots_open(REF)\n"
)
RETIRE_UNKNOWN = SW.probe_json(
    SW.GATHER_REFUSALS + READ_FIRST
    + "out['retire_unknown'] = _err(lambda: tools.decide([json.load(open('/inputs/attention/index.json'))['entries'][0]['entry']],"
    " 'retire', items=['notes/none/nothing.md'], reason='a decision that cannot take effect'))\n"
)
ACTOR_REFUSALS = SW.probe_json(
    SW.GATHER_REFUSALS + ROOT_CHECK
    + "REF = {'outside': _err(lambda: tools.read('/etc/hostname')),"
    " 'missing': _err(lambda: tools.read('/workspace/none.md'))}\n"
    "out['refusals'] = REF\n"
    "out['roots_not_open'] = _roots_open(REF)\n"
)
LIBRARIAN_REFUSALS = (
    SW.GATHER_REFUSALS + ROOT_CHECK
    + "REF = {'outside': _err(lambda: tools.read('/etc/hostname')), 'missing': _err(lambda: tools.read('/memory/none.md'))}\n"
    "out['refusals'] = REF\n"
    "out['roots_not_open'] = _roots_open(REF)\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_each_roles_refusals_name_only_roots_it_can_open(system, mode):
    """The shared refusal templates ("<path> is outside /inputs, /memory and /outputs ...", "... outside /inputs,
    /memory and /work", "... outside the library, /memory", "<path> does not exist; ..."): what each role is actually
    refused with names only roots that role can read."""
    s = system
    s.writer.plan = "keep"
    s.writer.probe = [WRITER_REFUSALS, SW.DELEGATE, SW.WAIT_HELPER]
    s.helper = SW.HelperLane(HELPER_PROBE)
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    await s.accepted_words(["glass"], "upper")
    assert s.writer.probes and s.helper.out, "no pass or no helper ran"
    writer = SW.marked(s.writer.pass_bodies and SW.tool_texts(s.writer.pass_bodies[1])[-1])
    helper = SW.marked(s.helper.out[0])
    actor: dict = {}
    lib: dict = {}

    def l_first(call):
        return reply(calls=[cell(SW.probe_json(LIBRARIAN_REFUSALS))])

    def l_second(call):
        lib["out"] = SW.marked(SW.last_tools(call))
        return SW.librarian_answer([SW.fit(f"capture:{a}")])

    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(ACTOR_REFUSALS, actor),
        librarian=[l_first, l_second, Always(lambda c: SW.librarian_answer([SW.fit(f"capture:{a}")]))]))
    roles = {"writer": writer, "helper": helper, "actor": actor["out"], "librarian": lib["out"]}
    bad = {r: o["roots_not_open"] for r, o in roles.items() if o["roots_not_open"]}
    empty = {r: k for r, o in roles.items() for k, t in o["refusals"].items() if not t}
    assert not empty, ("a refusal expected and none came", empty)
    assert not bad, bad


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_a_decision_that_cannot_take_effect_is_refused_with_its_reason(system, mode):
    """The lead's ruling on R9.6: a writer decision that cannot take effect (retiring an item the library does not
    hold) is refused, naming why; it is never recorded as if it had."""
    s = system
    s.writer.plan = "keep"
    s.writer.probe = [SW.READ_ALL, RETIRE_UNKNOWN]  # the material read first, so only the decision can be wrong
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")
    assert s.writer.probes, "no pass ran"
    got = SW.marked(s.writer.probes[0])["retire_unknown"]
    assert got and "notes/none/nothing.md" in got, ("the decision was accepted, or refused without its reason", got)
    assert not [e for e in s.entries() if e.decision == "retire"], "a retire that cannot take effect was recorded"


# ── the librarian's notes and search ─────────────────────────────────────────────────────────────────────────────

NOTE = "Word lists are read by the recorded procedures; search for the words of the request."


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_the_librarians_notes_reach_a_later_librarian_and_search_lists_every_item(system, mode):
    """``keep_note``: "A later librarian of this run reads every note when it starts"; ``search``: "Rank every item of
    the library (functions, notes and captured procedures) ...; else every item is listed ..., and the result says
    which"; "Each line is one item: its rank, its path under /memory (for read), its id, kind ..."."""
    s = system
    s.writer.plan = "land"
    a = "capture:" + s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    await s.accepted_words(["glass"], "upper")
    kept: dict = {}

    def keep(call):
        return reply(calls=[cell(f"print(tools.keep_note({NOTE!r}))\n")])

    def answer(call):
        kept["reply"] = SW.last_tools(call)
        return SW.librarian_answer([SW.fit(a)])

    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=[reply("Noted.")], librarian=[keep, answer, Always(lambda c: SW.librarian_answer([SW.fit(a)]))]))
    assert "kept" in kept.get("reply", ""), kept
    later: dict = {}

    def search(call):
        later["first"] = json.dumps(call.messages)
        return reply(calls=[cell("print(tools.search('reverse the words of a list'))\n")])

    def result(call):
        later["search"] = SW.last_tools(call)
        return SW.librarian_answer([SW.fit(a)])

    sink: dict = {}
    await s.request(SW.words_request(["elm"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json(
            "_try('items', lambda: [i.id for i in memory.items()])\n"
            # memory.get takes "an id, ``module:name``, a note's slug or ``topic/slug``, a path under ``/memory``, the
            # function itself, or an Item"
            f"_try('by_slug', lambda: str(memory.get('text/words'))[:80])\n"
            f"_try('by_path', lambda: str(memory.get('/memory/{SW.NOTE_ID}'))[:80])\n"
            f"_try('by_item', lambda: str(memory.get(memory.get({SW.NOTE_ID!r})))[:80])\n"), sink),
        librarian=[search, result, Always(lambda c: SW.librarian_answer([SW.fit(a)]))]))
    assert NOTE in later["first"], "the later librarian was not given the note an earlier one kept"
    text = later["search"]
    assert re.search(r"ranked|not ranked|unranked", text, re.I), text[:600]
    items = sink["out"]["items"]
    assert items and all(i in text for i in items), (items, text[:2000])
    for i in items:
        line = next(ln for ln in text.splitlines() if i in ln)
        assert "/memory/" in line, line
    for k in ("by_slug", "by_path", "by_item"):
        assert not str(sink["out"][k]).startswith("RAISED"), (k, sink["out"][k])
