"""S6: with memory off the actor and its teammates see no memory text, no memory agent runs, and their tools are the
memory-on tools less the memory ones only.

Design: R10 (the memory-off run names no writer, fork or librarian; tool texts match each role's real roots), change d
(``tools.request()`` in both arms), H1 (one knob); audit A16 (the off refusal names ``/inputs/attention/index.json``)
and guard J10 (memory-off text guard).
"""

from __future__ import annotations

import json
import re

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW

MEMORY_WORDS = re.compile(
    r"\b(memory|memories|librarian|library|writer|fork|forks|capture|captures|attention|recorded procedure)\b"
    r"|/inputs|/memory|@writer|@librarian|@fork|memory\.ask|ask_library",
    re.I,
)
PROBE = SW.probe_json(
    "_try('list', lambda: str(tools.list()))\n"
    "_names = [l.split('(')[0].strip() for l in str(out.get('list', '')).splitlines() if '(' in l]\n"
    "for _n in _names:\n"
    "    _try('help:' + _n, lambda _n=_n: str(tools.help(_n)))\n"
    "_try('request', lambda: str(tools.request()))\n"
    "_try('who', lambda: str(tools.who()))\n"
    "_try('read_missing', lambda: str(tools.read('/workspace/none.txt')))\n"
    "_try('read_outside', lambda: str(tools.read('/srv/elsewhere/x.md')))\n"
    "_try('grep_outside', lambda: str(tools.grep('x', '/srv/elsewhere')))\n"
    "_try('memory_name', lambda: str(memory))\n"
)
TASK = "Look at the words and say how many there are."


def _tools(out: dict) -> set[str]:
    return {k.split(":", 1)[1] for k in out if k.startswith("help:")}


async def _run(s, mode_texts: dict) -> None:
    """One request: the actor probes its own cell, then delegates the same probe to a teammate."""
    actor: dict = {}
    helper: dict = {}

    def h_first(call):
        helper["first"] = call
        return reply(calls=[cell(PROBE)])

    def h_second(call):
        helper["out"] = SW.marked(SW.last_tools(call))
        return reply(calls=[cell("tools.answer('counted')\n")])

    wait = ("got = ''\nfor _ in range(8):\n    got += str(tools.wait(timeout=120))\n"
            "    if '(done)' in got or '(reply)' in got:\n        break\nprint(got[-800:])\n")
    turns = SW.probe_turns(PROBE, actor, answer="")
    model = ScriptedModel(
        actor=[turns[0], lambda call: (actor.__setitem__("out", SW.marked(SW.last_tools(call))),
                                       reply(calls=[cell(f"print(tools.delegate({TASK!r}))\n")]))[1],
               reply(calls=[cell(wait)]), reply("Three.")],
        helper=[h_first, h_second, Always(reply("done"))],  # memory off: no librarian session exists
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[])),
    )
    sess = await s.request(SW.words_request(["a", "b", "c"]), model)
    mode_texts.update(actor=actor, helper=helper, model=model, session=sess)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["off"])
async def test_memory_off_has_no_memory_text_no_memory_agents_and_the_same_tools_but_memory(system, mode):
    s = system
    off: dict = {}
    await _run(s, off)
    # no memory agent ran, nothing of memory was written
    assert off["session"].episode_id is None
    assert not s.lanes.calls, [c.lane for c in s.lanes.calls]
    assert "librarian" not in off["model"].kinds()
    written = sorted(str(p.relative_to(s.home)) for top in ("memory-state", "memory", "memory-checkout")
                     for p in (s.home / top).rglob("*") if p.is_file()
                     and "memory-state/agents/" not in str(p))  # the team's run record (agents), not memory
    assert not written, written[:40]
    # no memory text in anything the actor or the teammate was given, or got back from a tool
    for role in ("actor", "helper"):
        texts = {"system": SW.system_text(off[role]["first"]), "first": SW.first_user(off[role]["first"]),
                 "tools": json.dumps(off[role]["first"].request.get("tools") or []),
                 **{k: str(v) for k, v in off[role]["out"].items() if k != "memory_name"}}
        hits = {k: sorted({m.group(0) for m in MEMORY_WORDS.finditer(t)}) for k, t in texts.items()
                if MEMORY_WORDS.search(t)}
        assert not hits, (role, hits)
        assert str(off[role]["out"]["memory_name"]).startswith("RAISED NameError"), off[role]["out"]["memory_name"]
    # the request reader is a harness tool in both arms
    assert "WORDS" in off["actor"]["out"]["request"]

    # ── the same request with memory on: the same tools, plus memory's only ──
    H.apply_settings(s.monkeypatch, {"UNIFY_MEMORY": "on"})
    on: dict = {}
    await _run(s, on)
    for role in ("actor", "helper"):
        t_off, t_on = _tools(off[role]["out"]), _tools(on[role]["out"])
        assert t_off and t_off <= t_on, (role, sorted(t_off - t_on))
        extra = {n: on[role]["out"][f"help:{n}"] for n in t_on - t_off}
        assert all(re.search(r"librar|memory", h, re.I) for h in extra.values()), (role, extra)
        same = {n for n in t_off if off[role]["out"][f"help:{n}"] != on[role]["out"][f"help:{n}"]}
        assert not [n for n in same if not re.search(r"librar|memory", on[role]["out"][f"help:{n}"], re.I)], \
            (role, sorted(same))
    # the actor (memory on) is never offered @librarian or @fork as an address (R8.2)
    text = SW.system_text(on["actor"]["first"])
    addresses = [ln for ln in re.split(r"(?<=[.;])\s", text) if "Address an agent" in ln or "a role:" in ln]
    offered = " ".join(addresses) + on["actor"]["out"]["who"] + on["actor"]["out"].get("help:send", "")
    assert "@librarian" not in offered and "@fork" not in offered, (addresses, on["actor"]["out"]["who"][:600])
