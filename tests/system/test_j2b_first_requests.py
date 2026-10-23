"""J2 for the dynamic texts: each role's first request in a real run (the system message, the first user message and
the tool list), with the scripted model and the toy worlds, per memory mode, equals its design-true golden
(``goldens/first_requests-<mode>.json``, normalised: ids, names, clocks and places are stand-ins).

Record (on a keyless tree, then review and apply the corrections): ``SYSTEM_GOLDENS_RECORD=<dir> pytest
tests/system/test_j2b_first_requests.py`` writes ``<dir>/first_requests-<mode>.recorded.json`` (the tree is read-only
to tests); copy it beside the golden and run ``python -m tests.system.goldens_apply <mode>`` for the design-true golden.
"""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.scripted_model import Always, ScriptedModel, cell, reply
from tests.system import sysworld as SW
from tests.system.drive import ALL_MODES
from tests.system.goldens import FIRST_REQUESTS, normalised

HERE = Path(__file__).parent
TASK = "Look at the words and say how many there are."
WAIT = ("got = ''\nfor _ in range(8):\n    got += str(tools.wait(timeout=120))\n"
        "    if '(done)' in got or '(reply)' in got:\n        break\nprint(got[-800:])\n")


def _shown(messages: list, tools: list) -> str:
    """A request as the role is shown it: its system message, its first (and, when there are more, last) user message
    and its tools' names."""
    system = next((SW.flat(m.get("content")) for m in messages if m.get("role") == "system"), "")
    users = [SW.flat(m.get("content")) for m in messages if m.get("role") == "user"]
    user = users[0] if users else ""
    last = f"### last user message\n{users[-1]}\n" if len(users) > 1 else ""  # e.g. the fork's instruction
    names = sorted((t.get("function") or {}).get("name") or t.get("name") or "" for t in tools or [])
    return normalised(f"### system\n{system}\n### first user message\n{user}\n{last}### tools\n{', '.join(names)}\n")


def _call(call) -> str:
    return _shown(call.messages, call.request.get("tools"))


def _body(body: dict) -> str:
    return _shown(body.get("messages") or [], body.get("tools"))


async def _request_with_teammate(s, *, item: str | None = None):
    """A request whose actor delegates to a teammate (and, with memory, whose librarian answers)."""
    kinds = {
        "actor": [reply(calls=[cell(f"print(tools.delegate({TASK!r}))\n")]), reply(calls=[cell(WAIT)]), reply("Three.")],
        "helper": [reply(calls=[cell("tools.answer('three')\n")]), Always(reply("done"))],
    }
    if item:
        kinds["librarian"] = Always(lambda c: SW.librarian_answer([SW.fit(item)]))
    model = ScriptedModel(**kinds)
    await s.request(SW.words_request(["a", "b", "c"]), model)
    return model


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ALL_MODES)
async def test_each_roles_first_request_reads_as_its_golden(system, mode):
    s = system
    got: dict[str, str] = {}
    item = None
    if mode != "off":
        s.writer.plan = "keep"
        s.writer.kin = "same_world"
        s.writer.probe = [SW.DELEGATE, SW.WAIT_HELPER]
        s.helper = SW.HelperLane("")
        item = "capture:" + s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
        await s.accepted_tally()  # the judge is asked about an unrelated capture
        await s.accepted_words(["glass"], "upper")  # a group, its pass, and the helper the writer starts
        forks = [c for c in s.lanes.of("fork") if not SW.is_segment(c.body)]
        got[f"fork/{mode}/first_request"] = _body(forks[0].body)
        if s.writer.judge_bodies:
            got[f"judge/{mode}/first_request"] = _body(s.writer.judge_bodies[0])
        got[f"writer/{mode}/first_request"] = _body(s.writer.pass_bodies[0])
        if s.helper.bodies:
            got[f"helper/{mode}/first_request"] = _body(s.helper.bodies[0])
    model = await _request_with_teammate(s, item=item)
    got[f"actor/{mode}/first_request"] = _call(model.of("actor")[0])
    helpers = [c for c in model.of("helper")]
    if helpers:
        got[f"teammate/{mode}/first_request"] = _call(helpers[0])
    if mode != "off":
        got[f"librarian/{mode}/first_request"] = _call(model.of("librarian")[0])
    path = HERE / FIRST_REQUESTS.format(mode=mode)
    record = os.environ.get("SYSTEM_GOLDENS_RECORD")
    if record:
        out = Path(record) / path.with_suffix(".recorded.json").name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(got, indent=1, sort_keys=True, ensure_ascii=False))
        pytest.skip(f"recorded to {out}")
    golden = json.loads(path.read_text())
    assert sorted(got) == sorted(golden), (sorted(got), sorted(golden))
    diffs = []
    for k in sorted(golden):
        if got[k] != golden[k]:
            d = difflib.unified_diff(golden[k].splitlines(), got[k].splitlines(), "golden", "run", n=0, lineterm="")
            diffs.append(f"== {k}\n" + "\n".join(list(d)[2:14]))
    assert not diffs, "\n".join(diffs)
