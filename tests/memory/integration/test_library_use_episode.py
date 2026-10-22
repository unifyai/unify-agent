"""Symbolic: one scripted library-mode request uses the library exactly as the actor's guide says, with no import
line (``memory.index(...)``, then a stored function imported by its id), and its recorded ``memory_use.json`` shows
it: the index helper's calls ok, the item's runtime calls ok, and ``memory_bound``.

The request runs end to end in process: the memory run begins (the read-only copy of a seeded library and its
index), the actor's cells run in the real sandboxed worker with tools in code and a persistent interpreter, and the
run finishes, recording the episode with no pass. The model is the scripted transport; the tracks the request
drives besides the actor are the recording fakes (``fake_tracks``). Keyless.
"""

from __future__ import annotations

import asyncio
import contextvars
import json

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.integration import fake_tracks
from tests.memory.integration.test_actor_export_library_mode import _library_repo
from tests.scripted_model import ScriptedModel, reply, scripted
from unify import sandbox
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.settings import SETTINGS

PD = "memory.text.dates:parse_date"
REQUEST = "Report the date 2026-10-09 in the library's date form."

#: The guide's own steps, with no import line for `memory`: look in the index, then use an item by its id.
LOOK = 'print(memory.index("text"))'
USE = "from memory.text.dates import parse_date\nprint('DATE', parse_date('2026-10-09'))"


@needs_bwrap
@pytest.mark.timeout(600)
def test_a_request_that_uses_the_library_as_the_guide_says_records_it(core_world, monkeypatch):  # noqa: F811
    from unify.actor.code_act_actor import CodeActActor
    from unify.memory.episodes import episode_dir
    from unify.memory.gitio import Repo

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    paths = Paths.under(core_world["state"])
    _library_repo(paths.memory)
    fake_tracks.install(monkeypatch)
    model = ScriptedModel(
        actor=[
            reply(calls=[("execute_code", {"code": LOOK})]),
            reply(calls=[("execute_code", {"code": USE})]),
            reply("2026-10-09"),
        ],
    )
    progress: list[str] = []

    async def episode():
        run = request_mod.RequestRun.begin(REQUEST)
        try:
            # rev 2: the actor owns nothing past a request; its in-code tools and slot close with the result
            actor = CodeActActor()
            with scripted(model):
                handle = await actor.act(REQUEST, persist=False)
                await asyncio.wait_for(handle.result(), 300)
        except BaseException:
            run.abort()
            raise
        await run.finish(None, progress=progress.append, consolidate=False)
        return run

    with asyncio.Runner() as runner:
        run = runner.run(episode(), context=contextvars.copy_context())
    outputs = [
        str(m.get("content"))
        for call in model.calls
        for m in call.messages
        if m.get("role") == "tool"
    ]
    assert any("DATE 2026-10-09" in o for o in outputs), outputs
    assert not any("NameError" in o or "ImportError" in o for o in outputs), outputs
    raw = Repo(paths.episodes).show("main", f"{episode_dir(run)}/memory_use.json")
    rec = json.loads(raw)
    assert rec["memory_bound"] is True, rec
    assert rec["library_helper_calls"]["index"]["ok"] >= 1, rec["library_helper_calls"]
    assert rec["items"][PD]["runtime"]["ok"] >= 1, rec["items"].get(PD)
    assert rec["library_calls_failed"] == 0 and rec["library_name_errors"] == 0, rec
