"""The actor's and a teammate's sandboxes with tools called from code: each finds ``tools``, ``help``, ``display``
and the model helpers bound in its first cell with no import, ``memory`` too in library mode (even for an empty
library), and never a bare name that ``tools`` replaced. Cells keep their variables by default; under
``UNIFY_CELL_STATE=fresh`` they do not, and the prompt says so. Each restart that loses variables is told in the
cell's output. Each agent's first request opens with the shared opening, which says once that the sandbox is
persistent and stateful and which names are already available as variables, with no import needed.

The model is scripted (tests/scripted_model.py): no key, no network. Cells run in the real sandboxed worker.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import (  # noqa: F401 (fixture)
    _act,
    _assert_ended,
    _system_texts,
    _tool_results,
    _why,
    code,
    in_code,
)
from tests.scripted_model import ScriptedModel, reply
from unify.actor import in_code_tools
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.library_export import generated_library_mode
from unify.settings import SETTINGS

# ==== names and sentences this file assumes (the spec gives them; adapt them here, in one place) ====================
#: design D.1, persistent form, its first sentence
PERSISTENT_SENTENCE = (
    "Your Python sandbox is persistent and stateful: variables, imports and functions you define in one cell are "
    "still there in the next, so build on what you already have instead of defining or loading it again."
)
#: design D.1, fresh form
FRESH_SENTENCE = (
    "Each cell runs in a fresh Python sandbox: files you write stay, but variables, imports and functions do not "
    "carry over to the next cell."
)
#: design D.1, the names sentence
NAMES_SENTENCE = re.compile(
    r"((?:`[A-Za-z_][A-Za-z0-9_]*`(?:, | and |, and )?)+) (?:is|are) already available as (?:a )?variables? in every "
    r"cell; no import needed\.",
)
#: design B.2: the actor's and teammate's pre-bound names (without a library), and what they must not have
PREBOUND = ("tools", "help", "display", "query_llm", "list_llms", "unillm", "run_coro_sync", "ModelRequestError")
MUST_NOT = ("install", "read_file", "grep", "record", "agents", "functions", "guidance")
#: the restart notes (box_interpreter._LOST and its callers; design C.3)
LOST = "the interpreter was restarted: variables are gone, files remain"
# =====================================================================================================================

CHECK = (*PREBOUND, "memory", *MUST_NOT)


def probe(names) -> str:
    """A first cell, before any import line: ``x = 41``, then which of *names* are bound (a bare name read in a
    ``try``, so it works with any builtins the box keeps; a lazy ``memory`` is not imported by it)."""
    lines = ["x = 41", "_bound = []"]
    for n in names:
        lines += ["try:", f"    {n}", f"    _bound.append({n!r})", "except NameError:", "    pass"]
    lines.append("print('BOUND', sorted(_bound))")
    return "\n".join(lines) + "\n"

PROBE = probe(CHECK)
NEXT = "print('NEXT', x + 1)\nhelp(len)\nprint('HELPED')\n"


def flat(text: str) -> str:
    return " ".join(str(text).split())


def _names(text: str) -> set[str]:
    found = NAMES_SENTENCE.findall(flat(text))
    assert len(found) == 1, f"expected one names sentence, found {found}"
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", found[0]))


def _bound(output: str) -> set[str]:
    line = next(ln for ln in output.splitlines() if "BOUND [" in ln)
    return set(json.loads(line.split("BOUND ", 1)[1].replace("'", '"')))


def _assert_first_request(call, agent: str, *, persistent: bool = True, library: bool = False) -> None:
    system = flat(_system_texts(call)[0])
    assert system.startswith("## How you work here"), (agent, system[:300])
    if persistent:
        assert system.count(PERSISTENT_SENTENCE) == 1, (agent, system[:2500])
        assert FRESH_SENTENCE not in system, agent
    else:
        assert system.count(FRESH_SENTENCE) == 1, (agent, system[:2500])
        assert "persistent and stateful" not in system, agent
    # the actor's own rules leave persistence to the opening: said once, never twice
    assert "cells share one persistent sandbox" not in system, agent
    names = _names(system)
    assert "tools" in names, (agent, names)
    assert ("memory" in names) == library, (agent, names)
    assert not names & set(MUST_NOT), (agent, names)
    assert not re.search(r"\bimport memory\b", system), agent
    assert "importable in your cells" not in system, agent


def _assert_probe(output: str, agent: str) -> None:
    bound = _bound(output)
    assert set(PREBOUND) <= bound, (agent, sorted(set(PREBOUND) - bound), output)
    assert not bound & set(MUST_NOT), (agent, sorted(bound & set(MUST_NOT)))


def _assert_next(output: str, agent: str) -> None:
    assert "NEXT 42" in output and "HELPED" in output, (agent, output)
    assert "NameError" not in output, (agent, output)


TEAMMATE_START = "print(tools.delegate('Run two cells in your box and tell me what they print, please.'))\n"
ACTOR_WAIT = (
    "got = tools.wait(timeout=200)\n"
    "print('GOT', got)\n"
    "tools.finish('teammate done')\n"
)


# --- the actor and a teammate, end to end ---------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_the_actor_has_its_names_bound_and_keeps_variables(in_code):
    model = ScriptedModel(
        actor=[reply(calls=[code(PROBE)]), reply(calls=[code(NEXT)]), reply("done")],
    )
    assert await _act(model, "Probe the sandbox.") == "done", _why(model)
    _assert_next(_tool_results(model.calls[2]), "actor")
    _assert_probe(_tool_results(model.calls[1]), "actor")
    _assert_first_request(model.calls[0], "actor")
    _assert_ended(in_code["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_a_teammate_has_its_names_bound_and_keeps_variables(in_code):
    model = ScriptedModel(
        actor=[reply(calls=[code(TEAMMATE_START)]), reply(calls=[code(ACTOR_WAIT)]), reply("not finished")],
        helper=[
            reply(calls=[code(PROBE)]),
            reply(calls=[code(NEXT)]),
            reply(calls=[code("tools.finish('probe finished')")]),
            reply("not finished"),
        ],
    )
    assert await _act(model, "Ask a teammate to probe its sandbox.") == "teammate done", _why(model)
    mates = model.of("helper")
    assert len(mates) >= 3, len(mates)
    _assert_next(_tool_results(mates[2]), "teammate")
    _assert_probe(_tool_results(mates[1]), "teammate")
    _assert_first_request(mates[0], "teammate")
    _assert_ended(in_code["opened"])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_under_fresh_cell_state_the_actor_and_a_teammate_lose_variables_and_are_told(in_code, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    model = ScriptedModel(
        actor=[
            reply(calls=[code("x = 41\nprint('SET')")]),
            reply(calls=[code("print('NEXT', x + 1)")]),
            reply(calls=[code(TEAMMATE_START)]),
            reply(calls=[code(ACTOR_WAIT)]),
            reply("not finished"),
        ],
        helper=[
            reply(calls=[code("x = 41\nprint('SET')")]),
            reply(calls=[code("print('NEXT', x + 1)")]),
            reply(calls=[code("tools.finish('probe finished')")]),
            reply("not finished"),
        ],
    )
    assert await _act(model, "Twice, then a teammate twice.") == "teammate done", _why(model)
    actor, mates = model.of("actor"), model.of("helper")
    for who, call in (("actor", actor[2]), ("teammate", mates[2])):
        shown = _tool_results(call)
        assert "NameError" in shown and "NEXT 42" not in shown, (who, shown)
    _assert_first_request(actor[0], "actor", persistent=False)
    _assert_first_request(mates[0], "teammate", persistent=False)
    _assert_ended(in_code["opened"])


# --- library mode: `memory` bound with no import, even for an empty library -----------------------------------


def _library_run(monkeypatch, home: Path) -> Paths:
    """A library-mode request whose read-only copy is the generated package of an empty library."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    paths = Paths.under(home)
    empty = home / "empty-library"
    empty.mkdir(parents=True)
    for rel, data in generated_library_mode(empty).items():
        (paths.checkout / rel).parent.mkdir(parents=True, exist_ok=True)
        (paths.checkout / rel).write_bytes(data)
    run = request_mod.RequestRun("r", paths)
    run.library_mode = True
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return paths


def _session(tmp_path: Path, agent: str, *, limit=None, start=None):
    work = tmp_path / f"{agent}-workspace"
    work.mkdir(exist_ok=True)
    return in_code_tools.InCodeTools(
        agent=agent,
        name="root" if agent == "actor" else "mate",
        record=SimpleNamespace(),
        workspace_root=str(work),
        shadow={},
        cell_limit_s=limit,
        events_folder=tmp_path / f"{agent}-events",
        start=start,
    )


def _out(res) -> str:
    from unify.actor.execution.types import parts_to_text

    return parts_to_text(res.stdout) + parts_to_text(res.stderr) + (res.error or "")


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.parametrize("agent", ["actor", "teammate"])
async def test_in_library_mode_memory_is_bound_with_no_import_even_when_empty(world, monkeypatch, agent):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    paths = _library_run(monkeypatch, world["state"])
    assert (paths.checkout / "memory" / "__init__.py").is_file()
    session = _session(world["state"], agent)
    try:
        first = _out(
            await session.run_cell(
                "y = 41\n"
                "print('LAZY', 'imported on first use' in repr(memory))\n"
                "print('EMPTY', 'The library is empty.' in memory.index())\n",
            ),
        )
        assert "LAZY True" in first and "EMPTY True" in first, first
        second = _out(await session.run_cell("print('NEXT', y + 1, type(memory).__name__)"))
        assert "NEXT 42 module" in second, second
        names = _names(session.opening())
        assert {"tools", "memory"} <= names, names
    finally:
        await session.close(completed=True)
    assert not session.box_alive()


# --- the restarts the actor is told about ------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_actor_is_told_each_restart_and_then_has_lost_its_variables(world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    session = _session(world["state"], "actor", limit=3.0)
    try:
        await session.run_cell("x = 41")
        timed = _out(await session.run_cell("import time\ntime.sleep(60)"))
        assert f"(the cell timed out after 3 s; {LOST})" in timed, timed
        lost = _out(await session.run_cell("print('AFTER TIMEOUT', 'x' in dir())\nx = 41"))
        assert "AFTER TIMEOUT False" in lost, lost
        left = _out(
            await session.run_cell(
                "import threading, time\nthreading.Thread(target=time.sleep, args=(60,), daemon=True).start()",
            ),
        )
        assert f"(a thread this cell started was still running after it ended; {LOST})" in left, left
        lost = _out(await session.run_cell("print('AFTER THREAD', 'x' in dir())"))
        assert "AFTER THREAD False" in lost, lost
        # the opening named these causes before they happened
        opening = flat(session.opening())
        for cause in ("ran out of time", "left something running", "sat idle too long"):
            assert cause in opening, cause
    finally:
        await session.close(completed=True)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_an_idle_actor_interpreter_is_restarted_and_says_so(world, monkeypatch):  # noqa: F811
    import asyncio

    from unify.common.box_interpreter import BoxInterpreter

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")

    async def start(**kw):
        kw["profile"] = dataclasses.replace(kw["profile"], idle_restart_s=0.5)
        return await BoxInterpreter.start(**kw)

    session = _session(world["state"], "actor", start=start)
    try:
        await session.run_cell("x = 41")
        await asyncio.sleep(3.0)
        out = _out(await session.run_cell("print('AFTER IDLE', 'x' in dir())"))
        assert "(the interpreter was idle for more than 0.5 s and was restarted: " in out, out
        assert "AFTER IDLE False" in out, out
    finally:
        await session.close(completed=True)

