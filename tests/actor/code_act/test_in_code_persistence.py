"""Symbolic: the actor's sandbox keeps its state between cells, and each
restart is told: a cell past its limit, a thread a cell left running and an idle interpreter each say so in the
next output, after which the cell's variables are gone.

The model is the scripted transport (no key, no network); cells run in the real sandboxed worker.
"""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Optional

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_in_code_tools_wiring import _Collect, _act, _tool_results, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify.actor import in_code_tools
from unify.actor.execution.types import parts_to_text


def _text(out) -> str:
    return parts_to_text(out.stdout) + parts_to_text(out.stderr) + (out.error or "")


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_variable_from_one_cell_is_there_in_the_next(core_world, monkeypatch):  # noqa: F811
    monkeypatch.setattr(in_code_tools, "SESSIONS", _Collect())
    first, second = ("x = 41\nprint('SET')", "print('GOT', x + 1)")
    model = ScriptedModel(
        actor=[reply(calls=[code(first)]), reply(calls=[code(second)]), reply("done")],
    )
    assert await _act(model, "Twice.") == "done"
    assert "GOT 42" in _tool_results(model.calls[2]), _why(model)


def _quick_idle_profile(seconds: float):
    from unify.common.box_interpreter import ActorProfile

    @dataclasses.dataclass
    class QuickIdle(ActorProfile):
        idle_restart_s: Optional[float] = seconds

    return QuickIdle


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_each_restart_is_told_and_the_variables_are_gone(core_world, monkeypatch, tmp_path):  # noqa: F811
    from unify.common import box_interpreter

    monkeypatch.setattr(box_interpreter, "ActorProfile", _quick_idle_profile(2.0))
    session = in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=3.0,
        events_folder=tmp_path / "events",
    )
    lost = "variables are gone, files remain"
    try:
        assert "41" in _text(await session.run_cell("x = 41\nprint(x)"))
        # a cell past its limit: the box is killed, and the output says so
        out = _text(await session.run_cell("import time\ntime.sleep(30)"))
        assert "timed out after 3 s" in out and lost in out, out
        assert "NameError" in _text(await session.run_cell("print(x)"))
        # a thread left running after the cell ended
        await session.run_cell("x = 41")
        out = _text(
            await session.run_cell(
                "import threading, time\nthreading.Thread(target=time.sleep, args=(30,), daemon=True).start()",
            ),
        )
        assert "a thread this cell started was still running after it ended" in out and lost in out, out
        assert "NameError" in _text(await session.run_cell("print(x)"))
        # an interpreter left idle past its limit is stopped; the next cell is told
        await session.run_cell("x = 41")
        await asyncio.sleep(4.0)
        out = _text(await session.run_cell("print(x)"))
        assert "was idle for more than 2 s" in out and "NameError" in out, out
    finally:
        await session.close()
