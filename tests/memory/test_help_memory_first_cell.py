"""``help(memory)`` as the first cell of a memory agent's box shows the library package, never its lazy stand-in.

``memory`` is bound in every cell before any runs and imported by the first cell that uses it
(``worker_child._LazyModule``): a cell that names it imports it before its own code runs (``Worker.load_named``), so
even its first line sees the module. Here the writer's and a helper's boxes, in a real pass over the fixture library;
the actor's, a teammate's and the librarian's are in ``test_memory_paths_one_way``.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory import test_prompt_runtime_contract_from_spec as C
from tests.memory.real_request_box import HELP_CELL, assert_package_help
from tests.memory.test_tool_channel_box import needs_box
from unify.settings import SETTINGS


@needs_box
@needs_bwrap
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("role", ["writer", "helper"])
def test_help_memory_as_the_first_cell_shows_the_package(role, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    seen: dict = {}

    def plan(steps, n):
        if n == 1:
            return HELP_CELL
        seen.setdefault("help", steps.outputs.get("c1", ""))
        return None

    steps = C.Steps(plan, C.MEMORY_END[role])
    if role == "helper":
        C._helper_waits_answer_at_once(monkeypatch)
        C._memory_pass(tmp_path, "writer", C._delegating_writer(), helper=steps)
    else:
        C._memory_pass(tmp_path, role, steps)
    assert_package_help(seen.get("help", ""))
