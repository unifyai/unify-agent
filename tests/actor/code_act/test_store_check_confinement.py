"""The store check never executes a model-written ``def`` in the harness's process.

Storing a function used to load it the way a search does: the ``def`` was
executed in a scratch namespace, which runs its default values and its
decorators. With Python in the sandboxed worker, ``functions.add`` from a cell
is served by the harness, so that ``def`` ran outside the sandbox. These tests
watch the two places a stored ``def`` is executed in this process
(``_create_in_process_callable`` and ``_inject_dependencies``) while functions
are stored.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import core_surface
from unify.settings import SETTINGS

DOUBLE = (
    "def double(x: int) -> int:\n"
    '    """Return twice its argument."""\n'
    "    return 2 * x\n"
)


@pytest.fixture
def executed(monkeypatch):
    """The names of the functions this process executed while storing."""
    from unify.function_manager.function_manager import FunctionManager

    seen: list[str] = []
    create = FunctionManager._create_in_process_callable
    inject = FunctionManager._inject_dependencies

    def watched_create(self, func_data, *args, **kwargs):
        seen.append(func_data.get("name"))
        return create(self, func_data, *args, **kwargs)

    def watched_inject(self, func_data, *args, **kwargs):
        seen.append(f"callees of {func_data.get('name')}")
        return inject(self, func_data, *args, **kwargs)

    monkeypatch.setattr(FunctionManager, "_create_in_process_callable", watched_create)
    monkeypatch.setattr(FunctionManager, "_inject_dependencies", watched_inject)
    return seen


class _Cells:
    """Cells of one core session, run through the actor's execute_code tool."""

    def __init__(self, actor):
        from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX

        self.actor = actor
        self.tools = actor.get_tools("act")
        self.sandbox = PythonExecutionSession(environments={})
        objects = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
        self.sandbox.global_state.update(objects)
        self.sandbox.core_globals = objects
        self._token = _CURRENT_SANDBOX.set(self.sandbox)

    async def __call__(self, code: str, **kwargs: Any):
        return await self.tools["execute_code"].fn(
            thought="A step.",
            code=code,
            **kwargs,
        )

    async def close(self) -> None:
        from unify.actor.execution import _CURRENT_SANDBOX

        _CURRENT_SANDBOX.reset(self._token)
        await self.sandbox.close()
        await self.actor.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_functions_add_from_a_cell_executes_nothing_in_the_harness(
    core_world,
    executed,
):
    cells = _Cells(new_actor())
    try:
        out = await cells(f"await functions.add({DOUBLE!r})")
        assert out.error is None, out.error
        assert executed == [], f"the harness executed {executed} while storing"
        out = await cells("await functions.run('double', x=21)")
        assert out.error is None and out.result == 42, out.error
    finally:
        await cells.close()


def test_the_store_check_executes_nothing_with_python_in_the_worker(
    monkeypatch,
    executed,
):
    from unify.function_manager.function_manager import FunctionManager

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DOUBLE])
    assert executed == []
    assert [f["name"] for f in fm.filter_functions()] == ["double"]


def test_with_python_in_process_the_check_still_loads_the_function(
    monkeypatch,
    executed,
):
    """Where cells run in this process anyway, the def is loaded and a broken one refused."""
    from unify.function_manager.function_manager import FunctionManager

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DOUBLE])
    assert "double" in executed
    with pytest.raises(ValueError, match="does not load the way a search loads it"):
        fm.add_functions(
            implementations=[
                "def broken(x: int = int('not a number')) -> int:\n"
                '    """Return its argument."""\n'
                "    return x\n",
            ],
        )
