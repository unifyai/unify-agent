"""A change to a stored function never replays its recorded cases in the harness's process.

``UNIFY_FUNCTION_CASES`` records each call of a stored function and, when
``functions.add(..., overwrite=True)`` or a patch would store a different
source, replays the calls that returned against the new source. A replay loads
the new ``def`` and calls it in the process serving the change: with Python in
the sandboxed worker that is the harness, so a cell could run code of its own
there, outside the sandbox and beside the credentials, and read the result
back in the refusal ("now returns ..."). With the worker on nothing is
replayed: a case that returned refuses the change, and one that raised does
not block. These tests watch the two places a stored ``def`` is executed in
this process (``_create_in_process_callable`` and ``_inject_dependencies``).
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_store_check_confinement import (  # noqa: F401
    _Cells,
    executed,
)
from tests.helpers import _handle_project
from unify.settings import SETTINGS

DOUBLE = (
    "def double(x: int) -> int:\n"
    '    """Return twice its argument."""\n'
    "    return 2 * x\n"
)
TRIPLE = DOUBLE.replace("2 * x", "3 * x")
DIVIDE = (
    "def divide(a: int, b: int) -> float:\n"
    '    """Divide a by b."""\n'
    "    return a / b\n"
)
SAFE_DIVIDE = DIVIDE.replace(
    "    return a / b\n",
    "    return a / b if b else 0.0\n",
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_an_overwrite_from_a_cell_executes_nothing_in_the_harness(
    core_world,
    executed,
):
    cells = _Cells(new_actor())
    try:
        out = await cells(f"await functions.add({DOUBLE!r})")
        assert out.error is None, out.error
        out = await cells("await functions.run('double', x=21)")
        assert out.error is None and out.result == 42, out.error
        out = await cells(f"await functions.add({TRIPLE!r}, overwrite=True)")
        assert executed == [], f"the harness executed {executed} while replaying"
        assert "could not be checked against the new source" in str(out.error)
        assert "now returns" not in str(out.error)
        out = await cells("await functions.run('double', x=21)")
        assert out.error is None and out.result == 42, out.error
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_a_function_whose_case_raised_is_still_fixed_from_a_cell(
    core_world,
    executed,
):
    cells = _Cells(new_actor())
    try:
        out = await cells(f"await functions.add({DIVIDE!r})")
        assert out.error is None, out.error
        out = await cells("await functions.run('divide', a=1, b=0)")
        assert "ZeroDivisionError" in str(out.error)
        out = await cells(f"await functions.add({SAFE_DIVIDE!r}, overwrite=True)")
        assert out.error is None, out.error
        assert executed == [], f"the harness executed {executed} while replaying"
        out = await cells("await functions.run('divide', a=1, b=0)")
        assert out.error is None and out.result == 0.0, out.error
    finally:
        await cells.close()


@_handle_project
def test_with_python_in_the_worker_a_change_is_not_replayed(monkeypatch, executed):
    from unify.function_manager.function_manager import FunctionManager

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[DOUBLE])
    namespace: dict = {}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    assert namespace["double"](21) == 42  # one recorded case
    executed.clear()
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    out = fm.add_functions(
        implementations=[TRIPLE],
        overwrite=True,
        raise_on_error=False,
    )
    assert "could not be checked against the new source" in out["double"]
    patched = fm.patch_function(name="double", old="2 * x", new="3 * x", why="probe")
    assert "could not be checked against the new source" in patched["error"]
    assert executed == [], f"the harness executed {executed} while replaying"
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    out = fm.add_functions(
        implementations=[TRIPLE],
        overwrite=True,
        raise_on_error=False,
    )
    assert "now returns 63" in out["double"] and "double" in executed
