"""A read that binds stored functions never executes them in the harness.

``functions.get`` / ``search`` / ``filter`` / ``list`` bind the functions they
return in the session, so the next cell can call them by name. That used to
load each one the in-process way (``_inject_callables_for_functions`` ->
``_inject_dependencies`` / ``_create_in_process_callable``): its ``def`` and
its stored callees were executed in the process serving the read, which with
Python in the sandboxed worker is the harness, beside the provider
credentials. The read also put the workspace environment's packages on the
harness's own ``sys.path`` (``environment.activate``), so whatever an install
put there could be imported by the harness.

With the worker on, a read now binds each function from its stored source,
which the worker defines and runs; the harness keeps nothing it could call.
The two places a stored ``def`` is executed here, and the stored-function
runner, refuse outright; the environment is read from its own path
(test_workspace_environment.py).
"""

from __future__ import annotations

import sys
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

HALVE = (
    "def halve(x: int) -> int:\n"
    '    """Return half its argument, rounded down."""\n'
    "    return x // 2\n"
)
DOUBLE = (
    "def double(x: int) -> int:\n"
    '    """Return twice its argument, by way of halve."""\n'
    "    return 4 * x - 2 * x + halve(0)\n"
)
SPARE = (
    "def spare(x: int) -> int:\n" '    """Return its argument."""\n' "    return x\n"
)


@pytest.fixture
def harness(monkeypatch):
    """What this process executed or activated while the cells ran."""
    from unify import environment
    from unify.function_manager.function_manager import FunctionManager

    seen: list[str] = []
    create = FunctionManager._create_in_process_callable
    inject = FunctionManager._inject_dependencies
    run = FunctionManager._execute_python_function
    activate = environment.activate

    def watched_create(self, func_data, *args, **kwargs):
        seen.append(func_data.get("name"))
        return create(self, func_data, *args, **kwargs)

    def watched_inject(self, func_data, *args, **kwargs):
        seen.append(f"callees of {func_data.get('name')}")
        return inject(self, func_data, *args, **kwargs)

    async def watched_run(self, *args, **kwargs):
        seen.append("execute_python_function")
        return await run(self, *args, **kwargs)

    def watched_activate(*args, **kwargs):
        seen.append("activate")
        return activate(*args, **kwargs)

    monkeypatch.setattr(FunctionManager, "_create_in_process_callable", watched_create)
    monkeypatch.setattr(FunctionManager, "_inject_dependencies", watched_inject)
    monkeypatch.setattr(FunctionManager, "_execute_python_function", watched_run)
    monkeypatch.setattr(environment, "activate", watched_activate)
    return seen


@pytest.fixture
def ranked(monkeypatch):
    """A semantic search without embeddings: every stored function, by name."""
    from unify.function_manager import function_manager as fm_module

    def by_name(rows, references, *, limit, id_field):
        ordered = sorted(rows, key=lambda row: str(row.get("name")))
        return [dict(row, _similarity=1.0) for row in ordered][:limit]

    monkeypatch.setattr(fm_module, "rank_by_similarity", by_name)


def _site_packages_on_path() -> bool:
    from unify import environment

    return str(environment.site_packages()) in sys.path


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
        out = await self.tools["execute_code"].fn(
            thought="A step.",
            code=code,
            **kwargs,
        )
        assert out.error is None, out.error
        return out

    async def close(self) -> None:
        from unify.actor.execution import _CURRENT_SANDBOX

        _CURRENT_SANDBOX.reset(self._token)
        await self.sandbox.close()
        await self.actor.close()


def _stdout(out) -> str:
    from unify.actor.execution.types import parts_to_text

    return (
        parts_to_text(out.stdout) if isinstance(out.stdout, list) else str(out.stdout)
    )


# ── (a) every library call from a cell, with the harness watched ────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_library_reads_from_cells_bind_without_executing_in_the_harness(
    core_world,
    harness,
    ranked,
):
    from unify import environment

    cells = _Cells(new_actor())
    try:
        await cells(
            f"await functions.add({HALVE!r}, dependencies=['packaging>=20'])\n"
            f"await functions.add({DOUBLE!r})\n"
            f"await functions.add({SPARE!r})",
        )
        out = await cells("row = await functions.get('double')\nrow['name']")
        assert out.result == "double"
        out = await cells("double(21)")
        assert out.result == 42
        out = await cells("halve(9)")
        assert out.result == 4
        out = await cells("help(double)")
        assert "double(x:" in _stdout(out)
        assert "Return twice its argument" in _stdout(out)
        out = await cells("[r['name'] for r in await functions.search('twice')]")
        assert "double" in out.result
        out = await cells(
            "[r['name'] for r in await functions.filter(filter=\"name = 'spare'\")]",
        )
        assert out.result == ["spare"]
        # Before any call of it records a case, which would refuse the change.
        out = await cells(
            "await functions.patch('spare', 'return x', 'return x + 0', "
            "why='Same value, written out.')",
        )
        out = await cells("(await functions.get('spare'))['implementation']")
        assert "return x + 0" in out.result
        out = await cells("spare(5)")
        assert out.result == 5
        out = await cells("sorted(await functions.list())")
        assert out.result == ["double", "halve", "spare"]
        out = await cells("await functions.run('double', x=5)")
        assert out.result == 10
        # The start-of-task shortlist and a guidance read's linked names
        # bind through the same read.
        library = cells.sandbox.core_globals["functions"]
        assert library._bind_names(["double"], sandbox=cells.sandbox) == {
            "double": False,
        }
    finally:
        await cells.close()
    assert harness == [], f"the harness executed or activated {harness}"
    assert str(environment.site_packages()) not in sys.path


# ── (b) the choke points ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_with_python_in_the_worker_nothing_stored_executes_in_process(
    monkeypatch,
):
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[HALVE, DOUBLE])
    double = fm.filter_functions(filter="name = 'double'")[0]
    message = "would execute model-written code in the harness"
    with pytest.raises(RuntimeError, match=message):
        fm._create_in_process_callable(dict(double), namespace={})
    with pytest.raises(RuntimeError, match=message):
        fm._inject_dependencies(dict(double), namespace={}, visited=set())
    with pytest.raises(RuntimeError, match=message):
        await fm.execute_function(function_name="double", call_kwargs={"x": 1})

    namespace: dict = {}
    loaded = fm.filter_functions(
        filter="name = 'double'",
        _return_callable=True,
        _namespace=namespace,
    )
    assert [f.__name__ for f in loaded] == ["double"]
    assert {"double", "halve"} <= set(namespace)
    # Bound from the stored source: nothing of it is callable here.
    for name in ("double", "halve"):
        with pytest.raises(RuntimeError, match="runs only in the sandboxed worker"):
            namespace[name](1)
    assert namespace["double"].source == double["implementation"]
    assert "Return twice its argument" in (namespace["double"].__doc__ or "")


# ── (c) an act: the workspace environment stays off the harness's sys.path ───


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_an_act_with_python_in_the_worker_leaves_sys_path_alone(
    core_world,
    harness,
    ranked,
):
    import asyncio

    from tests import cache_discipline_helpers as h

    actor = new_actor(can_store=False)
    actor.function_manager.add_functions(implementations=[HALVE, DOUBLE])
    replies = (
        lambda: h.completion(
            calls=[
                (
                    "execute_code",
                    {"thought": "Use it.", "code": "print(double(4), halve(8))"},
                ),
            ],
        ),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Double four.", persist=False)
            result = await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    assert result == "done"
    tool_replies = [
        m["content"]
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "8 4" in str(tool_replies), tool_replies
    assert harness == [], f"the harness executed or activated {harness}"
    assert not _site_packages_on_path()


# ── control: Python in process keeps loading in process ─────────────────────


def test_with_python_in_process_a_read_still_loads_in_process(
    monkeypatch,
    harness,
):
    from unify.function_manager.function_manager import FunctionManager

    monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[HALVE, DOUBLE])
    harness.clear()
    namespace: dict = {}
    fm.filter_functions(
        filter="name = 'double'",
        _return_callable=True,
        _namespace=namespace,
    )
    assert "double" in harness and "callees of double" in harness
    assert "activate" in harness
    assert namespace["double"](21) == 42
