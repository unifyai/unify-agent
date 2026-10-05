"""Symbolic: ``execute_function`` runs a stored entry point with its stored helpers.

``UNIFY_FUNCTION_HELPERS`` on the default (JSON) tool surface: a stored
entry point composed of stored helpers, run with ``execute_function`` as the
first step of a fresh session that has read nothing, in process and under
worker Python. The helpers are defined transitively (mutual recursion ends),
their declared packages installed, every level recorded (cases, usage,
trust) as ``functions.run`` records it under the core surface, a helper the
library no longer holds named in the error, and with the switch off the
shipped ``NameError``.

The hierarchy is ``tests/actor/code_act/library_world.py``'s. No model is
called; the worker runs in the real sandbox, so the tests are skipped where
bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401
from tests.actor.code_act.library_world import (
    HELPERS,
    HIERARCHY,
    SUMMARY,
    TEXT,
    new_actor,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_library_safety_e2e import GREET, PACKAGE, SHOUT, _wheel
from tests.helpers import _handle_project
from unify import db, environment
from unify.settings import SETTINGS

EVEN = (
    "def is_even(n: int) -> bool:\n"
    '    """Whether n is even, by mutual recursion with is_odd."""\n'
    "    return True if n == 0 else is_odd(n - 1)\n"
)
ODD = (
    "def is_odd(n: int) -> bool:\n"
    '    """Whether n is odd, by mutual recursion with is_even."""\n'
    "    return False if n == 0 else is_even(n - 1)\n"
)


@pytest.fixture(params=["in_process", "worker"])
def helpers_world(core_world, monkeypatch, request):  # noqa: F811
    """The default surface, Python in process or in the worker, the switch on,
    cases and trust recorded."""
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "")
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_WORKSPACE_PYTHON",
        "worker" if request.param == "worker" else "",
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_HELPERS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    return {**core_world, "python": request.param}


class Session:
    """A fresh act() session: the actor's own JSON tools over a bound sandbox."""

    def __init__(self) -> None:
        from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX

        self.actor = new_actor(can_store=False)
        self.tools = self.actor.get_tools("act")
        self.sandbox = PythonExecutionSession(environments={})
        self._token = _CURRENT_SANDBOX.set(self.sandbox)

    async def run(self, name: str, /, state_mode: str = "stateless", **kwargs):
        return await self.tools["execute_function"].fn(
            thought="Run it.",
            function_name=name,
            call_kwargs=kwargs,
            state_mode=state_mode,
            # read_only names the session it reads: the act() sandbox.
            session_id=0 if state_mode == "read_only" else None,
        )

    async def code(self, code: str, **kwargs):
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


def _store(*sources: str, **kwargs) -> None:
    from unify.function_manager.function_manager import FunctionManager

    FunctionManager(include_primitives=False).add_functions(
        implementations=list(sources),
        **kwargs,
    )


def _cases() -> dict[str, int]:
    out: dict[str, int] = {}
    for row in db.query(
        "SELECT f.name AS name FROM function_cases c "
        "JOIN functions f ON f.function_id = c.function_id",
    ):
        out[row["name"]] = out.get(row["name"], 0) + 1
    return out


def _trust() -> dict[str, tuple[int, int]]:
    return {
        r["name"]: (int(r["passes"]), int(r["failures"]))
        for r in db.query(
            "SELECT f.name AS name, t.passes AS passes, t.failures AS failures "
            "FROM function_trust t JOIN functions f ON f.function_id = t.function_id",
        )
    }


async def _usage(expected) -> dict[str, int]:
    """The usage counts once the off-loop usage writes have landed (or after 10 s)."""
    deadline = time.monotonic() + 10
    while True:
        usage = {
            r["name"]: int(r["usage_calls"] or 0)
            for r in db.query("SELECT name, usage_calls FROM functions")
        }
        if expected(usage) or time.monotonic() > deadline:
            return usage
        await asyncio.sleep(0.1)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_hierarchy_runs_in_a_fresh_session_and_every_level_is_recorded(
    helpers_world,
):
    """The first step of a session that has read nothing runs the entry
    point; its three helpers are found, and each call is recorded."""
    _store(*HIERARCHY)
    session = Session()
    try:
        out = await session.run("summarize_pairs", text=TEXT)
        assert out.error is None, out.error
        assert out.result == SUMMARY
        out = await session.run("summarize_pairs", text="c=7", sep=" | ")
        assert out.result == "c: 7", out.error
    finally:
        await session.close()
    cases = _cases()
    assert cases == {name: 2 for name in (*HELPERS, "summarize_pairs")}, cases
    trust = _trust()
    assert trust == {name: (2, 0) for name in (*HELPERS, "summarize_pairs")}, trust
    usage = await _usage(lambda u: all(u.get(n, 0) >= 2 for n in HELPERS))
    assert all(usage[n] == 2 for n in (*HELPERS, "summarize_pairs")), usage


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_helpers_stay_where_the_state_mode_keeps_definitions(helpers_world):
    """stateful: the entry point and its helpers stay defined in the session,
    so a later cell calls it by name; read_only: they do not stay."""
    _store(*HIERARCHY)
    session = Session()
    try:
        out = await session.code("x = 1")
        assert out.error is None, out.error
        out = await session.run("summarize_pairs", state_mode="read_only", text=TEXT)
        assert out.result == SUMMARY, out.error
        out = await session.code(
            "try:\n"
            "    format_totals\n"
            "    defined = True\n"
            "except NameError:\n"
            "    defined = False\n"
            "defined",
        )
        assert out.result is False, out.error
        out = await session.run("summarize_pairs", state_mode="stateful", text=TEXT)
        assert out.result == SUMMARY, out.error
        out = await session.code(f"summarize_pairs({TEXT!r}, sep=' / ')")
        assert out.error is None, out.error
        assert out.result == "a: 4 / b: 2"
    finally:
        await session.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_mutual_recursion_resolves_and_ends(helpers_world):
    _store(EVEN, ODD)
    session = Session()
    try:
        out = await session.run("is_even", n=7)
        assert out.error is None, out.error
        assert out.result is False
        out = await session.run("is_odd", n=7)
        assert out.result is True, out.error
    finally:
        await session.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_helper_the_library_no_longer_holds_is_named(helpers_world):
    from unify.function_manager.function_manager import FunctionManager

    _store(*HIERARCHY)
    fm = FunctionManager(include_primitives=False)
    fid = fm._get_function_data_by_name(name="format_totals")["function_id"]
    assert fm.delete_function(function_id=fid, delete_dependents=False) == {
        "format_totals": "deleted",
    }
    session = Session()
    try:
        out = await session.run("summarize_pairs", text=TEXT)
        assert out.result is None
        assert "NameError: name 'format_totals' is not defined" in out.error
        assert (
            "`summarize_pairs` calls the stored function `format_totals`, which "
            "the function library no longer holds"
        ) in out.error, out.error
    finally:
        await session.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_helpers_declared_dependency_is_installed(helpers_world, monkeypatch):
    """The helper declares a package; the entry point calling it declares
    nothing. Running the entry point first installs it."""
    directory = helpers_world["home"] / "wheels"
    directory.mkdir()
    requirement = f"{PACKAGE} @ {_wheel(directory).as_uri()}"
    monkeypatch.setenv("UV_OFFLINE", "1")
    _store(SHOUT, dependencies=[requirement])
    _store(GREET)
    assert environment.missing([requirement]) == [requirement]
    session = Session()
    try:
        out = await session.run("greet", name="ada")
        assert out.error is None, out.error
        assert out.result == "HELLO ADA!"
        assert environment.missing([requirement]) == []
    finally:
        await session.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_switch_off_is_the_shipped_name_error(helpers_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_HELPERS", False)
    _store(*HIERARCHY)
    session = Session()
    try:
        out = await session.run("summarize_pairs", text=TEXT)
        assert out.result is None
        assert "NameError" in out.error, out.error
        assert "is not defined" in out.error
        assert "no longer holds" not in out.error
        # Only the entry point was recorded, as shipped.
        assert set(_cases()) == {"summarize_pairs"}, _cases()
    finally:
        await session.close()
