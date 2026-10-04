"""Symbolic: the sandbox objects of ``UNIFY_TOOL_SURFACE=core``, called from cells.

``functions``, ``guidance``, ``install``, ``read_file`` and ``grep`` are harness
objects a cell reaches through the sandboxed worker's proxy. ``functions.run``
replaces the ``execute_function`` tool: it runs the stored code in the worker,
confined, and records the call as that tool does -- usage, a
``UNIFY_FUNCTION_CASES`` case with the environment calls it made (so a later
change is replayed against it), ``UNIFY_STORE_TRUST`` evidence (a failure the
caller caused is not held against the function, and the error says so), and
the declared dependencies installed first. These tests make the same calls
both ways and compare every stored record. A stored function called by name
is recorded the same way (as shipped the worker only notes its use).
``state`` is ``execute_function``'s ``state_mode``; writes the session may not
make are refused with the reason; ``help()`` prints the objects' docs.

Cells run through the actor's own ``execute_code`` tool in the real sandboxed
worker, against a fake registered environment; no model is called. The tests
are skipped where bubblewrap is missing.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor as _actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify import db
from unify.actor import core_surface
from unify.settings import SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"


# ── functions.run against execute_function: the same records ───────────────

REMOVE_BEFORE = (
    "def remove_tracks_before(year: int) -> int:\n"
    "    removed = 0\n"
    "    for track in primitives.music.list_tracks():\n"
    "        if track['year'] < year:\n"
    "            primitives.music.remove_track(track_id=track['id'])\n"
    "            removed += 1\n"
    "    return removed\n"
)
LOGIN = (
    "def fetch_profile(access_token: str) -> dict:\n"
    "    if not access_token.isalnum():\n"
    "        raise PermissionError(\n"
    "            f'401 Unauthorized: {access_token!r} is not a token'\n"
    "        )\n"
    "    return {'user': 'ada'}\n"
)


class _Music:
    def __init__(self) -> None:
        self.tracks = [
            {"id": 1, "year": 1990},
            {"id": 2, "year": 2005},
            {"id": 3, "year": 2020},
        ]

    def list_tracks(self, **kwargs):
        return [dict(t) for t in self.tracks]

    def remove_track(self, track_id: int):
        self.tracks = [t for t in self.tracks if t["id"] != track_id]
        return {"removed": track_id}


@pytest.fixture
def music():
    """A registered environment namespace whose calls cases record."""
    from unify.function_manager import function_manager as fm_module
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    clear_environment_namespaces()
    box = SimpleNamespace(world=_Music())
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="music",
                    methods=(
                        EnvironmentMethod(
                            name="list_tracks",
                            call=lambda **kw: box.world.list_tracks(**kw),
                            effect="read",
                        ),
                        EnvironmentMethod(
                            name="remove_track",
                            call=lambda track_id: box.world.remove_track(track_id),
                            effect="destructive",
                            signature="(track_id: int)",
                        ),
                    ),
                ),
            ),
        ),
        source="tests:music",
    )
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    yield box
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


def _unsalted(value: Any) -> Any:
    """*value* with each credential placeholder's per-call salt removed."""
    return json.loads(re.sub(r"<redacted:[0-9a-f]+>", "<redacted>", json.dumps(value)))


def _records() -> dict:
    """Every case and trust row, without timestamps, ids or redaction salts."""
    cases = []
    for row in db.query("SELECT * FROM function_cases ORDER BY case_id"):
        row = dict(row)
        for key in ("case_id", "recorded_at", "session"):
            row.pop(key, None)
        call = json.loads(row["call"]) if row.get("call") else None
        if isinstance(call, dict):
            call.pop("salt", None)
        row["call"] = call
        row["trace"] = json.loads(row["trace"]) if row.get("trace") else None
        cases.append(row)
    trust = []
    for row in db.query("SELECT * FROM function_trust ORDER BY function_id"):
        row = dict(row)
        row.pop("updated_at", None)
        trust.append(row)
    return _unsalted({"cases": cases, "trust": trust})


async def _reuse(monkeypatch, music, *, core: bool, calls: list[tuple[str, dict]]):
    """The given stored-function calls in one session, through the
    ``execute_function`` tool (switch off) or ``functions.run`` (switch on),
    in a fresh store; returns what each call returned, the records, the
    usage notes and the dependency installs."""
    from unify import environment
    from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
    from unify.function_manager.function_manager import FunctionManager
    from unify.function_manager.primitives.environment import namespace_object

    db.clear()
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core" if core else "")
    music.world = _Music()
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[REMOVE_BEFORE], dependencies=["tinydep>=1"])
    fm.add_functions(implementations=[LOGIN])
    installs: list = []
    monkeypatch.setattr(environment, "ensure", lambda specs: installs.append(specs))
    used: list = []
    note = fm._note_function_use
    monkeypatch.setattr(
        fm,
        "_note_function_use",
        lambda data: (used.append(data["name"]), note(data)),
    )
    actor = _actor(function_manager=fm, can_store=False)
    tools = actor.get_tools("act")
    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = SimpleNamespace(
        music=namespace_object("music"),
    )
    if core:
        objects = core_surface.sandbox_objects(
            actor,
            policy=core_surface.WritePolicy(),
        )
        sandbox.global_state.update(objects)
        sandbox.core_globals = objects
    token = _CURRENT_SANDBOX.set(sandbox)
    outs = []
    try:
        for name, kwargs in calls:
            if core:
                args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                out = await tools["execute_code"].fn(
                    thought="Reusing a stored function.",
                    code=f"await functions.run({name!r}, {args})",
                )
            else:
                tool = tools["execute_function"]
                out = await tool.fn(
                    thought="Reusing a stored function.",
                    function_name=name,
                    call_kwargs=kwargs,
                    state_mode="stateful",
                )
            outs.append(out)
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()
    return outs, _records(), used, installs


def _last_line(text: Any) -> str:
    return str(text).strip().splitlines()[-1] if text else ""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_functions_run_records_what_execute_function_records(
    core_world,
    music,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    calls = [
        ("remove_tracks_before", {"year": 2000}),
        ("fetch_profile", {"access_token": "expired-1"}),
        ("fetch_profile", {"access_token": "{{access_token}}"}),
    ]
    json_outs, json_records, json_used, json_installs = await _reuse(
        monkeypatch,
        music,
        core=False,
        calls=calls,
    )
    core_outs, core_records, core_used, core_installs = await _reuse(
        monkeypatch,
        music,
        core=True,
        calls=calls,
    )
    # The same results and errors reach the model.
    assert json_outs[0].result == core_outs[0].result == 1
    for json_out, core_out in zip(json_outs[1:], core_outs[1:]):
        assert json_out.result is None and core_out.result is None
        assert "401 Unauthorized" in core_out.error
    # A failure the caller caused is not held against the function, and the
    # reply says so, in both.
    assert "Not counted against the stored function `fetch_profile`" in (
        json_outs[2].error
    )
    assert "Not counted against the stored function `fetch_profile`" in (
        core_outs[2].error
    )
    # Usage, cases (with the environment calls in order) and trust: the same.
    assert core_used == json_used == [name for name, _ in calls]
    assert core_records == json_records, json.dumps(
        {"core": core_records, "json": json_records},
        default=str,
    )
    traced = core_records["cases"][0]
    assert [c["call"] for c in traced["trace"]] == [
        "music.list_tracks",
        "music.remove_track",
    ]
    assert traced["trace_complete"] == 1 and traced["args_shown"] == "year=2000"
    assert [r["failures"] for r in core_records["trust"]] == [0, 1]
    # The declared dependencies are installed before the call, once a call.
    assert core_installs == json_installs == [["tinydep>=1"]]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_stored_function_called_by_name_is_recorded_too(
    core_world,
    music,
    monkeypatch,
):
    """In core mode a call by name records a case and trust evidence, as the
    in-process boundary wrapper does; as shipped the worker only notes use."""
    from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
    from unify.function_manager.function_manager import FunctionManager
    from unify.function_manager.primitives.environment import namespace_object

    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[REMOVE_BEFORE, DOUBLE])
    actor = _actor(function_manager=fm, can_store=False)
    tools = actor.get_tools("act")
    sandbox = PythonExecutionSession(environments={})
    sandbox.global_state["primitives"] = SimpleNamespace(
        music=namespace_object("music"),
    )
    objects = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
    sandbox.global_state.update(objects)
    sandbox.core_globals = objects
    token = _CURRENT_SANDBOX.set(sandbox)
    try:
        found = await tools["execute_code"].fn(
            thought="Find them.",
            code="sorted(await functions.list())",
        )
        assert found.error is None, found.error
        assert sorted(found.result) == ["double", "remove_tracks_before"]
        ran = await tools["execute_code"].fn(
            thought="Call them by name.",
            code="(remove_tracks_before(2010), double(21))",
        )
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()
    assert ran.error is None and ran.result == (2, 42)
    records = _records()
    by_shown = {c["args_shown"]: c for c in records["cases"]}
    assert set(by_shown) == {"year=2010", "x=21"}
    assert [c["call"] for c in by_shown["year=2010"]["trace"]] == [
        "music.list_tracks",
        "music.remove_track",
        "music.remove_track",
    ]
    assert [r["passes"] for r in records["trust"]] == [1, 1]


# ── the sandbox objects ─────────────────────────────────────────────────────


class _Cells:
    """Cells of one core session, run through the actor's execute_code tool."""

    def __init__(self, actor, policy=None, extra=None):
        from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX

        self.actor = actor
        self.tools = actor.get_tools("act")
        self.sandbox = PythonExecutionSession(environments={})
        self.sandbox.global_state.update(extra or {})
        objects = core_surface.sandbox_objects(
            actor,
            policy=policy or core_surface.WritePolicy(),
        )
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


def _stdout(out) -> str:
    from unify.actor.execution.types import parts_to_text

    return (
        parts_to_text(out.stdout) if isinstance(out.stdout, list) else str(out.stdout)
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_help_prints_the_documentation_of_the_harness_objects(core_world):
    cells = _Cells(_actor(can_store=False))
    try:
        out = await cells("help()")
        index = _stdout(out)
        out = await cells("help(functions)")
        library = _stdout(out)
        out = await cells("help(functions.run)")
        run = _stdout(out)
        out = await cells("help(guidance.search)\nhelp(install)")
        more = _stdout(out)
        out = await cells(
            "def mine(a, b=2):\n    'Adds.'\n    return a + b\nhelp(mine)",
        )
        local = _stdout(out)
    finally:
        await cells.close()
    assert out.error is None, out.error
    for name in ("functions:", "guidance:", "install:", "read_file:", "grep:"):
        assert name in index, index
    assert "Methods:" in library
    assert "await functions.search(query: str = ''" in library
    assert "await functions.run(name" in library
    # Recording internals never show, and cannot be called (underscored).
    assert "_begin" not in library and "case_pending" not in library
    assert run.startswith(
        "await functions.run(name: str, /, *, state: str = 'stateless', **kwargs: Any)",
    )
    assert "stateless" in run and "read_only" in run
    assert "await guidance.search(references" in more
    assert "await install(packages" in more
    assert "mine(a, b=2)" in local and "Adds." in local


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_functions_run_states(core_world):
    actor = _actor(can_store=False)
    actor.function_manager.add_functions(
        implementations=[
            "def where() -> str:\n"
            "    try:\n"
            "        return f'sees {marker}'\n"
            "    except NameError:\n"
            "        return 'fresh'\n",
            "def bump() -> int:\n"
            "    global counter\n"
            "    counter = counter + 1\n"
            "    return counter\n",
        ],
    )
    cells = _Cells(actor)
    try:
        await cells("marker = 'cell'\ncounter = 1")
        out = await cells(
            "(await functions.run('where'), "
            "await functions.run('where', state='stateful'), "
            "await functions.run('where', state='read_only'))",
        )
        assert out.result == ("fresh", "sees cell", "sees cell"), out.error
        out = await cells(
            "a = await functions.run('bump', state='read_only')\n"
            "b = counter\n"
            "c = await functions.run('bump', state='stateful')\n"
            "(a, b, c, counter)",
        )
        assert out.result == (2, 1, 2, 2), out.error
        # stateful leaves it defined in the session
        out = await cells("where()")
        assert out.result == "sees cell", out.error
        out = await cells("await functions.run('where', state='global')")
        assert "state must be one of" in out.error
        out = await cells("await functions.run('nothing_by_this_name')")
        assert "NameError" in out.error
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_writes_the_session_may_not_make_are_refused_with_the_reason(
    core_world,
):
    cells = _Cells(
        _actor(),
        policy=core_surface.WritePolicy(can_store=False),
    )
    try:
        out = await cells(f"await functions.add({DOUBLE!r})")
        assert "PermissionError" in out.error and "can_store is off" in out.error
        # guidance.add stays allowed with can_store off, as shipped
        out = await cells("await guidance.add(title='t', content='c')")
        assert out.error is None, out.error
        # the recording API is not reachable from a cell
        out = await cells(
            "await functions._begin(name='x', mode='run', args=[], kwargs={})",
        )
        assert "do not cross the worker boundary" in out.error
    finally:
        await cells.close()
    cells = _Cells(
        _actor(),
        policy=core_surface.WritePolicy(admission_gated=True),
    )
    try:
        out = await cells("await guidance.add(title='t', content='c')")
        assert "UNIFY_STORE_ADMISSION" in out.error
        out = await cells("await functions.list()")
        assert out.error is None and out.result == {}
    finally:
        await cells.close()


def test_the_write_policy_matches_the_tools_the_json_surface_withholds():
    """Each refusal mirrors a JSON tool the shipped surface leaves out."""
    from unify.actor import code_act_actor

    source = open(code_act_actor.__file__).read()
    for method in sorted(core_surface._STORE_ONLY):
        family, name = method.split(".")
        assert core_surface.WritePolicy(can_store=False).refusal(method)
        assert core_surface.WritePolicy(admission_gated=True).refusal(method)
    for method in ("guidance.add", "guidance.update", "guidance.delete"):
        assert core_surface.WritePolicy(can_store=False).refusal(method) is None
        assert core_surface.WritePolicy(admission_gated=True).refusal(method)
    assert '"GuidanceManager_add_guidance"' in source


REMOVE_AT_OR_AFTER = REMOVE_BEFORE.replace(
    "track['year'] < year",
    "track['year'] >= year",
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_a_case_functions_run_recorded_refuses_a_change_of_behaviour(
    core_world,
    music,
    monkeypatch,
):
    """The recorded case replays (with its environment answers) against a
    new source, and a change that does something else is refused."""
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    outs, _records, _used, _installs = await _reuse(
        monkeypatch,
        music,
        core=True,
        calls=[("remove_tracks_before", {"year": 2010})],
    )
    assert outs[0].result == 2
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    out = fm.add_functions(
        implementations=[REMOVE_AT_OR_AFTER],
        overwrite=True,
        raise_on_error=False,
    )
    assert out["remove_tracks_before"].startswith(
        "error: 'remove_tracks_before' was not changed: the new source does "
        "something else on 1 recorded call(s) that worked before:",
    )
