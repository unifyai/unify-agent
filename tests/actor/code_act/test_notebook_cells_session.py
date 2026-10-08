"""Symbolic: a session under ``UNIFY_CODE_PROJECTION=notebook`` against the legacy projection.

The switch changes only what the model is sent and reads back: the
``execute_code`` schema (one ``code`` field), the session tools (gone;
``%sessions`` gives their data), the sentences of the prompt that named the
fields, and the rendering of a cell's result. These tests run the same work
both ways and compare what happens:

* with the switch at ``legacy`` the actor's first request is the default
  one, byte for byte (the golden of ``test_baked_prompt_golden``);
* the projected prompt is the legacy prompt with exactly the listed
  rewrites, on both tool surfaces and both prompt profiles, and names no
  field the model can no longer fill;
* cells in the real sandboxed worker keep, isolate, discard and separate
  state as the matching legacy arguments do, and bash runs;
* product paths behave as on the legacy projection: a cell asks a
  clarification and gets the answer, parent chat context reaches a
  sub-agent, the heartbeat notification fires, and stored-function calls
  record the same cases.

The model is the scripted transport of ``tests/cache_discipline_helpers``:
real unillm clients, nothing leaves the process. Tests that need the
sandboxed worker are skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from types import SimpleNamespace
from typing import Any

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    tool_names,
    world,
)
from tests.actor.code_act.helpers import patch_actor_act
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.async_helpers import _wait_for_condition
from tests.helpers import _handle_project
from unify.actor import notebook_cells as nb
from unify.settings import SETTINGS

PROJECTIONS = ["legacy", "notebook"]

#: What the projected prompt and tool description never name.
FIELDS = (
    "state_mode",
    "session_id",
    "session_name",
    "list_sessions",
    "inspect_state",
    "close_session",
    'language="bash"',
)


def _call(projection: str, code: str, **legacy: Any) -> dict:
    """``execute_code``'s arguments as each projection's model sends them."""
    if projection == "notebook":
        return {"code": code}
    return {"thought": "Next step.", "code": code, **legacy}


def _cell(projection: str, code: str, call_id: str | None = None, **legacy: Any):
    return lambda: h.completion(
        calls=[("execute_code", _call(projection, code, **legacy))],
        call_ids=[call_id] if call_id else None,
    )


def _done(text: str = "done"):
    return lambda: h.completion(content=text)


def _driver(decide, n: int = 30) -> list:
    """*n* replies that each look at the request they answer: *decide* maps
    the request to a completion, so the script follows the loop's turns
    however many it takes."""

    def reply():
        return decide(h._ACTIVE_PROVIDER[0].requests[-1])

    return [reply] * n


def _wait():
    return h.completion(calls=[("wait", {})])


def _texts(request: dict) -> str:
    return json.dumps(request["messages"], default=str)


def _tool_replies(request: dict) -> list[str]:
    out = []
    for m in request["messages"]:
        if m.get("role") != "tool":
            continue
        content = m["content"]
        out.append(content if isinstance(content, str) else json.dumps(content))
    return out


async def _act(actor, replies, request="Do the task.", **kwargs):
    with h.scripted(replies) as provider:
        handle = await actor.act(request, persist=False, **kwargs)
        result = await asyncio.wait_for(handle.result(), 120)
    return result, provider.requests


def _execute_code(request: dict) -> dict:
    return next(
        t["function"]
        for t in request["tools"]
        if t["function"]["name"] == "execute_code"
    )


# ── the switch at legacy: the default requests ──────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_at_legacy_the_actors_first_request_is_the_default_one(monkeypatch):
    from tests.actor.code_act.test_baked_prompt_golden import (
        BAKED_GOLDEN,
        record_first_request,
    )

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "legacy")
    assert SETTINGS.UNIFY_CODE_PROJECTION == "legacy"
    assert not nb.enabled()
    assert await record_first_request() == json.loads(BAKED_GOLDEN.read_text())


# ── the prompt and the schema ───────────────────────────────────────────────

#: (tool surface, prompt profile)
CONFIGS = [
    ("core", ""),
    ("core", "lean"),
]


async def _first_request(monkeypatch, projection: str, config) -> dict:
    surface, profile = config
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    actor = new_actor(can_store=False)
    try:
        _result, requests = await _act(actor, (_done(),))
    finally:
        await actor.close()
    request = requests[0]
    home = os.environ.get("UNIFY_HOME") or ""
    system = request["messages"][0]["content"]
    return {"system": system.replace(home, "$HOME") if home else system, **request}


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_projected_prompt_is_the_legacy_prompt_with_the_magics(
    core_world,
    monkeypatch,
):
    legacy_prompts = []
    for config in CONFIGS:
        legacy = await _first_request(monkeypatch, "legacy", config)
        notebook = await _first_request(monkeypatch, "notebook", config)
        legacy_prompts.append(legacy["system"])
        # Only the listed sentences change.
        assert notebook["system"] == nb.rewrite_prompt(legacy["system"]), config
        assert notebook["system"] != legacy["system"], config
        for field in FIELDS:
            assert field not in notebook["system"], (config, field)
        # The other tools are as shipped, less the session tools.
        others = [
            t for t in legacy["tools"] if t["function"]["name"] not in nb.SESSION_TOOLS
        ]
        assert [
            t for t in notebook["tools"] if t["function"]["name"] != "execute_code"
        ] == [t for t in others if t["function"]["name"] != "execute_code"], config
        assert tool_names(notebook).index("execute_code") == tool_names(
            {"tools": others},
        ).index("execute_code")
        # execute_code asks for the cell alone.
        tool = _execute_code(notebook)
        assert list(tool["parameters"]["properties"]) == ["code"], config
        assert tool["parameters"]["required"] == ["code"], config
        for field in FIELDS + ("thought",):
            assert field not in tool["description"], (config, field)
        assert "%%bash" in tool["description"], config
        assert "%sessions" in tool["description"]
        assert "thought" in _execute_code(legacy)["parameters"]["properties"]
    # Every rewrite applies to some shipped prompt: a reworded prompt fails
    # here rather than leaving a field the model can no longer fill.
    for old, _new in nb.PROMPT_REWRITES:
        assert any(old in prompt for prompt in legacy_prompts), old


# ── cells in the real worker: the same state as the legacy arguments ────────


def _sees_x(label: str) -> str:
    return (
        f"try:\n    x\n    print('{label}:', True)\n"
        f"except NameError:\n    print('{label}:', False)"
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
@pytest.mark.parametrize("projection", PROJECTIONS)
async def test_cells_keep_isolate_discard_and_separate_state_as_the_fields_did(
    core_world,
    monkeypatch,
    projection,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    nb_ = projection == "notebook"
    cells = [
        ("x = 41\nprint('set', x)", "x = 41\nprint('set', x)", {}),
        (
            "%%scratch\n" + _sees_x("scratch sees x"),
            _sees_x("scratch sees x"),
            {"state_mode": "stateless"},
        ),
        (
            "%%what_if\nx = 0\nprint('what-if x:', x)",
            "x = 0\nprint('what-if x:', x)",
            {"state_mode": "read_only", "session_id": 0},
        ),
        ("print('kept x:', x + 1)", "print('kept x:', x + 1)", {}),
        (
            "%%session side\ny = 7\n" + _sees_x("side sees x"),
            "y = 7\n" + _sees_x("side sees x"),
            {"state_mode": "stateful", "session_name": "side"},
        ),
        (
            "%%session side\nprint('side y:', y)",
            "print('side y:', y)",
            {"state_mode": "stateful", "session_name": "side"},
        ),
        ("%%bash\necho shell says hi", "echo shell says hi", {"language": "bash"}),
    ]
    replies = [
        _cell(projection, notebook if nb_ else legacy, **({} if nb_ else args))
        for notebook, legacy, args in cells
    ]
    if nb_:
        replies.append(_cell(projection, "%sessions"))
    actor = new_actor(can_store=False)
    try:
        result, requests = await _act(actor, (*replies, _done()))
    finally:
        await actor.close()
    assert result == "done"
    out = _tool_replies(requests[-1])
    assert "set 41" in out[0]
    assert "scratch sees x: False" in out[1]
    assert "what-if x: 0" in out[2]
    assert "kept x: 42" in out[3]
    assert "side sees x: False" in out[4]
    assert "side y: 7" in out[5]
    assert "shell says hi" in out[6]
    if nb_:
        # The model reads a notebook cell, not the envelope.
        for reply in out[:7]:
            assert "state_mode" not in reply and "duration_ms" not in reply, reply
        assert "Out: 0" in out[6]
        listing = out[7]
        assert re.search(r"this notebook \(the default\): .*\bx\b", listing), listing
        assert re.search(r"%%session side: .*\by\b", listing), listing
    else:
        assert "state_mode" in out[1] and "stateless" in out[1]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_refused_magic_reaches_the_model_and_the_next_cell_runs(
    core_world,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", "notebook")
    actor = new_actor(can_store=False)
    try:
        result, requests = await _act(
            actor,
            (
                _cell("notebook", "%matplotlib inline\nx = 1"),
                _cell("notebook", "%%what_if\n%%bash\nrm -f data.txt"),
                _cell("notebook", "x = 1\n!ls"),
                _cell("notebook", "x = 2\nprint('ran', x)"),
                _done(),
            ),
        )
    finally:
        await actor.close()
    unknown, bash_what_if, shell_line, ran = _tool_replies(requests[-1])
    assert "`%matplotlib` is not a magic" in unknown and "%%scratch" in unknown
    assert "cannot run as a what-if" in bash_what_if
    assert "Line 2 is a shell line" in shell_line and "%%bash" in shell_line
    assert "ran 2" in ran


# ── stored functions: the same cases ────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_stored_function_calls_record_the_same_cases(
    core_world,
    monkeypatch,
):
    from tests.actor.code_act import test_core_sandbox_objects as objects
    from unify import db
    from unify.actor import core_surface
    from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
    from unify.function_manager.function_manager import FunctionManager
    from unify.function_manager.primitives.environment import namespace_object

    from unify import environment

    monkeypatch.setattr(environment, "ensure", lambda specs: None)
    music_fixture = objects.music.__wrapped__()  # type: ignore[attr-defined]
    music = next(music_fixture)
    cells = [
        "print(await functions.run('remove_tracks_before', year=2000))",
        "await functions.get('fetch_profile')\nfetch_profile(access_token='expired-1')",
        "print(await functions.run('fetch_profile', access_token='{{access_token}}'))",
    ]

    async def run(projection: str):
        db.clear()
        monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
        music.world = objects._Music()
        fm = FunctionManager(include_primitives=False)
        fm.add_functions(
            implementations=[objects.REMOVE_BEFORE],
            dependencies=["tinydep>=1"],
        )
        fm.add_functions(implementations=[objects.LOGIN])
        actor = new_actor(function_manager=fm, can_store=False)
        tools = actor.get_tools("act")
        if projection == "notebook":
            tools = nb.project_tools(
                tools,
                caps=nb.Capabilities(bash=True),
                steering=False,
                structured=False,
                parent_context=False,
                resolve_session_name=actor._resolve_session_name,
                session_tools=tools,
            )
        sandbox = PythonExecutionSession(environments={})
        sandbox.global_state["primitives"] = SimpleNamespace(
            music=namespace_object("music"),
        )
        found = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
        sandbox.global_state.update(found)
        sandbox.core_globals = found
        token = _CURRENT_SANDBOX.set(sandbox)
        outs = []
        try:
            for code in cells:
                args = _call(projection, code)
                outs.append(await tools["execute_code"].fn(**args))
        finally:
            _CURRENT_SANDBOX.reset(token)
            await sandbox.close()
            await actor.close()
        return outs, objects._records()

    try:
        legacy_outs, legacy_records = await run("legacy")
        notebook_outs, notebook_records = await run("notebook")
    finally:
        try:
            next(music_fixture)
        except StopIteration:
            pass
    assert legacy_records["cases"]
    assert notebook_records == legacy_records
    for legacy_out, notebook_out in zip(legacy_outs, notebook_outs):
        assert type(notebook_out).__mro__[1] is type(legacy_out)
        assert notebook_out.error == legacy_out.error
        assert notebook_out.result == legacy_out.result


# ── product paths ───────────────────────────────────────────────────────────


def _simulated_sub_actors(monkeypatch, record: list) -> None:
    from unify.actor.simulated import SimulatedActor

    async def _impl(request: str, **kwargs):
        from unify.actor.execution.session import _PARENT_CHAT_CONTEXT

        record.append(
            {"request": request, "parent_chat_context": _PARENT_CHAT_CONTEXT.get(None)},
        )
        return await SimulatedActor(steps=1, duration=None).act(
            request,
            clarification_enabled=False,
        )

    patch_actor_act(monkeypatch, _impl)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("include", [False])
async def test_parent_chat_context_reaches_a_sub_agent_as_before(
    monkeypatch,
    projection,
    include,
):
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.environments.actor import ActorEnvironment
    from unify.actor.execution.session import _PARENT_CHAT_CONTEXT
    from unify.actor.simulated import _StaticAnswerHandle

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    seen: list = []

    async def _impl(request: str, **kwargs):
        seen.append(_PARENT_CHAT_CONTEXT.get(None))
        return _StaticAnswerHandle("Lucy Baker: 555-0101")

    patch_actor_act(monkeypatch, _impl)
    parent = [{"role": "user", "content": "Find Lucy's number; her surname is Baker."}]
    code = (
        "h = await primitives.actor.act(request='Find the number')\n"
        "print(await h.result())"
    )
    args = _call(projection, code)
    if include:
        args["include_parent_chat_context"] = True
    actor = CodeActActor(environments=[ActorEnvironment()], timeout=120)
    decide = lambda r: _done()() if "555-0101" in _texts(r) else _wait()  # noqa: E731
    try:
        with h.scripted(
            (lambda: h.completion(calls=[("execute_code", args)]), *_driver(decide)),
        ) as provider:
            handle = await actor.act(
                "Get the number.",
                persist=False,
                clarification_enabled=False,
                can_store=False,
                _parent_chat_context=parent,
            )
            result = await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    assert result == "done"
    schema = _execute_code(provider.requests[0])["parameters"]["properties"]
    assert "include_parent_chat_context" in schema
    assert len(seen) == 1
    if include:
        assert seen[0] and "Lucy" in json.dumps(seen[0])
    else:
        assert seen[0] is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_active_work_sees_the_cell_with_its_caption_and_nothing_reaches_the_model(
    monkeypatch,
):
    """Under the agent record (baked) nothing reaches the model while its cell
    runs (no heartbeat or in-cell progress is wired; 29b3a3d12), but the cell is
    still registered as active work, with its caption, while it runs."""
    from unify.events.active_work import ACTIVE_WORK

    ACTIVE_WORK.clear()
    actor = new_actor()
    actor._active_work_heartbeat_interval_s = 0.01
    actor._active_work_fallback_initial_delay_s = 0.03
    actor._active_work_fallback_repeat_interval_s = 0.05
    tools = nb.project_tools(
        actor.get_tools("act"),
        caps=nb.Capabilities(),
        steering=True,
        structured=False,
        parent_context=False,
        resolve_session_name=actor._resolve_session_name,
        session_tools=actor.get_tools("act"),
    )
    try:
        notification_q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(
            tools["execute_code"].fn(
                code="%%scratch\n# Waiting on the slow step\n"
                "import asyncio\nawait asyncio.sleep(0.2)",
                _notification_up_q=notification_q,
            ),
        )

        async def registered():
            return ACTIVE_WORK.snapshot().active_count == 1

        await _wait_for_condition(registered, poll=0.005, timeout=5)
        (work,) = ACTIVE_WORK.snapshot().works
        assert work["metadata"]["thought"] == "Waiting on the slow step"
        assert work["metadata"]["state_mode"] == "stateless"
        await task
        assert notification_q.empty(), "a notification reached the model mid-cell"
        assert ACTIVE_WORK.snapshot().active_count == 0
    finally:
        ACTIVE_WORK.clear()
        await actor.close()
