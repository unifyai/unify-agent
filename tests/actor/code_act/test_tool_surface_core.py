"""Symbolic: ``UNIFY_TOOL_SURFACE=core`` makes ``execute_code`` the actor's only JSON tool.

As shipped the actor sends 31 JSON tool schemas with every request (about
11.7k tokens; 18,898 fixed tokens with the system prompt on an ARC first
call), though its own design is that everything is code: the function and
guidance libraries alone are 16 of the tools. With the switch on, the model
sees ``execute_code`` (plus ``final_response`` with a response format, and
the loop's steering tools only for an actor that can start sub-actors); the
libraries are the sandbox's ``functions`` and ``guidance`` objects,
``functions.run`` replaces ``execute_function``, and ``install``,
``read_file``, ``grep`` and ``request_clarification`` are awaitables there.
Those objects live in the harness and a cell reaches them only through the
sandboxed worker's proxy, so the actor refuses to start without worker
Python, and with the discovery gate on (it can only force JSON tools).

``functions.run`` runs the stored code in the worker, confined, and records
the call as ``execute_function`` does: usage, a ``UNIFY_FUNCTION_CASES`` case
with the environment calls it made, ``UNIFY_STORE_TRUST`` evidence (a failure
the caller caused is not held against the function, and says so), and the
declared dependencies installed first. A stored function called by name is
recorded the same way.

The model is a scripted transport (tests/cache_discipline_helpers.py): real
unillm clients, real requests, nothing leaves the process. Cells run in the
real sandboxed worker; the tests that need bubblewrap are skipped where it
is missing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor as _actor,
    tool_names as _tool_names,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify.actor import core_surface
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code})],
    )


def _tool_replies(request: dict) -> list[str]:
    return [m["content"] for m in request["messages"] if m.get("role") == "tool"]


async def _act(actor, replies, request="Do the task.", **act_kwargs):
    with h.scripted(replies) as provider:
        handle = await actor.act(request, persist=False, **act_kwargs)
        result = await asyncio.wait_for(handle.result(), 120)
    return result, provider.requests


# ── the setting ──────────────────────────────────────────────────────────────


def test_the_switch_is_off_by_default_and_takes_only_core():
    assert ProductionSettings().UNIFY_TOOL_SURFACE == ""
    assert ProductionSettings(UNIFY_TOOL_SURFACE="Core").UNIFY_TOOL_SURFACE == "core"
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_TOOL_SURFACE="python")


# ── end to end ───────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_a_session_finds_runs_and_calls_a_stored_function_in_code(
    core_world,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    actor = _actor(can_store=False)
    actor.function_manager.add_functions(implementations=[DOUBLE])
    replies = (
        _cell(
            "row = await functions.get('double')\n"
            "print(row['name'])\n"
            "print(await functions.run('double', x=4))",
        ),
        _cell("print(double(5))\nhelp(functions.run)"),
        lambda: h.completion(content="done"),
    )
    try:
        result, requests = await _act(actor, replies)
    finally:
        await actor.close()
    assert result == "done"
    assert _tool_names(requests[0]) == ["execute_code"]
    first, second = (json.dumps(r) for r in _tool_replies(requests[-1]))
    assert "double" in first and "8" in first, first
    assert "10" in second and "functions.run" in second, second


# ── the surface: what the model is sent ─────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
@pytest.mark.parametrize("profile", ["", "lean"])
@pytest.mark.parametrize("structured", [False, True])
async def test_the_only_json_tool_is_execute_code(
    core_world,
    monkeypatch,
    structured,
    profile,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", profile)
    from pydantic import BaseModel

    class Answer(BaseModel):
        text: str

    actor = _actor(can_store=False)
    reply = (
        (lambda: h.completion(calls=[("final_response", {"answer": {"text": "ok"}})]))
        if structured
        else (lambda: h.completion(content="done"))
    )
    try:
        _result, requests = await _act(
            actor,
            (reply,),
            **({"response_format": Answer} if structured else {}),
        )
    finally:
        await actor.close()
    first = requests[0]
    expected = ["execute_code", "final_response"] if structured else ["execute_code"]
    assert _tool_names(first) == expected
    system = first["messages"][0]["content"]
    # The prompt names the sandbox's objects, never a library JSON tool.
    assert "### Sandbox Objects" in system and "`functions`" in system
    for absent in (
        "FunctionManager_",
        "GuidanceManager_",
        "execute_function",
        "install_python_packages",
        "send_notification",
        "list_sessions",
        "inspect_state",
        "store_skills` tool",
    ):
        assert absent not in system, absent
    # execute_code's own description no longer prefers execute_function, or
    # steers through tools this session does not have.
    code_tool = first["tools"][0]["function"]["description"]
    assert "execute_function" not in code_tool
    assert "list_sessions" not in code_tool
    assert "stop_execute_code" not in code_tool


@pytest.mark.asyncio
@_handle_project
async def test_with_the_switch_off_the_session_keeps_every_json_tool(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    actor = _actor(can_store=False)
    try:
        _result, requests = await _act(actor, (lambda: h.completion(content="done"),))
    finally:
        await actor.close()
    names = _tool_names(requests[0])
    for name in (
        "execute_code",
        "execute_function",
        "FunctionManager_search_functions",
        "GuidanceManager_search",
        "install_python_packages",
        "compress_context",
        "wait",
        "steer",
    ):
        assert name in names, name
    assert "### Sandbox Objects" not in requests[0]["messages"][0]["content"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_steering_tools_come_only_with_sub_actors(core_world):
    from unify.actor.environments import ActorEnvironment

    actor = _actor(can_store=False, environments=[ActorEnvironment()])
    try:
        _result, requests = await _act(actor, (lambda: h.completion(content="done"),))
    finally:
        await actor.close()
    assert _tool_names(requests[0]) == [
        "execute_code",
        "wait",
        "steer",
        "ask_about_completed_tool",
    ]
    assert "### Responding to a steering checkpoint" in (
        requests[0]["messages"][0]["content"]
    )


# ── refusals at start ───────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@_handle_project
async def test_the_actor_refuses_to_start_without_confinement_or_with_the_gate(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
    actor = _actor(can_store=False)
    try:
        # The discovery gate can only force JSON tools.
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", True)
        with pytest.raises(core_surface.ToolSurfaceError, match="DISCOVERY_GATE=0"):
            await actor.act("Do the task.")
        # Cells in this process would hold the real store.
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
        with pytest.raises(core_surface.ToolSurfaceError, match="PYTHON=worker"):
            await actor.act("Do the task.")
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
        with pytest.raises(core_surface.ToolSurfaceError, match="WORKSPACE=sandboxed"):
            await actor.act("Do the task.")
        # execute_code is the surface's only tool.
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", "sandboxed")
        with pytest.raises(core_surface.ToolSurfaceError, match="can_compose"):
            await actor.act("Do the task.", can_compose=False)
    finally:
        await actor.close()
    # Nothing was started: no sandbox slot is held.
    assert actor._act_semaphore._value == 20


# ── clarification and the forked review ────────────────────────────────────


@pytest.mark.asyncio
async def test_request_clarification_in_a_cell_is_the_json_tool_in_python():
    """Where the session can ask: the caller's queues and the request and
    answer events, as the loop's JSON tool; where it cannot: no such name."""
    up, down = asyncio.Queue(), asyncio.Queue()
    events: list = []
    session = core_surface.Session(
        tools={},
        prompt=core_surface.PromptSurface(),
        steering=False,
        objects={},
        clarification=core_surface._clarification_factory(
            (up, down),
            lambda q: events.append(("asked", q)),
            lambda a: events.append(("answered", a)),
        ),
    )
    namespace: dict = {"request_clarification": "shipped"}
    token = session.enter()
    try:
        restore = core_surface.bind_clarification(namespace, None, None)
        ask = namespace["request_clarification"]
        await down.put("blue")
        assert await ask("Which colour?") == "blue"
        assert await up.get() == "Which colour?"
        assert events == [("asked", "Which colour?"), ("answered", "blue")]
        restore()
        assert namespace == {"request_clarification": "shipped"}
    finally:
        core_surface.Session.leave(token)
    # A session that cannot ask has no request_clarification.
    token = core_surface._CLARIFICATION.set(None)
    try:
        restore = core_surface.bind_clarification(namespace, up, down)
        assert "request_clarification" not in namespace
        restore()
    finally:
        core_surface._CLARIFICATION.reset(token)
    # Outside a core session nothing is touched.
    assert core_surface.bind_clarification(namespace, up, down) is None
    assert namespace == {"request_clarification": "shipped"}


def test_a_forked_review_falls_back_to_the_standalone_one(monkeypatch):
    from unify.actor.code_act_actor import _review_fork_source

    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
    source, reason = _review_fork_source(object(), object())
    assert source is None and "no library tools" in reason
