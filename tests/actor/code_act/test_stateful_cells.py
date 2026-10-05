"""Symbolic and sandboxed: ``UNIFY_STATEFUL_CELLS``: every cell runs in the task's session.

In the lean-all ARC LOW screen runs the model chose ``state_mode="stateless"``
for 97% of its code cells (it fills in every argument, and the tools around
``execute_code`` speak of running stateless), then typed the same grid into
cell after cell. An enum led by the stateful default still drew "stateless"
in 15 of 18 replayed first cells, so with the switch on ``execute_code`` has
no mode or session argument at all: every cell runs in session 0, the four
session tools are not offered, and the prompt names neither. Off, the tools
and the prompt are as shipped.

The model, where there is one, is the scripted transport of
tests/cache_discipline_helpers.py; the cells of the last test run in the
real sandboxed worker.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.helpers import _handle_project
from unify.actor import cell_state
from unify.actor import prompt_builders as pb
from unify.common.llm_helpers import method_to_schema
from unify.settings import SETTINGS

SESSION_TOOLS = set(cell_state.SESSION_TOOLS)


def _actor(**kwargs):
    from unify.actor.code_act_actor import CodeActActor

    kwargs.setdefault("environments", [])
    return CodeActActor(**kwargs)


def _schema(tools, name):
    tool = tools[name]
    return method_to_schema(getattr(tool, "fn", tool), name)


def _prompt(tools, environments=None):
    return pb.build_code_act_prompt(
        environments=environments or {},
        tools=tools,
        can_store=True,
        persist=True,
        turn_reviews=False,
        can_clarify=False,
    )


@pytest.fixture(params=["", "lean"])
def profile(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", request.param)
    return request.param


def test_off_by_default():
    assert SETTINGS.UNIFY_STATEFUL_CELLS is False
    assert not cell_state.enabled()


@pytest.mark.parametrize("workspace", ["", "sandboxed"])
def test_off_the_cell_tool_and_the_session_tools_are_as_shipped(monkeypatch, workspace):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", workspace)
    tools = dict(_actor().get_tools("act"))
    params = _schema(tools, "execute_code")["function"]["parameters"]["properties"]
    assert {"state_mode", "session_id", "session_name"} <= set(params)
    assert SESSION_TOOLS <= set(tools)
    doc = _schema(tools, "execute_code")["function"]["description"]
    assert "**state_mode**: omit it and the cell runs" in doc


@pytest.mark.parametrize("workspace", ["", "sandboxed"])
def test_on_execute_code_has_no_mode_and_no_session(monkeypatch, workspace):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", workspace)
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    tools = dict(_actor().get_tools("act"))
    code = _schema(tools, "execute_code")["function"]
    assert not {"state_mode", "session_id", "session_name"} & set(
        code["parameters"]["properties"],
    )
    assert "**Session**: every cell runs in this task's persistent session" in (
        code["description"]
    )
    for gone in (
        "stateless",
        "read_only",
        "list_sessions",
        "session_id=0",
        "per session_id",
    ):
        assert gone not in code["description"].split("Output")[0], gone
    assert not SESSION_TOOLS & set(tools)
    function = _schema(tools, "execute_function")["function"]
    assert "state_mode" in function["parameters"]["properties"]
    assert not {"session_id", "session_name"} & set(
        function["parameters"]["properties"],
    )
    assert "keep ``execute_code`` semantics" not in function["description"]
    assert '``"stateful"`` runs it in the session' in function["description"]


def test_on_a_mode_or_session_argument_is_refused_by_name(monkeypatch):
    from unify.common._async_tool.tools_data import _normalise_kwargs_for_bound_method

    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    tools = dict(_actor().get_tools("act"))
    fn = tools["execute_code"].fn
    allowed, unknown = _normalise_kwargs_for_bound_method(
        fn,
        {
            "thought": "t",
            "code": "x = 1",
            "state_mode": "stateless",
            "session_id": None,
        },
    )
    assert set(unknown) == {"state_mode", "session_id"}
    assert set(allowed) == {"thought", "code"}


def test_the_switch_reaches_only_the_actor_built_with_it(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    on = dict(_actor().get_tools("act"))
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", False)
    off = dict(_actor().get_tools("act"))
    assert (
        "state_mode"
        in _schema(off, "execute_code")["function"]["parameters"]["properties"]
    )
    assert (
        "state_mode"
        not in _schema(on, "execute_code")["function"]["parameters"]["properties"]
    )


def test_the_prompt_names_no_cell_mode_and_no_session_tool(monkeypatch, profile):
    off_tools = dict(_actor().get_tools("act"))
    off = _prompt(off_tools)
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    tools = dict(_actor().get_tools("act"))
    on = _prompt(tools)
    assert 'state_mode="stateless"' in off and "list_sessions()" in off
    for gone in (
        'state_mode="stateless"',
        'state_mode="read_only"',
        "list_sessions",
        "inspect_state",
        "the session's\n`state_mode`",
    ):
        assert gone not in on, gone
    # The notebook framing and the function modes stay.
    assert "persistent" in on and "**stateless** | `await func.stateless(...)`" in on
    # Only these sentences change: what is left is the shipped text.
    assert len(on) < len(off)


def test_read_only_reads_the_one_session(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    actor = _actor()
    # Inside act() session 0 is the task's sandbox; here, stand it in.
    monkeypatch.setattr(actor, "_session_exists", lambda *, session_id: session_id == 0)
    assert (
        actor._resolve_session(
            state_mode="read_only",
            session_id=None,
            session_name=None,
        )
        == 0
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", False)
    from unify.common.tool_errors import ToolInputError

    with pytest.raises(ToolInputError):
        actor._resolve_session(
            state_mode="read_only",
            session_id=None,
            session_name=None,
        )


@pytest.mark.parametrize("delegation", ["on", "off"])
def test_composes_with_delegation(monkeypatch, delegation):
    from unify.actor.environments import ActorEnvironment

    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", delegation)
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    envs = [ActorEnvironment()] if delegation == "on" else []
    actor = _actor(environments=envs)
    tools = dict(actor.get_tools("act"))
    assert (
        "state_mode"
        not in _schema(tools, "execute_code")["function"]["parameters"]["properties"]
    )
    prompt = _prompt(tools, environments=actor.environments)
    assert 'state_mode="stateless"' not in prompt and "list_sessions" not in prompt


# ── a real session ───────────────────────────────────────────────────────────


def _cell(code: str, **extra):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code, **extra})],
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
@pytest.mark.parametrize("surface", ["", "core"])
async def test_a_cell_keeps_what_the_last_one_computed_in_the_sandboxed_worker(
    world,  # noqa: F811
    monkeypatch,
    surface,
):
    """Lean prompt, worker Python, the switch on: cell 2 reads cell 1's value.

    The model's second cell asks for "stateless", as it did in 97% of the
    recorded cells: the call is refused naming the parameters, and the next
    cell, without it, still sees ``grid``.
    """
    from unify import db
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    if surface == "core":
        (world["state"] / "store.sqlite").unlink()
        db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", surface)
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    actor = _actor(
        function_manager=FunctionManager(include_primitives=False),
        guidance_manager=GuidanceManager(),
        can_store=False,
    )
    replies = (
        _cell("grid = [[1, 2], [3, 4]]"),
        _cell("print(sum(map(sum, grid)))", state_mode="stateless"),
        _cell(
            "import sys\nprint('SUM', sum(map(sum, grid)), 'ISOLATED', sys.flags.isolated)",
        ),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Add up the grid.", persist=False)
            result = await asyncio.wait_for(handle.result(), 150)
    finally:
        await actor.close()
        if surface == "core":
            db.reset_store()
    assert result == "done"
    first = provider.requests[0]
    code_tool = next(
        t for t in first["tools"] if t["function"]["name"] == "execute_code"
    )
    assert "state_mode" not in code_tool["function"]["parameters"]["properties"]
    assert not SESSION_TOOLS & {t["function"]["name"] for t in first["tools"]}
    replies_seen = [
        str(m["content"])
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert any(
        "state_mode" in r and "parameter" in r.lower() for r in replies_seen
    ), replies_seen
    assert any("SUM 10" in r for r in replies_seen), replies_seen
    # The cells ran in the worker (``python -I``), not in this process.
    assert any("SUM 10 ISOLATED 1" in r for r in replies_seen), replies_seen
    assert not any("NameError" in r for r in replies_seen), replies_seen
    json.dumps(provider.requests)  # the requests are plain JSON
