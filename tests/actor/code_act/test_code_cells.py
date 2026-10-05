"""Symbolic: ``UNIFY_CODE_ONLY_CELLS``: the code-cell tool asks for code alone.

In the Python-tool-mode ARC LOW runs (af8958e5d) 54-66% of the model's cells
were narration: a printed sentence, a comment or ``None``, with the reasoning
in ``execute_code``'s required ``thought`` ("shown to the user as the
rationale for this step") and no reasoning tokens. With the switch on
(unify/actor/code_cells.py) ``code`` is the only required argument and there
is no ``thought``; without primitives, no ``include_parent_chat_context``.

The model, where there is one, is the scripted transport of
tests/cache_discipline_helpers.py.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import core_surface
from unify.common.llm_helpers import method_to_schema
from unify.settings import ProductionSettings, SETTINGS

SWITCHES = ("UNIFY_CODE_ONLY_CELLS",)


def _actor(environments=None, **kwargs):
    from unify.actor.code_act_actor import CodeActActor

    return CodeActActor(environments=environments or [], **kwargs)


def _code_tool(actor, *, core=False):
    tools = actor.get_tools("act")
    if core:
        tools = core_surface.core_tools(tools, steering=False)
    tool = tools["execute_code"]
    return method_to_schema(
        getattr(tool, "fn", tool),
        "execute_code",
        expose_context_control=True,
    )["function"]


@pytest.fixture(params=["", "sandboxed"])
def workspace(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", request.param)
    return request.param


def test_off_by_default():
    defaults = ProductionSettings()
    for name in SWITCHES:
        assert getattr(defaults, name) is False, name
        assert getattr(ProductionSettings(**{name: "1"}), name) is True, name
        assert getattr(SETTINGS, name) is False, name


# ── UNIFY_CODE_ONLY_CELLS ────────────────────────────────────────────────────


@pytest.mark.parametrize("core", [False, True])
def test_off_the_cell_tool_is_as_shipped(workspace, core):
    tool = _code_tool(_actor(), core=core)
    params = tool["parameters"]
    assert params["required"] == ["thought"]
    assert params["properties"]["code"] == {"type": ["string", "null"]}
    assert "Shown to the user" in params["properties"]["thought"]["description"]
    assert "include_parent_chat_context" in params["properties"]


@pytest.mark.parametrize("core", [False, True])
def test_on_code_is_the_only_required_argument(workspace, core, monkeypatch):
    shipped = _code_tool(_actor(), core=core)
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    tool = _code_tool(_actor(), core=core)
    params = tool["parameters"]
    assert params["required"] == ["code"]
    assert list(params["properties"])[0] == "code"
    assert params["properties"]["code"] == {"type": "string"}
    assert "thought" not in params["properties"]
    assert "include_parent_chat_context" not in params["properties"]
    # Everything else is as shipped: the other arguments and the description.
    kept = {
        k: v
        for k, v in shipped["parameters"]["properties"].items()
        if k not in ("thought", "code", "include_parent_chat_context")
    }
    assert {k: v for k, v in params["properties"].items() if k != "code"} == kept
    assert tool["description"] == shipped["description"]
    assert ("language" in params["properties"]) == (workspace == "sandboxed")


def test_with_primitives_the_parent_context_flag_stays(monkeypatch):
    from unify.actor.environments import ActorEnvironment

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    tool = _code_tool(_actor([ActorEnvironment()]))
    assert "include_parent_chat_context" in tool["parameters"]["properties"]
    assert tool["parameters"]["required"] == ["code"]


def test_composes_with_stateful_cells_and_the_trim(workspace, monkeypatch):
    for name in ("UNIFY_CODE_ONLY_CELLS", "UNIFY_STATEFUL_CELLS", "UNIFY_PROMPT_TRIM"):
        monkeypatch.setattr(SETTINGS, name, True)
    for core in (False, True):
        params = _code_tool(_actor(), core=core)["parameters"]
        expected = ["code", "language"] if workspace else ["code"]
        assert list(params["properties"]) == expected
        assert params["required"] == ["code"]


def test_execute_function_is_unchanged(monkeypatch):
    def schema():
        tool = _actor(function_manager=_fm()).get_tools("act")["execute_function"]
        return method_to_schema(getattr(tool, "fn", tool), "execute_function")

    shipped = schema()
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    assert schema() == shipped


def _fm():
    from unify.function_manager.function_manager import FunctionManager

    return FunctionManager(include_primitives=False)


def test_the_switch_reaches_only_the_actor_built_with_it(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    on = _actor()
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", False)
    off = _actor()
    assert "thought" in _code_tool(off)["parameters"]["properties"]
    on_tool = on.get_tools("act")["execute_code"]
    assert (
        "thought"
        not in method_to_schema(
            getattr(on_tool, "fn", on_tool),
            "execute_code",
        )[
            "function"
        ]["parameters"]["properties"]
    )


def _cell(code: str, **extra):
    return lambda: h.completion(calls=[("execute_code", {"code": code, **extra})])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_thought_is_refused_and_a_cell_without_one_runs(monkeypatch):
    """In process: the call that still passes ``thought`` is refused naming
    the tool's parameters; the next one, without it, runs."""
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    actor = _actor(can_store=False)
    replies = (
        _cell("print('A' * 2)", thought="Announcing the step."),
        _cell("print('B' * 2)"),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Print two letters.", persist=False)
            result = await asyncio.wait_for(handle.result(), 90)
    finally:
        await actor.close()
    assert result == "done"
    tool_messages = [
        json.dumps(m["content"])
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "thought" in tool_messages[0] and "code" in tool_messages[0]
    assert "AA" not in tool_messages[0]
    assert "BB" in tool_messages[1]
