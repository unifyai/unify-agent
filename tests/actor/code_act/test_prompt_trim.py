"""Symbolic: the prompt trim (baked in): prompt text describes only what the session has.

A lean-all ARC session has no primitives environment (``UNIFY_DELEGATION=off``,
no registered namespace), yet its 11.5k-character prompt and its code tools
told it how to route corrections to ``primitives.*`` handles, that function
search covers the primitives catalogue, that a sub-agent cannot take a reply
action for it, and that ``include_parent_chat_context`` passes the
conversation to ``execute_code`` (only primitives read it). Each of these is left out where its capability is absent, and kept, word
for word, where it is present.

The visibility message the loop appends is tested in
tests/async_tool_loop/test_visibility_trim.py.
"""

from __future__ import annotations

import pytest

from unify.actor import prompt_builders as pb
from unify.common.llm_helpers import method_to_schema
from unify.settings import SETTINGS


def _actor(environments=None):
    from unify.actor.code_act_actor import CodeActActor

    return CodeActActor(environments=environments or [])


def _render(actor):
    tools = dict(actor.get_tools("act"))
    prompt = pb.build_code_act_prompt(
        environments=actor.environments,
        tools=tools,
        can_store=True,
        persist=True,
        turn_reviews=False,
        can_clarify=False,
        search_when_useful=True,
    )
    schemas = {
        name: method_to_schema(
            getattr(t, "fn", t),
            name,
            expose_context_control=True,
        )
        for name, t in tools.items()
    }
    return prompt, schemas


def _with_primitives():
    from unify.actor.environments import ActorEnvironment

    return _actor([ActorEnvironment()])


@pytest.fixture(params=["", "lean"])
def profile(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", request.param)
    return request.param


@pytest.fixture
def trim(monkeypatch):
    """Baked in at the code freeze: the behaviour this pinned is the only path."""


def test_without_primitives_no_text_names_them(profile, trim, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    prompt, schemas = _render(_actor())
    for gone in (
        "primitives",
        "primitive call",
        "sub-agent",
        "SteerableToolHandle",
        "Discovery index scope",
        "steerable handle reaches",
    ):
        assert gone not in prompt, gone
    for name in ("execute_code", "store_skills"):
        description = schemas[name]["function"]["description"]
        assert "primitive" not in description, name
        assert (
            "include_parent_chat_context"
            not in schemas[name]["function"]["parameters"]["properties"]
        )
    # What the session does have is still described.
    for kept in (
        "### Responding to a steering checkpoint",
        "`query_llm`",
        "### Skill Storage",
    ):
        assert kept in prompt, kept


def test_with_primitives_the_text_is_as_shipped(profile, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "on")
    off = _render(_with_primitives())
    on = _render(_with_primitives())
    assert on == off
    assert "`primitives.*`" in on[0]
    assert (
        "include_parent_chat_context"
        in on[1]["execute_code"]["function"]["parameters"]["properties"]
    )


def test_session_text_follows_the_session_tools(trim, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    actor = _actor()
    tools = dict(actor.get_tools("act"))
    kwargs = dict(environments={}, can_store=True, persist=True, can_clarify=False)
    with_tools = pb.build_code_act_prompt(tools=tools, **kwargs)
    assert "`list_sessions()` and" in with_tools
    for name in (
        "list_sessions",
        "inspect_state",
        "close_session",
        "close_all_sessions",
    ):
        tools.pop(name)
    without = pb.build_code_act_prompt(tools=tools, **kwargs)
    assert "list_sessions" not in without and "inspect_state" not in without
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    shipped = pb.build_code_act_prompt(tools=tools, **kwargs)
    assert "list_sessions" not in shipped
    assert "Variables survive context compression" in shipped


@pytest.mark.parametrize("cells", [False, True])
def test_composes_with_stateful_cells(trim, monkeypatch, cells):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", cells)
    prompt, schemas = _render(_actor())
    assert "primitives" not in prompt
    assert ('state_mode="stateless"' in prompt) is not cells
    assert (
        "state_mode" in schemas["execute_code"]["function"]["parameters"]["properties"]
    ) is not cells


def test_a_core_session_prompt_names_no_primitive(trim, monkeypatch):
    """UNIFY_TOOL_SURFACE=core, lean, no primitives: the shared sections are
    trimmed as in the JSON-tool prompt."""
    from unify.actor.core_surface import PromptSurface

    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    core = PromptSurface(steering=True)
    kwargs = dict(environments={}, can_store=True, persist=True, core=core)
    on = pb.build_code_act_prompt(**kwargs)
    assert "primitives" not in on and "SteerableToolHandle" not in on
