"""Symbolic: ``UNIFY_PROMPT_TRIM``: prompt text describes only what the session has.

A lean-all ARC session has no primitives environment (``UNIFY_DELEGATION=off``,
no registered namespace), yet its 11.5k-character prompt and its code tools
told it how to route corrections to ``primitives.*`` handles, that function
search covers the primitives catalogue, that a sub-agent cannot take a reply
action for it, and that ``include_parent_chat_context`` passes the
conversation to ``execute_code`` (only primitives read it). With the switch
on each of these is left out where its capability is absent, and kept, word
for word, where it is present. Off, nothing changes.

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
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    return request.param


@pytest.fixture
def trim(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", True)


def test_off_by_default():
    assert SETTINGS.UNIFY_PROMPT_TRIM is False


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
    for name in ("execute_code", "execute_function", "store_skills"):
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
        "### Function & Guidance Library",
        "### Skill Storage",
    ):
        assert kept in prompt, kept


def test_with_primitives_the_text_is_as_shipped(profile, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "on")
    off = _render(_with_primitives())
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", True)
    on = _render(_with_primitives())
    assert on == off
    assert "`primitives.*`" in on[0]
    assert (
        "include_parent_chat_context"
        in on[1]["execute_code"]["function"]["parameters"]["properties"]
    )


def test_off_without_primitives_the_text_is_as_shipped(profile, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    prompt, schemas = _render(_actor())
    assert "When a correction concerns work already running in `primitives.*`" in prompt
    assert (
        "include_parent_chat_context"
        in schemas["execute_code"]["function"]["parameters"]["properties"]
    )


def test_each_trim_removes_only_its_own_text(trim, monkeypatch):
    """The lean profile, delegation off: every line the switch removes or
    changes names a primitive, a sub-agent, a handle or the search scope."""
    import difflib

    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    on, _ = _render(_actor())
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", False)
    off, _ = _render(_actor())
    removed = [
        line[1:]
        for line in difflib.unified_diff(off.splitlines(), on.splitlines(), n=0)
        if line.startswith("-") and not line.startswith("---")
    ]
    assert removed
    text = " ".join(removed)
    for subject in ("primitive", "sub-agent", "handle", "Discovery index scope"):
        assert subject in text, subject
    assert len(on) < len(off) - 1000


def test_the_reply_note_keeps_sub_agents_while_delegation_is_on(trim, monkeypatch):
    """A sub-agent can exist (delegation on) though this actor cannot start
    one: the note still says what a sub-agent should do."""
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "on")
    prompt, _ = _render(_actor())
    assert "If you are a sub-agent" in prompt
    assert "Do not delegate a sub-task" not in prompt


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


def test_a_core_session_prompt_names_no_primitive(trim, monkeypatch):
    """UNIFY_TOOL_SURFACE=core, lean, no primitives: the shared sections are
    trimmed as in the JSON-tool prompt."""
    from unify.actor.core_surface import PromptSurface

    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "off")
    core = PromptSurface(steering=True)
    kwargs = dict(environments={}, can_store=True, persist=True, core=core)
    on = pb.build_code_act_prompt(**kwargs)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_TRIM", False)
    off = pb.build_code_act_prompt(**kwargs)
    assert "primitives" in off
    assert "primitives" not in on and "SteerableToolHandle" not in on
