"""Symbolic: the prompt trim (baked in): prompt text describes only what the session has.

A lean-all ARC session has no primitives environment (no delegation,
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
from unify.actor.core_surface import PromptSurface
from unify.settings import SETTINGS


def _actor(environments=None):
    from unify.actor.code_act_actor import CodeActActor

    return CodeActActor(environments=environments or [])


def _render(actor):
    tools = dict(actor.get_tools("act"))
    prompt = pb.build_code_act_prompt(
        environments=actor.environments,
        can_store=True,
        persist=True,
        core=PromptSurface(clarification=False),
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
    return request.param


@pytest.fixture
def trim(monkeypatch):
    """Baked in at the code freeze: the behaviour this pinned is the only path."""


def test_without_primitives_no_text_names_them(profile, trim, monkeypatch):
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
        "`query_llm`",
        "### Skill Storage",
    ):
        assert kept in prompt, kept


def test_with_primitives_the_text_is_as_shipped(profile, monkeypatch):
    off = _render(_with_primitives())
    on = _render(_with_primitives())
    assert on == off
    assert "`primitives.*`" in on[0]
    assert (
        "include_parent_chat_context"
        in on[1]["execute_code"]["function"]["parameters"]["properties"]
    )


@pytest.mark.parametrize("cells", [False, True])
def test_composes_with_stateful_cells(trim, monkeypatch, cells):
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", cells)
    prompt, schemas = _render(_actor())
    assert "primitives" not in prompt
    assert ('state_mode="stateless"' in prompt) is not cells
    assert (
        "state_mode" in schemas["execute_code"]["function"]["parameters"]["properties"]
    ) is not cells


def test_a_core_session_prompt_names_no_primitive(trim, monkeypatch):
    """The core surface, lean, no primitives: the shared sections are
    trimmed."""
    on = pb.build_code_act_prompt(
        environments={},
        can_store=True,
        persist=True,
        core=PromptSurface(),
    )
    assert "primitives" not in on and "SteerableToolHandle" not in on
