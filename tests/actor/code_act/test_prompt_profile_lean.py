"""Symbolic: ``UNIFY_PROMPT_PROFILE=lean``: a requester-first prompt for a non-interactive session.

The shipped actor prompt was written for a chat colleague: a user who only
hears ``send_notification``, clarification norms for labelling a user's
emails, an Uncertainties section ending every answer, an attachments
table, pacing rules for browser and UI automation that was removed, and
about 70 absolute directives. On a benchmark none of these has a reader,
and the requester's own reply format ("one JSON line, last") comes only
after 7.6k tokens of it. The lean profile opens with the role and the
reply format, keeps the mechanisms, and drops what has no reader.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS


def _flat(text: str) -> str:
    return " ".join(text.split())


def _prompt(*, can_clarify: bool = False, persist: bool = True) -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
        persist=persist,
        turn_reviews=False,
        can_clarify=can_clarify,
    )


@pytest.fixture
def lean(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")


def test_lean_opens_with_the_role_and_the_reply_format(lean):
    prompt = _prompt()
    assert prompt.startswith(pb._LEAN_ROLE)
    assert "each reply follows that format exactly" in _flat(prompt.split("###")[1])


def test_lean_puts_the_reply_protocol_note_right_after_the_role(lean, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", True)
    prompt = _prompt()
    assert prompt.startswith(f"{pb._LEAN_ROLE}\n\n{pb._REPLY_PROTOCOL_NOTE}")
    assert prompt.count("### Actions Taken By Replying") == 1


@pytest.mark.parametrize(
    "gone",
    [
        "send_notification",  # notification channel
        "Uncertainties",  # the required ending
        "Proactive clarification",
        "request clarification",
        "request_clarification",
        "Attachments/",  # attachments table
        "Outputs/",
        "browser automation",  # pacing for removed tools
        "UI clicks",
        "Semantic downgrades are bugs",  # long query_llm doctrine
        "Triage my inbox",
        "**This is the preferred tool",
        "IMPORTANT — single-call rule",
        "Reach\nfor `execute_code` only",
    ],
)
def test_lean_drops_text_with_no_reader(lean, gone):
    assert gone not in _prompt()


def test_lean_keeps_the_mechanisms(lean):
    prompt = _prompt()
    for kept in (
        "### Tools",
        "| `query_llm` / `list_llms` |",  # the globals table
        "async def query_llm(",
        "### Code And Function Calls",
        "### Responding to a steering checkpoint",  # steering: its own switch
        "### Python First",
        "### Execution",
        "### Verify Before Scaling",
        "### Function & Guidance Library",  # library: its own switches
        "### Skill Storage",
        "### Current Time",
        "### Workspace",
    ):
        assert kept in prompt, kept


def test_lean_leaves_the_library_and_storage_sections_to_their_switches(
    lean,
    monkeypatch,
):
    lean_prompt = _prompt()
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", True)
    shipped = _prompt()
    start = shipped.index("### Function & Guidance Library")
    end = shipped.index("### Current Time")
    assert shipped[start:end] in lean_prompt


def test_lean_states_clarification_as_a_mechanism_when_the_session_has_it(lean):
    prompt = _prompt(can_clarify=True)
    assert "7. **Clarification**: `request_clarification` asks" in prompt
    assert "Proactive clarification" not in prompt


def test_lean_text_states_no_absolutes_and_names_no_benchmark():
    lean_text = "\n".join(
        [
            pb._LEAN_ROLE,
            pb._LEAN_QUERY_LLM,
            pb._LEAN_SUB_ACTOR_DIAL,
            pb._LEAN_TOOL_SELECTION,
            pb._LEAN_EXECUTION_RULES,
            pb._LEAN_CLARIFICATION_RULE,
            pb._LEAN_INCREMENTAL_EXECUTION,
        ],
    )
    for absolute in ("MUST", "HARD", "never", "always", "Always", "do not", "Do not"):
        assert absolute not in lean_text, absolute
    words = set(re.findall(r"[a-z]+", lean_text.lower()))
    for word in ("arc", "appworld", "scienceworld", "crafter", "grid", "demo"):
        assert word not in words


def test_lean_is_much_shorter(lean, monkeypatch):
    lean_prompt = _prompt()
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    shipped = _prompt()
    assert len(lean_prompt) < len(shipped) - 6_000


def test_lean_workspace_line_with_the_clock_in_the_first_message(lean):
    context = pb.build_session_context({"execute_code": object()})
    assert "### Workspace" in context and "Attachments" not in context


def test_off_the_prompt_is_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    prompt = _prompt()
    assert prompt.startswith("### Role\n\nYou are an expert agent")
    assert pb._EXECUTION_RULES in prompt


# ── the request a scripted act() sends ──────────────────────────────────


async def _first_request() -> dict:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act(
                "List the files in the workspace.",
                persist=False,
                clarification_enabled=False,
            )
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests)


def _descriptions(request: dict) -> dict[str, str]:
    return {
        t["function"]["name"]: t["function"]["description"]
        for t in request["tools"] or []
    }


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_lean_act_sends_the_lean_prompt_and_tools(lean):
    session = await _first_request()
    first, later = session[0], session[1]
    system = "\n".join(m["content"] for m in first["messages"] if m["role"] == "system")
    assert system.startswith(pb._LEAN_ROLE)
    assert "## Parent Chat Context" not in system  # every accuracy fix
    tools = _descriptions(later)
    assert "send_notification" not in tools
    assert "single-call rule" not in tools["execute_code"]
    assert "preferred tool" not in tools["execute_function"]
    assert "MUST" not in tools["install_python_packages"]
    assert "stop_execute_" not in json.dumps(tools)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_act_keeps_the_notification_tool(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    session = await _first_request()
    assert "send_notification" in _descriptions(session[1])


@pytest.mark.parametrize(
    "value, expected",
    [("lean", "lean"), ("LEAN", "lean"), ("", "")],
)
def test_the_setting_parses(value, expected):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_PROMPT_PROFILE=value).UNIFY_PROMPT_PROFILE == expected
    )


def test_the_setting_rejects_other_values():
    from unify.settings import ProductionSettings

    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_PROMPT_PROFILE="short")


def test_lean_implies_every_accuracy_fix():
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_PROMPT_PROFILE="lean").prompt_accuracy()
    assert not ProductionSettings().prompt_accuracy()
