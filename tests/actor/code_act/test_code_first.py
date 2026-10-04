"""Symbolic: ``UNIFY_CODE_FIRST``: compute a computable result with a program.

On Continual-ARC LOW (2 Oct) only 19 of the 190 forked storage reviews in
Unify's run came from episodes solved with executed code: the actor mostly
solved by reasoning in text, so its reviews had no program to store and a
return visit re-derived the rule. The published rows that solve with a
program and keep it (harness plus program-library procedure, 151-162 of
200; a controller-run program library, 164) beat plain harnesses (142-144)
and Unify (129-135). CodeAct, PAL / Program-of-Thoughts and Voyager report
the same direction for executable actions, code for computable reasoning
and code skills. The section asks for a program whenever a result can be
computed, with judgment kept as ``query_llm(...)`` inside it (the existing
query_llm dial), and defers to Tool Selection for a single call and to
Incremental Execution for side effects. It never tells the actor to check
the program against examples it was given (4 Oct): that discusses the task
under test with the agent, and stored functions are checked mechanically
instead (``UNIFY_FUNCTION_CASES`` replays their recorded calls). Sub-actors are
CodeActActors and build the same prompt. Deterministic: prompt text only.
"""

from __future__ import annotations

import re

import pytest

from unify.actor import prompt_builders as pb
from unify.settings import SETTINGS


def _prompt(monkeypatch, on: bool) -> str:
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_FIRST", on)
    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
    )


def test_on_adds_exactly_the_section_and_nothing_else(monkeypatch):
    off = _prompt(monkeypatch, False)
    on = _prompt(monkeypatch, True)
    assert "### Compute With Code" not in off
    assert on.count(pb._CODE_FIRST) == 1
    assert on.replace(pb._CODE_FIRST + "\n\n", "", 1) == off


@pytest.mark.parametrize("reply_note", [False, True])
def test_the_section_follows_the_execution_rules(monkeypatch, reply_note):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_PROTOCOL_NOTE", reply_note)
    prompt = _prompt(monkeypatch, True)
    start = prompt.index("### Compute With Code")
    assert prompt.index("### Execution Rules") < start
    assert start < prompt.index(pb._INCREMENTAL_EXECUTION[:40])
    if reply_note:
        assert prompt.index("### Actions Taken By Replying") < start
    # Static text: the cached prefix is the same for every session.
    again = _prompt(monkeypatch, True)
    assert again.split("### Current Time")[0] == prompt.split("### Current Time")[0]


def test_the_section_defers_to_the_rules_it_could_conflict_with():
    text = " ".join(pb._CODE_FIRST.split())
    # A single exact call stays execute_function, as Tool Selection says.
    assert "### Tool Selection" in pb._TOOL_SELECTION
    assert "If one stored function or primitive call is the whole task" in text
    # Side effects stay incremental.
    assert "### Incremental Execution" in pb._INCREMENTAL_EXECUTION
    assert "step by step, as Incremental Execution says" in text
    # Judgment stays semantic.
    assert "`query_llm(...)` calls inside the program" in text
    # Computing, not checking against given examples.
    assert "compute it with a program in `execute_code` rather than by hand" in text
    assert "A working program can be stored and reused" in text
    for phrase in ("example", "expected output", "known result", "check it"):
        assert phrase not in text.lower(), phrase


def test_the_section_names_no_benchmark():
    words = set(re.findall(r"[a-z]+", pb._CODE_FIRST.lower()))
    for word in (
        "arc",
        "grid",
        "grids",
        "puzzle",
        "demo",
        "demos",
        "submit",
        "appworld",
        "alfworld",
        "scienceworld",
        "benchmark",
    ):
        assert word not in words


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_parses_booleans(value, expected):
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_CODE_FIRST=value).UNIFY_CODE_FIRST is expected
