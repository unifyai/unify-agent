"""
Tests for CodeActActor prompt builder quality.

These tests are intentionally "high-signal string assertions" rather than
snapshot tests: the prompt is pinned byte for byte by
tests/actor/code_act/test_baked_prompt_golden.py; these check how it
composes environments and guidelines.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from unify.actor.prompt_builders import build_code_act_prompt


class _DummyEnv:
    """Minimal environment stub for build_code_act_prompt (prompt-context only)."""

    def __init__(self, prompt_context: str):
        self._prompt_context = prompt_context

    def get_prompt_context(self) -> str:
        return self._prompt_context


@pytest.mark.timeout(30)
def test_multiple_custom_environments_all_included():
    """Multiple environments each have their prompt context included."""
    marker_a = "### Alpha Environment\nAlpha-specific guidance for the LLM."
    marker_b = "### Beta Environment\nBeta-specific guidance for the LLM."
    envs: Mapping[str, Any] = {
        "alpha": _DummyEnv(marker_a),
        "beta": _DummyEnv(marker_b),
    }
    prompt = build_code_act_prompt(environments=envs)
    assert marker_a in prompt
    assert marker_b in prompt


@pytest.mark.timeout(30)
def test_custom_environment_empty_prompt_context_excluded():
    """Environments returning empty prompt context inject no noise."""
    envs: Mapping[str, Any] = {
        "empty_env": _DummyEnv(""),
        "whitespace_env": _DummyEnv("   \n  "),
    }
    assert build_code_act_prompt(environments=envs) == build_code_act_prompt(environments={})


@pytest.mark.timeout(30)
def test_guidelines_are_one_section_only_when_given():
    assert "### Guidelines" not in build_code_act_prompt(environments={}, guidelines=None)
    prompt = build_code_act_prompt(environments={}, guidelines="Check every input field.")
    assert prompt.count("### Guidelines") == 1
    assert "Check every input field." in prompt
    assert "Follow these guidelines throughout this session:" in prompt


@pytest.mark.timeout(30)
def test_a_structured_answer_adds_its_section():
    plain = build_code_act_prompt(environments={})
    structured = build_code_act_prompt(environments={}, structured=True)
    assert "### Answer" not in plain
    assert "### Answer" in structured


@pytest.mark.timeout(30)
def test_the_clock_is_the_host_clock_and_a_tasks_own_date_governs():
    """The prompt states the clock it was built at as the host's, for "today", and says that a date the task or its
    environment states as its own current date governs the task. It never says to use the host clock for "today" over
    such a date: a request dated in its own world would then meet two "todays" and the host's would win."""
    from unify.common import prompt_helpers

    prompt = build_code_act_prompt(environments={})
    section = prompt.split("### Current Time", 1)[1].split("###", 1)[0]
    text = " ".join(section.split())
    assert f"The host clock reads **{prompt_helpers.now()}**." in text, text
    assert 'Use it for "today" unless the task or its environment states its own current date' in text, text
    assert "then that date governs the task." in text, text
    assert "The current date and time is" not in text and 'Use this clock for "today"' not in text, text
