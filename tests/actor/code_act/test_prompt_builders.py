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
