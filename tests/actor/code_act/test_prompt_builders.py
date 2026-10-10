"""
Tests for CodeActActor prompt builder quality.

These tests are intentionally "high-signal string assertions" rather than
snapshot tests. Since the code freeze the prompt is the core tool surface's
lean prompt (tests/actor/code_act/test_baked_prompt_golden.py pins it byte
for byte); these check how it composes environments and guidelines.
"""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.actor.core_surface import PromptSurface
from unify.actor.prompt_builders import build_code_act_prompt


def _prompt(**kwargs) -> str:
    return build_code_act_prompt(core=PromptSurface(), **kwargs)


class _DummyEnv:
    """Minimal environment stub for build_code_act_prompt (prompt-context only)."""

    def __init__(self, prompt_context: str):
        self._prompt_context = prompt_context

    def get_prompt_context(self) -> str:
        return self._prompt_context

    def get_tools(self) -> dict:
        return {}


def _real_envs_mixed() -> Mapping[str, Any]:
    """Real environments that produce self-contained prompt context."""
    from unify.actor.environments.actor import ActorEnvironment
    from unify.actor.environments.base import _CompositeEnvironment

    composite = _CompositeEnvironment([ActorEnvironment()])
    return {"primitives": composite}


@pytest.mark.timeout(30)
def test_code_act_prompt_shows_the_actor_primitive_is_awaited():
    """Calling ``primitives.actor.act`` yields a coroutine, not a handle, so
    every place the prompt shows the call shows it awaited."""
    actor = CodeActActor()
    prompt = _prompt(
        environments=_real_envs_mixed(),
    )

    assert "`await primitives.actor.act(...)` spawns a sub-actor" in prompt
    assert "(`handle = await primitives.actor.act(...)`)" in prompt
    assert "**`async def primitives.actor.act(request" in prompt


@pytest.mark.timeout(30)
def test_multiple_custom_environments_all_included():
    """Multiple custom environments should each have their prompt context included."""
    actor = CodeActActor()

    marker_a = "### Alpha Environment\nAlpha-specific guidance for the LLM."
    marker_b = "### Beta Environment\nBeta-specific guidance for the LLM."
    envs: Mapping[str, Any] = {
        "alpha": _DummyEnv(marker_a),
        "beta": _DummyEnv(marker_b),
    }

    prompt = _prompt(
        environments=envs,
    )

    assert marker_a in prompt
    assert marker_b in prompt


@pytest.mark.timeout(30)
def test_custom_environment_empty_prompt_context_excluded():
    """Custom environments returning empty prompt context should not inject noise."""
    actor = CodeActActor()

    envs: Mapping[str, Any] = {
        "empty_env": _DummyEnv(""),
        "whitespace_env": _DummyEnv("   \n  "),
    }

    prompt = _prompt(
        environments=envs,
    )

    # The prompt should still be valid (no crash) and not contain stray whitespace blocks.
    assert "empty_env" not in prompt
    assert "whitespace_env" not in prompt


# ────────────────────────────────────────────────────────────────────────────
# External app integration section
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# Guidelines composition (constructor baseline + per-invocation overlay)
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.timeout(30)
def test_guidelines_neither_specified():
    """No guidelines at all -> no ### Guidelines section in the prompt."""
    actor = CodeActActor()
    prompt = _prompt(
        environments={},
        guidelines=None,
    )
    assert "### Guidelines" not in prompt


@pytest.mark.timeout(30)
def test_guidelines_constructor_only():
    """Constructor-level guidelines appear in a single ### Guidelines section."""
    actor = CodeActActor(guidelines="Always respond in formal English.")
    base = actor._base_guidelines
    effective = "\n\n".join(filter(None, [base, None])) or None

    prompt = _prompt(
        environments={},
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Always respond in formal English." in prompt


@pytest.mark.timeout(30)
def test_guidelines_per_invocation_only():
    """Per-invocation guidelines appear in a single ### Guidelines section."""
    actor = CodeActActor()
    per_invocation = "Check every input field."
    effective = (
        "\n\n".join(filter(None, [actor._base_guidelines, per_invocation])) or None
    )

    prompt = _prompt(
        environments={},
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Check every input field." in prompt


@pytest.mark.timeout(30)
def test_guidelines_both_compose():
    """Constructor + per-invocation guidelines compose into one ### Guidelines section."""
    actor = CodeActActor(guidelines="Always respond in formal English.")
    per_invocation = "Check every input field."
    effective = (
        "\n\n".join(
            filter(None, [actor._base_guidelines, per_invocation]),
        )
        or None
    )

    prompt = _prompt(
        environments={},
        guidelines=effective,
    )
    assert prompt.count("### Guidelines") == 1
    assert "Always respond in formal English." in prompt
    assert "Check every input field." in prompt
    # Constructor guidelines come first
    idx_base = prompt.index("Always respond in formal English.")
    idx_overlay = prompt.index("Check every input field.")
    assert idx_base < idx_overlay
