"""Tests for shared LLM client factories."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from unify.common.llm_client import new_llm_client, resolve_default_model
from unify.settings import SETTINGS


def test_resolve_default_model_reads_the_settings() -> None:
    """The run's model and effort are UNIFY_MODEL and UNIFY_REASONING_EFFORT."""
    assert resolve_default_model() == (
        SETTINGS.UNIFY_MODEL,
        SETTINGS.UNIFY_REASONING_EFFORT or None,
    )


def test_new_llm_client_pins_the_run_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run effort overrides the call site's effort."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MODEL", "openai/gpt-5.5@openrouter")
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "low")
    with patch("unify.common.llm_client.unillm.AsyncUnify") as mock_async:
        new_llm_client(reasoning_effort="high")
        args, kwargs = mock_async.call_args
        assert args == ("openai/gpt-5.5@openrouter",)
        assert kwargs["reasoning_effort"] == "low"


def test_new_llm_client_without_effort_keeps_call_site_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no run effort, each call site's effort stays."""
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "")
    with patch("unify.common.llm_client.unillm.AsyncUnify") as mock_async:
        new_llm_client(reasoning_effort="high")
        args, kwargs = mock_async.call_args
        assert args == (SETTINGS.UNIFY_MODEL,)
        assert kwargs["reasoning_effort"] == "high"


def test_new_llm_client_explicit_model_bypasses_the_run_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Call sites pinning a model keep it and their own effort."""
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "low")
    with patch("unify.common.llm_client.unillm.AsyncUnify") as mock_async:
        new_llm_client("claude-4.8-opus@anthropic", reasoning_effort="high")
        args, kwargs = mock_async.call_args
        assert args == ("claude-4.8-opus@anthropic",)
        assert kwargs["reasoning_effort"] == "high"
