"""Tests for CM eval assistant identity env overrides."""

from __future__ import annotations

import os

import pytest

from tests.legacy.conversation_manager.assistant_identity_env import (
    ensure_test_assistant_identity_env,
)

pytestmark = pytest.mark.no_unify_context


def test_ensure_blanks_ambient_assistant_id(monkeypatch):
    """An assistant id pinned in the machine's .env must not leak into CM
    test boots: populate_from_env skips blank values, so agent_id stays None
    and the boot binds the default local identity."""
    monkeypatch.setenv("ASSISTANT_ID", "524")

    ensure_test_assistant_identity_env()

    assert os.environ["ASSISTANT_ID"] == ""
