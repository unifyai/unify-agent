"""
tests/conversation_manager/core/test_brain.py
==============================================

Unit tests for the BrainSpec data structure and build_brain_spec helper
in ``domains/brain.py``.

Covers:
- The plain-text state message shape
- User details resolved from the session's user identity
- Assistant identity (bio) flowing into the system prompt
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from unify.common.prompt_helpers import PromptParts
from unify.legacy.conversation_manager.domains import brain as brain_module
from unify.legacy.conversation_manager.domains.brain import BrainSpec, build_brain_spec
from unify.session_details import SESSION_DETAILS

# =============================================================================
# Helpers
# =============================================================================


def _make_brain_spec(state_prompt: str = "<state>test</state>") -> BrainSpec:
    """Create a minimal BrainSpec for testing."""
    parts = PromptParts()
    parts.add("You are a helpful assistant.")
    return BrainSpec(system_prompt=parts, state_prompt=state_prompt)


def _make_cm():
    """The smallest ConversationManager-like object build_brain_spec needs."""
    return SimpleNamespace()


def _make_snapshot():
    return SimpleNamespace(full_render="<state>ready</state>")


@pytest.fixture
def assistant_identity(monkeypatch):
    """Pin the assistant identity fields build_brain_spec reads."""
    monkeypatch.setattr(SESSION_DETAILS.assistant, "job_title", "")
    monkeypatch.setattr(SESSION_DETAILS.assistant, "about", "Operations assistant.")
    return SESSION_DETAILS.assistant


@pytest.fixture
def user_identity(monkeypatch):
    """Pin the user identity fields build_brain_spec reads."""
    monkeypatch.setattr(SESSION_DETAILS.user, "first_name", "Dana")
    monkeypatch.setattr(SESSION_DETAILS.user, "surname", "Owner")
    return SESSION_DETAILS.user


# =============================================================================
# Tests
# =============================================================================


class TestBrainSpecStateMessage:
    """Tests for BrainSpec.state_message()."""

    def test_state_message_is_plain_text(self):
        """The state message is a plain text user turn tagged as a snapshot."""
        spec = _make_brain_spec(state_prompt="<state>hello</state>")
        msg = spec.state_message()

        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)
        assert msg["content"] == "<state>hello</state>"
        assert msg["_cm_state_snapshot"] is True

    def test_snapshot_clock_reaches_state_message(self):
        """The snapshot's trailing ``Current time`` pane reaches the model.

        The clock rides at the tail of the rendered snapshot rather than in
        the system prompt, so the state message must carry it through
        verbatim.
        """
        from unify.common.prompt_helpers import now
        from unify.legacy.conversation_manager.domains.chat_history import ChatHistory
        from unify.legacy.conversation_manager.domains.notifications import (
            NotificationBar,
        )
        from unify.legacy.conversation_manager.domains.renderer import Renderer

        snapshot = Renderer().render_state(
            ChatHistory(),
            NotificationBar(),
            in_flight_actions={},
            last_snapshot=datetime(2026, 2, 13, 11, 0, 0, tzinfo=timezone.utc),
        )
        expected_tail = f"Current time: {now()}."
        assert snapshot.full_render.endswith(expected_tail)

        plain = _make_brain_spec(state_prompt=snapshot.full_render).state_message()
        assert plain["content"].endswith(expected_tail)


class TestBuildBrainSpec:
    """Tests for build_brain_spec prompt construction."""

    def test_user_details_come_from_the_session_user(
        self,
        monkeypatch,
        assistant_identity,
        user_identity,
    ):
        """The main brain prompt reads the user's name from SESSION_DETAILS.user."""
        captured_prompt_kwargs = {}

        def fake_build_system_prompt(**kwargs):
            captured_prompt_kwargs.update(kwargs)
            parts = PromptParts()
            parts.add("system")
            return parts

        monkeypatch.setattr(
            brain_module,
            "build_system_prompt",
            fake_build_system_prompt,
        )

        build_brain_spec(_make_cm(), _make_snapshot())

        assert captured_prompt_kwargs["first_name"] == "Dana"
        assert captured_prompt_kwargs["surname"] == "Owner"

    def test_state_prompt_is_the_rendered_snapshot(
        self,
        assistant_identity,
        user_identity,
    ):
        """The snapshot's full render becomes the state prompt verbatim."""
        spec = build_brain_spec(_make_cm(), _make_snapshot())

        assert spec.state_prompt == "<state>ready</state>"

    def test_bio_carries_job_title_and_about(
        self,
        monkeypatch,
        assistant_identity,
        user_identity,
    ):
        """The Bio section names the role and carries the about text."""
        monkeypatch.setattr(assistant_identity, "job_title", "Ops lead")

        spec = build_brain_spec(_make_cm(), _make_snapshot())
        prompt = spec.system_prompt.flatten()

        assert "Role / specialization: Ops lead." in prompt
        assert "Operations assistant." in prompt

    def test_user_details_rendered(self, assistant_identity, user_identity):
        """The user's name appears under User details."""
        spec = build_brain_spec(_make_cm(), _make_snapshot())
        prompt = spec.system_prompt.flatten()

        assert "- First Name: Dana" in prompt
        assert "- Surname: Owner" in prompt

    def test_unconfigured_user_falls_back_to_placeholder_names(
        self,
        monkeypatch,
        assistant_identity,
    ):
        """A session without a user profile still yields a valid prompt."""
        monkeypatch.setattr(SESSION_DETAILS.user, "first_name", "")
        monkeypatch.setattr(SESSION_DETAILS.user, "surname", "")

        spec = build_brain_spec(_make_cm(), _make_snapshot())
        prompt = spec.system_prompt.flatten()

        assert "- First Name: Default" in prompt
        assert "- Surname: User" in prompt
