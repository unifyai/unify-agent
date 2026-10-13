"""
tests/conversation_manager/test_context_summarization.py
=============================================================

Tests for the brain's own LLM message list in ConversationManager.

This covers:
1. The _preprocess_messages() state snapshot deduplication logic
2. Brain message growth across LLM runs (the brain's model is scripted:
   ``tests/scripted_model.py``)
"""

from __future__ import annotations

import pytest

from tests.legacy.conversation_manager.scripted_brain import (
    WAIT,
    brain_requests,
    brain_turn,
    rendered_state,
    say,
)
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.legacy.conversation_manager.events import UnifyMessageReceived

# =============================================================================
# Test _preprocess_messages() - State Snapshot Deduplication
# =============================================================================


class TestPreprocessMessages:
    """Tests for the _preprocess_messages method that deduplicates state snapshots."""

    def test_returns_string_unchanged(self, initialized_cm):
        """String messages are returned unchanged."""
        result = initialized_cm.cm._preprocess_messages("simple string")
        assert result == "simple string"

    def test_returns_dict_unchanged(self, initialized_cm):
        """Dict messages are returned unchanged."""
        msg = {"role": "user", "content": "hello"}
        result = initialized_cm.cm._preprocess_messages(msg)
        assert result == msg

    def test_returns_non_list_unchanged(self, initialized_cm):
        """Non-list, non-string, non-dict inputs are returned unchanged."""
        result = initialized_cm.cm._preprocess_messages(42)
        assert result == 42

    def test_keeps_only_latest_state_snapshot(self, initialized_cm):
        """When multiple state snapshots exist, only the latest is kept."""
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "State v1", "_cm_state_snapshot": True},
            {"role": "assistant", "content": "response 1"},
            {"role": "user", "content": "State v2", "_cm_state_snapshot": True},
            {"role": "assistant", "content": "response 2"},
            {"role": "user", "content": "State v3", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep system message and latest state snapshot
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "State v3"
        assert result[1].get("_cm_state_snapshot") is True

    def test_preserves_system_messages(self, initialized_cm):
        """System messages are always preserved."""
        messages = [
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "user", "content": "State", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Both system messages should be kept
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 2

    def test_preserves_user_interjections(self, initialized_cm):
        """User messages that are NOT state snapshots are preserved."""
        messages = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "State v1", "_cm_state_snapshot": True},
            {"role": "user", "content": "User question"},  # Not a snapshot
            {"role": "user", "content": "State v2", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep: system, user question, latest state snapshot
        assert len(result) == 3
        contents = [m["content"] for m in result]
        assert "You are an assistant" in contents
        assert "User question" in contents
        assert "State v2" in contents

    def test_no_state_snapshots_returns_messages_unchanged(self, initialized_cm):
        """If there are no state snapshots, messages are returned as-is."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Response"},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        assert result == messages

    def test_handles_empty_list(self, initialized_cm):
        """Empty list is returned unchanged."""
        result = initialized_cm.cm._preprocess_messages([])
        assert result == []

    def test_handles_malformed_messages_gracefully(self, initialized_cm):
        """Non-dict items in the list are handled gracefully."""
        messages = [
            {"role": "system", "content": "System"},
            "not a dict",  # Should be skipped
            {"role": "user", "content": "State", "_cm_state_snapshot": True},
        ]

        result = initialized_cm.cm._preprocess_messages(messages)

        # Should keep system and state snapshot, skip the string
        assert len(result) == 2


# =============================================================================
# Test Brain Message Growth During LLM Runs
# =============================================================================


class TestBrainMessagesGrowth:
    """Tests for brain message growth during LLM runs."""

    @pytest.mark.asyncio
    async def test_brain_messages_grow_by_two_per_llm_run(self, initialized_cm):
        """Each LLM run adds 2 messages: input + assistant response."""
        initial_len = len(initialized_cm.cm.brain_messages)

        model = ScriptedModel(actor=[brain_turn(say("Got your test message."), WAIT)])
        event = UnifyMessageReceived(
            content="Test message",
        )
        with scripted(model):
            await initialized_cm.step_until_wait(event, max_steps=1)

        # Brain messages should have grown by 2 (input_message + assistant_content)
        new_len = len(initialized_cm.cm.brain_messages)
        assert new_len == initial_len + 2
        assert len(brain_requests(model)) == 1
        model.assert_used_up()

    @pytest.mark.asyncio
    async def test_brain_messages_format(self, initialized_cm):
        """Brain messages have correct format: the state the model was sent,
        then the text of its reply."""
        thought = "The user sent a test message; acknowledge it."
        model = ScriptedModel(
            actor=[
                reply(thought, calls=[say("Hi!"), WAIT]),
            ],
        )
        event = UnifyMessageReceived(
            content="Test message",
        )
        with scripted(model):
            await initialized_cm.step_until_wait(event, max_steps=1)

        # Should have at least 2 messages
        assert len(initialized_cm.cm.brain_messages) >= 2

        # Check structure of messages
        for msg in initialized_cm.cm.brain_messages:
            assert isinstance(msg, dict)
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant", "system"]

        state, answer = initialized_cm.cm.brain_messages[-2:]
        (request,) = brain_requests(model)
        assert state["role"] == "user"
        assert state["content"] == rendered_state(request)
        assert "Test message" in state["content"]
        assert answer == {"role": "assistant", "content": thought}
        model.assert_used_up()
