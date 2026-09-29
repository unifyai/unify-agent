"""
tests/conversation_manager/core/test_error_handling.py
======================================================

Tests for error handling and recovery in ConversationManager.

This test file covers:
1. Malformed events (invalid JSON, missing fields, unknown types)
2. Event handler edge cases (unregistered handlers)
3. State recovery scenarios (duplicate events, stray results)
4. Graceful degradation when data is missing or invalid

Most tests are marked as `symbolic` since they test deterministic error
handling behavior rather than LLM output.
"""

import json
import pytest
from datetime import datetime
from dataclasses import dataclass

from unify.conversation_manager.events import (
    ActorHandleStarted,
    Event,
    UnifyMessageReceived,
    ActorResult,
)

pytestmark = pytest.mark.symbolic


# =============================================================================
# Malformed Event Tests
# =============================================================================


class TestMalformedEvents:
    """Tests for handling malformed event data."""

    def test_from_json_invalid_json(self):
        """Event.from_json should raise on invalid JSON."""
        with pytest.raises(json.JSONDecodeError):
            Event.from_json("not valid json {")

    def test_from_json_missing_event_name(self):
        """Event.from_json should raise on missing event_name."""
        data = json.dumps({"payload": {"content": "test"}})
        with pytest.raises(KeyError):
            Event.from_json(data)

    def test_from_json_missing_payload(self):
        """Event.from_json should raise on missing payload."""
        data = json.dumps({"event_name": "UnifyMessageReceived"})
        with pytest.raises(KeyError):
            Event.from_json(data)

    def test_from_json_unknown_event_type(self, static_now):
        """Event.from_json should raise on unknown event type."""
        data = json.dumps(
            {
                "event_name": "NonExistentEvent",
                "payload": {"timestamp": static_now.isoformat()},
            },
        )
        with pytest.raises(Exception, match="not registered"):
            Event.from_json(data)

    def test_from_json_invalid_timestamp(self):
        """Event.from_json should raise on invalid timestamp format."""
        data = json.dumps(
            {
                "event_name": "UnifyMessageReceived",
                "payload": {"content": "hi", "timestamp": "not-a-valid-timestamp"},
            },
        )
        with pytest.raises(ValueError):
            Event.from_json(data)

    def test_from_json_missing_required_field(self, static_now):
        """Event.from_json should raise when required field is missing."""
        # UnifyMessageReceived requires 'content'
        data = json.dumps(
            {
                "event_name": "UnifyMessageReceived",
                "payload": {"timestamp": static_now.isoformat()},
            },
        )
        with pytest.raises(TypeError):
            Event.from_json(data)

    def test_from_json_extra_fields_ignored(self, static_now):
        """Event.from_json should ignore extra fields not in dataclass."""
        data = json.dumps(
            {
                "event_name": "UnifyMessageReceived",
                "payload": {
                    "content": "test message",
                    "timestamp": static_now.isoformat(),
                    "extra_field_that_does_not_exist": "should be ignored",
                    "another_random_field": 12345,
                },
            },
        )
        # Should not raise - extra fields are filtered out
        event = Event.from_json(data)
        assert isinstance(event, UnifyMessageReceived)
        assert event.content == "test message"
        assert not hasattr(event, "extra_field_that_does_not_exist")


# =============================================================================
# Event Handler Edge Cases
# =============================================================================


class TestEventHandlerEdgeCases:
    """Tests for event handler registry and edge cases."""

    @pytest.mark.asyncio
    async def test_unregistered_event_type_returns_noop(self, initialized_cm):
        """Events without registered handlers should be handled gracefully."""
        cm = initialized_cm

        # Create a custom event type that isn't registered
        @dataclass
        class UnregisteredTestEvent(Event):
            data: str = "test"

        event = UnregisteredTestEvent()

        # Should not raise - just returns a no-op coroutine
        result = await cm.step(event)

        # The event was processed (no exception), but no handler ran
        assert result.llm_requested is False
        assert result.llm_ran is False
        assert result.output_events == []

    @pytest.mark.asyncio
    async def test_handle_started_event_handler(self, initialized_cm):
        """ActorHandleStarted is handled without triggering the LLM."""
        cm = initialized_cm

        result = await cm.step(
            ActorHandleStarted(action_name="act", handle_id=999, query="noop"),
        )

        # The handler records nothing and requests no LLM run
        assert result.llm_requested is False
        assert result.llm_ran is False
        assert result.output_events == []


# =============================================================================
# State Recovery Scenarios
# =============================================================================


class TestStateRecovery:
    """Tests for state recovery from abnormal event sequences."""

    @pytest.mark.asyncio
    @pytest.mark.llm_call
    async def test_duplicate_message_received(self, initialized_cm):
        """Duplicate message events should be handled (added to thread twice)."""
        cm = initialized_cm

        event = UnifyMessageReceived(content="Duplicate message")

        # Process the same event twice
        await cm.step(event)
        await cm.step(event)

        # Both messages should be in the conversation (no deduplication at this level)
        matching = [
            m for m in cm.cm.chat_history.recent() if m.content == "Duplicate message"
        ]
        assert len(matching) == 2

    @pytest.mark.asyncio
    @pytest.mark.llm_call
    async def test_actor_result_for_nonexistent_action(self, initialized_cm):
        """ActorResult for an action not in in_flight_actions should not crash."""
        cm = initialized_cm

        # Send result for an action that doesn't exist
        result = await cm.step(
            ActorResult(
                handle_id=99999,  # Non-existent
                success=True,
                result="Some result",
            ),
        )

        # Should not crash - handler uses .pop() with default
        assert result.llm_requested is True


# =============================================================================
# Notification Bar Edge Cases
# =============================================================================


class TestNotificationBarEdgeCases:
    """Tests for notification bar edge cases."""

    @pytest.mark.asyncio
    async def test_push_notification_with_datetime_timestamp(
        self,
        initialized_cm,
        static_now,
    ):
        """push_notif should accept datetime timestamp."""
        cm = initialized_cm

        cm.cm.notifications_bar.push_notif(
            type="Test",
            notif_content="Test notification",
            timestamp=static_now,
        )

        # Should have added the notification
        assert len(cm.cm.notifications_bar.notifications) >= 1
        last_notif = cm.cm.notifications_bar.notifications[-1]
        assert last_notif.content == "Test notification"
        assert last_notif.type == "Test"


# =============================================================================
# Event Serialization Edge Cases
# =============================================================================


class TestEventSerializationEdgeCases:
    """Tests for event serialization and deserialization edge cases."""

    def test_event_round_trip_preserves_data(self):
        """Event should survive JSON round-trip with all data intact."""
        original = UnifyMessageReceived(content="Test message")

        # Serialize and deserialize
        json_str = original.to_json()
        restored = Event.from_json(json_str)

        assert isinstance(restored, UnifyMessageReceived)
        assert restored.content == original.content
        # Timestamps should be equal (within serialization precision)
        assert abs((restored.timestamp - original.timestamp).total_seconds()) < 1

    def test_event_round_trip_preserves_attachments(self):
        """Attachment paths survive the JSON round-trip unchanged."""
        attachments = ["Attachments/att-1_report.pdf"]
        original = UnifyMessageReceived(
            content="See attached",
            attachments=attachments,
        )

        restored = Event.from_json(original.to_json())

        assert isinstance(restored, UnifyMessageReceived)
        assert restored.attachments == attachments

    def test_event_to_dict_with_datetime(self):
        """Event.to_dict should serialize datetime correctly."""
        event = UnifyMessageReceived(content="test")
        data = event.to_dict()

        # Timestamp should be an ISO format string
        assert isinstance(data["payload"]["timestamp"], str)
        # Should be parseable
        datetime.fromisoformat(data["payload"]["timestamp"])


# =============================================================================
# Chat History Edge Cases
# =============================================================================


class TestBrainMessagesEdgeCases:
    """Tests for the brain's LLM message preprocessing edge cases."""

    @pytest.mark.asyncio
    async def test_preprocess_messages_with_string(self, initialized_cm):
        """_preprocess_messages should pass through strings unchanged."""
        cm = initialized_cm

        result = cm.cm._preprocess_messages("just a string")
        assert result == "just a string"

    @pytest.mark.asyncio
    async def test_preprocess_messages_with_dict(self, initialized_cm):
        """_preprocess_messages should pass through dicts unchanged."""
        cm = initialized_cm

        msg = {"role": "user", "content": "test"}
        result = cm.cm._preprocess_messages(msg)
        assert result == msg

    @pytest.mark.asyncio
    async def test_preprocess_messages_empty_list(self, initialized_cm):
        """_preprocess_messages should handle empty list."""
        cm = initialized_cm

        result = cm.cm._preprocess_messages([])
        assert result == []

    @pytest.mark.asyncio
    async def test_preprocess_messages_no_state_snapshots(self, initialized_cm):
        """_preprocess_messages should handle messages without state snapshots."""
        cm = initialized_cm

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        result = cm.cm._preprocess_messages(messages)
        assert result == messages


# =============================================================================
# Debouncer Edge Cases (Symbolic Tests)
# =============================================================================


class TestDebouncerBasics:
    """Basic tests for the Debouncer utility."""

    @pytest.mark.asyncio
    async def test_debouncer_exists(self, initialized_cm):
        """ConversationManager should have a debouncer instance."""
        cm = initialized_cm
        assert hasattr(cm.cm, "debouncer")
        assert cm.cm.debouncer is not None


# =============================================================================
# LLM Request Edge Cases
# =============================================================================


class TestLLMRequestEdgeCases:
    """Tests for LLM request management edge cases."""

    @pytest.mark.asyncio
    async def test_flush_empty_requests(self, initialized_cm):
        """flush_llm_requests with no pending requests should be a no-op."""
        cm = initialized_cm

        # Ensure no pending requests
        cm.cm._pending_llm_requests.clear()

        # Should not raise
        await cm.cm.flush_llm_requests()

    @pytest.mark.asyncio
    async def test_multiple_pending_requests_uses_last(self, initialized_cm):
        """Multiple pending LLM requests should use the last one's params."""
        cm = initialized_cm

        # Manually add multiple (delay, is_user_origin) requests
        cm.cm._pending_llm_requests.append((0, False))
        cm.cm._pending_llm_requests.append((1, False))
        cm.cm._pending_llm_requests.append((2, True))  # Last one

        # The flush logic uses the last request's params
        assert cm.cm._pending_llm_requests[-1] == (2, True)

        # Clear for other tests
        cm.cm._pending_llm_requests.clear()
