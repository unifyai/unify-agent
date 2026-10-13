"""
tests/conversation_manager/core/test_unify_attachments.py
=========================================================

Unit tests for chat message attachments: workspace paths carried on the
chat events and on the stored conversation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from unify.legacy.conversation_manager.domains.chat_history import ChatHistory
from unify.legacy.conversation_manager.events import (
    Event,
    UnifyMessageReceived,
    UnifyMessageSent,
)

REPORT_ATTACHMENT = "Attachments/att-uuid-1_report.pdf"
DATA_ATTACHMENT = "Attachments/att-uuid-2_data.xlsx"


class TestUnifyMessageReceivedAttachments:
    """Tests for UnifyMessageReceived event attachment handling."""

    def test_event_carries_attachment_paths(self):
        event = UnifyMessageReceived(
            content="Here's the document",
            attachments=[REPORT_ATTACHMENT, DATA_ATTACHMENT],
        )

        assert event.attachments == [REPORT_ATTACHMENT, DATA_ATTACHMENT]
        assert event.content == "Here's the document"

    def test_event_with_empty_attachments(self):
        event = UnifyMessageReceived(content="Just a message")

        assert event.attachments == []

    def test_attachments_survive_json_round_trip(self):
        event = UnifyMessageReceived(
            content="Here's the document",
            attachments=[REPORT_ATTACHMENT],
        )

        restored = Event.from_json(event.to_json())

        assert isinstance(restored, UnifyMessageReceived)
        assert restored.attachments == [REPORT_ATTACHMENT]


class TestUnifyMessageSentAttachments:
    """Tests for UnifyMessageSent event attachment handling."""

    def test_event_carries_attachment_paths(self):
        event = UnifyMessageSent(
            content="Sending you this file",
            attachments=[DATA_ATTACHMENT],
        )

        assert event.attachments == [DATA_ATTACHMENT]

    def test_sent_event_defaults_to_no_attachments(self):
        event = UnifyMessageSent(content="Here's the export")

        assert event.attachments == []


class TestChatHistoryAttachments:
    """The conversation keeps each message's attachment paths."""

    def test_message_keeps_attachment_paths(self):
        history = ChatHistory()
        message = history.append(
            role="user",
            content="Here's a file",
            attachments=[REPORT_ATTACHMENT],
            timestamp=datetime(2025, 6, 13, 12, 0, tzinfo=timezone.utc),
        )

        assert message.attachments == [REPORT_ATTACHMENT]

    def test_message_attachments_default_to_empty(self):
        history = ChatHistory()
        message = history.append(
            role="assistant",
            content="No file this time",
            timestamp=datetime(2025, 6, 13, 12, 0, tzinfo=timezone.utc),
        )

        assert message.attachments == []
