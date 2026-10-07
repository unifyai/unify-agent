"""
tests/conversation_manager/core/test_chat_history.py
====================================================

Symbolic tests for ``ChatHistory``: the conversation lives in memory and is
mirrored, one row per message, to the ``messages`` table so it survives a
restart.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from unify import db
from unify.legacy.conversation_manager.domains.chat_history import (
    ChatHistory,
    ChatMessage,
)

_BASE = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)


def _at(minutes: int) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _stored() -> list[dict]:
    rows = db.query(
        "SELECT id, role, content, timestamp, attachments FROM messages ORDER BY id",
    )
    for row in rows:
        row["attachments"] = db.loads(row["attachments"])
    return rows


class TestInMemory:
    """Before ``bind`` the history is an ordinary in-memory list."""

    def test_append_records_message_fields(self):
        history = ChatHistory()
        message = history.append(
            role="user",
            content="Hello",
            attachments=["Attachments/a.pdf"],
            timestamp=_at(0),
        )

        assert isinstance(message, ChatMessage)
        assert history.recent() == [message]
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.attachments == ["Attachments/a.pdf"]
        assert message.timestamp == _at(0)
        assert message.row_id is None
        assert history.is_bound is False

    def test_recent_returns_the_tail(self):
        history = ChatHistory()
        for i in range(5):
            history.append(role="user", content=f"m{i}", timestamp=_at(i))

        assert [m.content for m in history.recent(2)] == ["m3", "m4"]
        assert [m.content for m in history.recent()] == [f"m{i}" for i in range(5)]

    def test_max_messages_drops_the_oldest(self):
        history = ChatHistory(max_messages=3)
        for i in range(5):
            history.append(role="user", content=f"m{i}", timestamp=_at(i))

        assert [m.content for m in history.recent()] == ["m2", "m3", "m4"]

    def test_clear_forgets_messages(self):
        history = ChatHistory()
        history.append(role="user", content="Hello", timestamp=_at(0))
        history.clear()
        assert history.recent() == []


class TestPersistence:
    """Bound histories write through to the store and load from it."""

    def test_bind_writes_through_messages_held_in_memory(self):
        history = ChatHistory()
        held = history.append(role="user", content="before bind", timestamp=_at(0))

        history.bind()

        assert history.is_bound
        assert held.row_id is not None
        rows = _stored()
        assert [row["content"] for row in rows] == ["before bind"]
        assert rows[0]["id"] == held.row_id

    def test_append_after_bind_persists_immediately(self):
        history = ChatHistory()
        history.bind()

        history.append(
            role="assistant",
            content="reply",
            attachments=["Outputs/chart.png"],
            timestamp=_at(1),
        )

        (row,) = _stored()
        assert row["role"] == "assistant"
        assert row["content"] == "reply"
        assert row["attachments"] == ["Outputs/chart.png"]
        assert datetime.fromisoformat(row["timestamp"]) == _at(1)

    def test_load_restores_a_previous_session_in_order(self):
        earlier = ChatHistory()
        earlier.bind()
        earlier.append(role="user", content="first", timestamp=_at(0))
        earlier.append(role="assistant", content="second", timestamp=_at(1))
        earlier.append(role="user", content="third", timestamp=_at(2))

        later = ChatHistory()
        later.bind()
        restored = later.load()

        assert restored == 3
        assert [m.content for m in later.recent()] == ["first", "second", "third"]
        assert [m.role for m in later.recent()] == ["user", "assistant", "user"]
        assert all(m.row_id is not None for m in later.recent())

    def test_load_prepends_before_messages_that_arrived_mid_boot(self):
        earlier = ChatHistory()
        earlier.bind()
        earlier.append(role="user", content="last week", timestamp=_at(-10000))

        later = ChatHistory()
        later.append(role="user", content="just now", timestamp=_at(0))
        later.bind()
        restored = later.load()

        assert restored == 1
        assert [m.content for m in later.recent()] == ["last week", "just now"]

    def test_load_skips_rows_this_history_already_wrote(self):
        history = ChatHistory()
        history.bind()
        history.append(role="user", content="mine", timestamp=_at(0))

        assert history.load() == 0
        assert [m.content for m in history.recent()] == ["mine"]

    def test_load_keeps_only_the_most_recent_messages(self):
        earlier = ChatHistory()
        earlier.bind()
        for i in range(6):
            earlier.append(role="user", content=f"m{i}", timestamp=_at(i))

        later = ChatHistory(max_messages=4)
        later.bind()
        restored = later.load()

        assert restored == 4
        assert [m.content for m in later.recent()] == ["m2", "m3", "m4", "m5"]
