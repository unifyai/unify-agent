"""
ChatHistory: the one conversation between the user and the assistant.

Messages live in an in-memory list and are mirrored to the ``messages``
table, so the conversation survives a restart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime

from unify import db
from unify.common.prompt_helpers import now as prompt_now


@dataclass
class ChatMessage:
    """One chat message, from the user or the assistant.

    ``attachments`` are workspace paths of files that travelled with the
    message. ``row_id`` is the store row backing the message, ``None`` while
    the message is only in memory.
    """

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    attachments: list[str] = field(default_factory=list)
    row_id: int | None = None


class ChatHistory:
    """The chat as an ordered list of messages, persisted one row per message.

    Until ``bind`` runs (during manager initialisation), appended messages
    stay in memory; ``bind`` writes them through and ``load`` prepends
    whatever earlier sessions left in the table.
    """

    DEFAULT_MAX_MESSAGES = 100

    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES):
        self.max_messages = max_messages
        self.messages: list[ChatMessage] = []
        self._bound = False
        # ``append`` runs on the event loop while ``load`` may splice from a
        # worker thread; both mutate ``messages`` under this lock.
        self._lock = threading.Lock()

    @property
    def is_bound(self) -> bool:
        return self._bound

    def bind(self) -> None:
        """Start writing through, including any messages held in memory."""
        self._bound = True
        with self._lock:
            pending = [m for m in self.messages if m.row_id is None]
        for message in pending:
            self._persist(message)

    def load(self) -> int:
        """Prepend the messages earlier sessions stored; return how many."""
        rows = db.query(
            "SELECT id, role, content, timestamp, attachments FROM messages"
            " ORDER BY timestamp DESC, id DESC LIMIT ?",
            (self.max_messages,),
        )
        with self._lock:
            known = {m.row_id for m in self.messages if m.row_id is not None}
            restored = [
                self._from_row(row) for row in reversed(rows) if row["id"] not in known
            ]
            self.messages = (restored + self.messages)[-self.max_messages :]
        return len(restored)

    def append(
        self,
        *,
        role: str,
        content: str,
        attachments: list[str] | None = None,
        timestamp: datetime | None = None,
    ) -> ChatMessage:
        """Record a message, writing it to the store when bound."""
        message = ChatMessage(
            role=role,
            content=content or "",
            timestamp=timestamp or prompt_now(as_string=False),
            attachments=list(attachments or []),
        )
        with self._lock:
            self.messages.append(message)
            del self.messages[: -self.max_messages]
        if self._bound:
            self._persist(message)
        return message

    def recent(self, max_messages: int | None = None) -> list[ChatMessage]:
        with self._lock:
            messages = list(self.messages)
        if max_messages is None:
            return messages
        return messages[-max_messages:]

    def clear(self) -> None:
        """Forget the in-memory messages (the store is untouched)."""
        with self._lock:
            self.messages.clear()

    def _persist(self, message: ChatMessage) -> None:
        cursor = db.execute(
            "INSERT INTO messages (role, content, timestamp, attachments)"
            " VALUES (?, ?, ?, ?)",
            (
                message.role,
                message.content,
                message.timestamp.isoformat(),
                db.dumps(list(message.attachments)),
            ),
        )
        message.row_id = int(cursor.lastrowid)

    @staticmethod
    def _from_row(row: dict) -> ChatMessage:
        return ChatMessage(
            role=row["role"],
            content=row["content"] or "",
            timestamp=datetime.fromisoformat(row["timestamp"]),
            attachments=list(db.loads(row["attachments"]) or []),
            row_id=row["id"],
        )
