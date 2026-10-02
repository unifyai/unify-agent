"""Hold back a persistent session's reply that repeats one already answered.

With ``UNIFY_REPEAT_GUARD`` on, a persistent tool loop keeps the final reply
of each turn and the requester message that followed it. When a later turn
ends in a reply identical (after normalisation) to one the requester has
already answered, the reply is not surfaced yet: the loop appends one note
quoting that answer and gives the model another step. Sending the same reply
again surfaces it; the guard holds a given reply back at most once per
session. Replies that differ are surfaced as they always were.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# How much of the requester's answer the note quotes.
QUOTE_CHARS = 300

_WS = re.compile(r"\s+")


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_REPEAT_GUARD", False))


def normalise(reply: object) -> str:
    """The reply as compared: whitespace collapsed, JSON re-dumped with sorted keys."""
    text = "" if reply is None else str(reply)
    text = text.strip()
    if text[:1] in ("{", "["):
        try:
            return json.dumps(
                json.loads(text),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        except ValueError:
            pass
    return _WS.sub(" ", text)


@dataclass
class _Turn:
    number: int
    reply: str
    followed_by: Optional[str] = None


class RepeatGuard:
    """The final replies of one persistent session and what followed each."""

    def __init__(self) -> None:
        self._turns: list[_Turn] = []
        self._held: set[str] = set()

    def check(self, reply: object) -> Optional[str]:
        """The note to append instead of surfacing *reply*, or ``None``.

        A reply is held back only when it matches a reply the requester has
        answered and this exact reply has not been held back before.
        """
        norm = normalise(reply)
        if not norm or norm in self._held:
            return None
        for turn in self._turns:
            if turn.reply == norm and turn.followed_by is not None:
                self._held.add(norm)
                quote = turn.followed_by.strip()[:QUOTE_CHARS]
                return (
                    f"Your reply is identical to your reply in turn {turn.number}, "
                    f"after which the requester said: '{quote}'. If you still "
                    "intend it, send it again; otherwise revise."
                )
        return None

    def surfaced(self, reply: object) -> None:
        """Record a reply that ended a turn and was surfaced."""
        self._turns.append(_Turn(number=len(self._turns) + 1, reply=normalise(reply)))

    def requester_said(self, message: str) -> None:
        """Record a requester message: it answers the latest unanswered reply."""
        if not message or not self._turns:
            return
        last = self._turns[-1]
        if last.followed_by is None:
            last.followed_by = message
