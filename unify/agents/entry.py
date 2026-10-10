"""One record entry, and the @-mention grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

KINDS = ("post", "reply", "cancel", "system")
MAX_TEXT_BYTES = 16 * 1024
RESERVED_NAMES = ("user", "harness", "all", "root")
NAME_PATTERN = r"[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?"
_NAME_RE = re.compile(rf"^{NAME_PATTERN}$")
# An @ that does not follow a word character, another @ or a dot (so e-mail
# addresses and @@ are not mentions), then the longest name.
_MENTION_RE = re.compile(rf"(?<![\w@.])@({NAME_PATTERN})", re.IGNORECASE)


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def mention_tokens(text: str) -> list[str]:
    """The names @-mentioned in ``text``: lower-case, unique, in first-seen order."""
    seen: list[str] = []
    for match in _MENTION_RE.finditer(text or ""):
        token = match.group(1).lower()
        if token not in seen:
            seen.append(token)
    return seen


@dataclass(frozen=True)
class Entry:
    seq: int
    ts: str
    author: str
    kind: str
    text: str
    mentions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "author": self.author,
            "kind": self.kind,
            "text": self.text,
            "mentions": list(self.mentions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entry":
        """Readers ignore fields they do not know (later phases add some)."""
        return cls(
            seq=int(data["seq"]),
            ts=str(data["ts"]),
            author=str(data["author"]),
            kind=str(data["kind"]),
            text=str(data["text"]),
            mentions=tuple(str(m) for m in data.get("mentions") or ()),
        )
