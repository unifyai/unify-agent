"""One record entry, and the @-mention grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

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
    # The run record's fields (UNIFY_AGENTS_RUN_RECORD=on, unify/agents/run_record.py): the author's role, where it
    # works (its episode and pass), the entry it answers, typed pointers, and how many tool calls or cells other than
    # messages it had made when it posted (the storm notice's evidence count). A session record leaves them unset, and
    # :meth:`to_dict` writes them only when set, so its lines are as before.
    role: Optional[str] = None
    episode: Optional[str] = None
    pass_id: Optional[str] = None
    reply_to: Optional[int] = None
    refs: tuple[str, ...] = ()
    acts: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seq": self.seq,
            "ts": self.ts,
            "author": self.author,
            "kind": self.kind,
            "text": self.text,
            "mentions": list(self.mentions),
        }
        if self.role is not None:
            out.update(
                role=self.role,
                episode=self.episode,
                pass_id=self.pass_id,
                reply_to=self.reply_to,
                refs=list(self.refs),
                acts=self.acts,
            )
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entry":
        """Readers ignore fields they do not know (later phases add some). The run record's fields are read only from
        a run record's row (it always carries ``role``); a session record's line keeps them unset."""
        run = data.get("role") is not None
        reply_to = data.get("reply_to") if run else None
        acts = data.get("acts") if run else None
        return cls(
            seq=int(data["seq"]),
            ts=str(data["ts"]),
            author=str(data["author"]),
            kind=str(data["kind"]),
            text=str(data["text"]),
            mentions=tuple(str(m) for m in data.get("mentions") or ()),
            role=str(data["role"]) if run else None,
            episode=str(data["episode"]) if run and data.get("episode") is not None else None,
            pass_id=str(data["pass_id"]) if run and data.get("pass_id") is not None else None,
            reply_to=int(reply_to) if isinstance(reply_to, int) and not isinstance(reply_to, bool) else None,
            refs=tuple(str(r) for r in data.get("refs") or ()) if run else (),
            acts=int(acts) if isinstance(acts, int) and not isinstance(acts, bool) else None,
        )
