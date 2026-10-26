"""The standing-note digest: the library's preferences, rules and cautions, at the start of every request (rev 2.4).

Notes that state how work is to be done (a preference someone stated, a rule, a caution an episode proved) hold
whatever the request is, so they are shown to the actor at every request start, with no fit judgement: the librarian
still judges the fit of procedures and functions, never of these. Each entry names the note's id, its kind and status,
its title and description, what it applies to (``applies_to``) and who stated it (``stated_by``) when its front matter
says so.

**Which notes.** A note whose front matter's ``kind`` is ``preference``, ``rule`` or ``caution``
(:data:`STANDING_KINDS`) and no other: a note with no ``kind`` (one written before the field existed) and a ``lesson``
(the writer's record of what was tried) are not shown, and neither are retired and withdrawn notes. Every note stays
reachable through ``memory.items`` and ``memory.get``, as the digest's last line says.

**Order.** By status (``adopted`` first, then ``on-trial``), then trust (the note's positive uses less its negative
ones, :mod:`.serving`'s counts), then recency (the newest change first, the export's ``changed_order``), then id.

**A display limit, never a storage one.** At most :data:`DIGEST_ENTRIES` entries of at most :data:`ENTRY_CHARS`
characters each (the id is never cut). Nothing is dropped from the library: the digest's last line says how many it
shows of how many there are and names the calls that reach every note (``memory.items(kind='note')``,
``memory.get(id)``).

**Sent when it changes.** :class:`StandingDigest` gives the digest the first time and then only when it differs from
the one it last gave, so a persistent session's later requests repeat nothing (the prompt cache keeps its prefix).

Read from the actor's library copy only (``.memory/items.json``, the notes' own files and
``.memory/item_evidence.json``); the front matter is read leniently here (a field absent or unreadable is left out),
the gate's parser being the strict one. Nothing here reads a world, a word, an identifier or a request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

#: The note kinds the digest shows, and the only ones.
STANDING_KINDS = ("preference", "rule", "caution")
#: The display limit: entries shown, and characters of each entry after its id.
DIGEST_ENTRIES = 8
ENTRY_CHARS = 200
#: The digest's first line.
DIGEST_HEAD = (
    "Standing notes in your memory library (preferences, rules and cautions; shown at every request, without a fit "
    "judgement, so check what each applies to):"
)
#: The digest's last line: every note stays reachable.
DIGEST_REACH = "Every note: memory.items(kind='note'); open one with memory.get(id)."
#: The front-matter fields the digest reads beside the title and description.
STATED_BY = "stated_by"
APPLIES_TO = "applies_to"
KIND = "kind"
_STATUS_RANK = {"adopted": 0, "on-trial": 1}
_ITEMS_FILE = ".memory/items.json"
_EVIDENCE_FILE = ".memory/item_evidence.json"
_WITHDRAWN_FILE = ".memory/withdrawn_items.json"
_HIDDEN = ("retired",)
#: A note's id as the library layout names it (:data:`.layout.NOTE_ID`): only such a path is read from the copy.
_NOTE_ID = re.compile(r"notes/[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_-]*\.md\Z")


def front_fields(text: Any) -> dict[str, str]:
    """A note's front matter as ``key -> value`` (quotes stripped), read leniently: every ``key: value`` line between
    the opening and closing ``---`` lines; {} when there is none. Never raises."""
    if not isinstance(text, str):
        return {}
    lines = text.split("\n")
    if not lines or lines[0].rstrip("\r") != "---":
        return {}
    out: dict[str, str] = {}
    for raw in lines[1:]:
        line = raw.rstrip("\r")
        if line == "---":
            return out
        key, sep, value = line.partition(":")
        key = key.strip().lower()
        if not sep or not key or key in out:
            continue
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        out[key] = " ".join(v.split())
    return {}  # no closing line: not front matter


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _cut(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def standing_notes(copy: Path | str) -> list[dict]:
    """The standing notes of the library copy at *copy*, in digest order (module docstring): each ``{"id", "kind",
    "status", "title", "description", "applies_to", "stated_by", "trust", "changed_order"}``. [] when the copy's
    ``items.json`` cannot be read."""
    copy = Path(copy)
    data = _json(copy / _ITEMS_FILE)
    rows = data.get("items") if isinstance(data, dict) else None
    if not isinstance(rows, dict):
        return []
    evidence = _json(copy / _EVIDENCE_FILE)
    evidence = evidence.get("items") if isinstance(evidence, dict) else None
    evidence = evidence if isinstance(evidence, dict) else {}
    gone = _json(copy / _WITHDRAWN_FILE)
    gone = gone.get("items") if isinstance(gone, dict) else None
    gone = set(gone) if isinstance(gone, dict) else set()
    out: list[dict] = []
    for iid, row in rows.items():
        if not (
            isinstance(iid, str)
            and _NOTE_ID.match(iid)
            and isinstance(row, dict)
            and row.get("kind") == "note"
        ):
            continue
        status = (
            row.get("status")
            if isinstance(row.get("status"), str) and row.get("status")
            else "on-trial"
        )
        if status in _HIDDEN or iid in gone:
            continue
        try:
            text = (copy / iid).read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError):
            text = ""
        front = front_fields(text)
        kind = front.get(KIND, "").lower()
        if kind not in STANDING_KINDS:
            continue  # a lesson, a note with no kind, or a kind the digest does not show
        counts = (
            (evidence.get(iid) or {}).get("counts")
            if isinstance(evidence.get(iid), dict)
            else None
        )
        counts = counts if isinstance(counts, dict) else {}
        order = row.get("changed_order")
        out.append(
            {
                "id": iid,
                "kind": kind,
                "status": status,
                "title": str(row.get("title") or front.get("title") or ""),
                "description": str(
                    row.get("description") or front.get("description") or "",
                ),
                "applies_to": front.get(APPLIES_TO, ""),
                "stated_by": front.get(STATED_BY, ""),
                "trust": _int(counts.get("clean")) - _int(counts.get("failed")),
                "changed_order": (
                    order
                    if isinstance(order, int) and not isinstance(order, bool)
                    else None
                ),
            },
        )
    out.sort(
        key=lambda n: (
            _STATUS_RANK.get(n["status"], len(_STATUS_RANK)),
            -n["trust"],
            n["changed_order"] if n["changed_order"] is not None else float("inf"),
            n["id"],
        ),
    )
    return out


def entry_line(note: Mapping[str, Any]) -> str:
    """One digest entry: ``- <id>: [<kind>, <status>] <title>: <description> (applies to: …; stated by: …)``, the part
    after the id at most :data:`ENTRY_CHARS` characters."""
    title, about = str(note.get("title") or ""), str(note.get("description") or "")
    body = (
        f"{title}: {about}"
        if title and about and about != title
        else (title or about or "(no description)")
    )
    scope = [f"applies to: {note['applies_to']}"] if note.get("applies_to") else []
    scope += [f"stated by: {note['stated_by']}"] if note.get("stated_by") else []
    text = f"[{note.get('kind')}, {note.get('status')}] {body}" + (
        f" ({'; '.join(scope)})" if scope else ""
    )
    return f"- {note.get('id')}: {_cut(text, ENTRY_CHARS)}"


def digest_text(copy: Path | str | None) -> str:
    """The standing-note digest of the library copy at *copy* (module docstring); '' when it holds no standing note."""
    if copy is None:
        return ""
    notes = standing_notes(copy)
    if not notes:
        return ""
    shown = notes[:DIGEST_ENTRIES]
    lines = [DIGEST_HEAD, *(entry_line(n) for n in shown)]
    count = f"Showing {len(shown)} of {len(notes)} standing note{'' if len(notes) == 1 else 's'}. "
    lines.append(count + DIGEST_REACH)
    return "\n".join(lines)


class StandingDigest:
    """The digest of one session, sent when it changes: :meth:`update` gives the digest the first time it is non-empty
    and then only when it differs from the one it last gave; None otherwise (nothing to repeat).
    """

    def __init__(self, copy: Callable[[], Path | None]) -> None:
        self._copy = copy
        self._sent: str | None = None

    def update(self) -> str | None:
        try:
            text = digest_text(self._copy())
        except (
            Exception
        ):  # noqa: BLE001 - an unreadable copy sends nothing; the request goes on
            return None
        if text == (self._sent or ""):
            return None
        self._sent = text
        return text or None


__all__ = [
    "DIGEST_ENTRIES",
    "DIGEST_HEAD",
    "DIGEST_REACH",
    "ENTRY_CHARS",
    "STANDING_KINDS",
    "StandingDigest",
    "digest_text",
    "entry_line",
    "front_fields",
    "standing_notes",
]
