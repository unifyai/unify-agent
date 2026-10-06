"""``UNIFY_ENTRY_RECORD``: one record for every stored entry, function or guidance.

The retrieval design study of 5 Oct (research artifact retrieval-design-v1,
section 2) found that every piece of evidence the library keeps was built for
functions only: where an entry came from, how that session ended, how often
it was used and whether that went well. Guidance recorded no use, and its
origin was kept only under a separate switch (off in every arm that ran), so
a note written after a failed session read exactly like one that had worked
five times. This module gives both kinds the same record and says it the
same way.

* **Origin.** Guidance records the requests it is written for as functions
  do (:func:`unify.function_manager.task_origin.guidance_recorded`).
* **Use.** Each session that calls a stored function, reads a guidance entry
  (``get_guidance``) or, as its storage review judged, relied on an entry,
  keeps a hash of its request against that entry: the latest
  :data:`USES_KEPT` sessions per entry and kind of use, in the request log's
  ``entry_uses`` table (created only by this switch). A session counts once
  however often it used the entry. Calls and reads made by a storage review
  are not counted (:func:`reviewing`).
* **Relied on.** The storage review already reads the whole conversation; it
  names the entries the trajectory followed or called in one JSON key,
  ``relied_on`` (:data:`REVIEW_SECTION`, :func:`record_relied`). The acting
  model is asked nothing.
* **Status.** An entry is *verified* when the session that wrote it was
  accepted (the checker's outcome, else the review's judgement), or when a
  later accepted session called it or relied on it; otherwise *unverified*,
  saying whether its session was not accepted or its outcome is unknown
  (:func:`status`). The writers are the sessions its origin records (the
  latest three).
* **Record.** One line of text per entry (:func:`record_text`): what it was
  stored for (:meth:`~unify.function_manager.task_origin.Marker.origin_line`),
  how many other logged requests resemble that request (recurrence), its
  status, and its use with how those sessions ended. It informs; nothing
  is hidden or asked.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import task_origin

logger = logging.getLogger(__name__)

FUNCTION = "function"
GUIDANCE = "guidance"
CALL = "call"
READ = "read"
RELIED = "relied"
USES_KEPT = 5
"""The latest sessions kept per entry and kind of use."""

KEY = "relied_on"

REVIEW_SECTION = (
    "## Entries Relied On\n\n"
    "Also state which stored entries this trajectory relied on: the stored "
    "functions it called and the guidance entries it followed, as they were "
    "before this review, in one JSON key, `relied_on`, on the last line of "
    'your final reply: {"relied_on": ["function_name", "guidance 5"]}, or '
    '{"relied_on": []} when it used none. When you state another key on '
    "that line, put both in one object. Judge from the conversation alone.\n\n"
)

_REVIEWING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unify_entry_record_reviewing",
    default=False,
)


def enabled() -> bool:
    """``UNIFY_ENTRY_RECORD`` (with request records on)."""
    from unify.settings import SETTINGS

    return task_origin.enabled() and bool(
        getattr(SETTINGS, "UNIFY_ENTRY_RECORD", False),
    )


@contextlib.contextmanager
def reviewing() -> Iterator[None]:
    """While the block runs (and in the tasks it starts), uses are a review's, not the session's."""
    token = _REVIEWING.set(True)
    try:
        yield
    finally:
        _REVIEWING.reset(token)


def in_review() -> bool:
    return _REVIEWING.get()


# ── the uses table ────────────────────────────────────────────────────────


def _connect(path) -> sqlite3.Connection:
    conn = task_origin._connect_outcomes(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS entry_uses (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, kind TEXT NOT NULL, ident TEXT NOT NULL, text_key TEXT"
        " NOT NULL, how TEXT NOT NULL)",
    )
    return conn


def record_use(
    kind: str,
    ident: Any,
    how: str,
    *,
    text: Optional[str] = None,
    review: bool = False,
) -> bool:
    """Keep that the current session used entry *ident* of *kind* (*how*: call, read, relied).

    *text* is the session's bounded request (default the current one). A
    call or read made inside a storage review (:func:`reviewing`) is not
    kept unless *review* (the review's own ``relied_on``). Nothing is kept
    while the switch is off or outside a keyed request. Returns whether it
    was kept.
    """
    if not enabled() or (in_review() and not review):
        return False
    text = text if text is not None else task_origin.current_request()
    ident = str(ident or "").strip()
    if not text or not ident or how not in (CALL, READ, RELIED):
        return False
    try:
        path = task_origin.request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        key = task_origin.text_key(text)
        with closing(_connect(path)) as conn, conn:
            conn.execute(
                "DELETE FROM entry_uses WHERE kind = ? AND ident = ? AND how = ?"
                " AND text_key = ?",
                (kind, ident, how, key),
            )
            conn.execute(
                "INSERT INTO entry_uses (kind, ident, text_key, how) VALUES (?, ?, ?, ?)",
                (kind, ident, key, how),
            )
            conn.execute(
                "DELETE FROM entry_uses WHERE kind = ? AND ident = ? AND how = ?"
                " AND seq NOT IN (SELECT seq FROM entry_uses WHERE kind = ? AND"
                " ident = ? AND how = ? ORDER BY seq DESC LIMIT ?)",
                (kind, ident, how, kind, ident, how, USES_KEPT),
            )
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"entry use not logged: {type(exc).__name__}: {exc}")
        return False


@dataclass
class Uses:
    """The sessions that used one entry, newest first, by kind of use (request hashes)."""

    by_how: Dict[str, List[str]] = field(default_factory=dict)
    outcomes: Dict[str, Optional[bool]] = field(default_factory=dict)

    def keys(self, how: str) -> List[str]:
        return list(self.by_how.get(how) or [])

    def all_keys(self) -> List[str]:
        return list(dict.fromkeys(k for keys in self.by_how.values() for k in keys))


def _outcomes(
    conn: sqlite3.Connection,
    keys: Sequence[str],
) -> Dict[str, Optional[bool]]:
    keys = list(dict.fromkeys(keys))
    if not keys:
        return {}
    kept: Dict[Tuple[str, str], bool] = {
        (key, source): bool(solved)
        for key, source, solved in conn.execute(
            "SELECT text_key, source, solved FROM request_outcomes"
            f" WHERE text_key IN ({', '.join('?' for _ in keys)})",
            keys,
        )
    }
    return {
        key: kept.get((key, task_origin.CHECKER), kept.get((key, task_origin.REVIEW)))
        for key in keys
    }


def uses_of(entries: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], Uses]:
    """``{(kind, ident): Uses}`` for *entries*, with how each using session ended."""
    out = {(kind, str(ident)): Uses() for kind, ident in entries}
    if not out:
        return out
    path = task_origin.request_log_path()
    if not path.exists():
        return out
    try:
        with closing(_connect(path)) as conn:
            rows = conn.execute(
                "SELECT kind, ident, text_key, how FROM entry_uses ORDER BY seq DESC",
            ).fetchall()
            for kind, ident, key, how in rows:
                if (kind, ident) in out:
                    out[(kind, ident)].by_how.setdefault(how, []).append(key)
            keys = [key for uses in out.values() for key in uses.all_keys()]
            outcomes = _outcomes(conn, keys)
    except sqlite3.Error as exc:
        logger.warning(f"entry uses not read: {type(exc).__name__}: {exc}")
        return out
    for uses in out.values():
        uses.outcomes = {key: outcomes.get(key) for key in uses.all_keys()}
    return out


def outcome_of_key(key: Optional[str]) -> Optional[bool]:
    """The kept outcome of the session whose request hashes to *key* (checker, else review)."""
    if not key:
        return None
    path = task_origin.request_log_path()
    if not path.exists():
        return None
    try:
        with closing(task_origin._connect_outcomes(path)) as conn:
            return _outcomes(conn, [key]).get(key)
    except sqlite3.Error:
        return None


# ── status and use, in words ──────────────────────────────────────────────


def _sessions_phrase(keys: Sequence[str], outcomes: Dict[str, Optional[bool]]) -> str:
    n = len(keys)
    good = sum(1 for k in keys if outcomes.get(k) is True)
    bad = sum(1 for k in keys if outcomes.get(k) is False)
    unknown = n - good - bad
    parts = []
    if good:
        parts.append(f"{good} accepted")
    if bad:
        parts.append(f"{bad} not accepted")
    if unknown:
        parts.append(f"{unknown} with the outcome unknown")
    noun = "session" if n == 1 else "sessions"
    return f"{n} {noun} ({', '.join(parts)})"


def use_phrase(kind: str, uses: Uses) -> str:
    """How the entry was used, and how those sessions ended; ``not used yet`` otherwise."""
    parts = []
    if kind == FUNCTION and uses.keys(CALL):
        parts.append("called in " + _sessions_phrase(uses.keys(CALL), uses.outcomes))
    if kind == GUIDANCE and uses.keys(READ):
        parts.append("read in " + _sessions_phrase(uses.keys(READ), uses.outcomes))
    if uses.keys(RELIED):
        parts.append(
            "relied on in " + _sessions_phrase(uses.keys(RELIED), uses.outcomes),
        )
    if not parts:
        return "not called yet" if kind == FUNCTION else "not used yet"
    return "; ".join(parts)


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def writer_keys(row: Dict[str, Any]) -> List[str]:
    """The request hashes of the sessions that wrote *row*'s content.

    With ``UNIFY_PROTECT_VERIFIED`` the session that wrote its current content
    (``content_by``); otherwise every request its origin records.
    """
    from .verified_guard import content_by

    by = content_by(row)
    if by:
        return [by]
    texts = [
        t
        for t in (_metadata(row).get(task_origin.REQUESTS_FIELD) or [])
        if isinstance(t, str)
    ]
    return [task_origin.text_key(text) for text in texts]


def status(kind: str, row: Dict[str, Any], uses: Optional[Uses] = None) -> str:
    """``verified: ...`` or ``unverified: ...``: whether the entry's content has been accepted.

    Verified when every session that wrote it was accepted, or when a later
    session that called it or relied on it was accepted. Built-in entries
    are not lessons: ``built in``.
    """
    if row.get("is_builtin"):
        return "built in"
    verb = "stored" if kind == FUNCTION else "written"
    writers = writer_keys(row)
    outcomes = [outcome_of_key(key) for key in writers]
    if writers and all(o is True for o in outcomes):
        return f"verified: {verb} in a session whose answer was accepted"
    uses = uses or Uses()
    later = [
        key
        for how in (CALL, RELIED)
        for key in uses.keys(how)
        if key not in writers and uses.outcomes.get(key) is True
    ]
    if later:
        how = "called" if kind == FUNCTION and uses.keys(CALL) else "relied on"
        return f"verified: {how} in a later session whose answer was accepted"
    if any(o is False for o in outcomes):
        return f"unverified: {verb} after a session whose answer was not accepted"
    return f"unverified: {verb} in a session whose outcome is unknown"


def recurrence_phrase(marker: "task_origin.Marker", row: Dict[str, Any]) -> str:
    """How many earlier logged requests resemble the request *row* was recorded under, or ""."""
    from unify.settings import SETTINGS

    closest = marker._closest_origin(row)
    if closest is None:
        return ""
    _, origin = closest
    current = task_origin.current_request()
    earlier = [t for t in task_origin.logged_requests() if t not in (current, origin)]
    if not earlier:
        return ""
    threshold = SETTINGS.shortlist_gate_threshold() or 0.175
    weights = task_origin.token_weights([*earlier, origin])
    n = sum(
        1 for t in earlier if task_origin.similarity(origin, t, weights) >= threshold
    )
    if not n:
        return ""
    noun = "request resembles" if n == 1 else "requests resemble"
    return f"{n} other logged {noun} the one it was recorded under"


def record_text(
    marker: "task_origin.Marker",
    kind: str,
    row: Dict[str, Any],
    uses: Optional[Uses] = None,
) -> str:
    """The entry's record in one line: what it was stored for, its recurrence, status and use."""
    uses = (
        uses
        if uses is not None
        else uses_of([(kind, ident_of(kind, row))])[(kind, ident_of(kind, row))]
    )
    parts = [marker.origin_line(row, kind=kind)]
    recurred = recurrence_phrase(marker, row)
    if recurred:
        parts.append(recurred)
    parts += [status(kind, row, uses), use_phrase(kind, uses)]
    # UNIFY_PROTECT_VERIFIED=versioned: changes kept beside the content.
    from .verified_guard import versions

    pending = versions(row.get("metadata"))
    if pending:
        noun = "version" if len(pending) == 1 else "versions"
        parts.append(
            f"{len(pending)} unverified {noun} from later sessions kept beside it",
        )
    return "; ".join(parts)


def ident_of(kind: str, row: Dict[str, Any]) -> str:
    """The key an entry's uses are kept under: a function's name, a guidance entry's id."""
    return str(row.get("guidance_id") if kind == GUIDANCE else row.get("name") or "")


# ── what the storage review says it relied on ─────────────────────────────

_OBJECT = re.compile(r"\{[^{}]*\"" + KEY + r"\"[^{}]*\}")
_GUIDANCE_ITEM = re.compile(r"^\s*guidance[\s_#:-]*(\d+)\s*$", re.IGNORECASE)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_relied(text: Any) -> Optional[List[Tuple[str, str]]]:
    """``[(kind, ident), ...]`` from the last ``relied_on`` the review's reply states; ``None`` if none."""
    for match in reversed(_OBJECT.findall(str(text or ""))):
        try:
            data = json.loads(match)
        except ValueError:
            continue
        items = data.get(KEY) if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        out: List[Tuple[str, str]] = []
        for item in items:
            value = str(item or "").strip().strip("`")
            found = _GUIDANCE_ITEM.match(value)
            if found:
                out.append((GUIDANCE, str(int(found.group(1)))))
            elif value.startswith("function "):
                name = value[len("function ") :].strip().strip("`").split("(", 1)[0]
                if _NAME.match(name):
                    out.append((FUNCTION, name))
            else:
                name = value.split("(", 1)[0].strip()
                if _NAME.match(name):
                    out.append((FUNCTION, name))
        return list(dict.fromkeys(out))
    return None


def record_relied(text: Any) -> Optional[List[Tuple[str, str]]]:
    """Keep the entries the review's reply *text* says the session relied on; returns them."""
    if not enabled():
        return None
    relied = parse_relied(text)
    for kind, ident in relied or []:
        record_use(kind, ident, RELIED, review=True)
    if relied is not None:
        logger.info(f"entries relied on: {relied}")
    return relied


__all__ = [
    "CALL",
    "FUNCTION",
    "GUIDANCE",
    "READ",
    "RELIED",
    "REVIEW_SECTION",
    "USES_KEPT",
    "Uses",
    "enabled",
    "ident_of",
    "in_review",
    "outcome_of_key",
    "parse_relied",
    "record_relied",
    "record_text",
    "record_use",
    "reviewing",
    "status",
    "use_phrase",
    "uses_of",
    "writer_keys",
]
