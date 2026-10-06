"""``UNIFY_EVIDENCE_LEDGER``: evidence that arrived during a session is kept and shown on a return.

The cost decomposition of 6 Oct (research artifact cost-decomposition-v1;
Continual-ARC LOW, the first 136 instances) found no harness reduces what it
buys again when a task returns. Overhauled Unify bought 4.05 demonstration
pairs per return visit where the program library bought 0.74, and returned
to a task at no cost on 0 of 111 returns where the program library did so on
74. Unify keeps no evidence that arrived during a session: demonstrations,
feedback, clarification answers and the requester's follow-ups come in as
plain user messages and are gone when the session ends, so a return pays
for them again. On real work the same waste is asking a clarification that
was already answered.

With the switch on, a top-level session keeps what arrived after its
request, and a later session that returns to it sees that evidence:

* **What is kept** (phase 1). Every message that arrives after the request
  (``kind="message"``: an interjection, which is how a stream sends
  demonstrations, feedback and a scripted second turn; taken by the
  storage-check handle, so only in a session that may store) and every
  clarification question with its answer (``kind="clarification"``,
  ``"Q: ...\\nA: ..."``). Each item is redacted (:func:`redact`: credential
  values the process holds, values given for a credential-named key, and
  token shapes; credentials are never stored), cut to
  :data:`MAX_ITEM_CHARS` with a marker, and kept with its kind, size and time
  under the session's request (its ``text_key`` and task key) in the
  ``evidence`` table of ``<UNIFY_HOME>/request_log.sqlite``. A session keeps
  at most :data:`MAX_SESSION_CHARS`; the table keeps the latest
  :data:`LEDGER_SIZE` items. A store that cannot be written is skipped with a
  warning.
* **How a return finds it.** By request, not by stored entry (demos were
  bought again even when nothing was stored): the earlier logged requests
  that are this same request, or share a rare whole identifier with it (the
  rule of :func:`unify.function_manager.task_origin.shared_identifiers`, with
  the logged requests as the corpus). A request whose only link is an alias
  without digits (``amber-heron``) is not found.
* **What the model sees.** ``seen_before`` in the sandbox, plain data (a list
  of dicts: the earlier request's opening, which earlier visit, kind, label,
  content and age in seconds, newest first, at most :data:`MAX_SHOWN_ITEMS`
  and :data:`MAX_SHOWN_CHARS`), and one sentence in the first message
  saying it is there and what it holds. No instruction to check, reuse or
  trust anything: it informs, never forces. Nothing found, nothing bound and
  no sentence.

It runs for a top-level task only (a sub-agent's request is its caller's,
and what its caller tells it is not evidence from outside). It needs request
records (``UNIFY_TASK_ORIGIN`` or ``UNIFY_TRY_FIRST``), which it logs as the
stream corpus does. Off: nothing is kept, read, bound or said.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import logging
import re
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

TABLE = "evidence"
#: The name the evidence is bound to in the sandbox.
NAME = "seen_before"
MESSAGE = "message"
"""A message that arrived after the request (an interjection)."""
CLARIFICATION = "clarification"
"""A clarification question with its answer."""
#: The ledger keeps this many items, the latest.
LEDGER_SIZE = 2000
#: One item keeps at most this many characters (a marker says how many were cut).
MAX_ITEM_CHARS = 16_000
#: One session keeps at most this many characters in all; later items are dropped.
MAX_SESSION_CHARS = 64_000
#: ``seen_before`` holds at most this many items and characters of content.
MAX_SHOWN_ITEMS = 40
MAX_SHOWN_CHARS = 48_000
#: The opening of an earlier request that each item shows.
REQUEST_CHARS = 200

_CUT = "\n[... {n} more characters not kept]"


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_EVIDENCE_LEDGER", False))


@dataclass
class Session:
    """The request one top-level session keeps its evidence under."""

    key: str
    """The request's task key (:func:`task_origin.task_key`)."""
    text: str
    """The request's bounded copy (:func:`task_origin.bounded_text`)."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kept_chars: int = 0
    _questions: List[str] = field(default_factory=list)

    @property
    def text_key(self) -> str:
        from unify.function_manager import task_origin

        return task_origin.text_key(self.text)

    def question(self, question: Any) -> None:
        """A clarification question was asked; :meth:`answer` records it with its answer."""
        self._questions.append(str(question or ""))

    def answer(self, answer: Any) -> bool:
        """Record the oldest open question with *answer* (questions are answered in order)."""
        question = self._questions.pop(0) if self._questions else ""
        return record(
            CLARIFICATION,
            f"Q: {question}\nA: {answer if answer is not None else ''}",
            key=self,
        )


_SESSION: contextvars.ContextVar[Optional[Session]] = contextvars.ContextVar(
    "unify_evidence_ledger_session",
    default=None,
)


def enter() -> Optional[contextvars.Token]:
    """Start the ledger session of the current keyed request; ``None`` when off or unkeyed.

    Called after :func:`task_origin.enter` for a top-level task. A context
    that has a session already keeps it.
    """
    from unify.function_manager import task_origin

    if not enabled() or _SESSION.get() is not None:
        return None
    key, text = task_origin.current(), task_origin.current_request()
    if not key or not text:
        return None
    return _SESSION.set(Session(key=key, text=text))


def leave(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _SESSION.reset(token)
    except ValueError:  # entered in another context
        pass


def current_session() -> Optional[Session]:
    return _SESSION.get() if enabled() else None


# ── redaction ────────────────────────────────────────────────────────────


# ``name: value`` / ``name=value`` / ``"name": "value"`` (and ``Bearer``).
_ASSIGNMENT = re.compile(
    r"""(?P<name>[A-Za-z][\w.-]{1,63})(?P<sep>["']?\s*[:=]\s*["']?(?:Bearer\s+|Basic\s+|token\s+)?)"""
    r"""(?P<value>[^\s"',;]{8,})""",
)
# Well-known credential shapes (provider keys, personal access tokens).
_TOKEN_SHAPE = re.compile(
    r"\b(?:"
    r"(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r")",
)


def redact(text: str) -> str:
    """*text* without credentials.

    Credential values the process holds (``transcripts.scrub``), values given
    for a credential-named key (``store_trust.credential_key``; replaced by
    ``store_cases``'s salted placeholder, and wherever they recur), and
    well-known token shapes.
    """
    from unify import transcripts
    from unify.function_manager import store_cases
    from unify.function_manager.store_trust import credential_key

    text = transcripts.scrub(str(text))
    redactor = store_cases._Redactor.fresh()

    def named(match: "re.Match[str]") -> str:
        if not credential_key(match.group("name")):
            return match.group(0)
        value = match.group("value")
        return match.group("name") + match.group("sep") + redactor.placeholder(value)

    text = _ASSIGNMENT.sub(named, text)
    text = redactor.scrub(text)
    return _TOKEN_SHAPE.sub(lambda m: redactor.placeholder(m.group(0)), text)


# ── the ledger ───────────────────────────────────────────────────────────


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, text_key TEXT NOT NULL, task_key TEXT NOT NULL,"
        " session TEXT NOT NULL, kind TEXT NOT NULL, label TEXT,"
        " content TEXT NOT NULL, chars INTEGER NOT NULL, created_at TEXT NOT NULL)",
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {TABLE}_text_key ON {TABLE} (text_key)",
    )
    return conn


def record(
    kind: str,
    content: Any,
    *,
    label: Optional[str] = None,
    key: Optional[Session] = None,
) -> bool:
    """Keep *content* of *kind* under the session *key* (default: the current one); whether it was kept.

    Redacted, cut to :data:`MAX_ITEM_CHARS` and to what the session may still
    keep (:data:`MAX_SESSION_CHARS`); ``chars`` is its redacted size before
    any cut. Nothing is kept while off, outside a session or for empty
    content.
    """
    from unify.function_manager import task_origin

    session = key if key is not None else current_session()
    if not enabled() or session is None:
        return False
    text = redact(content if isinstance(content, str) else str(content))
    if not text.strip():
        return False
    size = len(text)
    room = min(MAX_ITEM_CHARS, MAX_SESSION_CHARS - session.kept_chars)
    if room <= len(_CUT.format(n=size)):
        logger.info(
            f"evidence not kept: the session kept {session.kept_chars} characters "
            f"already ({kind}, {size} chars)",
        )
        return False
    if size > room:
        marker = _CUT.format(n=size)
        head = room - len(marker)
        text = text[:head] + _CUT.format(n=size - head)
    try:
        path = task_origin.request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as conn, conn:
            conn.execute(
                f"INSERT INTO {TABLE} (text_key, task_key, session, kind, label,"
                " content, chars, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session.text_key,
                    session.key,
                    session.id,
                    str(kind),
                    label,
                    text,
                    size,
                    _now().isoformat(),
                ),
            )
            conn.execute(
                f"DELETE FROM {TABLE} WHERE seq NOT IN (SELECT seq FROM {TABLE}"
                " ORDER BY seq DESC LIMIT ?)",
                (LEDGER_SIZE,),
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"evidence not written: {type(exc).__name__}: {exc}")
        return False
    session.kept_chars += len(text)
    return True


def _related_requests(current_text: str) -> Dict[str, str]:
    """``{text_key: text}`` of the logged requests that are *current_text* or share a rare identifier with it."""
    from unify.function_manager import task_origin

    logged = task_origin.logged_requests()
    texts = list(dict.fromkeys([*logged, current_text]))
    mine = set(task_origin.identifiers(current_text))
    # ``shared_identifiers``'s rule, taken once over the whole log: an
    # identifier of this request that some, but not every, known request has.
    named = {text: mine & set(task_origin.identifiers(text)) for text in texts}
    df: Dict[str, int] = {}
    for found_ids in named.values():
        for low in found_ids:
            df[low] = df.get(low, 0) + 1
    rare = {low for low, count in df.items() if count < len(texts)}
    found: Dict[str, str] = {}
    for text in logged:
        if text == current_text or named[text] & rare:
            found[task_origin.text_key(text)] = text
    return found


def earlier(
    current_text: Optional[str],
    *,
    session: Optional[Session] = None,
) -> List[Dict[str, Any]]:
    """The evidence kept while handling earlier requests like *current_text*, newest first.

    *current_text* is the request's bounded copy. Earlier requests are the
    logged ones that are the same request or share a rare whole identifier
    with it; the rows of *session* (default: the current one) are left out.
    Each item is plain data: ``request`` (the opening of that request),
    ``visit`` (1 for the latest earlier session, 2 for the one before...),
    ``kind``, ``label``, ``content`` and ``age_seconds``. At most
    :data:`MAX_SHOWN_ITEMS` and :data:`MAX_SHOWN_CHARS` of content.
    """
    from unify.function_manager import task_origin

    if not enabled() or not current_text:
        return []
    session = session if session is not None else current_session()
    path = task_origin.request_log_path()
    if not path.exists():
        return []
    try:
        with closing(sqlite3.connect(path, timeout=30)) as conn:
            if not conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (TABLE,),
            ).fetchone():
                return []
            related = _related_requests(current_text)
            if not related:
                return []
            marks = ", ".join("?" for _ in related)
            rows = conn.execute(
                f"SELECT text_key, session, kind, label, content, created_at"
                f" FROM {TABLE} WHERE text_key IN ({marks}) AND session != ?"
                " ORDER BY seq DESC",
                (*related, session.id if session is not None else ""),
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning(f"evidence not read: {type(exc).__name__}: {exc}")
        return []
    now = _now()
    visits: Dict[str, int] = {}
    items: List[Dict[str, Any]] = []
    shown = 0
    for text_key, sid, kind, label, content, created_at in rows:
        if len(items) >= MAX_SHOWN_ITEMS or shown + len(content) > MAX_SHOWN_CHARS:
            break
        try:
            age = max(
                0,
                int((now - dt.datetime.fromisoformat(created_at)).total_seconds()),
            )
        except (TypeError, ValueError):
            age = None
        visits.setdefault(sid, len(visits) + 1)
        items.append(
            {
                "request": related[text_key][:REQUEST_CHARS],
                "visit": visits[sid],
                "kind": kind,
                "label": label,
                "content": content,
                "age_seconds": age,
            },
        )
        shown += len(content)
    return items


_KIND_NOUNS = ((MESSAGE, "message"), (CLARIFICATION, "clarification"))


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def line(items: List[Dict[str, Any]]) -> Optional[str]:
    """The first message's sentence about ``seen_before``; ``None`` for no items."""
    if not items:
        return None
    visits = len({item.get("visit") for item in items})
    counts = [
        _plural(sum(1 for item in items if item.get("kind") == kind), noun)
        for kind, noun in _KIND_NOUNS
        if any(item.get("kind") == kind for item in items)
    ]
    other = sum(1 for item in items if item.get("kind") not in dict(_KIND_NOUNS))
    if other:
        counts.append(_plural(other, "other item"))
    return (
        "Evidence received while handling "
        f"{_plural(visits, 'earlier request')} like this one is in "
        f"`{NAME}` ({', '.join(counts)})."
    )


def bind(sandbox: Any, items: List[Dict[str, Any]]) -> None:
    """Bind *items* as ``seen_before`` in *sandbox* and in the sessions it shares its objects with.

    Plain JSON data, so a worker sandbox gets it as a value copy.
    """
    if sandbox is None or not items:
        return
    data = json.loads(json.dumps(items))
    sandbox.global_state[NAME] = data
    core_globals = getattr(sandbox, "core_globals", None)
    if isinstance(core_globals, dict):
        core_globals[NAME] = data
