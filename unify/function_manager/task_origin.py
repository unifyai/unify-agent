"""Which request a stored function came from (``UNIFY_TASK_ORIGIN``, ``UNIFY_TRY_FIRST``).

A top-level ``act()`` keys its request (the session's first user message): a
short hash of the whitespace-collapsed text, and a bounded copy of that text
(at most 4,000 characters: all of it, or its first and last 2,000). The loop
and its storage review inherit the key through the context. With the switch
on, a function stored while handling that request records both in its
``metadata`` (``origin_tasks``: the hashes; ``origin_requests``: the copies of
the latest :data:`MAX_ORIGIN_REQUESTS` distinct requests; the field names are
kept from the first version), and a search from a later request adds
``similar_request: <score>`` to the function's result when the current request
is close to one it was stored from: a score of 1 for the same text, otherwise
the similarity below, shown rounded to two decimals from
:data:`SIMILAR_REQUEST_THRESHOLD` on. Neither origin field is ever shown in a
library result. Sub-agents inherit the key of the request they work for. With
the switch off nothing is recorded or marked. ``UNIFY_TASK_ORIGIN`` turns this
on by itself; ``UNIFY_TRY_FIRST`` turns it on together with its prompt
paragraph.

The similarity knows nothing about any request format. A request is reduced
to its set of lower-cased tokens: runs of letters and runs of digits, each a
token of its own (``Q3`` is ``q`` and ``3``; ``13x13`` is ``13``, ``x`` and
``13``), so a value written next to a word is compared apart from it. Every
token is kept. Each is weighted by how rare it is among the requests known at
the search (the origin requests of every function in the library, and the
current one): ``ln(N / df)`` over those N distinct requests, so wording that
every known request shares (a stream's common preamble) weighs nothing, and
the values one request alone carries weigh most. The score is the weighted
Jaccard index of the two token sets: the weight of the tokens both have over
the weight of the tokens either has. With few known requests there is little
to tell shared wording from distinctive wording by, so the score is low (with
one stored request it is 0 unless the texts are equal): a mark needs a
library of a few requests.

``UNIFY_SIMILAR_REQUEST_IDENTIFIERS`` also keeps every whole identifier as a
token of its own: an ASCII word of 6 or more letters, digits, ``_`` or ``-``
that mixes letters and digits (``task-ddc8a32b``, ``inv_2024q3``), next to the
runs it is split into. A shared id then weighs as one rare token instead of
a few short runs (``ddc``, ``8``, ``a``, ``32``, ``b``) that other ids share.
A request without such a word is compared exactly as without the switch.

``UNIFY_SIMILAR_REQUEST_CORPUS=stream`` weighs the tokens over more requests:
each top-level request is also logged (its key and bounded copy) in
``<UNIFY_HOME>/request_log.sqlite``, which keeps the latest
:data:`REQUEST_LOG_SIZE` distinct requests across restarts, and the weights
count those as well as the library's origin requests and the current one. A
function stored for one or two requests then scores on wording the stream's
other requests do not share, where the library alone could not tell it from
a preamble. A sub-agent's request is not logged (it is not a new task).

Calibrated offline on a split of recorded opening requests and hand-written
assistant requests (research artifact similar-request-v1): recurring requests
whose parameters change score about 0.25-0.9, different requests in one
domain mostly under 0.25 but up to about 0.45 when they share the object they
act on (the same report, the same requester), unrelated requests under 0.1.
A mark, and its score, is a hint to check, never proof.

``UNIFY_ORIGIN_PROVENANCE`` says why a function is marked (:meth:`Marker.provenance`):
the whole identifiers (the shape above, with or without
``UNIFY_SIMILAR_REQUEST_IDENTIFIERS``) that the closest request it was stored
from shares with the current one, rarest first, at most
:data:`MAX_SHARED_IDENTIFIERS`, leaving out any that every known request
has; or that it was stored for this same request. With ``UNIFY_OUTCOME`` a
session's checked outcome is kept under its request (:func:`record_outcome`,
in the request log's ``request_outcomes`` table) and the text adds whether
the checker accepted the answer of the session the function was stored
from. With ``UNIFY_REVIEW_OUTCOME`` the storage review's own judgement of
the conversation (was the final answer confirmed or rejected by the
requester or the environment?) is kept the same way, under its own source,
and the text says "confirmed (rejected), as judged by its review"; a
checker's outcome wins over it. Only the shared identifiers and the verdict
are shown, never the origin text.

``UNIFY_GUIDANCE_ORIGIN`` records the same origin fields for guidance entries
(:func:`guidance_enabled`), in the guidance table's ``origin`` column, which
no guidance read returns; the gated shortlist scores them like functions.

``UNIFY_SHORTLIST_RELATED`` logs every top-level request as the stream
corpus does, with a short hash of each of its lines (the whole request's,
in the log's ``request_lines`` table), so a later request's lines that most
logged requests share can be told from its own (:func:`line_keys`).

``UNIFY_REVIEW_RECURRENCE`` logs every top-level request as the stream
corpus does and counts, for the storage review, the earlier logged requests
whose ``similar_request`` to the current one reaches a threshold
(:func:`recurrence`).
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import math
import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

FIELD = "origin_tasks"
REQUESTS_FIELD = "origin_requests"
# Each stored function keeps the copies of this many distinct requests (the
# latest), so its metadata stays bounded however often it is overwritten.
MAX_ORIGIN_REQUESTS = 3
# Head and tail kept from a long request.
_HEAD = 2000
_TAIL = 2000
# A search result shows ``similar_request`` from this similarity on.
SIMILAR_REQUEST_THRESHOLD = 0.24
MARK = "similar_request"
# UNIFY_ORIGIN_PROVENANCE: the field a marked search row says why in.
ORIGIN_MARK = "origin"
# UNIFY_SIMILAR_REQUEST_CORPUS=stream: the request log keeps this many
# distinct top-level requests, the latest.
REQUEST_LOG_SIZE = 200
REQUEST_LOG_FILE = "request_log.sqlite"
# UNIFY_SHORTLIST_RELATED: at most this many line hashes per logged request.
MAX_LINE_KEYS = 500
# UNIFY_ORIGIN_PROVENANCE: at most this many shared identifiers are named.
MAX_SHARED_IDENTIFIERS = 2
# The request log keeps the checked outcomes of this many requests, the latest.
OUTCOME_LOG_SIZE = 1000

_WS = re.compile(r"\s+")
# A run of letters or a run of digits (any script).
_TOKEN = re.compile(r"[^\W\d_]+|\d+")
# UNIFY_SIMILAR_REQUEST_IDENTIFIERS: a whole identifier, 6 or more ASCII
# letters, digits, ``_`` or ``-`` mixing letters and digits. Kept with a
# prefix no run of letters or digits has, so it never merges with one.
_IDENTIFIER = re.compile(
    r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{6,}\b",
)
_IDENTIFIER_PREFIX = "#"


@dataclass(frozen=True)
class _Task:
    key: str
    text: str


_CURRENT: contextvars.ContextVar[Optional[_Task]] = contextvars.ContextVar(
    "unify_task_origin",
    default=None,
)


def enabled() -> bool:
    """Whether requests are keyed, recorded and marked (either switch)."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_TASK_ORIGIN", False)) or bool(
        getattr(SETTINGS, "UNIFY_TRY_FIRST", False),
    )


def _normalised(request: Any) -> str:
    if request is None:
        return ""
    if isinstance(request, str):
        text = request
    else:
        text = json.dumps(request, sort_keys=True, default=str)
    return _WS.sub(" ", text).strip()


def task_key(request: Any) -> Optional[str]:
    """The key of a request: 16 hex digits of the sha256 of its normalised text."""
    text = _normalised(request)
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def bounded_text(request: Any) -> str:
    """The copy of a request a function records: normalised, at most 4,000 characters."""
    text = _normalised(request)
    if len(text) > _HEAD + _TAIL:
        text = text[:_HEAD] + " " + text[-_TAIL:]
    return text


def _identifiers_enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", False))


def tokens(text: str) -> frozenset[str]:
    """The tokens a request is compared by: its runs of letters and of digits.

    With ``UNIFY_SIMILAR_REQUEST_IDENTIFIERS`` also its whole identifiers,
    each prefixed with ``#``.
    """
    return _tokens(text, _identifiers_enabled())


@lru_cache(maxsize=4096)
def _tokens(text: str, identifiers: bool) -> frozenset[str]:
    found = set(_TOKEN.findall(text.lower()))
    if identifiers:
        found.update(
            _IDENTIFIER_PREFIX + word.lower() for word in _IDENTIFIER.findall(text)
        )
    return frozenset(found)


def token_weights(texts: Iterable[str]) -> Dict[str, float]:
    """``ln(N / df)`` per token over the distinct *texts*; a token in all of them weighs 0."""
    sets = [tokens(text) for text in dict.fromkeys(texts)]
    df: Dict[str, int] = {}
    for token_set in sets:
        for token in token_set:
            df[token] = df.get(token, 0) + 1
    n = len(sets)
    return {token: math.log(n / count) for token, count in df.items()}


def similarity(a: str, b: str, weights: Dict[str, float]) -> float:
    """Weighted Jaccard index of the token sets of *a* and *b*."""
    ta, tb = tokens(a), tokens(b)
    union = sum(weights.get(t, 0.0) for t in ta | tb)
    if union <= 1e-12:
        return 1.0
    return sum(weights.get(t, 0.0) for t in ta & tb) / union


def enter(request: Any) -> Optional[contextvars.Token]:
    """Key the current context's task by *request*, unless it is keyed already.

    A sub-agent is started inside the task it works for, whose key it
    inherits and keeps. The caller resets the token with :func:`leave` in
    the same context once the task's handle is built.
    """
    if not enabled() or _CURRENT.get() is not None:
        return None
    key = task_key(request)
    if key is None:
        return None
    task = _Task(key=key, text=bounded_text(request))
    related = _related_enabled()
    if _stream_corpus() or recurrence_enabled() or related or _records_requests():
        _log_request(task)
    if related:
        _log_request_lines(task.key, line_keys(request))
    return _CURRENT.set(task)


def _stream_corpus() -> bool:
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", "") == "stream"


def _related_enabled() -> bool:
    """``UNIFY_SHORTLIST_RELATED``: its distinct lines are taken against the logged requests."""
    from unify.settings import SETTINGS

    return SETTINGS.shortlist_related() is not None


def _records_requests() -> bool:
    """``UNIFY_ENTRY_RECORD`` or ``UNIFY_PROTECT_VERIFIED``: requests are logged, outcomes and guidance origins kept.

    The log maps a use's request hash back to its text and weighs rare words
    over the stream, as ``UNIFY_SIMILAR_REQUEST_CORPUS=stream`` does.
    """
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_ENTRY_RECORD", False)) or bool(
        getattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", False),
    )


def guidance_enabled() -> bool:
    """``UNIFY_GUIDANCE_ORIGIN``, or ``UNIFY_ENTRY_RECORD`` (one origin for both kinds), with request records on."""
    from unify.settings import SETTINGS

    return enabled() and (
        bool(getattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", False))
        or bool(getattr(SETTINGS, "UNIFY_ENTRY_RECORD", False))
    )


def provenance_enabled() -> bool:
    """``UNIFY_ORIGIN_PROVENANCE`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", False))


def listing_provenance_enabled() -> bool:
    """``UNIFY_LISTING_PROVENANCE`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_LISTING_PROVENANCE", False))


def lesson_status_enabled() -> bool:
    """``UNIFY_LESSON_STATUS`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_LESSON_STATUS", False))


def listing_usage_enabled() -> bool:
    """``UNIFY_LISTING_USAGE`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_LISTING_USAGE", False))


def listing_notes_enabled() -> bool:
    """Whether any switch adds notes to the shortlist's lines."""
    return (
        listing_provenance_enabled()
        or lesson_status_enabled()
        or listing_usage_enabled()
    )


def guidance_recorded() -> bool:
    """Whether guidance entries record their requests.

    ``UNIFY_GUIDANCE_ORIGIN``, or a listing switch that shows what they were
    written for (``UNIFY_LISTING_PROVENANCE``, ``UNIFY_LESSON_STATUS``).
    """
    return (
        guidance_enabled()
        or listing_provenance_enabled()
        or lesson_status_enabled()
        or (enabled() and _records_requests())
    )


def recurrence_enabled() -> bool:
    """``UNIFY_REVIEW_RECURRENCE`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_REVIEW_RECURRENCE", False))


def review_outcome_enabled() -> bool:
    """``UNIFY_REVIEW_OUTCOME`` (with request records on)."""
    from unify.settings import SETTINGS

    return enabled() and bool(getattr(SETTINGS, "UNIFY_REVIEW_OUTCOME", False))


def require_origin_link_prerequisites() -> None:
    """Refuse a switch that reads or records under requests without request records.

    ``UNIFY_ORIGIN_PROVENANCE``, ``UNIFY_REVIEW_RECURRENCE``,
    ``UNIFY_REVIEW_OUTCOME``, ``UNIFY_GUIDANCE_ORIGIN``,
    ``UNIFY_LISTING_PROVENANCE``, ``UNIFY_LESSON_STATUS`` and
    ``UNIFY_LISTING_USAGE``.

    Each reads, keeps or records something under the requests that
    ``UNIFY_TASK_ORIGIN`` (or ``UNIFY_TRY_FIRST``) records; without them they
    would never say or record anything.
    """
    from unify.settings import SETTINGS

    if enabled():
        return
    for name in (
        "UNIFY_ORIGIN_PROVENANCE",
        "UNIFY_REVIEW_RECURRENCE",
        "UNIFY_REVIEW_OUTCOME",
        "UNIFY_GUIDANCE_ORIGIN",
        "UNIFY_LISTING_PROVENANCE",
        "UNIFY_LESSON_STATUS",
        "UNIFY_LISTING_USAGE",
        "UNIFY_ENTRY_RECORD",
        "UNIFY_SEARCH_IDENTIFIERS",
        "UNIFY_PROTECT_VERIFIED",
    ):
        if getattr(SETTINGS, name, False):
            raise ValueError(
                f"{name} needs UNIFY_TASK_ORIGIN=1 (or UNIFY_TRY_FIRST=1): it "
                "reads the requests stored entries were recorded under.",
            )


def request_log_path() -> Path:
    """``<UNIFY_HOME>/request_log.sqlite``: the log of top-level requests."""
    from unify import db

    return db.store_home() / REQUEST_LOG_FILE


def _connect_log(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, key TEXT NOT NULL UNIQUE, text TEXT NOT NULL)",
    )
    return conn


CHECKER = "checker"
"""An outcome the environment's checker posted (``UNIFY_OUTCOME``)."""
REVIEW = "review"
"""An outcome the session's storage review judged from the conversation (``UNIFY_REVIEW_OUTCOME``)."""


def _connect_outcomes(path: Path) -> sqlite3.Connection:
    conn = _connect_log(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS request_outcomes (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, text_key TEXT NOT NULL, source TEXT NOT NULL,"
        " solved INTEGER NOT NULL, UNIQUE (text_key, source))",
    )
    return conn


def text_key(text: str) -> str:
    """The key an outcome is kept under: 16 hex digits of the sha256 of a bounded copy.

    Taken over the copy a function records (not the full request), so the
    copy alone finds it.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _checker_kept() -> bool:
    return (
        provenance_enabled()
        or listing_notes_enabled()
        or (enabled() and _records_requests())
    )


def record_outcome(solved: Any, *, source: str = CHECKER) -> bool:
    """Keep the current request's outcome from *source*; whether it was kept.

    *source* :data:`CHECKER` is the outcome the environment posted
    (``UNIFY_OUTCOME``), kept while ``UNIFY_ORIGIN_PROVENANCE`` or a switch
    that shows outcomes in the shortlist (``UNIFY_LISTING_PROVENANCE``,
    ``UNIFY_LESSON_STATUS``, ``UNIFY_LISTING_USAGE``) is on;
    :data:`REVIEW` is the storage review's judgement from the conversation,
    kept while ``UNIFY_REVIEW_OUTCOME`` is on. *solved* is ``True`` or
    ``False``; anything else (unknown) keeps nothing. The latest outcome of
    a request from a source replaces the earlier one; the latest
    :data:`OUTCOME_LOG_SIZE` are kept. A log that cannot be written is
    skipped with a warning.
    """
    task = _CURRENT.get()
    allowed = {CHECKER: _checker_kept, REVIEW: review_outcome_enabled}.get(source)
    if allowed is None or not allowed() or task is None or not isinstance(solved, bool):
        return False
    try:
        path = request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_outcomes(path)) as conn, conn:
            key = text_key(task.text)
            conn.execute(
                "DELETE FROM request_outcomes WHERE text_key = ? AND source = ?",
                (key, source),
            )
            conn.execute(
                "INSERT INTO request_outcomes (text_key, source, solved)"
                " VALUES (?, ?, ?)",
                (key, source, int(solved)),
            )
            conn.execute(
                "DELETE FROM request_outcomes WHERE seq NOT IN (SELECT seq FROM"
                " request_outcomes ORDER BY seq DESC LIMIT ?)",
                (OUTCOME_LOG_SIZE,),
            )
        return True
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"request outcome not written: {type(exc).__name__}: {exc}")
        return False


def origin_outcome(text: str) -> Optional[tuple[bool, str]]:
    """``(solved, source)`` kept for the request whose bounded copy is *text*; ``None`` if none.

    The checker's outcome wins over the review's judgement.
    """
    path = request_log_path()
    if not path.exists():
        return None
    try:
        with closing(_connect_outcomes(path)) as conn:
            rows = dict(
                conn.execute(
                    "SELECT source, solved FROM request_outcomes WHERE text_key = ?",
                    (text_key(text),),
                ).fetchall(),
            )
    except sqlite3.Error as exc:
        logger.warning(f"request outcome not read: {type(exc).__name__}: {exc}")
        return None
    for source in (CHECKER, REVIEW):
        if source in rows:
            return bool(rows[source]), source
    return None


def _log_request(task: _Task) -> None:
    """Log *task* as the latest request; keep the latest :data:`REQUEST_LOG_SIZE`.

    A request seen before moves to the end. A log that cannot be written is
    skipped with a warning: the weights then come from the library alone.
    """
    try:
        path = request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_log(path)) as conn, conn:
            conn.execute("DELETE FROM requests WHERE key = ?", (task.key,))
            conn.execute(
                "INSERT INTO requests (key, text) VALUES (?, ?)",
                (task.key, task.text),
            )
            conn.execute(
                "DELETE FROM requests WHERE seq NOT IN"
                " (SELECT seq FROM requests ORDER BY seq DESC LIMIT ?)",
                (REQUEST_LOG_SIZE,),
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"request log not written: {type(exc).__name__}: {exc}")


def line_keys(request: Any) -> List[str]:
    """``UNIFY_SHORTLIST_RELATED``: a short hash of each distinct whitespace-collapsed line of *request*.

    Taken from the whole request (the logged copy keeps only its first and
    last 2,000 characters), at most :data:`MAX_LINE_KEYS`.
    """
    text = request if isinstance(request, str) else _normalised(request)
    keys: Dict[str, None] = {}
    for line in str(text or "").splitlines():
        line = _WS.sub(" ", line).strip()
        if line:
            keys[hashlib.sha256(line.encode("utf-8")).hexdigest()[:16]] = None
        if len(keys) >= MAX_LINE_KEYS:
            break
    return list(keys)


def _log_request_lines(key: str, lines: List[str]) -> None:
    """Keep the line hashes of the logged request *key*; drop those of requests no longer logged."""
    try:
        path = request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_log(path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS request_lines (key TEXT PRIMARY KEY,"
                " lines TEXT NOT NULL)",
            )
            conn.execute(
                "INSERT OR REPLACE INTO request_lines (key, lines) VALUES (?, ?)",
                (key, json.dumps(lines)),
            )
            conn.execute(
                "DELETE FROM request_lines WHERE key NOT IN (SELECT key FROM requests)",
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"request lines not written: {type(exc).__name__}: {exc}")


def logged_request_lines() -> List[frozenset]:
    """The line hashes of each logged request but the current one, oldest first.

    A request logged without them (before ``UNIFY_SHORTLIST_RELATED``) is
    left out. Empty without a log.
    """
    path = request_log_path()
    if not path.exists():
        return []
    current_key = current()
    try:
        with closing(_connect_log(path)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table'"
                " AND name = 'request_lines'",
            ).fetchone()
            if not exists:
                return []
            rows = conn.execute(
                "SELECT r.key, l.lines FROM requests r JOIN request_lines l"
                " ON l.key = r.key ORDER BY r.seq",
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning(f"request lines not read: {type(exc).__name__}: {exc}")
        return []
    out = []
    for key, lines in rows:
        if key == current_key:
            continue
        try:
            out.append(frozenset(json.loads(lines)))
        except ValueError:
            continue
    return out


def logged_requests() -> List[str]:
    """The logged requests' bounded copies, oldest first (empty without a log)."""
    path = request_log_path()
    if not path.exists():
        return []
    try:
        with closing(_connect_log(path)) as conn:
            return [
                text
                for (text,) in conn.execute(
                    "SELECT text FROM requests ORDER BY seq",
                )
            ]
    except sqlite3.Error as exc:
        logger.warning(f"request log not read: {type(exc).__name__}: {exc}")
        return []


def identifiers(text: str) -> Dict[str, str]:
    """The whole identifiers of *text*, lower-cased, each with its first spelling."""
    found: Dict[str, str] = {}
    for word in _IDENTIFIER.findall(text):
        found.setdefault(word.lower(), word)
    return found


def shared_identifiers(
    current: str,
    origin: str,
    known: Iterable[str],
    *,
    limit: int = MAX_SHARED_IDENTIFIERS,
) -> List[str]:
    """The whole identifiers *current* and *origin* share, rarest among *known* first.

    *known* are the requests the weights are taken over (they should include
    both). An identifier every known request has weighs nothing and is left
    out; ties keep the order of *current*. Spelled as *current* spells them.
    """
    mine = identifiers(current)
    shared = [low for low in mine if low in identifiers(origin)]
    if not shared:
        return []
    texts = list(dict.fromkeys([*known, current, origin]))
    sets = [frozenset(identifiers(text)) for text in texts]
    weighted = []
    for order, low in enumerate(shared):
        df = sum(1 for found in sets if low in found)
        weight = math.log(len(sets) / df) if df else 0.0
        if weight > 1e-12:
            weighted.append((-weight, order, mine[low]))
    weighted.sort()
    return [word for _, _, word in weighted[: max(limit, 0)]]


def identifier_matches(
    query: str,
    rows: Sequence[Dict[str, Any]],
    *,
    key: str = "guidance_id",
) -> Dict[Any, List[str]]:
    """``UNIFY_SEARCH_IDENTIFIERS``: ``{row[key]: shared identifiers}`` for the rows whose recorded requests name an identifier of *query*.

    Ordered by how rare the rarest shared identifier is among the recorded
    (and, with a request log, the logged) requests, then by how many are
    shared. An identifier every recorded request names says nothing and is
    ignored. Empty when *query* names no identifier.
    """
    wanted = identifiers(query)
    if not wanted:
        return {}
    known = [text for row in rows for text in _origins(row)[1]]
    known = list(dict.fromkeys([*logged_requests(), *known]))
    if not known:
        return {}
    sets = [frozenset(identifiers(text)) for text in known]
    weight: Dict[str, float] = {}
    for low in wanted:
        df = sum(1 for found in sets if low in found)
        if 0 < df < len(sets):
            weight[low] = math.log(len(sets) / df)
    scored = []
    for order, row in enumerate(rows):
        texts = _origins(row)[1]
        named = set().union(*(identifiers(text) for text in texts)) if texts else set()
        shared = sorted(
            (low for low in weight if low in named),
            key=lambda low: -weight[low],
        )
        if shared:
            scored.append((-weight[shared[0]], -len(shared), order, row, shared))
    scored.sort(key=lambda item: item[:3])
    return {row.get(key): [wanted[low] for low in shared] for *_, row, shared in scored}


@dataclass(frozen=True)
class Recurrence:
    """How many earlier logged requests resemble the current one (``UNIFY_REVIEW_RECURRENCE``)."""

    similar: int
    earlier: int
    threshold: float
    closest: Optional[float]


def recurrence(threshold: Optional[float] = None) -> Optional[Recurrence]:
    """The earlier logged requests whose ``similar_request`` to the current one is at least *threshold*.

    *threshold* defaults to :data:`SIMILAR_REQUEST_THRESHOLD`. Weights are
    taken over the logged requests and the current one. ``None`` while the
    switch is off or no request is current.
    """
    task = _CURRENT.get()
    if not recurrence_enabled() or task is None:
        return None
    t = SIMILAR_REQUEST_THRESHOLD if threshold is None else float(threshold)
    earlier = [text for text in logged_requests() if text != task.text]
    if not earlier:
        return Recurrence(0, 0, t, None)
    weights = token_weights([*earlier, task.text])
    scores = [similarity(task.text, text, weights) for text in earlier]
    return Recurrence(
        similar=sum(1 for score in scores if score >= t),
        earlier=len(earlier),
        threshold=t,
        closest=max(scores),
    )


def leave(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _CURRENT.reset(token)
    except ValueError:
        # Reset from another context (a handle cleaned up elsewhere).
        pass


def current() -> Optional[str]:
    task = _CURRENT.get()
    return task.key if task is not None else None


def current_request() -> Optional[str]:
    task = _CURRENT.get()
    return task.text if task is not None else None


def stamped(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """*metadata* with the current task added to its origins, or ``None`` when off."""
    task = _CURRENT.get()
    if not enabled() or task is None:
        return None
    out = dict(metadata or {})
    keys = [t for t in (out.get(FIELD) or []) if isinstance(t, str)]
    if task.key not in keys:
        keys.append(task.key)
    out[FIELD] = keys
    texts = [t for t in (out.get(REQUESTS_FIELD) or []) if isinstance(t, str)]
    texts = [t for t in texts if t != task.text] + [task.text]
    out[REQUESTS_FIELD] = texts[-MAX_ORIGIN_REQUESTS:]
    return out


def _origins(row: Dict[str, Any]) -> tuple[List[str], List[str]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return [], []
    keys = [t for t in (metadata.get(FIELD) or []) if isinstance(t, str)]
    texts = [t for t in (metadata.get(REQUESTS_FIELD) or []) if isinstance(t, str)]
    return keys, texts


def _has_origins(row: Dict[str, Any]) -> bool:
    metadata = row.get("metadata")
    return isinstance(metadata, dict) and (
        FIELD in metadata or REQUESTS_FIELD in metadata
    )


def strip(row: Dict[str, Any]) -> Dict[str, Any]:
    """*row* without the origin fields in its metadata (a copy when it had any)."""
    if not _has_origins(row):
        return row
    out = dict(row)
    out["metadata"] = {
        k: v for k, v in row["metadata"].items() if k not in (FIELD, REQUESTS_FIELD)
    }
    return out


# How a kept outcome reads, by (accepted, source).
_OUTCOME_TEXT = {
    (True, CHECKER): "the checker accepted that session's answer",
    (False, CHECKER): "the checker did not accept that session's answer",
    (True, REVIEW): "that session's answer was confirmed, as judged by its review",
    (False, REVIEW): "that session's answer was rejected, as judged by its review",
}


def lesson_status(row: Dict[str, Any]) -> Optional[str]:
    """``UNIFY_LESSON_STATUS``: why a guidance *row* is unverified, or ``None`` when every session that wrote it was accepted.

    *row* carries the entry's origin as ``metadata``. Unverified when a
    session that wrote it was not accepted (checker) or rejected (review),
    else when one has no kept outcome or no request was recorded.
    """
    _, texts = _origins(row)
    if not texts:
        return "unverified: written in a session whose outcome is unknown"
    outcomes = [origin_outcome(text) for text in texts]
    if any(kept is not None and not kept[0] for kept in outcomes):
        return "unverified: written after a session whose answer was not accepted"
    if any(kept is None for kept in outcomes):
        return "unverified: written in a session whose outcome is unknown"
    return None


CALLS_KEPT = 3
"""``UNIFY_LISTING_USAGE``: the latest calls whose request each function keeps."""


def record_call(name: str, text: Optional[str]) -> None:
    """``UNIFY_LISTING_USAGE``: keep that stored function *name* ran under the request *text*.

    The latest :data:`CALLS_KEPT` per function, in the request log's
    ``function_calls`` table (created only by this switch). A call outside a
    keyed request keeps nothing; a log that cannot be written is skipped.
    """
    if not name or not text:
        return
    try:
        path = request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect_calls(path)) as conn, conn:
            conn.execute(
                "INSERT INTO function_calls (name, text_key) VALUES (?, ?)",
                (name, text_key(text)),
            )
            conn.execute(
                "DELETE FROM function_calls WHERE name = ? AND seq NOT IN (SELECT"
                " seq FROM function_calls WHERE name = ? ORDER BY seq DESC LIMIT ?)",
                (name, name, CALLS_KEPT),
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"function call not logged: {type(exc).__name__}: {exc}")


def _connect_calls(path: Path) -> sqlite3.Connection:
    conn = _connect_outcomes(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS function_calls (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, name TEXT NOT NULL, text_key TEXT NOT NULL)",
    )
    return conn


def usage_note(name: str, calls: Any) -> str:
    """``UNIFY_LISTING_USAGE``: ``used N×`` and how the sessions of its last recorded calls ended."""
    count = int(calls or 0)
    if count <= 0:
        return "not called yet"
    note = f"used {count}×"
    path = request_log_path()
    keys: List[str] = []
    kept: Dict[tuple[str, str], bool] = {}
    if path.exists():
        try:
            with closing(_connect_calls(path)) as conn:
                keys = [
                    key
                    for (key,) in conn.execute(
                        "SELECT text_key FROM function_calls WHERE name = ?"
                        " ORDER BY seq DESC LIMIT ?",
                        (name, CALLS_KEPT),
                    )
                ]
                kept = (
                    {
                        (key, source): bool(solved)
                        for key, source, solved in conn.execute(
                            "SELECT text_key, source, solved FROM request_outcomes"
                            f" WHERE text_key IN ({', '.join('?' for _ in keys)})",
                            keys,
                        )
                    }
                    if keys
                    else {}
                )
        except sqlite3.Error as exc:
            logger.warning(f"function calls not read: {type(exc).__name__}: {exc}")
            keys = []
    if not keys:
        return note + "; the sessions of its calls were not recorded"
    failed = unknown = 0
    for key in keys:
        verdict = kept.get((key, CHECKER), kept.get((key, REVIEW)))
        if verdict is None:
            unknown += 1
        elif not verdict:
            failed += 1
    parts = [f"{failed} ran in a session whose answer was not accepted"]
    if unknown:
        parts.append(f"{unknown} with the outcome unknown")
    return note + f"; of its last {len(keys)} recorded calls, " + ", ".join(parts)


ORIGIN_LINE = "_origin_line"
LESSON = "_lesson"
USAGE = "_usage"


def listing_notes(marker: "Marker", kind: str, row: Dict[str, Any]) -> Dict[str, str]:
    """The notes the listing switches add to a listed entry's line, by key.

    *row* is the stored entry with its origin as ``metadata`` (and, for a
    function, ``name`` and ``usage_calls``; for guidance, ``is_builtin``).
    :data:`ORIGIN_LINE` (``UNIFY_LISTING_PROVENANCE``, every entry),
    :data:`LESSON` (``UNIFY_LESSON_STATUS``, an unverified guidance entry
    that is not built in), :data:`USAGE` (``UNIFY_LISTING_USAGE``, a
    function).
    """
    notes: Dict[str, str] = {}
    if listing_provenance_enabled():
        notes[ORIGIN_LINE] = marker.origin_line(row, kind=kind)
    if kind == "guidance" and lesson_status_enabled() and not row.get("is_builtin"):
        status = lesson_status(row)
        if status:
            notes[LESSON] = status
    if kind == "function" and listing_usage_enabled():
        notes[USAGE] = usage_note(str(row.get("name") or ""), row.get("usage_calls"))
    return notes


class Marker:
    """Marks the search rows of functions stored while handling a similar request.

    *library* holds the rows of every function the search could return: their
    origin requests and the current request weight the tokens (with
    ``UNIFY_SIMILAR_REQUEST_CORPUS=stream``, the logged requests too).
    """

    def __init__(self, library: Sequence[Dict[str, Any]] = ()) -> None:
        self._task = _CURRENT.get() if enabled() else None
        self._library = library
        self._weights: Optional[Dict[str, float]] = None
        self._known: Optional[List[str]] = None

    def _known_texts(self) -> List[str]:
        """The requests the weights are taken over, besides the current one."""
        if self._known is None:
            texts = [text for row in self._library for text in _origins(row)[1]]
            # UNIFY_ENTRY_RECORD logs requests too, and weighs over them.
            if _stream_corpus() or _records_requests():
                texts = [*logged_requests(), *texts]
            self._known = texts
        return self._known

    def _token_weights(self, task: _Task) -> Dict[str, float]:
        if self._weights is None:
            self._weights = token_weights([*self._known_texts(), task.text])
        return self._weights

    def score(self, row: Dict[str, Any]) -> Optional[float]:
        """How close the current request is to the closest one *row* was stored from.

        1 for the same text; ``None`` with no current request or no origins.
        """
        task = self._task
        if task is None:
            return None
        keys, texts = _origins(row)
        if task.key in keys:
            return 1.0
        if not texts:
            return None
        weights = self._token_weights(task)
        return max(similarity(task.text, text, weights) for text in texts)

    def provenance(self, row: Dict[str, Any]) -> Optional[str]:
        """``UNIFY_ORIGIN_PROVENANCE``: why *row* is close to the current request, or ``None``.

        Names the whole identifiers the closest request it was stored from
        shares with the current one (or says it was this same request), and
        whether the checker accepted that session's answer when an outcome
        was kept. ``None`` while the switch is off, with no current request
        or origin, or with nothing to say.
        """
        task = self._task
        if task is None or not provenance_enabled():
            return None
        keys, texts = _origins(row)
        if task.key in keys:
            same, origin = True, task.text
        elif texts:
            weights = self._token_weights(task)
            origin = max(texts, key=lambda text: similarity(task.text, text, weights))
            same = False
        else:
            return None
        parts: List[str] = []
        if same:
            parts.append("stored while handling this same request")
        else:
            shared = shared_identifiers(task.text, origin, self._known_texts())
            if shared:
                parts.append(
                    "stored while handling a request that also named "
                    + " and ".join(f"`{word}`" for word in shared),
                )
        kept = origin_outcome(origin)
        if kept is not None:
            parts.append(_OUTCOME_TEXT[kept])
        return "; ".join(parts) or None

    def _closest_origin(self, row: Dict[str, Any]) -> Optional[tuple[bool, str]]:
        """``(same, text)``: this same request, else the origin closest to it; ``None`` without one."""
        task = self._task
        keys, texts = _origins(row)
        if task is not None and task.key in keys:
            return True, task.text
        if not texts:
            return None
        if task is None:
            return False, texts[-1]
        weights = self._token_weights(task)
        return False, max(texts, key=lambda text: similarity(task.text, text, weights))

    def origin_line(self, row: Dict[str, Any], *, kind: str = "function") -> str:
        """``UNIFY_LISTING_PROVENANCE``: what *row* was stored (written) for, and how that session ended.

        Always says something: whether the closest request it was recorded
        under is this same request, shares rare identifiers with it
        (:meth:`provenance`'s rule) or shares none, or that no request was
        recorded; then the kept outcome of that session (the checker's,
        else its review's) or that it is unknown.
        """
        verb = "written" if kind == "guidance" else "stored"
        closest = self._closest_origin(row)
        if closest is None:
            return "no request recorded; outcome unknown"
        same, origin = closest
        if same:
            where = f"{verb} while handling this same request"
        else:
            shared = (
                shared_identifiers(self._task.text, origin, self._known_texts())
                if self._task is not None
                else []
            )
            where = (
                f"{verb} while handling a request that also named "
                + " and ".join(f"`{word}`" for word in shared)
                if shared
                else f"{verb} while handling another request (no rare identifier "
                "in common)"
            )
        kept = origin_outcome(origin)
        outcome = (
            _OUTCOME_TEXT[kept]
            if kept is not None
            else "that session's outcome is unknown"
        )
        return f"{where}; {outcome}"

    def annotate(self, row: Dict[str, Any]) -> None:
        """Drop the origin fields from *row*; add ``similar_request`` when close enough.

        With ``UNIFY_ORIGIN_PROVENANCE`` a marked row also gets ``origin``
        (:meth:`provenance`) when there is something to say.
        """
        if not _has_origins(row):
            return
        score = self.score(row)
        why = (
            self.provenance(row)
            if score is not None and score >= SIMILAR_REQUEST_THRESHOLD
            else None
        )
        row["metadata"] = strip(row)["metadata"]
        if score is not None and score >= SIMILAR_REQUEST_THRESHOLD:
            row[MARK] = round(score, 2)
            if why:
                row[ORIGIN_MARK] = why
