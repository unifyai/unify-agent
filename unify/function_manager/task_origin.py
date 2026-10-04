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
# UNIFY_SIMILAR_REQUEST_CORPUS=stream: the request log keeps this many
# distinct top-level requests, the latest.
REQUEST_LOG_SIZE = 200
REQUEST_LOG_FILE = "request_log.sqlite"

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
    if _stream_corpus():
        _log_request(task)
    return _CURRENT.set(task)


def _stream_corpus() -> bool:
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", "") == "stream"


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

    def _token_weights(self, task: _Task) -> Dict[str, float]:
        if self._weights is None:
            texts = [text for row in self._library for text in _origins(row)[1]]
            if _stream_corpus():
                texts = [*logged_requests(), *texts]
            self._weights = token_weights([*texts, task.text])
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

    def annotate(self, row: Dict[str, Any]) -> None:
        """Drop the origin fields from *row*; add ``similar_request`` when close enough."""
        if not _has_origins(row):
            return
        score = self.score(row)
        row["metadata"] = strip(row)["metadata"]
        if score is not None and score >= SIMILAR_REQUEST_THRESHOLD:
            row[MARK] = round(score, 2)
