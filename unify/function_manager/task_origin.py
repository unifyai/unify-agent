"""Which task a stored function came from (``UNIFY_TRY_FIRST``).

A top-level ``act()`` keys its task by its request (the session's first user
message): a short hash of the whitespace-collapsed text, and a bounded copy of
that text (at most 4,000 characters: all of it, or its first and last 2,000).
The task loop and its storage review inherit the key through the task context.
With the switch on, a function stored during that task records both in its
``metadata`` (``origin_tasks``: the hashes; ``origin_requests``: the copies of
the latest :data:`MAX_ORIGIN_REQUESTS` distinct requests), and a search from a
later task marks the function ``same_task: true`` when that task's request
matches one the function was stored from: the same hash, or a similarity of at
least :data:`SAME_TASK_THRESHOLD`. Neither field is ever shown in a library
result. Sub-agents inherit the key of the task they work for. With the switch
off nothing is recorded or marked.

The similarity tells instructions from instance data without knowing the
request's format. A request is reduced to its set of lower-cased alphanumeric
tokens, leaving out numbers and dimension-like runs (``21``, ``13x13``): the
grids, amounts and dates that change between visits to one task. Each token is
weighted by how rare it is among the requests known at the search (the origin
requests of every function in the library, and the current one): ``ln(N /
df)`` over those N distinct requests, so wording that every known request
shares (a stream's common preamble) weighs nothing and wording that names or
describes one task weighs most. The score is the weighted Jaccard index of the
two token sets: the weight of the tokens both have over the weight of the
tokens either has. Two requests with no weighted token apart from what all
known requests share score 1: they differ only in numbers and spacing.

Calibrated offline on recorded opening requests: return visits to one task
whose requests differ only in instance data score 1; differently worded
variants of one task score about 0.15-0.45 once the library holds requests of
a few other tasks, and different tasks mostly under 0.15. Different tasks that
share instance wording (the same requester's name and address) can score
0.2-0.6, so a mark is a hint to check, never proof.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

FIELD = "origin_tasks"
REQUESTS_FIELD = "origin_requests"
# Each stored function keeps the copies of this many distinct requests (the
# latest), so its metadata stays bounded however often it is overwritten.
MAX_ORIGIN_REQUESTS = 3
# Head and tail kept from a long request.
_HEAD = 2000
_TAIL = 2000
# A function is marked ``same_task`` from this similarity on.
SAME_TASK_THRESHOLD = 0.2

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
# Instance data: plain numbers and dimension-like runs (13x13, 3x4x5).
_NUMBERLIKE = re.compile(r"^\d+(?:x\d+)*$")


@dataclass(frozen=True)
class _Task:
    key: str
    text: str


_CURRENT: contextvars.ContextVar[Optional[_Task]] = contextvars.ContextVar(
    "unify_task_origin",
    default=None,
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_TRY_FIRST", False))


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


@lru_cache(maxsize=4096)
def tokens(text: str) -> frozenset[str]:
    """The tokens a request is compared by (numbers and dimension-like runs left out)."""
    return frozenset(
        token for token in _TOKEN.findall(text.lower()) if not _NUMBERLIKE.match(token)
    )


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
    return _CURRENT.set(_Task(key=key, text=bounded_text(request)))


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
    """Marks the search rows of functions stored from the current task.

    *library* holds the rows of every function the search could return: their
    origin requests and the current request weight the tokens.
    """

    def __init__(self, library: Sequence[Dict[str, Any]] = ()) -> None:
        self._task = _CURRENT.get() if enabled() else None
        self._library = library
        self._weights: Optional[Dict[str, float]] = None

    def _token_weights(self, task: _Task) -> Dict[str, float]:
        if self._weights is None:
            texts = [text for row in self._library for text in _origins(row)[1]]
            self._weights = token_weights([*texts, task.text])
        return self._weights

    def same_task(self, row: Dict[str, Any]) -> bool:
        task = self._task
        if task is None:
            return False
        keys, texts = _origins(row)
        if task.key in keys:
            return True
        if not texts:
            return False
        weights = self._token_weights(task)
        return any(
            similarity(task.text, text, weights) >= SAME_TASK_THRESHOLD
            for text in texts
        )

    def annotate(self, row: Dict[str, Any]) -> None:
        """Drop the origin fields from *row*; add ``same_task: true`` when it matches."""
        if not _has_origins(row):
            return
        matched = self.same_task(row)
        row["metadata"] = strip(row)["metadata"]
        if matched:
            row["same_task"] = True
