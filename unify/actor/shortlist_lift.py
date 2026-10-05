"""``UNIFY_SHORTLIST_LIFT``: rank the shortlist by how much more an entry matches this request than recent ones.

The ungated shortlist (``UNIFY_LIBRARY_SHORTLIST``) ranks entries by the
cosine of their text with the request. In a stream whose requests share a
long preamble that cosine is mostly the preamble's: on the Continual-ARC
paper-protocol run (5 Oct) one generic guidance entry, written after a
failed instance, topped all 74 lists, and guidance was the first entry of
354 of 361 lists across 14 runs, because prose resembles the shared rules
text more than any function does. With the switch an entry is scored by its
*lift*: its similarity to this request less its mean similarity to the last
``k`` top-level requests of this home. An entry that resembles every
request scores about 0 and gives way to one that resembles this request
more than the others.

The similarity is :func:`~unify.common.semantic_search.rank_by_similarity`'s
(the cosine of each field with the request, averaged over the fields with
text and clipped at 0; a function's ``name`` and ``docstring``, a guidance
entry's ``title`` and ``content``), computed from cached vectors only: the
fields and this request were embedded by the shipped ranking a moment
before, and each earlier request at its own task start. Each top-level task
start keeps its request's text hash (the embedding cache's key) in the
request log, in the ``shortlist_requests`` table, which only this switch
creates. An earlier request equal to this one is left out; one whose vector
is not in the cache (another embedder, a deleted cache) is skipped. With
fewer than ``k`` earlier vectors the shipped ranking stands: a new stream
has no "recent requests" to compare with.

An entry is listed only when its lift reaches the floor: by default a value
per embedder, calibrated offline (research artifact
``overhaul-lanes/memory-surface-v1``) as the highest floor that keeps the
recall of same-scenario entries on AppWorld and ScienceWorld within a few
points of the shipped ranking. It drops entries that match this request
clearly less than they match recent ones; it is not a match threshold.
"""

from __future__ import annotations

import contextvars
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from unify.common import embeddings

logger = logging.getLogger(__name__)

TABLE = "shortlist_requests"
# The log keeps the latest this many top-level requests (as the request log does).
LOG_SIZE = 200
# Fields compared, per entry kind (rank_by_similarity's references).
FIELDS = {"function": ("name", "docstring"), "guidance": ("title", "content")}
# Default floor per embedder (``Embedder.model``), from the offline calibration
# (MEMORY-SURFACE, 5 Oct, SEMANTIC-V2's 4,450 benchmark task starts): the
# highest floor at which the same-task recall of AppWorld and ScienceWorld
# stays within 4 points of the shipped ranking at k = 4. Any higher floor
# loses ScienceWorld's same-task entries first (their lift is often below 0,
# because the stream's recent requests are the same task's earlier visits).
DEFAULT_FLOORS = {
    embeddings.OPENROUTER.model: -0.08,
    embeddings.LOCAL.model: -0.03,
}
# An embedder with no calibrated floor lists every entry the ranking scores.
FALLBACK_FLOOR = float("-inf")

_TOP_LEVEL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unify_shortlist_lift_task",
    default=False,
)


@dataclass(frozen=True)
class Spec:
    """``k`` earlier requests; ``floor`` ``None`` means the embedder's default."""

    k: int
    floor: Optional[float]


def spec() -> Optional[Spec]:
    """The switch's setting; ``None`` when off."""
    from unify.settings import SETTINGS

    parsed = SETTINGS.shortlist_lift()
    return None if parsed is None else Spec(*parsed)


def floor_for(model: str, chosen: Optional[float]) -> float:
    """*chosen*, else the calibrated floor of *model*, else no floor."""
    if chosen is not None:
        return chosen
    return DEFAULT_FLOORS.get(model, FALLBACK_FLOOR)


def require_prerequisites() -> None:
    """Refuse the switch where it could rank nothing.

    Raises :class:`ValueError`: without the shortlist there is no list, and
    a gated list is ranked by request, not by embedding.
    """
    from unify.settings import SETTINGS

    if spec() is None:
        return
    if not SETTINGS.UNIFY_LIBRARY_SHORTLIST:
        raise ValueError(
            "UNIFY_SHORTLIST_LIFT needs UNIFY_LIBRARY_SHORTLIST=1: it ranks "
            "the shortlist's entries.",
        )
    if SETTINGS.shortlist_gate_threshold() is not None:
        raise ValueError(
            "UNIFY_SHORTLIST_LIFT ranks the shortlist by embedding; with "
            "UNIFY_SHORTLIST_GATE the list is chosen by request similarity "
            "instead, so set one or the other.",
        )


def enter() -> Optional[contextvars.Token]:
    """Mark the current context as a top-level task; ``None`` inside one (a sub-agent)."""
    if spec() is None or _TOP_LEVEL.get():
        return None
    return _TOP_LEVEL.set(True)


def leave(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _TOP_LEVEL.reset(token)
    except ValueError:
        pass


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, model TEXT NOT NULL, text_hash TEXT NOT NULL,"
        " UNIQUE (model, text_hash))",
    )
    return conn


def log_request(text: str) -> None:
    """Keep *text*'s hash as the latest top-level request; the latest :data:`LOG_SIZE`.

    A request seen before moves to the end. A log that cannot be written is
    skipped with a warning (later lists then compare with fewer requests).
    """
    from unify.common import embeddings
    from unify.function_manager import task_origin

    if not text:
        return
    model = embeddings.embedder().model
    digest = embeddings.text_hash(text)
    try:
        path = task_origin.request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as conn, conn:
            conn.execute(
                f"DELETE FROM {TABLE} WHERE model = ? AND text_hash = ?",
                (model, digest),
            )
            conn.execute(
                f"INSERT INTO {TABLE} (model, text_hash) VALUES (?, ?)",
                (model, digest),
            )
            conn.execute(
                f"DELETE FROM {TABLE} WHERE seq NOT IN"
                f" (SELECT seq FROM {TABLE} ORDER BY seq DESC LIMIT ?)",
                (LOG_SIZE,),
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"shortlist request not logged: {type(exc).__name__}: {exc}")


def recent_hashes(model: str, exclude: str) -> List[str]:
    """The kept request hashes of *model*, newest first, *exclude* left out."""
    from unify.function_manager import task_origin

    path = task_origin.request_log_path()
    if not path.exists():
        return []
    try:
        with closing(_connect(path)) as conn:
            return [
                digest
                for (digest,) in conn.execute(
                    f"SELECT text_hash FROM {TABLE} WHERE model = ?"
                    " ORDER BY seq DESC",
                    (model,),
                )
                if digest != exclude
            ]
    except sqlite3.Error as exc:
        logger.warning(f"shortlist requests not read: {type(exc).__name__}: {exc}")
        return []


def _field_texts(kind: str, row: Dict[str, Any]) -> List[str]:
    return [
        str(row[field])
        for field in FIELDS.get(kind, ())
        if str(row.get(field) or "").strip()
    ]


def rank(
    rows: Sequence[tuple[str, Dict[str, Any]]],
    text: str,
    chosen: Spec,
    *,
    k_list: int,
) -> Optional[List[tuple[str, Dict[str, Any]]]]:
    """*rows* (``(kind, row)``, every candidate) ranked by lift, at most *k_list*; ``None`` to rank as shipped.

    ``None`` when fewer than ``chosen.k`` earlier request vectors are cached
    or this request's vector is not. Entries whose vectors are not all
    cached keep a baseline of 0 (their lift is their similarity). Each kept
    row is a copy carrying ``_lift``. Ties go to the more similar entry,
    then to the earlier row.
    """
    from unify.common import embeddings

    if not text or not rows:
        return None
    model = embeddings.embedder().model
    current = embeddings.text_hash(text)
    earlier = recent_hashes(model, current)
    texts = {
        embeddings.text_hash(field): field
        for kind, row in rows
        for field in _field_texts(kind, row)
    }
    vectors = embeddings.cached_vectors([current, *earlier[: chosen.k * 4], *texts])
    if current not in vectors:
        return None
    recent = [vectors[digest] for digest in earlier if digest in vectors][: chosen.k]
    if len(recent) < chosen.k:
        return None
    request = vectors[current]
    baseline_requests = np.stack(recent)
    floor = floor_for(model, chosen.floor)
    scored = []
    for order, (kind, row) in enumerate(rows):
        digests = [embeddings.text_hash(f) for f in _field_texts(kind, row)]
        if not digests:
            continue
        if all(d in vectors for d in digests):
            fields = np.stack([vectors[d] for d in digests])
            similarity = max(0.0, float((fields @ request).mean()))
            baseline = float(
                np.clip((baseline_requests @ fields.T).mean(axis=1), 0.0, None).mean(),
            )
        else:  # not embedded by the shipped ranking: its score as shipped
            similarity = float(row.get("_similarity") or 0.0)
            baseline = 0.0
        if similarity <= 0:
            continue
        lift = similarity - baseline
        if lift < floor:
            continue
        scored.append(
            (-lift, -similarity, order, kind, {**row, "_lift": round(lift, 4)}),
        )
    scored.sort(key=lambda item: item[:3])
    return [(kind, row) for *_, kind, row in scored[: max(k_list, 0)]]


__all__ = [
    "DEFAULT_FLOORS",
    "Spec",
    "enter",
    "floor_for",
    "leave",
    "log_request",
    "rank",
    "recent_hashes",
    "require_prerequisites",
    "spec",
]
