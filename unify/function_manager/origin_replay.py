"""``UNIFY_ORIGIN_REPLAY_STATUS``: show whether a stored function returns the answer of the request it was stored for.

With ``UNIFY_CAPTURE_ACCEPTED`` the storage review is shown the code cell
behind the session's answer, and every function it adds or updates is run
on that cell's values (:func:`unify.function_manager.origin_capture.record_answering_call`).
The review is told the result, but later sessions never see it. Offline
(research artifact memory-a-v1/capture-v1, ARC), a review-written function
that returned its origin request's answer was right on the job's next visit
37 of 45 times; one that returned something else 0 of 3, and one that could
not be run on those values 0 of 59.

With this switch the result is kept per function and source -- ``returns``,
``differs`` or ``not_run`` with the reason -- and shown as one line on the
function's evidence record and in search results. It informs; nothing is
refused, ranked or hidden. A function changed since it was run shows
nothing (the status belongs to the source that was run). Off: no table, no
line.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from contextlib import closing
from typing import Any, Optional

logger = logging.getLogger(__name__)

RETURNS = "returns"
DIFFERS = "differs"
NOT_RUN = "not_run"

_TABLE = """
CREATE TABLE IF NOT EXISTS origin_replay (
    function_id INTEGER NOT NULL,
    source_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    why TEXT,
    recorded_at TEXT,
    PRIMARY KEY (function_id, source_hash)
)
"""

_LINES = {
    RETURNS: "run on the values of the request it was stored for, it returns that request's answer",
    DIFFERS: "run on the values of the request it was stored for, it does not return that request's answer",
}


def enabled() -> bool:
    """``UNIFY_ORIGIN_REPLAY_STATUS`` with ``UNIFY_CAPTURE_ACCEPTED`` on."""
    from unify.settings import SETTINGS

    from . import origin_capture

    return bool(getattr(SETTINGS, "UNIFY_ORIGIN_REPLAY_STATUS", False)) and (
        origin_capture.enabled()
    )


def _hash(source: Any) -> str:
    return hashlib.sha256(str(source or "").encode()).hexdigest()


def _connect() -> sqlite3.Connection:
    from unify import db

    path = db.store_home() / "origin_replay.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(_TABLE)
    return conn


def record(function_id: Any, source: Any, status: str, why: str = "") -> None:
    """Keep the replay status of *source* (no-op while off); never raises."""
    if not enabled() or status not in (RETURNS, DIFFERS, NOT_RUN):
        return
    try:
        from unify import db

        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO origin_replay VALUES (?, ?, ?, ?, ?)",
                (int(function_id), _hash(source), status, why[:200], db.now_iso()),
            )
    except Exception as exc:  # noqa: BLE001 - a status must never break a store
        logger.warning("origin replay status not kept: %s: %s", type(exc).__name__, exc)


def line(function_id: Any, source: Any) -> Optional[str]:
    """The line for the function as stored now, or ``None`` (off, never run, or changed since)."""
    if not enabled() or function_id is None:
        return None
    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT status, why FROM origin_replay WHERE function_id = ? AND source_hash = ?",
                (int(function_id), _hash(source)),
            ).fetchone()
    except (sqlite3.Error, TypeError, ValueError) as exc:
        logger.warning("origin replay status not read: %s", exc)
        return None
    if row is None:
        return None
    status, why = row
    if status == NOT_RUN:
        return "not run on the values of the request it was stored for" + (
            f" ({why})" if why else ""
        )
    return _LINES.get(status)


__all__ = ["DIFFERS", "NOT_RUN", "RETURNS", "enabled", "line", "record"]
