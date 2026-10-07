"""What is left of the request records the removed request-memory switches kept.

The research switches that recorded which request a stored function or
guidance entry came from (``UNIFY_TASK_ORIGIN``, ``UNIFY_TRY_FIRST`` and the
listings, gate, evidence list and review records built on them) were removed
at the code freeze; they are recoverable from the tag ``pre-freeze-3bb760e54``.
Nothing records a request any more. Two things stay:

- :func:`strip`: a store written while those switches were on can still hold
  the origin fields in a function's ``metadata``; a library read never shows
  them, as before.
- The request log's outcome table and the current request, which
  :mod:`unify.function_manager.run_summary` (``UNIFY_FUNCTION_SUMMARY``)
  still reads. No request is keyed, so :func:`current_request` is ``None``.
  These go with that module.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

FIELD = "origin_tasks"
REQUESTS_FIELD = "origin_requests"

REQUEST_LOG_FILE = "request_log.sqlite"

CHECKER = "checker"
"""An outcome the environment's checker posted."""
REVIEW = "review"
"""An outcome the session's storage review judged from the conversation."""


def current_request() -> Optional[str]:
    """The current request's bounded text: always ``None``, since no request is keyed."""
    return None


def request_log_path() -> Path:
    """``<UNIFY_HOME>/request_log.sqlite``: the log of top-level requests."""
    from unify import db

    return db.store_home() / REQUEST_LOG_FILE


def _connect_outcomes(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS requests (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, key TEXT NOT NULL UNIQUE, text TEXT NOT NULL)",
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS request_outcomes (seq INTEGER PRIMARY KEY"
        " AUTOINCREMENT, text_key TEXT NOT NULL, source TEXT NOT NULL,"
        " solved INTEGER NOT NULL, UNIQUE (text_key, source))",
    )
    return conn


def text_key(text: str) -> str:
    """The key an outcome is kept under: 16 hex digits of the sha256 of a bounded copy."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


__all__ = [
    "CHECKER",
    "FIELD",
    "REQUESTS_FIELD",
    "REVIEW",
    "current_request",
    "request_log_path",
    "strip",
    "text_key",
]
