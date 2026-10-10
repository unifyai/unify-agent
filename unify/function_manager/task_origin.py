"""What is left of the request records the removed request-memory switches kept.

The research switches that recorded which request a stored function or
guidance entry came from (``UNIFY_TASK_ORIGIN``, ``UNIFY_TRY_FIRST`` and the
listings, gate, evidence list and review records built on them) were removed
at the code freeze; they are recoverable from the tag ``pre-freeze-3bb760e54``.
Nothing records a request any more. A store written while those switches were
on can still hold the origin fields in a function's ``metadata``;
:func:`strip` keeps a library read from showing them, as before.
"""

from __future__ import annotations

from typing import Any, Dict

FIELD = "origin_tasks"
REQUESTS_FIELD = "origin_requests"


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


__all__ = ["FIELD", "REQUESTS_FIELD", "strip"]
