"""The signal arrival rows (``signal-arrivals.jsonl``, :mod:`unify.memory.integration.outcomes`) and the evidence store's
regime changes, as the tests read them."""

from __future__ import annotations

import json
from pathlib import Path

from unify.memory.integration.outcomes import ARRIVALS_FILE


def read_arrivals(state_dir: Path | str) -> list[dict]:
    """Every arrival row, oldest first; a torn or malformed line is skipped."""
    try:
        text = (Path(state_dir) / ARRIVALS_FILE).read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("v") == 1:
            out.append(row)
    return out


def regime_changes(evidence, eid: str | None = None) -> list[dict]:
    """Every regime change the evidence store logged (only *eid*'s when given), oldest first."""
    made = evidence.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='regime_changes'").fetchone()
    if not made:
        return []
    q = "SELECT episode_id, at, before, after, basis, signal_id FROM regime_changes"
    args: tuple = ()
    if eid is not None:
        q, args = q + " WHERE episode_id=?", (eid,)
    keys = ("episode_id", "at", "before", "after", "basis", "signal_id")
    return [dict(zip(keys, r)) for r in evidence.db.execute(q + " ORDER BY seq", args).fetchall()]
