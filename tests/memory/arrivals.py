"""The signal arrival rows as the tests read them (``signal-arrivals.jsonl``, :mod:`unify.memory.integration.outcomes`)."""

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
