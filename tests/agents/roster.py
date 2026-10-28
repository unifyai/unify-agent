"""The run record's roster as the tests read it: ``roster.json`` with each agent's effective state."""

from __future__ import annotations

from pathlib import Path

from unify.agents import run_record as rr


def read_roster(home: Path | str) -> dict:
    """``roster.json`` under *home*'s agents folder, each agent's state its effective one (an agent whose process is
    gone is ``ended``)."""
    doc = rr._load_roster(rr.agents_dir(home) / rr.ROSTER_FILE)
    for row in doc["agents"]:
        row["state"] = rr.effective_state(row)
    return doc
