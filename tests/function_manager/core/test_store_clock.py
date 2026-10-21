"""Symbolic: every timestamp on a stored function reads the store's clock.

``created_at`` and the usage trace (``usage_last_called_at``,
``usage_recent_calls``) reach the model in search results, and activation
ranks by their age, so they all read ``db.utc_now``. The test session
freezes it, which keeps those fields, and with them the model's requests,
the same on every run. As shipped it is the current UTC time. No model is
called.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.helpers import _handle_project
from unify import db
from unify.function_manager.function_manager import FunctionManager

FROZEN = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
SOURCE = "def scale(x: int) -> int:\n    return x * 2\n"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _stored(name: str) -> dict:
    row = dict(db.query_one("SELECT * FROM functions WHERE name = ?", (name,)))
    row["usage_recent_calls"] = db.loads(row["usage_recent_calls"])
    return row


def test_the_test_session_freezes_the_store_clock():
    assert db.utc_now() == FROZEN
    assert db.now_iso() == FROZEN.isoformat()


@_handle_project
def test_a_stored_function_is_stamped_by_the_store_clock():
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=SOURCE)
    row = _stored("scale")
    assert row["created_at"] == FROZEN.isoformat()

    fm._note_function_use(row)
    used = _stored("scale")
    assert used["usage_calls"] == 1
    assert used["usage_last_called_at"] == FROZEN.isoformat()
    assert used["usage_recent_calls"] == [FROZEN.isoformat()]


@_handle_project
def test_search_ranks_by_age_on_the_store_clock(monkeypatch):
    """A function created a moment ago on the store's clock is fresh, not
    dormant: activation measures ages from the same clock."""
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=SOURCE)
    row = _stored("scale")
    ranked = fm._activation_rank([dict(row)], n=5, include_dormant=False)
    assert [r["name"] for r in ranked] == ["scale"]
    later = FROZEN + timedelta(days=3650)
    monkeypatch.setattr(db, "utc_now", lambda: later)
    assert fm._activation_rank([dict(row)], n=5, include_dormant=False) == []


def test_as_shipped_the_store_clock_is_the_current_utc_time():
    """In a fresh interpreter, without the test session's patches."""
    script = (
        "import json; from unify import db; "
        "print(json.dumps([db.utc_now().isoformat(), db.now_iso()]))"
    )
    before = datetime.now(timezone.utc)
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        timeout=120,
        check=True,
    ).stdout
    after = datetime.now(timezone.utc)
    stamps = json.loads(out.strip().splitlines()[-1])
    for stamp in stamps:
        when = datetime.fromisoformat(stamp)
        assert when.tzinfo is not None and when.utcoffset() == timedelta(0)
        assert stamp.endswith("+00:00")
        assert before <= when <= after
