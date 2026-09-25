"""Symbolic: search_functions ranks by standing and drops the lapsed.

The ranking step is monkeypatched, so these tests pin the wiring —
overfetch, activation ordering, scope dropout, include_dormant, primitive
immunity, and the disabled master switch — without depending on what the
store holds.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest

import unify.function_manager.function_manager as fm_module
from tests.helpers import _handle_project
from unify.function_manager.activation import ActivationSettings
from unify.function_manager.function_manager import FunctionManager

NOW = datetime.now(timezone.utc)


def _FM(**kwargs: Any) -> FunctionManager:
    kwargs.setdefault("include_primitives", False)
    return FunctionManager(**kwargs)


def _row(
    name: str,
    *,
    score: float,
    days_dormant: float | None = None,
    calls: int = 0,
    is_primitive: bool = False,
    created_days_ago: float = 400,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "function_id": abs(hash(name)) % 10_000,
        "name": name,
        "_similarity": score,
        "is_primitive": is_primitive,
        "created_at": (NOW - timedelta(days=created_days_ago)).isoformat(),
    }
    if days_dormant is not None:
        stamps = [
            (NOW - timedelta(days=days_dormant + i)).isoformat()
            for i in range(max(1, calls))
        ]
        row["usage_calls"] = calls
        row["usage_recent_calls"] = stamps
        row["usage_last_called_at"] = stamps[0]
    return row


def _patch_search(monkeypatch: pytest.MonkeyPatch, rows: List[Dict[str, Any]]):
    calls: Dict[str, Any] = {}

    def _fake(candidates, terms, *, word_references, limit, id_field, backfill):
        calls["limit"] = limit
        return [dict(r) for r in rows]

    monkeypatch.setattr(fm_module, "rank_rows", _fake)
    # Search-hit bumps would write usage for rows that do not exist.
    monkeypatch.setattr(
        FunctionManager,
        "_bump_search_hits",
        lambda self, rows: None,
    )
    return calls


@_handle_project
def test_dormant_functions_drop_out_of_search(monkeypatch) -> None:
    fm = _FM()
    seen = _patch_search(
        monkeypatch,
        [
            _row("fresh", score=0.5, days_dormant=1, calls=8),
            _row("lapsed_daily", score=0.5, days_dormant=90, calls=8),
        ],
    )
    rows = fm.search_functions(query="anything", n=5)
    names = [r["name"] for r in rows]
    assert names == ["fresh"]
    # Overfetch: the ranking step was asked for more than n.
    assert seen["limit"] > 5


@_handle_project
def test_include_dormant_restores_the_lapsed(monkeypatch) -> None:
    fm = _FM()
    _patch_search(
        monkeypatch,
        [
            _row("fresh", score=0.5, days_dormant=1, calls=8),
            _row("lapsed_daily", score=0.5, days_dormant=90, calls=8),
        ],
    )
    rows = fm.search_functions(query="anything", n=5, include_dormant=True)
    assert {r["name"] for r in rows} == {"fresh", "lapsed_daily"}


@_handle_project
def test_standing_breaks_similarity_ties(monkeypatch) -> None:
    fm = _FM()
    _patch_search(
        monkeypatch,
        [
            _row("dusty", score=0.5, days_dormant=25, calls=1),
            _row("workhorse", score=0.5, days_dormant=1, calls=15),
        ],
    )
    rows = fm.search_functions(query="anything", n=5)
    assert [r["name"] for r in rows][0] == "workhorse"


@_handle_project
def test_similarity_still_dominates_standing(monkeypatch) -> None:
    fm = _FM()
    _patch_search(
        monkeypatch,
        [
            _row("strong_match_newborn", score=1.0, created_days_ago=0.5),
            _row("weak_match_workhorse", score=0.2, days_dormant=1, calls=20),
        ],
    )
    rows = fm.search_functions(query="anything", n=5)
    assert [r["name"] for r in rows][0] == "strong_match_newborn"
    # The querying model sees the components, not just the order.
    assert {"_similarity", "_standing", "_retrieval_score"} <= set(rows[0])
    assert rows[0]["_similarity"] > rows[1]["_similarity"]
    assert rows[0]["_standing"] < rows[1]["_standing"]


@_handle_project
def test_primitives_never_drop_out(monkeypatch) -> None:
    fm = _FM()
    _patch_search(
        monkeypatch,
        [_row("platform_method", score=0.5, is_primitive=True)],
    )
    rows = fm.search_functions(query="anything", n=5)
    assert [r["name"] for r in rows] == ["platform_method"]


@_handle_project
def test_master_switch_off_restores_pre_activation_behaviour(monkeypatch) -> None:
    fm = _FM()
    seen = _patch_search(
        monkeypatch,
        [
            _row("lapsed_daily", score=0.5, days_dormant=90, calls=8),
            _row("fresh", score=0.5, days_dormant=1, calls=8),
        ],
    )
    monkeypatch.setattr(
        FunctionManager,
        "activation_settings",
        property(lambda self: ActivationSettings(enabled=False)),
    )
    rows = fm.search_functions(query="anything", n=5)
    # Untouched: ranking order preserved, nothing filtered, no overfetch.
    assert [r["name"] for r in rows] == ["lapsed_daily", "fresh"]
    assert seen["limit"] == 5
