"""D43: E = 100k recorded tokens under v2.1; the pass wall bound; the proxy journal path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.memory_v2.integration.consolidate import sol_settings
from unify.memory_v2.integration.switch import (
    sol_journal,
    v21_experience_budget,
    v21_pass_wall_s,
)
from unify.memory_v2.trigger import EXPERIENCE_BUDGET, EXPERIENCE_BUDGET_V21, Trigger


def test_v21_e_is_100k_and_v2_keeps_its_own():
    assert EXPERIENCE_BUDGET_V21 == 100_000 and EXPERIENCE_BUDGET == 150_000
    assert (
        sol_settings(SimpleNamespace(UNIFY_MEMORY_V2_E=150000)).experience_budget
        == 150000
    )
    on = SimpleNamespace(UNIFY_MEMORY_V2_E=150000, UNIFY_MEMORY_V21="on")
    assert sol_settings(on).experience_budget == 100000
    assert (
        sol_settings(
            SimpleNamespace(UNIFY_MEMORY_V21="on", UNIFY_MEMORY_V21_E=75000),
        ).experience_budget
        == 75000
    )
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V21_E"):
        v21_experience_budget(SimpleNamespace(UNIFY_MEMORY_V21_E="1e5"))
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V21_E"):
        v21_experience_budget(SimpleNamespace(UNIFY_MEMORY_V21_E=0))


def test_the_trigger_fires_at_e(tmp_path):
    from tests.memory_v2.test_batch_map import _ep
    from unify.memory_v2.evidence import EvidenceStore

    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(_ep(eid="e1"), "1" * 40)
    tokens = Trigger(ev, experience_budget=10**9).pending()[0][1]
    assert Trigger(ev, experience_budget=tokens).after_episode("e1")
    assert Trigger(ev, experience_budget=tokens + 1).after_episode("e1") == []


def test_wall_bound_and_journal():
    assert v21_pass_wall_s(SimpleNamespace()) == 2700
    assert v21_pass_wall_s(SimpleNamespace(UNIFY_MEMORY_V21_PASS_WALL_S="600")) == 600
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V21_PASS_WALL_S"):
        v21_pass_wall_s(SimpleNamespace(UNIFY_MEMORY_V21_PASS_WALL_S="-1"))
    assert sol_journal(SimpleNamespace()) is None
    assert (
        sol_journal(
            SimpleNamespace(UNIFY_MEMORY_V2_SOL_JOURNAL="/srv/proxy/costs.jsonl"),
        )
        == "/srv/proxy/costs.jsonl"
    )
    with pytest.raises(ValueError, match="absolute"):
        sol_journal(SimpleNamespace(UNIFY_MEMORY_V2_SOL_JOURNAL="costs.jsonl"))


def test_the_settings_load_with_their_defaults():
    from unify.settings import SETTINGS

    assert SETTINGS.UNIFY_MEMORY_V21_E == 100000
    assert SETTINGS.UNIFY_MEMORY_V21_PASS_WALL_S == 2700
    assert SETTINGS.UNIFY_MEMORY_V2_SOL_JOURNAL == ""


def test_the_v2_max_calls_map_still_parses():
    """The v2.1 integer parsers must not replace v2's per-effort validator (it is looked up at call time)."""
    from unify.memory_v2.integration.switch import sol_max_calls_map

    assert sol_max_calls_map("low:40,medium:60,high:80") == {
        "low": 40,
        "medium": 60,
        "high": 80,
    }
    with pytest.raises(ValueError):
        sol_max_calls_map("low:0,medium:60,high:80")
