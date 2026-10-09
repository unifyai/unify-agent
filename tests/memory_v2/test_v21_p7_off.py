"""P7 Task 10: with UNIFY_MEMORY_V21 off, P7 changes nothing: blocking passes, E = 150k, v2's brief and tools,
no async files, no served head and no main lock."""

from __future__ import annotations

import asyncio

from tests.memory_v2.integration import test_consolidate as tc
from unify.memory_v2 import memory_writer as mw
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State
from unify.memory_v2.sol_pass import PassConfig, sol_tools

ASYNC_FILES = (
    "pass.lock",
    "pass-inflight.json",
    "pass-results.jsonl",
    "pass-worker.log",
    "pass-results.cursor",
)


def test_off_passes_block_and_create_no_async_files(tmp_path, monkeypatch):
    calls = []

    async def fake(*a, **k):
        calls.append(k)
        return []

    monkeypatch.setattr(consolidate, "run_due_passes", fake)
    paths = Paths.under(tmp_path)
    run = request_mod.RequestRun("r", paths)
    run.stores = consolidate.open_stores(paths)
    run.state = State.load(paths.state)
    run.effort = "low"
    assert run.v21 is False
    asyncio.run(run._consolidate("e1", "1" * 40, lambda s: None, None))
    assert len(calls) == 1 and "supervise" not in calls[0]
    for name in ASYNC_FILES:
        assert not (paths.state_dir / name).exists(), name


def test_off_settings_brief_and_tools_are_v2s():
    s = tc._settings()
    s.UNIFY_MEMORY_V2_E = ""
    s.UNIFY_MEMORY_V21_E = 5  # read only under v2.1
    assert consolidate.sol_settings(s).experience_budget == 150000
    assert PassConfig().deadline_s == 900.0 and PassConfig().v21 is False
    check = next(t for t in sol_tools() if t["function"]["name"] == "check")[
        "function"
    ]["description"]
    assert "covers and their channels" in check


def test_off_a_pass_leaves_no_served_head_lock_or_lag(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", tc.FakeSol())
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    (out,) = tc._run(stores, "e1", sha)
    assert not mw.lock_path(stores.memory).exists()
    assert mw.served_head(stores.memory) == stores.memory.head()  # no served ref: main
    assert stores.memory.run("for-each-ref", "refs/heads/served").strip() == ""
    events = tc._events(stores)
    assert all("served_lag_commits" not in e and "ended" not in e for e in events)
    ledger = consolidate.ledger_path(stores.paths).read_text()
    assert "unknown_usd_each" not in ledger
    for name in ASYNC_FILES:
        assert not (stores.paths.state_dir / name).exists(), name
