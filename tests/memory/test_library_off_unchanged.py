"""With UNIFY_MEMORY_LIBRARY off, the library mode changes nothing: blocking passes, E = 150k, v2's brief and tools,
no async files, no served head and no main lock."""

from __future__ import annotations

import asyncio

from unify.memory.integration import consolidate
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State

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
    assert run.library_mode is False
    asyncio.run(run._consolidate("e1", "1" * 40, lambda s: None, None))
    assert len(calls) == 1 and "supervise" not in calls[0]
    for name in ASYNC_FILES:
        assert not (paths.state_dir / name).exists(), name


