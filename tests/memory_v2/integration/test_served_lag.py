"""P7 Amendment C: every v2.1 end event carries how far ``served`` is behind memory ``main``, and a lag above 0
after publishing raises a ``served_lag`` warning."""

from __future__ import annotations

import asyncio

from tests.memory_v2.integration import test_consolidate as tc
from unify.memory_v2 import memory_writer as mw
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration.state import State


def _land(mem, files):
    base = mem.head()
    with mem.temp_checkout(base) as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "consolidation pass px: x", {"Pass": "px"})
    return mw.land(mem, sha, base)


def test_after_passes_failing_twice_leaves_a_lag_of_two_and_warns_twice(
    tmp_path,
    monkeypatch,
):
    def boom(*a, **k):
        raise RuntimeError("records not written")

    monkeypatch.setattr(consolidate, "unillm_turn", tc.FakeSol())
    # P5's after_passes (absent before integration): it fails, so no landed commit gets its item records
    monkeypatch.setattr(consolidate, "after_passes", boom, raising=False)
    stores = tc._stores(tmp_path)
    for i, eid in enumerate(("e1", "e2")):
        _land(stores.memory, {f"memory/x{i}.py": "X = 1\n"})
        sha, _ = tc._record(stores, eid, minute=i)
        asyncio.run(
            consolidate.run_due_passes(
                stores,
                eid,
                sha,
                State(stores.paths.state),
                effort="low",
                settings=tc._v21_settings("on"),
                emit=None,
            ),
        )
    events = tc._events(stores)
    ends = [e for e in events if e.get("phase") == "end"]
    assert [e["served_lag_commits"] for e in ends] == [1, 2]
    warnings = [e for e in events if e.get("phase") == "served_lag"]
    assert [e["served_lag_commits"] for e in warnings] == [1, 2]
    assert mw.served_lag(stores.memory) == 2


def test_v2_end_events_carry_no_lag(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", tc.FakeSol())
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    tc._run(stores, "e1", sha)
    events = tc._events(stores)
    assert all("served_lag_commits" not in e for e in events)
    assert not any(e.get("phase") in ("served_lag", "published") for e in events)
