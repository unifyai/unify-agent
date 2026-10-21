"""Every library-mode end event carries how far ``served`` is behind memory ``main``, and a lag above 0
after publishing raises a ``served_lag`` warning."""

from __future__ import annotations


from tests.memory.integration import test_consolidate as tc
from unify.memory import memory_writer as mw
from unify.memory.integration import consolidate


def _land(mem, files):
    base = mem.head()
    with mem.temp_checkout(base) as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "consolidation pass px: x", {"Pass": "px"})
    return mw.land(mem, sha, base)


def test_v2_end_events_carry_no_lag(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", tc.FakeWriter())
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    tc._run(stores, "e1", sha)
    events = tc._events(stores)
    assert all("served_lag_commits" not in e for e in events)
    assert not any(e.get("phase") in ("served_lag", "published") for e in events)
