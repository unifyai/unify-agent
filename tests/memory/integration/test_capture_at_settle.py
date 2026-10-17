"""UNIFY_MEMORY_LIBRARY_CAPTURE_AT_SETTLE: an attempt is captured as soon as its fork settles, by the CLI that recorded
it, and a request that begins first captures any settled fork not captured yet (off by default)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from unify.memory.integration import consolidate, request, switch


def test_the_switch_is_off_by_default_and_needs_the_fork():
    assert switch.parse_library_capture_at_settle("") == "off"
    assert switch.parse_library_capture_at_settle("on") == "on"
    with pytest.raises(ValueError):
        switch.parse_library_capture_at_settle("maybe")
    assert not switch.library_capture_at_settle(SimpleNamespace())
    assert not switch.library_capture_at_settle(
        SimpleNamespace(UNIFY_MEMORY_LIBRARY_CAPTURE_AT_SETTLE="on", UNIFY_MEMORY_LIBRARY_FORK="off"),
    )
    assert switch.library_capture_at_settle(
        SimpleNamespace(UNIFY_MEMORY_LIBRARY_CAPTURE_AT_SETTLE="on", UNIFY_MEMORY_LIBRARY_FORK="on"),
    )


def test_capture_at_settle_captures_the_settled_forks_and_never_raises(monkeypatch):
    opened, seen, notes = [], [], []
    monkeypatch.setattr(consolidate, "open_stores", lambda paths, busy_timeout_s=None: opened.append((paths, busy_timeout_s)) or "stores")
    monkeypatch.setattr(consolidate, "capture_settled_forks", lambda stores, settings: seen.append(stores) or [{"episode": "e1"}])
    rows = consolidate.capture_at_settle("paths", SimpleNamespace(), 60.0, notes.append)
    assert rows == [{"episode": "e1"}] and opened == [("paths", 60.0)] and seen == ["stores"]
    assert notes == ["memory: 1 captured at settle"]

    def boom(*_a, **_k):
        raise OSError("store busy")

    monkeypatch.setattr(consolidate, "open_stores", boom)
    notes.clear()
    assert consolidate.capture_at_settle("paths", SimpleNamespace(), None, notes.append) == []
    assert notes == ["memory: capture at settle skipped (OSError)"]


def test_the_request_captures_after_the_fork_settles_and_sweeps_before_the_snapshot():
    finish = inspect.getsource(request.RequestRun.finish)
    assert finish.index("_fork.finish, fork, SETTINGS, progress") < finish.index("consolidate.capture_at_settle")
    assert "if library_capture_at_settle(SETTINGS):" in finish
    opening = inspect.getsource(request.RequestRun._open)
    sweep = opening.index("consolidate.capture_settled_forks(self.stores, SETTINGS)")
    assert opening.index("if library_capture_at_settle(SETTINGS):") < sweep < opening.index("apply_results(self.state, paths)")
