"""A writer pass per stored capture, in the library mode's worker (``UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on``).

Every stored capture starts a pass for its episode, whatever its outcome and whether it reproduced the value its
attempt handed back; the experience budget alone starts none; one pass covers at most
``UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS`` capture episodes, and the worker runs the next pass right after it, one
writer at a time. The worker's stand-ins are the verdict tests' (:class:`World`): no box runs and no model is called.
"""

from __future__ import annotations

import os

import pytest

from tests.memory.integration.test_verdict_passes import POSITIVE, World, _record, _settings, _settle
from unify.memory import captures
from unify.memory.integration import async_pass as ap
from unify.memory.integration import consolidate


def _each(per_pass=None, **kw):
    s = _settings(**kw)
    s.UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE = "on"
    if per_pass is not None:
        s.UNIFY_MEMORY_LIBRARY_CAPTURES_PER_PASS = per_pass
    return s


def _row(n, **kw):
    return {"capture_id": f"{n:016x}", **kw}


@pytest.fixture
def world(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "row",
    [
        {"outcome": "failed", "reproduces": "differs"},
        {"outcome": "failed", "reproduces": "reproduces"},
        {"outcome": "positive", "reproduces": "unknown"},
        {"outcome": "none", "reproduces": "differs"},
        POSITIVE,
    ],
)
def test_every_stored_capture_starts_a_pass_whatever_its_outcome(world, row):
    sha = _record(world.stores, "e1")
    world.settled["e1"] = _row(1, **row)
    world.worker("e1", sha, _each())
    [p] = world.passes
    assert (p["kind"], p["episodes"], p["verdicts"]) == ("capture", ["e1"], ["e1"])
    [start] = world.starts()
    assert start["pass_kind"] == "capture" and start["verdict_episodes"] == ["e1"]


def test_a_step_that_stored_nothing_starts_no_pass_and_the_budget_alone_starts_none(world):
    sha = _record(world.stores, "e1", chars=4000)
    world.settled["e1"] = {"capture_id": None, "safety": "no procedure"}
    ran, _ = world.worker("e1", sha, _each(e=1))  # E is reached, but nothing was captured
    assert world.passes == [] and ran == []


def test_captures_waiting_together_get_one_pass_each_in_one_worker(world):
    for i in range(3):
        _record(world.stores, f"e{i}")
        world.settled[f"e{i}"] = _row(i, outcome="none", reproduces="unknown")
    sha = _record(world.stores, "e3")  # recorded, not captured (yet)
    ran, slots = world.worker("e3", sha, _each())
    assert [(p["kind"], p["episodes"], p["verdicts"]) for p in world.passes] == [
        ("capture", [f"e{i}"], [f"e{i}"]) for i in range(3)
    ]
    assert world.most == 1 and len(ran) == 3
    assert slots == ["e3.p0", "e3.p2", "e3.p4", "e3.p6"]  # the last slot found nothing new due


def test_the_bound_sets_how_many_captures_one_pass_covers(world):
    for i in range(3):
        _record(world.stores, f"e{i}")
        world.settled[f"e{i}"] = _row(i, outcome="failed", reproduces="differs")
    sha = _record(world.stores, "e3")
    world.worker("e3", sha, _each(per_pass=2))
    assert [(p["episodes"], p["verdicts"]) for p in world.passes] == [
        (["e0", "e1"], ["e0", "e1"]),
        (["e2", "e3"], ["e2"]),  # e3, recorded with no capture, rides with the pass before the bound is reached
    ]


def test_off_the_built_trigger_runs_unchanged(world):
    sha = _record(world.stores, "e1")
    world.settled["e1"] = _row(1, outcome="failed", reproduces="differs")
    world.worker("e1", sha, _settings())
    assert world.passes == []  # not a verdict, and E not reached: no pass, as built
    assert captures.episode_rows(world.store)["e1"]["outcome"] == "failed"


def test_a_request_end_starts_the_pass_a_stored_capture_made_due(world):
    stores = world.stores
    spawned = []

    class _Proc:
        pid = os.getpid()  # a live process, so the in-flight record has a start time

    def popen(argv, **kw):
        spawned.append(argv)
        return _Proc()

    sha = _record(stores, "e1")
    _settle(stores, "e1")
    captures._event(world.store, {"episode": "e1", "safety": "ok", **_row(1, outcome="failed", reproduces="differs")})
    ev = ap.maybe_spawn(stores, "e1", sha, effort="low", settings=_each(), emit=None, popen=popen)
    ap.clear_inflight(stores.paths, ev.get("pass_id") or "")
    assert ev["phase"] == "spawned" and ev["pass_kind"] == "capture" and ev["verdict_episodes"] == ["e1"]
    # as built, the same failed capture makes nothing due
    ev = ap.maybe_spawn(stores, "e1", sha, effort="low", settings=_settings(), emit=None, popen=popen)
    assert ev["phase"] != "spawned"


def test_the_switch_without_the_fork_is_refused_where_a_pass_would_start():
    with pytest.raises(Exception) as got:
        consolidate.writer_settings(_each(fork="off"))
    assert "UNIFY_MEMORY_LIBRARY_REVIEW_EACH_CAPTURE=on needs UNIFY_MEMORY_LIBRARY_FORK=on" in str(got.value)
