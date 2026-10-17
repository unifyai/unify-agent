"""Writer passes at the verdict, in the library mode's worker: one writer at a time.

A capture that is a verdict (it reproduces its handed-back value and nothing contradicted it: positive, or no
signal at all) starts a writer pass for its episode at
once. The worker runs its due pass, then, slot after slot, every pass a verdict made due meanwhile, each starting only
after the one before it ended: two writers never run side by side, and a verdict that comes due during a pass (a
batched one or another verdict's) waits for it and then runs. A request's end starts a worker for a verdict, or to
capture a settled fork (positive ones first; one with no signal too).

The writer is a stand-in that records when each pass starts and ends and how many passes run at once; the capture
step is a stand-in that writes the capture rows the real one would (no box runs). No model is called.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from decimal import Decimal

import pytest

from tests.memory.integration import test_consolidate as tc
from tests.memory.test_trigger import _big
from unify.memory import captures
from unify.memory import memory_writer
from unify.memory.episodes import EpisodeWriter
from unify.memory.integration import async_pass as ap
from unify.memory.integration import consolidate
from unify.memory.redact import Redactor
from unify.memory.signals import Signal, post_signal
from unify.memory.writer_pass import PassOutcome

POSITIVE = {"outcome": "positive", "reproduces": "reproduces"}


def _settings(e=10**9, fork="on", visible=None):
    s = tc._settings(e=1)
    s.UNIFY_MEMORY_LIBRARY, s.UNIFY_MEMORY_LIBRARY_E, s.UNIFY_MEMORY_LIBRARY_FORK = "on", e, fork
    if visible:
        s.UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE = visible
    return s


def _record(stores, eid, chars=400):
    """An episode recorded and indexed, its experience about *chars* / 2 tokens."""
    ep = _big(eid, chars)
    ep.started_at = "2026-10-14T01:00:00Z"
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    return sha


class World:
    """The worker's stand-ins: the capture step writes the rows of the episodes whose forks have settled; the writer
    records each pass (when it started and ended, what it covered) and how many ran at once."""

    def __init__(self, monkeypatch, tmp_path):
        self.stores = tc._stores(tmp_path)
        self.store = captures.store_dir(self.stores.paths.state_dir)
        self.settled: dict[str, dict] = {}  # forks settled, with the row their capture gives
        self.passes: list[dict] = []
        self.during: dict[int, object] = {}  # pass number -> what happens while it runs
        self.active = 0
        self.most = 0
        world = self

        def capture(stores, settings):
            done = captures.episode_rows(world.store)
            rows = []
            for eid, row in list(world.settled.items()):
                if eid not in done:
                    captures._event(world.store, {"episode": eid, "safety": "ok", **row})
                    rows.append(row)
            return rows

        class Writer:
            def __init__(self, *a, **k):
                pass

            async def run(self, req, pass_id):
                world.active += 1
                world.most = max(world.most, world.active)
                row = {
                    "pass_id": pass_id,
                    "kind": req.kind,
                    "episodes": list(req.episodes),
                    "verdicts": list(req.verdicts),
                    "start": time.monotonic(),
                }
                world.passes.append(row)
                happen = world.during.pop(len(world.passes), None)
                try:
                    if happen is not None:
                        happen()
                    await asyncio.sleep(0.05)
                    world.stores.evidence.record_pass(
                        {"pass_id": pass_id, "kind": req.kind, "passed": 0, "reasons": "[]", "usd": "0"},
                    )
                    return PassOutcome(pass_id, False, None, "0", 0, ["nothing to merge"])
                finally:
                    row["end"] = time.monotonic()
                    world.active -= 1

            def transcript(self, pass_id):
                return []

        monkeypatch.setattr(consolidate, "capture_settled_forks", capture)
        monkeypatch.setattr(consolidate, "WriterPass", Writer)
        monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: None)
        monkeypatch.setattr(consolidate, "_curate_state", lambda stores, lookup, *rules: None)
        monkeypatch.setattr(consolidate, "after_passes", lambda *a, **k: None)
        monkeypatch.setattr(memory_writer, "publish", lambda *a, **k: None)

    def worker(self, eid, sha, settings):
        """One worker, as the pass worker runs it: its slots, each under its own supervisor."""
        slots = []

        async def go():
            return await consolidate.run_slots(
                self.stores,
                eid,
                sha,
                ap.StateView.load(self.stores.paths.state),
                effort="low",
                settings=settings,
                supervisor=lambda k: ap.Supervisor(30.0, asyncio.Event(), models=("openai/gpt-6-sol",)),
                on_slot=lambda k, pass_id: slots.append(pass_id),
            )

        return asyncio.run(go()), slots

    def starts(self):
        lines = consolidate.events_path(self.stores.paths).read_text().splitlines()
        return [e for e in map(json.loads, lines) if e.get("phase") == "start"]


@pytest.fixture
def world(monkeypatch, tmp_path):
    return World(monkeypatch, tmp_path)


def test_a_verdict_starts_a_pass_for_its_episode_at_once(world):
    sha = _record(world.stores, "e1")
    world.settled["e1"] = POSITIVE
    ran, _ = world.worker("e1", sha, _settings())
    [p] = world.passes
    assert (p["kind"], p["episodes"], p["verdicts"], p["pass_id"]) == ("verdict", ["e1"], ["e1"], "e1.p0")
    [start] = world.starts()
    assert start["pass_kind"] == "verdict" and start["verdict_episodes"] == ["e1"] and start["episodes"] == ["e1"]
    # the batched pass's budget: E x the rate (at low effort, scale 1); the run guard books it
    cfg = consolidate.writer_settings(_settings())
    budget = Decimal(cfg.experience_budget) * cfg.usd_per_token * cfg.effort_scale["low"]
    assert start["cap_usd"] == consolidate._usd(budget) and start["experience_budget"] == 10**9
    assert len(ran) == 1


@pytest.mark.parametrize(
    "row",
    [
        {"outcome": "failed", "reproduces": "differs"},
        {"outcome": "failed", "reproduces": "reproduces"},
        {"outcome": "positive", "reproduces": "unknown"},
        {"outcome": "none", "reproduces": "differs"},
    ],
)
def test_a_capture_that_failed_or_did_not_reproduce_starts_no_pass(world, row):
    sha = _record(world.stores, "e1")
    world.settled["e1"] = row
    ran, slots = world.worker("e1", sha, _settings())
    assert world.passes == [] and ran == [] and slots == ["e1.p0"]
    assert captures.episode_rows(world.store)["e1"]["outcome"] == row["outcome"]  # captured all the same


def test_a_reproducing_capture_with_no_signal_starts_a_pass(world):
    # an environment with no reward or feedback: the capture reproduces and nothing contradicts it
    sha = _record(world.stores, "e1")
    world.settled["e1"] = {"outcome": "none", "reproduces": "reproduces"}
    world.worker("e1", sha, _settings())
    assert [(p["kind"], p["verdicts"]) for p in world.passes] == [("verdict", ["e1"])]


def test_without_a_verdict_the_batched_pass_runs_as_before(world):
    sha = _record(world.stores, "e1")
    ran, _ = world.worker("e1", sha, _settings(e=1))
    [p] = world.passes
    assert (p["kind"], p["episodes"], p["verdicts"]) == ("batched", ["e1"], [])
    [start] = world.starts()
    assert "pass_kind" not in start and "verdict_episodes" not in start
    assert world.worker("e1", sha, _settings(e=1))[0] == []  # covered: nothing is due again


def test_episodes_in_a_row_each_get_their_pass_and_nothing_caps_them(world):
    for i in range(6):
        eid = f"e{i}"
        sha = _record(world.stores, eid)
        world.settled[eid] = POSITIVE
        world.worker(eid, sha, _settings())
    assert [(p["kind"], p["episodes"], p["verdicts"]) for p in world.passes] == [
        ("verdict", [f"e{i}"], [f"e{i}"]) for i in range(6)
    ]


def test_two_verdicts_close_together_run_one_after_the_other_never_side_by_side(world):
    sha = _record(world.stores, "e1")
    world.settled["e1"] = POSITIVE

    def second_verdict():  # while e1's pass runs, e2 is recorded and its fork settles with a verdict
        _record(world.stores, "e2")
        world.settled["e2"] = POSITIVE

    world.during[1] = second_verdict
    ran, slots = world.worker("e1", sha, _settings())
    one, two = world.passes
    assert (one["kind"], one["episodes"], one["verdicts"]) == ("verdict", ["e1"], ["e1"])
    assert (two["kind"], two["episodes"], two["verdicts"]) == ("verdict", ["e2"], ["e2"])
    assert two["start"] >= one["end"] and world.most == 1  # never two writers at once
    assert (one["pass_id"], two["pass_id"]) == ("e1.p0", "e1.p2") and slots == ["e1.p0", "e1.p2", "e1.p4"]
    assert len(ran) == 2


def test_a_verdict_during_a_batched_pass_waits_for_it(world):
    sha = _record(world.stores, "e1", chars=4000)
    e = world.stores.evidence.experience_of("e1")[0]  # e1 alone reaches E; e2 alone does not

    def verdict():
        _record(world.stores, "e2", chars=40)
        world.settled["e2"] = POSITIVE

    world.during[1] = verdict
    world.worker("e1", sha, _settings(e=e))
    batched, then = world.passes
    assert (batched["kind"], batched["episodes"]) == ("batched", ["e1"])
    assert (then["kind"], then["episodes"], then["verdicts"]) == ("verdict", ["e2"], ["e2"])
    assert then["start"] >= batched["end"] and world.most == 1


def test_a_verdict_a_worker_did_not_reach_stays_due_for_the_next(world):
    sha = _record(world.stores, "e1")
    captures._event(world.store, {"episode": "e1", "safety": "ok", **POSITIVE})
    settings = _settings()
    # a worker holds the lock (another pass runs): the request's end starts nothing, and the verdict stays due
    held = ap.try_lock(ap.lock_path(world.stores.paths))
    try:
        assert ap.maybe_spawn(world.stores, "e1", sha, effort="low", settings=settings, emit=None)["phase"] == "busy"
    finally:
        os.close(held)
    world.worker("e1", sha, settings)
    [p] = world.passes
    assert p["verdicts"] == ["e1"]


def _settle(stores, eid):
    d = stores.paths.state_dir / "memory-staging" / eid
    d.mkdir(parents=True, exist_ok=True)
    (d / "fork.json").write_text(json.dumps({"status": "ok"}))


def _seen_pass(stores, eid, sha, label):
    post_signal(
        Signal(f"{eid}.checker.1", eid, "checker", label, "2026-10-14T01:00:01Z", refers_to=eid, visible_to_actor=True),
        stores.episodes,
        sha,
        stores.evidence,
    )


class _Proc:
    pid = os.getpid()  # a live process, so the record has a start time


def test_a_request_end_starts_a_worker_for_a_verdict_or_to_capture_a_settled_fork(world):
    stores = world.stores
    settings = _settings(visible="on")
    spawned = []

    def popen(argv, **kw):
        spawned.append(argv)
        return _Proc()

    def end(eid, sha):
        out = ap.maybe_spawn(stores, eid, sha, effort="low", settings=settings, emit=None, popen=popen)
        ap.clear_inflight(stores.paths, out.get("pass_id") or "")
        return out

    sha1 = _record(stores, "e1")
    assert end("e1", sha1)["phase"] == "not_due" and spawned == []  # nothing settled, nothing due
    _settle(stores, "e1")
    # settled with no signal at all: still a candidate, so it is captured and may make a verdict
    assert consolidate.verdict_candidates(stores, settings) == ["e1"]
    ev = end("e1", sha1)
    assert ev["phase"] == "spawned_for_capture" and ev["capture_episodes"] == ["e1"] and len(spawned) == 1
    # once captured as a verdict, a request's end starts the verdict pass itself
    captures._event(world.store, {"episode": "e1", "safety": "ok", **POSITIVE})
    assert consolidate.verdict_candidates(stores, settings) == []
    ev = end("e1", sha1)
    assert ev["phase"] == "spawned" and ev["pass_kind"] == "verdict" and ev["verdict_episodes"] == ["e1"]
    # positive first, then no signal, then contradicted (captured as failed: it makes no pass)
    sha2 = _record(stores, "e2")
    _settle(stores, "e2")
    _seen_pass(stores, "e2", sha2, "fail")
    _record(stores, "e3")
    _settle(stores, "e3")
    sha4 = _record(stores, "e4")
    _settle(stores, "e4")
    _seen_pass(stores, "e4", sha4, "pass")
    assert consolidate.verdict_candidates(stores, settings) == ["e4", "e3", "e2"]
    # with the fork off there is nothing to capture
    assert consolidate.verdict_candidates(stores, _settings(fork="off", visible="on")) == []


def test_a_waiting_request_follows_the_workers_next_slot(tmp_path):
    """Under UNIFY_MEMORY_LIBRARY_WAIT_SLOT the request waits for the worker; a verdict slot the worker starts after
    its first gets its own bound, so the wait never cuts a queued pass short."""
    from dataclasses import replace

    from unify.memory.integration.paths import Paths

    paths = Paths.under(tmp_path)
    now = [1000.0]
    me = os.getpid()
    first = ap.InFlight("e1.p0", me, os.getpgid(0), ap.proc_start(me) or "", now[0], now[0] + 10, "e1")
    ap.write_inflight(paths, first)
    bound = ap.failsafe_s(60, passes=ap.WORKER_PASSES) + ap.FAILSAFE_EXTRA_S
    sleeps: list[float] = []

    def sleep(_s):
        sleeps.append(now[0])
        if len(sleeps) == 1:  # near the first slot's bound, the worker starts its next slot
            now[0] += bound - 5
            ap.write_inflight(paths, replace(first, pass_id="e1.p2", started_at=now[0], deadline_at=now[0] + 10))
        else:
            now[0] += 10

    drained: list[float] = []
    out = ap.wait_slot(
        paths,
        60,
        clock=lambda: now[0],
        sleep=sleep,
        drain_fn=lambda p, wait_s: drained.append(now[0]) or {"terminated": True},
    )
    assert out["ended"] == "drained"
    assert drained[0] >= 1000 + 2 * bound - 5  # the second slot's own bound, not the first's
