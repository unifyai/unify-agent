"""``UNIFY_MEMORY_WRITER_PASSES=off``: the captures without the writer.

In the library mode with the episode-end fork on, each settled fork's attempt is still captured (the host's capture
step) and the actor still reads and runs the captures, and nothing else runs after a request: no pass (WRITE, CURATE,
a pass a verdict makes due) starts, no writer setting is read, no writer call is made, and the library never changes.
A request's end starts a worker only to capture a settled fork. Without the library mode, the fork, or with
``UNIFY_MEMORY_PASSES=off`` the setting is refused before anything runs.

No model is called: the writer, the capture step and the fork are stand-ins; the request runs on the real tracks.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory.integration import test_memory_passes_off as mpo
from tests.memory.integration.test_episode_journal import (  # noqa: F401 (the home fixture)
    REQUEST,
    TURNS,
    _publish,
    home,
)
from tests.memory.integration.test_verdict_passes import (  # noqa: F401 (the world fixture)
    POSITIVE,
    _Proc,
    _record,
    _settings,
    _settle,
    world,
)
from unify.memory import captures
from unify.memory.gitio import Repo
from unify.memory.integration import async_pass as ap
from unify.memory.integration import consolidate, switch
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import request as request_mod
from unify.memory.integration.state import State
from unify.settings import SETTINGS, ProductionSettings


def _off(**kw):
    s = _settings(**kw)
    s.UNIFY_MEMORY_WRITER_PASSES = "off"
    return s


def test_the_setting_is_on_by_default_and_takes_on_or_off():
    assert ProductionSettings.model_fields["UNIFY_MEMORY_WRITER_PASSES"].default == ""
    assert (
        switch.parse_writer_passes("") == "on"
        and switch.parse_writer_passes(" OFF ") == "off"
    )
    assert (
        ProductionSettings(UNIFY_MEMORY_WRITER_PASSES="off").UNIFY_MEMORY_WRITER_PASSES
        == "off"
    )
    with pytest.raises(ValueError, match="UNIFY_MEMORY_WRITER_PASSES"):
        switch.parse_writer_passes("captures")
    assert switch.writer_passes_on(SimpleNamespace()) is True
    assert (
        switch.writer_passes_on(SimpleNamespace(UNIFY_MEMORY_WRITER_PASSES="off"))
        is False
    )


def test_off_is_refused_without_the_library_mode_the_fork_or_with_passes_off():
    assert (
        switch.profile_refusal(_off()) is None
    )  # the library mode, the fork, passes on: runs
    assert (
        switch.profile_refusal(_settings()) is None
    )  # on (the default) is never refused
    for name, value, named in (
        ("UNIFY_MEMORY_LIBRARY", "off", "UNIFY_MEMORY_LIBRARY=on"),
        ("UNIFY_MEMORY_LIBRARY_FORK", "off", "UNIFY_MEMORY_LIBRARY_FORK=on"),
        ("UNIFY_MEMORY_PASSES", "off", "UNIFY_MEMORY_PASSES not off"),
    ):
        s = _off()
        setattr(s, name, value)
        refusal = switch.profile_refusal(s)
        assert (
            refusal is not None
            and refusal.startswith("UNIFY_MEMORY_WRITER_PASSES=off")
            and named in refusal
        )
    # refused before any writer setting is read: the writer route refuses too
    s = _off(fork="off")
    with pytest.raises(switch.WriterRouteRefused, match="UNIFY_MEMORY_WRITER_PASSES"):
        consolidate.writer_settings(s)


def test_a_verdict_is_captured_and_no_pass_runs(world):  # noqa: F811
    sha = _record(world.stores, "e1")
    world.settled["e1"] = POSITIVE
    ran, slots = world.worker("e1", sha, _off())
    assert world.passes == [] and ran == [] and slots == ["e1.p0"]
    assert (
        captures.episode_rows(world.store)["e1"]["outcome"] == "positive"
    )  # captured as with the writer on
    assert not consolidate.events_path(
        world.stores.paths
    ).exists()  # no pass event of any kind
    # the same verdict with the writer on: its pass runs at once
    sha2 = _record(world.stores, "e2")
    world.settled["e2"] = POSITIVE
    world.worker("e2", sha2, _settings())
    assert [(p["kind"], p["verdicts"]) for p in world.passes] == [("verdict", ["e2"])]


def test_a_due_batch_never_runs_and_no_writer_setting_is_read(
    world,  # noqa: F811
    monkeypatch,
):
    read: list = []
    real = consolidate.writer_settings
    monkeypatch.setattr(
        consolidate, "writer_settings", lambda s: read.append(1) or real(s)
    )
    sha = _record(world.stores, "e1")
    out = asyncio.run(
        consolidate.run_due_passes(
            world.stores,
            "e1",
            sha,
            State(world.stores.paths.state),
            effort="low",
            settings=_off(e=1),
            emit=None,
        ),
    )
    assert out == [] and world.passes == [] and read == []
    # with the writer on the same batch is due and its pass runs
    asyncio.run(
        consolidate.run_due_passes(
            world.stores,
            "e1",
            sha,
            State(world.stores.paths.state),
            effort="low",
            settings=_settings(e=1),
            emit=None,
        ),
    )
    assert [p["kind"] for p in world.passes] == ["batched"] and read


def test_a_request_end_starts_a_worker_only_to_capture(world):  # noqa: F811
    stores, spawned = world.stores, []

    def popen(argv, **kw):
        spawned.append(argv)
        return _Proc()

    def end(eid, sha, settings):
        out = ap.maybe_spawn(
            stores, eid, sha, effort="low", settings=settings, emit=None, popen=popen
        )
        ap.clear_inflight(stores.paths, out.get("pass_id") or "")
        return out

    sha1 = _record(stores, "e1")
    # a batch is due (E=1) and nothing is settled: with the writer off no worker starts
    assert end("e1", sha1, _off(e=1))["phase"] == "not_due" and spawned == []
    assert (
        end("e1", sha1, _settings(e=1))["phase"] == "spawned" and len(spawned) == 1
    )  # on: the pass's worker
    _settle(stores, "e1")
    ev = end("e1", sha1, _off(e=1))
    assert (
        ev["phase"] == "spawned_for_capture"
        and ev["capture_episodes"] == ["e1"]
        and len(spawned) == 2
    )
    # captured as a verdict: with the writer on its pass would start; off, nothing is left to do
    captures._event(world.store, {"episode": "e1", "safety": "ok", **POSITIVE})
    assert end("e1", sha1, _off())["phase"] == "not_due" and len(spawned) == 2
    assert end("e1", sha1, _settings())["phase"] == "spawned" and len(spawned) == 3


def _capture_dir(state_dir) -> str:
    """One capture in the store, as the capture step leaves it."""
    cid = "0707294761b569b2"
    d = captures.store_dir(state_dir) / cid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text("def run(value):\n    return value\n")
    (d / "record.json").write_text(
        json.dumps(
            {
                "applies": True,
                "capture_id": cid,
                "cases": [],
                "deterministic": True,
                "episodes": ["e0"],
                "holds": True,
                "level": "unchecked",
                "outcome": "positive",
                "record_line": "",
                "recurrence": [],
                "reproduces": "reproduces",
                "same_as": [],
                "status": "on-trial",
                "status_basis": None,
                "status_reason": None,
                "status_rule": None,
                "status_text": "on-trial",
                "track": {},
                "uses": [],
            },
        ),
    )
    return cid


@pytest.mark.parametrize("writer", ["on", "off"])
def test_a_request_keeps_its_surface_and_its_captures_and_starts_the_fork_and_a_worker(  # noqa: F811
    home,  # noqa: F811
    monkeypatch,
    writer,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_FORK", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_E", 1)
    cid = _capture_dir(home.paths.state_dir)
    work: list[str] = []
    monkeypatch.setattr(fork_mod, "start", lambda *a, **k: work.append("fork"))
    monkeypatch.setattr(
        ap, "maybe_spawn", lambda *a, **k: work.append("worker") or {"phase": "not_due"}
    )
    monkeypatch.setattr(
        consolidate, "unillm_turn", lambda *a, **k: work.append("writer call")
    )
    probe = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        on_surface = mpo._surface(probe)
    finally:
        home.ctx.run(probe._drop_journal)
        home.ctx.run(probe.abort)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_WRITER_PASSES", writer)
    memory = Repo(home.paths.memory)
    before = memory.head()
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        surface = mpo._surface(run)
        assert (
            surface == on_surface
        )  # the actor's memory surface does not depend on the switch
        assert any(
            cid in path and path.endswith("procedure.py")
            for path, _ in surface["files"]
        )  # memory.run reads it
        _publish(home, TURNS)
        assert run.take_outcome({"solved": True})["accepted"] is True
        mpo._finish(home, run)
    finally:
        home.ctx.run(run.abort)
    # the fork starts (it settles into a capture), and the request's end starts the worker that captures it; no
    # writer call is made in the request either way
    assert work == ["fork", "worker"]
    assert memory.head() == before
