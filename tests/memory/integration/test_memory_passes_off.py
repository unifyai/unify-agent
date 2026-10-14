"""``UNIFY_MEMORY_PASSES=off``: the memory surface without the memory work.

The actor's memory surface is exactly as with passes on (its prompt section, its read-only library copy and the
names bound in its cells), and after a request nothing of memory runs: no pass (WRITE, CURATE, a pass a verdict makes
due) starts, no writer call is made, the episode-end fork does not start, and the library never changes. Every other
memory setting keeps its value. The episode is recorded as before, its meta saying ``"memory_passes": "off"``.

No model is called: turns, the fork and the pass worker are spies; the request runs on the real tracks.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.integration import test_consolidate as tc
from tests.memory.integration.test_episode_journal import (  # noqa: F401 (the home fixture)
    REQUEST,
    TURNS,
    _publish,
    _recorded,
    home,
)
from tests.memory.test_batch_map import _ep
from unify.memory.gitio import Repo
from unify.memory.integration import async_pass as ap
from unify.memory.integration import request as request_mod
from unify.memory.integration import consolidate, hooks, switch
from unify.memory.integration import fork as fork_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.settings import SETTINGS, ProductionSettings


def test_the_setting_is_on_by_default_and_takes_on_or_off():
    assert ProductionSettings.model_fields["UNIFY_MEMORY_PASSES"].default == ""
    assert switch.parse_memory_passes("") == "on" and switch.parse_memory_passes(" OFF ") == "off"
    assert ProductionSettings(UNIFY_MEMORY_PASSES="off").UNIFY_MEMORY_PASSES == "off"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_PASSES"):
        switch.parse_memory_passes("later")
    assert switch.memory_passes_on(SimpleNamespace()) is True
    assert switch.memory_passes_on(SimpleNamespace(UNIFY_MEMORY_PASSES="off")) is False


@pytest.mark.parametrize("library", ["off", "on"])
def test_a_due_pass_never_runs_and_no_turn_is_built(tmp_path, monkeypatch, library):
    made: list = []
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: made.append(1))
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    settings = tc._library_settings(library)
    settings.UNIFY_MEMORY_PASSES = "off"
    got: list[dict] = []
    out = asyncio.run(
        consolidate.run_due_passes(
            stores, "e1", sha, State(stores.paths.state), effort="low", settings=settings, emit=got.append
        ),
    )
    assert out == [] and made == [] and got == [] and tc._events(stores) == []
    # the same batch is due: with passes on, a pass starts and its writer turn is built
    fake = tc.FakeWriter()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    settings.UNIFY_MEMORY_PASSES = "on"
    asyncio.run(
        consolidate.run_due_passes(
            stores, "e1", sha, State(stores.paths.state), effort="low", settings=settings, emit=got.append
        ),
    )
    assert fake.made[:1] == [("openai/gpt-6-sol", "low")] and any(e.get("phase") == "start" for e in got)


def test_no_worker_starts_for_a_pass_or_a_capture(tmp_path):
    stores = consolidate.open_stores(Paths.under(tmp_path))
    stores.evidence.index_episode(_ep(eid="e1"), "1" * 40)
    settings = tc._library_settings("on")
    settings.UNIFY_MEMORY_PASSES = "off"
    calls: list = []
    ev = ap.maybe_spawn(
        stores,
        "e1",
        "1" * 40,
        effort="low",
        settings=settings,
        emit=calls.append,
        popen=lambda *a, **k: calls.append(a),
    )
    assert ev["phase"] == "passes_off" and calls == [] and ap.read_inflight(stores.paths) is None
    assert not consolidate.events_path(stores.paths).exists()


def _surface(run) -> dict:
    """What the actor is given of memory for *run*: its prompt section, its library copy and its cell names."""
    checkout = Path(run.paths.checkout)
    files = sorted(
        (str(p.relative_to(checkout)), p.read_bytes()) for p in checkout.rglob("*") if p.is_file()
    )
    return {
        "section": hooks.system_prompt("ROLE"),
        "files": files,
        "globals": sorted(hooks.worker_globals()),
        "mounts": len(hooks.worker_mounts()),
    }


def _finish(env, run) -> list[dict]:
    emitted: list[dict] = []
    with asyncio.Runner() as runner:
        runner.run(run.finish(None, progress=lambda _t: None, emit=emitted.append), context=env.ctx)
    return emitted


@pytest.mark.parametrize("passes", ["on", "off"])
def test_a_request_with_a_capture_runs_no_memory_work_when_off_and_keeps_its_surface(  # noqa: F811
    home,
    monkeypatch,
    passes,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_FORK", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_E", 1)
    work: list[str] = []
    monkeypatch.setattr(fork_mod, "start", lambda *a, **k: work.append("fork"))
    monkeypatch.setattr(ap, "maybe_spawn", lambda *a, **k: work.append("pass") or {"phase": "not_due"})
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: work.append("writer call"))
    # the surface with passes on, from a request that is then abandoned before it records anything
    probe = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        on_surface = _surface(probe)
    finally:
        home.ctx.run(probe._drop_journal)
        home.ctx.run(probe.abort)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_PASSES", passes)
    memory = Repo(home.paths.memory)
    before = memory.head()
    run = home.ctx.run(request_mod.RequestRun.begin, REQUEST)
    try:
        assert _surface(run) == on_surface  # the actor's memory surface does not depend on the switch
        _publish(home, TURNS)
        # the episode carries a verdict: a checked outcome the driver posted
        assert run.take_outcome({"solved": True})["accepted"] is True
        emitted = _finish(home, run)
    finally:
        home.ctx.run(run.abort)
    if passes == "off":
        assert work == []
        assert [e for e in emitted if e.get("type") == "consolidation"] == []
        assert not (Path(home.paths.state_dir) / "memory-staging").exists()
    else:
        assert work == ["fork", "pass"]  # the same request with passes on: the fork and the pass worker start
    assert memory.head() == before  # the library never changed
    repo, rel, ep = _recorded(home.paths, run.episode_id)
    meta = json.loads(repo.show("main", f"{rel}/meta.json"))
    assert ep.memory_passes == ("off" if passes == "off" else "")
    assert meta.get("memory_passes") == ("off" if passes == "off" else None)
    assert ("memory_passes" in meta) is (passes == "off")
