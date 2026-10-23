"""The request's memory run and its CLI wiring (integration Task 25, online build).

RequestRun is driven without an actor: ``begin``, a hand-written transcript, then ``finish``. The modules
it drives (the work-tree capture, trajectory, consolidation and cost) are recording fakes at their fixed
interfaces (``fake_tracks``), so what is checked is the lifecycle: the order, what each call is given, the
cleanup, and that only pass/fail of an outcome is kept.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.integration import fake_tracks
from tests.memory.integration.test_checkout import _seed_library
from unify import sandbox
from unify.memory.integration import hooks
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import acquire_lock, release_lock
from unify.settings import SETTINGS

SENTINEL = "SENTINEL-7f3a"
#: rev 2: the episode's fork is on whenever memory is; with no fork lane configured (as here) it is refused, and finish
#: says so first
FORK_REFUSED = "memory: fork refused: no fork lane"


@pytest.fixture
def mv2(monkeypatch, tmp_path):
    """A ``UNIFY_HOME`` with a seeded memory, the switch on and the other tracks faked."""
    home = tmp_path / "unify"
    home.mkdir()
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    mem, sha = _seed_library(home)  # rev 2: the actor's export is the library layout
    fakes = fake_tracks.install(monkeypatch)
    # The run is opened and closed in one context, as the CLI's task does; the test's own context
    # never sees its context variables.
    return SimpleNamespace(
        home=home,
        paths=Paths.under(home),
        sha=sha,
        fakes=fakes,
        ctx=contextvars.copy_context(),
    )


def _begin(mv2, request: str = "hi"):
    return mv2.ctx.run(request_mod.RequestRun.begin, request)


def _abort(mv2, run) -> None:
    mv2.ctx.run(run.abort)


def _in_ctx(mv2, coro):
    with asyncio.Runner() as runner:
        return runner.run(coro, context=mv2.ctx)


def _transcript(run, *lines: dict) -> Path:
    path = run.paths.home / "transcripts" / f"{run.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = lines or (
        {"seq": 0, "type": "message", "role": "user", "content": run.request},
        {"seq": 1, "type": "cell"},
        {"seq": 2, "type": "message", "role": "assistant", "content": "done"},
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _lock_is_free(paths: Paths) -> bool:
    try:
        fd = acquire_lock(paths.lock, timeout_s=0.1)
    except TimeoutError:
        return False
    release_lock(fd)
    return True


def _finish(mv2, run, **kw):
    progress: list[str] = []
    emitted: list[dict] = []
    _in_ctx(
        mv2,
        run.finish(None, progress=progress.append, emit=emitted.append, **kw),
    )
    return progress, emitted


def _left_nothing(paths: Paths) -> None:
    assert not paths.checkout.exists()
    assert _lock_is_free(paths)
    assert request_mod.current() is None
    assert hooks.worker_paths() == [] and hooks.worker_mounts() == []


# ── begin ────────────────────────────────────────────────────────────────────


def test_a_stale_export_from_a_killed_process_is_replaced(mv2):
    mv2.paths.checkout.mkdir()
    (mv2.paths.checkout / "junk.py").write_text("x = 1\n")
    run = _begin(mv2, "hi")
    try:
        assert not (mv2.paths.checkout / "junk.py").exists()
        assert (mv2.paths.checkout / "memory/acct/users.py").exists()
    finally:
        _abort(mv2, run)


def test_begin_refuses_when_a_cell_could_read_harness_state(mv2, monkeypatch):
    real = sandbox.build_policy(fresh=True)

    class Leaky:
        workspace = real.workspace

        def readable_violation(self, path):
            return None if Path(path) == mv2.paths.episodes else ("x", "hidden")

    monkeypatch.setattr(sandbox, "build_policy", lambda fresh=False: Leaky())
    with pytest.raises(request_mod.MemoryUnavailable, match="episodes.git"):
        _begin(mv2, "hi")
    _left_nothing(mv2.paths)
    assert mv2.fakes.cost_active == []


def test_a_begin_failing_after_the_capture_began_aborts_it(mv2, monkeypatch):
    from unify import transcripts

    def broken(_eid):
        raise ValueError("no session")

    monkeypatch.setattr(transcripts, "resume_session", broken)
    with pytest.raises(ValueError, match="no session"):
        _begin(mv2, "hi")
    assert mv2.fakes.names()[-2:] == ["worktree.begin", "worktree.abort"]
    _left_nothing(mv2.paths)


def test_hooks_open_no_run_while_the_switch_is_off(mv2, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")  # rev 2: empty means on
    assert hooks.begin_request("hi") is None
    assert mv2.fakes.calls == [] and request_mod.current() is None


def test_hooks_open_the_run_under_the_switch(mv2):
    run = mv2.ctx.run(hooks.begin_request, "hi")
    try:
        assert isinstance(run, request_mod.RequestRun) and request_mod.current() is run
    finally:
        _abort(mv2, run)


# ── The writer's effort matches the actor's unless a mismatch ablation fixes it ────


@pytest.mark.parametrize(
    "effort,want",
    [("low", "low"), ("medium", "medium"), ("", "high")],
)
def test_the_effort_is_the_actors(mv2, monkeypatch, effort, want):
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", effort)
    assert request_mod.actor_effort() == want
    run = _begin(mv2, "hi")
    _transcript(run)
    _finish(mv2, run)
    # the run records the actor's effort, the effort its writer passes run at
    assert run.effort == want


# ── finish ───────────────────────────────────────────────────────────────────


def test_finish_records_the_episode_and_runs_the_passes(mv2):
    run = _begin(mv2, "Say hi to ada.")
    transcript = _transcript(run)
    progress, emitted = _finish(mv2, run)

    f = mv2.fakes
    order = [
        n
        for n in f.names()
        if n
        in (
            "read_jsonl",
            "worktree.finish",
            "assemble",
            "checker_signal",
            "request_end",
        )
    ]
    # rev 2: the hidden checker's verdict is made a signal (checker_signal) and posted with the inferred regime
    assert order == [
        "read_jsonl",
        "worktree.finish",
        "assemble",
        "checker_signal",
        "request_end",
    ]
    (path,), _ = f.of("read_jsonl")
    assert path == transcript
    (cells,), _ = f.of("worktree.finish")
    assert cells == [("cell", 1)]  # trajectory.timed_cells over the folded transcript
    (a_run, lines, ended_at), kw = f.of("assemble")
    assert a_run is run and [x["seq"] for x in lines] == [0, 1, 2]
    assert kw == {
        "extra_actions": [fake_tracks.WT_ACTION],
        "worktree_before": fake_tracks.WT_BEFORE,
        "worktree_after": fake_tracks.WT_AFTER,
        "worktree_diff": fake_tracks.WT_DIFF,
    }
    # one episode commit, indexed
    stores = f.of("open_stores")
    episodes = mv2.paths.episodes
    log = os.popen(f"git --git-dir {episodes} log --format=%s main").read()
    assert log.splitlines()[0] == f"episode {run.episode_id}"
    (_, eid, solved, _ts), _ = f.of("checker_signal")
    assert eid == run.episode_id and solved is None
    from unify.memory.evidence import EvidenceStore

    sha = EvidenceStore(mv2.paths.evidence).episode_ref(eid)[0]
    assert len(sha) == 40 and sha == os.popen(f"git --git-dir {episodes} rev-parse main").read().strip()
    # the writer service is made sure of, with the settings and an emitter; the request never waits for a pass
    (_, p_eid), kw = f.of("request_end")
    assert p_eid == eid and kw["settings"] is SETTINGS and kw["emit"] is not None
    assert mv2.paths.state.exists()
    # the service event reaches the CLI's emitter
    assert [(e["phase"], e["episode_id"]) for e in emitted] == [("service", eid)]
    assert progress == [FORK_REFUSED]
    assert f.cost_active == [True, False]
    _left_nothing(mv2.paths)
    assert not mv2.paths.errors.exists()


def test_the_worktree_redactor_is_built_at_finish(mv2, monkeypatch):
    """The capture's zero-argument redactor factory is called at finish and knows the environment's
    secrets, including one set after the request began."""
    run = _begin(mv2, "hi")
    assert mv2.fakes.redactors == []
    monkeypatch.setenv("FAKE_SERVICE_TOKEN", "env-secret-12345678")
    _transcript(run)
    _finish(mv2, run)
    (redactor,) = mv2.fakes.redactors
    assert "env-secret-12345678" not in redactor.text("x env-secret-12345678 y")


def test_without_jsonl_the_driver_gets_no_emitter(mv2):
    run = _begin(mv2, "hi")
    _transcript(run)
    _in_ctx(mv2, run.finish(None, progress=lambda _t: None))
    _, kw = mv2.fakes.of("request_end")
    assert kw["emit"] is None


def test_finish_never_raises_and_logs_a_failed_pass(mv2):
    mv2.fakes.passes_raise = RuntimeError("writer fell over")
    run = _begin(mv2, "hi")
    _transcript(run)
    progress, _ = _finish(mv2, run)
    errors = [json.loads(x) for x in mv2.paths.errors.read_text().splitlines()]
    assert [e["stage"] for e in errors] == ["passes"]
    assert errors[0]["episode_id"] == run.episode_id
    assert "RuntimeError" in errors[0]["error"]
    assert len(progress) == 2 and progress[0] == FORK_REFUSED and "memory v2" in progress[1]
    assert mv2.paths.state.exists()  # generations are kept even when the passes fail
    _left_nothing(mv2.paths)


def test_a_failed_episode_runs_no_pass(mv2):
    mv2.fakes.assemble_raise = ValueError("bad transcript")
    run = _begin(mv2, "hi")
    _transcript(run)
    progress, _ = _finish(mv2, run)
    assert "request_end" not in mv2.fakes.names()
    assert "checker_signal" not in mv2.fakes.names()
    errors = [json.loads(x) for x in mv2.paths.errors.read_text().splitlines()]
    assert [e["stage"] for e in errors] == ["episode"]
    assert progress
    _left_nothing(mv2.paths)


def test_a_missing_transcript_is_logged_not_raised(mv2):
    run = _begin(mv2, "hi")
    _finish(mv2, run)
    assert (
        mv2.fakes.names()[-1] == "worktree.abort"
    )  # the capture never reached its finish
    errors = [json.loads(x) for x in mv2.paths.errors.read_text().splitlines()]
    assert [e["stage"] for e in errors] == ["episode"]
    _left_nothing(mv2.paths)


def test_one_process_begins_one_run_for_memorys_spend_cap(mv2, monkeypatch):
    """The run's start (the spend cap's count) is written once per controller process, at its first memory request,
    never at each request or by the service."""
    from unify.memory import writer_service as ws

    monkeypatch.setattr(request_mod, "_RUNS_BEGUN", set())
    for text in ("one", "two"):
        run = _begin(mv2, text)
        _abort(mv2, run)
    rows = ws._rows(ws.service_dir(mv2.home) / ws.SPEND_FILE)
    assert [r["op"] for r in rows if r.get("op") == "run"] == ["run"]
    # the same home by another spelling (a link to it) is the same run: no second run row resets the count
    alias = mv2.home.parent / "alias"
    alias.symlink_to(mv2.home, target_is_directory=True)
    monkeypatch.setenv("UNIFY_HOME", str(alias))
    _abort(mv2, _begin(mv2, "three"))
    rows = ws._rows(ws.service_dir(mv2.home) / ws.SPEND_FILE)
    assert [r["op"] for r in rows if r.get("op") == "run"] == ["run"]


def test_a_run_row_that_cannot_be_written_never_fails_the_request(mv2, monkeypatch):
    """Fail safe on spend: the request goes on, a warning says so, and the count runs on from the previous run's row
    (it only counts more)."""
    from unify.memory import writer_service as ws

    def broken(home):
        raise OSError("disk full")

    monkeypatch.setattr(request_mod, "_RUNS_BEGUN", set())
    monkeypatch.setattr(ws, "begin_run", broken)
    warned: list = []
    monkeypatch.setattr(request_mod.logger, "warning", lambda msg, *args: warned.append(msg % args))
    _abort(mv2, _begin(mv2, "hi"))
    assert any("the run's start was not recorded (OSError)" in w for w in warned), warned
    assert not [r for r in ws._rows(ws.service_dir(mv2.home) / ws.SPEND_FILE) if r.get("op") == "run"]


def test_abort_records_nothing(mv2):
    run = _begin(mv2, "hi")
    head = os.popen(f"git --git-dir {mv2.paths.episodes} rev-parse main").read()
    _transcript(run)
    _abort(mv2, run)
    _abort(mv2, run)  # idempotent
    assert "assemble" not in mv2.fakes.names()
    assert "worktree.finish" not in mv2.fakes.names()
    assert mv2.fakes.names().count("worktree.abort") == 1
    assert os.popen(f"git --git-dir {mv2.paths.episodes} rev-parse main").read() == head
    _left_nothing(mv2.paths)


def test_abort_after_finish_changes_nothing(mv2):
    run = _begin(mv2, "hi")
    _transcript(run)
    _finish(mv2, run)
    calls = list(mv2.fakes.calls)
    assert "worktree.abort" not in mv2.fakes.names()  # finish ended the capture
    _abort(mv2, run)
    assert mv2.fakes.calls == calls
    _left_nothing(mv2.paths)


def _cell_lines(code: str, system: str, error: str | None = None) -> list[dict]:
    meta = {"duration_ms": 1, **({"error": error} if error else {})}
    return [
        {"seq": 0, "type": "system_prompt", "content": system},
        {
            "seq": 1,
            "type": "message",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {
                            "name": "execute_code",
                            "arguments": json.dumps({"code": code}),
                        },
                    },
                ],
            },
        },
        {
            "seq": 2,
            "type": "message",
            "message": {
                "role": "tool",
                "tool_call_id": "c0",
                "content": [{"type": "text", "text": json.dumps(meta)}],
            },
        },
    ]


# ── the outcome: pass/fail only ──────────────────────────────────────────────


def _scan(value) -> str:
    """Every string reachable from *value* (attributes, containers), for a sentinel scan; walked with an explicit
    stack, since a run's object graph is deeper than the recursion limit."""
    seen: set[int] = set()
    out: list[str] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if isinstance(v, (str, bytes)):
            out.append(v if isinstance(v, str) else v.decode("utf-8", "replace"))
        elif isinstance(v, dict):
            for k, x in v.items():
                stack += [k, x]
        elif isinstance(v, (list, tuple, set, frozenset)):
            stack += list(v)
        elif isinstance(v, types.ModuleType):
            continue  # code the run refers to (a store module), not data it holds
        elif hasattr(v, "__dict__") and not isinstance(v, type):
            stack.append(vars(v))
        else:
            out.append(repr(v))
    return " ".join(out)


def test_take_outcome_keeps_only_pass_fail(mv2):
    run = _begin(mv2, "hi")
    try:
        bad = run.take_outcome({"solved": "yes", "summary": SENTINEL})
        assert bad["type"] == "outcome" and bad["accepted"] is False
        assert SENTINEL not in json.dumps(bad) and bad["reason"]
        assert run.take_outcome("not an object")["accepted"] is False
        good = run.take_outcome(
            {
                "solved": False,
                "summary": SENTINEL,
                "checks": [{"name": SENTINEL, "passed": False, "reason": SENTINEL}],
            },
        )
        assert good == {
            "type": "outcome",
            "accepted": True,
            "solved": False,
            "checks": 1,
        }
        assert SENTINEL not in _scan(vars(run))
        _transcript(run)
        _finish(mv2, run)
    finally:
        _abort(mv2, run)
    (_, _, solved, _), _ = mv2.fakes.of("checker_signal")
    assert solved is False
    assert SENTINEL.encode() not in fake_tracks.dump_home(mv2.home)
    assert SENTINEL not in _scan(mv2.fakes.calls)


# ── the CLI ──────────────────────────────────────────────────────────────────


class _Handle:
    """A session that ends when stopped (persistent) or at once (one-shot)."""

    def __init__(self, *, persist: bool) -> None:
        self._stopped = asyncio.Event()
        if not persist:
            self._stopped.set()
        self.run_stats: dict = {}

    async def result(self):
        await self._stopped.wait()
        return "done"

    def done(self) -> bool:
        return self._stopped.is_set()

    async def stop(self, reason=None) -> None:
        self._stopped.set()

    async def next_notification(self):
        await asyncio.Event().wait()

    async def next_clarification(self):
        await asyncio.Event().wait()


class _Run:
    #: the request's episode (the CLI's ``episode`` line names it whenever a memory run is open)
    episode_id = "20261022T000000Z-ab12cd34"

    def __init__(self, log: list) -> None:
        self.log = log

    def take_outcome(self, raw):
        self.log.append(("take_outcome", raw))
        return {"type": "outcome", "accepted": True, "solved": True, "checks": 0}

    async def finish(self, handle, *, progress, emit=None, consolidate=True):
        self.log.append(("finish", handle, emit is not None))
        self.consolidate = consolidate
        if emit is not None:
            emit({"type": "consolidation", "phase": "start", "cap_usd": "0.1"})

    def abort(self) -> None:
        self.log.append(("abort",))


def _drive(monkeypatch, argv: list[str], stdin: bytes, run: _Run | None):
    from unify.cli import Act, _parse_args

    log = run.log if run is not None else []
    read_fd, write_fd = os.pipe()
    os.write(write_fd, stdin)
    os.close(write_fd)
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    args = _parse_args(argv)
    session = Act(args)
    handle = _Handle(persist=args.persist)

    class _Actor:
        async def act(self, request, **kw):
            log.append(("act", request))
            return handle

        async def close(self):
            pass

    async def _start():
        session._actor = _Actor()

    def _begin_request(request):
        log.append(("begin", request))
        return run

    monkeypatch.setattr(session, "start", _start)
    monkeypatch.setattr(hooks, "begin_request", _begin_request)
    out: list[dict] = []
    monkeypatch.setattr(session, "_emit", lambda **p: out.append(p))

    async def _go():
        try:
            return await asyncio.wait_for(session.run(args.request), 30)
        finally:
            await session.close()

    code = asyncio.run(_go())
    return code, out, log, handle


def test_cli_opens_the_run_before_the_actor_and_finishes_before_ended(monkeypatch):
    log: list = []
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"outcome": {"solved": true}}\n{"quit": true}\n',
        _Run(log),
    )
    assert code == 0
    assert [e[0] for e in log] == ["begin", "act", "take_outcome", "finish", "abort"]
    assert log[0] == ("begin", "Say hi") and log[3] == ("finish", handle, True)
    types_ = [o["type"] for o in out]
    assert types_ == ["episode", "outcome", "result", "consolidation", "ended"]
    assert out[0] == {"type": "episode", "episode_id": _Run.episode_id}
    assert out[1] == {"type": "outcome", "accepted": True, "solved": True, "checks": 0}


def test_cli_without_jsonl_finishes_with_no_emitter(monkeypatch):
    log: list = []
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--quiet", "--no-clarify", "Say hi"],
        b"",
        _Run(log),
    )
    assert code == 0
    assert log == [
        ("begin", "Say hi"),
        ("act", "Say hi"),
        ("finish", handle, False),
        ("abort",),
    ]
    assert out == []  # without --jsonl the result is printed, not emitted


def test_quit_without_consolidation_records_the_episode_and_starts_no_pass(mv2):
    # a driver ending its stream: {"quit": true, "consolidate": false}; no pass may be tied to the stream's end
    run = _begin(mv2, "Say hi to ada.")
    _transcript(run)
    progress, emitted = _finish(mv2, run, consolidate=False)
    names = mv2.fakes.names()
    assert "assemble" in names and "checker_signal" in names
    assert "request_end" not in names
    episodes = mv2.paths.episodes
    log = os.popen(f"git --git-dir {episodes} log --format=%s main").read()
    assert log.splitlines()[0] == f"episode {run.episode_id}"
    assert emitted == [
        {
            "type": "consolidation",
            "phase": "held",
            "episode_id": run.episode_id,
            "reason_codes": ["quit_without_consolidation"],
        },
    ]
    held = [
        json.loads(x) for x in mv2.paths.events.read_text().splitlines() if x.strip()
    ]
    # rev 2: the events file opens with the request's capture-serving row (written at begin, never emitted)
    assert held[0] == {
        "type": "capture_serving",
        "episode_id": run.episode_id,
        "withdrawn": 0,
        "withdrawn_replaced": 0,
        "offered_in_place": [],
    }
    assert held[1:] == emitted
    assert any("no consolidation pass started" in p for p in progress)
    _left_nothing(mv2.paths)
