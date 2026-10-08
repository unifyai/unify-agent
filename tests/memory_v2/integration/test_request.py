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
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory_v2.integration import fake_tracks
from tests.memory_v2.integration.test_checkout import _seed
from unify import sandbox
from unify.function_manager.primitives import observers
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import acquire_lock, release_lock
from unify.settings import SETTINGS

SENTINEL = "SENTINEL-7f3a"


@pytest.fixture
def mv2(monkeypatch, tmp_path):
    """A ``UNIFY_HOME`` with a seeded memory, the switch on and the other tracks faked."""
    home = tmp_path / "unify"
    home.mkdir()
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    mem, sha = _seed(home)
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


@pytest.mark.parametrize("surfacing", ["index", "catalogue"])
def test_begin_exports_memory_and_opens_the_scope(mv2, monkeypatch, surfacing):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SURFACING", surfacing)
    run = _begin(mv2, "Say hi to ada.")
    try:
        paths = mv2.paths
        assert request_mod.current() is run
        assert run.pin == mv2.sha and run.request == "Say hi to ada."
        assert (paths.checkout / "env/spotify/__init__.py").exists()
        if surfacing == "catalogue":
            # the generated catalogue sits beside the commit's files; the prompt holds the constant guide
            readme = (paths.checkout / "README.md").read_text()
            assert "- `hello(apis, name)`: Say hi." in readme
            assert (paths.checkout / ".memory/catalog.json").is_file()
            assert (paths.checkout / "memory.py").is_file()
            assert set(run.generated) == {
                "README.md",
                "memory.py",
                ".memory/catalog.json",
                ".memory/shapes.py",
            }
            from unify.memory_v2.integration.prompt import GUIDE
            from unify.memory_v2.integration.state import State

            assert run.index == GUIDE
            # shown once, kept for the rest of the run (saved at once, so an abort keeps it too)
            assert run.state.guide is True and State.load(paths.state).guide is True
        else:  # v2: the index in the prompt, nothing generated beside the export
            assert "`hello(apis, name)`" in run.index
            assert run.state.guide is False and not paths.state.exists()
            assert run.generated == {}
            assert not (paths.checkout / "README.md").exists()
            assert not (paths.checkout / "memory.py").exists()
            assert not (paths.checkout / ".memory").exists()
        assert hooks.system_prompt("S").endswith(run.index)
        assert hooks.worker_mounts() == [paths.checkout]
        assert not _lock_is_free(paths)
        # the work-tree capture saw the policy's workspace and took its before snapshot
        (wc_paths, workspace), _ = mv2.fakes.of("WorktreeCapture")
        assert wc_paths == paths
        assert workspace == sandbox.build_policy().workspace
        assert mv2.fakes.names()[:3] == [
            "open_stores",
            "WorktreeCapture",
            "worktree.begin",
        ]
        # the transcript continues the episode id; costs are recorded; no tool observer (not wired)
        from unify import transcripts

        assert mv2.ctx.run(transcripts._REQUESTED_ID.get) == run.episode_id
        assert transcripts._REQUESTED_ID.get() is None  # only the run's context
        assert run.observer is None and mv2.ctx.run(observers._OBSERVERS.get) == ()
        assert mv2.fakes.cost_active == [True]
        assert run.episode_id[:8].isdigit() and run.episode_id[8] == "T"
    finally:
        _abort(mv2, run)
    _left_nothing(mv2.paths)
    assert mv2.ctx.run(transcripts._REQUESTED_ID.get) is None
    assert mv2.fakes.names()[-1] == "worktree.abort"
    assert mv2.fakes.cost_active == [True, False]


def test_a_stale_export_from_a_killed_process_is_replaced(mv2):
    mv2.paths.checkout.mkdir()
    (mv2.paths.checkout / "junk.py").write_text("x = 1\n")
    run = _begin(mv2, "hi")
    try:
        assert not (mv2.paths.checkout / "junk.py").exists()
        assert (mv2.paths.checkout / "env/spotify/__init__.py").exists()
    finally:
        _abort(mv2, run)


def test_begin_refuses_when_a_cell_could_read_harness_state(mv2, monkeypatch):
    real = sandbox.build_policy(fresh=True)

    class Leaky:
        workspace = real.workspace

        def readable_violation(self, path):
            return None if Path(path) == mv2.paths.episodes else ("x", "hidden")

    monkeypatch.setattr(sandbox, "build_policy", lambda fresh=False: Leaky())
    with pytest.raises(request_mod.MemoryV2Unavailable, match="episodes.git"):
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
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    assert hooks.begin_request("hi") is None
    assert mv2.fakes.calls == [] and request_mod.current() is None


def test_hooks_open_the_run_under_the_switch(mv2):
    run = mv2.ctx.run(hooks.begin_request, "hi")
    try:
        assert isinstance(run, request_mod.RequestRun) and request_mod.current() is run
    finally:
        _abort(mv2, run)


# ── Sol's effort matches the actor's unless a mismatch ablation fixes it ────


@pytest.mark.parametrize(
    "effort,want",
    [("low", "low"), ("medium", "medium"), ("", "high")],
)
def test_the_effort_is_the_actors(mv2, monkeypatch, effort, want):
    from unify.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", effort)
    assert request_mod.actor_effort() == want
    run = _begin(mv2, "hi")
    _transcript(run)
    _finish(mv2, run)
    _, kw = mv2.fakes.of("run_due_passes")
    # Sol's effort matches the actor's by default (the lead, 8 Oct)
    assert run.effort == want and kw["effort"] == want


def test_an_assistant_default_model_brings_its_effort(mv2, monkeypatch):
    from unify.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "openai/x")
    monkeypatch.setattr(
        SESSION_DETAILS.assistant,
        "default_reasoning_effort",
        "minimal",
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", "high")
    assert request_mod.actor_effort() == "minimal"
    assert request_mod.actor_model() == "openai/x"


# ── finish ───────────────────────────────────────────────────────────────────


def test_finish_records_the_episode_and_runs_the_passes(mv2):
    run = _begin(mv2, "Say hi to ada.")
    transcript = _transcript(run)
    (mv2.paths.checkout / "env/spotify/scratch.py").write_text("x = 1\n")
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
            "post_checker",
            "run_due_passes",
        )
    ]
    assert order == [
        "read_jsonl",
        "worktree.finish",
        "assemble",
        "post_checker",
        "run_due_passes",
    ]
    (path,), _ = f.of("read_jsonl")
    assert path == transcript
    (cells,), _ = f.of("worktree.finish")
    assert cells == [("cell", 1)]  # trajectory.timed_cells over the folded transcript
    (a_run, lines, memory_diff, ended_at), kw = f.of("assemble")
    assert a_run is run and [x["seq"] for x in lines] == [0, 1, 2]
    assert "scratch.py" in memory_diff
    # the untouched generated catalogue is not something the request wrote
    assert "README.md" not in memory_diff and "catalog.json" not in memory_diff
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
    (_, eid, sha, solved, _ts), _ = f.of("post_checker")
    assert eid == run.episode_id and solved is None and len(sha) == 40
    from unify.memory_v2.evidence import EvidenceStore

    assert EvidenceStore(mv2.paths.evidence).episode_ref(eid)[0] == sha
    # the passes: inherited effort, the settings, and an emitter
    (_, p_eid, p_sha, state), kw = f.of("run_due_passes")
    assert (p_eid, p_sha) == (eid, sha)
    assert kw["settings"] is SETTINGS and kw["effort"] == run.effort
    assert mv2.paths.state.exists()
    # the events reach the CLI's emitter, money as plain decimal strings (never an exponent)
    assert [e["phase"] for e in emitted] == ["start", "end"]
    assert emitted[0]["cap_usd"] == "0.00000073" and emitted[1]["usd"] == "0.0000001"
    assert emitted[0]["sol_effort"] == run.effort
    assert emitted[1]["reason_codes"] == ["no_manifest"]
    assert progress == []
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
    _, kw = mv2.fakes.of("run_due_passes")
    assert kw["emit"] is None


def test_finish_never_raises_and_logs_a_failed_pass(mv2):
    mv2.fakes.passes_raise = RuntimeError("sol fell over")
    run = _begin(mv2, "hi")
    _transcript(run)
    progress, _ = _finish(mv2, run)
    errors = [json.loads(x) for x in mv2.paths.errors.read_text().splitlines()]
    assert [e["stage"] for e in errors] == ["passes"]
    assert errors[0]["episode_id"] == run.episode_id
    assert "RuntimeError" in errors[0]["error"]
    assert progress and "memory v2" in progress[0]
    assert mv2.paths.state.exists()  # generations are kept even when the passes fail
    _left_nothing(mv2.paths)


def test_a_failed_episode_runs_no_pass(mv2):
    mv2.fakes.assemble_raise = ValueError("bad transcript")
    run = _begin(mv2, "hi")
    _transcript(run)
    progress, _ = _finish(mv2, run)
    assert "run_due_passes" not in mv2.fakes.names()
    assert "post_checker" not in mv2.fakes.names()
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


def test_finish_records_how_the_request_used_the_library(mv2):
    """The record is computed at finish from the transcript against the items at the pin, written as the
    episode's ``memory_use.json`` and indexed per item. The cell edited ``hello`` in its scratch copy, so
    the refusal it then raised is booked as the edit's, never the stored function's."""
    import hashlib

    from unify.memory_v2.episodes import episode_dir
    from unify.memory_v2.evidence import EvidenceStore
    from unify.memory_v2.gitio import Repo

    run = _begin(mv2, "Say hi to ada.")
    assert run.item_ids == ["env/spotify:hello"]
    # a function the cell writes into its scratch copy is not a memory item
    (mv2.paths.checkout / "env/spotify/__init__.py").write_text(
        "class MemoryInputError(ValueError):\n    pass\n\n"
        "def hello(apis, name):\n    raise MemoryInputError(name)\n\n"
        "def extra():\n    return 1\n",
    )
    code = "from env.spotify import hello as hi, extra\nextra()\nhi(None, 'ada')\n"
    module = mv2.paths.checkout / "env/spotify/__init__.py"
    error = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 3, in <module>\n'
        f'  File "{module}", line 5, in hello\n'
        "    raise MemoryInputError(name)\n"
        "env.spotify.MemoryInputError: ada\n"
    )
    _transcript(run, *_cell_lines(code, f"core\n\n{run.index}", error))
    # the code tool's structured result, as the tool loop hands it over (hooks.tool_result); the
    # rendered metadata block above is never read for it
    from unify.actor.execution.types import ExecutionResult

    hooks.tool_result("execute_code", "c0", ExecutionResult(error=error, duration_ms=1))
    assert run.cell_status["c0"]["exits"] == [["refused", "spotify", "hello"]]
    _finish(mv2, run)
    rel = episode_dir(run)
    raw = Repo(mv2.paths.episodes).show("main", f"{rel}/memory_use.json")
    rec = json.loads(raw)
    assert rec["items_at_pin"] == ["env/spotify:hello"]
    assert set(rec["items"]) == {"env/spotify:hello"}
    assert rec["export_roots"][0] == str(mv2.paths.checkout)
    assert rec["modified_channels"] == ["spotify"]
    hello = rec["items"]["env/spotify:hello"]
    assert (hello["imported"], hello["called"]) == (1, 1)
    assert (hello["refused"], hello["refused_modified"]) == (0, 1)
    assert rec["outcomes_known"] and rec["cells_without_metadata"] == 0
    assert rec["items_outcome_unknown"] == [] and rec["cells_outcome_unknown"] == 0
    assert set(rec["cells_without_metadata_by_cause"].values()) == {0}
    assert [c["status"] for c in rec["cell_status"]] == ["error"]
    assert hello["modified_in_request"] is True
    shown = rec["memory_section_shown"]
    assert shown["shown"]
    assert shown["sha256"] == hashlib.sha256(run.index.encode()).hexdigest()
    assert rec["shown_items"] == ["env/spotify:hello"]
    assert rec["shown_channels"] == ["spotify"]
    # from the renderer's own record, confirmed against the prompt the transcript holds
    assert rec["exposure_source"] == "record" and shown["prompt_confirmed"] is True
    assert rec["shown_record"]["renderer"] == "index"
    assert rec["shown_record"]["items"] == ["env/spotify:hello"]
    assert run.index not in raw.decode()  # names and a digest, never the section's text
    totals = EvidenceStore(mv2.paths.evidence).item_use()
    row = totals["env/spotify:hello"]
    assert (row["called"], row["refused"], row["refused_modified"]) == (1, 0, 1)
    assert (row["shown"], row["channel_shown"], row["modified"]) == (1, 1, 1)
    assert (row["exposure_record"], row["exposure_legacy_text"]) == (1, 0)
    assert (row["outcome_unknown"], row["prompt_unconfirmed"]) == (0, 0)
    assert not mv2.paths.errors.exists()
    _left_nothing(mv2.paths)


# ── the outcome: pass/fail only ──────────────────────────────────────────────


def _scan(value, seen=None) -> str:
    """Every string reachable from *value* (attributes, containers), for a sentinel scan."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return ""
    seen.add(id(value))
    if isinstance(value, (str, bytes)):
        return value if isinstance(value, str) else value.decode("utf-8", "replace")
    if isinstance(value, dict):
        return " ".join(_scan(k, seen) + " " + _scan(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_scan(v, seen) for v in value)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _scan(vars(value), seen)
    return repr(value)


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
    (_, _, _, solved, _), _ = mv2.fakes.of("post_checker")
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
    def __init__(self, log: list) -> None:
        self.log = log

    def take_outcome(self, raw):
        self.log.append(("take_outcome", raw))
        return {"type": "outcome", "accepted": True, "solved": True, "checks": 0}

    async def finish(self, handle, *, progress, emit=None):
        self.log.append(("finish", handle, emit is not None))
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
    assert types_ == ["outcome", "result", "consolidation", "ended"]
    assert out[0] == {"type": "outcome", "accepted": True, "solved": True, "checks": 0}


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


def test_cli_with_the_switch_off_is_as_shipped(monkeypatch):
    code, out, log, handle = _drive(
        monkeypatch,
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"],
        b'{"outcome": {"solved": true}}\n{"quit": true}\n',
        None,
    )
    assert code == 0
    assert [e[0] for e in log] == ["begin", "act"]
    assert out[0]["type"] == "outcome" and out[0]["accepted"] is False
    assert "takes no outcome" in out[0]["reason"]
    assert [o["type"] for o in out] == ["outcome", "result", "ended"]


@pytest.mark.parametrize("actor_effort", ["low", "medium", "high"])
@pytest.mark.parametrize("sol_effort", ["actor", "low", "medium"])
def test_a_fixed_sol_effort_overrides_the_actors_only_when_declared(
    mv2,
    monkeypatch,
    actor_effort,
    sol_effort,
):
    from unify.session_details import SESSION_DETAILS

    monkeypatch.setattr(SESSION_DETAILS.assistant, "default_model", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_REASONING_EFFORT", actor_effort)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SOL_EFFORT", sol_effort)
    run = _begin(mv2, "hi")
    _transcript(run)
    _finish(mv2, run)
    _, kw = mv2.fakes.of("run_due_passes")
    want = actor_effort if sol_effort == "actor" else sol_effort
    assert kw["effort"] == want and run.effort == actor_effort


def test_the_sol_effort_setting_is_validated():
    from unify.memory_v2.integration import switch

    assert switch.parse_sol_effort("") == "actor"
    assert switch.parse_sol_effort("low") == "low"
    assert switch.parse_sol_effort(" Medium ") == "medium"
    with pytest.raises(ValueError):
        switch.parse_sol_effort("xhigh")


def test_quit_without_consolidation_records_the_episode_and_starts_no_pass(mv2):
    # a driver ending its stream: {"quit": true, "consolidate": false}; no pass may be tied to the stream's end
    run = _begin(mv2, "Say hi to ada.")
    _transcript(run)
    progress, emitted = _finish(mv2, run, consolidate=False)
    names = mv2.fakes.names()
    assert "assemble" in names and "post_checker" in names
    assert "run_due_passes" not in names
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
    assert held == emitted
    assert any("no consolidation pass started" in p for p in progress)
    _left_nothing(mv2.paths)
