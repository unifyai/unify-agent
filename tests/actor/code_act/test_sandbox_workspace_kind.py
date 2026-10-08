"""Symbolic: the configured workspace is its own root kind, and the store is hidden inside it.

The workspace (``UNIFY_LOCAL_ROOT``, or ``<UNIFY_HOME>/workspace``) is where
the work is, so a top-level directory such as ``/data`` is allowed; only ``/``,
a home itself, ``UNIFY_HOME`` itself, a system directory and a directory that
is or holds a configuration, cache or credential directory are refused.
Inside it, secret files are masked (by an incremental, bounded scan), and a
state directory or log directory it holds stays hidden (the store,
``internal-transcripts/``, the logs) while ``transcripts/`` stays readable. Each test checks access only, on files it
creates itself, and nothing connects anywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    ENV_SECRET,
    STATE_SECRET,
    bash,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.settings import SETTINGS

STORE_FILES = ("store.sqlite", "store.sqlite-wal", "store.sqlite-shm")


def _seed_state(state: Path) -> dict[str, Path]:
    """A state directory with a store, internal and ordinary transcripts."""
    (state / "transcripts").mkdir(parents=True, exist_ok=True)
    (state / "transcripts" / "s.jsonl").write_text('{"seq": 0}\n')
    (state / "internal-transcripts").mkdir()
    (state / "internal-transcripts" / "review.jsonl").write_text(STATE_SECRET + "\n")
    for name in STORE_FILES:
        (state / name).write_text(STATE_SECRET)
    return {
        "transcript": state / "transcripts" / "s.jsonl",
        "internal": state / "internal-transcripts" / "review.jsonl",
        **{name: state / name for name in STORE_FILES},
    }


def _use(monkeypatch, *, state: Path, workspace: str, log_dir: Path) -> None:
    monkeypatch.setenv("UNIFY_HOME", str(state))
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", workspace)
    monkeypatch.setenv("UNILLM_LOG_DIR", str(log_dir))
    monkeypatch.setenv("UNIFY_OTEL_LOG_DIR", str(log_dir.parent / "otel"))
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)


# ── the predicate ───────────────────────────────────────────────────────────


def test_a_top_level_workspace_is_allowed_and_broad_ones_refused():
    homes = [Path("/home/someone")]
    guarded = sandbox._workspace_guarded(homes)

    def refusal(path: str):
        return sandbox._workspace_refusal(Path(path), homes=homes, guarded=guarded)

    for path in ("/data", "/srv/project", "/mnt/project", "/home/someone/proj"):
        assert refusal(path) is None, path
    # The derived roots' rule still refuses a top-level directory.
    assert sandbox._root_refusal(Path("/data")) is not None
    for path in ("/", "/home/someone", "/etc", "/etc/app", "/usr", "/proc", "/var"):
        assert refusal(path) is not None, path
    # Equal to or holding a configuration, cache or credential directory.
    for path in (
        "/home/someone/.config",
        "/home/someone/.cache",
        "/home/someone/.ssh",
        "/home/someone/.local",
        "/home/someone/.local/share",
        "/home",
    ):
        assert refusal(path) is not None, path
    # Below them is fine: the runners' attempts sit under ~/.local/share.
    runner = (
        "/home/someone/.local/share/continual-harness-research/runtime-arc/"
        "a1/system/state/unify/workspace"
    )
    assert refusal(runner) is None


def test_home_state_and_ancestors_of_a_fake_config_or_cache_are_refused(
    world,
    monkeypatch,
):
    home = world["home"]
    (home / ".config").mkdir()
    (home / ".cache").mkdir()
    assert sandbox._workspace_refusal(home) is not None
    assert sandbox._workspace_refusal(home.parent) is not None
    for rel in (".config", ".cache"):
        assert sandbox._workspace_refusal(home / rel) is not None, rel
    # A workspace that holds only a fake ~/.config or ~/.cache (not the home).
    outer = home.parent / "outer"
    for rel in (".config", ".cache"):
        guarded = [outer / "inner" / rel]
        assert sandbox._workspace_refusal(outer, homes=[], guarded=guarded)
        assert sandbox._workspace_refusal(outer / "inner", homes=[], guarded=guarded)
        assert (
            sandbox._workspace_refusal(outer / "elsewhere", homes=[], guarded=guarded)
            is None
        )
    # A workspace holding a log directory is allowed (the log is hidden).
    project = home.parent / "project"
    monkeypatch.setenv("UNILLM_LOG_DIR", str(project / "logs" / "unillm"))
    assert sandbox._workspace_refusal(project) is None
    assert sandbox._workspace_refusal(home / "work") is None
    # UNIFY_HOME itself is refused; a workspace strictly holding it is not.
    state = project / ".unify"
    monkeypatch.setenv("UNIFY_HOME", str(state))
    assert sandbox._workspace_refusal(state) is not None
    assert sandbox._workspace_refusal(project) is None
    assert sandbox._workspace_refusal(state / "workspace") is None


# ── secrets inside the workspace ────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_a_dot_env_inside_the_workspace_is_masked_in_a_cell(world):
    ws = world["workspace"]
    (ws / "a" / "b").mkdir(parents=True)
    secrets = [ws / ".env", ws / "a" / "b" / ".envrc", ws / "a" / "service-key.json"]
    for path in secrets:
        path.write_text(f"KEY={ENV_SECRET}\n")
    (ws / "a" / "plain.txt").write_text("plain\n")
    policy = sandbox.build_policy(fresh=True)
    masked = {p for p, _ in policy.workspace_masked}
    assert set(secrets) <= masked
    for path in secrets:
        assert policy.readable_violation(path) is not None, path
    assert policy.readable_violation(ws / "a" / "plain.txt") is None
    ex = SessionExecutor()
    try:
        out, _ = await bash(
            ex,
            "cat " + " ".join(map(str, secrets)) + f" 2>&1; cat {ws}/a/plain.txt",
        )
        assert ENV_SECRET not in out, out
        assert "rule mask-env-file" in out and "plain" in out, out
        out, _ = await bash(ex, f"grep -r {ENV_SECRET} {ws} 2>/dev/null | wc -l")
        assert out.strip() == "0"
        # The workspace stays writable around the masks.
        out, _ = await bash(ex, "echo made > a/made.txt && cat a/made.txt")
        assert out.strip() == "made"
    finally:
        await ex.close()
    # The host's files are untouched.
    for path in secrets:
        assert path.read_text() == f"KEY={ENV_SECRET}\n"


# ── a workspace that holds or is UNIFY_HOME ─────────────────────────────────


@needs_bwrap
def test_a_workspace_that_is_unify_home_itself_is_refused(world, monkeypatch):
    state = world["home"].parent / "state-workspace"
    state.mkdir()
    _seed_state(state)
    log_dir = world["home"].parent / "logs-outside" / "unillm"
    _use(monkeypatch, state=state, workspace=str(state), log_dir=log_dir)
    assert sandbox._workspace_refusal(state) is not None
    policy = sandbox.build_policy(fresh=True)
    # The harness's own file tools refuse the store there all the same.
    for name in STORE_FILES:
        assert policy.readable_violation(state / name) is not None, name
    with pytest.raises(sandbox.SandboxRefusal) as raised:
        sandbox.wrap_argv(["true"], policy)
    assert raised.value.rule == "root-allowlist"
    assert "state directory" in str(raised.value)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "logs",
    ["outside the workspace", "under UNIFY_HOME", "in the workspace"],
)
async def test_a_workspace_holding_the_state_hides_the_store(
    world,
    monkeypatch,
    logs,
):
    root = world["home"].parent
    workspace = root / "project"
    state = workspace / ".unify"
    state.mkdir(parents=True)
    seeded = _seed_state(state)
    (workspace / "notes.txt").write_text("needle one\n")
    (workspace / "config").mkdir()
    key = workspace / "config" / "service-key.json"
    key.write_text(f'{{"k": "{ENV_SECRET}"}}\n')
    log_dir = {
        "outside the workspace": root / "logs-outside" / "unillm",
        "under UNIFY_HOME": state / "logs" / "unillm",
        "in the workspace": workspace / "logs" / "unillm",
    }[logs]
    log_dir.mkdir(parents=True)
    (log_dir / "request.json").write_text(STATE_SECRET + "\n")
    _use(monkeypatch, state=state, workspace=str(workspace), log_dir=log_dir)
    assert sandbox._workspace_refusal(workspace) is None

    policy = sandbox.build_policy(fresh=True)
    sandbox.wrap_argv(["true"], policy)  # builds without a refusal
    hidden = [
        *(seeded[name] for name in STORE_FILES),
        seeded["internal"],
        log_dir / "request.json",
        key,
    ]
    for path in hidden:
        assert policy.readable_violation(path) is not None, path
    assert policy.readable_violation(seeded["transcript"]) is None
    assert policy.readable_violation(workspace / "notes.txt") is None

    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, "cat " + " ".join(map(str, hidden)) + " 2>&1")
        assert STATE_SECRET not in out and ENV_SECRET not in out, out
        out, _ = await bash(ex, f"cat {seeded['transcript']} {workspace}/notes.txt")
        assert out.split() == ['{"seq":', "0}", "needle", "one"], out
        out, _ = await bash(
            ex,
            f"grep -rl -e {STATE_SECRET} -e {ENV_SECRET} {workspace} {log_dir} "
            "2>/dev/null | wc -l",
        )
        assert out.strip() == "0", out
        out, _ = await bash(ex, f"ls -A {state}/internal-transcripts")
        assert "review.jsonl" not in out
        # Transcripts are readable, never writable; the store stays as it was.
        out, _ = await bash(ex, f"echo x >> {seeded['transcript']} 2>&1")
        assert "Read-only file system" in out, out
        await bash(ex, f"echo x > {seeded['store.sqlite']} 2>&1")
        out, _ = await bash(ex, "echo made > made.txt && cat made.txt")
        assert out.strip() == "made"
    finally:
        await ex.close()
    assert seeded["transcript"].read_text() == '{"seq": 0}\n'
    for name in STORE_FILES:
        assert seeded[name].read_text() == STATE_SECRET
    assert (workspace / "made.txt").read_text() == "made\n"


# ── the benchmark runners' layout ───────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
async def test_the_runner_layout_builds_runs_and_hides_the_state(world, monkeypatch):
    """``<attempt>/system``, ``HOME=state/home``, ``UNIFY_HOME=state/unify``, the
    default workspace ``state/unify/workspace`` as the cwd, and the LLM request
    log under ``UNIFY_HOME``: as ``UnifyAgentLearner`` sets them."""
    system = world["home"].parent / "runtime-bench" / "attempt-1" / "system"
    home = system / "state" / "home"
    state = system / "state" / "unify"
    workspace = state / "workspace"
    home.mkdir(parents=True)
    workspace.mkdir(parents=True)
    seeded = _seed_state(state)
    log_dir = state / "logs" / "unillm"
    log_dir.mkdir(parents=True)
    (log_dir / "request.json").write_text(STATE_SECRET + "\n")
    (workspace / "task.txt").write_text("the task\n")
    monkeypatch.setenv("HOME", str(home))
    _use(monkeypatch, state=state, workspace="", log_dir=log_dir)
    monkeypatch.chdir(workspace)

    policy = sandbox.build_policy(fresh=True)
    assert policy.workspace == Path(os.path.realpath(workspace))
    assert sandbox._workspace_refusal(policy.workspace) is None
    sandbox.wrap_argv(["true"], policy)  # builds without a refusal
    hidden = [
        *(seeded[name] for name in STORE_FILES),
        seeded["internal"],
        log_dir / "request.json",
    ]
    for path in hidden:
        assert policy.readable_violation(path) is not None, path
    assert policy.readable_violation(seeded["transcript"]) is None

    ex = SessionExecutor()
    try:
        out, _ = await bash(ex, "pwd -P; cat task.txt")
        assert out.split() == [str(policy.workspace), "the", "task"], out
        out, _ = await bash(ex, "cat " + " ".join(map(str, hidden)) + " 2>&1")
        assert STATE_SECRET not in out, out
        out, _ = await bash(ex, f"ls -A {state}")
        assert set(out.split()) == {
            sandbox.MASK_NOTICE_NAME,
            "transcripts",
            "workspace",
        }, out
        out, _ = await bash(ex, f"cat {seeded['transcript']}")
        assert out.strip() == '{"seq": 0}'
    finally:
        await ex.close()


# ── the workspace scan: incremental, bounded, noted in the run's record ─────


def _age(paths, seconds: int) -> None:
    """Set each directory's mtime *seconds* in the past (past the settle window)."""
    import time

    t = time.time_ns() - seconds * 1_000_000_000
    for path in paths:
        os.utime(path, ns=(t, t))


def test_the_workspace_scan_reads_only_changed_directories(world, monkeypatch):
    """The runner layout with a representative workspace: about 5k files in
    nested directories and a ``.git`` with many objects, which is not entered."""
    state = world["home"].parent / "runtime-bench" / "a1" / "system" / "state"
    workspace = state / "unify" / "workspace"
    dirs = []
    for a in range(10):
        for b in range(10):
            d = workspace / "src" / f"pkg{a}" / f"mod{b}"
            d.mkdir(parents=True)
            dirs.append(d)
            for i in range(50):
                (d / f"f{i}.py").write_text("x = 1\n")
    (dirs[37] / ".env").write_text(f"KEY={ENV_SECRET}\n")
    objects = workspace / ".git" / "objects"
    for a in range(40):
        (objects / f"{a:02x}").mkdir(parents=True)
        for i in range(100):
            (objects / f"{a:02x}" / f"{i:038x}").write_text("blob\n")
    every = [p for p in workspace.rglob("*") if p.is_dir()] + [workspace]
    _age(every, 60)
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCANS", {})

    first = sandbox._scan_workspace(workspace, [])
    assert not first.capped
    assert first.entries < sandbox._WORKSPACE_SCAN_LIMIT // 10, first.entries
    assert first.reused == 0 and first.rescanned == first.visited
    assert (dirs[37] / ".env", "mask-env-file") in first.files
    assert not any(p.is_relative_to(objects) for p, _ in first.files)
    # .git is listed by its parent but never entered.
    assert first.visited == len([p for p in every if ".git" not in p.parts])

    # One directory changes: only it is read again.
    (dirs[5] / "service-key.json").write_text(f'{{"k": "{ENV_SECRET}"}}\n')
    _age([dirs[5]], 30)
    second = sandbox._scan_workspace(workspace, [])
    assert second.rescanned == 1, second
    assert second.reused == first.visited - 1
    assert (dirs[5] / "service-key.json", "mask-credentials") in second.files
    assert (dirs[37] / ".env", "mask-env-file") in second.files

    # Nothing changed: nothing is read.
    third = sandbox._scan_workspace(workspace, [])
    assert third.rescanned == 0 and third.entries == 0


def test_a_capped_scan_is_noted_in_the_runs_record(world, monkeypatch):
    workspace = world["workspace"]
    for i in range(20):
        (workspace / f"d{i}").mkdir()
        (workspace / f"d{i}" / "x.txt").write_text("x\n")
    noted = []

    class _Record:
        path = world["home"].parent / "records" / "run-1" / "record.jsonl"

        def append_harness(self, author, text, *, kind="post", mentions=None):
            noted.append((author, text, kind, mentions))

    class _Pool:
        record = _Record()

    from unify.agents import binding

    monkeypatch.setattr(binding, "current_root_pool", lambda: _Pool())
    monkeypatch.setattr(sandbox, "_current_run_records", lambda: None)
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCAN_LIMIT", 5)
    monkeypatch.setattr(sandbox, "_WORKSPACE_SCANS", {})
    monkeypatch.setattr(sandbox, "_NOTED", set())
    policy = sandbox.build_policy(fresh=True)
    assert policy.workspace_scan is not None and policy.workspace_scan.capped
    assert policy.scan_note and "cap of 5 entries" in policy.scan_note
    assert noted == [("harness", policy.scan_note, "system", [])]
    # Once per run, not on every rebuild.
    sandbox.build_policy(fresh=True)
    assert len(noted) == 1
