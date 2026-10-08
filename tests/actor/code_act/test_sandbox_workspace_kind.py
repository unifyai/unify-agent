"""Symbolic: the configured workspace is its own root kind, and the store is hidden inside it.

The workspace (``UNIFY_LOCAL_ROOT``, or ``<UNIFY_HOME>/workspace``) is where
the work is, so a top-level directory such as ``/data`` is allowed; only ``/``,
a home itself, a system directory and a directory that is or holds a log,
configuration, cache or credential directory are refused. Inside it, secret
files are masked, and a state directory it holds (or is) keeps the store,
``internal-transcripts/`` and the log directories hidden while
``transcripts/`` stays readable. Each test checks access only, on files it
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


def test_home_and_ancestors_of_a_fake_config_or_cache_are_refused(
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
    # A workspace holding a log directory.
    project = home.parent / "project"
    monkeypatch.setenv("UNILLM_LOG_DIR", str(project / "logs" / "unillm"))
    assert sandbox._workspace_refusal(project) is not None
    assert sandbox._workspace_refusal(project / "src") is None
    assert sandbox._workspace_refusal(home / "work") is None


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
@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["UNIFY_HOME inside", "UNIFY_HOME itself"])
async def test_a_workspace_holding_the_state_hides_the_store(
    world,
    monkeypatch,
    layout,
):
    root = world["home"].parent
    if layout == "UNIFY_HOME inside":
        workspace = root / "project"
        state = workspace / ".unify"
    else:
        workspace = state = root / "state-workspace"
    state.mkdir(parents=True)
    seeded = _seed_state(state)
    (workspace / "notes.txt").write_text("needle one\n")
    (workspace / "config").mkdir()
    key = workspace / "config" / "service-key.json"
    key.write_text(f'{{"k": "{ENV_SECRET}"}}\n')
    log_dir = root / "logs-outside" / "unillm"
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
