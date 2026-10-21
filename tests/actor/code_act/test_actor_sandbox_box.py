"""Symbolic: the actor's sandbox shows fixed paths, runs under resource limits
with an explicit environment, and is killed verifiably.

Nothing here reaches a model. The command lines are the real ``wrap_argv``'s,
and the kills are the real ``process_tree.kill_tree``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    TOKEN_VALUE,
    needs_bwrap,
    world,
)
from unify import sandbox
from unify.common import process_tree

GIB = 1024**3


def _pairs(argv: list[str], option: str, width: int) -> list[list[str]]:
    end = argv.index("--")
    return [argv[i + 1 : i + 1 + width] for i in range(end) if argv[i] == option]


# ── the command line ─────────────────────────────────────────────────────────


@needs_bwrap
def test_a_neutral_sandbox_binds_the_workspace_at_a_fixed_path_and_starts_there(
    world,
):
    policy = sandbox.build_policy(fresh=True)
    workspace = str(Path(os.path.realpath(world["workspace"])))
    env = sandbox.sandbox_env(policy)
    argv = sandbox.wrap_argv(
        ["true"],
        policy,
        cwd=str(world["workspace"]),
        env=env,
        neutral=True,
    )
    binds = _pairs(argv, "--bind", 2)
    assert [workspace, "/workspace"] in binds
    # The host path is a bind source only, never a place in the sandbox.
    destinations = [
        b[1]
        for opt in ("--bind", "--ro-bind")
        for b in _pairs(argv, opt, 2)
    ]
    assert not any(d == workspace or d.startswith(workspace + "/") for d in destinations)
    transcripts = str(Path(os.path.realpath(world["state"] / "transcripts")))
    assert [transcripts, "/transcripts"] in _pairs(argv, "--ro-bind", 2)
    # The command starts in the workspace, as the sandbox shows it.
    assert argv[argv.index("--chdir") + 1] == "/workspace"
    assert ["PWD", "/workspace"] in _pairs(argv, "--setenv", 2)
    # Unchanged without neutral: the workspace at its own path.
    plain = sandbox.wrap_argv(["true"], policy, cwd=str(world["workspace"]))
    assert [workspace, workspace] in _pairs(plain, "--bind", 2)
    assert "--clearenv" not in plain[: plain.index("--")]


@needs_bwrap
def test_the_sandbox_gets_an_explicit_environment_and_resource_limits(
    world,
    monkeypatch,
):
    policy = sandbox.build_policy(fresh=True)
    env = sandbox.sandbox_env(policy)
    argv = sandbox.wrap_argv(["true"], policy, env=env, neutral=True)
    options = argv[: argv.index("--")]
    assert "--clearenv" in options
    setenv = dict(_pairs(argv, "--setenv", 2))
    assert setenv.pop("PWD") == "/workspace"
    assert setenv == env
    # The allow-list and its second filter: a declared probe passes, the
    # declared credentials do not, by name or by value.
    assert setenv["UNIFY_SANDBOX_PROBE"] == "visible"
    assert "FAKE_SERVICE_TOKEN" not in argv and "DB_PASSWORD" not in argv
    assert TOKEN_VALUE not in argv
    assert "UNIFY_HOME" not in setenv
    # A credential handed in explicitly is still dropped from the command line.
    planted = {**env, "PLANTED_API_KEY": "planted-key-for-the-env-test"}
    again = sandbox.wrap_argv(["true"], policy, env=planted, neutral=True)
    assert "PLANTED_API_KEY" not in again
    assert "planted-key-for-the-env-test" not in again
    # Inside, prlimit sets the limits before the command runs.
    inside = argv[argv.index("--") + 1 :]
    assert inside == [
        sandbox.PRLIMIT,
        f"--as={16 * GIB}:{16 * GIB}",
        "--core=0:0",
        f"--fsize={16 * 1024**3}:{16 * 1024**3}",
        "--nofile=8192:8192",
        "--nproc=4096:4096",
        "--",
        "true",
    ]


def test_a_host_hard_limit_below_the_sandbox_limit_is_kept(monkeypatch):
    """prlimit cannot raise a hard limit, so a host with a lower one keeps its own and the sandbox still starts."""
    import resource

    real = resource.getrlimit

    def low(which):
        if which == resource.RLIMIT_NOFILE:
            return (512, 2048)
        if which == resource.RLIMIT_NPROC:
            return (100, 300)
        return real(which)

    monkeypatch.setattr(resource, "getrlimit", low)
    monkeypatch.setattr(sandbox.os, "access", lambda *a, **k: True)
    args = sandbox._rlimit_args()
    assert "--nofile=2048:2048" in args
    assert "--nproc=300:300" in args


def test_paths_map_between_the_sandbox_and_the_host(world):
    policy = sandbox.build_policy(fresh=True)
    workspace = Path(os.path.realpath(world["workspace"]))
    assert sandbox.host_path("/workspace/data.txt", policy) == str(
        workspace / "data.txt",
    )
    assert sandbox.host_path("/workspace", policy) == str(workspace)
    # Leaving the mount leaves the mapping; the caller's checks then apply.
    assert sandbox.host_path("/workspace/../etc/hosts", policy) == "/etc/hosts"
    assert sandbox.host_path("data.txt", policy) == "data.txt"
    assert sandbox.box_path(workspace / "a" / "b.txt", policy) == "/workspace/a/b.txt"
    assert sandbox.box_path("/usr/bin/env", policy) == "/usr/bin/env"
    text = f"at {workspace}/x and {world['state']}/logs/unify.log"
    neutral = sandbox.neutral_text(text, policy)
    assert str(world["state"]) not in neutral and "/workspace/x" in neutral


# ── verified termination ─────────────────────────────────────────────────────


def test_the_actor_and_the_memory_boxes_share_one_kill():
    from unify.memory import sandbox_run

    assert sandbox_run._descendants is process_tree.descendants
    assert sandbox_run._alive is process_tree.alive


@pytest.mark.asyncio
async def test_the_kill_ends_a_process_group_and_every_descendant():
    proc = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        "sleep 60 & sleep 60",
        start_new_session=True,
    )
    await asyncio.sleep(0.3)
    children = process_tree.descendants(proc.pid)
    assert children
    process_tree.kill_tree(proc.pid)
    assert not any(process_tree.alive(pid, start) for pid, start in children)
    await asyncio.wait_for(proc.wait(), 10)


@pytest.mark.asyncio
async def test_a_survivor_makes_the_kill_raise(monkeypatch):
    proc = await asyncio.create_subprocess_exec(
        "sleep",
        "60",
        start_new_session=True,
    )
    survivor = (2_000_000_000, "stubbed")
    real_alive = process_tree.alive
    monkeypatch.setattr(process_tree, "KILL_GRACE_S", 0.3)
    monkeypatch.setattr(
        process_tree,
        "descendants",
        lambda pid: [survivor],
    )
    monkeypatch.setattr(
        process_tree,
        "alive",
        lambda pid, start: (pid, start) == survivor or real_alive(pid, start),
    )
    try:
        with pytest.raises(RuntimeError, match="survived SIGKILL"):
            process_tree.kill_tree(proc.pid, grace_s=0.3)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        await asyncio.wait_for(proc.wait(), 10)


def test_the_teammates_request_names_the_record_by_its_sandbox_path(tmp_path):
    from unify.agents.binding import helper_request

    path = tmp_path / "run-label-7q" / "records" / "x" / "record.jsonl"
    text = helper_request("h1", "root", "Count the rows of a.csv please.", 3, path)
    assert "/record/record.jsonl" in text
    assert "run-label-7q" not in text and str(tmp_path) not in text


@pytest.mark.skipif(sys.platform != "linux", reason="needs /proc")
def test_a_plain_process_is_killed_and_verified(tmp_path):
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    process_tree.kill_tree(proc.pid, exited=lambda: proc.poll() is not None)
    assert proc.poll() is not None
