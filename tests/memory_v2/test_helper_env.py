"""The code miner's and replay verification's helper processes never inherit the pass worker's environment (a live
token audit found ``tar`` and the box's ``bwrap`` holding the writer's base URL). Each spawn site runs through its
real code; every ``subprocess.run`` it makes is answered by a real child (``env -0``) started with exactly the
environment the site passed, and that child's environment must hold no route name and no sentinel value.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from unify.memory_v2 import mining
from unify.memory_v2 import mining_behaviour
from unify.memory_v2 import s0_pass

SENTINEL = "http://127.0.0.1:1/writer/v1/sentinel-route"
ROUTE_NAMES = (
    "UNIFY_MEMORY_V2_WRITER_BASE_URL",
    "UNIFY_MEMORY_V2_WRITER_TOKEN_FD",
    "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
    "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
)
REAL_RUN = subprocess.run


@pytest.fixture
def spawns(monkeypatch):
    """Every helper spawn as (argv[0] basename, the child's real environment)."""
    for name in ROUTE_NAMES:
        monkeypatch.setenv(name, SENTINEL)
    seen: list[tuple[str, dict]] = []

    def run(argv, *args, **kwargs):
        env = kwargs.get("env")
        probe = REAL_RUN(
            ["/usr/bin/env", "-0"],
            env=env,
            capture_output=True,
            check=True,
        )
        child = dict(
            kv.split("=", 1) for kv in probe.stdout.decode().split("\0") if "=" in kv
        )
        seen.append((os.path.basename(str(argv[0])), child))
        out: bytes | str = b""
        if "ls-tree" in argv:
            out = b"2026/10/ep1/meta.json\n"
        elif "show" in argv:
            out = json.dumps({"worktree_before": "abc123"}).encode()
        if kwargs.get("text"):
            out = "{}"
        return subprocess.CompletedProcess(
            argv,
            0,
            out,
            b"" if not kwargs.get("text") else "",
        )

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def _clean(seen):
    assert seen
    for name, env in seen:
        assert not any(k in env for k in ROUTE_NAMES), name
        assert SENTINEL not in "".join(env.values()), name
        assert set(env) <= {"PATH", "HOME"}, (name, sorted(env))


def test_s0_export_tree_runs_git_and_tar_without_the_route(spawns, tmp_path):
    assert (
        s0_pass._export_tree("/no/worktree.git", "abc123", tmp_path / "t") is not None
    )
    assert [n for n, _ in spawns] == ["git", "tar"]
    _clean(spawns)


def test_behaviour_export_trees_runs_git_and_tar_without_the_route(spawns, tmp_path):
    out = mining_behaviour.export_trees(
        "/no/worktree.git",
        "/no/episodes.git",
        "main",
        ["ep1"],
        tmp_path,
    )
    assert list(out) == ["ep1"]
    assert {n for n, _ in spawns} == {"git", "tar"}
    _clean(spawns)


def test_the_box_s_bwrap_starts_without_the_route(spawns, monkeypatch):
    monkeypatch.setattr(
        mining_behaviour.shutil,
        "which",
        lambda name: "/usr/bin/" + name,
    )
    assert mining_behaviour.in_box({"mode": "replay"}, 5) == {}
    assert [n for n, _ in spawns] == ["bwrap"]
    _clean(spawns)


def test_the_recorded_episodes_git_reader_runs_without_the_route(spawns):
    mining._git("/no/episodes.git", "ls-tree", "-r", "--name-only", "main")
    _clean(spawns)


def test_the_helper_env_is_path_and_a_scratch_home_only():
    env = mining.helper_env()
    assert set(env) == {"PATH", "HOME"} and not Path(env["HOME"]).exists()
