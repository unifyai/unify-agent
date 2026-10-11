"""Memory's git processes never carry a harness-only route name (M4 qualification on w174, 11 Oct: the runner's
live token audit caught an ``sh`` holding the Sol base URL; a local push runs ``git-receive-pack`` under ``sh -c`` with
the environment git was given). Every git environment builder starts from :func:`gitio.git_base_env`; a real local
push's receive hook, a descendant of that shell, sees none of the names.
"""

from __future__ import annotations

import subprocess

import pytest

from unify.memory_v2 import gitio, snapshot
from unify.memory_v2.integration import hardgit, request
from unify.memory_v2.sandbox_run import HARNESS_ONLY_ENV

SENTINEL = "http://127.0.0.1:1/route-sentinel"
REAL_RUN = subprocess.run


@pytest.fixture
def routed(monkeypatch):
    for name in HARNESS_ONLY_ENV:
        monkeypatch.setenv(name, SENTINEL)
    monkeypatch.setenv(
        "unify_memory_v2_sol_base_url",
        SENTINEL,
    )  # any case of a route name
    monkeypatch.setenv("KEEP_ME", "kept")


def _clean(env: dict) -> None:
    assert not {k.upper() for k in env} & HARNESS_ONLY_ENV
    assert SENTINEL not in "".join(env.values())


def test_a_local_push_s_receive_hook_sees_no_route(routed, tmp_path):
    bare = gitio.Repo.init_bare(
        tmp_path / "m.git",
    )  # init_bare itself does a local push
    out = tmp_path / "hook-env"
    hook = bare.git_dir / "hooks" / "post-receive"
    hook.write_text(f"#!/bin/sh\nenv > {out}\n")
    hook.chmod(0o755)
    wt = tmp_path / "wt"
    gitio._git(["init", "-q", "-b", "main", str(wt)])
    gitio._git(["-C", str(wt), "commit", "-q", "--allow-empty", "-m", "probe"])
    gitio._git(["push", "-q", str(bare.git_dir), "HEAD:refs/heads/probe"], cwd=wt)
    env = dict(
        line.split("=", 1) for line in out.read_text().splitlines() if "=" in line
    )
    assert env.get("KEEP_ME") == "kept"  # the rest of the caller's environment is kept
    _clean(env)


@pytest.fixture
def recorded(routed, monkeypatch):
    seen: list[dict] = []

    def run(argv, *args, **kwargs):
        seen.append(dict(kwargs.get("env") or {}))
        return REAL_RUN(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    return seen


def test_every_git_env_builder_drops_the_routes(recorded, tmp_path):
    repo = gitio.Repo.init_bare(tmp_path / "m.git")
    snapshot._git_bytes(repo, ["rev-parse", "HEAD"])
    hardgit.git(repo.git_dir, "rev-parse", "HEAD")
    request.build_id()
    assert len(recorded) >= 5
    for env in recorded:
        assert env, "a git call without an explicit environment"
        _clean(env)
        assert env.get("KEEP_ME") == "kept"
