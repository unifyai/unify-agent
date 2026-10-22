"""The memory home's fixed paths, hardened git, the state file and the request lock (integration Task 18)."""

import subprocess

import pytest

from unify.memory.fingerprint import Generations
from unify.memory.gitio import GitError, Repo
from unify.memory.integration import hardgit
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State, acquire_lock, release_lock

MOD = 'def hello(apis, name):\n    """Say hi.\n\n    Effect: read\n    """\n    return f"hi {name}"\n'


def _seed(tmp_path, extra=None):
    mem = Repo.init_bare(tmp_path / "memory")
    base = mem.head()
    with mem.temp_checkout() as wt:
        (wt / "env/spotify").mkdir(parents=True)
        (wt / "env/spotify/__init__.py").write_text(MOD)
        for rel, text in (extra or {}).items():
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "seed", {})
    mem.fast_forward("main", sha, expected_old=base)
    return mem, sha


def _seed_library(tmp_path):
    """A memory repo whose main holds one library package in the library layout (the actor's export copies its
    modules, docstrings and notes only): memory/acct with ``users.user_id``."""
    from tests.memory import library_example_world as lw

    mem = Repo.init_bare(tmp_path / "memory")
    base = mem.head()
    with mem.temp_checkout() as wt:
        for rel, text in {
            lw.PACKAGE_INIT: '"""Accounts."""\n',
            lw.README: "Reading account responses.\n",
            lw.MODULE: lw.MODULE_SOURCE,
        }.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "seed", {})
    mem.fast_forward("main", sha, expected_old=base)
    return mem, sha


def test_paths_are_fixed_under_home(tmp_path):
    p = Paths.under(tmp_path)
    assert p.memory == tmp_path / "memory"
    assert p.checkout == tmp_path / "memory-checkout"
    assert p.episodes == tmp_path / "episodes.git"
    assert p.blobs == tmp_path / "episodes-blobs"
    assert p.evidence == tmp_path / "memory-evidence.sqlite"
    assert p.state_dir == tmp_path / "memory-state"
    assert p.lock == p.state_dir / "request.lock"
    assert p.state == p.state_dir / "state.json"
    assert p.errors == p.state_dir / "errors.jsonl"
    assert p.worktree_git == tmp_path / "worktree.git"


def test_hardgit_ignores_global_config_and_inherited_git_env(tmp_path, monkeypatch):
    mem, sha = _seed(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".gitconfig").write_text("[alias]\n\tfoo = log\n")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    assert hardgit.git(mem.git_dir, "rev-parse", "main").decode().strip() == sha
    with pytest.raises(GitError):
        hardgit.git(mem.git_dir, "foo")


def test_hardgit_timeout_is_a_git_error(tmp_path, monkeypatch):
    seen = {}

    def slow(argv, **kw):
        seen.update(kw)
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    monkeypatch.setattr(hardgit.subprocess, "run", slow)
    with pytest.raises(GitError, match="timed out"):
        hardgit.git(tmp_path, "status", timeout=0.2)
    assert seen["timeout"] == 0.2


def test_state_roundtrip_and_default(tmp_path):
    path = tmp_path / "memory-state" / "state.json"
    s = State.load(path)
    assert s.drift == set() and s.suspect == set()
    s.generations.observe({"venmo.me": {"shapes": ["{a:int}"], "errors": []}})
    s.generations.observe({"venmo.me": {"shapes": ["{b:int}"], "errors": []}})
    s.drift.add("venmo")
    s.suspect.update({"venmo", "spotify"})
    s.save()
    back = State.load(path)
    assert back.drift == {"venmo"} and back.suspect == {"spotify", "venmo"}
    assert back.generations.generation("venmo") == 1
    assert isinstance(back.generations, Generations)
    assert [p.name for p in path.parent.iterdir()] == ["state.json"]
    # the catalogue guide flag: absent until set (the v2 file is unchanged), then kept
    assert back.guide is False and "guide" not in path.read_text()
    back.guide = True
    back.save()
    assert State.load(path).guide is True and '"guide": true' in path.read_text()


def test_lock_is_exclusive_and_times_out(tmp_path):
    lock = tmp_path / "memory-state" / "request.lock"
    fd = acquire_lock(lock)
    try:
        with pytest.raises(TimeoutError):
            acquire_lock(lock, timeout_s=0.2)
    finally:
        release_lock(fd)
    release_lock(acquire_lock(lock, timeout_s=0.2))


