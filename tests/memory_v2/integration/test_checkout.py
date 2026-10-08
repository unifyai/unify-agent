"""The per-request export of memory main, its diff, the state file and the request lock (integration Task 18)."""

import os
import subprocess

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.fingerprint import Generations
from unify.memory_v2.gitio import GitError, Repo
from unify.memory_v2.integration import hardgit
from unify.memory_v2.integration.checkout import (
    checkout_diff,
    export_checkout,
    remove_checkout,
)
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State, acquire_lock, release_lock

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


def test_paths_are_fixed_under_home(tmp_path):
    p = Paths.under(tmp_path)
    assert p.memory == tmp_path / "memory"
    assert p.checkout == tmp_path / "memory-checkout"
    assert p.episodes == tmp_path / "episodes.git"
    assert p.blobs == tmp_path / "episodes-blobs"
    assert p.evidence == tmp_path / "memory-evidence.sqlite"
    assert p.state_dir == tmp_path / "memory-v2"
    assert p.lock == p.state_dir / "request.lock"
    assert p.state == p.state_dir / "state.json"
    assert p.errors == p.state_dir / "errors.jsonl"
    assert p.worktree_git == tmp_path / "worktree.git"


def test_export_has_no_git_and_diff_sees_new_changed_and_ignored(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "memory-checkout"
    export_checkout(mem.git_dir, sha, dest)
    assert (dest / "env/spotify/__init__.py").read_text() == MOD
    assert not (dest / ".git").exists()
    (dest / "env/spotify/scratch.py").write_text("x = 1\n")
    (dest / "env/spotify/__init__.py").write_text(MOD + "\nY = 2\n")
    (dest / ".gitignore").write_text("hidden.py\n")
    (dest / "hidden.py").write_text("y = 2\n")
    (dest / "env/spotify/__pycache__").mkdir()
    (dest / "env/spotify/__pycache__/a.pyc").write_bytes(b"\0")
    (dest / "top.pyc").write_bytes(b"\0")
    d = checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b"))
    assert "scratch.py" in d and "hidden.py" in d and "+Y = 2" in d
    assert "__pycache__" not in d and "top.pyc" not in d
    assert mem.head() == sha  # the memory repo is untouched


def test_unchanged_export_diffs_empty(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    assert checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b")) == ""


def test_export_ignores_gitattributes_export_rules(tmp_path):
    mem, sha = _seed(
        tmp_path,
        {".gitattributes": "env/** export-ignore\n"},
    )
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    assert (dest / "env/spotify/__init__.py").read_text() == MOD


def test_planted_git_file_never_runs_code(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    evil = tmp_path / "evil"
    evil.mkdir()
    marker = tmp_path / "ran"
    (evil / "config").write_text(f"[core]\n\tfsmonitor = touch {marker}\n")
    (dest / ".git").write_text(f"gitdir: {evil}\n")
    checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b"))
    assert not marker.exists()


def test_planted_gitattributes_filter_never_runs(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    (dest / ".gitattributes").write_text("*.py filter=evil diff=evil\n")
    (dest / "new.py").write_text("z = 3\n")
    d = checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b"))
    assert "new.py" in d and "+z = 3" in d


def test_symlink_is_recorded_not_followed(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    secret = tmp_path / "secret.txt"
    secret.write_text("SENTINEL-link-body\n")
    os.symlink(secret, dest / "link")
    d = checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b"))
    assert "link" in d and "SENTINEL-link-body" not in d


def test_reexport_discards_previous_writes(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    (dest / "junk.py").write_text("1")
    export_checkout(mem.git_dir, sha, dest)
    assert not (dest / "junk.py").exists()
    remove_checkout(dest)
    assert not dest.exists()


def test_large_diff_is_capped_into_a_blob(tmp_path):
    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    (dest / "big.txt").write_text("z" * 400_000)
    blobs = BlobStore(tmp_path / "b")
    d = checkout_diff(mem.git_dir, sha, dest, blobs, cap=1000)
    assert len(d) < 2000 and "truncated" in d
    blob = d.rsplit("blob ", 1)[1].split("]")[0]
    assert b"big.txt" in blobs.get(blob)


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
    path = tmp_path / "memory-v2" / "state.json"
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
    lock = tmp_path / "memory-v2" / "request.lock"
    fd = acquire_lock(lock)
    try:
        with pytest.raises(TimeoutError):
            acquire_lock(lock, timeout_s=0.2)
    finally:
        release_lock(fd)
    release_lock(acquire_lock(lock, timeout_s=0.2))


def test_untouched_generated_files_stay_out_of_the_diff(tmp_path):
    from unify.memory_v2.catalogue import write_generated

    mem, sha = _seed(tmp_path)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    generated = write_generated(dest)
    assert (dest / "README.md").is_file() and (dest / ".memory/catalog.json").is_file()
    blobs = BlobStore(tmp_path / "b")
    assert checkout_diff(mem.git_dir, sha, dest, blobs, generated=generated) == ""
    # without the list, the same files are new to the commit
    assert "README.md" in checkout_diff(mem.git_dir, sha, dest, blobs)
    # a generated file the request changed is recorded; the others stay out
    (dest / "README.md").write_text("my notes on the library\n")
    (dest / ".memory/proposal.txt").write_text("merge two readers\n")
    d = checkout_diff(mem.git_dir, sha, dest, blobs, generated=generated)
    assert "+my notes on the library" in d and "proposal.txt" in d
    assert "catalog.json" not in d and "memory.py" not in d
    # a link put in place of a generated file is recorded, never followed
    (dest / "memory.py").unlink()
    secret = tmp_path / "secret.txt"
    secret.write_text("SENTINEL-generated-link\n")
    os.symlink(secret, dest / "memory.py")
    d = checkout_diff(mem.git_dir, sha, dest, blobs, generated=generated)
    assert "memory.py" in d and "SENTINEL-generated-link" not in d


def test_an_older_commits_bytecode_is_never_exported(tmp_path):
    """v2.1 I4: a commit from before the gate refused what manifest.unsafe_path names (bytecode, extensions,
    root entries outside the layout) is exported without them, so a cell can never import a sourceless
    ``.pyc`` from the library; the untouched export still diffs empty (what was left out is not something
    the request deleted).
    """
    # not a dict literal: test_gate_layout's fixture sweep reads those as libraries the gate must admit
    planted = dict.fromkeys(
        [
            "env/__init__.pyc",
            "json.pyc",
            "sitecustomize.pyc",
            "env/spotify/__pycache__/__init__.cpython-312.pyc",
            "env/spotify/fast.so",
            "pytest_shadow.txt",
        ],
        "x",
    )
    mem = Repo.init_bare(tmp_path / "memory")
    base = mem.head()
    with mem.temp_checkout() as wt:
        (wt / "env/spotify/__pycache__").mkdir(parents=True)
        (wt / "env/spotify/__init__.py").write_text(MOD)
        for rel, text in planted.items():
            (wt / rel).write_text(text)
        sha = mem.commit_all(wt, "old", {})
    mem.fast_forward("main", sha, expected_old=base)
    dest = tmp_path / "co"
    export_checkout(mem.git_dir, sha, dest)
    exported = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert exported == ["env/spotify/__init__.py"]
    assert checkout_diff(mem.git_dir, sha, dest, BlobStore(tmp_path / "b")) == ""
