"""Every git child memory v2 starts: no harness-only Sol route or credential in its environment, and no hook runs.

A stand-in ``git`` first on ``PATH`` records each child's environment (``/proc/self/environ`` as the token
audit reads it) and then runs the real git, so each spawn site is checked through its real call.
"""

import os
import shutil
import stat

import pytest

from unify.memory_v2 import snapshot
from unify.memory_v2.gitio import Repo, git_child_env
from unify.memory_v2.integration import hardgit
from unify.memory_v2.integration import request as request_mod

REAL_GIT = shutil.which("git")
needs_git = pytest.mark.skipif(REAL_GIT is None, reason="git required")
FAKE_TOKEN = "tok-" + "x" * 32  # pragma: allowlist secret
FAKE_KEY = "sk-or-v1-" + "a" * 64  # pragma: allowlist secret
ROUTE = {
    "UNIFY_MEMORY_V2_SOL_BASE_URL": "http://127.0.0.1:9/sol/v1",
    "UNIFY_MEMORY_V2_SOL_TOKEN": FAKE_TOKEN,
    "UNIFY_MEMORY_V2_SOL_TOKEN_FD": "9",
}
SECRET = {"OPENROUTER_API_KEY": FAKE_KEY}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """The environments of the git children started from now on, one dict per child."""
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "git").write_text(
        "#!/bin/sh\n"
        f'tr "\\000" "\\n" < /proc/self/environ > "{dumps}/$$.env"\n'
        f'exec {REAL_GIT} "$@"\n',
    )
    (shim / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ.get('PATH', '')}")
    for k, v in {**ROUTE, **SECRET, "GIT_DIR": str(tmp_path / "elsewhere")}.items():
        monkeypatch.setenv(k, v)

    def envs() -> list[dict[str, str]]:
        out = []
        for f in sorted(dumps.glob("*.env")):
            pairs = [ln.split("=", 1) for ln in f.read_text().splitlines() if "=" in ln]
            out.append(dict(pairs))
        return out

    return envs


def _clean(env: dict[str, str]) -> bool:
    return not (set(env) & (set(ROUTE) | set(SECRET))) and env.get("GIT_DIR") is None


def test_git_child_env_drops_the_route_credentials_and_inherited_git_settings(
    monkeypatch,
):
    for k, v in {**ROUTE, **SECRET, "GIT_DIR": "/x", "KEEP_ME": "1"}.items():
        monkeypatch.setenv(k, v)
    env = git_child_env({"GIT_TERMINAL_PROMPT": "0"})
    assert _clean(env)
    assert env["KEEP_ME"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0" and "PATH" in env


@needs_git
def test_every_spawn_site_starts_git_without_the_route(tmp_path, recorded):
    repo = Repo.init_bare(tmp_path / "memory.git")  # gitio._git, several children
    base = repo.head()
    with repo.temp_checkout() as wt:
        (wt / "a.txt").write_text("a\n")
        sha = repo.commit_all(wt, "add a", {})
    repo.fast_forward("main", sha, expected_old=base)
    snapshot._git_bytes(repo, ["rev-parse", "main"])  # snapshot
    hardgit.git(repo.git_dir, "rev-parse", "main")  # hardgit (checkout_diff's)
    request_mod.build_id()  # the harness checkout's commit
    envs = recorded()
    assert len(envs) >= 5
    assert all(_clean(e) for e in envs), [
        sorted(set(e) & (set(ROUTE) | set(SECRET))) for e in envs
    ]


@needs_git
def test_a_planted_hook_never_runs(tmp_path):
    marker = tmp_path / "hook-ran"
    repo = Repo.init_bare(tmp_path / "memory.git")
    hooks = tmp_path / "planted-hooks"
    hooks.mkdir()
    for name in (
        "pre-commit",
        "post-commit",
        "reference-transaction",
        "post-update",
        "pre-receive",
        "update",
        "post-receive",
    ):
        h = hooks / name
        h.write_text(f'#!/bin/sh\necho {name} >> "{marker}"\nexit 0\n')
        h.chmod(h.stat().st_mode | stat.S_IXUSR)
        shutil.copy2(h, repo.git_dir / "hooks" / name)  # the repo's own hooks directory
    # a repo-local config naming the planted directory, as a writable config would
    hardgit.git(repo.git_dir, "config", "core.hooksPath", str(hooks))
    base = repo.head()
    with (
        repo.temp_checkout() as wt,
    ):  # a worktree of the planted repo: its hooks and config apply
        (wt / "b.txt").write_text("b\n")
        sha = repo.commit_all(wt, "add b", {})
    repo.fast_forward("main", sha, expected_old=base)
    assert repo.head() == sha
    repo.add_note(sha, "n")
    hardgit.git(repo.git_dir, "update-ref", "refs/heads/probe", repo.head())
    assert not marker.exists()
