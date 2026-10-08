"""Thin git CLI wrapper used by the three memory-v2 histories. Fixed argv lists; never a shell."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator, Mapping, Sequence


class GitError(RuntimeError):
    pass


_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",  # no user config: hooks, fsmonitor, filters
    "GIT_AUTHOR_NAME": "unify-memory",
    "GIT_AUTHOR_EMAIL": "memory@unify.invalid",
    "GIT_COMMITTER_NAME": "unify-memory",
    "GIT_COMMITTER_EMAIL": "memory@unify.invalid",
}


#: Never handed to a git child, by exact name: memory v2's route for Sol's calls
#: (``unify.sandbox.HARNESS_ONLY_ENV``), kept here too for when ``unify.sandbox`` cannot be imported.
_HARNESS_ONLY = frozenset(
    {
        "UNIFY_MEMORY_V2_SOL_BASE_URL",
        "UNIFY_MEMORY_V2_SOL_TOKEN",
        "UNIFY_MEMORY_V2_SOL_TOKEN_FD",
    },
)


def git_child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of every git child memory v2 starts: this process's without credentials and
    the harness-only Sol route (``unify.sandbox.scrubbed_env``), without inherited ``GIT_*``, then
    *extra* (the call's own git settings). Git needs none of what is dropped; a git child is a process
    the token audit sees, so it holds nothing only the controller may hold."""
    try:
        from unify.sandbox import scrubbed_env

        base = scrubbed_env()
    except ImportError:  # outside the harness package: drop the route by name at least
        base = dict(os.environ)
    env = {
        k: v
        for k, v in base.items()
        if not k.startswith("GIT_") and k.upper() not in _HARNESS_ONLY
    }
    env.update(extra or {})
    return env


def _trailer_block(trailers: Mapping[str, str | Sequence[str]]) -> str:
    lines = []
    for key, val in trailers.items():
        for v in ([val] if isinstance(val, str) else list(val)):
            if "\n" in v:
                raise GitError(f"trailer {key} contains a newline")
            lines.append(f"{key}: {v}")
    return "\n".join(lines)


class Repo:
    def __init__(self, git_dir: Path, work_tree: Path | None = None) -> None:
        # Absolute: git runs with other working directories (temporary checkouts), where a relative path breaks.
        self.git_dir = Path(git_dir).absolute()
        self.work_tree = Path(work_tree).absolute() if work_tree else None

    # -- construction -------------------------------------------------------------------------------
    @classmethod
    def init_bare(cls, path: Path) -> "Repo":
        """A bare repo whose `main` starts at an empty root commit "init"."""
        path = Path(path).absolute()
        _git(["init", "--bare", "-q", "-b", "main", str(path)])
        repo = cls(path)
        tmp = Path(tempfile.mkdtemp(prefix="memv2-init-"))
        try:
            wt = tmp / "wt"
            _git(["init", "-q", "-b", "main", str(wt)])
            repo.commit_all(wt, "init", {}, allow_empty=True)
            _git(["push", "-q", str(path), "HEAD:refs/heads/main"], cwd=wt)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return repo

    @classmethod
    def init_snapshot(cls, git_dir: Path, work_tree: Path) -> "Repo":
        """A separate git dir tracking `work_tree`; no `.git` ever appears inside the work tree."""
        _git(["--git-dir", str(git_dir), "init", "-q", "-b", "main"])
        _git(["--git-dir", str(git_dir), "config", "core.bare", "false"])
        return cls(git_dir, work_tree)

    # -- plumbing -------------------------------------------------------------------------------------
    def run(self, *args: str, input: str | None = None, cwd: Path | None = None) -> str:
        base = ["--git-dir", str(self.git_dir)]
        if self.work_tree is not None and cwd is None:
            base += ["--work-tree", str(self.work_tree)]
        return _git(base + list(args), input=input, cwd=cwd)

    def head(self, ref: str = "main") -> str:
        return self.run("rev-parse", f"refs/heads/{ref}").strip()

    @contextlib.contextmanager
    def temp_checkout(self, rev: str = "main") -> Iterator[Path]:
        tmp = Path(tempfile.mkdtemp(prefix="memv2-wt-"))
        wt = tmp / "wt"
        self.run("worktree", "add", "-q", "--detach", str(wt), rev)
        try:
            yield wt
        finally:
            with contextlib.suppress(GitError):
                self.run("worktree", "remove", "--force", str(wt))
            shutil.rmtree(tmp, ignore_errors=True)
            with contextlib.suppress(GitError):
                self.run("worktree", "prune")

    def commit_all(
        self,
        path: Path,
        message: str,
        trailers: Mapping[str, str | Sequence[str]],
        allow_empty: bool = False,
    ) -> str:
        # --force: a .gitignore in the tree never hides a file from the commit (the gate refuses .gitignore)
        _git(["add", "-A", "--force"], cwd=path)
        if not allow_empty and not _git(["status", "--porcelain"], cwd=path).strip():
            return _git(["rev-parse", "HEAD"], cwd=path).strip()
        block = _trailer_block(trailers)
        body = message.strip() + ("\n\n" + block if block else "") + "\n"
        args = ["-c", "commit.gpgsign=false", "commit", "-q", "-F", "-"] + (
            ["--allow-empty"] if allow_empty else []
        )
        _git(args, input=body, cwd=path)
        return _git(["rev-parse", "HEAD"], cwd=path).strip()

    def fast_forward(self, branch: str, new_sha: str, expected_old: str) -> None:
        current = self.head(branch)
        if current != expected_old:
            raise GitError(
                f"{branch} moved: expected {expected_old[:12]}, found {current[:12]}",
            )
        try:
            self.run("merge-base", "--is-ancestor", expected_old, new_sha)
        except GitError as exc:
            raise GitError(
                f"{new_sha[:12]} does not descend from {expected_old[:12]}",
            ) from exc
        self.run("update-ref", f"refs/heads/{branch}", new_sha, expected_old)

    def snapshot(self, message: str) -> str:
        if self.work_tree is None:
            raise GitError("snapshot needs a work tree")
        self.run("add", "-A")
        self.run(
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            message,
        )
        return self.run("rev-parse", "HEAD").strip()

    def add_note(self, sha: str, text: str, ref: str = "signals") -> None:
        if "\n" in text:
            raise GitError("a note line must be a single line")
        self.run("notes", f"--ref={ref}", "append", "-m", text, sha)

    def append_note_lines(self, sha: str, lines: Sequence[str], ref: str) -> None:
        """Append *lines* (each a single line) to *sha*'s note on *ref* in one call; the text goes by stdin."""
        if any("\n" in ln or "\r" in ln for ln in lines):
            raise GitError("a note line must be a single line")
        if lines:
            self.run(
                "notes",
                f"--ref={ref}",
                "append",
                "-F",
                "-",
                sha,
                input="\n".join(lines) + "\n",
            )

    def notes(self, sha: str, ref: str = "signals") -> list[str]:
        try:
            out = self.run("notes", f"--ref={ref}", "show", sha)
        except GitError:
            return []
        return [ln for ln in out.splitlines() if ln.strip()]

    def changed_paths(self, a: str, b: str) -> list[str]:
        return [p for p in self.run("diff", "--name-only", a, b).splitlines() if p]

    def diff(self, a: str, b: str) -> str:
        return self.run("diff", a, b)

    def show(self, rev: str, path: str) -> bytes:
        return self.run("show", f"{rev}:{path}").encode()

    def log_shas(self, ref: str = "main") -> list[str]:
        return self.run("rev-list", "--reverse", f"refs/heads/{ref}").split()

    def blame_lines(self, rev: str, path: str) -> list[str]:
        out = self.run("blame", "--porcelain", rev, "--", path)
        shas = []
        for ln in out.splitlines():
            parts = ln.split(" ")
            if (
                len(parts) >= 3
                and len(parts[0]) == 40
                and all(c in "0123456789abcdef" for c in parts[0])
            ):
                shas.append((int(parts[2]), parts[0]))
        return [s for _, s in sorted(shas)]


#: Every call: no hooks, no fsmonitor and no user-level excludes file, whatever a repo's own config says.
_HARD = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.excludesFile=/dev/null",
)
GIT_TIMEOUT_S = 120


def _git(args: list[str], input: str | None = None, cwd: Path | None = None) -> str:
    env = git_child_env(_ENV)
    try:
        proc = subprocess.run(
            ["git", *_HARD, *args],
            input=input,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {' '.join(args[:3])}… timed out") from exc
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:3])}… failed: {proc.stderr.strip()[:400]}")
    return proc.stdout
