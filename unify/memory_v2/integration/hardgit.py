"""git for harness-side repos over trees model code may have written: no hooks, no global or system config, bounded."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..gitio import GitError, git_child_env

_HARD = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    "protocol.allow=never",
)


def git(
    git_dir: Path,
    *args: str,
    work_tree: Path | None = None,
    env: dict | None = None,
    input: bytes | None = None,
    cwd: Path | None = None,
    timeout: float = 120.0,
) -> bytes:
    e = git_child_env(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            **(env or {}),
        },
    )
    argv = [
        "git",
        *_HARD,
        "--git-dir",
        str(git_dir),
        *(["--work-tree", str(work_tree)] if work_tree else []),
        *args,
    ]
    try:
        p = subprocess.run(
            argv,
            input=input,
            env=e,
            cwd=cwd or work_tree,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0] if args else ''} timed out") from exc
    if p.returncode != 0:
        raise GitError(
            f"git {' '.join(args[:2])} failed: {p.stderr.decode(errors='replace')[:400]}",
        )
    return p.stdout
