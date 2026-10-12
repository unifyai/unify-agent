"""Recorded office workspaces for replay: each episode's ``worktree_before`` commit, exported once from the run's
worktree repo into a cache folder beside the state, so replayed code that reads its working directory finds the
files it read when it was recorded.

An export is made the first time an episode's workspace is needed and kept; a failed export leaves nothing behind
and is retried on the next request. The helper processes (``git archive`` and ``tar``) get only
:func:`.mining.helper_env`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from .mining import helper_env

EXPORT_TIMEOUT_S = 300  # one export (operational: a stuck git or tar must end)


def export_tree(worktree_git: str, sha: str, dest: Path) -> Path | None:
    """The workspace at commit *sha* under *dest* (exported once; *dest* is kept), or None when it cannot be."""
    if dest.exists():
        return dest
    try:
        data = subprocess.run(
            ["git", "--git-dir", worktree_git, "archive", sha],
            capture_output=True,
            timeout=EXPORT_TIMEOUT_S,
            env=helper_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if data.returncode:
        return None
    tmp = dest.with_name(dest.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        untar = subprocess.run(
            ["tar", "x", "-C", str(tmp)],
            input=data.stdout,
            capture_output=True,
            timeout=EXPORT_TIMEOUT_S,
            env=helper_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        untar = None
    if untar is None or untar.returncode:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    tmp.rename(dest)
    return dest


def trees_for(
    load: Callable,
    episodes: Iterable[str],
    worktree_git: str | None,
    cache: Path | None,
) -> dict[str, Path]:
    """Episode id -> its exported workspace, for each of *episodes* that recorded one (``worktree_before``) and
    could be exported; {} without a worktree repo or a cache folder. An episode that cannot be loaded is left out.
    """
    if not worktree_git or cache is None:
        return {}
    out: dict[str, Path] = {}
    for eid in dict.fromkeys(str(e) for e in episodes):
        if not eid or "/" in eid or "\\" in eid or eid.startswith("."):
            continue  # a folder name under the cache, never a path
        try:
            sha = getattr(load(eid), "worktree_before", None)
        except (
            Exception
        ):  # noqa: BLE001 - an unknown episode has no workspace; the replay reports it
            continue
        if not sha:
            continue
        tree = export_tree(worktree_git, str(sha), Path(cache) / eid)
        if tree is not None:
            out[eid] = tree
    return out
