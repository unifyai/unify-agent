"""Reads and writes of the memory stores that only tests make: the harness itself never calls them, so they live with
the tests (each is the SQL or git call the store method it replaces made, unchanged)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from unify.memory.gitio import Repo, _git


# --- the evidence store ----------------------------------------------------------------------------------------------


def seq_of(ev: Any, eid: str) -> int:
    """The index position of episode *eid*; KeyError if it is not indexed."""
    row = ev.db.execute("SELECT seq FROM episodes WHERE episode_id=?", (eid,)).fetchone()
    if row is None:
        raise KeyError(eid)
    return int(row[0])


def experience_of(ev: Any, eid: str) -> tuple[int, str]:
    """(experience tokens, counter) recorded for *eid*; KeyError if none."""
    row = ev.db.execute("SELECT tokens, counter FROM experience WHERE episode_id=?", (eid,)).fetchone()
    if row is None:
        raise KeyError(eid)
    return int(row[0]), str(row[1])


def regime_record(ev: Any, eid: str) -> tuple[str, str] | None:
    """``(regime, basis)`` inferred for *eid*, or None when its regime was never inferred."""
    if not ev._has_table("episode_regimes"):
        return None
    row = ev.db.execute("SELECT regime, basis FROM episode_regimes WHERE episode_id=?", (eid,)).fetchone()
    return (row[0], row[1]) if row is not None else None


def add_cover(ev: Any, item: str, eid: str, action_index: int) -> None:
    """A cover row as a landed merge before rev 2.1 recorded one (the store still reads them: ``covers_of``)."""
    with ev.db:
        ev.db.execute("INSERT OR IGNORE INTO covers VALUES(?,?,?)", (item, eid, action_index))


def covered(ev: Any) -> set[tuple[str, int]]:
    """Every recorded cover's ``(episode_id, action_index)``."""
    return {(r[0], int(r[1])) for r in ev.db.execute("SELECT episode_id, action_index FROM covers")}


# --- the attention ledger --------------------------------------------------------------------------------------------


def guaranteed(report: Any) -> bool:
    """The ledger's guarantee at a run's end: every entry is closed, unreadable or not reached."""
    return all(k.open == 0 for k in report.kinds.values())


# --- git ------------------------------------------------------------------------------------------------------------


def init_snapshot(git_dir: Path, work_tree: Path) -> Repo:
    """A separate git dir tracking *work_tree*; no ``.git`` ever appears inside the work tree."""
    _git(["--git-dir", str(git_dir), "init", "-q", "-b", "main"])
    _git(["--git-dir", str(git_dir), "config", "core.bare", "false"])
    return Repo(git_dir, work_tree)


def changed_paths(repo: Repo, a: str, b: str) -> list[str]:
    return [p for p in repo.run("diff", "--name-only", a, b).splitlines() if p]


def blame_lines(repo: Repo, rev: str, path: str) -> list[str]:
    """The commit of each line of *path* at *rev*, in line order."""
    out = repo.run("blame", "--porcelain", rev, "--", path)
    shas = []
    for ln in out.splitlines():
        parts = ln.split(" ")
        if len(parts) >= 3 and len(parts[0]) == 40 and all(c in "0123456789abcdef" for c in parts[0]):
            shas.append((int(parts[2]), parts[0]))
    return [s for _, s in sorted(shas)]
