"""One writer for memory ``main``, and the head requests may pin (memory v2.1: spec §4.4, §6; Amendment A of P7).

* **One writer.** Every v2.1 move of ``main`` (a pass's merge, a status commit, publishing) holds
  :func:`main_lock`, an ``flock`` on ``<memory>.main.flock`` beside the bare repo.
* **Retargeting.** A pass's commit is one commit on the parent it was built on. If ``main`` has since moved only
  by harness status commits (:func:`is_status_commit`: one parent, the same tree as that parent, subject
  ``status: `` and ``Status:`` trailers), the commit is re-made with the same tree and message on the new head and
  the gate is not re-run: no code changed. If any other commit moved ``main``, :class:`StaleParent` is raised and
  nothing is written; the gate refuses the pass and keeps its patch as a draft.
* **The served head.** v2.1 requests pin ``refs/heads/served``, never ``main``. :func:`publish` advances it to the
  newest first-parent commit of ``main`` that carries its own item-records map on ``refs/notes/items`` (P5), so no
  request pins a commit whose statuses are still to be written, and every request on one pin sees the same
  statuses (cache rule). It is created at ``main``'s head by the first v2.1 write, before ``main`` moves.
* **Lag is visible** (Amendment C): :func:`served_lag` counts the first-parent commits ``served`` is behind
  ``main``; the driver reports it on every v2.1 end event and warns when it is above 0 after publishing.
* **Copies carry the notes.** A plain ``git clone`` does not fetch ``refs/notes/items``. :func:`archive` writes a
  bundle of every ref and checks that ``main``, ``served`` and the notes are in it; restore with
  ``git clone --mirror <bundle>``.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path
from typing import Callable, Iterator

from .gitio import GitError, Repo

SERVED = "served"
NOTES_REF = "refs/notes/items"  # P5's item_records.NOTES_REF ("items") as a full ref
LOCK_TIMEOUT_S = 600.0
STATUS_SUBJECT = "status: "  # P5's MemoryRepo.status_commit subject
_SEP = "\x1e"


class StaleParent(GitError):
    """``main`` moved past a pass's parent by a commit that changed the tree: the pass must be refused."""


def lock_path(repo: Repo) -> Path:
    git_dir = Path(repo.git_dir)
    return git_dir.with_name(git_dir.name + ".main.flock")


@contextlib.contextmanager
def main_lock(repo: Repo, timeout_s: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """Hold the one writer lock of *repo*'s ``main``. Not re-entrant: a holder never takes it again."""
    path = lock_path(repo)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    deadline = time.monotonic() + float(timeout_s)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"memory main lock {path.name} held for {timeout_s}s",
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def _has_ref(repo: Repo, ref: str) -> bool:
    try:
        repo.run("rev-parse", "--verify", "--quiet", ref)
        return True
    except GitError:
        return False


def served_head(repo: Repo) -> str:
    """The commit a v2.1 request pins: ``served``, or ``main`` while no v2.1 write has happened yet."""
    return repo.head(SERVED) if _has_ref(repo, f"refs/heads/{SERVED}") else repo.head()


def ensure_served(repo: Repo) -> None:
    """Create ``served`` at ``main``'s head when it is missing. Call under :func:`main_lock`, before ``main``
    moves."""
    if not _has_ref(repo, f"refs/heads/{SERVED}"):
        repo.run("update-ref", f"refs/heads/{SERVED}", repo.head(), "0" * 40)


def _parents(repo: Repo, sha: str) -> list[str]:
    return repo.run("rev-list", "--parents", "-n", "1", sha).split()[1:]


def is_status_commit(repo: Repo, sha: str) -> bool:
    """A harness status commit: one parent, the same tree as it, subject ``status: `` and a ``Status:`` trailer."""
    fmt = "%P%x1e%T%x1e%s%x1e%(trailers:key=Status,valueonly,separator=%x1f)"
    fields = repo.run("show", "-s", f"--format={fmt}", sha).rstrip("\n").split(_SEP)
    if len(fields) != 4:
        return False
    parents, tree, subject, status = fields
    ps = parents.split()
    if len(ps) != 1 or not subject.startswith(STATUS_SUBJECT) or not status.strip():
        return False
    return repo.run("rev-parse", f"{ps[0]}^{{tree}}").strip() == tree


def moved_only_by_status(repo: Repo, parent: str, head: str) -> bool:
    """Whether *head* descends from *parent* through status commits alone (deterministic: no tree changed)."""
    if parent == head:
        return True
    try:
        repo.run("merge-base", "--is-ancestor", parent, head)
    except GitError:
        return False
    between = repo.run("rev-list", f"{parent}..{head}").split()
    return bool(between) and all(is_status_commit(repo, c) for c in between)


def _remake(repo: Repo, sha: str, onto: str) -> str:
    """*sha*'s tree and message as one commit on *onto*."""
    tree = repo.run("rev-parse", f"{sha}^{{tree}}").strip()
    # the raw message, byte for byte (``show --format=%B`` appends a newline that commit-tree would keep)
    message = repo.run("cat-file", "commit", sha).split("\n\n", 1)[1]
    return repo.run(
        "-c",
        "commit.gpgsign=false",
        "commit-tree",
        tree,
        "-p",
        onto,
        "-F",
        "-",
        input=message,
    ).strip()


def land(
    repo: Repo,
    sha: str,
    parent: str,
    *,
    timeout_s: float = LOCK_TIMEOUT_S,
) -> str:
    """Move ``main`` from *parent* to *sha* (one commit on *parent*) as the one writer, and return the commit that
    landed: *sha*, or its remake on a head that moved only by status commits. :class:`StaleParent` when ``main``
    moved by any other commit, or *sha* is not one commit on *parent*; nothing is written then.
    """
    with main_lock(repo, timeout_s):
        ensure_served(repo)
        head = repo.head()
        if head != parent:
            if not moved_only_by_status(repo, parent, head):
                raise StaleParent(
                    f"stale: main moved from {parent[:12]} to {head[:12]} by a commit that changed the library",
                )
            if _parents(repo, sha) != [parent]:
                raise StaleParent(
                    f"stale: {sha[:12]} is not one commit on {parent[:12]}",
                )
            sha = _remake(repo, sha, head)
        repo.fast_forward("main", sha, expected_old=head)
        return sha


def publish(
    repo: Repo,
    *,
    has_records: Callable[[str], bool] | None = None,
    scan: int = 200,
) -> str:
    """Advance ``served`` to the newest first-parent commit of ``main``, back to ``served``, that has its own
    item-records map, and return ``served``'s commit. Run after the maps are written (the end of
    ``run_due_passes``, after P5's ``after_passes``)."""
    check = has_records
    if check is None:
        from .item_records import read_records  # P5

        check = lambda c: read_records(repo, c) is not None  # noqa: E731
    with main_lock(repo):
        ensure_served(repo)
        served = repo.head(SERVED)
        for c in repo.run(
            "rev-list",
            "--first-parent",
            f"--max-count={int(scan)}",
            "main",
        ).split():
            if c == served:
                break
            if check(c):
                repo.run(
                    "merge-base",
                    "--is-ancestor",
                    served,
                    c,
                )  # GitError: never move served sideways
                repo.run("update-ref", f"refs/heads/{SERVED}", c, served)
                return c
        return served


def served_lag(repo: Repo) -> int:
    """How many first-parent commits ``served`` is behind ``main`` (P7 Amendment C; 0 while there is no
    ``served``). Above 0 after :func:`publish`: commits whose item records were not written, which no request
    can pin yet."""
    if not _has_ref(repo, f"refs/heads/{SERVED}"):
        return 0
    return int(
        repo.run(
            "rev-list",
            "--first-parent",
            "--count",
            f"refs/heads/{SERVED}..main",
        ).strip(),
    )


def archive(repo: Repo, dest: Path) -> dict:
    """A bundle of every ref of the memory repo at *dest*, verified to hold ``main``, ``served`` and
    ``refs/notes/items`` wherever the repo has them (a plain clone drops the notes). Restore with
    ``git clone --mirror <dest>``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo.run("bundle", "create", str(dest), "--all")
    repo.run("bundle", "verify", str(dest))
    refs = {
        ln.split()[1]
        for ln in repo.run("bundle", "list-heads", str(dest)).splitlines()
        if len(ln.split()) == 2
    }
    need = {
        r
        for r in ("refs/heads/main", f"refs/heads/{SERVED}", NOTES_REF)
        if _has_ref(repo, r)
    }
    missing = sorted(need - refs)
    if missing:
        raise GitError(f"archive {dest.name} lacks {missing}")
    return {"path": str(dest), "refs": sorted(refs)}
