"""Memory v2.1 r5 (§2-§4): staging, the findings an episode's reader leaves for the writer.

Three readers write the same files under ``<state>/memory-staging/<episode>/``: the actor's episode-end fork (arm
B, :mod:`.integration.fork_worker`, which writes ``fork.json`` last), Sol analysts (arm C, :mod:`.analysts`, which
write ``analyst.json`` last) and the offline replay of arm B. A directory without its status file is still being
written and counts as no staging (a pass may batch an episode before its fork ends).

Staging is untrusted input. :func:`read` takes only regular files, never through a link, under the paths the fork
worker allows (:func:`.integration.fork_worker.allowed`), within its operational quota; anything else is listed as
refused with why. :func:`export` copies what was taken to the writer's ``/inputs/staging/<episode>/``. Nothing here
imports or runs a staged file.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from .integration.fork_worker import (
    ALLOWED,
    QUOTA_BYTES,
    QUOTA_FILES,
    allowed,
    parse_reply,
    write_files,
)

__all__ = [
    "ALLOWED",
    "QUOTA_BYTES",
    "QUOTA_FILES",
    "STAGING_DIR",
    "STATUS_FILES",
    "StagingView",
    "export",
    "parse_reply",
    "read",
    "write_files",
]

STAGING_DIR = "memory-staging"
STATUS_FILES = {
    "fork.json": "fork",
    "analyst.json": "sol_analyst",
    "replay_fork.json": "replay_fork",
}


@dataclass
class StagingView:
    status: str
    source: str
    files: dict[str, bytes] = field(default_factory=dict)
    refused: list[dict] = field(default_factory=list)


def _status(d: Path) -> tuple[str, str] | None:
    for name, source in STATUS_FILES.items():
        p = d / name
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if not stat.S_ISREG(st.st_mode) or st.st_size > 1024 * 1024:
            return f"error: unreadable {name}", source
        try:
            doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except ValueError:
            return f"error: unreadable {name}", source
        return (
            str(doc.get("status", "unknown"))[:200]
            if isinstance(doc, dict)
            else "unknown"
        ), source
    return None


def read(root: Path | str | None, eid: str) -> StagingView | None:
    """*eid*'s staging under *root* (``<state>/memory-staging``), or None: no directory, a link, or no status file
    yet."""
    if not root:
        return None
    d = Path(root) / eid
    try:
        if not stat.S_ISDIR(os.lstat(d).st_mode):
            return None
    except OSError:
        return None
    head = _status(d)
    if head is None:
        return None
    view = StagingView(status=head[0], source=head[1])
    total = 0
    for base, dirs, names in os.walk(d, followlinks=False):
        for name in sorted(dirs + names):
            p = Path(base) / name
            rel = p.relative_to(d).as_posix()
            if name in STATUS_FILES and Path(base) == d:
                continue
            st = os.lstat(p)
            if stat.S_ISDIR(st.st_mode):
                continue
            why = None
            if stat.S_ISLNK(st.st_mode):
                why = "a link"
            elif not stat.S_ISREG(st.st_mode):
                why = "not a regular file"
            else:
                why = allowed(rel)
            if why is None and (
                len(view.files) >= QUOTA_FILES or total + st.st_size > QUOTA_BYTES
            ):
                why = "operational quota reached"
            if why:
                view.refused.append({"path": rel[:200], "why": why})
                continue
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as fh:
                data = fh.read(QUOTA_BYTES + 1)
            total += len(data)
            view.files[rel] = data
        dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(base, x))]
    return view


def export(view: StagingView | None, dest: Path) -> dict:
    """Copy *view*'s files under *dest* (the writer's ``/inputs/staging/<episode>``); the batch-map entry."""
    if view is None:
        return {"status": "none"}
    for rel, data in view.files.items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return {
        "status": view.status,
        "source": view.source,
        "files": sorted(view.files),
        "refused": view.refused,
        "path": f"/inputs/staging/{dest.name}",
    }
