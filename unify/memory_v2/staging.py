"""Staging: the findings an episode's reader leaves for the writer.

Readers write the same files under ``<state>/memory-staging/<episode>/``: the actor's episode-end fork
(:mod:`.integration.fork_worker`, which writes ``fork.json`` last), writer-side analysts (:mod:`.analysts`, which
write ``analyst.json`` last) and the offline replay of the fork. A directory without its status file is still being
written and counts as no staging; with the fork on, admission (:func:`.integration.fork.settled`) keeps such an
episode out of a pass batch until it settles.

Staging is untrusted input. :func:`read` takes only regular files, never through a link, under the paths the fork
worker allows (:func:`.integration.fork_worker.allowed`), within its operational quota; anything else is listed as
refused with why. The fork's own record files (``fork-blocks.json``, the raw ``fork-reply.json``, ``fork.lock``) are
never staged files; every block the fork's parser refused or dropped (``fork-blocks.json``) is listed beside the
read-time refusals, so the writer sees what did not arrive and why. :func:`export` copies what was taken to the
writer's ``/inputs/staging/<episode>/``. Nothing here imports or runs a staged file.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

from .integration.fork_worker import (
    ALLOWED,
    BLOCKS_JSON,
    QUOTA_BYTES,
    QUOTA_FILES,
    RECORD_FILES,
    allowed,
    parse_reply,
    write_files,
)

__all__ = [
    "ALLOWED",
    "RECORD_FILES",
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
    "analyst.json": "writer_analyst",
    "replay_fork.json": "replay_fork",
}


@dataclass
class StagingView:
    status: str
    source: str
    files: dict[str, bytes] = field(default_factory=dict)
    refused: list[dict] = field(default_factory=list)
    #: the fork's block counts from ``fork-blocks.json`` (blocks, parsed, refused, dropped), when it wrote one
    blocks: dict | None = None


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
            if Path(base) == d and (name in STATUS_FILES or name in RECORD_FILES):
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
                view.refused.append({"path": rel[:200], "why": why, "stage": "read"})
                continue
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as fh:
                data = fh.read(QUOTA_BYTES + 1)
            total += len(data)
            view.files[rel] = data
        dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(base, x))]
    _parse_record(d, view)
    return view


def _parse_record(d: Path, view: StagingView) -> None:
    """Add the fork's parse-time refusals and drops (``fork-blocks.json``) to *view*'s refused list."""
    p = d / BLOCKS_JSON
    try:
        st = os.lstat(p)
    except OSError:
        return
    if not stat.S_ISREG(st.st_mode) or st.st_size > 8 * 1024 * 1024:
        view.refused.append({"path": BLOCKS_JSON, "why": "unreadable block record", "stage": "parse"})
        return
    try:
        doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        view.refused.append({"path": BLOCKS_JSON, "why": "unreadable block record", "stage": "parse"})
        return
    if not isinstance(doc, dict):
        return
    if isinstance(doc.get("counts"), dict):
        view.blocks = {k: v for k, v in doc["counts"].items() if isinstance(v, int)}
    for b in doc.get("blocks") or []:
        if not isinstance(b, dict) or b.get("status") not in ("refused", "dropped"):
            continue
        view.refused.append(
            {
                "path": str(b.get("path"))[:200] if b.get("path") is not None else None,
                "why": str(b.get("reason"))[:200],
                "status": b.get("status"),
                "form": b.get("form"),
                "block": b.get("index"),
                "stage": "parse",
            },
        )


def export(view: StagingView | None, dest: Path) -> dict:
    """Copy *view*'s files under *dest* (the writer's ``/inputs/staging/<episode>``); the batch-map entry."""
    if view is None:
        return {"status": "none"}
    for rel, data in view.files.items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    entry = {
        "status": view.status,
        "source": view.source,
        "files": sorted(view.files),
        "refused": view.refused,
        "path": f"/inputs/staging/{dest.name}",
    }
    if view.blocks is not None:
        entry["blocks"] = view.blocks
    return entry
