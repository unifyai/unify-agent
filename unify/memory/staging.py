"""Staging: the findings an episode's reader leaves for the writer.

Readers write the same files under ``<state>/memory-staging/<episode>/``: the actor's episode-end fork
(:mod:`.integration.fork_worker`, which writes ``fork.json`` last) and the offline replay of the fork (which writes
``replay_fork.json`` last). Writer-launched helpers write the same format to their own folder (:mod:`.helpers`). A directory without its status file is still being
written and counts as no staging; with the fork on, admission (:func:`.integration.fork.settled`) keeps such an
episode out of a pass batch until it settles.

Staging is untrusted input. :func:`read` takes only regular files, never through a link, under the paths the fork
worker allows (:func:`.integration.fork_worker.allowed`), within its operational quota; anything else is listed as
refused with why. The fork's own record files (``fork-blocks.json``, the raw ``fork-reply.json``, ``fork.lock``) are
never staged files; every block the fork's parser refused or dropped (``fork-blocks.json``) is listed beside the
read-time refusals, so the writer sees what did not arrive and why. :func:`export` copies what was taken to the
writer's ``/inputs/staging/<episode>/``. Nothing here imports or runs a staged file.

**A reader that did not finish hands the writer nothing.** Only a status of ``ok`` (or ``ok: <note>``, such as the
operational quota) means the reader finished. For any other status (a fork that timed out, was lost, errored or was
refused) the view holds no files: each file it left is listed as refused with :data:`WITHHELD_WHY` and kept on the
host for the record, and the batch-map row's status reads ``no reader output: <status>`` (:data:`NO_OUTPUT`), which
the writer brief explains.
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
#: The batch-map row's status prefix when the reader did not finish: nothing of the episode's staging is handed on.
NO_OUTPUT = "no reader output"
#: Why a file an unfinished reader left is not handed on.
WITHHELD_WHY = "the reader did not finish"
#: The staged summary of the episode (:data:`.integration.fork_worker.SUMMARY_FILE`).
SUMMARY_FILE = "summary.md"
#: The file that marks a staging dir settled, and the source it names. Any of them settles an episode for admission.
STATUS_FILES = {
    "fork.json": "fork",
    "replay_fork.json": "replay_fork",
    "helper.json": "helper",
}


@dataclass
class StagingView:
    status: str
    source: str
    files: dict[str, bytes] = field(default_factory=dict)
    refused: list[dict] = field(default_factory=list)
    #: the fork's block counts from ``fork-blocks.json`` (blocks, parsed, refused, dropped), when it wrote one
    blocks: dict | None = None
    #: whether the reader finished (:func:`finished`); when not, :attr:`files` is empty
    finished: bool = True
    #: the files an unfinished reader left, kept on the host and not handed on
    withheld: list[str] = field(default_factory=list)


def finished(status: str) -> bool:
    """Whether a reader's recorded *status* says it finished: ``ok``, or ``ok: <note>``."""
    return status == "ok" or status.startswith("ok:")


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
    view = StagingView(status=head[0], source=head[1], finished=finished(head[0]))
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
            if why is None and not view.finished:
                why = WITHHELD_WHY
                view.withheld.append(rel[:200])
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
        view.refused.append(
            {"path": BLOCKS_JSON, "why": "unreadable block record", "stage": "parse"},
        )
        return
    try:
        doc = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        view.refused.append(
            {"path": BLOCKS_JSON, "why": "unreadable block record", "stage": "parse"},
        )
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
    """Copy *view*'s files under *dest* (the writer's ``/inputs/staging/<episode>``); the batch-map entry. Its
    ``summary`` is the path of the episode's ``summary.md`` (None without one); the status of a reader that did not
    finish reads ``no reader output: <status>``. The entry never names the reader (the view's ``source``): that stays
    in the host-side records (the pool entry's provenance and the pass's end event)."""
    if view is None:
        return {"status": "none"}
    for rel, data in view.files.items():
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    path = f"/inputs/staging/{dest.name}"
    entry = {
        "status": view.status if view.finished else f"{NO_OUTPUT}: {view.status}",
        "files": sorted(view.files),
        "refused": view.refused,
        "path": path,
        "summary": f"{path}/{SUMMARY_FILE}" if SUMMARY_FILE in view.files else None,
    }
    if view.blocks is not None:
        entry["blocks"] = view.blocks
    return entry
