"""Work-tree capture for one memory-v2 request (spec §C2 work-tree kind, §E5; integration Task 30).

:meth:`WorktreeCapture.begin` snapshots the request's workspace into ``<UNIFY_HOME>/worktree.git`` (a
separate bare repo, harness-side, never mounted into a cell) and makes the capture the active one, so
``hooks.worker_audit()`` sends the workspace root to the worker child, whose audit hook (``adapters/audit.py``)
then records what each cell opened, listed, removed or renamed there. ``hooks.worker_cell_done()`` hands
each cell's drained records here, stamped on the harness clock. :meth:`WorktreeCapture.finish` gives each
cell's records to the approved work-tree adapter (``adapters/worktree.py``) under the cell whose transcript
window holds the stamp, snapshots the workspace again, and returns the adapter's rows, both snapshot shas
and the redacted work-tree diff.

The adapter does all host-side reading (no-follow fds from the root, size caps, shapes in a bounded child,
text-only exported blobs, R26-R30). This module adds three things:

- the redactor is built by *redactor_factory* at :meth:`finish`, so it knows every secret the request
  learned, and it is set on the recorder before any record is read, shaped, stored or cut;
- what the sandbox hides from cells inside the workspace (secret-named files, the store, a state directory
  under it) is never snapshotted, read or listed: by default the sandbox policy's ``readable_violation``,
  failing closed (a policy that cannot be built hides everything);
- only path records are kept (process starts belong to the shell adapter, not wired in this build), with a
  per-request byte bound.

Records are processed at request end, so a read of a file that is not in the before snapshot, or that an
earlier cell wrote, sees the file as it is at request end, not as it was right after the reading cell.
Nothing here raises into the request: a capture that cannot start records nothing, and :meth:`finish`
never raises.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from ..blobs import BlobStore
from ..episodes import Action
from ..gitio import Repo, _git
from ..redact import Redactor
from .adapters.audit import PATH_EVENTS, RECORD_OVERHEAD
from .adapters.worktree import WorkTreeRecorder
from .paths import Paths

logger = logging.getLogger(__name__)

#: Bytes of recorded path strings kept per request (the audit hook bounds each cell at 1 MiB).
MAX_AUDIT_BYTES = 16 * 1024 * 1024

_ACTIVE: "WorktreeCapture | None" = None
_LOCK = threading.Lock()


def active() -> "WorktreeCapture | None":
    """The capture of the request in progress, between a successful ``begin`` and ``finish``."""
    return _ACTIVE


@dataclass
class WorktreeResult:
    actions: list[Action]  # kind="worktree" rows, channel "worktree:workspace"
    before: str | None  # snapshot commit shas in worktree.git (harness-side only)
    after: str | None
    diff: (
        str  # WorkTreeRecorder.diff(): redacted text hunks, hashes and sizes otherwise
    )


def cell_at(ts: float, cells: Sequence[Any]) -> int:
    """The index of the first cell whose ``start <= ts <= end`` (harness clock), else -1."""
    for i, cell in enumerate(cells):
        try:
            if cell.start <= ts <= cell.end:
                return i
        except TypeError:  # a cell with no recorded window
            continue
    return -1


def snapshot_repo(git_dir: Path) -> Repo:
    """The bare sha1 snapshot repo at *git_dir*, created on first use (no work tree is ever attached)."""
    git_dir = Path(git_dir)
    if not (git_dir / "HEAD").exists():
        git_dir.mkdir(parents=True, exist_ok=True)
        _git(
            [
                "init",
                "--bare",
                "-q",
                "--object-format=sha1",
                "-b",
                "main",
                str(git_dir),
            ],
        )
    return Repo(git_dir)


def sandbox_hidden(workspace: Path) -> Callable[[str], bool]:
    """What the workspace sandbox hides from cells, as a predicate over work-tree-relative paths."""
    from unify import sandbox

    policy = sandbox.build_policy()

    def hidden(rel: str) -> bool:
        return policy.readable_violation(workspace / rel) is not None

    return hidden


class WorktreeCapture:
    """One request's work-tree capture (see the module docstring)."""

    def __init__(
        self,
        paths: Paths,
        workspace: Path,
        redactor_factory: Callable[[], Redactor],
        *,
        hidden: Callable[[str], bool] | None = None,
    ) -> None:
        self.paths = paths
        self.workspace = Path(os.path.realpath(workspace))
        self._redactor_factory = redactor_factory
        self._hidden = hidden
        self._recorder: WorkTreeRecorder | None = None
        self._events: list[tuple[float, list]] = []
        self._kept_bytes = 0
        self.dropped = 0  # audit records lost to the hook's caps or to MAX_AUDIT_BYTES
        self.failed = 0  # audit events the hook failed on

    def begin(self) -> None:
        """Snapshot the workspace and become the active capture. Installs nothing in the worker (the
        worker asks ``hooks.worker_audit()`` when it starts); never raises."""
        global _ACTIVE
        try:
            git_dir = Path(os.path.realpath(self.paths.worktree_git))
            if git_dir == self.workspace or self.workspace in git_dir.parents:
                raise ValueError("the snapshot repo must live outside the workspace")
            hidden = self._hidden
            if hidden is None:
                hidden = sandbox_hidden(self.workspace)
            recorder = WorkTreeRecorder(
                self.workspace,
                snapshot_repo(self.paths.worktree_git),
                BlobStore(self.paths.blobs),
                hidden=hidden,
            )
            recorder.begin()
        except Exception as exc:  # noqa: BLE001 - a capture never fails the request
            logger.warning(
                "memory v2: no work-tree capture for this request (%s)",
                type(exc).__name__,
            )
            return
        self._recorder = recorder
        with _LOCK:
            _ACTIVE = self

    def cell_done(self, stamp: float, drained: Any) -> None:
        """Keep one cell's drained audit records (path events only), stamped *stamp* on the harness clock."""
        if self._recorder is None or not isinstance(drained, dict):
            return
        records = drained.get("records")
        if not isinstance(records, list):
            return
        for key in ("dropped", "failed"):
            n = drained.get(key)
            if isinstance(n, int) and not isinstance(n, bool) and n > 0:
                setattr(self, key, getattr(self, key) + n)
        kept = [
            r for r in records if isinstance(r, dict) and r.get("event") in PATH_EVENTS
        ]
        size = sum(
            RECORD_OVERHEAD
            + sum(len(v) for v in (r.get("path"), r.get("dst")) if isinstance(v, str))
            for r in kept
        )
        if self._kept_bytes + size > MAX_AUDIT_BYTES:
            self.dropped += len(kept)
            return
        self._kept_bytes += size
        self._events.append((stamp, kept))

    def finish(self, cells: Sequence[Any]) -> WorktreeResult:
        """The request's rows, snapshot shas and diff; *cells* are the transcript's timed cells, in order
        (``start``/``end`` on the harness clock). Never raises."""
        global _ACTIVE
        with _LOCK:
            if _ACTIVE is self:
                _ACTIVE = None
        recorder, self._recorder = self._recorder, None
        events, self._events = self._events, []
        if recorder is None:
            return WorktreeResult([], None, None, "")
        try:
            # before anything is read, shaped, stored or cut: the request's full redactor
            recorder.redactor = self._redactor_factory()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory v2: no work-tree rows (redactor failed: %s)",
                type(exc).__name__,
            )
            return WorktreeResult([], recorder.before, None, "")
        try:
            for stamp, records in events:
                recorder.record_cell(cell_at(stamp, cells), records)
            recorder.finish()
        except Exception as exc:  # noqa: BLE001 - keep what was recorded
            logger.warning(
                "memory v2: work-tree capture incomplete (%s)",
                type(exc).__name__,
            )
        diff = ""
        if recorder.after is not None:
            try:
                diff = recorder.diff()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - worktree_diff never raises; diff() guards
                logger.warning("memory v2: no work-tree diff (%s)", type(exc).__name__)
        return WorktreeResult(
            list(recorder.actions),
            recorder.before,
            recorder.after,
            diff,
        )
