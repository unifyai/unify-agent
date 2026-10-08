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
  under it) is never snapshotted, read or entered: by default the sandbox policy's ``readable_violation``,
  failing closed (a policy that cannot be built hides everything);
- only the known fields of path records are kept, type-checked (process starts belong to the shell adapter,
  not wired in this build), and every kept byte counts toward ``MAX_AUDIT_BYTES`` per request; past it,
  records are dropped, counted and noted;
- :meth:`finish` runs within ``FINISH_SECONDS``: the records get the first half, the after snapshot, the
  write rows and the diff the rest; each git call gets only the time left, and what is cut off is counted
  (``cut_off``).

There is one active capture per process: ``begin`` replaces it. That relies on the CLI's one request per
process (``request.lock``). :meth:`abort` ends a capture without rows.

Records are processed at request end, so a read of a file that is not in the before snapshot, or that an
earlier cell wrote, sees the file as it is at request end, not as it was right after the reading cell.
Nothing here raises into the request: a capture that cannot start records nothing, and :meth:`finish`
never raises.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
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

#: Bytes of audit records kept per request, every kept field counted (the hook bounds a cell at 1 MiB).
MAX_AUDIT_BYTES = 16 * 1024 * 1024
#: Wall-clock seconds of :meth:`WorktreeCapture.finish`, git included; the records get the first half.
FINISH_SECONDS = 30.0
#: The record fields kept: the audit adapter's schema for path events.
MODE_CHARS = 16
_FLAGS = ("clipped", "cwd_assumed", "cell_thread")

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
    """The index of the first cell whose ``start <= ts <= end`` (harness clock), else -1; never raises."""
    try:
        for i, cell in enumerate(cells):
            try:
                if cell.start <= ts <= cell.end:
                    return i
            except Exception:  # noqa: BLE001 - a cell with no usable window
                continue
    except Exception:  # noqa: BLE001 - cells that cannot be iterated
        pass
    return -1


def _project(record: Any) -> tuple[dict, int] | None:
    """The known, type-checked fields of a path record and their size, or None if it is not one."""
    if not isinstance(record, dict):
        return None
    event = record.get("event")
    if not isinstance(event, str) or event not in PATH_EVENTS:
        return None
    path, dst = record.get("path"), record.get("dst")
    path = path if isinstance(path, str) else None
    dst = dst if isinstance(dst, str) else None
    if path is None and dst is None:
        return None
    out: dict = {"event": event, "path": path}
    if dst is not None:
        out["dst"] = dst
    mode = record.get("mode")
    if isinstance(mode, str):
        out["mode"] = mode[:MODE_CHARS]
    for flag in _FLAGS:
        if record.get(flag) is True:
            out[flag] = True
    return out, RECORD_OVERHEAD + len(json.dumps(out))


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
    """What the capture never records, as a predicate over work-tree-relative paths: what the sandbox policy
    hides from cells, and any path one of whose components the sandbox's own secret rule masks in a mounted root
    (``.env*`` files, credential directories and files, private ``.pem`` keys, ``*key*.json``). The workspace is the
    user's own data and the policy may show such a file to cells; the snapshot still never keeps it (fail closed).
    A failure to evaluate either rule hides the path."""
    from unify import sandbox

    policy = sandbox.build_policy()

    def hidden(rel: str) -> bool:
        try:
            if any(sandbox._secret_rule(part) for part in Path(rel).parts):
                return True
            return policy.readable_violation(workspace / rel) is not None
        except Exception:  # noqa: BLE001 - fail closed
            return True

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
        self.cut_off = (
            0  # records and changed paths left when finish()'s deadline lapsed
        )
        self._over_cap = False

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
        kept: list = []
        for record in records:
            got = _project(record)
            if got is None:
                continue
            if self._kept_bytes + got[1] > MAX_AUDIT_BYTES:
                self.dropped += 1
                if not self._over_cap:
                    self._over_cap = True
                    logger.warning(
                        "memory v2: audit records over the request cap (%d bytes) are dropped",
                        MAX_AUDIT_BYTES,
                    )
                self._recorder.skip_counts["audit records over the request cap"] += 1
                continue
            self._kept_bytes += got[1]
            kept.append(got[0])
        if kept:
            self._kept_bytes += RECORD_OVERHEAD
            self._events.append((stamp, kept))

    def _end(self) -> tuple["WorkTreeRecorder | None", list]:
        global _ACTIVE
        with _LOCK:
            if _ACTIVE is self:
                _ACTIVE = None
        recorder, self._recorder = self._recorder, None
        events, self._events = self._events, []
        return recorder, events

    def abort(self) -> None:
        """End the capture with no rows and no after snapshot (the request was aborted); never raises."""
        try:
            recorder, _ = self._end()
            if recorder is not None:
                recorder.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory v2: work-tree capture abort (%s)",
                type(exc).__name__,
            )

    def finish(self, cells: Sequence[Any]) -> WorktreeResult:
        """The request's rows, snapshot shas and diff; *cells* are the transcript's timed cells, in order
        (``start``/``end`` on the harness clock). Within ``FINISH_SECONDS``; never raises.
        """
        recorder, events = self._end()
        if recorder is None:
            return WorktreeResult([], None, None, "")
        try:
            return self._finish(recorder, events, cells)
        except Exception as exc:  # noqa: BLE001 - never into the request
            logger.warning(
                "memory v2: work-tree capture failed (%s)",
                type(exc).__name__,
            )
            return WorktreeResult([], recorder.before, None, "")
        finally:
            recorder.close()

    def _finish(
        self,
        recorder: WorkTreeRecorder,
        events: list,
        cells: Sequence[Any],
    ) -> WorktreeResult:
        start = time.monotonic()
        deadline = start + FINISH_SECONDS
        try:
            # before anything is read, shaped, stored or cut: the request's full redactor
            recorder.redactor = self._redactor_factory()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "memory v2: no work-tree rows (redactor failed: %s)",
                type(exc).__name__,
            )
            return WorktreeResult([], recorder.before, None, "")
        records_until = start + FINISH_SECONDS / 2
        for stamp, records in events:
            try:
                recorder.record_cell(
                    cell_at(stamp, cells),
                    records,
                    deadline=records_until,
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - one cell's records never lose the rest
                logger.warning(
                    "memory v2: a cell's work-tree records failed (%s)",
                    type(exc).__name__,
                )
        try:
            recorder.finish(deadline=deadline)
        except Exception as exc:  # noqa: BLE001 - keep what was recorded
            logger.warning(
                "memory v2: no work-tree after snapshot (%s)",
                type(exc).__name__,
            )
        diff = ""
        if recorder.after is not None:
            try:
                diff = recorder.diff(seconds=max(deadline - time.monotonic(), 0.1))
            except Exception as exc:  # noqa: BLE001 - worktree_diff never raises
                logger.warning("memory v2: no work-tree diff (%s)", type(exc).__name__)
        self.cut_off = sum(
            n
            for reason, n in recorder.skip_counts.items()
            if reason.startswith("deadline:")
        )
        return WorktreeResult(
            list(recorder.actions),
            recorder.before,
            recorder.after,
            diff,
        )
