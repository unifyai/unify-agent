"""Work-tree and shell observation for memory v2 (spec §C2, option A of §C5).

This module runs *inside the worker child* and imports the standard library
only, so the worker can load it by file path without the ``unify`` package.
One ``sys.addaudithook`` hook records, while a cell runs, what the cell
opened, listed, removed, renamed or created under the work-tree roots, and
which processes it started. The harness drains the buffer after each cell.

It produces RAW RECORDS only. Action rows are built elsewhere from these
records: the shell adapter owns ``kind="shell"`` rows and the work-tree
adapter owns ``kind="worktree"`` rows.

Record schema
-------------
``CellAudit.drain()`` returns a JSON-safe dict::

    {"records": [record, ...],  # in event order, at most MAX_RECORDS
     "dropped": int,            # events lost to the count or byte cap
     "failed": int,             # events the hook failed on (swallowed)
     "bytes": int}              # the byte budget used (see MAX_BYTES)

Every record has:

* ``event`` (str): the audit event name, one of the names below.
* ``tid`` (int): ``threading.get_ident()`` of the thread that raised it.
* ``cell_thread`` (bool): whether ``tid`` is the thread that called
  ``begin()`` (the cell's thread).

Path events. ``path`` and ``dst`` are absolute, normalised
(``os.path.abspath``, symlinks not resolved) paths in the child's path
space, always under a root; events wholly outside every root are never
recorded.

* ``"open"``: ``path``, ``mode`` (str). ``mode`` is the builtin ``open``
  mode string (``"r"``, ``"wb"``, ``"a+"`` ...); for a low-level ``os.open``
  it is ``"r"`` or ``"w"``, derived from the flags. Directory opens
  (``O_DIRECTORY``) are not recorded, nor are low-level ``os.open`` calls
  made from standard-library modules (``shutil.rmtree``, ``os.fwalk``,
  ``tempfile``: directory handles and dir_fd-relative names; their effects
  arrive as their own events). A relative ``os.open`` from cell code is
  resolved against the cwd: its ``dir_fd`` is not in the audit event, so
  ``os.open(name, dir_fd=fd)`` with a relative name is unresolvable; such
  records carry ``"cwd_assumed": true``.
* ``"os.listdir"``, ``"os.scandir"``: ``path`` (the listed directory; the
  cwd when called with no argument; resolved from a directory fd).
* ``"os.remove"``, ``"os.rmdir"``, ``"os.mkdir"``: ``path``, resolved against
  ``dir_fd`` when one is given (Linux ``/proc/self/fd``; dropped when that
  cannot be read).
* ``"shutil.rmtree"``: ``path`` (the tree removed). The removals it makes
  are recorded as well, each at its real path.
* ``"os.rename"``: ``path`` (source) and ``dst``; either is ``None`` when
  that side is outside every root (at least one side is inside).

Process events. Executables and argv are kept wherever the executable is.

* ``"subprocess.Popen"``, ``"os.posix_spawn"``, ``"os.exec"``: ``exe`` (str
  or None), ``argv`` (list of str-or-None, or None). A start seen again
  through a lower layer (Popen then posix_spawn with the same argv) is
  recorded once. ``argv_truncated`` (true, present only when set): there
  were more than MAX_ARGV items.
* ``"os.system"``: ``exe`` None, ``argv`` ``[command]``.

String handling: every recorded string has key-shaped substrings masked
*before* it is clipped to MAX_STR characters; a clipped string loses its
trailing token run (so no partial secret survives the cut) and ends with
``CLIP_MARK``, and its record has ``"clipped": true``. Consumers still run
the harness's exact-value ``Redactor`` before anything is persisted.

What the hook guarantees, and what it cannot
--------------------------------------------
* It never records environment variables (the ``env`` argument of the spawn
  events is never read) or file contents (only paths and modes).
* It is bounded per cell: MAX_RECORDS records and MAX_BYTES of recorded
  strings (plus RECORD_OVERHEAD per record); the excess is counted in
  ``dropped``.
* It swallows its own failures (counted in ``failed``), ignores its own
  activity (a thread-local re-entrancy guard, which also hides any user
  code run inside the hook, such as an ``__fspath__`` that opens a file),
  checks ``on`` by identity so a hostile ``__bool__`` is never called, and
  returns at once when the stack is within RECURSION_MARGIN frames of the
  recursion limit. One case cannot be fixed in Python: an audited event
  raised at the recursion limit itself raises ``RecursionError`` on entry
  to the hook frame.
* ``on`` is process-wide. Events from other threads, and from asyncio tasks
  or executor jobs left running by earlier cells, are credited to the cell
  that is running (``tid``/``cell_thread`` let consumers filter threads;
  tasks on the cell's own thread cannot be told apart), and their events
  between cells are lost.

Trust: records are the agent's own unconfirmed self-reports. A cell can
find the hook (``gc.get_objects()``), switch it off, clear or rewrite its
buffer, forge events with ``sys.audit`` and flood it so later events drop.
That only pollutes the agent's own trajectory, the same power it already
has through its reply, file names and tool arguments, so it is not a
privilege escalation. Records are never used for confinement,
authorisation, checker or outcome signals, or cost. Writes should be
cross-checked against snapshot diffs, content comes only from work-tree
snapshots, and harness-side code must never open a recorded path without
refusing symlinks (no-follow) and re-checking that it is inside the root.
"""

from __future__ import annotations

import os
import re
import sys
import threading
from typing import Any, Iterable, Optional

PATH_EVENTS = frozenset(
    {
        "open",
        "os.listdir",
        "os.scandir",
        "os.remove",
        "os.rename",
        "os.mkdir",
        "os.rmdir",
        "shutil.rmtree",
    },
)
SPAWN_EVENTS = frozenset({"subprocess.Popen", "os.posix_spawn", "os.exec"})
SYSTEM_EVENT = "os.system"
#: Audit events kept; everything else returns after one set lookup.
AUDITED = PATH_EVENTS | SPAWN_EVENTS | {SYSTEM_EVENT}

#: Caps per cell and per record.
MAX_RECORDS = 2000
MAX_BYTES = 1 << 20
MAX_ARGV = 64
MAX_STR = 4096
#: How much of a string is scanned for keys before clipping.
MAX_SCAN = MAX_STR * 16
RECORD_OVERHEAD = 64
RECURSION_MARGIN = 50
CLIP_MARK = "<clipped>"

#: The process-global sentinel: the one installed hook, whatever module
#: object (a second load of this file) asks for it.
SENTINEL = "_unify_memory_v2_cell_audit"

# Same shapes as unify.memory_v2.redact.KEY_SHAPED; copied so this module
# stays stdlib-only.
_KEY_SHAPED = re.compile(
    r"sk-or-v1-[0-9a-f]{64}"
    r"|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}"
    r"|AKIA[0-9A-Z]{16}",
)
_KEY_MASK = "<redacted:key-shaped>"
# The characters of a token that a clip may have cut through: a clipped text drops its trailing run of them.
_TOKEN_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-+/=.:~")


def _drop_trailing_token(text: str) -> str:
    """*text* without its trailing run of token characters: a linear scan from the end (a ``[...]+$`` search
    retries from every start position and is quadratic on a long run)."""
    # ``$`` also matched before one final newline, which the old search kept: same here
    tail = "\n" if text.endswith("\n") else ""
    end = len(text) - len(tail)
    while end and text[end - 1] in _TOKEN_CHARS:
        end -= 1
    return text[:end] + tail

_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) - {"__main__"}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
#: dir_fd's position in each path event's audit args.
_DIR_FD_AT = {"os.remove": 1, "os.rmdir": 1, "os.mkdir": 2, "shutil.rmtree": 1}


def _is_fd(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _decode(value: Any) -> Optional[str]:
    """A str/bytes/path-like argument as text; None for anything else."""
    if value is None or isinstance(value, (int, bool)):
        return None
    try:
        return os.fsdecode(value)
    except Exception:  # noqa: BLE001
        return None


def _clean(text: str) -> tuple[str, bool]:
    """Mask key-shaped substrings, then clip; returns (text, clipped)."""
    masked = _KEY_SHAPED.sub(_KEY_MASK, text[:MAX_SCAN])
    if len(text) <= MAX_SCAN and len(masked) <= MAX_STR:
        return masked, False
    cut = _drop_trailing_token(masked[:MAX_STR])
    return cut + CLIP_MARK, True


def _fd_path(fd: Any) -> Optional[str]:
    """The path a directory fd refers to, best effort (Linux /proc)."""
    if not _is_fd(fd):
        return None
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except Exception:  # noqa: BLE001
        return None
    return target if target.startswith("/") else None


class CellAudit:
    """The per-process hook and its bounded per-cell buffer.

    ``roots`` are the work-tree roots *as the child sees them*; path events
    outside every root are dropped here, so the buffer holds only work-tree
    paths and process starts.
    """

    def __init__(self, roots: Iterable[str]) -> None:
        self.roots = self.prefixes(roots)
        self.records: list = []
        self.dropped = 0  # over the count or byte cap
        self.failed = 0  # swallowed hook failures
        self.bytes = 0
        self.on = False
        self.cell_tid: Optional[int] = None
        self._local = threading.local()

    @staticmethod
    def prefixes(roots: Iterable[str]) -> tuple:
        out: list = []
        for root in roots or ():
            if not isinstance(root, str) or not root:
                continue
            for variant in (os.path.abspath(root), os.path.realpath(root)):
                prefix = os.path.join(variant, "")
                if prefix not in out:
                    out.append(prefix)
        return tuple(out)

    # The hook. The fast path comes first: most events are not ours.
    def __call__(self, event: str, args: tuple) -> None:
        if event not in AUDITED:
            return
        try:
            if self.on is not True:
                return
            if len(self.records) >= MAX_RECORDS:
                self.dropped += 1
                return
            local = self._local
            if getattr(local, "busy", False):
                return
            limit = sys.getrecursionlimit()
            if limit > RECURSION_MARGIN:
                try:
                    sys._getframe(limit - RECURSION_MARGIN)
                except ValueError:
                    pass  # the stack is not that deep: carry on
                else:
                    return  # too near the limit to do anything safely
            if event == "open" and not isinstance(
                args[1] if len(args) > 1 else "",
                str,
            ):
                # A low-level os.open made by the standard library itself
                # (shutil.rmtree, os.fwalk, tempfile): directory handles and
                # dir_fd-relative names the event cannot resolve. Their
                # effects arrive as their own events (rmtree, remove, scandir).
                caller = sys._getframe(1).f_globals.get("__name__") or ""
                if caller.partition(".")[0] in _STDLIB:
                    return
            local.busy = True
            try:
                self._record(event, args)
            finally:
                local.busy = False
        except Exception:  # noqa: BLE001 - an audit never changes the cell
            try:
                self.failed += 1
            except Exception:  # noqa: BLE001
                pass

    def _inside(self, path: Optional[str]) -> bool:
        if path is None:
            return False
        probe = os.path.join(path, "")
        return any(probe.startswith(r) for r in self.roots)

    def _resolve(self, value: Any, dir_fd: Any = None) -> Optional[str]:
        """An absolute path for a path argument; a relative one is joined to
        its ``dir_fd``'s directory when one is given, else to the cwd.
        None when unresolvable."""
        text = _decode(value)
        if text is None:
            return None
        if not os.path.isabs(text) and _is_fd(dir_fd):
            base = _fd_path(dir_fd)
            if base is None:
                return None
            text = os.path.join(base, text)
        return os.path.abspath(text)

    def _append(self, record: dict, size: int) -> None:
        if len(self.records) >= MAX_RECORDS or self.bytes + size > MAX_BYTES:
            self.dropped += 1
            return
        self.bytes += size
        tid = threading.get_ident()
        record["tid"] = tid
        record["cell_thread"] = tid == self.cell_tid
        self.records.append(record)

    @staticmethod
    def _string(text: str, record: dict) -> str:
        cleaned, clipped = _clean(text)
        if clipped:
            record["clipped"] = True
        return cleaned

    def _record(self, event: str, args: tuple) -> None:
        if event in SPAWN_EVENTS:
            self._spawn(event, args)
        elif event == SYSTEM_EVENT:
            command = _decode(args[0]) if args else None
            if command is not None:
                record: dict = {"event": event, "exe": None}
                record["argv"] = [self._string(command, record)]
                self._append(record, RECORD_OVERHEAD + len(record["argv"][0]))
        else:
            self._path(event, args)

    def _spawn(self, event: str, args: tuple) -> None:
        # (executable, argv, ...): the env argument is never touched.
        record: dict = {"event": event}
        exe = _decode(args[0]) if args else None
        raw = args[1] if len(args) > 1 else None
        argv: Optional[list] = None
        size = RECORD_OVERHEAD
        if isinstance(raw, (str, bytes)) or hasattr(raw, "__fspath__"):
            raw = [raw]
        if raw is not None:
            argv = []
            for i, item in enumerate(raw):
                if i >= MAX_ARGV:
                    record["argv_truncated"] = True
                    break
                text = _decode(item)
                if text is None:
                    argv.append(None)
                    continue
                text = self._string(text, record)
                size += len(text)
                argv.append(text)
        last = self.records[-1] if self.records else None
        if (
            last is not None
            and last["event"] in SPAWN_EVENTS
            and last["event"] != event
            and last.get("argv") == argv
        ):
            return  # the same start seen through a lower layer
        record["exe"] = self._string(exe, record) if exe else None
        record["argv"] = argv
        size += len(record["exe"] or "")
        self._append(record, size)

    def _path(self, event: str, args: tuple) -> None:
        if event == "os.rename":
            # (src, dst, src_dir_fd, dst_dir_fd)
            src = self._resolve(args[0], args[2] if len(args) > 2 else None)
            dst = self._resolve(
                args[1] if len(args) > 1 else None,
                args[3] if len(args) > 3 else None,
            )
            src_in, dst_in = self._inside(src), self._inside(dst)
            if not (src_in or dst_in):
                return
            record: dict = {"event": event}
            record["path"] = self._string(src, record) if src_in else None
            record["dst"] = self._string(dst, record) if dst_in else None
            size = RECORD_OVERHEAD + len(record["path"] or "")
            self._append(record, size + len(record["dst"] or ""))
            return
        first = args[0] if args else None
        extra: dict = {}
        if event in ("os.listdir", "os.scandir"):
            if first is None:
                path = os.path.abspath(".")
            elif _is_fd(first):
                path = _fd_path(first)
            else:
                path = self._resolve(first)
        elif event == "open":
            if isinstance(first, int):
                return  # an fd, not a path
            mode = args[1] if len(args) > 1 else None
            if isinstance(mode, str):
                extra["mode"] = mode[:16]
            else:
                flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
                if flags & _O_DIRECTORY:
                    return  # a directory handle, not a file access
                extra["mode"] = "w" if flags & _WRITE_FLAGS else "r"
                text = _decode(first)
                if text is not None and not os.path.isabs(text):
                    extra["cwd_assumed"] = True
            path = self._resolve(first)
        else:
            at = _DIR_FD_AT.get(event)
            dir_fd = args[at] if at is not None and len(args) > at else None
            path = self._resolve(first, dir_fd)
        if not self._inside(path):
            return
        record = {"event": event}
        record["path"] = self._string(path, record)
        record.update(extra)
        self._append(record, RECORD_OVERHEAD + len(record["path"]))

    # Controls, called from the worker's cell loop on the cell's thread.
    def begin(self) -> None:
        self.records, self.dropped, self.failed, self.bytes = [], 0, 0, 0
        self.cell_tid = threading.get_ident()
        self.on = True

    def end(self) -> None:
        self.on = False

    def drain(self) -> dict:
        """The cell's records, emptied; safe to send as JSON."""
        self.on = False
        out = {
            "records": self.records,
            "dropped": self.dropped,
            "failed": self.failed,
            "bytes": self.bytes,
        }
        self.records, self.dropped, self.failed, self.bytes = [], 0, 0, 0
        return out


_INSTALL_LOCK = threading.Lock()


def install(roots: Iterable[str]) -> Any:
    """Install the hook once per *process*, even if this file is loaded
    twice: the hook is kept in ``sys.modules[SENTINEL]``. A second call
    returns that hook with its roots replaced (a hook can never be removed
    or doubled)."""
    with _INSTALL_LOCK:
        existing = sys.modules.get(SENTINEL)
        if existing is not None:
            existing.roots = existing.prefixes(roots)
            return existing
        audit = CellAudit(roots)
        sys.modules[SENTINEL] = audit  # type: ignore[assignment]
        sys.addaudithook(audit)
        return audit
