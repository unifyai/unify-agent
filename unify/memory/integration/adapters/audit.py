"""Work-tree and shell observation for memory v2, by an audit hook in the worker child.

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

Memory calls. The generated library package (``library_helper``) raises
``sys.audit("memory.call", item_id, args_digest, outcome)`` once per call of
a library function, with ``outcome`` one of ``ok``, ``refused`` or
``error``. They are kept apart from the path and process records, so a
flood of either never hides the other, and ``drain()`` adds, only when a
cell raised any::

    "calls": {"counts": {item_id: {"ok": n, "refused": n, "error": n}},
              "records": [{"item", "args_digest", "outcome", "tid",
                           "cell_thread"}, ...],   # at most MAX_CALL_RECORDS
              "dropped": int,          # calls of items over MAX_CALL_ITEMS
              "records_dropped": int,  # calls counted but not listed
              "malformed": int,        # events whose arguments were not
                                       # an item id, a hex digest and an outcome
              "helpers": {helper_id: {"ok": n, "refused": n, "error": n}}}
                                       # only when a helper call was seen

The package's own calls raise ``memory.helper_call`` with the same
arguments, under the ids ``memory:get`` and ``memory:items`` (lookups),
``memory:ask`` (a question to the librarian), and, for the calls they replace
and their aliases, ``memory:index``, ``memory:show``, ``memory:find``,
``memory:match`` (also an item's applicability check, ``Item.fits``) and
``memory:run`` (also a function item's tracked call); they are counted under
``helpers`` only, never as items.

``memory.match`` and ``Item.fits`` also raise ``memory.match_result`` with
two tuples of ``(id, level, status, matched)`` rows (the procedures listed,
the failed attempts listed apart), and ``memory.run`` and a function item's
call raise ``memory.run_result``
with its target id, its outcome (``returned``, ``refused`` or ``error``),
whether two runs agreed (``yes``, ``no`` or ``unchecked``), its level, its
conditions (``none``, ``held``, ``not-held`` or ``error``) and a tuple of
the ids of the matching procedures that gave the same result. Ids are
library item ids or ``capture:<16 hex digits>`` (a captured procedure's run
also raises ``memory.call`` under its capture id). ``calls`` then also
holds, in order::

    "matches": [{"listed": [{"id", "level", "status", "matched"}, ...],
                 "failed": [...], "calls_before": n, "runs_before": n,
                 "no_outcome": [{"id", "read", "applies", "reason"}, ...]}, ...]
    "runs": [{"target", "outcome", "repeatable", "level", "conditions",
              "agrees_with": [id, ...], "calls_before": n}, ...]

where ``calls_before`` and ``runs_before`` count the cell's ``records`` and
``runs`` kept before it, so a consumer can tell which calls came after.
``no_outcome`` is there only when ``memory.match`` ran the stored readers
(the librarian's dispatch file) and some gave no outcome: each such reader's
procedures with the label ``not-started`` or ``failed`` and a fixed short
reason (:data:`_ASK_REASON`), the event's optional third element.

An ``open`` record made by the import system, by traceback formatting or by
the library package itself carries ``via`` (``import``, ``traceback`` or
``library``), so a reader can tell the cell's own file reads from them.

A learned skill's fit (``memory.learned(id).fit(examples)``, :mod:`unify.memory.learned`) raises
``sys.audit("memory.learned_fit", item_id, state_text, reproduced, steps)``:
the fitted state as JSON text ("" when it was over the package's bound),
whether it gave back every example's output, and the optimiser steps. It is
kept apart, under ``calls``, only when a cell raised one::

    "learned": [{"item", "state", "reproduced", "steps"}, ...]
                                       # at most MAX_LEARNED_FITS, and
                                       # MAX_LEARNED_BYTES of state text
    "learned_dropped": int

A state holding a key-shaped string (``_KEY_SHAPED``) is refused and counted
under ``malformed``: refused, not masked, so a kept state is always the
cell's own JSON. The state is the one value this hook keeps: it is the fitted parameters of a
library item, read only harness-side by the consolidation after the episode
(:mod:`unify.memory.learned_store`), which gates it on the item's own stored
fixtures before anything is stored; a forged one is refused there.

A call that raised (an item's or a helper's) also raises
``sys.audit("memory.call_failed", id, exception_type, message_line,
args_shape)``: the exception's type name, the first line of its message and,
for a helper, the arguments' types and sizes (never values). The block then
also has::

              "failures": [{"item", "exception", "message", "shape", "tid",
                            "cell_thread"}, ...],  # at most MAX_CALL_FAILURES
              "failures_dropped": int}             # failures not listed

``message`` and ``shape`` are key-masked and clipped to MAX_FAILURE_TEXT
characters, like every string here; a failure whose id is neither an item
id nor one of FAILURE_HELPERS counts as malformed.

``counts`` holds every well-formed call of up to MAX_CALL_ITEMS items, so it
is exact unless ``dropped`` is non-zero. Like every record here, these are
the cell's own unconfirmed reports (see Trust below).

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
Memory call counts feed the library's item statuses, which were already
built from the cells' own code and tracebacks; a forged call is the same
self-report as a call site the cell wrote, never more authority.
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
CALL_EVENT = "memory.call"
CALL_OUTCOMES = ("ok", "refused", "error")
#: The library package's own helpers raise this event, with the same arguments, under these ids.
HELPER_EVENT = "memory.helper_call"
HELPER_ITEMS = frozenset(
    {
        "memory:get",
        "memory:items",
        "memory:ask",
        "memory:index",
        "memory:show",
        "memory:find",
        "memory:match",
        "memory:run",
    },
)
#: A failed call of an item or a helper: (id, exception type, first message line, arguments' shape).
FAILED_EVENT = "memory.call_failed"
#: The helper ids a failure may name.
FAILURE_HELPERS = HELPER_ITEMS
#: What ``memory.match`` or ``Item.fits`` listed and how a tracked call (``memory.run``, or calling a function item)
#: went (ids and outcomes only).
MATCH_EVENT = "memory.match_result"
RUN_EVENT = "memory.run_result"
#: What ``memory.ask`` got back (UNIFY_MEMORY_LIBRARIAN=on): the ids the answer listed as fitting, whether it said
#: nothing fits, and the recognisers' outcomes as ``(id, read, applies)`` labels, with a fixed short reason after
#: them when the reader gave no outcome (``not-started``, ``failed``); never text or values.
ASK_EVENT = "memory.ask_result"
#: A learned skill's fit: (item id, fitted state as JSON text, reproduced, steps).
LEARNED_EVENT = "memory.learned_fit"
ASK_READ = ("parsed", "did-not-parse", "not-started", "failed")
ASK_NO_OUTCOME = ("not-started", "failed")
ASK_APPLIES = ("held", "did-not-hold", "raised", "not-run", "none")
RUN_OUTCOMES = ("returned", "refused", "error")
RUN_REPEATABLE = ("yes", "no", "unchecked")
RUN_CONDITIONS = ("none", "held", "not-held", "error")
#: Audit events kept; everything else returns after one set lookup.
AUDITED = PATH_EVENTS | SPAWN_EVENTS | {
    SYSTEM_EVENT,
    CALL_EVENT,
    HELPER_EVENT,
    FAILED_EVENT,
    MATCH_EVENT,
    RUN_EVENT,
    ASK_EVENT,
    LEARNED_EVENT,
}
#: The callers whose opens are marked ``via``: the import system, traceback formatting, the library package.
_VIA_IMPORT = ("importlib", "_frozen_importlib", "zipimport")
_VIA_TRACEBACK = frozenset({"linecache", "tokenize", "traceback"})

#: Caps per cell and per record.
MAX_RECORDS = 2000
MAX_BYTES = 1 << 20
MAX_ARGV = 64
MAX_STR = 4096
#: How much of a string is scanned for keys before clipping.
MAX_SCAN = MAX_STR * 16
RECORD_OVERHEAD = 64
RECURSION_MARGIN = 50
#: Memory calls per cell: distinct items counted, and calls listed one by one.
MAX_CALL_ITEMS = 512
MAX_CALL_RECORDS = 1000
MAX_ITEM_ID = 300
#: Failed calls listed per cell, and the characters kept of a failure's message line and arguments' shape.
MAX_CALL_FAILURES = 50
MAX_FAILURE_TEXT = 300
#: match results and runs listed per cell, and ids per match result.
MAX_MATCH_RECORDS = 200
MAX_RUN_RECORDS = 1000
MAX_MATCH_IDS = 500
#: learned-skill fits kept per cell, and the state text they may hold in all.
MAX_LEARNED_FITS = 20
MAX_LEARNED_BYTES = 1 << 20
CLIP_MARK = "<clipped>"

#: The process-global sentinel: the one installed hook, whatever module
#: object (a second load of this file) asks for it.
SENTINEL = "_unify_memory_cell_audit"

# Same shapes as unify.memory.redact.KEY_SHAPED; copied so this module
# stays stdlib-only.
_KEY_SHAPED = re.compile(
    r"sk-or-v1-[0-9a-f]{64}"
    r"|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}"
    r"|AKIA[0-9A-Z]{16}",
)
_KEY_MASK = "<redacted:key-shaped>"
_ITEM_ID = re.compile(r"memory(?:\.[A-Za-z_][A-Za-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_]*")
_ITEM_ID_OR_MODULE = re.compile(r"memory(?:\.[A-Za-z_][A-Za-z0-9_]*)+(?::[A-Za-z_][A-Za-z0-9_]*)?")
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}")
_HEX_DIGEST = re.compile(r"[0-9a-f]{1,64}")
_CAPTURE_ID = re.compile(r"capture:[0-9a-f]{16}")
#: A level, a status or how a match was made: a short lower-case label.
_LABEL = re.compile(r"[a-z][a-z0-9-]{0,39}")
#: Why a stored reader gave no outcome: one of the package's fixed forms (``library_helper._REASON``), at most 80
#: characters, never a value.
_ASK_REASON = re.compile(
    r"timeout after \d{1,4}(?:\.\d{1,3})? s(?: for all readers)?|output overrun|no marker|no report|malformed report"
    r"|(?:start|runner|load): [A-Za-z_][A-Za-z0-9_]{0,60}",
)
# The characters of a token that a clip may have cut through: a clipped text drops its trailing run of them.
_TOKEN_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-+/=.:~",
)


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


def _line(text: str) -> str:
    """The first line of *text*, key-masked, then clipped to MAX_FAILURE_TEXT."""
    masked = _KEY_SHAPED.sub(_KEY_MASK, text[:MAX_SCAN])
    lines = masked.splitlines()
    line = lines[0] if lines else ""
    if len(line) <= MAX_FAILURE_TEXT:
        return line
    return _drop_trailing_token(line[:MAX_FAILURE_TEXT]) + CLIP_MARK


def _fd_path(fd: Any) -> Optional[str]:
    """The path a directory fd refers to, best effort (Linux /proc)."""
    if not _is_fd(fd):
        return None
    try:
        target = os.readlink(f"/proc/self/fd/{fd}")
    except Exception:  # noqa: BLE001
        return None
    return target if target.startswith("/") else None


def _via(caller: str) -> Optional[str]:
    """``import``, ``traceback`` or ``library`` for an open made by the import system, by traceback formatting or
    by the library package (its helper, its matcher or one of its modules); None for anything else."""
    if caller.startswith(_VIA_IMPORT):
        return "import"
    if caller in _VIA_TRACEBACK:
        return "traceback"
    if caller == "memory" or caller.startswith(("memory.", "_memory_")):
        return "library"
    return None


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
        self._reset_calls()

    def _reset_calls(self) -> None:
        self.call_counts: dict = {}
        self.call_records: list = []
        self.calls_dropped = 0
        self.call_records_dropped = 0
        self.calls_malformed = 0
        self.helper_counts: dict = {}
        self.call_failures: list = []
        self.call_failures_dropped = 0
        self.match_records: list = []
        self.run_records: list = []
        self.ask_records: list = []
        self.learned_records: list = []
        self.learned_dropped = 0
        self.learned_bytes = 0

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
            if event == CALL_EVENT:
                self._call(args)
                return
            if event == HELPER_EVENT:
                self._helper_call(args)
                return
            if event == FAILED_EVENT:
                self._failure(args)
                return
            if event == MATCH_EVENT:
                self._match_result(args)
                return
            if event == RUN_EVENT:
                self._run_result(args)
                return
            if event == ASK_EVENT:
                self._ask_result(args)
                return
            if event == LEARNED_EVENT:
                self._learned_fit(args)
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
            via = None
            if event == "open":
                caller = sys._getframe(1).f_globals.get("__name__") or ""
                caller = caller if type(caller) is str else ""
                if not isinstance(args[1] if len(args) > 1 else "", str):
                    # A low-level os.open made by the standard library itself
                    # (shutil.rmtree, os.fwalk, tempfile): directory handles and
                    # dir_fd-relative names the event cannot resolve. Their
                    # effects arrive as their own events (rmtree, remove, scandir).
                    if caller.partition(".")[0] in _STDLIB:
                        return
                via = _via(caller)
            local.busy = True
            try:
                self._record(event, args, via)
            finally:
                local.busy = False
        except Exception:  # noqa: BLE001 - an audit never changes the cell
            try:
                self.failed += 1
            except Exception:  # noqa: BLE001
                pass

    def _call(self, args: tuple) -> None:
        """Count one ``memory.call`` event. Only exact ``str`` arguments of the
        expected shapes are read, so no code of a forged argument ever runs."""
        if type(args) is not tuple or len(args) != 3:
            self.calls_malformed += 1
            return
        item, digest, outcome = args
        if (
            type(item) is not str
            or type(digest) is not str
            or type(outcome) is not str
            or outcome not in CALL_OUTCOMES
            or len(item) > MAX_ITEM_ID
            or (_ITEM_ID.fullmatch(item) is None and _CAPTURE_ID.fullmatch(item) is None)
            or _HEX_DIGEST.fullmatch(digest) is None
        ):
            self.calls_malformed += 1
            return
        counts = self.call_counts.get(item)
        if counts is None:
            if len(self.call_counts) >= MAX_CALL_ITEMS:
                self.calls_dropped += 1
                return
            counts = self.call_counts[item] = dict.fromkeys(CALL_OUTCOMES, 0)
        counts[outcome] += 1
        if len(self.call_records) >= MAX_CALL_RECORDS:
            self.call_records_dropped += 1
            return
        tid = threading.get_ident()
        self.call_records.append(
            {
                "item": item,
                "args_digest": digest,
                "outcome": outcome,
                "tid": tid,
                "cell_thread": tid == self.cell_tid,
            },
        )

    def _helper_call(self, args: tuple) -> None:
        """Count one ``memory.helper_call`` event by helper and outcome; nothing else is kept."""
        if type(args) is not tuple or len(args) != 3:
            self.calls_malformed += 1
            return
        item, digest, outcome = args
        if (
            type(item) is not str
            or type(digest) is not str
            or type(outcome) is not str
            or item not in HELPER_ITEMS
            or outcome not in CALL_OUTCOMES
            or _HEX_DIGEST.fullmatch(digest) is None
        ):
            self.calls_malformed += 1
            return
        counts = self.helper_counts.get(item)
        if counts is None:
            counts = self.helper_counts[item] = dict.fromkeys(CALL_OUTCOMES, 0)
        counts[outcome] += 1

    def _failure(self, args: tuple) -> None:
        """Keep one ``memory.call_failed`` event: the id, the exception's type, the message's first line and the
        arguments' shape. Only exact ``str`` arguments are read, as for ``memory.call``."""
        if type(args) is not tuple or len(args) != 4 or any(type(a) is not str for a in args):
            self.calls_malformed += 1
            return
        item, exception, message, shape = args
        if (
            len(item) > MAX_ITEM_ID
            or (
                item not in FAILURE_HELPERS
                and _ITEM_ID.fullmatch(item) is None
                and _CAPTURE_ID.fullmatch(item) is None
            )
            or _EXCEPTION_NAME.fullmatch(exception) is None
        ):
            self.calls_malformed += 1
            return
        if len(self.call_failures) >= MAX_CALL_FAILURES:
            self.call_failures_dropped += 1
            return
        tid = threading.get_ident()
        self.call_failures.append(
            {
                "item": item,
                "exception": exception,
                "message": _line(message),
                "shape": _line(shape),
                "tid": tid,
                "cell_thread": tid == self.cell_tid,
            },
        )

    @staticmethod
    def _id(item: Any) -> Optional[str]:
        """*item* if it is a library item id or a capture id, else None."""
        if (
            type(item) is not str
            or len(item) > MAX_ITEM_ID
            or (_ITEM_ID.fullmatch(item) is None and _CAPTURE_ID.fullmatch(item) is None)
        ):
            return None
        return item

    @classmethod
    def _ids(cls, value: Any) -> Optional[list]:
        """A tuple of ids (:meth:`_id`) as a list, or None if it is not one."""
        if type(value) is not tuple or len(value) > MAX_MATCH_IDS:
            return None
        out = [cls._id(v) for v in value]
        return None if any(v is None for v in out) else out

    @classmethod
    def _rows(cls, value: Any) -> Optional[list]:
        """A tuple of ``(id, level, status, matched)`` rows as dicts, or None if it is not one."""
        if type(value) is not tuple or len(value) > MAX_MATCH_IDS:
            return None
        out = []
        for row in value:
            if type(row) is not tuple or len(row) != 4:
                return None
            item = cls._id(row[0])
            labels = row[1:]
            if item is None or any(type(x) is not str or _LABEL.fullmatch(x) is None for x in labels):
                return None
            out.append({"id": item, "level": labels[0], "status": labels[1], "matched": labels[2]})
        return out

    def _match_result(self, args: tuple) -> None:
        """Keep one ``memory.match_result`` event: the rows listed and listed as failed, how many of the cell's
        call records and runs came before it, and the procedures whose reader gave no outcome (an optional third
        element: :meth:`_facts` rows)."""
        if type(args) is not tuple or len(args) not in (2, 3):
            self.calls_malformed += 1
            return
        listed, failed = self._rows(args[0]), self._rows(args[1])
        missing = self._facts(args[2]) if len(args) == 3 else []
        if (
            listed is None
            or failed is None
            or missing is None
            or any(f["read"] not in ASK_NO_OUTCOME for f in missing)
        ):
            self.calls_malformed += 1
            return
        if len(self.match_records) >= MAX_MATCH_RECORDS:
            self.call_records_dropped += 1
            return
        record = {
            "listed": listed,
            "failed": failed,
            "calls_before": len(self.call_records),
            "runs_before": len(self.run_records),
        }
        if missing:
            record["no_outcome"] = missing
        self.match_records.append(record)

    def _run_result(self, args: tuple) -> None:
        """Keep one ``memory.run_result`` event: its target, outcome, repeatability, level, conditions and the ids
        of the matching procedures that agreed."""
        if type(args) is not tuple or len(args) != 6:
            self.calls_malformed += 1
            return
        target, outcome, repeatable, level, conditions, agrees = args
        agrees = self._ids(agrees)
        if (
            self._id(target) is None
            or agrees is None
            or type(outcome) is not str
            or outcome not in RUN_OUTCOMES
            or type(repeatable) is not str
            or repeatable not in RUN_REPEATABLE
            or type(level) is not str
            or _LABEL.fullmatch(level) is None
            or type(conditions) is not str
            or conditions not in RUN_CONDITIONS
        ):
            self.calls_malformed += 1
            return
        if len(self.run_records) >= MAX_RUN_RECORDS:
            self.call_records_dropped += 1
            return
        self.run_records.append(
            {
                "target": target,
                "outcome": outcome,
                "repeatable": repeatable,
                "level": level,
                "conditions": conditions,
                "agrees_with": agrees,
                "calls_before": len(self.call_records),
            },
        )

    @classmethod
    def _facts(cls, value: Any) -> Optional[list]:
        """A tuple of recogniser facts as dicts, or None if it is not one: ``(id, read, applies)``, or ``(id, read,
        applies, reason)`` for a reader that gave no outcome, the reason one of :data:`_ASK_REASON`'s forms."""
        if type(value) is not tuple or len(value) > MAX_MATCH_IDS:
            return None
        out = []
        for row in value:
            if (
                type(row) is not tuple
                or len(row) not in (3, 4)
                or cls._id(row[0]) is None
                or row[1] not in ASK_READ
                or row[2] not in ASK_APPLIES
                or (len(row) == 4) != (row[1] in ASK_NO_OUTCOME)
                or (
                    len(row) == 4
                    and (type(row[3]) is not str or _ASK_REASON.fullmatch(row[3]) is None or _KEY_SHAPED.search(row[3]))
                )
            ):
                return None
            fact = {"id": row[0], "read": row[1], "applies": row[2]}
            if len(row) == 4:
                fact["reason"] = row[3]
            out.append(fact)
        return out

    def _ask_result(self, args: tuple) -> None:
        """Keep one ``memory.ask_result`` event: the ids listed as fitting, whether nothing fits, the facts'
        outcome labels, and how many of the cell's call records and runs came before it."""
        if type(args) is not tuple or len(args) != 3:
            self.calls_malformed += 1
            return
        listed, nothing, facts = args
        listed = self._ids(listed)
        rows = self._facts(facts)
        if listed is None or type(nothing) is not bool or rows is None:
            self.calls_malformed += 1
            return
        if len(self.ask_records) >= MAX_MATCH_RECORDS:
            self.call_records_dropped += 1
            return
        self.ask_records.append(
            {
                "listed": listed,
                "nothing_fits": nothing,
                "facts": rows,
                "calls_before": len(self.call_records),
                "runs_before": len(self.run_records),
            },
        )

    def _learned_fit(self, args: tuple) -> None:
        """Keep one ``memory.learned_fit`` event: only exact ``str``, ``bool`` and ``int`` arguments of the
        expected shapes, within the per-cell count and byte bounds."""
        if type(args) is not tuple or len(args) != 4:
            self.calls_malformed += 1
            return
        item, state, reproduced, steps = args
        if (
            type(item) is not str
            or type(state) is not str
            or type(reproduced) is not bool
            or type(steps) is not int
            or len(item) > MAX_ITEM_ID
            or _ITEM_ID_OR_MODULE.fullmatch(item) is None
        ):
            self.calls_malformed += 1
            return
        if _KEY_SHAPED.search(state) is not None:
            # a key-shaped string in the state (its keys and config are the cell's): refused, never masked, so what is
            # kept is always the cell's JSON as given; counted as malformed
            self.calls_malformed += 1
            return
        if len(self.learned_records) >= MAX_LEARNED_FITS or self.learned_bytes + len(state) > MAX_LEARNED_BYTES:
            self.learned_dropped += 1
            return
        self.learned_bytes += len(state)
        self.learned_records.append(
            {"item": item, "state": state, "reproduced": reproduced, "steps": steps},
        )

    def _calls_seen(self) -> bool:
        return bool(
            self.learned_records
            or self.learned_dropped
            or self.call_counts
            or self.calls_dropped
            or self.calls_malformed
            or self.helper_counts
            or self.call_failures
            or self.call_failures_dropped
            or self.match_records
            or self.run_records
            or self.ask_records,
        )

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

    def _record(self, event: str, args: tuple, via: Optional[str] = None) -> None:
        if event in SPAWN_EVENTS:
            self._spawn(event, args)
        elif event == SYSTEM_EVENT:
            command = _decode(args[0]) if args else None
            if command is not None:
                record: dict = {"event": event, "exe": None}
                record["argv"] = [self._string(command, record)]
                self._append(record, RECORD_OVERHEAD + len(record["argv"][0]))
        else:
            self._path(event, args, via)

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

    def _path(self, event: str, args: tuple, via: Optional[str] = None) -> None:
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
        if via is not None and event == "open":
            record["via"] = via
        self._append(record, RECORD_OVERHEAD + len(record["path"]))

    # Controls, called from the worker's cell loop on the cell's thread.
    def begin(self) -> None:
        self.records, self.dropped, self.failed, self.bytes = [], 0, 0, 0
        self._reset_calls()
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
        if self._calls_seen():
            out["calls"] = {
                "counts": self.call_counts,
                "records": self.call_records,
                "dropped": self.calls_dropped,
                "records_dropped": self.call_records_dropped,
                "malformed": self.calls_malformed,
            }
            if self.helper_counts:
                out["calls"]["helpers"] = self.helper_counts
            if self.call_failures or self.call_failures_dropped:
                out["calls"]["failures"] = self.call_failures
                out["calls"]["failures_dropped"] = self.call_failures_dropped
            if self.match_records:
                out["calls"]["matches"] = self.match_records
            if self.run_records:
                out["calls"]["runs"] = self.run_records
            if self.ask_records:
                out["calls"]["asks"] = self.ask_records
            if self.learned_records or self.learned_dropped:
                out["calls"]["learned"] = self.learned_records
                out["calls"]["learned_dropped"] = self.learned_dropped
        self.records, self.dropped, self.failed, self.bytes = [], 0, 0, 0
        self._reset_calls()
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
