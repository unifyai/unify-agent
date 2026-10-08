"""Work-tree adapter (spec §C2 "work tree" kind, §C3 shapes, §C5 option A, §E5 snapshot history).

:class:`WorkTreeRecorder` turns the per-cell audit records of the worker (the raw ``sys.audit`` names the
separate audit adapter emits: ``open``, ``os.listdir``, ``os.scandir``, ``os.remove``, ``os.rename``,
``os.mkdir``, ``os.rmdir``, ``shutil.rmtree``; the bare names are accepted too) plus two snapshots of the
work tree (before and after the request) into ``Action(kind="worktree")`` rows:

- ``read`` and ``list`` rows (and ``mkdir`` / ``rmdir`` / ``rmtree`` rows) are emitted per cell by
  :meth:`WorkTreeRecorder.record_cell`;
- ``write`` rows are emitted at request end by :meth:`WorkTreeRecorder.finish` from the snapshot diff,
  with the blob ids before and after: one row per cell that opened the path for writing, or one row with
  ``cell=-1`` when no audited cell did (e.g. a subprocess wrote it).

Host-side file access (the threat model)
---------------------------------------
The work tree is written by model code, which may plant symlinks, FIFOs, odd names, or swap files while
the recorder runs. Every host-side access therefore goes through file descriptors: each path component is
opened from the work-tree root with ``O_NOFOLLOW | O_DIRECTORY`` (``dir_fd``-relative), the final file with
``O_NOFOLLOW | O_NONBLOCK``; the fd is ``fstat``-ed and must be a regular file within the size cap, and the
bytes are read from that same fd (at most cap + 1 bytes). A symlink at any component, and anything that is
not a regular file, is refused with a counted note. Paths are never resolved and then reopened.
Names that are not valid UTF-8, contain ``\\r``, ``\\n`` or NUL, or have a ``.git`` component (any case)
are refused with a counted note, as is a read of the work-tree root itself and a path the audit adapter
clipped. A record that still fails is counted and skipped; it never aborts the cell. Snapshots have a
per-file cap and a total byte and file budget.

Shapes: no file format is parsed in this process
------------------------------------------------
Every parser has its own blow-up on hostile input (PyYAML's base-60 ints are quadratic; a zip lies about
its sizes), so CSV, TSV, JSON, JSONL, YAML, XLSX and JSON-looking text are parsed by
:mod:`worktree_shapes` in a child process (:func:`_run_child`). The host reads the bytes through the fds
above, redacts them, then cuts them to ``PARSE_CAP`` and sends them on the child's stdin. The child gets
no path, an empty environment, isolated interpreter flags, an import allowlist (the standard library and
the parsers), rlimits on address space, CPU, file size (0), core size (1), open files and processes (0),
``PARSE_TIMEOUT`` seconds of wall clock, and a cap on what it may print. A child that times out, hits a
limit or fails gives a fixed marker shape. Only plain text and binary files are shaped here (a decode and
a line count). Each recorder (one request) has a budget of parsed bytes and one of ``PARSE_BUDGET_S``
wall-clock seconds; once either is spent, shapes are budget markers and no child is spawned.

The child is bounded, not sandboxed: it runs as the harness user, and ``NPROC=0`` stops ``fork`` and
threads but not ``execve``. That suits internal runs; a run whose work tree can hold third-party or
adversarial bytes needs it under ``run_confined`` first. As root, ``NPROC`` is not enforced at all (see
:func:`_reap`).

Every shape passes one gate before it can enter a record (:func:`_checked`): the child's answer is parsed
with bounded depth and no ``NaN`` / ``Infinity``, checked against the shape schema of
:mod:`worktree_shapes` (only that format's keys and types, bounded strings and counts), and any violation
gives ``{"format", "unparsed": "invalid"}``; then every string in it, keys and values, is redacted.

What holds raw content (controller rulings on review item I5, R26 and R28)
-------------------------------------------------------------------------
The snapshot git dir holds the same bytes as the user's own workspace, unredacted: it lives harness-side,
is never mounted into a cell or into Sol's box, and blame / exact checkout need the real bytes.
Everything this adapter exports, i.e. the rows and the blob-store blobs they reference (which reach the
evidence store and Sol), is redacted first. The default redactor knows the secrets in ``os.environ``.

A redactor scans text, so it cannot see inside compression or any other binary encoding, wherever in the
file a stream starts. Exported blobs are therefore text only (ruling R28, which replaced R27's list of
container signatures): a file's bytes are exported only when they decode as strict UTF-8 and hold no NUL
byte (:func:`exportable_text`), whatever its name. Every other file (a zip, gzip, tar, PDF, image, a zlib
stream, a gzip appended to a log, latin-1 text) is ``withheld: "binary"``: its row holds the SHA-256 of
its bytes, its size and its (redacted) shape, and no blob. The same rule holds for the request's
work-tree diff (``Episode.worktree_diff``), which only :meth:`WorkTreeRecorder.diff` builds.

No exported hash or id is of raw text from which a secret was redacted (ruling R30): a short password's
file could otherwise be found by hashing guesses offline. Such a row has ``redacted: True`` and no git
``oid`` (its blob id, when it has a blob, is the SHA-256 of the redacted text), and the diff hashes text
after redaction. Raw-content ids are exported only for text with no redaction, and for binary files.

Shapes hold names, not data: column names and keys, and for an ``.xlsx`` its sheet titles and the cells
of each sheet's first row, which are cell values. Those come from the child, which reads an ``.xlsx``
unredacted (its bytes are deflated); the harness redacts them before they are recorded. A CSV or TSV
that is not UTF-8 is read as latin-1, so its header cells are names even when its bytes are withheld;
the redactor reads them as text like any other name.

Git runs with a minimal environment (``PATH``, a scratch ``HOME``, ``GIT_CONFIG_GLOBAL=/dev/null``,
``GIT_CONFIG_NOSYSTEM=1``), ``core.hooksPath=/dev/null``, ``core.fsmonitor=false``, an empty scratch work
tree and a bounded timeout. Blobs are written as loose objects from the bytes read here, so no git
command ever opens a work-tree path and no ``.gitattributes`` filter can run.

This module is standalone: nothing in the actor calls it yet. ``file_shape`` (and
:mod:`worktree_shapes`) must be unified with the parallel ``unify/memory_v2/analysis/shapes.py`` at
merge.
"""

from __future__ import annotations

import collections
import difflib
import functools
import hashlib
import importlib.util
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ...blobs import BlobStore
from ...episodes import Action
from ...gitio import _ENV, GitError, Repo
from ...redact import Redactor
from ...sandbox_run import PRLIMIT
from . import worktree_shapes
from .worktree_shapes import (
    ANSWER_BYTES,
    FORMATS,
    LIMIT_EXIT,
    PARSE_CAP,
    SHAPE_BYTES,
    XLSX_INFLATED,
    plain_text_shape,
)

CHANNEL = "worktree"
BLOB_CAP = 2 * 1024 * 1024  # bytes stored per file in the blob store
SNAPSHOT_CAP = 16 * 1024 * 1024  # files larger than this are left out of snapshots
SNAPSHOT_BUDGET_BYTES = 512 * 1024 * 1024  # total bytes per snapshot
SNAPSHOT_BUDGET_FILES = 50_000  # files per snapshot
MAX_DEPTH = 1024  # directory depth walked by a snapshot
GIT_TIMEOUT = 120  # seconds per git command
SHAPE_BUDGET = 64 * 1024 * 1024  # bytes parsed for shapes per recorder (request)
PARSE_TIMEOUT = 2.0  # wall-clock seconds one shape child may run
PARSE_BUDGET_S = 6.0  # wall-clock seconds of shaping per recorder (request)
PARSE_MIN_SLICE = 0.1  # seconds left below which the time budget counts as spent
# A shape child peaks near 34 MiB of address space on small input; CPU is the backstop if the
# harness never gets to check the wall clock (it dies first, say).
CHILD_AS_BYTES = 512 * 1024 * 1024  # address space of a shape child
CHILD_CPU_S = 3  # CPU seconds of a shape child
CHILD_NOFILE = 32  # open files of a shape child
CHILD_NPROC = 0  # shape child: no fork, no thread (execve still works; root ignores it)
# Core size 1, not 0: for a piped ``core_pattern`` (systemd-coredump, apport, WSL's crash capture) the
# kernel ignores RLIMIT_CORE except that 1 aborts the dump, and a file core needs at least a page. A
# crashed child's memory holds an inflated .xlsx, unredacted. A socket pattern (``@…``, Linux 6.16+)
# is not covered: its server decides, so each worker's core_pattern must be checked before wiring.
CHILD_CORE = 1
CHILD_STDOUT_CAP = ANSWER_BYTES  # bytes read from a shape child, then killed
CHILD_KILL_GRACE_S = 5.0  # seconds a SIGKILLed shape child may take to be gone
REDACT_MARGIN = 64 * 1024  # bytes past PARSE_CAP redacted before the cut
DIFF_LINES = 2000  # lines a side of a text file may have to be diffed line by line
DIFF_CAP = 256 * 1024  # characters of a work-tree diff, cut after redaction
DIFF_PATHS = 1000  # changed paths a work-tree diff shows; the rest are counted
DIFF_SECONDS = 20.0  # wall-clock seconds a work-tree diff may take, git included
LIST_ENTRIES = 200  # names kept for a directory listing
MAX_NOTES = 200  # skip notes kept verbatim (all are counted)
REQUEST_END_CELL = -1  # cell of a write that no audited cell opened
CLIP_MARK = "<clipped>"  # the audit adapter's suffix for a clipped string

_WRITE_MODE_CHARS = set("wax+")
_WRITE_EVENTS = {"remove", "unlink", "rename", "replace"}
_DIR_EVENTS = {"mkdir", "rmdir", "rmtree"}
_LIST_EVENTS = {"listdir", "scandir"}
# The audit adapter emits the raw ``sys.audit`` names; both spellings are accepted.
_EVENT_NAMES = {
    "os.listdir": "listdir",
    "os.scandir": "scandir",
    "os.remove": "remove",
    "os.rename": "rename",
    "os.mkdir": "mkdir",
    "os.rmdir": "rmdir",
    "shutil.rmtree": "rmtree",
}
_LIMIT_SIGNALS = {signal.SIGKILL, signal.SIGXCPU, signal.SIGXFSZ}
_CHILD_SCRIPT = Path(worktree_shapes.__file__)

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def _gap(stage: str, rel: str) -> None:
    """Test seam between enumerating a name ("listed") or checking its fd ("checked") and reading it."""


# -- shapes (spec §C3): parsed in a bounded child, never in this process ----------------------------
def _format_of(path: str) -> str:
    return FORMATS.get(Path(path).suffix.lower(), "text")


@functools.lru_cache(maxsize=1)
def _lib_dirs() -> tuple[str, ...]:
    """The directories holding the parser libraries (PyYAML, openpyxl and its ``et_xmlfile``), from this
    interpreter's own import path; the child appends them after its standard library."""
    dirs: list[str] = []
    for name in ("yaml", "openpyxl", "et_xmlfile"):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        found = os.path.dirname(
            os.path.abspath(list(spec.submodule_search_locations)[0]),
        )
        if found not in dirs:
            dirs.append(found)
    return tuple(dirs)


def exportable_text(data: bytes) -> bool:
    """Whether *data* may leave the harness as bytes (controller ruling R28): strict UTF-8 with no NUL.

    The only test, whatever the file's name or signature. Anything else (any binary, container or
    embedded stream, and text in another encoding) exports its SHA-256, size and redacted shape only.
    """
    if b"\0" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def file_shape(
    path: str,
    data: bytes,
    *,
    timeout: float = PARSE_TIMEOUT,
    redactor: Redactor | None = None,
) -> dict:
    """The detected format plus a structural summary of *data*, checked and redacted; never raises.

    Only the first ``PARSE_CAP`` bytes are parsed (an ``.xlsx`` whole, if at most ``XLSX_INFLATED``
    bytes). Plain text and binary files are shaped here (a decode and a line count); every other format is
    parsed by :mod:`worktree_shapes` in a child process (see :func:`_run_child`), bounded by *timeout*
    seconds of wall clock and by its rlimits. A child that times out, hits a limit or fails gives
    ``{"format", "unparsed": "timeout" | "limit" | "error"}``. Every shape then passes :func:`_checked`
    with *redactor* (by default one for the secrets in ``os.environ``). Nothing about the input or the
    child's answer can make this raise; only a child that survives SIGKILL (a host fault) raises
    ``RuntimeError``.
    """
    fmt = _format_of(path)
    if redactor is None:
        redactor = Redactor.from_environ(os.environ)
    if fmt == "xlsx":
        if len(data) > XLSX_INFLATED:
            return _checked(
                fmt,
                {"format": "xlsx", "error": "inflated_too_large"},
                redactor,
            )
        payload, truncated = data, False
    else:
        payload, truncated = data[:PARSE_CAP], len(data) > PARSE_CAP
        if fmt == "text":
            plain = plain_text_shape(payload, truncated)
            if plain is not None:
                return _checked(fmt, plain, redactor)
    return _checked(fmt, _run_child(fmt, payload, truncated, timeout), redactor)


def _checked(fmt: str, answer: bytes | dict, redactor: Redactor) -> dict:
    """The one gate between a shape and a record, for every format.

    *answer* is a child's stdout (bytes) or a shape made here. Bytes are parsed by
    :func:`worktree_shapes.parse_answer` (strict UTF-8, nesting checked before parsing, no ``NaN`` /
    ``Infinity``); the result must pass :func:`worktree_shapes.valid_shape`. Anything else gives
    ``{"format": fmt, "unparsed": "invalid"}``. Every string in a valid shape, keys and values, is then
    redacted whole, and only then are key and column names cut to ``MAX_NAME``
    (:func:`worktree_shapes.cut_names`); a shape still over ``SHAPE_BYTES`` serialised is refused as
    ``{"format", "error": "shape_too_large", "bytes"}``.
    """
    shape = answer if isinstance(answer, dict) else worktree_shapes.parse_answer(answer)
    if not worktree_shapes.valid_shape(fmt, shape):
        return {"format": fmt, "unparsed": "invalid"}
    shape = worktree_shapes.cut_names(redactor.obj(shape))
    size = len(json.dumps(shape))
    if size > SHAPE_BYTES:
        return {"format": shape["format"], "error": "shape_too_large", "bytes": size}
    return shape


def _run_child(fmt: str, data: bytes, truncated: bool, timeout: float) -> bytes | dict:
    """Shape *data* as *fmt* in a child process; its stdout (unchecked: see :func:`_checked`), or
    ``{"format", "unparsed": why}``.

    The child is ``prlimit`` exec-ing ``python -I -S -B worktree_shapes.py``: rlimits on address space,
    CPU, file size (0: it can write no file), core size (``CHILD_CORE``: no file or piped core dump), open
    files and processes (0: it cannot fork or start a thread, though ``execve`` still works; root ignores
    this), set by ``prlimit`` on itself before the interpreter starts (no ``preexec_fn``, which is unsafe
    in this threaded process); an empty environment, so no credential is passed to it (a same-user process
    can still read the harness's ``/proc/<pid>/environ``: see the module docstring on confinement); ``/``
    as its working directory; its own session, so the whole group is killed. It gets the bytes on stdin
    and never a path. Its stdout is read up to ``CHILD_STDOUT_CAP`` and stderr is discarded. On the
    wall-clock *timeout*, an oversized output or any error the group is SIGKILLed; it is always reaped and
    verified gone before this returns.
    """

    def marker(why: str) -> dict:
        return {"format": fmt, "unparsed": why}

    deadline = time.monotonic() + max(timeout, 0.0)
    argv = [
        PRLIMIT,
        f"--as={CHILD_AS_BYTES}",
        f"--cpu={CHILD_CPU_S}",
        "--fsize=0",
        f"--core={CHILD_CORE}",
        f"--nofile={CHILD_NOFILE}",
        f"--nproc={CHILD_NPROC}",
        "--",
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(_CHILD_SCRIPT),
        fmt,
        "1" if truncated else "0",
        *_lib_dirs(),
    ]
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env={},
            start_new_session=True,
        )
    except OSError:
        return marker("error")
    out, why = b"", "error"
    try:
        out, why = _exchange(proc, data, deadline)
        if why is None and not _exited(proc.pid, deadline):
            why = "timeout"
    except OSError:
        why = "error"
    finally:
        _reap(proc)
    if why is not None:
        return marker(why)
    if proc.returncode == LIMIT_EXIT or -proc.returncode in _LIMIT_SIGNALS:
        return marker("limit")
    if proc.returncode != 0:
        return marker("error")
    return out


def _exchange(
    proc: subprocess.Popen,
    data: bytes,
    deadline: float,
) -> tuple[bytes, str | None]:
    """Write *data* to the child's stdin and read its stdout until EOF, both without blocking past
    *deadline*; ``(stdout, None)``, or ``(partial, "timeout" | "limit")``."""
    out = bytearray()
    view = memoryview(data)
    wfd, rfd = proc.stdin.fileno(), proc.stdout.fileno()
    os.set_blocking(wfd, False)
    with selectors.DefaultSelector() as sel:
        sel.register(rfd, selectors.EVENT_READ)
        if view:
            sel.register(wfd, selectors.EVENT_WRITE)
        else:
            proc.stdin.close()
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return bytes(out), "timeout"
            for key, _ in sel.select(left):
                if key.fd == wfd:
                    try:
                        view = view[os.write(wfd, view[: 1 << 16]) :]
                    except BlockingIOError:
                        continue
                    except BrokenPipeError:
                        # the child stopped reading: its answer (or its failure) follows
                        view = view[:0]
                    if not view:
                        sel.unregister(wfd)
                        proc.stdin.close()
                    continue
                chunk = os.read(rfd, 1 << 16)
                if not chunk:
                    return bytes(out), None
                out += chunk
                if len(out) > CHILD_STDOUT_CAP:
                    return bytes(out), "limit"


def _exited(pid: int, deadline: float) -> bool:
    """Wait until *pid* has exited or *deadline* passes, without reaping it (its pid, and so its process
    group id, stays reserved until :func:`_reap`)."""
    while True:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)


def _reap(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group, reap the child and verify the group is empty.

    The group is signalled before the child is reaped, so its id cannot have been reused. The child cannot
    fork (``CHILD_NPROC``), so a group that outlives ``CHILD_KILL_GRACE_S`` is a fault, raised. This
    waits at most two graces: one for the child, one for its group.

    Limit as root: ``NPROC`` is not enforced, so a compromised child can fork a grandchild that calls
    ``setsid`` and leaves the group, where neither the kill nor the check sees it; and in a container whose
    PID 1 never reaps, a killed grandchild's zombie keeps the group alive, so every shape raises here
    after the grace. Root workers need the child under ``run_confined``, whose PID namespace solves both.
    """
    for pipe in (proc.stdin, proc.stdout):
        try:
            pipe.close()
        except OSError:
            pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=CHILD_KILL_GRACE_S)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"shape child {proc.pid} survived SIGKILL") from exc
    deadline = time.monotonic() + CHILD_KILL_GRACE_S
    while True:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f"shape child group {proc.pid} survived SIGKILL")
        time.sleep(0.01)


# -- fd-relative host access ----------------------------------------------------------------------
class Refused(OSError):
    """A host path the recorder will not read (symlink, not a regular file, over a cap, bad name)."""


def _bad_name(name: str) -> str | None:
    if "\n" in name or "\r" in name or "\0" in name:
        return "control character in name"
    try:
        name.encode("utf-8")
    except UnicodeEncodeError:
        return "non-UTF-8 name"
    if name.lower() == ".git":
        return ".git component"
    return None


def _open_dir(root_fd: int, parts: Iterable[str]) -> int:
    """An fd for the directory *parts* under *root_fd*; every component opened with ``O_NOFOLLOW``."""
    fd = os.open(".", _DIR_FLAGS, dir_fd=root_fd)
    try:
        for part in parts:
            nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


@dataclass
class _Read:
    data: bytes | None  # the whole file when within the cap, else None
    size: int
    prefix: bytes  # the first PARSE_CAP + REDACT_MARGIN bytes (redacted, then cut, for a shape)


def _read_regular(dir_fd: int, name: str, cap: int, rel: str) -> _Read:
    """Read *name* under *dir_fd* through one ``O_NOFOLLOW | O_NONBLOCK`` fd that must be a regular file."""
    _gap("listed", rel)
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        raise Refused(
            exc.errno,
            f"cannot open without following links: {exc.strerror}",
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise Refused(0, "not a regular file")
        _gap("checked", rel)
        want = cap + 1 if st.st_size <= cap else PARSE_CAP + REDACT_MARGIN
        chunks, got = [], 0
        while got < want:
            chunk = os.read(fd, min(1 << 20, want - got))
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(fd)
    prefix = data[: PARSE_CAP + REDACT_MARGIN]
    if st.st_size > cap or len(data) > cap:
        return _Read(None, max(st.st_size, len(data)), prefix)
    return _Read(data, len(data), prefix)


class GitTimeout(GitError):
    """A git command that ran past the time it was given."""


# -- git plumbing (spec §E5) -------------------------------------------------------------------------
def _git_raw(
    repo: Repo,
    args: list[str],
    *,
    input: bytes | None = None,
    scratch: Path | None = None,
    index: Path | None = None,
    timeout: float = GIT_TIMEOUT,
) -> bytes:
    if scratch is None:
        with tempfile.TemporaryDirectory(prefix="memv2-git-") as tmp:
            return _git_raw(
                repo,
                args,
                input=input,
                scratch=Path(tmp),
                index=index,
                timeout=timeout,
            )
    home, empty = scratch / "home", scratch / "empty"
    home.mkdir(exist_ok=True)
    empty.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": os.devnull,
        **_ENV,  # GIT_CONFIG_NOSYSTEM, no prompt, the fixed snapshot identity
    }
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    argv = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "--git-dir",
        str(repo.git_dir),
        "--work-tree",
        str(empty),
        *args,
    ]
    try:
        proc = subprocess.run(
            argv,
            input=input,
            env=env,
            cwd=str(empty),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitTimeout(
            f"git {' '.join(args[:2])}… timed out after {timeout}s",
        ) from exc
    if proc.returncode != 0:
        raise GitError(f"git {' '.join(args[:2])}… failed: {proc.stderr[:400]!r}")
    return proc.stdout


def _write_loose(git_dir: Path, data: bytes) -> str:
    """Store *data* as a loose sha1 blob object; return its id."""
    raw = b"blob %d\0" % len(data) + data
    oid = hashlib.sha1(
        raw,
    ).hexdigest()  # noqa: S324 - git's object id, not a security hash
    path = Path(git_dir) / "objects" / oid[:2] / oid[2:]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix="tmp_obj_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(zlib.compress(raw))
            os.chmod(tmp, 0o444)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    return oid


@dataclass(frozen=True)
class _Stored:
    """What a row records of one file's bytes."""

    blob: str | None  # redacted bytes in the blob store; None over the cap or withheld
    shape: dict | None  # None when the shape was not asked for
    size: int
    sha256: str | None = None  # of the raw bytes of a withheld file (R28)
    redacted: bool = False  # text with at least one secret redacted (R30)

    def withheld(self, suffix: str = "") -> dict:
        """The row fields of a withheld file (none otherwise); *suffix* is ``_before`` or ``_after``."""
        if self.sha256 is None:
            return {}
        return {f"withheld{suffix}": "binary", f"sha256{suffix}": self.sha256}

    def ids(self, oid: str, suffix: str = "") -> dict:
        """The row's git id of the file: the raw bytes' *oid*, or None and ``redacted: True`` when a
        secret was redacted from it (ruling R30: a raw-content id would let a short secret be guessed
        offline; the blob id, when there is a blob, is of the redacted text)."""
        if self.redacted:
            return {f"oid{suffix}": None, f"redacted{suffix}": True}
        return {f"oid{suffix}": oid}


@dataclass
class Snapshot:
    sha: str
    files: dict[str, str]  # path -> git blob id
    notes: list[str]
    counts: collections.Counter


class _Notes:
    def __init__(self) -> None:
        self.notes: list[str] = []
        self.counts: collections.Counter = collections.Counter()

    def add(self, reason: str, rel: str) -> None:
        self.counts[reason] += 1
        if len(self.notes) < MAX_NOTES:
            safe = rel.encode("utf-8", "backslashreplace").decode(
                "ascii",
                "backslashreplace",
            )
            self.notes.append(f"{reason}: {safe[:200]!r}")


def snapshot(
    repo: Repo,
    work_tree: Path,
    message: str,
    *,
    cap: int = SNAPSHOT_CAP,
    budget_bytes: int = SNAPSHOT_BUDGET_BYTES,
    budget_files: int = SNAPSHOT_BUDGET_FILES,
) -> Snapshot:
    """Commit the work tree onto ``main`` of *repo*, reading every file through a checked fd."""
    notes = _Notes()
    if _git_raw(repo, ["rev-parse", "--show-object-format"]).decode().strip() != "sha1":
        raise GitError("the work-tree snapshot repo must use sha1 objects")
    files: dict[str, str] = {}
    modes: dict[str, str] = {}
    total = 0
    root_fd = os.open(str(work_tree), _DIR_FLAGS)
    try:
        pending: list[tuple[str, ...]] = [()]
        while pending:
            parts = pending.pop()
            prefix = "/".join(parts)
            try:
                dfd = _open_dir(root_fd, parts)
            except OSError:
                notes.add("directory refused (symlink or vanished)", prefix)
                continue
            try:
                entries = sorted(os.listdir(dfd))
                subdirs = []
                for name in entries:
                    rel = f"{prefix}/{name}" if prefix else name
                    bad = _bad_name(name)
                    if bad:
                        notes.add(bad, rel)
                        continue
                    try:
                        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                    except OSError:
                        notes.add("vanished or unreadable", rel)
                        continue
                    if stat.S_ISDIR(st.st_mode):
                        if len(parts) + 1 > MAX_DEPTH:
                            notes.add("deeper than the depth cap", rel)
                        else:
                            subdirs.append(parts + (name,))
                        continue
                    if not stat.S_ISREG(st.st_mode):
                        notes.add("not a regular file", rel)
                        continue
                    if st.st_size > cap:
                        notes.add("over the per-file cap", rel)
                        continue
                    if len(files) >= budget_files or total + st.st_size > budget_bytes:
                        notes.add("over the snapshot budget", rel)
                        continue
                    try:
                        got = _read_regular(dfd, name, cap, rel)
                    except Refused as exc:
                        notes.add(f"refused ({exc.strerror})", rel)
                        continue
                    if got.data is None:
                        notes.add("over the per-file cap", rel)
                        continue
                    if total + got.size > budget_bytes:
                        notes.add("over the snapshot budget", rel)
                        continue
                    total += got.size
                    files[rel] = _write_loose(repo.git_dir, got.data)
                    modes[rel] = "100755" if st.st_mode & 0o111 else "100644"
                pending.extend(reversed(subdirs))
            finally:
                os.close(dfd)
    finally:
        os.close(root_fd)
    with tempfile.TemporaryDirectory(prefix="memv2-wt-snap-") as tmp:
        scratch = Path(tmp)
        index = scratch / "index"
        info = b"".join(
            f"{modes[rel]} {oid}\t".encode() + rel.encode("utf-8") + b"\0"
            for rel, oid in files.items()
        )
        _git_raw(
            repo,
            ["update-index", "-z", "--index-info"],
            input=info,
            scratch=scratch,
            index=index,
        )
        tree = (
            _git_raw(repo, ["write-tree"], scratch=scratch, index=index)
            .decode()
            .strip()
        )
        try:
            parent = (
                _git_raw(
                    repo,
                    ["rev-parse", "--verify", "-q", "refs/heads/main"],
                    scratch=scratch,
                )
                .decode()
                .strip()
            )
        except GitError:
            parent = ""
        args = ["commit-tree", tree, "-m", message] + (["-p", parent] if parent else [])
        sha = _git_raw(repo, args, scratch=scratch).decode().strip()
        _git_raw(
            repo,
            ["update-ref", "refs/heads/main", sha] + ([parent] if parent else []),
            scratch=scratch,
        )
    return Snapshot(sha, files, notes.notes, notes.counts)


def tree_files(repo: Repo, sha: str, *, timeout: float = GIT_TIMEOUT) -> dict[str, str]:
    """``path -> git blob id`` of every file in commit *sha*."""
    out: dict[str, str] = {}
    listing = _git_raw(
        repo,
        ["ls-tree", "-r", "-z", "--full-tree", sha],
        timeout=timeout,
    )
    for rec in listing.split(b"\0"):
        if not rec:
            continue
        meta, _, raw = rec.partition(b"\t")
        _mode, kind, oid = meta.decode("ascii").split(" ")
        if kind == "blob":
            out[raw.decode("utf-8", errors="surrogateescape")] = oid
    return out


def worktree_diff(
    repo: Repo,
    before: str,
    after: str,
    *,
    redactor: Redactor,
    blob_cap: int = BLOB_CAP,
    cap: int = DIFF_CAP,
    max_paths: int = DIFF_PATHS,
    seconds: float = DIFF_SECONDS,
) -> str:
    """The text of ``Episode.worktree_diff`` between snapshot commits *before* and *after* (ruling R28).

    Built here from the snapshot blobs, never by ``git diff``: ``--binary`` emits base85 of every file,
    and git's own text test only looks for a NUL in the first 8000 bytes, so a small compressed file would
    be printed as text. A changed path whose sides are both :func:`exportable_text` within *blob_cap*
    bytes and ``DIFF_LINES`` lines gets a unified diff of its sides, each redacted whole first (so a
    multi-line secret is never split by the diff's line prefixes). Every other changed path gets one line
    with each side's SHA-256 (of its redacted text for ``text``, of its raw bytes for ``binary``: ruling
    R30) and size and no bytes: ``binary``, ``text`` (over a limit, or an empty file added or deleted),
    or ``text …: changed; content redacted`` when redaction hides the whole change. A path whose blobs
    cannot be read gets ``failed <path>: <error type>``. Every part is redacted again
    (paths included).

    Bounded three ways, every path cut away by one counted in a closing ``[N changed paths not shown:
    why]`` line: at most *max_paths* paths; *seconds* of wall clock over every git call, the two tree
    listings included (each call is given only the time left, and one that times out cuts the diff off
    there); and *cap* characters. Parts are kept whole within the cap; only a first part longer than the
    cap is cut, after redaction, and marked. Never raises: snapshot trees that cannot be listed in time,
    or at all, give a single marker line.

    *redactor* has no default: it must be the one the request's records use (the recorder's), so the diff
    never redacts less than the rows do. :meth:`WorkTreeRecorder.diff` passes it.
    """
    deadline = time.monotonic() + seconds

    def left() -> float:
        return max(deadline - time.monotonic(), 0.1)

    try:
        old = tree_files(repo, before, timeout=left())
        new = tree_files(repo, after, timeout=left())
    except GitTimeout:
        return "[work-tree diff not shown: deadline]\n"
    except Exception as exc:  # noqa: BLE001 - a diff never aborts the caller
        return f"[work-tree diff unavailable: {type(exc).__name__}]\n"
    paths = sorted(p for p in set(old) | set(new) if old.get(p) != new.get(p))
    parts: list[str] = []
    size, why, cut = 0, None, False
    for path in paths:
        if len(parts) >= max_paths:
            why = "path limit"
        elif time.monotonic() >= deadline:
            why = "deadline"
        if why:
            break
        try:
            part = _path_diff(
                repo,
                path,
                old.get(path),
                new.get(path),
                redactor,
                blob_cap,
                deadline,
            )
        except GitTimeout:
            why = "deadline"  # this path and the rest are cut off, and counted
            break
        except Exception as exc:  # noqa: BLE001 - one path never loses the rest
            part = f"failed {path}: {type(exc).__name__}\n"
        part = redactor.text(part)
        if size + len(part) > cap:
            why = "size cap"
            if parts:
                break  # parts are shown whole or not at all, and those not shown are counted
            # a first part alone over the cap: cut, after redaction
            part, cut = part[:cap], True
        parts.append(part)
        size += len(part)
        if cut:
            break
    out = "".join(parts)
    if cut:
        out += f"\n[work-tree diff cut at {cap} characters]\n"
    if len(paths) > len(parts):
        out += f"[{len(paths) - len(parts)} changed paths not shown: {why}]\n"
    return out


def _path_diff(
    repo: Repo,
    path: str,
    a_oid: str | None,
    b_oid: str | None,
    redactor: Redactor,
    blob_cap: int,
    deadline: float,
) -> str:
    a, b = (
        (
            None
            if oid is None
            else _git_raw(
                repo,
                ["cat-file", "blob", oid],
                timeout=max(deadline - time.monotonic(), 0.1),
            )
        )
        for oid in (a_oid, b_oid)
    )
    sides = [x for x in (a, b) if x is not None]
    text = all(exportable_text(x) for x in sides)
    # text is hashed as redacted (ruling R30): a raw hash would let a short secret be guessed offline
    ra, rb = (
        redactor.text(x.decode("utf-8")) if text and x is not None else None
        for x in (a, b)
    )
    note = ""
    if text and all(len(x) <= blob_cap for x in sides):
        a_lines, b_lines = (_lines(x) if x is not None else [] for x in (ra, rb))
        if max(len(a_lines), len(b_lines)) <= DIFF_LINES:
            hunks = "".join(
                difflib.unified_diff(
                    a_lines,
                    b_lines,
                    "/dev/null" if a is None else f"a/{path}",
                    "/dev/null" if b is None else f"b/{path}",
                ),
            )
            if hunks:
                return hunks
            if a is not None and b is not None:
                note = " changed; content redacted,"

    def digest(x: bytes | None, red: str | None) -> str:
        if x is None:
            return "absent"
        return hashlib.sha256(red.encode("utf-8") if text else x).hexdigest()

    def length(x: bytes | None) -> str:
        return "absent" if x is None else str(len(x))

    return (
        f"{'text' if text else 'binary'} {path}:{note} sha256 {digest(a, ra)} -> "
        f"{digest(b, rb)}, bytes {length(a)} -> {length(b)}\n"
    )


def _lines(text: str) -> list[str]:
    """*text* as lines that each end in a newline (a missing final one is marked, as git does).

    Split on ``\\n`` only: ``str.splitlines`` also breaks on ``\\r``, form feeds, ``\\x85``, U+2028 and
    the like, which end no output line, so the diff would glue its next line onto them.
    """
    lines = [line + "\n" for line in text.split("\n")]
    # what followed the final newline, plus one: "\\n" alone when there was nothing
    last = lines.pop()
    if last != "\n":
        lines.append(last + "\\ No newline at end of file\n")
    return lines


# -- the recorder ---------------------------------------------------------------------------------------
class WorkTreeRecorder:
    """Records one request's reads, writes and listings of a work tree (see the module docstring)."""

    def __init__(
        self,
        work_tree: Path,
        snapshot_repo: Repo,
        blobs: BlobStore,
        *,
        redactor: Redactor | None = None,
        blob_cap: int = BLOB_CAP,
        snapshot_cap: int = SNAPSHOT_CAP,
        budget_bytes: int = SNAPSHOT_BUDGET_BYTES,
        budget_files: int = SNAPSHOT_BUDGET_FILES,
        shape_budget: int = SHAPE_BUDGET,
        parse_seconds: float = PARSE_BUDGET_S,
    ) -> None:
        self.work_tree = Path(work_tree).resolve()
        git_dir = Path(snapshot_repo.git_dir).resolve()
        if git_dir == self.work_tree or self.work_tree in git_dir.parents:
            raise ValueError("the snapshot git dir must live outside the work tree")
        self.repo = snapshot_repo
        self.blobs = blobs
        self.redactor = (
            redactor if redactor is not None else Redactor.from_environ(os.environ)
        )
        self.blob_cap = blob_cap
        self.snapshot_cap = snapshot_cap
        self.budget_bytes = budget_bytes
        self.budget_files = budget_files
        self._shape_left = shape_budget  # bytes still to be parsed for shapes
        self._parse_left = (
            parse_seconds  # wall-clock seconds still to be spent on shapes
        )
        self.before: str | None = None
        self.after: str | None = None
        self._notes = _Notes()
        self._before_files: dict[str, str] = {}
        self._written: dict[str, list[int]] = (
            {}
        )  # path -> cells that opened it for writing, in order
        self._tree_ops: list[tuple[str, int]] = (
            []
        )  # (directory, cell) of rmtree / rmdir / rename
        self._seen: set[tuple[int, str, str]] = set()
        self._stored: dict[tuple[str, str], _Stored] = {}  # (oid, ext) -> row data
        self.actions: list[Action] = []

    @property
    def skipped(self) -> list[str]:
        """Verbatim notes for the first ``MAX_NOTES`` refused or skipped entries."""
        return self._notes.notes

    @property
    def skip_counts(self) -> collections.Counter:
        """Every refused or skipped entry, counted by reason."""
        return self._notes.counts

    def _snapshot(self, message: str) -> Snapshot:
        snap = snapshot(
            self.repo,
            self.work_tree,
            message,
            cap=self.snapshot_cap,
            budget_bytes=self.budget_bytes,
            budget_files=self.budget_files,
        )
        for reason, n in snap.counts.items():
            self._notes.counts[f"{message}: {reason}"] += n
        room = MAX_NOTES - len(self._notes.notes)
        self._notes.notes.extend(f"{message}: {n}" for n in snap.notes[: max(room, 0)])
        return snap

    # -- request boundaries --------------------------------------------------------------------------
    def begin(self, message: str = "worktree_before") -> str:
        snap = self._snapshot(message)
        self.before, self._before_files = snap.sha, snap.files
        return snap.sha

    def finish(self, message: str = "worktree_after") -> list[Action]:
        """Snapshot after the request; one ``write`` row per (writing cell, changed or write-opened path)."""
        if self.before is None:
            raise RuntimeError("finish() before begin()")
        snap = self._snapshot(message)
        self.after, after_files = snap.sha, snap.files
        changed = {
            p
            for p in set(self._before_files) | set(after_files)
            if self._before_files.get(p) != after_files.get(p)
        }
        rows = []
        for path in sorted(changed | set(self._written)):
            before_oid = self._before_files.get(path)
            after_oid = after_files.get(path)
            try:
                status, response = self._write_response(path, before_oid, after_oid)
            except Exception as exc:  # noqa: BLE001 - one path never loses the rest
                self._notes.add("write record failed", path)
                # whether a secret was redacted from it is unknown, so no raw git id (R30)
                status, response = "unrecorded", {
                    "blob_before": None,
                    "blob_after": None,
                    "oid_before": None,
                    "oid_after": None,
                    "failed": type(exc).__name__,
                }
            for cell in self._writers(path):
                rows.append(self._action(cell, "write", path, dict(response), status))
        self.actions.extend(rows)
        return rows

    def _write_response(
        self,
        path: str,
        before_oid: str | None,
        after_oid: str | None,
    ) -> tuple[str, dict]:
        if before_oid is None and after_oid is None:
            # opened for writing but absent from both snapshots (deleted again, refused or skipped)
            return "unrecorded", {"blob_before": None, "blob_after": None}
        # the before side is stored but not shaped: only the after shape is recorded
        before = self._content(path, before_oid, shape=False) if before_oid else None
        after = self._content(path, after_oid) if after_oid else None
        response: dict[str, Any] = {
            "blob_before": before.blob if before else None,
            "blob_after": after.blob if after else None,
            "oid_before": None,
            "oid_after": None,
            "shape": after.shape if after else None,
            "size": after.size if after else before.size,
            "deleted": after_oid is None,
        }
        if before:
            response.update(before.ids(before_oid, "_before"))
            response.update(before.withheld("_before"))
        if after:
            response.update(after.ids(after_oid, "_after"))
            response.update(after.withheld("_after"))
        return "ok", response

    def diff(self, **bounds: Any) -> str:
        """:func:`worktree_diff` of this request (``begin`` to ``finish`` snapshots) with this recorder's
        redactor and blob cap; *bounds* are its ``cap``, ``max_paths`` and ``seconds``. The one way to fill
        ``Episode.worktree_diff``."""
        if self.before is None or self.after is None:
            raise RuntimeError("diff() before finish()")
        return worktree_diff(
            self.repo,
            self.before,
            self.after,
            redactor=self.redactor,
            blob_cap=self.blob_cap,
            **bounds,
        )

    def _writers(self, path: str) -> list[int]:
        if path in self._written:
            return self._written[path]
        cells = [c for d, c in self._tree_ops if d == "." or path.startswith(d + "/")]
        return list(dict.fromkeys(cells)) or [REQUEST_END_CELL]

    # -- per cell ------------------------------------------------------------------------------------
    def record_cell(self, cell: int, audit: Iterable[dict]) -> list[Action]:
        """Rows for one cell's audit records; write opens are remembered for :meth:`finish`."""
        if self.before is None:
            raise RuntimeError("record_cell() before begin()")
        rows: list[Action] = []
        for rec in audit:
            try:
                self._record(cell, rec, rows)
            except Exception as exc:  # noqa: BLE001 - one record never aborts the cell
                self._notes.add("audit record failed", type(exc).__name__)
        self.actions.extend(rows)
        return rows

    def _record(self, cell: int, rec: dict, rows: list[Action]) -> None:
        event = rec.get("event")
        event = _EVENT_NAMES.get(event, event) if isinstance(event, str) else None
        if event == "open":
            mode = str(rec.get("mode") or "r")
            method = "write" if _WRITE_MODE_CHARS & set(mode) else "read"
        elif event in _LIST_EVENTS:
            method = "list"
        elif event in _WRITE_EVENTS or event in _DIR_EVENTS:
            method = event
        else:
            return
        paths = [rec.get("path")]
        if event in ("rename", "replace"):
            paths.append(rec.get("dest", rec.get("dst")))
        for raw in paths:
            if rec.get("clipped") and isinstance(raw, str) and raw.endswith(CLIP_MARK):
                # not a real path; the note keeps (redacted) what the audit adapter kept of it
                self._notes.add("clipped path", self.redactor.text(raw))
                continue
            parts = self._relative(raw)
            if parts is None:
                continue
            rel = "/".join(parts) or "."
            if method in ("write",) or method in _WRITE_EVENTS:
                self._written.setdefault(rel, [])
                if cell not in self._written[rel]:
                    self._written[rel].append(cell)
                if event in ("rename", "replace"):
                    self._tree_ops.append((rel, cell))
                continue
            if (cell, rel, method) in self._seen:
                continue
            if method == "read":
                rows.append(self._read_row(cell, rel, parts))
            elif method == "list":
                rows.append(self._list_row(cell, rel, parts))
            else:
                if method in ("rmdir", "rmtree"):
                    self._tree_ops.append((rel, cell))
                rows.append(self._dir_row(cell, method, rel, parts))
            # only once its row exists: a record that failed is retried by the next identical one
            self._seen.add((cell, rel, method))

    def _read_row(self, cell: int, rel: str, parts: tuple[str, ...]) -> Action:
        if rel not in self._written and rel in self._before_files:
            oid = self._before_files[rel]
            got = self._content(rel, oid)
            response = {
                "blob": got.blob,
                **got.ids(oid),
                "source": "before",
                "shape": got.shape,
                "size": got.size,
                **got.withheld(),
            }
            return self._action(cell, "read", rel, response, "ok")
        if not parts:  # the work-tree root itself (open(<work tree>), os.open("."))
            return self._refused(cell, "read", rel, "not a regular file")
        # written earlier in this request, or not in the snapshot: the disk state after this cell,
        # read through a checked fd
        try:
            dfd = self._dir_fd(parts[:-1])
        except (OSError, ValueError):  # ValueError: a NUL in a component
            return self._refused(
                cell,
                "read",
                rel,
                "directory refused (symlink, not a directory or absent)",
            )
        try:
            got = _read_regular(dfd, parts[-1], self.snapshot_cap, rel)
        except (OSError, ValueError) as exc:
            why = getattr(exc, "strerror", None) or type(exc).__name__
            return self._refused(cell, "read", rel, f"refused ({why})")
        finally:
            os.close(dfd)
        if got.data is None:
            shape = {**self._shape(rel, self._redact(got.prefix)), "truncated": True}
            response = {
                "blob": None,
                "oid": None,
                "source": "disk",
                "shape": shape,
                "size": got.size,
            }
        else:
            stored = self._store(rel, got.data)
            response = {
                "blob": stored.blob,
                **stored.ids(None),
                "source": "disk",
                "shape": stored.shape,
                "size": stored.size,
                **stored.withheld(),
            }
        return self._action(cell, "read", rel, response, "ok")

    def _list_row(self, cell: int, rel: str, parts: tuple[str, ...]) -> Action:
        try:
            dfd = self._dir_fd(parts)
        except (OSError, ValueError):
            return self._refused(
                cell,
                "list",
                rel,
                "directory refused (symlink, not a directory or absent)",
            )
        try:
            names = sorted(os.listdir(dfd))
        except OSError:
            return self._refused(cell, "list", rel, "directory unreadable")
        finally:
            os.close(dfd)
        shown = [
            os.fsencode(n).decode("utf-8", "backslashreplace")
            for n in names
            if n.lower() != ".git"
        ]
        response = {"entries": shown[:LIST_ENTRIES], "count": len(shown)}
        return self._action(cell, "list", rel, response, "ok")

    def _dir_row(
        self,
        cell: int,
        method: str,
        rel: str,
        parts: tuple[str, ...],
    ) -> Action:
        try:
            os.close(self._dir_fd(parts))
            exists = True
        except (OSError, ValueError):
            exists = False
        return self._action(cell, method, rel, {"exists_after_cell": exists}, "ok")

    def _refused(self, cell: int, method: str, rel: str, reason: str) -> Action:
        self._notes.add(reason, rel)
        response = {"blob": None, "shape": None, "size": None, "refused": reason}
        if method == "list":
            response = {"entries": None, "count": None, "refused": reason}
        return self._action(cell, method, rel, response, "unrecorded")

    def _dir_fd(self, parts: Iterable[str]) -> int:
        """An fd for the directory *parts*, walked from a fresh root fd with ``O_NOFOLLOW`` throughout."""
        root = os.open(str(self.work_tree), _DIR_FLAGS)
        try:
            return _open_dir(root, parts)
        finally:
            os.close(root)

    # -- content -------------------------------------------------------------------------------------
    def _content(self, rel: str, oid: str, *, shape: bool = True) -> _Stored:
        """:meth:`_store` of snapshot blob *oid*, cached per (oid, suffix); shaped only if *shape*."""
        key = (oid, Path(rel).suffix.lower())
        hit = self._stored.get(key)
        if hit is None or (shape and hit.shape is None):
            hit = self._store(
                rel,
                _git_raw(self.repo, ["cat-file", "blob", oid]),
                shape=shape,
            )
            self._stored[key] = hit
        return hit

    def _store(self, rel: str, data: bytes, *, shape: bool = True) -> _Stored:
        """Redact and (up to the cap) store *data*, and shape it if *shape*; the shape is taken from the
        redacted bytes.

        Only text is ever stored (:func:`exportable_text`, controller ruling R28): for any other file the
        row gets the SHA-256 of the raw bytes, the size and the shape instead. Text is redacted whole,
        over the blob cap too, so ``redacted`` (ruling R30) knows of every secret in it.
        """
        size = len(data)
        text = exportable_text(data)
        over = size > self.blob_cap
        hits = self.redactor.hits
        if text:
            clean = self._redact(data)
        else:
            # over the cap, redacted before file_shape cuts it to PARSE_CAP, so no secret is cut first
            clean = self._redact(data[: PARSE_CAP + REDACT_MARGIN] if over else data)
        redacted = text and self.redactor.hits > hits
        if over:
            clean = clean[: PARSE_CAP + REDACT_MARGIN]
        blob = self.blobs.put(clean) if text and not over else None
        got = None
        if shape:
            got = self._shape(rel, clean)
            if over:
                got = {**got, "truncated": True}
        digest = None if text else hashlib.sha256(data).hexdigest()
        return _Stored(blob, got, size, digest, redacted)

    def _shape(self, rel: str, data: bytes) -> dict:
        """``file_shape`` within the recorder's budgets of parsed bytes and of wall-clock seconds.

        Once either is spent, a shape is ``{"format", "error": "budget"}`` and no child is spawned. Each
        child's timeout is the per-file ``PARSE_TIMEOUT`` or the seconds left, whichever is less, and its
        whole cost (kill and reaping included) is charged to the budget. So one request spends at most
        ``PARSE_BUDGET_S`` plus, for the last child, up to two ``CHILD_KILL_GRACE_S`` if the host fails
        to kill it (one for the child, one for its group; see :func:`_reap`).
        """
        fmt = _format_of(rel)
        cost = len(data) if fmt == "xlsx" else min(len(data), PARSE_CAP)
        if cost > self._shape_left:
            self._notes.add("shape budget spent", rel)
            return {"format": fmt, "error": "budget"}
        if self._parse_left < PARSE_MIN_SLICE:
            self._notes.add("shape time budget spent", rel)
            return {"format": fmt, "error": "budget"}
        self._shape_left -= cost
        start = time.monotonic()
        try:
            return file_shape(
                rel,
                data,
                timeout=min(PARSE_TIMEOUT, self._parse_left),
                redactor=self.redactor,
            )
        finally:
            self._parse_left -= time.monotonic() - start

    def _redact(self, data: bytes) -> bytes:
        try:
            return self.redactor.text(data.decode("utf-8")).encode("utf-8")
        except UnicodeDecodeError:
            # latin-1 round-trips every byte, so key-shaped ASCII secrets in binary files are caught too
            return self.redactor.text(data.decode("latin-1")).encode("latin-1")

    # -- helpers -------------------------------------------------------------------------------------
    def _relative(self, path: Any) -> tuple[str, ...] | None:
        """The lexical components of *path* under the work tree, or None (counted) if refused."""
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, (str, os.PathLike)):
            return None
        p = Path(os.fspath(path))
        if not p.is_absolute():
            p = self.work_tree / p
        p = Path(os.path.normpath(p))
        if p != self.work_tree and self.work_tree not in p.parents:
            return None  # another root; this adapter records the work tree only
        parts = p.relative_to(self.work_tree).parts
        for part in parts:
            bad = _bad_name(part)
            if bad:
                self._notes.add(
                    bad,
                    os.fsencode(str(p)).decode("utf-8", "backslashreplace"),
                )
                return None
        return tuple(parts)

    def _action(
        self,
        cell: int,
        method: str,
        rel: str,
        response: dict,
        status: str,
    ) -> Action:
        return Action(
            cell=cell,
            channel=CHANNEL,
            method=method,
            args=[rel],
            kwargs={},
            response=response,
            status=status,
            effect="read" if method in ("read", "list") else "write",
            kind="worktree",
        )
