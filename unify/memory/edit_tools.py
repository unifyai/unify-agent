"""The writer's and the curator's file tools: ``write`` (a whole file) and ``edit`` (one exact replacement) under
/memory, behind ``UNIFY_MEMORY_LIBRARY_EDIT_TOOLS`` (the spec flag ``edit``).

Where they run: in the pass process on the host, as ``fixture`` and ``pool`` already do, not in the box. A box per
call would cost a confined process and a cell slot for a file write; on the host the open discipline below is the
boundary instead of the mount, and it holds while cell code changes /memory at the same moment (a cell can make
links there at any time):

- /memory's directory is opened once; every parent component is opened from the directory before it with
  ``openat`` and ``O_NOFOLLOW | O_DIRECTORY``, so a link or a non-directory on the way is refused, never followed.
  No path is checked and then opened by name (Python has no ``openat2``; the walk gives what ``RESOLVE_BENEATH |
  RESOLVE_NO_SYMLINKS`` would).
- The new bytes go to an ``O_EXCL`` temporary file in the leaf's own directory, made mode 0644 with ``fchmod``,
  and replace the leaf with ``renameat`` within that directory. A rename replaces the directory entry itself: a
  link swapped in at the leaf after the check is replaced, never written through, and a hard link elsewhere keeps
  its old bytes.

Paths are relative to /memory; absolute paths, ``..``, ``.``, empty components, NUL and text that is not UTF-8 are
refused, as is anything the gate treats as reserved: a component starting with ``.git``, the generated files
(:data:`.layout.GENERATED_ROOT_FILES`, ``.memory/``) and ``.pass/`` apart from the manifest. No size limit is set
here: the pass measures the run-safety quotas after each call, as after a cell.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path

from .layout import DATA_DIR, GENERATED_ROOT_FILES
from .tool_spec import ToolSpec
from .writer_tools import MANIFEST, Reply, path_row, refused

#: the spec flag the tools need (``UNIFY_MEMORY_LIBRARY_EDIT_TOOLS=on``)
FLAG = "edit"
#: the mode every written file gets
FILE_MODE = 0o644
_DIR_MODE = 0o755
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
_ROLES = frozenset({"write", "curate"})

_PATH = {
    "type": "string",
    "description": "relative to /memory, e.g. memory/<package>/<module>.py",
}

_PATH_REFUSALS = {
    "bad_path": "{path!r}: {reason}",
    "reserved": "{path} is reserved: {reason}",
    "link": "{path}: {what} is a link",
    "not_regular": "{path} is not a regular file",
    "not_directory": "{path}: {where} is not a directory",
    "not_written": "cannot write {path}: {error}",
}

WRITE = ToolSpec(
    name="write",
    description=(
        "Write a whole file under /memory: `content` (text) becomes the file at `path`, replacing it if it exists; "
        "missing directories are made. The file gets mode 0644. Refused: links, .git, the files the harness "
        "generates, and .pass/ except .pass/manifest.json."
    ),
    parameters={
        "type": "object",
        "properties": {"path": _PATH, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    roles=_ROLES,
    requires=FLAG,
    refusals={
        **_PATH_REFUSALS,
        "bad_content": "content must be a string of UTF-8 text; nothing was written",
    },
    summary="write a whole file under /memory (path relative to /memory).",
)

EDIT = ToolSpec(
    name="edit",
    description=(
        "Replace text in a UTF-8 file under /memory: `old` must occur exactly once in the file and becomes `new`. "
        "Exact text, no patterns; 0 or several occurrences are refused with the count (add surrounding lines to "
        "make `old` unique). The same path rules as write."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": _PATH,
            "old": {"type": "string"},
            "new": {"type": "string"},
        },
        "required": ["path", "old", "new"],
    },
    roles=_ROLES,
    requires=FLAG,
    refusals={
        **_PATH_REFUSALS,
        "not_found": "{path} does not exist; nothing was changed (write makes a new file)",
        "not_text": "{path} is not UTF-8 text; nothing was changed",
        "bad_old": "{reason}; nothing was changed",
        "match_count": "`old` occurs {count} times in {path}; it must occur exactly once (nothing was changed)",
    },
    summary="replace text that occurs exactly once in a /memory file.",
)

SPECS: tuple[ToolSpec, ...] = (WRITE, EDIT)
NAMES = tuple(s.name for s in SPECS)


class _Refused(Exception):
    def __init__(self, form: str, **fields: object) -> None:
        super().__init__(form)
        self.form, self.fields = form, fields


def memory_path(path: object) -> list[str]:
    """*path*'s components, relative to /memory, or :class:`_Refused` (``bad_path`` or ``reserved``)."""
    if not isinstance(path, str):
        raise _Refused("bad_path", path=str(path)[:80], reason="path must be a string")
    shown = path[:200]
    if "\x00" in path:
        raise _Refused("bad_path", path=shown, reason="a path holds no NUL")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        raise _Refused("bad_path", path=shown, reason="the path is not UTF-8") from None
    if not path:
        raise _Refused("bad_path", path=shown, reason="the path is empty")
    if path.startswith("/"):
        raise _Refused(
            "bad_path",
            path=shown,
            reason="give the path relative to /memory, not absolute",
        )
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise _Refused(
            "bad_path",
            path=shown,
            reason="a path component is empty, '.' or '..'",
        )
    rel = "/".join(parts)
    if any(p.lower().startswith(".git") for p in parts):
        raise _Refused("reserved", path=shown, reason="git metadata is never written")
    if rel in GENERATED_ROOT_FILES or parts[0] == DATA_DIR:
        raise _Refused(
            "reserved",
            path=shown,
            reason="the harness generates it in every copy of the library",
        )
    if parts[0] == ".pass" and rel != MANIFEST:
        raise _Refused(
            "reserved",
            path=shown,
            reason=f"under .pass/ only {MANIFEST} is written",
        )
    return parts


def _openat(name: str, flags: int, dir_fd: int, mode: int = 0o777) -> int:
    """``openat(dir_fd, name, flags)``: the one place a component is opened (tests stand in here to change the tree
    between the check and the open)."""
    return os.open(name, flags, mode, dir_fd=dir_fd)


def _what(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, type(exc).__name__)


def _parent(root_fd: int, parts: list[str], shown: str, create: bool) -> int:
    """An fd of the leaf's directory, reached component by component from *root_fd*, never through a link."""
    fd = os.dup(root_fd)
    try:
        for i, name in enumerate(parts[:-1]):
            if create:
                try:
                    os.mkdir(name, _DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise _Refused(
                        "not_written",
                        path=shown,
                        error=_what(exc),
                    ) from None
            try:
                nxt = _openat(name, _DIR_FLAGS, fd)
            except OSError as exc:
                where = "/".join(parts[: i + 1])
                if exc.errno in (errno.ELOOP, errno.ENOTDIR) and _is_link(fd, name):
                    # O_DIRECTORY may answer a link with ENOTDIR; the entry itself says which it is
                    raise _Refused(
                        "link",
                        path=shown,
                        what=f"the directory {where}",
                    ) from None
                if exc.errno == errno.ENOENT and not create:
                    raise _Refused("not_found", path=shown) from None
                if exc.errno == errno.ENOTDIR:
                    raise _Refused("not_directory", path=shown, where=where) from None
                raise _Refused("not_written", path=shown, error=_what(exc)) from None
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _is_link(dir_fd: int, name: str) -> bool:
    """Whether the entry *name* in *dir_fd* is a link (read after a refused open, only to name the refusal)."""
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def _leaf_kind(dir_fd: int, leaf: str) -> int | None:
    """The leaf's ``st_mode`` (never following a link), or None when it does not exist."""
    try:
        return os.stat(leaf, dir_fd=dir_fd, follow_symlinks=False).st_mode
    except FileNotFoundError:
        return None


def _check_leaf(mode: int | None, shown: str) -> None:
    if mode is None:
        return
    if stat.S_ISLNK(mode):
        raise _Refused("link", path=shown, what="the file")
    if not stat.S_ISREG(mode):
        raise _Refused("not_regular", path=shown)


def _replace(dir_fd: int, leaf: str, data: bytes, shown: str) -> None:
    """*data* into a new ``O_EXCL`` file beside *leaf*, mode 0644, then renamed over *leaf* within *dir_fd*."""
    tmp = f".edit-{secrets.token_hex(8)}.tmp"
    try:
        fd = _openat(tmp, _TEMP_FLAGS, dir_fd, FILE_MODE)
    except OSError as exc:
        raise _Refused("not_written", path=shown, error=_what(exc)) from None
    try:
        try:
            os.fchmod(fd, FILE_MODE)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
        os.rename(tmp, leaf, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except OSError as exc:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        if exc.errno in (errno.EISDIR, errno.ENOTEMPTY, errno.EEXIST):
            raise _Refused("not_regular", path=shown) from None
        raise _Refused("not_written", path=shown, error=_what(exc)) from None


def _read(dir_fd: int, leaf: str, shown: str) -> bytes:
    try:
        fd = _openat(leaf, _READ_FLAGS, dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _Refused("link", path=shown, what="the file") from None
        if exc.errno == errno.ENOENT:
            raise _Refused("not_found", path=shown) from None
        raise _Refused("not_written", path=shown, error=_what(exc)) from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _Refused("not_regular", path=shown)
        chunks = []
        while chunk := os.read(fd, 1 << 20):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def occurrences(text: str, old: str) -> int:
    """How many times *old* occurs in *text*, overlapping ones included (so "exactly once" means one place)."""
    n, i = 0, text.find(old)
    while i != -1:
        n += 1
        i = text.find(old, i + 1)
    return n


def _text(value: object, form: str, **fields: object) -> bytes:
    if not isinstance(value, str):
        raise _Refused(form, **fields)
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise _Refused(form, **fields) from None


def _write(root_fd: int, parts: list[str], shown: str, args: dict) -> str:
    data = _text(args.get("content"), "bad_content")
    dir_fd = _parent(root_fd, parts, shown, create=True)
    try:
        existed = _leaf_kind(dir_fd, parts[-1])
        _check_leaf(existed, shown)
        _replace(dir_fd, parts[-1], data, shown)
    finally:
        os.close(dir_fd)
    return f"ok: {'replaced' if existed is not None else 'created'} {shown} ({len(data)} bytes)"


def _edit(root_fd: int, parts: list[str], shown: str, args: dict) -> str:
    bad = {"reason": "old and new must be strings of UTF-8 text"}
    old = _text(args.get("old"), "bad_old", **bad).decode("utf-8")
    new = _text(args.get("new"), "bad_old", **bad).decode("utf-8")
    if not old:
        raise _Refused("bad_old", reason="old is empty")
    if old == new:
        raise _Refused("bad_old", reason="old and new are the same")
    dir_fd = _parent(root_fd, parts, shown, create=False)
    try:
        try:
            text = _read(dir_fd, parts[-1], shown).decode("utf-8")
        except UnicodeDecodeError:
            raise _Refused("not_text", path=shown) from None
        count = occurrences(text, old)
        if count != 1:
            raise _Refused("match_count", path=shown, count=count)
        data = text.replace(old, new, 1).encode("utf-8")
        _replace(dir_fd, parts[-1], data, shown)
    finally:
        os.close(dir_fd)
    return f"ok: edited {shown}: 1 occurrence replaced ({len(data)} bytes now)"


def call(name: str, args: dict, box: Path) -> Reply:
    """The ``write`` or ``edit`` tool on the box mounted at /memory. A refusal is a reply, never raised."""
    raw = args.get("path")
    where = (path_row(f"/memory/{str(raw)[:200]}", "write"),)
    try:
        parts = memory_path(raw)
        shown = "/".join(parts)
        try:
            root_fd = os.open(Path(box), _DIR_FLAGS)
        except OSError as exc:
            raise _Refused("not_written", path=shown, error=_what(exc)) from None
        try:
            text = (_write if name == "write" else _edit)(root_fd, parts, shown, args)
        finally:
            os.close(root_fd)
    except _Refused as r:
        return refused(name, r.form, where, **r.fields)
    return Reply(text, paths=where)
