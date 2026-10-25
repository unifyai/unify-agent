"""Writing a file under /memory from the pass process on the host, never through a link: the draft manifest
(:func:`.tool_handlers.write_draft_manifest`) and a landed capture's files (:mod:`.land_capture`).

On the host the open discipline below is the boundary instead of the mount, and it holds while cell code changes
/memory at the same moment (a cell can make links there at any time):

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
(:data:`.layout.GENERATED_ROOT_FILES`, ``.memory/``) and ``.pass/`` apart from the manifest.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path

from unify.common.tool_registry import REFUSED

from . import safe_paths as _safe
from .layout import DATA_DIR, GENERATED_ROOT_FILES
from .writer_tools import MANIFEST

#: the mode every written file gets
FILE_MODE = 0o644
_DIR_MODE = 0o755
# the open flags every descriptor-relative host handler shares (:mod:`.safe_paths`)
_DIR_FLAGS = _safe.DIR_FLAGS
_TEMP_FLAGS = _safe.TEMP_FLAGS

#: Why a write is refused, by form code.
REFUSALS = {
    "bad_path": "{path!r}: {reason}",
    "reserved": "{path} is reserved: {reason}",
    "link": "{path}: {what} is a link",
    "not_regular": "{path} is not a regular file",
    "not_directory": "{path}: {where} is not a directory",
    "not_written": "cannot write {path}: {error}",
    "bad_content": "content must be a string of UTF-8 text; nothing was written",
}


class Refused(Exception):
    """A write refused: its form code (:data:`REFUSALS`) and the fields its text names."""

    def __init__(self, form: str, **fields: object) -> None:
        super().__init__(form)
        self.form, self.fields = form, fields

    @property
    def text(self) -> str:
        return REFUSED + REFUSALS[self.form].format(**self.fields)


def memory_path(path: object) -> list[str]:
    """*path*'s components, relative to /memory, or :class:`Refused` (``bad_path`` or ``reserved``)."""
    if not isinstance(path, str):
        raise Refused("bad_path", path=str(path)[:80], reason="path must be a string")
    shown = path[:200]
    if "\x00" in path:
        raise Refused("bad_path", path=shown, reason="a path holds no NUL")
    try:
        path.encode("utf-8")
    except UnicodeEncodeError:
        raise Refused("bad_path", path=shown, reason="the path is not UTF-8") from None
    if not path:
        raise Refused("bad_path", path=shown, reason="the path is empty")
    if path.startswith("/"):
        raise Refused(
            "bad_path",
            path=shown,
            reason="give the path relative to /memory, not absolute",
        )
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise Refused(
            "bad_path",
            path=shown,
            reason="a path component is empty, '.' or '..'",
        )
    rel = "/".join(parts)
    if any(p.lower().startswith(".git") for p in parts):
        raise Refused("reserved", path=shown, reason="git metadata is never written")
    if rel in GENERATED_ROOT_FILES or parts[0] == DATA_DIR:
        raise Refused(
            "reserved",
            path=shown,
            reason="the harness generates it in every copy of the library",
        )
    if parts[0] == ".pass" and rel != MANIFEST:
        raise Refused(
            "reserved",
            path=shown,
            reason=f"under .pass/ only {MANIFEST} is written",
        )
    return parts


def _openat(name: str, flags: int, dir_fd: int, mode: int) -> int:
    """``openat(dir_fd, name, flags)``: the one place a component is opened (tests stand in here to change the tree
    between the check and the open)."""
    return _safe.openat(name, flags, dir_fd, mode)


def _what(exc: OSError) -> str:
    return _safe.errno_name(exc)


def _parent(root_fd: int, parts: list[str], shown: str) -> int:
    """An fd of the leaf's directory, reached component by component from *root_fd*, never through a link."""
    fd = os.dup(root_fd)
    try:
        for i, name in enumerate(parts[:-1]):
            try:
                os.mkdir(name, _DIR_MODE, dir_fd=fd)
            except FileExistsError:
                pass
            except OSError as exc:
                raise Refused("not_written", path=shown, error=_what(exc)) from None
            try:
                nxt = _openat(name, _DIR_FLAGS, fd, _safe.NO_CREATE_MODE)
            except OSError as exc:
                where = "/".join(parts[: i + 1])
                if exc.errno in (errno.ELOOP, errno.ENOTDIR) and _is_link(fd, name):
                    # O_DIRECTORY may answer a link with ENOTDIR; the entry itself says which it is
                    raise Refused(
                        "link",
                        path=shown,
                        what=f"the directory {where}",
                    ) from None
                if exc.errno == errno.ENOTDIR:
                    raise Refused("not_directory", path=shown, where=where) from None
                raise Refused("not_written", path=shown, error=_what(exc)) from None
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


_is_link = _safe.is_link


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
        raise Refused("link", path=shown, what="the file")
    if not stat.S_ISREG(mode):
        raise Refused("not_regular", path=shown)


def _replace(dir_fd: int, leaf: str, data: bytes, shown: str) -> None:
    """*data* into a new ``O_EXCL`` file beside *leaf*, mode 0644, then renamed over *leaf* within *dir_fd*."""
    tmp = f".write-{secrets.token_hex(8)}.tmp"
    try:
        fd = _openat(tmp, _TEMP_FLAGS, dir_fd, FILE_MODE)
    except OSError as exc:
        raise Refused("not_written", path=shown, error=_what(exc)) from None
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
            raise Refused("not_regular", path=shown) from None
        raise Refused("not_written", path=shown, error=_what(exc)) from None


def _text(value: object, form: str, **fields: object) -> bytes:
    if not isinstance(value, str):
        raise Refused(form, **fields)
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError:
        raise Refused(form, **fields) from None


def write(box: Path, path: str, content: str) -> str:
    """Write *content* as the file at *path* (relative to /memory, the box at *box*), replacing it if it exists;
    missing directories are made. Returns what was done; :class:`Refused` when the path or content is refused."""
    parts = memory_path(path)
    shown = "/".join(parts)
    data = _text(content, "bad_content")
    try:
        root_fd = os.open(Path(box), _DIR_FLAGS)
    except OSError as exc:
        raise Refused("not_written", path=shown, error=_what(exc)) from None
    try:
        dir_fd = _parent(root_fd, parts, shown)
        try:
            existed = _leaf_kind(dir_fd, parts[-1])
            _check_leaf(existed, shown)
            _replace(dir_fd, parts[-1], data, shown)
        finally:
            os.close(dir_fd)
    finally:
        os.close(root_fd)
    return f"{'replaced' if existed is not None else 'created'} {shown} ({len(data)} bytes)"
