"""Open box paths on the host from a root directory descriptor, one component at a time, never through a link.

Box code can change /memory, /work and /outputs while a host handler reads or writes them: it can swap a directory
for a link, a file for a FIFO, or a file for a link to a file outside the roots, at any moment. A check by path
followed by an open by path is therefore never used. Instead:

- a box path is parsed lexically first: it must be one of the role's root prefixes or lie under one, and it holds
  no ``..``, no NUL and at most :data:`PATH_MAX_BYTES` bytes (empty and ``.`` components are dropped);
- each root is opened once as a directory descriptor; every directory on the way is opened from the one before it
  with ``openat`` and ``O_NOFOLLOW | O_DIRECTORY``; the leaf is opened with ``O_NOFOLLOW | O_NONBLOCK`` (a FIFO
  planted by the box cannot block the host) and its descriptor is checked with ``fstat``: only a regular file or
  a directory is used, anything else is refused;
- directories are listed with ``os.scandir`` on the opened descriptor;
- a write goes to a new ``O_EXCL`` file in the leaf's own directory and is renamed over the leaf within that
  directory, so a link at the leaf is replaced, never written through, and a hard link keeps its other name's bytes.

A link anywhere on the way is refused, whether it points inside the roots or not. Python has no ``openat2``; this
walk gives what ``RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS`` would.

Roots are given per call (:class:`Roots`, from root descriptors or from root paths opened once), never held as
module state.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
import time
from collections.abc import Callable, Iterator, Mapping

#: The longest box path parsed, in UTF-8 bytes (the kernel's own path limit).
PATH_MAX_BYTES = 4096
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: a root's own host path is chosen by the host (its parents are never box-writable), so it may be reached by a link
ROOT_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
FILE_MODE = 0o644
DIR_MODE = 0o755


class PathError(OSError):
    """A box path that is not opened; ``where`` is the box path of the component that stopped the walk."""

    kind = "unreadable"

    def __init__(self, reason: str, where: str = "") -> None:
        super().__init__(reason)
        self.reason, self.where = reason, where

    def __str__(self) -> str:
        return self.reason


class Outside(PathError, PermissionError):
    """Not under one of the roots (lexically: another prefix, ``..``, NUL, too long)."""

    kind = "outside"


class LinkFound(PathError):
    """A link on the way or at the leaf: never followed."""

    kind = "link"


class Missing(PathError, FileNotFoundError):
    """The path, or a directory on the way to it, does not exist (or is a file where a directory is needed)."""

    kind = "not_found"


class NotRegular(PathError):
    """The leaf is neither a regular file nor a directory (a FIFO, a socket, a device)."""

    kind = "not_regular"


class TooLarge(PathError):
    """A file larger than the reader's limit."""

    kind = "too_large"


def openat(name: str, flags: int, dir_fd: int, mode: int = 0o777) -> int:
    """``openat(dir_fd, name, flags)``."""
    return os.open(name, flags, mode, dir_fd=dir_fd)


def errno_name(exc: OSError) -> str:
    """The error's symbolic name (``ELOOP``), a refused walk's reason, or the error's type: never an error's own
    text, which can name host paths."""
    if isinstance(exc, PathError):
        return exc.reason
    return errno.errorcode.get(exc.errno or 0, type(exc).__name__)


def is_link(dir_fd: int, name: str) -> bool:
    """Whether the entry *name* in *dir_fd* is a link (read after a refused open, only to name the refusal)."""
    try:
        return stat.S_ISLNK(os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode)
    except OSError:
        return False


def lstat_at(dir_fd: int, name: str) -> os.stat_result | None:
    """The entry's own ``stat`` (a link is not followed), or None when it does not exist."""
    try:
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


# --- lexical parsing -----------------------------------------------------------------------------------------


def _check_text(path: object) -> str:
    if not isinstance(path, str):
        raise Outside("the path is not text")
    if "\x00" in path:
        raise Outside("a path holds no NUL")
    try:
        size = len(path.encode("utf-8"))
    except UnicodeEncodeError:
        raise Outside("the path is not UTF-8") from None
    if size > PATH_MAX_BYTES:
        raise Outside(f"the path is longer than {PATH_MAX_BYTES} bytes")
    return path


def rel_parts(rel: object) -> list[str]:
    """The components of a path relative to one root: no ``..``, no NUL, bounded; empty and ``.`` dropped.
    An absolute path is refused (:class:`Outside`)."""
    text = _check_text(rel)
    if text.startswith("/"):
        raise Outside("the path is absolute")
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise Outside("a path component is '..'")
    return parts


def split(vpath: object, prefixes: Iterator[str] | list[str] | tuple[str, ...]) -> tuple[str, list[str]]:
    """``(prefix, components)`` of a box path under one of *prefixes* (the first that matches), else :class:`Outside`."""
    text = _check_text(vpath)
    for prefix in prefixes:
        if text == prefix or text.startswith(prefix + "/"):
            parts = [p for p in text[len(prefix) :].split("/") if p not in ("", ".")]
            if ".." in parts:
                raise Outside("a path component is '..'")
            return prefix, parts
    raise Outside("outside the readable roots")


def virtual(prefix: str, parts: list[str]) -> str:
    """The box path of *parts* under *prefix*, normalised (``/memory/a/b``)."""
    return "/".join([prefix, *parts]) if parts else prefix


# --- roots ---------------------------------------------------------------------------------------------------


class Roots:
    """Root directory descriptors by box prefix (``"/memory"``). Built from descriptors (borrowed: the caller
    closes them) or from host paths, each opened once on first use and closed by :meth:`close`.
    """

    def __init__(self, roots: Mapping[str, int | str | os.PathLike]) -> None:
        self._given = dict(roots)
        self._fds: dict[str, int] = {}
        self._owned: list[int] = []

    @classmethod
    def of(cls, roots: "Roots | Mapping[str, int | str | os.PathLike]") -> "Roots":
        return roots if isinstance(roots, Roots) else cls(roots)

    @property
    def prefixes(self) -> list[str]:
        return list(self._given)

    def fd(self, prefix: str) -> int:
        """*prefix*'s root descriptor (opened now if it was given as a path)."""
        if prefix not in self._fds:
            root = self._given[prefix]
            if isinstance(root, int):
                self._fds[prefix] = root
            else:
                try:
                    fd = os.open(root, ROOT_FLAGS)
                except FileNotFoundError:
                    raise Missing("the root does not exist", prefix) from None
                self._owned.append(fd)
                self._fds[prefix] = fd
        return self._fds[prefix]

    def split(self, vpath: object) -> tuple[str, list[str]]:
        return split(vpath, self.prefixes)

    def close(self) -> None:
        for fd in self._owned:
            os.close(fd)
        self._owned.clear()
        self._fds.clear()

    def __enter__(self) -> "Roots":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Borrowed:
    """A context that closes *roots* only if it opened them."""

    def __init__(self, roots: "Roots | Mapping[str, int | str | os.PathLike]") -> None:
        self.mine = not isinstance(roots, Roots)
        self.roots = Roots.of(roots)

    def __enter__(self) -> Roots:
        return self.roots

    def __exit__(self, *exc: object) -> None:
        if self.mine:
            self.roots.close()


def using(roots: "Roots | Mapping[str, int | str | os.PathLike]") -> _Borrowed:
    """``with using(roots) as r``: *roots* as a :class:`Roots`, closed at the end only when opened here."""
    return _Borrowed(roots)


# --- walking -------------------------------------------------------------------------------------------------


def walk_dirs(
    root_fd: int,
    parts: list[str],
    *,
    create: bool = False,
    shown: str = "",
    opener: Callable[[str, int, int], int] | None = None,
) -> int:
    """A new descriptor of the directory ``root/parts...``, reached component by component, never through a link.
    *create*: missing directories are made (mode 0755). The caller closes the descriptor."""
    op = opener or (lambda name, flags, dfd: openat(name, flags, dfd))
    fd = os.dup(root_fd)
    try:
        for i, name in enumerate(parts):
            where = "/".join([shown, *parts[: i + 1]]) if shown else "/".join(parts[: i + 1])
            if create:
                try:
                    os.mkdir(name, DIR_MODE, dir_fd=fd)
                except FileExistsError:
                    pass
            try:
                nxt = op(name, DIR_FLAGS, fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR) and is_link(fd, name):
                    raise LinkFound("a link on the way is not followed", where) from None
                if exc.errno == errno.ELOOP:
                    raise LinkFound("a link on the way is not followed", where) from None
                if exc.errno in (errno.ENOENT, errno.ENOTDIR):
                    raise Missing("does not exist", where) from None
                raise
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def open_leaf(dir_fd: int, name: str, where: str = "") -> int:
    """A descriptor of the entry *name* in *dir_fd*: a regular file or a directory, opened without following a
    link and without blocking (``O_NONBLOCK``), checked with ``fstat``. The caller closes it."""
    st = lstat_at(dir_fd, name)
    if st is None:
        raise Missing("does not exist", where)
    if stat.S_ISLNK(st.st_mode):
        raise LinkFound("a link is not followed", where)
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        raise NotRegular("not a regular file or directory", where)
    try:
        fd = openat(name, READ_FLAGS, dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise LinkFound("a link is not followed", where) from None
        if exc.errno == errno.ENOENT:
            raise Missing("does not exist", where) from None
        if exc.errno in (errno.ENXIO, errno.ENODEV):
            raise NotRegular("not a regular file or directory", where) from None
        raise
    mode = os.fstat(fd).st_mode
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        os.close(fd)
        raise NotRegular("not a regular file or directory", where)
    return fd


def open_path(roots: Roots, vpath: object) -> tuple[int, str, list[str]]:
    """``(fd, prefix, parts)`` for a box path: a regular file or a directory, the caller closes *fd*."""
    prefix, parts = roots.split(vpath)
    root = roots.fd(prefix)
    if not parts:
        return os.dup(root), prefix, parts
    dfd = walk_dirs(root, parts[:-1], shown=prefix)
    try:
        return open_leaf(dfd, parts[-1], virtual(prefix, parts)), prefix, parts
    finally:
        os.close(dfd)


def is_dir_fd(fd: int) -> bool:
    return stat.S_ISDIR(os.fstat(fd).st_mode)


def listing(dir_fd: int) -> list[str]:
    """The directory's entry names, sorted, a directory's with a trailing ``/`` (a link is never followed)."""
    with os.scandir(dir_fd) as it:
        return sorted(e.name + ("/" if e.is_dir(follow_symlinks=False) else "") for e in it)


def read_fd(fd: int, limit: int | None = None) -> bytes:
    """All of an open regular file's bytes (:class:`TooLarge` past *limit*)."""
    chunks, total = [], 0
    while chunk := os.read(fd, 1 << 20):
        chunks.append(chunk)
        total += len(chunk)
        if limit is not None and total > limit:
            raise TooLarge(f"larger than {limit} bytes")
    return b"".join(chunks)


def read_at(dir_fd: int, name: str, limit: int | None = None, where: str = "") -> bytes:
    """The bytes of the regular file *name* in *dir_fd* (a directory is :class:`NotRegular`)."""
    fd = open_leaf(dir_fd, name, where)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise NotRegular("not a regular file", where)
        return read_fd(fd, limit)
    finally:
        os.close(fd)


def read_rel(root_fd: int, parts: list[str], limit: int | None = None) -> bytes:
    """The bytes of the regular file ``root/parts...``."""
    shown = "/".join(parts)
    dfd = walk_dirs(root_fd, parts[:-1])
    try:
        return read_at(dfd, parts[-1], limit, shown)
    finally:
        os.close(dfd)


def check_leaf(dir_fd: int, name: str, where: str = "") -> int | None:
    """The leaf's mode when it is a regular file, None when it does not exist; a link or another kind is refused."""
    st = lstat_at(dir_fd, name)
    if st is None:
        return None
    if stat.S_ISLNK(st.st_mode):
        raise LinkFound("a link is not written through", where)
    if not stat.S_ISREG(st.st_mode):
        raise NotRegular("not a regular file", where)
    return st.st_mode


def replace_at(
    dir_fd: int,
    name: str,
    data: bytes,
    mode: int = FILE_MODE,
    opener: Callable[[str, int, int, int], int] | None = None,
) -> None:
    """*data* into a new ``O_EXCL`` file beside *name* (mode *mode*), then renamed over *name* within *dir_fd*.
    An ``OSError`` leaves *name* as it was."""
    op = opener or openat
    tmp = f".edit-{secrets.token_hex(8)}.tmp"
    fd = op(tmp, TEMP_FLAGS, dir_fd, mode)
    try:
        try:
            os.fchmod(fd, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view) :]
        finally:
            os.close(fd)
        os.rename(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except OSError:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except OSError:
            pass
        raise


def write_rel(root_fd: int, parts: list[str], data: bytes, mode: int = FILE_MODE) -> bool:
    """Write the file ``root/parts...`` (missing directories made), never through a link: an existing link or
    non-regular leaf is refused, and the bytes replace the leaf by a rename. True when the file existed."""
    shown = "/".join(parts)
    dfd = walk_dirs(root_fd, parts[:-1], create=True)
    try:
        existed = check_leaf(dfd, parts[-1], shown)
        replace_at(dfd, parts[-1], data, mode)
        return existed is not None
    finally:
        os.close(dfd)


def write_in(root: str | os.PathLike, rel: str, data: bytes, mode: int = FILE_MODE) -> bool:
    """:func:`write_rel` under the host directory *root* (opened here), *rel* parsed by :func:`rel_parts`."""
    parts = rel_parts(rel)
    if not parts:
        raise Outside("the path names the root itself")
    fd = os.open(root, DIR_FLAGS)
    try:
        return write_rel(fd, parts, data, mode)
    finally:
        os.close(fd)


def read_in(root: str | os.PathLike, rel: str, limit: int | None = None) -> bytes:
    """:func:`read_rel` under the host directory *root* (opened here), *rel* parsed by :func:`rel_parts`."""
    parts = rel_parts(rel)
    if not parts:
        raise NotRegular("the path names the root itself")
    fd = os.open(root, DIR_FLAGS)
    try:
        return read_rel(fd, parts, limit)
    finally:
        os.close(fd)


def link_in(root: str | os.PathLike, rel: str) -> bool:
    """Whether a link (or an unopenable entry) stands on the way to *rel* under *root*, or at *rel* itself; checked
    from *root*'s descriptor. A missing part is no link."""
    try:
        parts = rel_parts(rel)
        fd = os.open(root, DIR_FLAGS)
    except OSError:
        return True
    try:
        try:
            dfd = walk_dirs(fd, parts[:-1])
        except Missing:
            return False
        except OSError:
            return True
        try:
            st = lstat_at(dfd, parts[-1]) if parts else None
        finally:
            os.close(dfd)
        return st is not None and stat.S_ISLNK(st.st_mode)
    finally:
        os.close(fd)


def iter_files(
    roots: Roots,
    vpath: object,
    *,
    want: Callable[[str], bool] | None = None,
    deadline: float | None = None,
) -> Iterator[tuple[str, bytes]]:
    """``(box path, bytes)`` of every regular file at or under *vpath*, in sorted path order (each directory's
    names sorted, a directory's files where its name sorts). Links, FIFOs, sockets and devices are skipped, never
    opened through; an entry that cannot be opened or read is skipped. *want*: a box path filter applied before a
    file is read. *deadline* (``time.monotonic()``): :class:`TimeoutError` once it passes.

    A missing *vpath* yields nothing; a link at or on the way to *vpath* is :class:`LinkFound`.
    """
    try:
        top, prefix, parts = open_path(roots, vpath)
    except (Missing, NotRegular):
        return
    base = virtual(prefix, parts)
    if not is_dir_fd(top):
        try:
            data = read_fd(top) if want is None or want(base) else None
        except OSError:
            data = None
        finally:
            os.close(top)
        if data is not None:
            yield base, data
        return
    # one open descriptor per directory level, closed as the walk leaves it
    stack: list[tuple[int, str, list[str]]] = []
    try:
        try:
            with os.scandir(top) as it:
                names = sorted(e.name for e in it)
        except OSError:
            names = []
        stack.append((top, base, names[::-1]))
        while stack:
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("the walk ran out of time")
            dfd, here, todo = stack[-1]
            if not todo:
                os.close(dfd)
                stack.pop()
                continue
            name = todo.pop()
            vp = f"{here}/{name}"
            st = lstat_at(dfd, name)
            if st is None or stat.S_ISLNK(st.st_mode):
                continue
            if stat.S_ISDIR(st.st_mode):
                try:
                    cfd = openat(name, DIR_FLAGS, dfd)
                except OSError:
                    continue
                try:
                    with os.scandir(cfd) as it:
                        sub = sorted(e.name for e in it)
                except OSError:
                    sub = []
                stack.append((cfd, vp, sub[::-1]))
                continue
            if not stat.S_ISREG(st.st_mode) or (want is not None and not want(vp)):
                continue
            try:
                fd = open_leaf(dfd, name, vp)
            except OSError:
                continue
            try:
                data = read_fd(fd) if stat.S_ISREG(os.fstat(fd).st_mode) else None
            except OSError:
                data = None
            finally:
                os.close(fd)
            if data is not None:
                yield vp, data
    finally:
        for dfd, _, _ in stack:
            os.close(dfd)

