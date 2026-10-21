"""Bubblewrap-confined subprocesses for model-written code and its tests.

The box sees an allowlist of the host, nothing else: ``/usr`` and the ``/bin``, ``/lib*``, ``/sbin`` links
or directories, a few files of ``/etc`` (the loader's cache, ``passwd``, ``group``, the time zone, CA
certificates), its own ``/dev``, ``/proc`` and ``/tmp``, an empty ``/home``, the interpreter's venv and base
install, and the paths given in ``ro``/``rw``. The root is read-only. ``--unshare-all`` gives it no network
and private IPC, PID, UTS and user namespaces; a seccomp filter limits sockets to AF_UNIX, AF_INET and
AF_INET6 (WSL2's AF_VSOCK reaches the Windows host whatever the network namespace) and refuses new user
namespaces and io_uring. The environment is cleared; rlimits bound memory, file size, open files and
processes; on timeout the whole box is killed and verified dead. Every command is a fixed argv list; no
shell is involved.
"""

from __future__ import annotations

import contextlib
import os
import platform
import json
import posixpath
import pwd
import select
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

try:
    from unify.common import process_tree
except ImportError:  # run as a script outside the package: the stdlib-only module by path
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location(
        "process_tree",
        Path(__file__).resolve().parents[1] / "common" / "process_tree.py",
    )
    process_tree = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(process_tree)

if TYPE_CHECKING:
    from .example_calls import CallRecord


def _account_home() -> Path:
    """The account's home from the password database; ``$HOME`` may point elsewhere (tests move it)."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


# The box's interpreter: ``MEMORY_PYTHON``, else the controller's own (a fixed host path would not exist elsewhere).
PYTHON = Path(os.environ.get("MEMORY_PYTHON") or sys.executable)

# The host paths the box sees (read-only), besides the interpreter roots and the caller's binds.
SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32")
ETC_ALLOW = (
    "ld.so.cache",
    "ld.so.conf",
    "ld.so.conf.d",
    "ssl/certs",
    "localtime",
    "passwd",
    "group",
    "nsswitch.conf",
    "alternatives",
)

#: Never in a box's environment, whatever a caller passes: memory v2's route for the writer's own model calls.
HARNESS_ONLY_ENV = frozenset(
    {
        "UNIFY_MEMORY_WRITER_BASE_URL",
        "UNIFY_MEMORY_WRITER_TOKEN",
        "UNIFY_MEMORY_WRITER_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_FORK_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_FORK_TOKEN_FD",
        "UNIFY_MEMORY_LIBRARY_HELPER_BASE_URL",
        "UNIFY_MEMORY_LIBRARY_HELPER_TOKEN_FD",
    },
)

# Resource bounds of every process in the box (the laptop VM has 8 GB).
RLIMIT_AS_BYTES = 4 * 1024**3
RLIMIT_NPROC_COUNT = 256
RLIMIT_FSIZE_BYTES = 512 * 1024**2
RLIMIT_NOFILE_COUNT = 1024
PRLIMIT = "/usr/bin/prlimit"

JUNIT_MAX_BYTES = 8 * 1024**2
# Each of stdout and stderr keeps at most its last CAPTURE_MAX_BYTES; the rest is read and dropped.
CAPTURE_MAX_BYTES = 1024**2
# library mode: a spooled stream keeps at most SPOOL_MAX_BYTES on the harness's disk
# (each of stdout and stderr of one run); past it the spool stops and ends with a marker naming the bytes not kept.
SPOOL_MAX_BYTES = 64 * 1024**2

_KILL_GRACE_S = 10.0


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _python_roots() -> list[Path]:
    """The interpreter's venv and base install, by every name the interpreter is reached through.

    The venv's ``bin/python`` links to a uv base install by its minor-version name
    (``cpython-3.12-...``), itself a link to the patch release (``cpython-3.12.11-...``), and the venv's
    ``pyvenv.cfg`` names the former. Each name is mounted (from its real target), so the chain resolves
    inside the box although the home directory above it is a tmpfs.
    """
    venv = Path(os.path.abspath(PYTHON)).parent.parent
    roots = [venv]
    link = Path(os.path.abspath(PYTHON))
    for _ in range(40):
        if not link.is_symlink():
            break
        link = Path(os.path.normpath(link.parent / os.readlink(link)))
        roots.append(link.parent.parent)
    roots.append(Path(os.path.realpath(PYTHON)).parent.parent)
    cfg = venv / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and value.strip():
                roots.append(Path(os.path.abspath(value.strip())).parent)
    out: list[Path] = []
    for root in roots:
        for p in (root, Path(os.path.realpath(root))):
            if p not in out:
                out.append(p)
    return out


def _homes() -> list[Path]:
    """``/home``, the account's home and ``$HOME`` (real paths, outside /tmp): never exposed whole."""
    out: list[Path] = []
    for raw in (Path("/home"), _account_home(), Path.home()):
        real = Path(os.path.realpath(raw))
        if not _within(real, Path("/tmp")) and real not in out:
            out.append(real)
    return out


def _check_bind_source(src: Path, homes: list[Path]) -> None:
    real = Path(os.path.realpath(src))
    for home in homes:
        if _within(home, real):
            raise ValueError(
                f"refusing to expose {src}: it contains the home directory {home}",
            )


def _root_args() -> list[str]:
    """The allowlisted host paths, read-only, on the box's own root, with an empty ``/home``."""
    args: list[str] = []
    for path in SYSTEM_DIRS:
        if os.path.islink(path):
            args += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            args += ["--ro-bind", path, path]
    for name in ETC_ALLOW:
        path = "/etc/" + name
        if os.path.islink(path):
            args += ["--symlink", os.readlink(path), path]
        elif os.path.exists(path):
            args += ["--ro-bind", path, path]
    args += [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
    ]
    return args


# --- seccomp -------------------------------------------------------------------------------------------

_AUDIT_ARCH_X86_64 = 0xC000003E
_X32_SYSCALL_BIT = 0x40000000
_NR = {
    "socket": 41,
    "socketpair": 53,
    "clone": 56,
    "unshare": 272,
    "io_uring_setup": 425,
    "io_uring_enter": 426,
    "io_uring_register": 427,
    "clone3": 435,
}
_ALLOWED_FAMILIES = (1, 2, 10)  # AF_UNIX, AF_INET, AF_INET6
_CLONE_NEWUSER = 0x10000000
_EPERM, _ENOSYS, _EAFNOSUPPORT = 1, 38, 97
_RET_ALLOW = 0x7FFF0000
_RET_KILL_PROCESS = 0x80000000


def _ret_errno(errno: int) -> int:
    return 0x00050000 | errno


def seccomp_program() -> bytes:
    """A classic-BPF seccomp filter for x86_64, assembled by hand (``struct sock_filter`` array).

    * another audit arch (i386 via ``int 0x80``) kills the process; x32 syscall numbers get ENOSYS;
    * ``socket``/``socketpair`` with a family other than AF_UNIX, AF_INET or AF_INET6 get EAFNOSUPPORT;
    * ``clone``/``unshare`` with CLONE_NEWUSER get EPERM; ``clone3`` (flags in memory, which a filter
      cannot read) gets ENOSYS, so the C library falls back to ``clone``;
    * io_uring, which can open sockets without the ``socket`` syscall, gets ENOSYS.
    """
    ld_w_abs, jeq, jge, jset, ret = 0x20, 0x15, 0x35, 0x45, 0x06
    nr_off, arch_off, arg0_off = (
        0,
        4,
        16,
    )  # struct seccomp_data; arg0's low word (little endian)
    prog: list[tuple] = [
        ("ld", arch_off),
        ("jeq", _AUDIT_ARCH_X86_64, "nr", "kill"),
        ("label", "nr"),
        ("ld", nr_off),
        ("jge", _X32_SYSCALL_BIT, "enosys", None),
        ("jeq", _NR["socket"], "family", None),
        ("jeq", _NR["socketpair"], "family", None),
        ("jeq", _NR["clone"], "newuser", None),
        ("jeq", _NR["unshare"], "newuser", None),
        ("jeq", _NR["clone3"], "enosys", None),
        ("jeq", _NR["io_uring_setup"], "enosys", None),
        ("jeq", _NR["io_uring_enter"], "enosys", None),
        ("jeq", _NR["io_uring_register"], "enosys", None),
        ("ret", _RET_ALLOW),
        ("label", "family"),
        ("ld", arg0_off),
        *[("jeq", fam, "allow", None) for fam in _ALLOWED_FAMILIES],
        ("ret", _ret_errno(_EAFNOSUPPORT)),
        ("label", "newuser"),
        ("ld", arg0_off),
        ("jset", _CLONE_NEWUSER, "eperm", "allow"),
        ("label", "allow"),
        ("ret", _RET_ALLOW),
        ("label", "eperm"),
        ("ret", _ret_errno(_EPERM)),
        ("label", "enosys"),
        ("ret", _ret_errno(_ENOSYS)),
        ("label", "kill"),
        ("ret", _RET_KILL_PROCESS),
    ]
    labels: dict[str, int] = {}
    count = 0
    for ins in prog:
        if ins[0] == "label":
            labels[ins[1]] = count
        else:
            count += 1
    codes = {"jeq": jeq, "jge": jge, "jset": jset}
    out = b""
    index = 0
    for ins in prog:
        op = ins[0]
        if op == "label":
            continue
        if op == "ld":
            out += struct.pack("=HBBI", ld_w_abs, 0, 0, ins[1])
        elif op == "ret":
            out += struct.pack("=HBBI", ret, 0, 0, ins[1])
        else:
            _, k, jt, jf = ins
            offsets = [0 if t is None else labels[t] - (index + 1) for t in (jt, jf)]
            if not all(0 <= o < 256 for o in offsets):
                raise ValueError("seccomp jump out of range")
            out += struct.pack("=HBBI", codes[op], offsets[0], offsets[1], k)
        index += 1
    return out


# --- termination ---------------------------------------------------------------------------------------


# Shared with the actor's sandbox (unify/common/process_tree.py).
_proc_stat = process_tree.proc_stat
_descendants = process_tree.descendants
_alive = process_tree.alive


class _Tail:
    """Drain one pipe of the box on a thread, keeping only its last ``cap`` bytes (bounded memory).

    The box can always reach the host's pipes (its pid 1 holds them, and ``/proc/1/fd/1`` is open to the
    same user), so the host must never buffer them whole or decode them strictly: the text is decoded as
    UTF-8 with replacement characters, prefixed by a marker when bytes were dropped. With *spool* (library
    mode) every byte is also written to that harness-side file, up to *spool_max* (:data:`SPOOL_MAX_BYTES`),
    after which it ends with ``[spool limit reached: N further bytes not kept]``; memory stays bounded.
    """

    def __init__(
        self,
        stream,
        cap: int = CAPTURE_MAX_BYTES,
        spool: Path | None = None,
        spool_max: int | None = None,
    ) -> None:
        self._stream, self._cap = stream, cap
        self._buf = bytearray()
        self._dropped = 0
        self._spool = (
            open(spool, "wb") if spool is not None else None
        )  # noqa: SIM115 - closed by the drain
        self._spool_room = SPOOL_MAX_BYTES if spool_max is None else spool_max
        self._unspooled = 0
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(1 << 16)
                if not chunk:
                    break
                if self._spool is not None:
                    kept = chunk[: self._spool_room]
                    if kept:
                        self._spool.write(kept)
                        self._spool_room -= len(kept)
                    self._unspooled += len(chunk) - len(kept)
                self._buf += chunk
                if len(self._buf) > 2 * self._cap:
                    cut = len(self._buf) - self._cap
                    self._dropped += cut
                    del self._buf[:cut]
        except (OSError, ValueError):
            pass
        finally:
            try:
                self._stream.close()
            except OSError:
                pass
            if self._spool is not None:
                try:
                    if self._unspooled:
                        self._spool.write(
                            f"\n[spool limit reached: {self._unspooled} further bytes not kept]\n".encode(),
                        )
                    self._spool.close()
                except OSError:
                    pass

    def text(self, timeout: float) -> str:
        self._thread.join(timeout)
        data = bytes(self._buf)
        dropped = self._dropped
        if len(data) > self._cap:
            dropped += len(data) - self._cap
            data = data[-self._cap :]
        text = data.decode("utf-8", errors="replace")
        return f"[{dropped} earlier bytes dropped]\n{text}" if dropped else text


def _kill_box(proc: subprocess.Popen) -> None:
    """SIGKILL bubblewrap's process group, every process below it and every process in the box's pid namespace
    (held open since it started, :func:`_spawn_box`); raise unless all are gone. The namespace is released after."""
    try:
        process_tree.kill_tree(
            proc.pid,
            exited=lambda: proc.poll() is not None,
            grace_s=_KILL_GRACE_S,
            pid_ns=getattr(proc, "box_pid_ns", None),
            pid_ns_fd=getattr(proc, "box_pid_ns_fd", None),
        )
    finally:
        _release_namespace(proc)


def _release_namespace(proc: subprocess.Popen) -> None:
    """Close the fd that holds the box's pid namespace (:func:`_spawn_box`). Safe to repeat."""
    fd = getattr(proc, "box_pid_ns_fd", None)
    proc.box_pid_ns_fd = None
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _check_box_request(env: dict[str, str] | None, pass_fds: tuple[int, ...]) -> str:
    """Refuse what never enters a box (a harness-only variable, a descriptor that is not a pipe) and check the
    host has what every box needs; returns bubblewrap's path."""
    for name in env or {}:
        if name.upper() in HARNESS_ONLY_ENV:
            raise ValueError(f"{name} never enters the box")
    for fd in pass_fds:
        if (
            not isinstance(fd, int)
            or isinstance(fd, bool)
            or not stat.S_ISFIFO(os.fstat(fd).st_mode)
        ):
            raise ValueError("only pipe descriptors are passed into the box")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required")
    if platform.machine() != "x86_64":
        raise RuntimeError("the seccomp filter is built for x86_64 only")
    if not os.access(PRLIMIT, os.X_OK):
        raise RuntimeError(f"{PRLIMIT} is required to bound the box's processes")
    return bwrap


def _box_args(
    bwrap: str,
    argv: list[str],
    *,
    ro: dict[Path, str] | None,
    rw: dict[Path, str] | None,
    late_ro: dict[Path, str] | None,
    cwd: str,
    env_fd: int,
    bpf_fd: int,
) -> list[str]:
    """The whole command line of one box: bubblewrap's binds and options, the rlimits, then *argv*."""
    homes = _homes()
    args = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        *_root_args(),
    ]
    for root in _python_roots():
        real = Path(os.path.realpath(root))
        if root.exists() and not any(_within(home, real) for home in homes):
            args += ["--ro-bind", str(real), str(root)]
    for src, dst in (ro or {}).items():
        _check_bind_source(src, homes)
        args += ["--ro-bind", str(src), dst]
    for src, dst in (rw or {}).items():
        _check_bind_source(src, homes)
        args += ["--bind", str(src), dst]
    for src, dst in (late_ro or {}).items():
        _check_bind_source(src, homes)
        if Path(src).is_symlink() or not Path(src).exists():
            raise ValueError(
                f"refusing a late read-only bind of {src}: missing or a link",
            )
        args += ["--ro-bind", str(src), dst]
    args += [
        "--remount-ro",
        "/",
        "--chdir",
        cwd,
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
    ]
    args += ["--args", str(env_fd), "--seccomp", str(bpf_fd), "--"]
    # Set inside the box, not by a preexec hook (unsafe in threaded parents). There RLIMIT_NPROC counts
    # the box's own user namespace; outside it would count all of this user's processes, and
    # bubblewrap could not create its namespaces on a busy machine.
    args += [
        PRLIMIT,
        f"--as={RLIMIT_AS_BYTES}:{RLIMIT_AS_BYTES}",
        f"--fsize={RLIMIT_FSIZE_BYTES}:{RLIMIT_FSIZE_BYTES}",
        f"--nofile={RLIMIT_NOFILE_COUNT}:{RLIMIT_NOFILE_COUNT}",
        f"--nproc={RLIMIT_NPROC_COUNT}:{RLIMIT_NPROC_COUNT}",
        "--",
    ]
    return args + list(argv)


def _spawn_box(
    argv: list[str],
    *,
    ro: dict[Path, str] | None,
    rw: dict[Path, str] | None,
    cwd: str,
    env: dict[str, str] | None,
    late_ro: dict[Path, str] | None,
    pass_fds: tuple[int, ...],
    stdin: int,
) -> subprocess.Popen:
    """Start one box (the checks and command line every box shares) and return bubblewrap's process.

    The descriptors the box inherits are exactly the environment and seccomp files bubblewrap reads and closes,
    the standard streams and *pass_fds* (``close_fds`` stays on); no hook runs in the child before it executes.
    """
    bwrap = _check_box_request(env, pass_fds)
    # Unlinked temporary files: the entries never appear in a command line or a named file.
    with tempfile.TemporaryFile() as env_file, tempfile.TemporaryFile() as bpf_file:
        for k, v in (env or {}).items():
            for part in ("--setenv", k, v):
                if "\0" in part:
                    raise ValueError("environment entries must not contain NUL")
                env_file.write(part.encode() + b"\0")
        env_file.flush()
        env_file.seek(0)
        bpf_file.write(seccomp_program())
        bpf_file.flush()
        bpf_file.seek(0)
        env_fd, bpf_fd = env_file.fileno(), bpf_file.fileno()
        args = _box_args(
            bwrap,
            argv,
            ro=ro,
            rw=rw,
            late_ro=late_ro,
            cwd=cwd,
            env_fd=env_fd,
            bpf_fd=bpf_fd,
        )
        proc = subprocess.Popen(
            args,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
            pass_fds=(env_fd, bpf_fd, *pass_fds),
        )
        # the box's pid namespace, pinned (held open) as it starts: the verified kill finds every process in it,
        # whatever its parent, group or session, and a later box can never take its number meanwhile
        # (process_tree.kill_tree); released by the kill or when the box ends
        pinned = process_tree.pin_box_namespace(proc.pid)
        proc.box_pid_ns, proc.box_pid_ns_fd = pinned if pinned is not None else (None, None)
        return proc


def run_confined(
    argv: list[str],
    *,
    ro: dict[Path, str] | None = None,
    rw: dict[Path, str] | None = None,
    cwd: str = "/tmp",
    timeout_s: float = 120.0,
    env: dict[str, str] | None = None,
    spool: tuple[Path, Path] | None = None,
    late_ro: dict[Path, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    on_spawn: Callable[[], None] | None = None,
) -> SandboxResult:
    """Run *argv* in a bubblewrap box; ``ro``/``rw`` map host paths to their paths inside the box.

    *pass_fds* are host descriptors the box inherits under the same numbers (the write end of an open-log pipe,
    :class:`BoxOpenLog`); never a descriptor of a host file. *on_spawn* runs once the box has started, before the
    host waits on it (:meth:`BoxOpenLog.close_write`: the host drops its copy of the write end, so the pipe reaches
    end of file when the box exits).

    *late_ro* paths are bound read-only after the ``rw`` binds, so a path inside a writable bind stays
    read-only (library mode's generated files in the writer's box). Each source must be an
    existing path that is not a link.

    *env* is the box's whole environment beyond PATH, HOME and PYTHONDONTWRITEBYTECODE. It reaches
    bubblewrap through an unlinked file (``--args``), not its command line; it must never hold credentials.
    ``stdout`` and ``stderr`` are bounded tails (:data:`CAPTURE_MAX_BYTES` each), decoded with replacement.
    *spool* (library mode) also writes every byte of stdout and stderr to those two harness-side files (each at
    most :data:`SPOOL_MAX_BYTES`, marked past it); it changes nothing about the box.
    """
    proc = _spawn_box(
        argv,
        ro=ro,
        rw=rw,
        cwd=cwd,
        env=env,
        late_ro=late_ro,
        pass_fds=pass_fds,
        stdin=subprocess.DEVNULL,
    )
    if on_spawn is not None:
        try:
            on_spawn()
        except BaseException:
            _kill_box(proc)
            raise
    tails: list[_Tail] = []

    def result(returncode: int, timed_out: bool) -> SandboxResult:
        out, err = (t.text(_KILL_GRACE_S) for t in tails)
        return SandboxResult(returncode, out, err, timed_out)

    try:
        # inside the try: a failure to start a drain thread still kills the box
        tails.append(_Tail(proc.stdout, spool=spool[0] if spool is not None else None))
        tails.append(_Tail(proc.stderr, spool=spool[1] if spool is not None else None))
        proc.wait(timeout=timeout_s)
        # the box ended by itself: its namespace went with it, and its pin is released
        _release_namespace(proc)
        return result(proc.returncode, False)
    except subprocess.TimeoutExpired:
        _kill_box(proc)
        return result(-9, True)
    except BaseException as exc:
        try:
            _kill_box(proc)
        except BaseException as kill_error:
            raise kill_error from exc
        raise


#: Bytes of a long-lived box's stderr the host keeps (its last bytes), for a start failure's message.
BOX_STDERR_TAIL_BYTES = 8192


class ConfinedBox:
    """A box started by :func:`start_confined` and left running: bubblewrap's process, its protocol pipes and a
    bounded drain of its stderr. :meth:`kill` is the verified kill of :func:`run_confined`'s timeout."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc
        self._stderr = _Tail(proc.stderr, cap=BOX_STDERR_TAIL_BYTES)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def stdin(self):
        return self.proc.stdin

    @property
    def stdout(self):
        return self.proc.stdout

    def running(self) -> bool:
        """Bubblewrap itself has not exited (its processes may still be checked with :meth:`alive`)."""
        return self.proc.poll() is None

    def alive(self) -> bool:
        """Bubblewrap or any process below it is still alive."""
        if self.proc.poll() is None:
            return True
        return bool(_descendants(self.proc.pid))

    def stderr_tail(self, timeout: float = 0.0) -> str:
        return self._stderr.text(timeout)

    def kill(self) -> None:
        """SIGKILL the box's process group and every descendant, verified gone (raises otherwise); then close the
        host's ends of its pipes. Safe to repeat."""
        _kill_box(self.proc)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


def start_confined(
    argv: list[str],
    *,
    ro: dict[Path, str] | None = None,
    rw: dict[Path, str] | None = None,
    cwd: str = "/tmp",
    env: dict[str, str] | None = None,
    late_ro: dict[Path, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    on_spawn: Callable[[], None] | None = None,
) -> ConfinedBox:
    """Start *argv* in the same box as :func:`run_confined` (the same checks, binds, ``--unshare-all``,
    ``--die-with-parent``, ``--new-session``, ``--clearenv``, seccomp filter and rlimits; only pipe descriptors in
    *pass_fds*) without waiting for it: the caller talks to it over its stdin and stdout pipes and ends it with
    :meth:`ConfinedBox.kill`. *on_spawn* runs once the box has started (a failure kills the box)."""
    proc = _spawn_box(
        argv,
        ro=ro,
        rw=rw,
        cwd=cwd,
        env=env,
        late_ro=late_ro,
        pass_fds=pass_fds,
        stdin=subprocess.PIPE,
    )
    try:
        box = ConfinedBox(proc)
        if on_spawn is not None:
            on_spawn()
    except BaseException:
        _kill_box(proc)
        raise
    return box


# --- the box's open log ---------------------------------------------------------------------------------------

#: The observe-only audit hook, staged read-only as ``/inputs/sitecustomize.py`` (see :mod:`.box_audit_hook`).
OPEN_HOOK_SOURCE = Path(__file__).with_name("box_audit_hook.py")
#: The environment variable naming the open log's descriptor in the box (``fd:<n>``).
OPEN_LOG_ENV = "MEMORY_BOX_OPEN_LOG"
#: Bytes of open-log lines kept per call; later lines are still drained (so the hook keeps running) and counted
#: as dropped.
BOX_OPEN_LOG_BYTES = 1 << 20
_OPEN_AREAS = ("/inputs", "/memory")
_OPEN_MODES = frozenset({"r", "w", "a", "x"})
_OPEN_LINE_MAX = 4096  # the hook never writes a longer line (one atomic pipe write)
_OPEN_DRAIN_WAIT_S = 5.0


def stage_open_hook(inputs: Path) -> None:
    """Write the audit hook as ``<inputs>/sitecustomize.py`` (created new, never through a link, read-only)."""
    data = OPEN_HOOK_SOURCE.read_bytes()
    fd = os.open(
        Path(inputs) / "sitecustomize.py",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o444,
    )
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _open_row(line: bytes) -> dict | None:
    """The ``{event, path, mode}`` of one hook line, or None when the line is not one the hook writes. Any other
    field from the box (a pass id, a call number) is dropped: the host stamps those itself.
    """
    if len(line) > _OPEN_LINE_MAX:
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    if not isinstance(obj, dict) or obj.get("event") != "open":
        return None
    path, mode = obj.get("path"), obj.get("mode")
    if not isinstance(path, str) or mode not in _OPEN_MODES:
        return None
    if posixpath.normpath(path) != path or not any(
        path == a or path.startswith(a + "/") for a in _OPEN_AREAS
    ):
        return None
    return {"event": "open", "path": path, "mode": mode}


class BoxOpenLog:
    """The host end of one call's open log: a pipe whose write end the box inherits (``pass_fds``, set
    non-blocking, named by ``MEMORY_BOX_OPEN_LOG=fd:<n>``), drained by a host thread.

    Lines are kept up to *cap_bytes* (default :data:`BOX_OPEN_LOG_BYTES`); the rest are read and discarded, and
    counted in ``dropped`` with lines that are not hook lines. A full pipe makes the hook stop (a missing
    measurement); nothing here can stall the call: the box never blocks on the pipe, and the drain is bounded.
    Use it as a context manager around :func:`run_confined` and read :attr:`result` after it.
    """

    def __init__(self, cap_bytes: int | None = None) -> None:
        self.cap = BOX_OPEN_LOG_BYTES if cap_bytes is None else int(cap_bytes)
        self.rows: list[dict] = []
        self.dropped = 0
        self.seen = 0  # hook lines read, kept or not
        self.started = (
            False  # the hook's start line arrived: it was installed and could write
        )
        self.failed = False
        self.result: dict | None = None
        self._kept = 0
        self._buf = b""
        self._stop = threading.Event()
        self._r, self._w = os.pipe()
        try:
            os.set_blocking(self._w, False)
            self._thread = threading.Thread(
                target=self._drain,
                name="box-open-log",
                daemon=True,
            )
            self._thread.start()
        except BaseException:
            os.close(self._r)
            os.close(self._w)
            raise

    @property
    def env(self) -> dict[str, str]:
        return {OPEN_LOG_ENV: f"fd:{self._w}"}

    @property
    def pass_fds(self) -> tuple[int, ...]:
        return (self._w,) if self._w is not None else ()

    def close_write(self) -> None:
        """Close the host's copy of the write end (pass it as :func:`run_confined`'s *on_spawn*): from then on only
        the box holds it, so the drain reads end of file when the box exits, and the host never holds a writable
        end while the box runs. Idempotent."""
        if self._w is not None:
            os.close(self._w)
            self._w = None

    def __enter__(self) -> "BoxOpenLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()

    def _drain(self) -> None:
        try:
            while True:
                ready = select.select(
                    [self._r],
                    [],
                    [],
                    0 if self._stop.is_set() else 0.2,
                )[0]
                if not ready:
                    if self._stop.is_set():
                        return
                    continue
                chunk = os.read(self._r, 65536)
                if not chunk:
                    return
                self._take(chunk)
        except (
            Exception
        ):  # a broken reader is a missing measurement, never a failed call
            self.failed = True

    def _take(self, chunk: bytes) -> None:
        *lines, self._buf = (self._buf + chunk).split(b"\n")
        if len(self._buf) > _OPEN_LINE_MAX:  # no hook line is this long
            self._buf = b""
            self.dropped += 1
        for line in lines:
            if line.strip() == b'{"event": "start"}':
                self.started = True
                continue
            row = _open_row(line)
            if row is None:
                self.dropped += 1
                continue
            self.seen += 1
            if self._kept + len(line) + 1 > self.cap:
                self.dropped += 1
                continue
            self._kept += len(line) + 1
            self.rows.append(row)

    def finish(self) -> dict:
        """Close the host's write end, drain what the box wrote and return ``{"rows", "dropped", "dead",
        "reason"}``. ``dead`` is True only when the hook never reported its start (it was not installed, or the
        process ended before Python's start-up ran it) or the reader failed; a cell that opens nothing under /inputs
        or /memory is measured, with no rows. ``reason`` names what happened: ``ok``, ``no_start`` or
        ``reader_failed``; lines past the byte cap are counted in ``dropped``. Idempotent.
        """
        if self.result is not None:
            return self.result
        self.close_write()
        self._thread.join(_OPEN_DRAIN_WAIT_S)
        self._stop.set()
        self._thread.join(1.0)
        if self._thread.is_alive():
            self.failed = True  # the reader still holds the read end; it is left to the daemon thread
        else:
            os.close(self._r)
            if self._buf:
                self.dropped += 1
                self._buf = b""
        self.result = {
            "rows": list(self.rows),
            "dropped": self.dropped,
            "dead": self.failed or not self.started,
            "reason": (
                "reader_failed"
                if self.failed
                else ("ok" if self.started else "no_start")
            ),
        }
        return self.result


def box_open_fields(result: dict | None) -> dict:
    """The tool-event fields of an ``execute_code`` call from its :meth:`BoxOpenLog.finish` result (None: no box
    ran, so no measurement)."""
    if result is None:
        return {
            "box_opens": 0,
            "box_opens_dropped": 0,
            "box_hook_dead": None,
            "box_hook_reason": None,
        }
    return {
        "box_opens": len(result.get("rows") or []),
        "box_opens_dropped": int(result.get("dropped") or 0),
        "box_hook_dead": bool(result.get("dead")),
        "box_hook_reason": result.get("reason"),
    }


def append_open_rows(path: Path, rows: list[dict]) -> bool:
    """Append host-stamped *rows* to the ``box-opens.jsonl`` at *path*: a regular file with one link, created if
    absent, never opened through a link. Returns False (writing nothing) otherwise or on an OS error.
    """
    if not rows:
        return True
    try:
        folder = Path(path).parent
        folder.mkdir(parents=True, exist_ok=True)
        if folder.is_symlink() or not folder.is_dir():
            return False
        fd = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o644,
        )
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                return False
            data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode(
                "utf-8",
            )
            os.write(fd, data)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


# --- pytest --------------------------------------------------------------------------------------------


@dataclass
class PytestOutcome:
    """Test ids are ``<path relative to the tests dir>::[Class::]test``.

    Untrusted when model code runs in the tested process: that code shares pytest's process and can
    rewrite the junit report or the exit status. ``valid`` is False when the report is missing, a link or
    another non-regular file, too large, malformed, or contradicts pytest's exit status (0 needs no failure, 1
    at least one, 5 no test; any other status is invalid); a gate must then treat the run as failed. With
    ``import_skips_fail`` (:func:`run_pytest`) a skip caused by a failed import is a failure, 5 is also valid
    when every collected entry is a module skipped for an import (:data:`MODULE_SKIPPED`), and a report the
    marking plugin did not write is invalid.
    """

    passed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    skipped: set[str] = field(default_factory=set)
    returncode: int = 0
    timed_out: bool = False
    output: str = ""
    valid: bool = True
    # library mode: the run's output uncut by this module: stdout then stderr as run_confined kept
    # them (each its last CAPTURE_MAX_BYTES, prefixed "[N earlier bytes dropped]" when bytes were dropped, so
    # nothing is cut silently). ``output`` stays the 4,000-character tail every v2 reason quotes; empty means
    # the same text as ``output``.
    full_output: str = ""
    # library mode: the whole output's blob id (the gate sets it; :func:`.library_checks.keep_output`)
    output_blob: str | None = None
    # library mode: the calls of the observed functions this run recorded (:mod:`.example_calls`), when the run
    # asked for them; None otherwise
    calls: "CallRecord | None" = None

    def __post_init__(self) -> None:
        if not self.full_output:
            self.full_output = self.output


def _case_id(case: ET.Element, rootdir: str, tests_dir: str) -> str:
    name = case.get("name") or "?"
    classname = case.get("classname") or ""
    file = case.get("file")
    classes: list[str] = []
    if file:
        module = file[:-3].replace("/", ".") if file.endswith(".py") else file
        if classname.startswith(module + "."):
            classes = classname[len(module) + 1 :].split(".")
    else:
        parts = classname.split(".") if classname else []
        while len(parts) > 1 and parts[-1].startswith("Test"):
            classes.insert(0, parts.pop())
        file = "/".join(parts) + ".py"
    path = posixpath.normpath(posixpath.join(rootdir, file))
    base = posixpath.normpath(tests_dir)
    if path.startswith(base.rstrip("/") + "/"):
        file = posixpath.relpath(path, base)
    return "::".join([file, *classes, name])


def _read_junit(junit_dir: Path) -> ET.Element | None:
    """The root element of ``r.xml`` in the junit directory, or None when it cannot be trusted as a report.

    The box can replace ``r.xml`` with anything, so the host never follows it: it is opened with
    O_NOFOLLOW (a link is refused, not read) and O_NONBLOCK (a FIFO cannot stall the harness); the raw
    descriptor must be a regular file (not a directory, FIFO or device) of at most JUNIT_MAX_BYTES, is
    parsed from that descriptor and is closed on every path. The directory itself is
    the host's own temporary directory, bind-mounted into the box, so only its entries can change.
    """
    try:
        fd = os.open(
            junit_dir / "r.xml",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
    except OSError:
        return None
    try:
        st = os.fstat(
            fd,
        )  # the raw descriptor: a directory or FIFO is refused before any read
        if not stat.S_ISREG(st.st_mode) or st.st_size > JUNIT_MAX_BYTES:
            return None
        chunks: list[bytes] = []
        size = 0
        while size <= JUNIT_MAX_BYTES:
            chunk = os.read(fd, JUNIT_MAX_BYTES + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > JUNIT_MAX_BYTES:
            return None
        return ET.fromstring(b"".join(chunks))
    except (ET.ParseError, OSError, ValueError):
        return None
    finally:
        os.close(fd)


# --- skips caused by a failed import (opt-in) ----------------------------------------------------------------
#
# A test whose import is missing in one place it runs must not pass there silently; the gate counts such a skip
# as a failure when it asks to (``import_skips_fail``: a stage-5 switch is on or the library's tests use the
# test kit). The signal is structural, never pytest's message text: a plugin loaded into the box's pytest marks
# a skip whose exception came from ``pytest.importorskip`` or was raised while an ImportError was being
# handled (its ``__cause__``/``__context__`` chain), with the junit property SKIP_PROPERTY: value "test" on a
# test's entry, "collect" on the entry of a module or package skipped at collection. It also writes the
# suite property SKIPMARK_LOADED, and a report without it is invalid (the plugin did not run). The plugin
# reaches pytest's junit writer through its stash key (``_pytest.junitxml.xml_key``); a pytest without it is a
# usage error, so the run is invalid, never silently unmarked. Model code shares pytest's process and could
# forge the properties, as it could the report (model code is assumed careless, not malicious).
SKIP_PROPERTY = "memory_import_skip"
SKIPMARK_LOADED = "memory_skipmark"
SKIPMARK_MODULE = "_memory_skipmark"
SKIPMARK_DIR = "/memory-skipmark"  # the plugin's read-only directory in the box (on PYTHONPATH, last)
MODULE_SKIPPED = "test module skipped"
# The plugin, written into a fresh host directory per run; the host never imports it (importing it patches
# pytest.importorskip in that process).
SKIPMARK_SOURCE = r'''"""Mark skips caused by a failed import in the junit report (the memory gate's runs; see sandbox_run)."""
import unittest

import pytest
from _pytest import outcomes as _outcomes
from _pytest.junitxml import xml_key

PROPERTY = "memory_import_skip"
LOADED = "memory_skipmark"
_MARK = "_memory_import_skip"
_SKIPS = (_outcomes.Skipped, unittest.SkipTest)
_importorskip = _outcomes.importorskip


def importorskip(*args, **kwargs):
    __tracebackhide__ = True
    try:
        return _importorskip(*args, **kwargs)
    except _outcomes.Skipped as exc:
        setattr(exc, _MARK, True)  # the module is missing (or older than minversion)
        raise


pytest.importorskip = importorskip
_outcomes.importorskip = importorskip


def _from_import(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(exc, _MARK, False) or isinstance(exc, ImportError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _writer(config):
    xml = config.stash.get(xml_key, None)
    if xml is None or not hasattr(xml, "node_reporter") or not hasattr(xml, "add_global_property"):
        raise pytest.UsageError("_memory_skipmark needs pytest's junit writer (--junitxml)")
    return xml


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    _writer(config).add_global_property(LOADED, "1")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    exc = call.excinfo.value if call.excinfo is not None else None
    if (
        rep.skipped
        and not hasattr(rep, "wasxfail")
        and isinstance(exc, _SKIPS)
        and _from_import(exc)
        and (PROPERTY, "test") not in item.user_properties
    ):
        item.user_properties.append((PROPERTY, "test"))  # junit writes them with the teardown report
    return rep


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    rep = yield
    if rep.skipped:
        call = rep.__dict__.get("call")  # pytest keeps the collection's CallInfo here until it is reported
        exc = call.excinfo.value if call is not None and call.excinfo is not None else None
        if exc is None or _from_import(exc):  # unreachable exception: counted, never silently a skip
            _writer(collector.config).node_reporter(rep).add_property(PROPERTY, "collect")
    return rep
'''


def _skip_mark(case: ET.Element) -> str | None:
    """The plugin's mark on a junit entry: ``"test"``, ``"collect"`` or None."""
    props = case.find("properties")
    for p in props.iter("property") if props is not None else ():
        if p.get("name") == SKIP_PROPERTY and p.get("value") in ("test", "collect"):
            return p.get("value")
    return None


def _suite_property(root: ET.Element, name: str) -> bool:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        props = suite.find("properties")
        for p in props.iter("property") if props is not None else ():
            if p.get("name") == name:
                return True
    return False


def _skipmark_loaded(root: ET.Element) -> bool:
    return _suite_property(root, SKIPMARK_LOADED)


def run_pytest(
    tests_dir_in_box: str,
    *,
    python: Path,
    ro: dict[Path, str],
    rw: dict[Path, str],
    cwd: str,
    timeout_s: float = 300.0,
    env: dict[str, str] | None = None,
    import_skips_fail: bool = False,
    spool_dir: Path | None = None,
    calls: list[str] | None = None,
) -> PytestOutcome:
    """Run pytest on *tests_dir_in_box* in the box; see :class:`PytestOutcome` for ids and validity.

    *import_skips_fail* (default off: a skip is a skip): load the skip-marking plugin and count a skip caused
    by a failed import as a failure, ``<file>::test module skipped`` for a module skipped at collection and
    the test's id for a test. *spool_dir* (library mode) receives the run's whole ``stdout`` and ``stderr``
    (:func:`run_confined`'s *spool*); without it the call is v2's.

    *calls* (library mode): the function ids whose calls from test code the run records. The observe-only plugin
    (:data:`.example_calls.CALLS_SOURCE`) is mounted read-only and loaded like the skip-marking plugin, a fresh host
    directory is mounted read-write at ``/calls`` for its record, and :attr:`PytestOutcome.calls` holds the parsed
    record. A report without the plugin's suite property is invalid: nothing goes unrecorded silently.
    """
    more = (
        {"spool": (spool_dir / "stdout", spool_dir / "stderr")}
        if spool_dir is not None
        else {}
    )
    if spool_dir is not None:
        spool_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as stack:
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="memory-junit-"))
        rw2 = dict(rw)
        rw2[Path(tmp)] = "/junit"
        ro2, env2, plugin = ro, env, []
        if import_skips_fail:
            plug = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="memory-skipmark-"),
                ),
            )
            (plug / f"{SKIPMARK_MODULE}.py").write_text(SKIPMARK_SOURCE)
            ro2 = {**ro, plug: SKIPMARK_DIR}
            env2 = dict(env or {})
            path = env2.get("PYTHONPATH")
            env2["PYTHONPATH"] = f"{path}:{SKIPMARK_DIR}" if path else SKIPMARK_DIR
            plugin = ["-p", SKIPMARK_MODULE]
        record_dir: Path | None = None
        if calls is not None:
            from . import example_calls

            plug = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="memory-calls-plugin-"),
                ),
            )
            example_calls.plugin_files(plug)
            record_dir = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="memory-calls-"),
                ),
            )
            ro2 = {**ro2, plug: example_calls.CALLS_DIR}
            rw2[record_dir] = posixpath.dirname(example_calls.CALLS_OUT)
            env2 = {**(env2 or {}), **example_calls.calls_env(calls)}
            path = env2.get("PYTHONPATH")
            env2["PYTHONPATH"] = (
                f"{path}:{example_calls.CALLS_DIR}" if path else example_calls.CALLS_DIR
            )
            plugin = [*plugin, "-p", example_calls.CALLS_MODULE]
        r = run_confined(
            [
                str(python),
                "-m",
                "pytest",
                tests_dir_in_box,
                "-q",
                "-p",
                "no:cacheprovider",
                *plugin,
                "--rootdir",
                cwd,
                "--junitxml",
                "/junit/r.xml",
                "-o",
                "addopts=",
                "-o",
                "junit_family=xunit1",
            ],
            ro=ro2,
            rw=rw2,
            cwd=cwd,
            timeout_s=timeout_s,
            env=env2,
            **more,
        )
        out = PytestOutcome(
            returncode=r.returncode,
            timed_out=r.timed_out,
            output=(r.stdout + r.stderr)[-4000:],
            full_output=r.stdout + r.stderr,
        )
        root = None if r.timed_out else _read_junit(Path(tmp))
        if record_dir is not None:
            from .example_calls import CALLS_LOADED, CALLS_OUT, read_calls

            loaded = root is not None and _suite_property(root, CALLS_LOADED)
            out.calls = read_calls(
                record_dir / posixpath.basename(CALLS_OUT),
                loaded_property=loaded,
            )
            if not loaded:
                out.valid = False
        if r.timed_out or root is None:
            out.valid = False
            return out
        cases = list(root.iter("testcase"))
        # skipped for a failed import: failures, as an ImportError would be
        missing: set[str] = set()
        module_skips = 0
        for case in cases:
            name = _case_id(case, cwd, tests_dir_in_box)
            tags = {child.tag for child in case}
            if tags & {"failure", "error"}:
                out.failed.add(name)
            elif "skipped" in tags:
                mark = _skip_mark(case) if import_skips_fail else None
                if mark == "collect":
                    module_skips += 1
                    missing.add(name.split("::", 1)[0] + "::" + MODULE_SKIPPED)
                elif mark == "test":
                    missing.add(name)
                else:
                    out.skipped.add(name)
            else:
                out.passed.add(name)
        out.passed -= out.failed
        out.skipped -= out.failed
        rc = r.returncode
        out.valid = (
            (rc == 0 and not out.failed and bool(cases))
            or (rc == 1 and bool(out.failed))
            # pytest exits 5 when no test was collected: no entry, or (marked) only modules skipped for an import
            or (rc == 5 and len(cases) == module_skips)
        )
        if import_skips_fail and not _skipmark_loaded(root):
            out.valid = False
        if record_dir is not None and not _suite_property(root, CALLS_LOADED):
            out.valid = False
        out.failed |= missing
        return out
