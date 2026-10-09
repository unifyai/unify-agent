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
import posixpath
import pwd
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


def _account_home() -> Path:
    """The account's home from the password database; ``$HOME`` may point elsewhere (tests move it)."""
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


# The box's interpreter: ``MEMORY_V2_PYTHON``, else the controller's own (a fixed host path would not exist elsewhere).
PYTHON = Path(os.environ.get("MEMORY_V2_PYTHON") or sys.executable)

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

#: Never in a box's environment, whatever a caller passes: memory v2's route for Sol's own model calls.
HARNESS_ONLY_ENV = frozenset(
    {
        "UNIFY_MEMORY_V2_SOL_BASE_URL",
        "UNIFY_MEMORY_V2_SOL_TOKEN",
        "UNIFY_MEMORY_V2_SOL_TOKEN_FD",
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


def _proc_stat(pid: int) -> tuple[int, str, str] | None:
    """(ppid, state, starttime) of *pid*, or None when it no longer exists."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = raw[raw.rfind(")") + 2 :].split()
    return int(fields[1]), fields[0], fields[19]


def _descendants(pid: int) -> list[tuple[int, str]]:
    """Every live descendant of *pid* as (pid, starttime), so a reused pid is never mistaken for it."""
    children: dict[int, list[tuple[int, str]]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        st = _proc_stat(int(entry))
        if st is not None:
            children.setdefault(st[0], []).append((int(entry), st[2]))
    found: list[tuple[int, str]] = []
    todo = [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child[0])
    return found


def _alive(pid: int, starttime: str) -> bool:
    st = _proc_stat(pid)
    return st is not None and st[2] == starttime and st[1] not in ("Z", "X")


class _Tail:
    """Drain one pipe of the box on a thread, keeping only its last ``cap`` bytes (bounded memory).

    The box can always reach the host's pipes (its pid 1 holds them, and ``/proc/1/fd/1`` is open to the
    same user), so the host must never buffer them whole or decode them strictly: the text is decoded as
    UTF-8 with replacement characters, prefixed by a marker when bytes were dropped.
    """

    def __init__(self, stream, cap: int = CAPTURE_MAX_BYTES) -> None:
        self._stream, self._cap = stream, cap
        self._buf = bytearray()
        self._dropped = 0
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(1 << 16)
                if not chunk:
                    break
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
    """SIGKILL bubblewrap's process group and every process below it; raise unless all are gone."""
    victims = _descendants(proc.pid)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for pid, start in victims:
        if _alive(pid, start):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        proc.wait(timeout=_KILL_GRACE_S)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + _KILL_GRACE_S
    while True:
        survivors = [pid for pid, start in victims if _alive(pid, start)]
        if proc.poll() is not None and not survivors:
            return
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"sandbox processes survived SIGKILL: bwrap {proc.pid}, others {survivors}",
            )
        time.sleep(0.1)


def run_confined(
    argv: list[str],
    *,
    ro: dict[Path, str] | None = None,
    rw: dict[Path, str] | None = None,
    cwd: str = "/tmp",
    timeout_s: float = 120.0,
    env: dict[str, str] | None = None,
    late_ro: dict[Path, str] | None = None,
) -> SandboxResult:
    """Run *argv* in a bubblewrap box; ``ro``/``rw`` map host paths to their paths inside the box.

    *late_ro* paths are bound read-only after the ``rw`` binds, so a path inside a writable bind stays
    read-only (memory v2.1's generated files in the writer's box, P3 Amendment C). Each source must be an
    existing path that is not a link.

    *env* is the box's whole environment beyond PATH, HOME and PYTHONDONTWRITEBYTECODE. It reaches
    bubblewrap through an unlinked file (``--args``), not its command line; it must never hold credentials.
    ``stdout`` and ``stderr`` are bounded tails (:data:`CAPTURE_MAX_BYTES` each), decoded with replacement.
    """
    for name in env or {}:
        if name.upper() in HARNESS_ONLY_ENV:
            raise ValueError(f"{name} never enters the box")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required")
    if platform.machine() != "x86_64":
        raise RuntimeError("the seccomp filter is built for x86_64 only")
    if not os.access(PRLIMIT, os.X_OK):
        raise RuntimeError(f"{PRLIMIT} is required to bound the box's processes")
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
        args += list(argv)
        proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin"},
            pass_fds=(env_fd, bpf_fd),
        )
    tails: list[_Tail] = []

    def result(returncode: int, timed_out: bool) -> SandboxResult:
        out, err = (t.text(_KILL_GRACE_S) for t in tails)
        return SandboxResult(returncode, out, err, timed_out)

    try:
        # inside the try: a failure to start a drain thread still kills the box
        tails.append(_Tail(proc.stdout))
        tails.append(_Tail(proc.stderr))
        proc.wait(timeout=timeout_s)
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
# forge the properties, as it could the report (R16: careless, not malicious).
SKIP_PROPERTY = "memv2_import_skip"
SKIPMARK_LOADED = "memv2_skipmark"
SKIPMARK_MODULE = "_memv2_skipmark"
SKIPMARK_DIR = "/memv2-skipmark"  # the plugin's read-only directory in the box (on PYTHONPATH, last)
MODULE_SKIPPED = "test module skipped"
# The plugin, written into a fresh host directory per run; the host never imports it (importing it patches
# pytest.importorskip in that process).
SKIPMARK_SOURCE = r'''"""Mark skips caused by a failed import in the junit report (the memory gate's runs; see sandbox_run)."""
import unittest

import pytest
from _pytest import outcomes as _outcomes
from _pytest.junitxml import xml_key

PROPERTY = "memv2_import_skip"
LOADED = "memv2_skipmark"
_MARK = "_memv2_import_skip"
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
        raise pytest.UsageError("_memv2_skipmark needs pytest's junit writer (--junitxml)")
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


def _skipmark_loaded(root: ET.Element) -> bool:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        props = suite.find("properties")
        for p in props.iter("property") if props is not None else ():
            if p.get("name") == SKIPMARK_LOADED:
                return True
    return False


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
) -> PytestOutcome:
    """Run pytest on *tests_dir_in_box* in the box; see :class:`PytestOutcome` for ids and validity.

    *import_skips_fail* (default off: a skip is a skip): load the skip-marking plugin and count a skip caused
    by a failed import as a failure, ``<file>::test module skipped`` for a module skipped at collection and
    the test's id for a test.
    """
    with contextlib.ExitStack() as stack:
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="memv2-junit-"))
        rw2 = dict(rw)
        rw2[Path(tmp)] = "/junit"
        ro2, env2, plugin = ro, env, []
        if import_skips_fail:
            plug = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="memv2-skipmark-"),
                ),
            )
            (plug / f"{SKIPMARK_MODULE}.py").write_text(SKIPMARK_SOURCE)
            ro2 = {**ro, plug: SKIPMARK_DIR}
            env2 = dict(env or {})
            path = env2.get("PYTHONPATH")
            env2["PYTHONPATH"] = f"{path}:{SKIPMARK_DIR}" if path else SKIPMARK_DIR
            plugin = ["-p", SKIPMARK_MODULE]
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
        )
        out = PytestOutcome(
            returncode=r.returncode,
            timed_out=r.timed_out,
            output=(r.stdout + r.stderr)[-4000:],
        )
        if r.timed_out:
            out.valid = False
            return out
        root = _read_junit(Path(tmp))
        if root is None:
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
        out.failed |= missing
        return out
