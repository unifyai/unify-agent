"""OS confinement for the shell and file access the actor gets.

Every shell cell, every subprocess a Python cell starts, and the sandboxed
Python worker run inside bubblewrap (Linux) under one policy (the
``UNIFY_WORKSPACE=sandboxed`` switch until the code freeze baked it in):

* The root is an allowlist, read-only (:data:`_ROOT_ALLOWLIST`): the system
  directories (``/usr``, ``/bin``, ``/lib*``, ``/sbin``), a few files of
  ``/etc`` named one by one, the interpreter (its venv and base install,
  followed through their links), the Unify package and the editable installs
  the venv's ``.pth`` files add, and the files ``LD_PRELOAD`` names when the
  harness itself runs with them. Nothing else of the host exists: no
  ``/home`` beyond those paths, ``/root``, ``/var``, ``/opt``, ``/mnt``,
  ``/workspaces``, ``/srv``, ``/media``, ``/run``. Never ``/`` itself, and
  never a whole home directory (rule ``root-allowlist``).
* The workspace (``<UNIFY_HOME>/workspace`` or ``UNIFY_LOCAL_ROOT``) and a
  private ``/tmp`` are the only writable places.
* A seccomp filter (:func:`seccomp_program`) lets ``socket`` and
  ``socketpair`` create only AF_UNIX, AF_INET and AF_INET6 sockets (on WSL2
  AF_VSOCK reaches the Windows host whatever the network namespace), refuses
  new user namespaces and io_uring, and kills a process that enters the
  kernel through another architecture's system calls (rule
  ``socket-families``).
* The Unify state directory (``UNIFY_HOME``) is hidden behind an empty tmpfs,
  except read-only views of the transcripts directory, the store file and the
  package venv, and the writable workspace. The harness's internal
  transcripts (``internal-transcripts``: the storage review, whose prompt
  carries the environment's checked outcome) stay hidden even when a mount
  that is seen contains them (a workspace configured as ``UNIFY_HOME``).
* The harness's log directories (``UNILLM_LOG_DIR``, which holds every LLM
  request and reply, ``UNILLM_OTEL_LOG_DIR``, ``UNIFY_LOG_DIR`` and
  ``UNIFY_OTEL_LOG_DIR``, wherever they are configured) are hidden too.
* Credential locations (``~/.ssh``, ``~/.config``, ``~/.aws``, ``~/.gnupg`` and
  a few other well-known ones) are hidden behind a notice, and so is every
  ``.env`` file found in the working directory and its parents, the home
  directory and its non-hidden subdirectories two levels down, the Unify
  checkout and the editable installs, where a mount would show it (outside
  every mount it does not exist at all).
* The environment loses every variable whose name contains KEY, TOKEN, SECRET,
  PASSWORD or CREDENTIAL.
* The network namespace is private: nothing but a loopback of its own, unless
  ``UNIFY_WORKSPACE_NETWORK=proxy``, in which case one port on that loopback
  forwards to the proxy at ``127.0.0.1:UNIFY_WORKSPACE_PROXY_PORT`` on the
  host, and nothing else is reachable.

Each rule has a name. A refusal the harness makes itself (a missing
bubblewrap, a path :func:`check_readable` rejects) raises
:class:`SandboxRefusal` naming it; a hidden directory contains one file,
``UNIFY_SANDBOX_MASKED``, and a hidden file reads as one line, each naming the
rule that hid it; an OS error in a shell cell's output gets a note naming the
rule behind it (:func:`annotate_refusals`).

The harness's package installer (unify/environment.py) runs under the same
policy with two additions it asks for itself: the workspace environment and
the installer's cache are writable, and it keeps the host's network to reach
the package index.

Without bubblewrap nothing runs: the harness refuses rather than run the
command unconfined. What this does not confine is Python cells themselves:
they run in the harness's own process (``exec``), so only the subprocesses they
start go through the sandbox.
"""

from __future__ import annotations

import contextvars
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

from unify.common.tool_errors import ToolInputError

__all__ = [
    "RULES",
    "SandboxPolicy",
    "SandboxRefusal",
    "annotate_refusals",
    "build_policy",
    "check_readable",
    "confined_subprocesses",
    "scrubbed_env",
    "unconfined",
    "wrap_argv",
]

# Rule name -> what it does, as a refused action reports it.
RULES: dict[str, str] = {
    "sandbox-required": (
        "shell commands and Python cells run only inside bubblewrap, never "
        "unconfined"
    ),
    "workspace-write": (
        "only the workspace and a private /tmp are writable; everything else is "
        "mounted read-only"
    ),
    "root-allowlist": (
        "only the system directories, a few /etc files, the interpreter, the "
        "Unify package and the paths the policy names exist; the rest of the "
        "host (other home directories' contents, /root, /var, /mnt, /opt, "
        "/workspaces) does not"
    ),
    "socket-families": (
        "only AF_UNIX, AF_INET and AF_INET6 sockets can be created; new user "
        "namespaces and io_uring are refused"
    ),
    "mask-unify-state": (
        "the Unify state directory is hidden, except read-only views of the "
        "transcripts, the store file and the package venv, and the workspace"
    ),
    "mask-harness-logs": (
        "the harness's log directories (every LLM request and reply, traces) "
        "are hidden"
    ),
    "mask-credentials": "credential directories and files are hidden",
    "mask-env-file": ".env files are hidden",
    "mask-proc": "the harness's /proc (its environment and memory) is not readable",
    "private-tmp": "/tmp is private to the sandbox; the host's /tmp is not readable",
    "regular-files-only": "only regular files can be read",
    "env-scrub": (
        "environment variables whose names contain KEY, TOKEN, SECRET, PASSWORD "
        "or CREDENTIAL are removed"
    ),
    "network-off": "the sandbox has no network, only a loopback of its own",
    "network-proxy-only": (
        "the only network is one loopback port forwarded to the configured proxy"
    ),
}

SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")

# Relative to the home directory. Directories are masked with an empty tmpfs,
# files with a one-line notice.
CREDENTIAL_PATHS = (
    ".ssh",
    ".config",
    ".aws",
    ".gnupg",
    ".docker",
    ".kube",
    ".azure",
    ".claude",
    ".codex",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".pgpass",
)

_ENV_FILE_ALLOWED = {".env.example", ".env.sample", ".env.template"}
_ENV_SCAN_SKIP = {
    "node_modules",
    ".cache",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "site-packages",
    ".npm",
    ".cargo",
    ".rustup",
}
_ENV_SCAN_DEPTH = 2
_POLICY_TTL_S = 300.0

MASK_NOTICE_NAME = "UNIFY_SANDBOX_MASKED"
_PROXY_MOUNT = "/tmp/.unify-proxy"

# The host paths every sandboxed command sees besides those derived at start
# (_derived_roots: the interpreter, the Unify package, editable installs, the
# trusted LD_PRELOAD) and those the policy mounts (workspace, state views,
# proxy). Each is mounted read-only where it exists; a symlink whose target
# is itself visible stays a symlink, any other is mounted from its target
# (WSL's /etc/resolv.conf links into /mnt/wsl). The root they sit on is an
# empty tmpfs, remounted read-only once everything is in place. /etc is never
# mounted whole: it holds the host's configuration, some of it secret
# (shadow, ssh host keys, pip.conf index credentials, docker).
_ROOT_ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("/usr", "programs, shared libraries, Python's system packages, zoneinfo"),
    ("/bin", "the shells and core utilities (a link to usr/bin when merged)"),
    ("/sbin", "system programs some tools call by path (a link when merged)"),
    ("/lib", "shared libraries and the dynamic loader (a link when merged)"),
    ("/lib64", "the x86_64 dynamic loader's path (a link when merged)"),
    ("/lib32", "32-bit libraries, where installed (a link when merged)"),
    ("/libx32", "x32 libraries, where installed (a link when merged)"),
    ("/etc/ld.so.cache", "the dynamic loader's library cache"),
    ("/etc/ld.so.conf", "the dynamic loader's search path"),
    ("/etc/ld.so.conf.d", "the dynamic loader's search path, per package"),
    ("/etc/alternatives", "the links Debian's update-alternatives makes"),
    ("/etc/nsswitch.conf", "which databases name lookups use"),
    ("/etc/hosts", "localhost's name"),
    ("/etc/host.conf", "the resolver's options"),
    ("/etc/gai.conf", "address ordering for getaddrinfo"),
    ("/etc/resolv.conf", "DNS for the installer's network (its real target)"),
    ("/etc/protocols", "protocol names (getprotobyname)"),
    ("/etc/services", "service names (getservbyname)"),
    ("/etc/ssl/certs", "CA certificates for TLS"),
    ("/etc/ssl/openssl.cnf", "OpenSSL's configuration"),
    ("/etc/ca-certificates", "the CA bundle's local configuration"),
    ("/etc/pki/tls/certs", "CA certificates on Red Hat systems"),
    ("/etc/pki/ca-trust", "CA certificates on Red Hat systems"),
    ("/etc/localtime", "the host's time zone"),
    ("/etc/timezone", "the host's time zone name"),
    ("/etc/os-release", "the distribution's name and version"),
    ("/etc/mime.types", "file types for Python's mimetypes"),
    ("/etc/fonts", "fontconfig's configuration (plots)"),
    ("/etc/passwd", "generated: root, this account and nobody, never the host's"),
    ("/etc/group", "generated: root, this account's group and nogroup"),
)

# Generated, never bound from the host: the other accounts are not the cell's
# business.
_GENERATED_ETC = ("/etc/passwd", "/etc/group")

_SYSTEM_DIRS = tuple(p for p, _ in _ROOT_ALLOWLIST if not p.startswith("/etc/"))

# Environment variables a sandboxed command gets only from the harness's own
# environment (set by the trusted runner, e.g. the office benchmark's fake
# clock), never from what a cell passes to a subprocess.
TRUSTED_ENV = ("LD_PRELOAD", "FAKETIME", "FAKETIME_SHARED", "TZ")


class SandboxRefusal(ToolInputError):
    """An action the workspace sandbox refuses; ``rule`` names the rule."""

    def __init__(self, rule: str, detail: str, *, suggestion: str | None = None):
        self.rule = rule
        super().__init__(
            f"Refused by workspace sandbox rule `{rule}` ({RULES[rule]}): {detail}",
            suggestion=suggestion,
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_ENV_MARKERS)


def scrubbed_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """*env* (default: this process's) without credential-named variables."""
    source = os.environ if env is None else env
    return {k: v for k, v in source.items() if not is_secret_name(k)}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class SandboxPolicy:
    """Everything the bubblewrap command line and the path checks derive from."""

    workspace: Path
    state_dir: Path
    readonly_state: list[Path] = field(default_factory=list)
    masked_dirs: list[tuple[Path, str]] = field(default_factory=list)
    masked_files: list[tuple[Path, str]] = field(default_factory=list)
    # Harness-only directories, each with the rule that hides it: hidden after
    # every mount, so no mount that contains them shows them.
    hidden: list[tuple[Path, str]] = field(default_factory=list)
    network: str = ""  # "" (off) or "proxy"
    proxy_port: int = 0
    notices_dir: Optional[Path] = None
    # The allowlisted root (_root_mounts): bubblewrap arguments, and the host
    # paths (resolved) they show, which the harness's own file tools check.
    root_args: list[str] = field(default_factory=list)
    root_visible: list[Path] = field(default_factory=list)
    created: float = field(default_factory=time.monotonic)

    # -- path checks ---------------------------------------------------------
    def readable_violation(self, path: Path) -> Optional[tuple[str, str]]:
        """``(rule, detail)`` when *path* (resolved) is outside the readable set."""
        resolved = Path(os.path.realpath(path))
        if _within(resolved, Path("/proc")):
            return "mask-proc", f"{resolved} is under /proc"
        seen = (self.workspace, *self.readonly_state)
        for hidden, rule in self.hidden:
            if _within(resolved, hidden) and not any(
                _within(resolved, v) and _within(v, hidden) and v != hidden
                for v in seen
            ):
                return rule, f"{resolved} is inside the harness's {hidden}"
        # Mounted last, so visible whatever contains them.
        if any(_within(resolved, v) for v in (self.workspace, *self.readonly_state)):
            return None
        if _within(resolved, self.state_dir):
            return (
                "mask-unify-state",
                f"{resolved} is inside the Unify state directory {self.state_dir}",
            )
        for masked, rule in self.masked_dirs:
            if _within(resolved, masked):
                return rule, f"{resolved} is inside {masked}"
        for masked, rule in self.masked_files:
            if resolved == masked:
                return rule, f"{resolved} is hidden"
        if _is_env_file(resolved.name):
            return "mask-env-file", f"{resolved} is a .env file"
        if _within(resolved, Path("/tmp")):
            return "private-tmp", f"{resolved} is in the host's /tmp"
        if not any(_within(resolved, r) for r in self.root_visible):
            return (
                "root-allowlist",
                f"{resolved} is outside every path the sandbox mounts",
            )
        return None


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_env_file(name: str) -> bool:
    return (name == ".env" or name.startswith(".env.")) and (
        name not in _ENV_FILE_ALLOWED
    )


def _find_env_files(roots: Sequence[tuple[Path, int]], skip: Sequence[Path]):
    """``.env`` files under each ``(root, depth)``; never descends into *skip*."""
    found: set[Path] = set()
    for root, depth in roots:
        stack = [(root, 0)]
        while stack:
            directory, level = stack.pop()
            if any(_within(directory, s) for s in skip):
                continue
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False) and _is_env_file(
                        entry.name,
                    ):
                        found.add(Path(entry.path))
                    elif (
                        level < depth
                        and not entry.name.startswith(".")
                        and entry.name not in _ENV_SCAN_SKIP
                        and entry.is_dir(follow_symlinks=False)
                    ):
                        stack.append((Path(entry.path), level + 1))
                except OSError:
                    continue
    return sorted(found)


def _write_notices(directory: Path) -> Path:
    """The notice files, the generated /etc files and the seccomp program.

    Under a private ``mkdtemp`` directory of the host's /tmp, which no
    sandboxed command sees (its /tmp is its own).
    """
    directory.mkdir(parents=True, exist_ok=True)
    for rule, text in RULES.items():
        (directory / rule).write_text(
            f"Hidden by the Unify workspace sandbox (rule {rule}: {text}).\n",
        )
    etc = directory / "etc"
    etc.mkdir(exist_ok=True)
    passwd, group = _account_files()
    (etc / "passwd").write_text(passwd)
    (etc / "group").write_text(group)
    (directory / "seccomp.bpf").write_bytes(seccomp_program())
    return directory


def _account_files() -> tuple[str, str]:
    """Minimal ``/etc/passwd`` and ``/etc/group``: root, this account, nobody.

    Enough for ``id``, ``whoami``, ``getpass.getuser()`` and tools that look
    up their own user; the host's other accounts (and their home
    directories) are not listed. No real names (the GECOS field is empty).
    """
    import grp
    import pwd

    users: list[str] = []
    for uid in dict.fromkeys((0, os.getuid(), 65534)):
        try:
            e = pwd.getpwuid(uid)
        except KeyError:
            continue
        users.append(
            f"{e.pw_name}:x:{e.pw_uid}:{e.pw_gid}::{e.pw_dir}:{e.pw_shell}\n",
        )
    groups: list[str] = []
    for gid in dict.fromkeys((0, os.getgid(), 65534)):
        try:
            g = grp.getgrgid(gid)
        except KeyError:
            continue
        groups.append(f"{g.gr_name}:x:{g.gr_gid}:\n")
    return "".join(users), "".join(groups)


_NOTICES_DIR: Optional[Path] = None
_POLICY_CACHE: Optional[SandboxPolicy] = None
_POLICY_LOCK = threading.Lock()


def _notices_dir() -> Path:
    global _NOTICES_DIR
    if _NOTICES_DIR is None or not (_NOTICES_DIR / "seccomp.bpf").is_file():
        _NOTICES_DIR = _write_notices(
            Path(tempfile.mkdtemp(prefix="unify-sandbox-notices-")),
        )
    return _NOTICES_DIR


def _current_run_records() -> Optional[Path]:
    """The folder holding this run's agent record, if one is open."""
    from unify.agents.binding import current_root_pool

    pool = current_root_pool()
    path = pool.record.path if pool is not None else None
    return Path(os.path.realpath(path.parent)) if path is not None else None


LOG_DIR_SETTINGS = (
    "UNILLM_LOG_DIR",
    "UNILLM_OTEL_LOG_DIR",
    "UNIFY_LOG_DIR",
    "UNIFY_OTEL_LOG_DIR",
)


def _log_dir_settings() -> list[str]:
    """Each harness log directory as configured (environment, then settings)."""
    from unify.settings import SETTINGS

    try:
        import unillm

        unillm_settings = unillm.SETTINGS
    except Exception:
        unillm_settings = None
    out = []
    for name in LOG_DIR_SETTINGS:
        source = unillm_settings if name.startswith("UNILLM_") else SETTINGS
        value = os.environ.get(name, "").strip() or str(
            getattr(source, name, "") or "",
        )
        out.append(value.strip())
    return out


def _log_dirs() -> list[Path]:
    """The configured log directories, created, to hide from every cell.

    A directory that holds the interpreter or the Unify package is not hidden
    (a cell could not start), and a warning names it.
    """
    import logging

    keep = [
        Path(os.path.realpath(sys.prefix)),
        Path(os.path.realpath(sys.executable)),
        Path(__file__).resolve().parents[1],
    ]
    dirs: list[Path] = []
    for raw in _log_dir_settings():
        if not raw:
            continue
        path = Path(os.path.realpath(Path(raw).expanduser()))
        if any(_within(k, path) for k in keep):
            logging.getLogger(__name__).warning(
                "workspace sandbox: log directory %s holds the interpreter or the "
                "Unify package, so cells can read it",
                path,
            )
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if path not in dirs:
            dirs.append(path)
    return dirs


def _homes() -> list[Path]:
    """``/home``, the account's home and ``$HOME``, resolved: never mounted whole.

    The account's home comes from the password database, since ``$HOME`` may
    point elsewhere (the tests move it). A home under ``/tmp`` is the
    sandbox's private /tmp inside, so it is not counted.
    """
    import pwd

    raw = [Path("/home"), Path(pwd.getpwuid(os.getuid()).pw_dir), Path.home()]
    out: list[Path] = []
    for path in raw:
        real = Path(os.path.realpath(path))
        if not _within(real, Path("/tmp")) and real not in out:
            out.append(real)
    return out


def _too_broad(path: Path, homes: Sequence[Path]) -> bool:
    """Whether a mount of *path* would show ``/`` or a whole home directory."""
    for p in (Path(os.path.abspath(path)), Path(os.path.realpath(path))):
        if p == Path("/") or any(_within(home, p) for home in homes):
            return True
    return False


def _is_python_install(root: Path) -> bool:
    """A venv or a Python installation (not, say, ``~/.local`` above a link)."""
    return (root / "pyvenv.cfg").is_file() or any(
        (root / "lib").glob("python3*"),
    )


def _interpreter_roots() -> list[Path]:
    """The interpreter's venv and base install, by every name it is reached through.

    A uv venv's ``bin/python`` links to the base install by its minor-version
    name (``cpython-3.12-...``), itself a link to the patch release
    (``cpython-3.12.11-...``), and ``pyvenv.cfg`` names the former. Each name
    is listed (and mounted from its real target), so the chain resolves inside
    the sandbox although the home directory above it does not exist there.
    """
    roots = [
        Path(p)
        for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
    ]
    link = Path(os.path.abspath(sys.executable))
    for _ in range(40):
        roots.append(link.parent.parent)
        if not link.is_symlink():
            break
        link = Path(os.path.normpath(link.parent / os.readlink(link)))
    roots.append(Path(os.path.realpath(sys.executable)).parent.parent)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    try:
        lines = cfg.read_text().splitlines() if cfg.is_file() else []
    except OSError:
        lines = []
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip() == "home" and value.strip():
            roots.append(Path(os.path.abspath(value.strip())).parent)
    return [r for r in roots if _is_python_install(r)]


def _editable_roots() -> list[Path]:
    """The ``sys.path`` entries the venv's ``.pth`` files add: editable installs.

    That is how unillm (and any package installed with ``pip install -e``)
    reaches the harness's, and so the worker's, ``sys.path``. Entries a
    script directory or the working directory put there are not mounted;
    ``PYTHONPATH``'s are (:func:`_pythonpath_roots`).
    """
    import site

    on_path = {os.path.abspath(p) for p in sys.path if p}
    sites = {os.path.abspath(p) for p in site.getsitepackages()}
    sites |= {p for p in on_path if p.endswith(("site-packages", "dist-packages"))}
    out: list[Path] = []
    for directory in sorted(sites):
        try:
            names = sorted(n for n in os.listdir(directory) if n.endswith(".pth"))
        except OSError:
            continue
        for name in names:
            try:
                lines = (Path(directory) / name).read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith(("#", "import ", "import\t")):
                    continue
                path = os.path.abspath(os.path.join(directory, line))
                if path in on_path and os.path.exists(path):
                    out.append(Path(path))
    return out


def _pythonpath_roots() -> list[Path]:
    """The harness's own ``PYTHONPATH`` entries that are on ``sys.path``.

    Set by the trusted runner: the benchmark adapters put their relay
    client's directory there (``<attempt>/system/<bench>-client``, files
    read-only), and the worker reuses the harness's ``sys.path``.
    """
    on_path = {os.path.abspath(p) for p in sys.path if p}
    return [
        Path(os.path.abspath(p))
        for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if p and os.path.abspath(p) in on_path and os.path.exists(p)
    ]


def _preload_files() -> list[Path]:
    """The libraries the harness's own ``LD_PRELOAD`` names (the fake clock).

    Only what the trusted parent runs with; a cell never adds to it
    (:data:`TRUSTED_ENV`).
    """
    raw = os.environ.get("LD_PRELOAD", "")
    return [
        Path(p)
        for p in raw.replace(":", " ").split()
        if os.path.isabs(p) and os.path.isfile(p)
    ]


def _derived_roots() -> list[tuple[Path, str]]:
    """The read-only roots taken from this process at policy time, with reasons."""
    out: list[tuple[Path, str]] = []
    out += [
        (p, "the interpreter (venv and base install)") for p in _interpreter_roots()
    ]
    # The worker runs worker_child.py from the package by path.
    out.append((Path(__file__).resolve().parent, "the Unify package (the worker)"))
    out += [(p, "an editable install on sys.path (.pth)") for p in _editable_roots()]
    out += [(p, "the harness's PYTHONPATH (relay client)") for p in _pythonpath_roots()]
    out += [(p, "LD_PRELOAD of the harness (fake clock)") for p in _preload_files()]
    return out


def _is_system(path: Path) -> bool:
    return any(
        _within(path, Path(os.path.realpath(d)))
        for d in _SYSTEM_DIRS
        if os.path.isdir(d)
    )


def _root_mounts(notices: Path) -> tuple[list[str], list[Path]]:
    """The allowlisted root: bubblewrap arguments, and the host paths they show.

    :data:`_ROOT_ALLOWLIST` first, then :func:`_derived_roots`, each by its
    own name and its resolved one, outer paths before inner ones. A derived
    root that would show ``/`` or a whole home directory is left out, with a
    warning; nothing of it is then visible.
    """
    import logging

    homes = _homes()
    args: list[str] = []
    visible: list[Path] = []
    for path, _reason in _ROOT_ALLOWLIST:
        if path in _GENERATED_ETC:
            args += ["--ro-bind", str(notices / "etc" / Path(path).name), path]
            continue
        if not os.path.lexists(path):
            continue
        real = Path(os.path.realpath(path))
        if os.path.islink(path) and (
            path in _SYSTEM_DIRS or (real.exists() and _is_system(real))
        ):
            # A link stays a link; its target is visible through /usr.
            args += ["--symlink", os.readlink(path), path]
        elif real.exists():
            args += ["--ro-bind", str(real), path]
            visible.append(real)
    wanted: list[Path] = []
    for root, reason in _derived_roots():
        for p in (Path(os.path.abspath(root)), Path(os.path.realpath(root))):
            if not p.exists() or _is_system(Path(os.path.realpath(p))):
                continue
            if _too_broad(p, homes):
                logging.getLogger(__name__).warning(
                    "workspace sandbox: not mounting %s (%s): it would show / "
                    "or a whole home directory",
                    p,
                    reason,
                )
                continue
            if p not in wanted:
                wanted.append(p)
    # Outer before inner; an inner path under an outer one is already shown
    # (its resolved name is listed too, so links inside resolve).
    kept: list[Path] = []
    for p in sorted(wanted, key=lambda q: (len(q.parts), str(q))):
        if any(_within(p, k) for k in kept):
            continue
        kept.append(p)
        real = Path(os.path.realpath(p))
        args += ["--ro-bind", str(real), str(p)]
        visible.append(real)
    return args, visible


def build_policy(*, fresh: bool = False) -> SandboxPolicy:
    """The policy for the current settings and filesystem.

    Cached for five minutes (``fresh=True`` rebuilds it), since every sandboxed
    cell asks for it and finding ``.env`` files walks part of the home
    directory; a ``.env`` file created meanwhile is still refused by name by
    ``read_file`` and ``grep``, but a shell cell sees it until the rebuild.
    """
    global _POLICY_CACHE
    from unify.db import store_home, store_path
    from unify.settings import SETTINGS
    from unify.workspace import get_local_root

    with _POLICY_LOCK:
        cached = _POLICY_CACHE
        key = (
            str(store_home()),
            get_local_root(),
            store_path(),
            str(Path.home()),
            os.getcwd(),
            getattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", ""),
            getattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0),
            str(_current_run_records() or ""),
            tuple(_log_dir_settings()),
            sys.prefix,
            tuple(sys.path),
            os.environ.get("LD_PRELOAD", ""),
            os.environ.get("PYTHONPATH", ""),
        )
        if (
            not fresh
            and cached is not None
            and getattr(cached, "_key", None) == key
            and time.monotonic() - cached.created < _POLICY_TTL_S
        ):
            return cached

        state_dir = Path(os.path.realpath(store_home()))
        workspace = Path(os.path.realpath(get_local_root()))
        workspace.mkdir(parents=True, exist_ok=True)
        store = Path(os.path.realpath(store_path()))
        # Mounted only if present, and sessions pointed at it must find it.
        (state_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        # Cells may read (grep, tail) their own run's agent record, as
        # transcripts; never another run's, and only the harness writes it.
        records = _current_run_records()
        readonly = [
            p
            for p in (
                state_dir / "transcripts",
                *((records,) if records is not None else ()),
                store,
                Path(f"{store}-wal"),
                Path(f"{store}-shm"),
                state_dir / "venv",
            )
            if p.exists()
        ]
        home = Path(os.path.realpath(Path.home()))
        masked_dirs: list[tuple[Path, str]] = []
        masked_files: list[tuple[Path, str]] = []
        for rel in CREDENTIAL_PATHS:
            p = home / rel
            if p.is_dir():
                masked_dirs.append((Path(os.path.realpath(p)), "mask-credentials"))
            elif p.is_file():
                masked_files.append((Path(os.path.realpath(p)), "mask-credentials"))

        cwd = Path(os.getcwd())
        roots: list[tuple[Path, int]] = [(home, _ENV_SCAN_DEPTH)]
        roots += [(d, 0) for d in (cwd, *cwd.parents)]
        roots.append((Path(__file__).resolve().parents[1], 1))
        # The editable installs the root mounts (unillm's checkout).
        roots += [(p, 1) for p in _editable_roots()]
        skip = [state_dir, workspace, *(d for d, _ in masked_dirs)]
        for env_file in _find_env_files(roots, skip):
            masked_files.append((Path(os.path.realpath(env_file)), "mask-env-file"))

        network = str(getattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "") or "")
        port = int(getattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0) or 0)
        from unify.transcripts import INTERNAL_DIRNAME

        notices = _notices_dir()
        root_args, root_visible = _root_mounts(notices)
        policy = SandboxPolicy(
            workspace=workspace,
            state_dir=state_dir,
            readonly_state=readonly,
            masked_dirs=masked_dirs,
            masked_files=masked_files,
            hidden=[
                (state_dir / INTERNAL_DIRNAME, "mask-unify-state"),
                *((d, "mask-harness-logs") for d in _log_dirs()),
            ],
            network=network,
            proxy_port=port,
            notices_dir=notices,
            root_args=root_args,
            root_visible=root_visible,
        )
        policy._key = key  # type: ignore[attr-defined]
        _POLICY_CACHE = policy
        return policy


def check_readable(path: str | os.PathLike, policy: SandboxPolicy) -> Path:
    """Resolve *path* (relative to the workspace) or raise naming the rule."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = policy.workspace / p
    violation = policy.readable_violation(p)
    if violation is not None:
        rule, detail = violation
        raise SandboxRefusal(rule, detail)
    return Path(os.path.realpath(p))


# ---------------------------------------------------------------------------
# bubblewrap
# ---------------------------------------------------------------------------


def bwrap_path() -> Optional[str]:
    return shutil.which("bwrap")


def require_bwrap() -> str:
    path = bwrap_path()
    if path is None or not sys.platform.startswith("linux"):
        raise SandboxRefusal(
            "sandbox-required",
            "bubblewrap (bwrap) is not available on this machine, so the command "
            "was not run",
            suggestion="Install bubblewrap (apt install bubblewrap).",
        )
    return path


# ---------------------------------------------------------------------------
# seccomp: socket families, user namespaces, io_uring
# ---------------------------------------------------------------------------

# Pattern from the memory-v2 runner (unify/memory_v2/sandbox_run.py at
# 6ec1e9821), extended to aarch64.
_SECCOMP_FD = 9
_SECCOMP_EXEC = 'exec "$@" 9<"$0"'

# Per architecture: its AUDIT_ARCH value, whether x32 system call numbers
# (bit 30) must be refused, and the numbers of the calls the filter looks at.
_SECCOMP_ARCHES: dict[str, tuple[int, bool, dict[str, int]]] = {
    "x86_64": (
        0xC000003E,
        True,
        {
            "socket": 41,
            "socketpair": 53,
            "clone": 56,
            "unshare": 272,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "clone3": 435,
        },
    ),
    "aarch64": (
        0xC00000B7,
        False,
        {
            "socket": 198,
            "socketpair": 199,
            "clone": 220,
            "unshare": 97,
            "io_uring_setup": 425,
            "io_uring_enter": 426,
            "io_uring_register": 427,
            "clone3": 435,
        },
    ),
}
_MACHINE_ARCH = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}
ALLOWED_SOCKET_FAMILIES = (1, 2, 10)  # AF_UNIX, AF_INET, AF_INET6
_X32_SYSCALL_BIT = 0x40000000
_CLONE_NEWUSER = 0x10000000
_EPERM, _ENOSYS, _EAFNOSUPPORT = 1, 38, 97
_RET_ALLOW = 0x7FFF0000
_RET_KILL_PROCESS = 0x80000000


def _ret_errno(errno: int) -> int:
    return 0x00050000 | errno


def seccomp_arch(machine: Optional[str] = None) -> str:
    """The filter's architecture for *machine* (default: this one), or refuse."""
    import platform

    machine = machine or platform.machine()
    arch = _MACHINE_ARCH.get(machine.lower())
    if arch is None:
        raise SandboxRefusal(
            "sandbox-required",
            f"the socket-family filter is built for x86_64 and aarch64, not "
            f"{machine}, so the command was not run",
        )
    return arch


def seccomp_program(arch: Optional[str] = None) -> bytes:
    """A classic-BPF seccomp filter (``struct sock_filter`` array), assembled here.

    * a system call made through another architecture's entry (i386's
      ``int 0x80`` on x86_64, 32-bit ARM on aarch64) kills the process, and
      on x86_64 x32 system call numbers get ENOSYS;
    * ``socket`` / ``socketpair`` with a family other than AF_UNIX, AF_INET
      or AF_INET6 get EAFNOSUPPORT (AF_VSOCK, AF_NETLINK, AF_PACKET, ...);
    * ``clone`` / ``unshare`` with CLONE_NEWUSER get EPERM; ``clone3``
      (flags in memory, which a filter cannot read) gets ENOSYS, so the C
      library falls back to ``clone``;
    * io_uring, which can open sockets without the ``socket`` call, gets
      ENOSYS.

    Network reach is unchanged: AF_INET follows the network namespace
    (``--unshare-net`` or ``--share-net``) and the proxy mode as before.
    """
    import struct

    audit_arch, x32, nr = _SECCOMP_ARCHES[arch or seccomp_arch()]
    ld_w_abs, jeq, jge, jset, ret = 0x20, 0x15, 0x35, 0x45, 0x06
    # struct seccomp_data: nr at 0, arch at 4, args[0]'s low word at 16 (both
    # architectures are little endian; the kernel reads these as int).
    nr_off, arch_off, arg0_off = 0, 4, 16
    prog: list[tuple] = [
        ("ld", arch_off),
        ("jeq", audit_arch, "nr", "kill"),
        ("label", "nr"),
        ("ld", nr_off),
        *([("jge", _X32_SYSCALL_BIT, "enosys", None)] if x32 else []),
        ("jeq", nr["socket"], "family", None),
        ("jeq", nr["socketpair"], "family", None),
        ("jeq", nr["clone"], "newuser", None),
        ("jeq", nr["unshare"], "newuser", None),
        ("jeq", nr["clone3"], "enosys", None),
        ("jeq", nr["io_uring_setup"], "enosys", None),
        ("jeq", nr["io_uring_enter"], "enosys", None),
        ("jeq", nr["io_uring_register"], "enosys", None),
        ("ret", _RET_ALLOW),
        ("label", "family"),
        ("ld", arg0_off),
        *[("jeq", fam, "allow", None) for fam in ALLOWED_SOCKET_FAMILIES],
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


def wrap_argv(
    argv: Sequence[str],
    policy: SandboxPolicy,
    *,
    cwd: Optional[str] = None,
    writable: Sequence[Path] = (),
    readonly: Sequence[Path] = (),
    share_network: bool = False,
) -> list[str]:
    """*argv* as a bubblewrap command line under *policy*.

    *writable* paths are bound read-write after everything else, *readonly*
    ones (a program the harness runs from outside the allowlisted root: the
    installer's ``uv``) read-only on the root, and *share_network* keeps the
    host's network instead of the policy's; only the harness's package
    installer asks for these (unify/environment.py).

    The command line starts ``/bin/sh -c 'exec "$@" 9<"$0"' <seccomp.bpf>``:
    the shell opens the seccomp program on descriptor 9 for bubblewrap
    (``--seccomp 9``), which reads and closes it, and replaces itself with
    bubblewrap (same process, same process group).
    """
    bwrap = require_bwrap()
    notices = policy.notices_dir
    root_args = policy.root_args
    if notices is None or not (notices / "seccomp.bpf").is_file() or not root_args:
        notices = _notices_dir()
        root_args = _root_mounts(notices)[0]
    homes = _homes()
    args: list[str] = [bwrap, *root_args]
    for path in readonly:
        for p in dict.fromkeys(
            (Path(os.path.abspath(path)), Path(os.path.realpath(path))),
        ):
            if p.exists():
                args += ["--ro-bind", str(Path(os.path.realpath(p))), str(p)]
    # $HOME exists (empty, read-only) unless something mounted shows it.
    home = Path(os.path.abspath(os.environ.get("HOME", "") or "/"))
    if home != Path("/") and not any(
        _within(home, Path(args[i + 2]))
        for i in range(1, len(args) - 2)
        if args[i] in _BIND_OPTIONS
    ):
        args += ["--dir", str(home)]
    args += [
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--seccomp",
        str(_SECCOMP_FD),
    ]
    extra = [Path(os.path.realpath(p)) for p in writable]
    shown = (*policy.readonly_state, policy.workspace, *extra)

    def _inside_shown(path: Path) -> bool:
        return any(_within(path, p) and path != p for p in shown)

    def _hide(path: Path, rule: str) -> list[str]:
        # Its mount point must exist on the host when the mount above it is
        # read-only.
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            if not path.is_dir():
                raise
        return ["--tmpfs", str(path), "--ro-bind", str(notices / rule), _notice(path)]

    # Hide the state directory, then put back what may be seen.
    args += ["--tmpfs", str(policy.state_dir)]
    args += ["--ro-bind", str(notices / "mask-unify-state"), _notice(policy.state_dir)]
    for path, rule in policy.masked_dirs:
        args += ["--tmpfs", str(path), "--ro-bind", str(notices / rule), _notice(path)]
    # A hidden file outside everything mounted does not exist in the sandbox
    # at all; a notice there would only reveal its directory's name.
    mounted = (*policy.root_visible, *shown)
    for path, rule in policy.masked_files:
        if any(_within(path, m) for m in mounted):
            args += ["--ro-bind", str(notices / rule), str(path)]
    # A harness-only directory outside the state directory and outside every
    # mount below is masked with the others; what is seen inside it is put
    # back below.
    for path, rule in policy.hidden:
        if not _within(path, policy.state_dir) and not _inside_shown(path):
            args += _hide(path, rule)
    # After every mask, so what may be seen is seen wherever it lives, and the
    # workspace last of all, so it is writable even inside a masked directory.
    for path in policy.readonly_state:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(policy.workspace), str(policy.workspace)]
    for path in extra:
        args += ["--bind", str(path), str(path)]
    # A harness-only directory inside a mount above is hidden last of all,
    # unless it holds a mount itself (a log directory that holds the
    # workspace): the harness's own file tools still refuse it then. One
    # inside the state directory that no mount contains is already behind its
    # tmpfs.
    for path, rule in policy.hidden:
        if _inside_shown(path) and not any(_within(p, path) for p in shown):
            args += _hide(path, rule)
    command = list(argv)
    if share_network:
        args.append("--share-net")
    elif policy.network == "proxy":
        bridge = _proxy_bridge(policy.proxy_port)
        args += ["--ro-bind", str(bridge.directory), _PROXY_MOUNT]
        command = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _FORWARDER_SRC,
            str(policy.proxy_port),
            f"{_PROXY_MOUNT}/proxy.sock",
            *command,
        ]
    workdir = cwd or os.getcwd()
    bound = any(_within(Path(os.path.realpath(workdir)), p) for p in extra)
    if (
        not bound and policy.readable_violation(Path(workdir)) is not None
    ) or not os.path.isdir(workdir):
        workdir = str(policy.workspace)
    # Every mount point on the root's tmpfs exists now; nothing more is
    # written there.
    args += ["--remount-ro", "/", "--chdir", workdir, "--"]
    _refuse_broad_binds(args, homes)
    return [
        "/bin/sh",
        "-c",
        _SECCOMP_EXEC,
        str(notices / "seccomp.bpf"),
        *args,
        *command,
    ]


_BIND_OPTIONS = ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try")


def _refuse_broad_binds(args: Sequence[str], homes: Sequence[Path]) -> None:
    """Refuse a command line that would mount ``/`` or a whole home directory.

    Raised for the policy's own mounts (a workspace configured as the home
    directory); derived roots that would are already left out.
    """
    end = args.index("--") if "--" in args else len(args)
    for i in range(end - 2):
        if args[i] in _BIND_OPTIONS:
            for p in (args[i + 1], args[i + 2]):
                if _too_broad(Path(p), homes):
                    raise SandboxRefusal(
                        "root-allowlist",
                        f"{p} would be mounted ({args[i]}), which would show / "
                        "or a whole home directory",
                        suggestion=(
                            "Point UNIFY_LOCAL_ROOT (or UNIFY_HOME) at a "
                            "directory of its own, not a home directory."
                        ),
                    )


def _notice(directory: Path) -> str:
    return str(directory / MASK_NOTICE_NAME)


def sandbox_env(
    policy: SandboxPolicy,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """The environment a sandboxed command gets."""
    out = scrubbed_env(env)
    # The fake clock's variables come from the harness only, as it has them.
    for name in TRUSTED_ENV:
        out.pop(name, None)
        if name in os.environ:
            out[name] = os.environ[name]
    out["TMPDIR"] = "/tmp"
    if policy.network == "proxy":
        url = f"http://127.0.0.1:{policy.proxy_port}"
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            out[name] = url
            out[name.lower()] = url
        out.pop("NO_PROXY", None)
        out.pop("no_proxy", None)
    return out


def annotate_refusals(output: str, policy: SandboxPolicy) -> Optional[str]:
    """A note naming the rule behind an OS error in *output*, if any."""
    notes = []
    if "Read-only file system" in output:
        notes.append(
            f"`workspace-write`: {RULES['workspace-write']} "
            f"(workspace: {policy.workspace})",
        )
    network_errors = (
        "Network is unreachable",
        "Could not resolve host",
        "Temporary failure in name resolution",
        "Name or service not known",
    )
    if any(e in output for e in network_errors):
        rule = "network-proxy-only" if policy.network == "proxy" else "network-off"
        notes.append(f"`{rule}`: {RULES[rule]}")
    if not notes:
        return None
    return "Refused by workspace sandbox rule " + "; rule ".join(notes)


# ---------------------------------------------------------------------------
# Subprocesses started by Python cells
# ---------------------------------------------------------------------------

_CONFINE: contextvars.ContextVar[Optional[SandboxPolicy]] = contextvars.ContextVar(
    "unify_sandbox_confine",
    default=None,
)
_PATCH_LOCK = threading.Lock()
_PATCHED = False


@contextmanager
def confined_subprocesses(policy: Optional[SandboxPolicy]) -> Iterator[None]:
    """Run every subprocess started in this context (and its tasks) under *policy*."""
    if policy is None:
        yield
        return
    _install_popen_patch()
    token = _CONFINE.set(policy)
    try:
        yield
    finally:
        _CONFINE.reset(token)


@contextmanager
def unconfined() -> Iterator[None]:
    """Harness-owned subprocesses (package installs) started inside a cell."""
    token = _CONFINE.set(None)
    try:
        yield
    finally:
        _CONFINE.reset(token)


def _install_popen_patch() -> None:
    """Wrap ``Popen.__init__`` and ``os.system`` once; inert outside a confined context.

    Every stdlib route to a child process that takes a command line --
    ``subprocess.run``/``call``/``check_output``, ``Popen``, ``os.popen`` and
    asyncio's subprocess transports -- constructs a ``Popen``.
    """
    global _PATCHED
    with _PATCH_LOCK:
        if _PATCHED:
            return
        import inspect

        original_init = subprocess.Popen.__init__
        signature = inspect.signature(original_init)

        def confined_init(self, *args, **kwargs):
            policy = _CONFINE.get()
            if policy is None:
                return original_init(self, *args, **kwargs)
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            command = a["args"]
            if a.get("shell"):
                shell = a.get("executable") or "/bin/sh"
                if isinstance(command, (str, bytes, os.PathLike)):
                    argv = [shell, "-c", os.fsdecode(command)]
                else:
                    argv = [shell, "-c", *map(os.fsdecode, command)]
            elif isinstance(command, (str, bytes, os.PathLike)):
                argv = [os.fsdecode(command)]
            else:
                argv = [os.fsdecode(c) for c in command]
            if not a.get("shell") and a.get("executable"):
                argv[0] = os.fsdecode(a["executable"])
            cwd = a.get("cwd")
            a["args"] = wrap_argv(
                argv,
                policy,
                cwd=os.fsdecode(cwd) if cwd is not None else None,
            )
            a["shell"] = False
            a["executable"] = None
            a["env"] = sandbox_env(policy, a.get("env"))
            return original_init(*bound.args, **bound.kwargs)

        original_system = os.system

        def confined_system(command):
            if _CONFINE.get() is None:
                return original_system(command)
            proc = subprocess.Popen(command, shell=True)
            code = proc.wait()
            # os.system reports a wait status, not an exit code.
            return code << 8 if code >= 0 else -code

        subprocess.Popen.__init__ = confined_init  # type: ignore[method-assign]
        os.system = confined_system
        _PATCHED = True


# ---------------------------------------------------------------------------
# Proxy bridge: host loopback port <- unix socket <- sandbox loopback port
# ---------------------------------------------------------------------------

# Runs inside the sandbox: listens on the sandbox's own loopback at the proxy
# port, relays each connection to the host bridge's unix socket, and runs the
# command as its child, exiting with its status.
_FORWARDER_SRC = r"""
import os, socket, subprocess, sys, threading
port, sock_path, argv = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", port))
srv.listen(64)
def pump(a, b):
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
def serve():
    while True:
        conn, _ = srv.accept()
        up = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            up.connect(sock_path)
        except OSError:
            conn.close()
            continue
        threading.Thread(target=pump, args=(conn, up), daemon=True).start()
        threading.Thread(target=pump, args=(up, conn), daemon=True).start()
threading.Thread(target=serve, daemon=True).start()
sys.exit(subprocess.call(argv))
"""


class _ProxyBridge:
    """Relays connections on a private unix socket to ``127.0.0.1:<port>``."""

    def __init__(self, port: int) -> None:
        if not 0 < port < 65536:
            raise SandboxRefusal(
                "network-proxy-only",
                f"UNIFY_WORKSPACE_PROXY_PORT must name the proxy's port, not {port}",
            )
        self.port = port
        self.directory = Path(tempfile.mkdtemp(prefix="unify-proxy-"))
        self.path = self.directory / "proxy.sock"
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(64)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            try:
                up = socket.create_connection(("127.0.0.1", self.port), timeout=10)
                up.settimeout(None)
            except OSError:
                conn.close()
                continue
            for a, b in ((conn, up), (up, conn)):
                threading.Thread(target=_pump, args=(a, b), daemon=True).start()


def _pump(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            data = a.recv(65536)
            if not data:
                break
            b.sendall(data)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


_BRIDGES: dict[int, _ProxyBridge] = {}


def _proxy_bridge(port: int) -> _ProxyBridge:
    with _PATCH_LOCK:
        bridge = _BRIDGES.get(port)
        if bridge is None or not bridge.path.exists():
            bridge = _BRIDGES[port] = _ProxyBridge(port)
        return bridge
