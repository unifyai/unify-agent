"""OS confinement for the shell and file access the actor gets (``UNIFY_WORKSPACE``).

With ``UNIFY_WORKSPACE=sandboxed`` every shell cell, and every subprocess a
Python cell starts, runs inside bubblewrap (Linux) under one policy:

* ``/`` is mounted read-only; the workspace (``<UNIFY_HOME>/workspace`` or
  ``UNIFY_LOCAL_ROOT``) and a private ``/tmp`` are the only writable places.
* The Unify state directory (``UNIFY_HOME``) is hidden behind an empty tmpfs,
  except read-only views of the transcripts directory, the store file and the
  package venv, and the writable workspace.
* Credential locations (``~/.ssh``, ``~/.config``, ``~/.aws``, ``~/.gnupg`` and
  a few other well-known ones) are hidden, and so is every ``.env`` file found
  in the working directory and its parents, the home directory and its
  non-hidden subdirectories two levels down, and the Unify checkout.
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
    "enabled",
    "scrubbed_env",
    "unconfined",
    "wrap_argv",
]

# Rule name -> what it does, as a refused action reports it.
RULES: dict[str, str] = {
    "sandbox-required": (
        "UNIFY_WORKSPACE=sandboxed runs shell commands only inside bubblewrap, "
        "never unconfined"
    ),
    "workspace-write": (
        "only the workspace and a private /tmp are writable; everything else is "
        "mounted read-only"
    ),
    "mask-unify-state": (
        "the Unify state directory is hidden, except read-only views of the "
        "transcripts, the store file and the package venv, and the workspace"
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


class SandboxRefusal(ToolInputError):
    """An action the workspace sandbox refuses; ``rule`` names the rule."""

    def __init__(self, rule: str, detail: str, *, suggestion: str | None = None):
        self.rule = rule
        super().__init__(
            f"Refused by workspace sandbox rule `{rule}` ({RULES[rule]}): {detail}",
            suggestion=suggestion,
        )


def enabled() -> bool:
    """Whether ``UNIFY_WORKSPACE=sandboxed``."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_WORKSPACE", "") == "sandboxed"


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
    network: str = ""  # "" (off) or "proxy"
    proxy_port: int = 0
    notices_dir: Optional[Path] = None
    created: float = field(default_factory=time.monotonic)

    # -- path checks ---------------------------------------------------------
    def readable_violation(self, path: Path) -> Optional[tuple[str, str]]:
        """``(rule, detail)`` when *path* (resolved) is outside the readable set."""
        resolved = Path(os.path.realpath(path))
        if _within(resolved, Path("/proc")):
            return "mask-proc", f"{resolved} is under /proc"
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
    directory.mkdir(parents=True, exist_ok=True)
    for rule, text in RULES.items():
        (directory / rule).write_text(
            f"Hidden by the Unify workspace sandbox (rule {rule}: {text}).\n",
        )
    return directory


_NOTICES_DIR: Optional[Path] = None
_POLICY_CACHE: Optional[SandboxPolicy] = None
_POLICY_LOCK = threading.Lock()


def _notices_dir() -> Path:
    global _NOTICES_DIR
    if _NOTICES_DIR is None or not _NOTICES_DIR.is_dir():
        _NOTICES_DIR = _write_notices(
            Path(tempfile.mkdtemp(prefix="unify-sandbox-notices-")),
        )
    return _NOTICES_DIR


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
        if getattr(SETTINGS, "UNIFY_TRANSCRIPTS", False):
            # Mounted only if present, and sessions pointed at it must find it.
            (state_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        readonly = [
            p
            for p in (
                state_dir / "transcripts",
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
        skip = [state_dir, workspace, *(d for d, _ in masked_dirs)]
        for env_file in _find_env_files(roots, skip):
            masked_files.append((Path(os.path.realpath(env_file)), "mask-env-file"))

        network = str(getattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "") or "")
        port = int(getattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0) or 0)
        policy = SandboxPolicy(
            workspace=workspace,
            state_dir=state_dir,
            readonly_state=readonly,
            masked_dirs=masked_dirs,
            masked_files=masked_files,
            network=network,
            proxy_port=port,
            notices_dir=_notices_dir(),
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
            suggestion=(
                "Install bubblewrap (apt install bubblewrap), or unset "
                "UNIFY_WORKSPACE to run without shell cells and file tools."
            ),
        )
    return path


def wrap_argv(
    argv: Sequence[str],
    policy: SandboxPolicy,
    *,
    cwd: Optional[str] = None,
) -> list[str]:
    """*argv* as a bubblewrap command line under *policy*."""
    bwrap = require_bwrap()
    notices = policy.notices_dir or _notices_dir()
    args: list[str] = [
        bwrap,
        "--ro-bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
    ]
    # Hide the state directory, then put back what may be seen.
    args += ["--tmpfs", str(policy.state_dir)]
    args += ["--ro-bind", str(notices / "mask-unify-state"), _notice(policy.state_dir)]
    for path, rule in policy.masked_dirs:
        args += ["--tmpfs", str(path), "--ro-bind", str(notices / rule), _notice(path)]
    for path, rule in policy.masked_files:
        args += ["--ro-bind", str(notices / rule), str(path)]
    # After every mask, so what may be seen is seen wherever it lives, and the
    # workspace last of all, so it is writable even inside a masked directory.
    for path in policy.readonly_state:
        args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(policy.workspace), str(policy.workspace)]
    command = list(argv)
    if policy.network == "proxy":
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
    if policy.readable_violation(Path(workdir)) is not None or not os.path.isdir(
        workdir,
    ):
        workdir = str(policy.workspace)
    args += ["--chdir", workdir, "--"]
    return args + command


def _notice(directory: Path) -> str:
    return str(directory / MASK_NOTICE_NAME)


def sandbox_env(
    policy: SandboxPolicy,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """The environment a sandboxed command gets."""
    out = scrubbed_env(env)
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
