"""OS confinement for the shell and file access the actor gets.

Every shell cell, every subprocess a Python cell starts, and the sandboxed
Python worker run inside bubblewrap (Linux) under one policy (the
``UNIFY_WORKSPACE=sandboxed`` switch until the code freeze baked it in):

* The root is an allowlist, read-only (:data:`_ROOT_ALLOWLIST`): the system
  directories (``/usr``, ``/bin``, ``/lib*``, ``/sbin``), a few files of
  ``/etc`` named one by one, and the roots taken from the harness process,
  each of an enumerated kind (:func:`_derived_candidates`): the interpreter's
  prefixes (its venv and base install, and the link names between them), the
  Unify package, the packages (never the checkout) of the editable installs
  the venv's ``.pth`` files add, the harness's ``PYTHONPATH`` entries, and
  the ``*.so`` files ``LD_PRELOAD`` names when the harness itself runs with
  them. Each is bound once, from its resolved path; a name that reaches it
  through a link is the same link inside. Nothing else of the host exists:
  no ``/home`` beyond those paths, ``/root``, ``/var``, ``/opt``, ``/mnt``,
  ``/workspaces``, ``/srv``, ``/media``, ``/run``. Never ``/``, a top-level
  directory, a whole home or its ``.local`` / ``.config`` / ``.cache``, nor
  the state or log directories (:func:`_root_refusal`, rule
  ``root-allowlist``). ``python -I -m unify.sandbox --print-roots`` lists
  what would be mounted and what is refused, with reasons.
* The workspace (``<UNIFY_HOME>/workspace`` or ``UNIFY_LOCAL_ROOT``) and a
  private ``/tmp`` are the only writable places.
* A seccomp filter (:func:`seccomp_program`) lets ``socket`` and
  ``socketpair`` create only AF_UNIX, AF_INET and AF_INET6 sockets (on WSL2
  AF_VSOCK reaches the Windows host whatever the network namespace), refuses
  new user namespaces and io_uring, and kills a process that enters the
  kernel through another architecture's system calls (rule
  ``socket-families``).
* The Unify state directory (``UNIFY_HOME``) is hidden behind an empty tmpfs,
  except read-only views of the transcripts directory (every session's) and
  the package venv, and the writable workspace; the store file is not mounted
  (cells reach the library through its API). The harness's internal
  transcripts (``internal-transcripts``: the storage review, whose prompt
  carries the environment's checked outcome) stay hidden even when a mount
  that is seen contains them (a workspace configured as ``UNIFY_HOME``).
* The harness's log directories (``UNILLM_LOG_DIR``, which holds every LLM
  request and reply, ``UNILLM_OTEL_LOG_DIR``, ``UNIFY_LOG_DIR`` and
  ``UNIFY_OTEL_LOG_DIR``, wherever they are configured) are hidden too.
* Credential locations (``~/.ssh``, ``~/.config``, ``~/.aws``, ``~/.gnupg`` and
  a few other well-known ones) are hidden behind a notice, and so is every
  ``.env`` file found in the working directory and its parents, the home
  directory and its non-hidden subdirectories two levels down and the Unify
  checkout, where a mount would show it (outside every mount it does not
  exist at all). Inside every derived root, at every depth, so are ``.env*``,
  ``*.pem`` (but public CA bundles), ``*key*.json`` and the credential names.
* The environment loses every variable whose name contains KEY, TOKEN, SECRET,
  PASSWORD or CREDENTIAL, or the word PAT, AUTH, PASSWD or PASS, and every
  value holding a URL with a password.
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
the installer's cache are writable, and its network namespace, private like
every other, has one route out: a loopback port forwarded to an
allow-listing CONNECT proxy the harness runs (:class:`EgressProxy`), which
opens tunnels only to the package index hosts and never to a loopback,
link-local or private address, whatever a name resolves to (rule
``installer-index-only``). The host's loopback services and the cloud
metadata server (``169.254.169.254``) are unreachable from it.

Without bubblewrap nothing runs: the harness refuses rather than run the
command unconfined. Python cells run in the sandboxed worker
(unify/actor/execution/worker.py), a persistent ``python -I -S`` started under
this policy, never in the harness's own process: a cell's code and every
subprocess it starts are confined alike.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import shutil
import socket
import stat
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
    "EgressProxy",
    "SandboxPolicy",
    "SandboxRefusal",
    "annotate_refusals",
    "build_policy",
    "check_readable",
    "confined_subprocesses",
    "egress_proxy",
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
        "transcripts and the package venv, and the workspace"
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
        "or CREDENTIAL, or the word PAT, AUTH, PASSWD or PASS, and values "
        "holding a URL with a password, are removed"
    ),
    "network-off": "the sandbox has no network, only a loopback of its own",
    "network-proxy-only": (
        "the only network is one loopback port forwarded to the configured proxy"
    ),
    "installer-index-only": (
        "the package installer has no network of its own; its one route out is "
        "the harness's proxy, which opens tunnels only to the package index "
        "hosts and never to a loopback, link-local or private address"
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
# The workspace's secret scan (:func:`_find_secret_files`): the directories it
# does not enter (a repository's objects, installed packages, bytecode: large,
# and the harness's file tools still refuse a secret name in them) and the most
# entries it reads in one scan (it reads only changed directories again,
# _scan_workspace), so a huge workspace costs a bounded walk per policy build.
_WORKSPACE_SCAN_SKIP = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__"},
)
_WORKSPACE_SCAN_LIMIT = 100_000
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


# Whole words of a variable's name (split at anything but letters and digits)
# that name a credential, beside SECRET_ENV_MARKERS' substrings: GITHUB_PAT,
# NPM_AUTH, MYSQL_PASSWD, REDIS_PASS. Words, so PATH and AUTHOR stay.
_SECRET_ENV_WORDS = frozenset({"PAT", "AUTH", "PASSWD", "PASS"})
# A URL with a password in it (a database URL with user and password, an
# index URL with a token as its password).
_URL_CREDENTIALS = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^/\s@:]*:[^/\s@]*@")


def _is_secret_env(name: str, value: str) -> bool:
    if is_secret_name(name):
        return True
    words = re.split(r"[^A-Z0-9]+", name.upper())
    if any(w in _SECRET_ENV_WORDS for w in words):
        return True
    return bool(_URL_CREDENTIALS.search(value or ""))


def scrubbed_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """*env* (default: this process's) without credential variables.

    Removed: names containing a :data:`SECRET_ENV_MARKERS` marker or one of
    :data:`_SECRET_ENV_WORDS` as a word, and any value holding a URL with a
    password (a ``DATABASE_URL`` with a user and password).
    """
    source = os.environ if env is None else env
    return {k: v for k, v in source.items() if not _is_secret_env(k, v)}


# What a sandboxed command (the Python worker that runs every cell, a bash
# cell, a cell's subprocess, the sandboxed grep) gets from the harness's
# environment: an allow-list, not the harness's environment minus a deny-list.
# On 7 Oct 2026 a model-written cell enumerated os.environ and printed a
# provider key; a key under a name no marker catches would pass a deny-list.
# HOME is the harness's (the sandbox shows that path, empty unless mounted);
# PYTHONPATH's entries are derived roots, checked before they are mounted.
# The benchmark adapters' relay clients read *_RELAY_SOCKET, *_VERIFY_SOCKET
# and *_RELAY_TIMEOUT in cell processes, and their sitecustomize reads
# UNIFY_ENV_NAMESPACES (a factory's module:attribute) to leave a registered
# namespace alone. TRUSTED_ENV, TMPDIR, USER/LOGNAME and the proxy variables
# are set by :func:`sandbox_env` itself.
CELL_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "TERM",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "PYTHONIOENCODING",
        "PYTHONDONTWRITEBYTECODE",
        "UNIFY_ENV_NAMESPACES",
    },
)
CELL_ENV_PATTERNS = ("LC_*", "*_RELAY_SOCKET", "*_VERIFY_SOCKET", "*_RELAY_TIMEOUT")

# A value that is a provider key or other credential whatever its name: an
# OpenAI/OpenRouter/Anthropic style ``sk-`` key, a Google API key, GitHub,
# Hugging Face, Slack, GitLab and AWS access key formats, a PEM private key.
_KEY_VALUE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-[A-Za-z0-9_-]{16,}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    r"|hf_[A-Za-z0-9]{20,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|glpat-[A-Za-z0-9_-]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r")|-----BEGIN [A-Z ]*PRIVATE KEY-----",
)


def _looks_like_key(value: str) -> bool:
    """A key-shaped value (:data:`_KEY_VALUE`) or a URL with a password."""
    return bool(_KEY_VALUE.search(value or "") or _URL_CREDENTIALS.search(value or ""))


def _env_pattern_matches(pattern: str, name: str) -> bool:
    if pattern.startswith("*"):
        return name.endswith(pattern[1:])
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def _cell_env_patterns() -> tuple[str, ...]:
    """:data:`CELL_ENV_PATTERNS` and ``UNIFY_CELL_ENV_ALLOW`` (names and
    ``PREFIX_*`` patterns a runner declares for its own cell variables)."""
    from unify.settings import SETTINGS

    declared = str(getattr(SETTINGS, "UNIFY_CELL_ENV_ALLOW", "") or "")
    return (*CELL_ENV_PATTERNS, *(p.strip() for p in declared.split(",") if p.strip()))


def allowed_env(env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The allow-listed variables of *env* (default: this process's).

    Even an allow-listed variable is left out when its name is a credential's
    (:func:`_is_secret_env`) or its value looks like a key or holds a URL with
    a password (:func:`_looks_like_key`): the deny-list is a second filter.
    """
    source = os.environ if env is None else env
    patterns = _cell_env_patterns()
    return {
        k: v
        for k, v in source.items()
        if (k in CELL_ENV_NAMES or any(_env_pattern_matches(p, k) for p in patterns))
        and not _is_secret_env(k, v)
        and not _looks_like_key(v)
    }


def _account_name() -> Optional[str]:
    """This account's name as the sandbox's generated ``/etc/passwd`` lists it."""
    import pwd

    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return None


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
    # Inside the workspace, masked after its bind (a file with a notice, a
    # directory with an empty tmpfs): the secret scan's finds there and the
    # store's files (store.sqlite, -wal, -shm) when the workspace holds them.
    workspace_masked: list[tuple[Path, str]] = field(default_factory=list)
    # The workspace's secret scan (_scan_workspace), and the warning written
    # to the log and the run's record when it stopped at its cap.
    workspace_scan: Optional["WorkspaceScan"] = None
    scan_note: Optional[str] = None
    network: str = ""  # "" (off) or "proxy"
    proxy_port: int = 0
    notices_dir: Optional[Path] = None
    # The allowlisted root (_root_mounts): bubblewrap arguments, and the host
    # paths (resolved) they show, which the harness's own file tools check.
    root_args: list[str] = field(default_factory=list)
    root_visible: list[Path] = field(default_factory=list)
    created: float = field(default_factory=time.monotonic)
    # The raw store's files (store.sqlite, -wal, -shm) at store_path(), and
    # by name in the state directory: refused to the harness's file tools
    # wherever they are, whatever mount would otherwise show them.
    store_files: list[Path] = field(default_factory=list)

    # -- path checks ---------------------------------------------------------
    def readable_violation(self, path: Path) -> Optional[tuple[str, str]]:
        """``(rule, detail)`` when *path* (resolved) is outside the readable set."""
        resolved = Path(os.path.realpath(path))
        if _within(resolved, Path("/proc")):
            return "mask-proc", f"{resolved} is under /proc"
        if self.workspace == self.state_dir:
            # wrap_argv refuses this workspace (_workspace_refusal); the
            # harness's file tools refuse it too, so nothing is read through
            # a policy no sandbox would run.
            return (
                "root-allowlist",
                f"the workspace {self.workspace} is the Unify state directory",
            )
        if resolved in self.store_files:
            return "mask-unify-state", f"{resolved} is the Unify store"
        seen = (self.workspace, *self.readonly_state)
        for hidden, rule in self.hidden:
            if _within(resolved, hidden) and not any(
                _within(resolved, v) and _within(v, hidden) and v != hidden
                for v in seen
            ):
                return rule, f"{resolved} is inside the harness's {hidden}"
        if _within(resolved, self.workspace):
            for masked, rule in self.workspace_masked:
                if _within(resolved, masked):
                    return rule, f"{resolved} is hidden"
            # A secret-named file made since the policy was built, or in a
            # directory the scan does not enter, is refused by name.
            parts = resolved.relative_to(self.workspace).parts
            for i, part in enumerate(parts):
                rule = _secret_rule(part)
                if rule is not None and (i == len(parts) - 1 or not _is_env_file(part)):
                    return rule, f"{resolved} has a credential's name"
            # A private key under any name, read again on every request (the
            # scan's finds can be older than the file's content).
            if _holds_private_key(resolved):
                return "mask-credentials", f"{resolved} holds a private key"
            # A state directory inside the workspace is hidden again over the
            # workspace's mount, but for the views put back.
            if (
                self.state_dir != self.workspace
                and _within(self.state_dir, self.workspace)
                and _within(resolved, self.state_dir)
                and not any(_within(resolved, v) for v in self.readonly_state)
            ):
                return (
                    "mask-unify-state",
                    f"{resolved} is inside the Unify state directory {self.state_dir}",
                )
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


def _store_files(store: Path, state_dir: Path) -> list[Path]:
    """The raw store's files: at *store* (``store_path()``, resolved) and, by
    their default name, in *state_dir*."""
    out: list[Path] = []
    for base in (store, state_dir / "store.sqlite"):
        for name in (base.name, f"{base.name}-wal", f"{base.name}-shm"):
            path = Path(os.path.realpath(base.parent / name))
            if path not in out:
                out.append(path)
    return out


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_env_file(name: str) -> bool:
    """``.env``, ``.env.local``, ``.envrc``, ...: every ``.env*`` but the templates."""
    return name.startswith(".env") and name not in _ENV_FILE_ALLOWED


# Public CA bundles: certificates only, and TLS needs them (certifi, grpc).
_PUBLIC_PEM = {"cacert.pem", "roots.pem", "ca-bundle.pem", "ca-certificates.pem"}
_CREDENTIAL_NAMES = {Path(p).name for p in CREDENTIAL_PATHS}


# Private keys and secret stores by their own name: OpenSSH's default key
# files (and any ``id_*`` that starts with one of these stems but for ``.pub``
# public halves), and the suffixes of key, keystore and password-database
# files. ``.netrc``'s Windows name. Public halves (``*.pub``), ``known_hosts``
# and ``authorized_keys`` never match.
_SSH_KEY_STEMS = (
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_ecdsa_sk",
    "id_ed25519_sk",
)
_SECRET_SUFFIXES = (".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk", ".kdbx")
_SECRET_FILE_NAMES = {"_netrc"}


def _secret_rule(name: str) -> Optional[str]:
    """The mask rule for a file or directory named *name* in a mounted root."""
    if _is_env_file(name):
        return "mask-env-file"
    lower = name.lower()
    if name in _CREDENTIAL_NAMES or lower in _SECRET_FILE_NAMES:
        return "mask-credentials"
    if lower.endswith(".pem") and lower not in _PUBLIC_PEM:
        return "mask-credentials"
    if lower.endswith(".json") and "key" in lower:
        return "mask-credentials"
    if lower.endswith(".pub"):
        return None
    if lower.startswith(_SSH_KEY_STEMS) or lower.endswith(_SECRET_SUFFIXES):
        return "mask-credentials"
    return None


# A workspace file is also masked by its content: a regular file of at most
# _ARMOUR_MAX_SIZE bytes whose first _ARMOUR_READ bytes hold a private key's
# PEM or PGP armour (PKCS#1/#8, OpenSSH, SEC1, DSA, encrypted PKCS#8, a PGP
# secret key block; a JSON service-account key holds it too). Public-key and
# certificate armour never matches.
_ARMOUR_MAX_SIZE = 64 * 1024
_ARMOUR_READ = 4096
# The shortest armour line: a smaller file cannot hold one and is never read.
_ARMOUR_MIN_SIZE = len(b"-----BEGIN PRIVATE KEY-----")  # pragma: allowlist secret
_PRIVATE_ARMOUR = re.compile(
    rb"-----BEGIN (?:(?:OPENSSH|RSA|EC|DSA|ENCRYPTED) )?PRIVATE KEY-----"  # pragma: allowlist secret
    rb"|-----BEGIN PGP PRIVATE KEY BLOCK-----",  # pragma: allowlist secret
)


def _holds_private_key(path: str | os.PathLike) -> bool:
    """Whether *path* is a regular file of at most :data:`_ARMOUR_MAX_SIZE`
    bytes whose first :data:`_ARMOUR_READ` bytes hold private-key armour.

    Opened with ``O_NOFOLLOW`` (a link is never followed: its target is
    checked by its own path) and ``O_NONBLOCK`` (a FIFO swapped in never
    blocks), only when it is a regular file by ``lstat`` of a size that can
    hold armour, then checked again on the open descriptor. A file that
    cannot be opened or read is not masked by content.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_NOCTTY", 0)
    try:
        st = os.lstat(path)
        # Never opened unless it is a small regular file now (a device is
        # never opened); checked again on the descriptor.
        if not (
            stat.S_ISREG(st.st_mode)
            and _ARMOUR_MIN_SIZE <= st.st_size <= _ARMOUR_MAX_SIZE
        ):
            return False
        fd = os.open(path, flags)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _ARMOUR_MAX_SIZE:
            return False
        head = os.read(fd, _ARMOUR_READ)
    except OSError:
        return False
    finally:
        os.close(fd)
    return _PRIVATE_ARMOUR.search(head) is not None


# The interpreter roots' scan (_find_secret_files with ``cached``): about 60k
# entries, a second or so, the same in every harness process until a package
# is installed. It is kept in this process (_SECRET_SCAN_CACHE) and on disk
# (_scan_cache_dir), keyed on the root and its fingerprint (_root_fingerprint).
_SECRET_SCAN_CACHE: dict[Path, tuple[str, list, list]] = {}
# Bump when the scan's matching or walking rules change (_secret_rule,
# _secret_entries): every cache entry written under another version is stale.
# The rule tables themselves (_CREDENTIAL_NAMES, _PUBLIC_PEM,
# _ENV_FILE_ALLOWED, _SSH_KEY_STEMS, _SECRET_SUFFIXES, _SECRET_FILE_NAMES)
# are part of the key as they are. Version 2: private keys by name.
_ROOT_SCAN_RULES_VERSION = 2
_ROOT_SCAN_SCHEMA = 2
# How deep below the root and below each of its site-packages the fingerprint
# lists directories: a new entry at depth 1 to 3 below either changes it.
_ROOT_SCAN_FINGERPRINT_DEPTH = 2
# A disk entry is rescanned after a day whatever its fingerprint, which bounds
# the residual (an entry made deeper than the fingerprint) to a day.
_ROOT_SCAN_MAX_AGE_S = 86_400.0
# A fingerprinted directory changed this recently may change again within the
# file system's timestamp granularity without moving it: not cached.
_ROOT_SCAN_SETTLE_NS = 2_000_000_000
_SCAN_RULES = frozenset({"mask-env-file", "mask-credentials"})


def _rules_digest() -> str:
    """The scan's rules as data: its version and the name tables it matches."""
    data = [
        _ROOT_SCAN_RULES_VERSION,
        sorted(_CREDENTIAL_NAMES),
        sorted(_PUBLIC_PEM),
        sorted(_ENV_FILE_ALLOWED),
        sorted(_SSH_KEY_STEMS),
        sorted(_SECRET_SUFFIXES),
        sorted(_SECRET_FILE_NAMES),
    ]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def _root_fingerprint(root: Path, skip: Sequence[Path]) -> tuple[str, int]:
    """``(digest, newest change in ns)`` of what can add a secret file to *root*.

    The digest covers the scan's rules (:func:`_rules_digest`), the root's
    name, the *skip* paths that meet it (they change what the walk enters),
    and, for the root and every directory down to
    :data:`_ROOT_SCAN_FINGERPRINT_DEPTH` below it and below each of its
    ``lib/python3*/site-packages``, the directory's device, inode, mtime and
    ctime and the sorted names of all its entries with each entry's inode,
    kind and size (links are not followed; no file is read). An entry made,
    removed or renamed at depth 1 to 3 below either base changes a listed
    name whatever the clock: a directory's mtime alone does not move when the
    entry is made within the file system's timestamp tick (seen on ext4).
    """
    h = hashlib.sha256()
    meeting = sorted(str(s) for s in skip if _within(s, root) or _within(root, s))
    h.update(json.dumps([_rules_digest(), str(root), meeting]).encode())
    newest = 0
    bases = [root, *sorted(root.glob("lib/python3*/site-packages"))]
    for base in bases:
        try:
            st = os.stat(base)
        except OSError:
            h.update(f"{base}\0missing\n".encode())
            continue
        level = [(base, st)]
        for depth in range(_ROOT_SCAN_FINGERPRINT_DEPTH + 1):
            below = []
            for directory, st in level:
                h.update(
                    f"{directory}\0{st.st_dev}\0{st.st_ino}\0{st.st_mtime_ns}"
                    f"\0{st.st_ctime_ns}\n".encode(),
                )
                newest = max(newest, st.st_mtime_ns, st.st_ctime_ns)
                try:
                    entries = sorted(os.scandir(directory), key=lambda e: e.name)
                except OSError:
                    h.update(f"{directory}\0unreadable\n".encode())
                    continue
                for entry in entries:
                    try:
                        est = entry.stat(follow_symlinks=False)
                    except OSError:
                        h.update(f"{entry.path}\0unreadable\n".encode())
                        continue
                    kind = stat.S_IFMT(est.st_mode)
                    h.update(
                        f"\1{entry.name}\0{est.st_ino}\0{kind}\0{est.st_size}\n".encode(
                            "utf-8",
                            "surrogateescape",
                        ),
                    )
                    if depth < _ROOT_SCAN_FINGERPRINT_DEPTH and stat.S_ISDIR(
                        est.st_mode,
                    ):
                        below.append((Path(entry.path), est))
            level = below
    return h.hexdigest(), newest


def _scan_cache_dir(avoid: Sequence[Path] = ()) -> Optional[Path]:
    """Where the interpreter roots' scans are kept across processes, or ``None``.

    ``$XDG_CACHE_HOME/unify/sandbox-scans`` (an absolute ``XDG_CACHE_HOME``
    only), else ``~/.cache/unify/sandbox-scans``. No cell can write it: cells
    write only the workspace, a private ``/tmp`` and (the installer) the
    package venv and its cache, and a home's ``.cache`` is never mounted
    (:func:`_root_refusal` refuses a root that is or holds it, and
    :func:`_workspace_refusal` a workspace that is or holds it). ``None``, so
    the scan is kept in memory only, when the directory meets one of *avoid*
    (the workspace, the state and log directories, the scanned roots): an
    unusual ``XDG_CACHE_HOME`` inside something a cell can see or write.
    """
    raw = os.environ.get("XDG_CACHE_HOME", "")
    base = Path(raw) if raw and os.path.isabs(raw) else Path.home() / ".cache"
    path = Path(os.path.realpath(base / "unify" / "sandbox-scans"))
    for a in avoid:
        a = Path(os.path.realpath(a))
        if _within(path, a) or _within(a, path):
            return None
    return path


def _scan_cache_file(cache_dir: Path, root: Path) -> Path:
    return cache_dir / (hashlib.sha256(str(root).encode()).hexdigest()[:32] + ".json")


def _scan_cache_load(
    cache_dir: Path,
    root: Path,
    key: str,
) -> Optional[tuple[list, list]]:
    """The cached ``(files, dirs)`` of *root* under *key*, ``None`` if absent,
    unreadable, malformed, of another root or key, older than
    :data:`_ROOT_SCAN_MAX_AGE_S`, not this user's own mode-0600 file, or
    naming a found path that is gone."""
    path = _scan_cache_file(cache_dir, root)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None
    try:
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            st = os.fstat(fh.fileno())
            if not (
                path.parent.stat().st_uid == os.getuid() == st.st_uid
                and st.st_mode & 0o7777 == 0o600
                and (st.st_mode & 0o170000) == 0o100000
            ):
                return None
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    if not (
        isinstance(data, dict)
        and set(data) == {"schema", "root", "key", "created", "files", "dirs"}
        and data["schema"] == _ROOT_SCAN_SCHEMA
        and data["root"] == str(root)
        and data["key"] == key
        and isinstance(data["created"], (int, float))
        and 0 <= time.time() - data["created"] <= _ROOT_SCAN_MAX_AGE_S
    ):
        return None
    out: list[list[tuple[Path, str]]] = [[], []]
    for i, name, exists in ((0, "files", os.path.isfile), (1, "dirs", os.path.isdir)):
        entries = data[name]
        if not isinstance(entries, list):
            return None
        for item in entries:
            if not (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and os.path.isabs(item[0])
                and item[1] in _SCAN_RULES
                and exists(item[0])
            ):
                return None
            out[i].append((Path(item[0]), item[1]))
    return out[0], out[1]


def _scan_cache_store(
    cache_dir: Path,
    root: Path,
    key: str,
    files: list[tuple[Path, str]],
    dirs: list[tuple[Path, str]],
) -> None:
    """Write *root*'s scan atomically (a temporary file, then ``os.replace``),
    mode 0600 in a mode-0700 directory of this user's. A failure is logged at
    debug and leaves the in-memory cache only."""
    log = logging.getLogger(__name__)
    payload = {
        "schema": _ROOT_SCAN_SCHEMA,
        "root": str(root),
        "key": key,
        "created": time.time(),
        "files": [[str(p), r] for p, r in files],
        "dirs": [[str(p), r] for p, r in dirs],
    }
    tmp = None
    try:
        cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(cache_dir)
        if (st.st_mode & 0o170000) != 0o040000 or st.st_uid != os.getuid():
            log.debug("sandbox scan cache: %s is not this user's directory", cache_dir)
            return
        if st.st_mode & 0o077:
            os.chmod(cache_dir, 0o700)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=".scan-", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o600)
            json.dump(payload, fh)
        os.replace(tmp, _scan_cache_file(cache_dir, root))
        tmp = None
    except OSError as exc:
        log.debug("sandbox scan cache: not written to %s: %s", cache_dir, exc)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _secret_entries(
    directory: Path,
    skip_names: frozenset[str] = frozenset(),
    *,
    content: bool = False,
) -> Optional[tuple[list, list, list, int]]:
    """One directory's ``(files, dirs, subdirectories, entries)`` for the scan.

    ``files`` and ``dirs`` are what to mask there, with rules (a link of a
    secret name masks its target; a directory of a :data:`CREDENTIAL_PATHS`
    name is masked whole and not entered); ``subdirectories`` are the ones to
    enter (not those named in *skip_names*). ``None`` if it cannot be read.

    With *content* (the workspace scan only, never the interpreter roots',
    whose packages carry test keys), a regular file of another name that
    :func:`_holds_private_key` is masked too (``mask-credentials``); each
    file read (one of a size that can hold armour) counts once more in
    ``entries``, so the scan's cap bounds the reads as well.
    """
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return None
    files: list[tuple[Path, str]] = []
    dirs: list[tuple[Path, str]] = []
    subdirs: list[Path] = []
    reads = 0
    for entry in entries:
        rule = _secret_rule(entry.name)
        try:
            if entry.is_symlink():
                if rule is not None:
                    target = Path(os.path.realpath(entry.path))
                    if target.is_file():
                        files.append((target, rule))
                    elif target.is_dir():
                        dirs.append((target, rule))
            elif entry.is_dir():
                if rule is not None and not _is_env_file(entry.name):
                    dirs.append((Path(entry.path), rule))
                elif entry.name not in skip_names:
                    subdirs.append(Path(entry.path))
            elif rule is not None and entry.is_file():
                files.append((Path(entry.path), rule))
            elif (
                content
                and entry.is_file(follow_symlinks=False)
                and _ARMOUR_MIN_SIZE
                <= entry.stat(follow_symlinks=False).st_size
                <= _ARMOUR_MAX_SIZE
            ):
                reads += 1
                if _holds_private_key(entry.path):
                    files.append((Path(entry.path), "mask-credentials"))
        except OSError:
            continue
    return files, dirs, subdirs, len(entries) + reads


def _walk_secret_files(
    root: Path,
    skip: Sequence[Path],
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """``(files, dirs)`` to mask under *root*, read now, at every depth."""
    f_out: list[tuple[Path, str]] = []
    d_out: list[tuple[Path, str]] = []
    stack = [root]
    while stack:
        directory = stack.pop()
        if any(_within(directory, s) for s in skip):
            continue
        found = _secret_entries(directory)
        if found is None:
            continue
        f_out += found[0]
        d_out += found[1]
        stack += found[2]
    return f_out, d_out


def _find_secret_files(
    roots: Sequence[Path],
    skip: Sequence[Path],
    *,
    cached: Sequence[Path] = (),
    cache_dir: Optional[Path] = None,
) -> tuple[list[tuple[Path, str]], list[tuple[Path, str]]]:
    """``(files, dirs)`` to mask, with rules, anywhere under each mounted root.

    Every depth, hidden directories included: ``.env*`` files, ``*.pem``
    other than public CA bundles, ``*key*.json``, the names of
    :data:`CREDENTIAL_PATHS` (a directory of those names is masked whole and
    not entered), OpenSSH private keys (``id_rsa``, ``id_ed25519``, ... but
    ``*.pub``), ``_netrc`` and the key and keystore suffixes
    (:data:`_SECRET_SUFFIXES`). A link of such a name masks its target. No
    file is read here (the workspace scan alone checks content).

    The scan of a root in *cached* (the interpreter's, about 60k entries) is
    reused while its fingerprint (:func:`_root_fingerprint`) is unchanged: in
    this process, and across processes from *cache_dir*
    (:func:`_scan_cache_dir`) when one is given, so it runs once per venv
    rather than once per harness process. A disk entry is used only if its
    schema, root, key and age check out, it is this user's mode-0600 file and
    every path it names still exists; anything else rescans and rewrites it.
    A cache that cannot be written leaves the in-memory one; masking never
    depends on it.

    Residual: a secret-named file made deeper than the fingerprint (below a
    directory at depth 2 under the root or a site-packages, e.g.
    ``site-packages/pkg/sub/deep/x.pem``) in a tree that is otherwise
    unchanged is not masked until the fingerprint moves or the disk entry is
    a day old. Installers make a ``.dist-info`` at site-packages' top level, so
    an install refreshes it; nothing a cell runs can write these roots.
    """
    files: list[tuple[Path, str]] = []
    dirs: list[tuple[Path, str]] = []
    for root in roots:
        if root not in cached:
            f_out, d_out = _walk_secret_files(root, skip)
            files += f_out
            dirs += d_out
            continue
        key, newest = _root_fingerprint(root, skip)
        hit = _SECRET_SCAN_CACHE.get(root)
        if hit is not None and hit[0] == key:
            files += hit[1]
            dirs += hit[2]
            continue
        loaded = (
            _scan_cache_load(cache_dir, root, key) if cache_dir is not None else None
        )
        if loaded is not None:
            f_out, d_out = loaded
        else:
            f_out, d_out = _walk_secret_files(root, skip)
        if time.time_ns() - newest > _ROOT_SCAN_SETTLE_NS:
            _SECRET_SCAN_CACHE[root] = (key, f_out, d_out)
            if loaded is None and cache_dir is not None:
                _scan_cache_store(cache_dir, root, key, f_out, d_out)
        files += f_out
        dirs += d_out
    return files, dirs


@dataclass
class _ScannedDir:
    """One workspace directory's scan, valid while its mtime is unchanged."""

    mtime_ns: int
    files: list
    dirs: list
    subdirs: list


# Per workspace, for the life of the harness process (its session): each
# directory's last scan. A directory's mtime changes when an entry in it is
# made, removed or renamed, which is all the scan looks at (names), so an
# unchanged one is reused and only changed directories are read again.
_WORKSPACE_SCANS: dict[Path, dict[Path, _ScannedDir]] = {}
# A directory changed this recently is read again next time, whatever its
# mtime says: a change within the file system's timestamp granularity would
# not move it.
_WORKSPACE_SCAN_SETTLE_NS = 2_000_000_000


@dataclass
class WorkspaceScan:
    """What the workspace's secret scan did: directories visited, read again
    and reused, the entries read, and whether it stopped at the cap."""

    files: list
    dirs: list
    visited: int = 0
    rescanned: int = 0
    reused: int = 0
    entries: int = 0
    capped: bool = False


def _scan_workspace(
    root: Path,
    skip: Sequence[Path],
    *,
    limit: Optional[int] = None,
) -> WorkspaceScan:
    """The secret files and directories to mask in the workspace, incrementally.

    Every directory is stat'ed; only those whose mtime changed since the last
    scan (or that are new, or changed within the settle window) are read
    again, so the *limit* on entries read per scan rarely binds. Directories
    named in :data:`_WORKSPACE_SCAN_SKIP` and paths in *skip* are not entered.
    Past *limit*, unchanged directories are still reused but changed ones are
    not read (``capped``): their secrets are not masked in cells, and the
    harness's file tools refuse them by name.

    Files are also masked by content (private-key armour in a small regular
    file, :func:`_holds_private_key`), read when their directory is read.
    Residual: a directory's mtime moves when an entry is made, removed or
    renamed, not when a file's content is rewritten in place, so armour
    written into an existing file of an unchanged directory is not masked in
    cells until that directory changes (or the harness process restarts);
    the harness's file tools read the file again on every request and refuse
    it.
    """
    if limit is None:
        limit = _WORKSPACE_SCAN_LIMIT
    cache = _WORKSPACE_SCANS.get(root, {})
    kept: dict[Path, _ScannedDir] = {}
    out = WorkspaceScan(files=[], dirs=[])
    now = time.time_ns()
    stack = [root]
    while stack:
        directory = stack.pop()
        if any(_within(directory, s) for s in skip):
            continue
        try:
            mtime = os.stat(directory).st_mtime_ns
        except OSError:
            continue
        out.visited += 1
        hit = cache.get(directory)
        if hit is not None and hit.mtime_ns == mtime:
            out.reused += 1
            scanned = hit
        elif out.entries >= limit:
            out.capped = True
            continue
        else:
            found = _secret_entries(directory, _WORKSPACE_SCAN_SKIP, content=True)
            if found is None:
                continue
            out.rescanned += 1
            out.entries += found[3]
            scanned = _ScannedDir(mtime, found[0], found[1], found[2])
        if now - mtime > _WORKSPACE_SCAN_SETTLE_NS:
            kept[directory] = scanned
        out.files += scanned.files
        out.dirs += scanned.dirs
        stack += scanned.subdirs
    if out.capped:
        # What was not reached keeps its last scan for next time.
        kept = {**cache, **kept}
    _WORKSPACE_SCANS[root] = kept
    return out


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


def _pwd_home() -> Path:
    """The account's home from the password database, never ``$HOME``.

    ``$HOME`` may point elsewhere (the tests move it); both count as homes
    (:func:`_homes`), but the ``/home`` rule of :func:`_root_refusal` keys on
    this one.
    """
    import pwd

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _homes() -> list[Path]:
    """The account's home and ``$HOME``, resolved: never mounted whole.

    A home under ``/tmp`` is the sandbox's private /tmp inside, so it is not
    counted.
    """
    out: list[Path] = []
    for path in (_pwd_home(), Path.home()):
        real = Path(os.path.realpath(path))
        if not _within(real, Path("/tmp")) and real not in out:
            out.append(real)
    return out


def _too_broad(path: Path, homes: Sequence[Path]) -> bool:
    """Whether a mount of *path* would show ``/``, ``/home`` or a whole home."""
    for p in (Path(os.path.abspath(path)), Path(os.path.realpath(path))):
        if p == Path("/") or any(_within(h, p) for h in (*homes, Path("/home"))):
            return True
    return False


# Host directories no derived root or policy mount may be, or contain: the
# rest of the machine (other users, mounted drives, the system's state and
# configuration, devices).
_DENIED_ANCESTORS = (
    "/workspaces",
    "/mnt",
    "/home",
    "/root",
    "/opt",
    "/var",
    "/srv",
    "/media",
    "/run",
    "/etc",
    "/sys",
    "/proc",
    "/dev",
    "/boot",
)
# Under each home: where keyrings, tool credentials (uv, gh, gcloud), other
# programs' data and caches live. A mount may sit inside them (the uv Python
# under ``~/.local/share/uv/python``), never be or contain them.
_HOME_PRIVATE = (".local", ".local/share", ".config", ".cache")


def _root_refusal(path: Path, guarded: Sequence[Path] = ()) -> Optional[str]:
    """Why a mount of *path* (by its own name or its resolved one) is refused.

    ``None`` when it may be mounted. Refused: ``/`` and every top-level
    directory; anything that is or contains a home, ``/home``, the other
    directories of :data:`_DENIED_ANCESTORS` or a home's
    :data:`_HOME_PRIVATE` directories; anything in ``/home`` outside the
    account's own home; and anything that is or contains one of *guarded*
    (the state directory, the log directories).
    """
    homes = _homes()
    account = Path(os.path.realpath(_pwd_home()))
    for p in dict.fromkeys((Path(os.path.abspath(path)), Path(os.path.realpath(path)))):
        if p == Path("/"):
            return "it is /"
        if len(p.parts) == 2:
            return f"{p} is a top-level directory"
        if len(p.parts) == 3 and p.parts[1] in ("mnt", "media"):
            return f"{p} is a mounted drive's root"
        for home in homes:
            if _within(home, p):
                return f"it would show the whole home directory {home}"
            for rel in _HOME_PRIVATE:
                if _within(home / rel, p):
                    return f"it would show all of {home / rel}"
        for d in _DENIED_ANCESTORS:
            if _within(Path(d), p):
                return f"it would show all of {d}"
        if _within(p, Path("/home")) and not _within(p, account):
            return f"{p} is in /home outside this account's home"
        for g in guarded:
            if _within(g, p):
                return f"it would show the harness's {g}"
    return None


# The system's own directories: never the workspace. The first group may not
# hold it either (the host's configuration, the kernel's and devices' views).
_WORKSPACE_SYSTEM_DIRS = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/boot",
    "/usr",
    "/var",
    "/root",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
)
_WORKSPACE_SYSTEM_ANCESTORS = ("/etc", "/proc", "/sys", "/dev", "/boot")
# Under each home: what a workspace may not be or contain, beside the
# CREDENTIAL_PATHS (.ssh, .config, .aws, ...): caches and the keyrings.
_WORKSPACE_HOME_GUARDED = (".cache", ".local/share/keyrings", ".config/gcloud")


def _workspace_guarded(homes: Sequence[Path]) -> list[Path]:
    """What the workspace may not be or contain: each home's configuration,
    cache and credential directories (by their own and their resolved names;
    they need not exist). Log directories may be inside it: they are hidden
    after its bind (``mask-harness-logs``)."""
    out: list[Path] = []
    for home in homes:
        for rel in (*CREDENTIAL_PATHS, *_WORKSPACE_HOME_GUARDED):
            out.append(home / rel)
    return list(
        dict.fromkeys(q for p in out for q in (p, Path(os.path.realpath(p)))),
    )


def _workspace_refusal(
    path: Path,
    *,
    homes: Optional[Sequence[Path]] = None,
    guarded: Optional[Sequence[Path]] = None,
    state: Optional[Path] = None,
    code: Optional[Sequence[tuple[Path, bool]]] = None,
) -> Optional[str]:
    """Why the read-write workspace bind of *path* is refused, ``None`` if not.

    The workspace is its own kind: the user's choice of where the work is, so
    a top-level directory (``/data``, ``/srv/project``), one that strictly
    holds ``UNIFY_HOME`` (the state directory is hidden inside it, whole,
    :func:`wrap_argv`) or a log directory (hidden too) is allowed. Refused
    only: ``/``; a home itself (the account's and ``$HOME``); the state
    directory itself (*state*, default ``UNIFY_HOME``: its store and records
    would be the workspace); a system directory
    (:data:`_WORKSPACE_SYSTEM_DIRS`), or a path inside the first group of
    them; a path that is or contains a home's configuration, cache or
    credential directory (:func:`_workspace_guarded`); and a path that is or
    contains the harness's own code, or is inside the part of it marked so
    (*code*, default :func:`_workspace_code_roots`).
    """
    if homes is None:
        homes = list(
            dict.fromkeys(
                Path(os.path.realpath(h)) for h in (_pwd_home(), Path.home())
            ),
        )
    if guarded is None:
        guarded = _workspace_guarded(homes)
    if state is None:
        from unify.db import store_home

        state = store_home()
    if code is None:
        code = _workspace_code_roots()
    states = {Path(os.path.abspath(state)), Path(os.path.realpath(state))}
    for p in dict.fromkeys((Path(os.path.abspath(path)), Path(os.path.realpath(path)))):
        if p == Path("/"):
            return "it is /"
        if p in states:
            return f"it is the Unify state directory {state} (UNIFY_HOME)"
        for home in homes:
            if p == home:
                return f"it is the home directory {home}"
        for d in _WORKSPACE_SYSTEM_DIRS:
            for s in dict.fromkeys((Path(d), Path(os.path.realpath(d)))):
                if p == s:
                    return f"it is the system directory {d}"
                if d in _WORKSPACE_SYSTEM_ANCESTORS and _within(p, s):
                    return f"it is inside the system directory {d}"
        for g in guarded:
            if _within(g, p):
                return f"it would show {g}"
        for root, inside_too in code:
            if _within(root, p):
                return f"it would make the harness's code {root} writable"
            if inside_too and _within(p, root):
                return f"it is inside the harness's code {root}"
    return None


def _workspace_code_roots() -> list[tuple[Path, bool]]:
    """The harness's code a workspace may not be or hold, each by both names.

    The workspace is bound read-write after every other mount, so a
    workspace holding the Unify package, the interpreter's prefixes or an
    editable root would make that code writable to cells. One above the
    package would also let a cell plant a ``.env`` that the CLI's
    ``load_dotenv()`` (which walks up from the package) loads into the
    harness. ``True``: a workspace inside it is refused too (the package and
    the prefixes, and each editable root's packages); an editable root itself
    is often a checkout whose other directories are the user's own.
    """
    out: list[tuple[Path, bool]] = [(Path(__file__).resolve().parent, True)]
    out += [(p, True) for p in _interpreter_prefixes()]
    for root in _editable_roots():
        out.append((root, False))
        out += [(p, True) for p in _editable_packages(root)]
    return list(
        dict.fromkeys(
            (q, inside_too)
            for p, inside_too in out
            for q in (Path(os.path.abspath(p)), Path(os.path.realpath(p)))
        ),
    )


@dataclass(frozen=True)
class DerivedRoot:
    """A candidate for the allowlisted root taken from this process.

    ``kind`` is one of: ``interpreter`` (the venv, the base install and the
    names of the link chain between them), ``unify-package``,
    ``editable-root`` (a ``.pth`` entry: an empty directory, so ``sys.path``
    resolves), ``editable-package`` (a package or module at an editable
    root's top level), ``pythonpath`` and ``preload``. ``refusal`` says why
    it is left out, ``None`` when it is mounted.
    """

    path: Path
    kind: str
    reason: str
    refusal: Optional[str] = None


_REASONS = {
    "interpreter": "the interpreter (venv, base install, the link chain between)",
    "unify-package": "the Unify package (the worker runs worker_child.py by path)",
    "editable-root": "an editable install's sys.path entry (.pth), directory only",
    "editable-package": "a package or module of an editable install (.pth)",
    "pythonpath": "the harness's PYTHONPATH (relay client)",
    "preload": "LD_PRELOAD of the harness (fake clock)",
}


def _interpreter_candidates() -> list[Path]:
    """The interpreter's venv and base install, by every name it is reached through.

    A uv venv's ``bin/python`` links to the base install by its minor-version
    name (``cpython-3.12-...``), itself a link to the patch release
    (``cpython-3.12.11-...``), and ``pyvenv.cfg`` names the former. Each name
    is a candidate; :func:`_derived_candidates` keeps only the interpreter's
    prefixes and paths under them.
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
    return list(dict.fromkeys(Path(os.path.abspath(r)) for r in roots))


def _interpreter_prefixes() -> list[Path]:
    """``sys.prefix``, ``sys.base_prefix`` (and the exec ones), each name resolved too."""
    out: list[Path] = []
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        for q in (Path(os.path.abspath(p)), Path(os.path.realpath(p))):
            if q not in out:
                out.append(q)
    return out


def _editable_roots() -> list[Path]:
    """The ``sys.path`` entries the venv's ``.pth`` files add: editable installs.

    That is how unillm (and any package installed with ``pip install -e``)
    reaches the harness's, and so the worker's, ``sys.path``. Entries a
    script directory or the working directory put there are not counted;
    ``PYTHONPATH``'s are counted in :func:`_derived_candidates`.
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
                if path in on_path and os.path.exists(path) and Path(path) not in out:
                    out.append(Path(path))
    return out


# Never bound from an editable root even when importable: a checkout's own
# tests and build scripts are not what the installed package imports.
_EDITABLE_SKIP = {"tests", "test", "setup.py", "conftest.py", "noxfile.py"}


def _editable_packages(root: Path) -> list[Path]:
    """What an editable root's ``sys.path`` entry imports: its packages and modules.

    Each directory with an ``__init__.py`` and each ``*.py`` file at the
    root's top level whose name is an identifier; nothing else there (a
    checkout's ``.env``, ``.git``, tests, logs). Namespace packages (no
    ``__init__.py``) are not found.
    """
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except OSError:
        return []
    out: list[Path] = []
    for entry in entries:
        name = entry.name
        if name.startswith(".") or name in _EDITABLE_SKIP:
            continue
        try:
            if (
                entry.is_dir(follow_symlinks=False)
                and name.isidentifier()
                and (Path(entry.path) / "__init__.py").is_file()
            ):
                out.append(Path(entry.path))
            elif (
                entry.is_file(follow_symlinks=False)
                and name.endswith(".py")
                and name[:-3].isidentifier()
            ):
                out.append(Path(entry.path))
        except OSError:
            continue
    return out


def _is_shared_library(name: str) -> bool:
    return name.endswith(".so") or ".so." in name


def _preload_files() -> list[Path]:
    """The libraries the harness's own ``LD_PRELOAD`` names (the fake clock).

    Only what the trusted parent runs with; a cell never adds to it
    (:data:`TRUSTED_ENV`). Only absolute paths of regular files named
    ``*.so`` or ``*.so.*``; :func:`_derived_candidates` refuses the rest.
    """
    raw = os.environ.get("LD_PRELOAD", "")
    return [Path(p) for p in raw.replace(":", " ").split() if os.path.isabs(p)]


def _guarded_dirs() -> list[Path]:
    """The state directory and the log directories: never inside a derived root."""
    from unify.db import store_home

    return [Path(os.path.realpath(store_home())), *_log_dirs()]


def _derived_candidates(guarded: Optional[Sequence[Path]] = None) -> list[DerivedRoot]:
    """Every candidate for the root taken from this process, accepted or refused.

    A candidate is accepted only as one of the enumerated kinds (see
    :class:`DerivedRoot`), and only if :func:`_root_refusal` does not refuse
    it:

    * ``interpreter``: exactly ``sys.prefix``, ``sys.base_prefix`` (and the
      exec ones, each by its own and its resolved name) or a link-chain name
      under one of them; never a parent (``~/.local`` above a
      ``~/.local/bin/python3`` link, or ``pyvenv.cfg``'s ``home``);
    * ``unify-package``: the package directory itself;
    * ``editable-root`` / ``editable-package``: a ``.pth`` entry on
      ``sys.path`` is an empty directory; only its packages and modules
      (:func:`_editable_packages`) are bound;
    * ``pythonpath``: a ``PYTHONPATH`` entry on ``sys.path``, exactly;
    * ``preload``: a ``*.so`` file ``LD_PRELOAD`` names, exactly.
    """
    if guarded is None:
        guarded = _guarded_dirs()
    out: list[DerivedRoot] = []

    def add(path: Path, kind: str, refusal: Optional[str] = None) -> bool:
        if refusal is None:
            refusal = _root_refusal(path, guarded)
        out.append(DerivedRoot(path, kind, _REASONS[kind], refusal))
        return refusal is None

    prefixes = _interpreter_prefixes()
    for path in _interpreter_candidates():
        under = any(
            _within(q, p)
            for q in (path, Path(os.path.realpath(path)))
            for p in prefixes
        )
        add(
            path,
            "interpreter",
            None if under else "not the interpreter's prefix or a path under it",
        )
    package = Path(__file__).resolve().parent
    add(
        package,
        "unify-package",
        None if (package / "__init__.py").is_file() else "not a package directory",
    )
    for root in _editable_roots():
        if not os.path.isdir(root):
            add(root, "editable-root", "not a directory")
            continue
        packages = _editable_packages(root)
        if not add(
            root,
            "editable-root",
            None if packages else "no package or module at its top level",
        ):
            continue
        for p in packages:
            add(p, "editable-package")
    on_path = {os.path.abspath(p) for p in sys.path if p}
    for raw in dict.fromkeys(os.environ.get("PYTHONPATH", "").split(os.pathsep)):
        if not raw:
            continue
        path = Path(os.path.abspath(raw))
        if str(path) not in on_path:
            # Under ``python -I`` (the dry run) PYTHONPATH is not on sys.path.
            add(path, "pythonpath", "not on sys.path")
        elif not path.exists():
            add(path, "pythonpath", "does not exist")
        else:
            add(path, "pythonpath")
    for path in _preload_files():
        if not _is_shared_library(path.name):
            add(path, "preload", "not a shared library (*.so, *.so.*)")
        elif not os.path.isfile(path):
            add(path, "preload", "not a regular file")
        else:
            add(path, "preload")
    return out


def _derived_roots() -> list[tuple[Path, str]]:
    """The accepted read-only roots taken from this process, with reasons."""
    return [(d.path, d.reason) for d in _derived_candidates() if d.refusal is None]


def _is_system(path: Path) -> bool:
    return any(
        _within(path, Path(os.path.realpath(d)))
        for d in _SYSTEM_DIRS
        if os.path.isdir(d)
    )


def _link_args(
    name: Path,
    kept: Sequence[Path],
    made: dict[Path, Path],
) -> list[str]:
    """``--symlink`` arguments that make *name* resolve, inside, as on the host.

    Each linked component of *name* (itself or a parent: ``<worktree>/.venv``
    linking to another worktree's venv) becomes the same link to its resolved
    target, so a path through the link reaches the one mount of the target,
    and the masks there. Never one inside a mount (the link is there already)
    or a system directory (the allowlist keeps those links).
    """
    args: list[str] = []
    path = Path(os.path.abspath(name))
    for _ in range(40):
        if Path(os.path.realpath(path)) == path:
            break
        link: Optional[Path] = None
        current = Path(path.parts[0])
        for part in path.parts[1:]:
            current = current / part
            if os.path.islink(current):
                link = current
                break
        if link is None:
            break
        target = Path(os.path.realpath(link))
        inside = any(_within(link, k) for k in kept) or any(
            _within(link, Path(p)) for p in _SYSTEM_DIRS
        )
        if not inside and link not in made:
            made[link] = target
            args += ["--symlink", str(target), str(link)]
        path = target.joinpath(*path.parts[len(link.parts) :])
    return args


def _root_mounts(
    notices: Path,
    derived: Optional[Sequence[DerivedRoot]] = None,
) -> tuple[list[str], list[Path]]:
    """The allowlisted root: bubblewrap arguments, and the host paths they show.

    :data:`_ROOT_ALLOWLIST` first, then the accepted
    :func:`_derived_candidates`: each bound once, from its resolved path,
    outer paths before inner ones; an editable root is only a directory. A
    name that reaches one through a link gets the same link
    (:func:`_link_args`), so masks on the resolved path hold whatever name a
    cell uses. A refused candidate is left out with a warning; nothing of it
    is then visible.
    """
    import logging

    if derived is None:
        derived = _derived_candidates()
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
    dirs_only: list[Path] = []
    names: list[Path] = []
    for d in derived:
        if d.refusal is not None:
            logging.getLogger(__name__).warning(
                "workspace sandbox: not mounting %s (%s): %s",
                d.path,
                d.kind,
                d.refusal,
            )
            continue
        real = Path(os.path.realpath(d.path))
        if not real.exists() or _is_system(real):
            continue
        names.append(Path(os.path.abspath(d.path)))
        target = dirs_only if d.kind == "editable-root" else wanted
        if real not in target:
            target.append(real)
    # Outer before inner; an inner path under an outer one is already shown.
    kept: list[Path] = []
    for p in sorted(wanted, key=lambda q: (len(q.parts), str(q))):
        if any(_within(p, k) for k in kept):
            continue
        kept.append(p)
        args += ["--ro-bind", str(p), str(p)]
        visible.append(p)
    for p in dirs_only:
        if not any(_within(p, k) for k in kept):
            args += ["--dir", str(p)]
    made: dict[Path, Path] = {}
    for name in names:
        args += _link_args(name, kept, made)
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
            store_path(),
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
        # Mounted only if present, and sessions pointed at it must find it.
        (state_dir / "transcripts").mkdir(parents=True, exist_ok=True)
        # Cells may read (grep, tail) their own run's agent record, as
        # transcripts; never another run's, and only the harness writes it.
        records = _current_run_records()
        # What a cell sees of the state directory, read-only (the harness's
        # file tools check the same list):
        # - transcripts/: every session's and every agent's, across sessions,
        #   by the lead's design (8 Oct: "the model should be able to grep any
        #   transcript even other sessions' and other agents' transcripts too.
        #   That's by design."). Every line is outcome-free (the outcome
        #   section the harness rendered is redacted, unify/outcome.py), and
        #   the harness-internal sessions (reviews, gates) are written to
        #   internal-transcripts/, which is never mounted (``hidden`` below).
        # - not the raw store (store.sqlite, -wal, -shm): cells reach the
        #   library through the functions/guidance API, and the file holds
        #   what the API does not give (recorded cases, trust, history).
        readonly = [
            p
            for p in (
                state_dir / "transcripts",
                *((records,) if records is not None else ()),
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
        # Every directory the root mounts from this process, at every depth.
        log_dirs = _log_dirs()
        derived = _derived_candidates([state_dir, *log_dirs])
        bound = sorted(
            {
                Path(os.path.realpath(d.path))
                for d in derived
                if d.refusal is None
                and d.kind != "editable-root"
                and os.path.isdir(d.path)
            },
        )
        from unify.environment import installer_cache

        found_files, found_dirs = _find_secret_files(
            bound,
            skip,
            cached=[
                Path(os.path.realpath(d.path))
                for d in derived
                if d.kind == "interpreter"
            ],
            cache_dir=_scan_cache_dir(
                [workspace, state_dir, installer_cache(), *log_dirs, *bound],
            ),
        )
        for path, rule in found_dirs:
            if all(path != d for d, _ in masked_dirs):
                masked_dirs.append((path, rule))
        for path, rule in found_files:
            if all(path != f for f, _ in masked_files):
                masked_files.append((path, rule))
        # The workspace, scanned for the same names, incrementally and bounded
        # (_scan_workspace): it can be any size (a checkout, a data
        # directory) and changes every cell. What inside it is masked or hidden anyway, its views put back
        # read-only, the installer's cache and a state directory inside it
        # (hidden whole, wrap_argv) are not entered.
        from unify.transcripts import INTERNAL_DIRNAME

        state_inside = state_dir != workspace and _within(state_dir, workspace)
        ws_skip = [
            d
            for d in (
                *(m for m, _ in masked_dirs),
                *log_dirs,
                *readonly,
                state_dir,
                state_dir / INTERNAL_DIRNAME,
                Path(os.path.realpath(installer_cache())),
            )
            if d != workspace and _within(d, workspace)
        ]
        scan = _scan_workspace(workspace, ws_skip)
        scan_note = None
        if scan.capped:
            scan_note = (
                f"workspace sandbox: the secret scan of {workspace} read its cap "
                f"of {_WORKSPACE_SCAN_LIMIT} entries and stopped; secret-named "
                "files in directories it did not read are not masked in cells "
                "(the harness's file tools still refuse them by name)"
            )
        # A find outside the workspace (a link's target) joins the other masks.
        workspace_masked: list[tuple[Path, str]] = []
        for found, out in ((scan.dirs, masked_dirs), (scan.files, masked_files)):
            for path, rule in found:
                target = workspace_masked if _within(path, workspace) else out
                if all(path != m for m, _ in target):
                    target.append((path, rule))
        # The store's files, wherever the workspace bind would show them (a
        # UNIFY_STORE_PATH inside it; a workspace that is UNIFY_HOME itself is
        # refused); inside a state directory hidden whole they are hidden
        # already.
        store = Path(os.path.realpath(store_path()))
        for name in (store.name, f"{store.name}-wal", f"{store.name}-shm"):
            path = store.parent / name
            if _within(path, workspace) and not (
                state_inside and _within(path, state_dir)
            ):
                workspace_masked.append((path, "mask-unify-state"))

        network = str(getattr(SETTINGS, "UNIFY_WORKSPACE_NETWORK", "") or "")
        port = int(getattr(SETTINGS, "UNIFY_WORKSPACE_PROXY_PORT", 0) or 0)
        notices = _notices_dir()
        root_args, root_visible = _root_mounts(notices, derived)
        policy = SandboxPolicy(
            workspace=workspace,
            state_dir=state_dir,
            readonly_state=readonly,
            masked_dirs=masked_dirs,
            masked_files=masked_files,
            hidden=[
                (state_dir / INTERNAL_DIRNAME, "mask-unify-state"),
                *((d, "mask-harness-logs") for d in log_dirs),
            ],
            workspace_masked=workspace_masked,
            workspace_scan=scan,
            scan_note=scan_note,
            network=network,
            proxy_port=port,
            notices_dir=notices,
            root_args=root_args,
            root_visible=root_visible,
            store_files=_store_files(store, state_dir),
        )
        policy._key = key  # type: ignore[attr-defined]
        _POLICY_CACHE = policy
    if scan_note is not None:
        _note_for_the_run(scan_note)
    return policy


_NOTED: set[tuple[str, str]] = set()


def _note_for_the_run(text: str) -> None:
    """*text* in the log and, once per run, in the run's agent record.

    The record (``records/<run>``) is what a reader of the run sees; the
    entry is the harness's ``system`` kind with no mentions, so under the
    default delivery (mentions) it wakes no agent and enters no prompt.
    """
    import logging

    logging.getLogger(__name__).warning("%s", text)
    try:
        from unify.agents.binding import current_root_pool
        from unify.agents.record import HARNESS

        pool = current_root_pool()
        if pool is None:
            return
        key = (str(pool.record.path or id(pool.record)), text)
        if key in _NOTED:
            return
        _NOTED.add(key)
        pool.record.append_harness(HARNESS, text, kind="system", mentions=[])
    except Exception:
        logging.getLogger(__name__).exception("could not note %r in the record", text)


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

    Network reach is unchanged: AF_INET follows the network namespace (never
    the host's), and the proxy mode and the installer's egress reach out only
    through their forwarders' unix sockets, as before.
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
    egress: Optional["EgressProxy"] = None,
) -> list[str]:
    """*argv* as a bubblewrap command line under *policy*.

    *writable* paths are bound read-write after everything else, *readonly*
    ones (a program the harness runs from outside the allowlisted root: the
    installer's ``uv``) read-only on the root, and *egress* replaces the
    policy's network with one loopback port (:data:`INSTALLER_PROXY_PORT`)
    forwarded to that allow-listing proxy; only the harness's package
    installer asks for these (unify/environment.py). No command ever gets the
    host's network.

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
    # What the workspace bind covers of the harness's: a state directory
    # inside it is hidden again, whole, and the views inside the workspace
    # are put back read-only (transcripts stay readable, never writable).
    workspace = policy.workspace
    state = policy.state_dir
    if state != workspace and _within(state, workspace):
        args += ["--tmpfs", str(state)]
        args += ["--ro-bind", str(notices / "mask-unify-state"), _notice(state)]
    for path in policy.readonly_state:
        if _within(path, workspace):
            args += ["--ro-bind", str(path), str(path)]
    # Then the masks inside it: the secret scan's finds and the store's files.
    # Only what exists now: a mount point made on this read-write mount
    # would be made on the host.
    for path, rule in policy.workspace_masked:
        if os.path.islink(path):
            continue
        if os.path.isdir(path):
            args += ["--tmpfs", str(path)]
            args += ["--ro-bind", str(notices / rule), _notice(path)]
        elif os.path.exists(path):
            args += ["--ro-bind", str(notices / rule), str(path)]
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
    if egress is not None:
        # The installer: the same forwarder as the proxy mode, relaying to the
        # harness's allow-listing proxy instead of the operator's.
        _refuse_shown_socket(
            egress.directory,
            policy,
            (*extra, *readonly),
            "installer-index-only",
        )
        args += ["--ro-bind", str(egress.directory), _PROXY_MOUNT]
        command = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _FORWARDER_SRC,
            str(INSTALLER_PROXY_PORT),
            f"{_PROXY_MOUNT}/{egress.path.name}",
            *command,
        ]
    elif policy.network == "proxy":
        bridge = _proxy_bridge(policy.proxy_port)
        _refuse_shown_socket(
            bridge.directory,
            policy,
            (*extra, *readonly),
            "network-proxy-only",
        )
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
    # The /bin/sh wrapper runs outside the sandbox and exports the harness's
    # own working directory as PWD; the command gets its sandbox one instead.
    args += ["--setenv", "PWD", workdir]
    args += ["--remount-ro", "/", "--chdir", workdir, "--"]
    _refuse_broad_binds(
        args,
        homes,
        workspace=policy.workspace,
        state=policy.state_dir,
    )
    return [
        "/bin/sh",
        "-c",
        _SECCOMP_EXEC,
        str(notices / "seccomp.bpf"),
        *args,
        *command,
    ]


_BIND_OPTIONS = ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try")


def _refuse_shown_socket(
    directory: Path,
    policy: SandboxPolicy,
    bound: Sequence[Path],
    rule: str,
) -> None:
    """Refuse when a proxy socket's *directory* is visible in the sandbox
    anywhere but :data:`_PROXY_MOUNT`: under the workspace, a mounted view
    or one of the command's own binds (*bound*), a cell could connect to the
    socket directly or replace it. It lives in the harness's temporary
    directory, which the sandbox replaces with its own (rule
    ``private-tmp``), unless ``TMPDIR`` points somewhere the sandbox shows.
    """
    real = Path(os.path.realpath(directory))
    if policy.readable_violation(real) is None or any(
        _within(real, Path(os.path.realpath(p))) for p in bound
    ):
        raise SandboxRefusal(
            rule,
            f"the proxy's socket directory {real} would be visible inside the "
            "sandbox; point TMPDIR at a directory the sandbox does not show "
            "(the default, /tmp, is private to it)",
        )


def _refuse_broad_binds(
    args: Sequence[str],
    homes: Sequence[Path],
    *,
    workspace: Optional[Path] = None,
    state: Optional[Path] = None,
) -> None:
    """Refuse a command line that would mount ``/`` or a whole home directory.

    Raised for the policy's own mounts (a workspace configured as the home
    directory); derived roots that would are already left out. The workspace's
    read-write bind is its own kind (:func:`_workspace_refusal`); every other
    mount outside the allowlist gets the derived roots' rule.
    """
    end = args.index("--") if "--" in args else len(args)
    allowlisted = {p for p, _ in _ROOT_ALLOWLIST}
    ws = str(workspace) if workspace is not None else None
    for i in range(end - 2):
        if args[i] in _BIND_OPTIONS:
            is_workspace = args[i] == "--bind" and args[i + 1] == args[i + 2] == ws
            for p in (args[i + 1], args[i + 2]):
                why = None
                if is_workspace:
                    why = _workspace_refusal(Path(p), state=state)
                elif _too_broad(Path(p), homes):
                    why = "it would show / or a whole home directory"
                elif args[i + 2] not in allowlisted:
                    # The allowlist's own entries (/usr, ...) are top-level
                    # on purpose; every other mount gets the derived roots'
                    # rule.
                    why = _root_refusal(Path(p))
                if why is not None:
                    raise SandboxRefusal(
                        "root-allowlist",
                        f"{p} would be mounted ({args[i]}): {why}",
                        suggestion=(
                            "Point UNIFY_LOCAL_ROOT (or UNIFY_HOME) at a "
                            "directory of its own: not a home, a system or "
                            "credential directory, or UNIFY_HOME itself."
                        ),
                    )


def _notice(directory: Path) -> str:
    return str(directory / MASK_NOTICE_NAME)


_PYTHON_NAMES = ("python", "python3")


def interpreter_bin_dirs() -> list[str]:
    """The directories that give a sandboxed command ``python`` and ``python3``.

    Cells run with this interpreter (the worker starts ``sys.executable``),
    which the root already mounts by every name it is reached through
    (:func:`_interpreter_candidates`). Its own directory comes first (a venv's
    ``bin``, which usually has both names); the resolved install's directory
    follows only when the first lacks one of them. Nothing is created: a name
    neither directory has stays unresolved.
    """
    dirs: list[str] = []
    missing = set(_PYTHON_NAMES)
    for candidate in (
        os.path.dirname(os.path.abspath(sys.executable)),
        os.path.dirname(os.path.realpath(sys.executable)),
    ):
        if not missing or candidate in dirs:
            continue
        found = {
            name
            for name in missing
            if os.access(os.path.join(candidate, name), os.X_OK)
        }
        if found:
            dirs.append(candidate)
            missing -= found
    return dirs


def _interpreter_first_on_path(path: Optional[str]) -> str:
    """*path* with :func:`interpreter_bin_dirs` first, each entry once."""
    entries = (path if path is not None else os.defpath).split(os.pathsep)
    return os.pathsep.join(
        dict.fromkeys(entry for entry in (*interpreter_bin_dirs(), *entries) if entry),
    )


def sandbox_env(
    policy: SandboxPolicy,
    env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """The environment a sandboxed command gets.

    Without an explicit *env*, it is built from an allow-list of the harness's
    variables (:func:`allowed_env`), never from the harness's environment as a
    whole, so a provider key is not in it whatever its name. ``PATH`` starts
    with the cell interpreter's directories (:func:`interpreter_bin_dirs`), so
    ``python`` and ``python3`` in a cell's subprocess or a bash cell are the
    interpreter cells run with, whether or not the host's ``PATH`` has them.
    They are already mounted; no mount changes. ``USER`` and ``LOGNAME`` are
    the account the generated ``/etc/passwd`` lists. An explicit *env* (a
    cell's ``env=``, already inside the sandbox) is kept as given, without
    credential variables (:func:`scrubbed_env`).
    """
    if env is None:
        out = allowed_env()
        out["PATH"] = _interpreter_first_on_path(out.get("PATH"))
        account = _account_name()
        if account:
            out["USER"] = out["LOGNAME"] = account
    else:
        out = scrubbed_env(env)
    # The fake clock's variables come from the harness only, as it has them.
    for name in TRUSTED_ENV:
        out.pop(name, None)
        value = os.environ.get(name)
        if value is not None and not _looks_like_key(value):
            out[name] = value
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


# ---------------------------------------------------------------------------
# The installer's egress: an allow-listing CONNECT proxy, harness-side
# ---------------------------------------------------------------------------

# The port the forwarder listens on inside the installer's sandbox, on its
# own loopback (the namespace is private, so any port would do).
INSTALLER_PROXY_PORT = 3128
# A CONNECT request head: at most this many bytes, all within this many
# seconds of the connection (one deadline, not one per read).
_PROXY_HEAD_LIMIT = 8192
_PROXY_HANDSHAKE_S = 30.0
_PROXY_CONNECT_S = 10.0
_TUNNEL_IDLE_S = 300.0
# The client's first TLS bytes in a tunnel: the whole ClientHello must
# arrive within this many bytes (record headers included) and seconds.
_CLIENT_HELLO_LIMIT = 16384
_CLIENT_HELLO_S = 10.0
# The encrypted_client_hello extension (TLS ECH): it hides the real server
# name, so a hello carrying it is refused.
_ECH_EXTENSION = 0xFE0D
# At most this many tunnels are open at once; more are refused.
_MAX_TUNNELS = 32
# Of one proxy's refusals, the first this many are kept, logged and shown
# to the model (as fixed notes); the rest are only counted. Each kept target
# and reason is cut to _REFUSAL_TEXT characters.
_REFUSALS_KEPT = 20
_REFUSAL_TEXT = 200
# The well-known NAT64 prefix (RFC 6052): the last 32 bits are the IPv4
# address a translator reaches, link-local and private ones included. The
# local-use prefix (64:ff9b:1::/48, RFC 8215) embeds it at a position its
# length decides, so it is refused with the rest outside 2000::/3.
_NAT64_WELL_KNOWN = "64:ff9b::/96"


def _resolve(host: str, port: int) -> list[tuple]:
    """The addresses *host* resolves to, as :func:`socket.getaddrinfo` gives them."""
    return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)


def public_address(address: str) -> bool:
    """Whether *address* is a globally routable unicast address.

    False for loopback, link-local (the cloud metadata server,
    ``169.254.169.254``), private, shared (CGNAT), reserved, unspecified and
    multicast addresses. An IPv6 address must be global unicast, inside
    ``2000::/3``, and outside every special range: so IPv4-compatible
    (``::a9fe:a9fe``), IPv4-mapped, SIIT (``::ffff:0:a9fe:a9fe``),
    site-local (``fec0::/10``) and unique-local addresses are refused, and
    so is 6to4 (``2002::/16``: not globally reachable for
    :mod:`ipaddress`, and never around a non-public IPv4 address whatever
    its version says). The one exception is the
    well-known NAT64 prefix (``64:ff9b::/96``), whose last 32 bits are the
    IPv4 address a translator reaches: public only if that address is.
    """
    import ipaddress

    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6:
        if ip in ipaddress.ip_network(_NAT64_WELL_KNOWN):
            return public_address(str(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)))
        if ip not in ipaddress.ip_network("2000::/3"):
            return False
        if ip.sixtofour is not None and not public_address(str(ip.sixtofour)):
            return False
    return ip.is_global and not ip.is_multicast


def _clip(text: str) -> str:
    """*text* cut to :data:`_REFUSAL_TEXT` characters."""
    return text if len(text) <= _REFUSAL_TEXT else text[:_REFUSAL_TEXT] + "..."


def _host_key(host: str) -> str:
    return host.strip().strip("[]").lower().rstrip(".")


def _authority(text: str) -> Optional[tuple[str, int]]:
    """``(host, port)`` of a CONNECT request's ``host:port``, or ``None``.

    The port is ASCII digits only: ``str.isdigit`` also holds for ``"²"``,
    which ``int`` refuses, and for other scripts' digits, which it reads.
    """
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    else:
        host, _, port = text.rpartition(":")
    if not host or not (port.isascii() and port.isdigit()) or not 0 < int(port) < 65536:
        return None
    return _host_key(host), int(port)


class _HelloRefused(ValueError):
    """The tunnel's first bytes are not a ClientHello the proxy accepts."""


class _Reader:
    """Bounds-checked reads over a TLS structure."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def done(self) -> bool:
        return self.pos == len(self.data)

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise _HelloRefused("the ClientHello is malformed (truncated field)")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def number(self, width: int) -> int:
        return int.from_bytes(self.take(width), "big")

    def vector(self, width: int) -> bytes:
        """A length-prefixed field whose length takes *width* bytes."""
        return self.take(self.number(width))


def _read_client_hello(
    conn: socket.socket,
    buffered: bytes,
    deadline: float,
) -> tuple[bytes, str]:
    """``(bytes read, server name)``: the client's ClientHello, read from
    *conn* after the *buffered* bytes the request head was followed by.

    Reads TLS handshake records (``content type 22``, version ``3.x``,
    length 1..16384), reassembling a ClientHello fragmented over several of
    them, until the whole handshake message is in; never more than
    :data:`_CLIENT_HELLO_LIMIT` bytes or past *deadline*. Raises
    :class:`_HelloRefused` on anything else: other first bytes (plain HTTP,
    say), a record or message over the limit, a first handshake message
    that is not a ClientHello, a malformed one, the stream ending or the
    time running out first, and a ClientHello without exactly one
    ``host_name`` in its ``server_name`` extension.
    """
    data = bytearray(buffered)

    def need(n: int) -> None:
        if n > _CLIENT_HELLO_LIMIT:
            raise _HelloRefused(
                f"the ClientHello is larger than {_CLIENT_HELLO_LIMIT} bytes",
            )
        while len(data) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise _HelloRefused(
                    f"no complete ClientHello within {_CLIENT_HELLO_S:g} s",
                )
            conn.settimeout(left)
            try:
                chunk = conn.recv(min(4096, n - len(data)))
            except (socket.timeout, TimeoutError):
                raise _HelloRefused(
                    f"no complete ClientHello within {_CLIENT_HELLO_S:g} s",
                ) from None
            if not chunk:
                raise _HelloRefused("the tunnel closed before a complete ClientHello")
            data.extend(chunk)

    handshake = bytearray()
    pos = 0
    while True:
        need(pos + 5)
        kind, major, length = (
            data[pos],
            data[pos + 1],
            data[pos + 3] << 8 | data[pos + 4],
        )
        if kind != 22 or major != 3:
            raise _HelloRefused("the tunnel's first bytes are not a TLS handshake")
        if not 0 < length <= 16384:
            raise _HelloRefused(f"a TLS record of {length} bytes is out of bounds")
        need(pos + 5 + length)
        handshake += data[pos + 5 : pos + 5 + length]
        pos += 5 + length
        if len(handshake) < 4:
            continue
        if handshake[0] != 1:
            raise _HelloRefused("the first handshake message is not a ClientHello")
        size = 4 + int.from_bytes(handshake[1:4], "big")
        if size > _CLIENT_HELLO_LIMIT:
            raise _HelloRefused(
                f"the ClientHello is larger than {_CLIENT_HELLO_LIMIT} bytes",
            )
        if len(handshake) >= size:
            return bytes(data), _client_hello_sni(bytes(handshake[4:size]))


def _client_hello_sni(body: bytes) -> str:
    """The one ``host_name`` a ClientHello's *body* names (RFC 8446 4.1.2,
    RFC 6066 3), lower-cased without a trailing dot."""
    r = _Reader(body)
    r.take(2 + 32)  # legacy_version, random
    if len(r.vector(1)) > 32:  # legacy_session_id
        raise _HelloRefused("the ClientHello is malformed (session id)")
    suites = r.vector(2)
    if not suites or len(suites) % 2:
        raise _HelloRefused("the ClientHello is malformed (cipher suites)")
    if not r.vector(1):  # legacy_compression_methods
        raise _HelloRefused("the ClientHello is malformed (compression methods)")
    if r.done():
        raise _HelloRefused("the ClientHello names no TLS server (no SNI)")
    extensions = _Reader(r.vector(2))
    if not r.done():
        raise _HelloRefused("the ClientHello is malformed (trailing bytes)")
    seen: set[int] = set()
    name: Optional[bytes] = None
    while not extensions.done():
        kind = extensions.number(2)
        data = extensions.vector(2)
        if kind in seen:
            raise _HelloRefused(f"the ClientHello repeats extension {kind}")
        seen.add(kind)
        if kind == _ECH_EXTENSION:
            # Encrypted Client Hello hides the real server name behind an
            # outer (public) one, so the SNI checked here could be a decoy.
            raise _HelloRefused("the ClientHello uses encrypted_client_hello")
        if kind != 0:  # server_name
            continue
        entries = _Reader(data)
        names = _Reader(entries.vector(2))
        if not entries.done() or names.done():
            raise _HelloRefused("the ClientHello is malformed (server_name)")
        while not names.done():
            if names.number(1) != 0 or name is not None:
                raise _HelloRefused(
                    "the ClientHello's server_name holds more than one host_name",
                )
            name = names.vector(2)
    if name is None:
        raise _HelloRefused("the ClientHello names no TLS server (no SNI)")
    text = name.decode("ascii", "replace")
    if not text or not all(c.isascii() and (c.isalnum() or c in "-._") for c in text):
        raise _HelloRefused("the ClientHello's server name is not a host name")
    return _host_key(text)


class EgressProxy:
    """An HTTP CONNECT proxy, on a private unix socket, for the installer.

    It opens a tunnel only to an *allowed* ``(host, port)`` pair (the
    package index hosts, unify/environment.py), and only to the public
    addresses that host resolves to: a name that resolves to loopback,
    link-local (the cloud metadata server), private or other non-public
    addresses is refused, and the address connected to is the one checked,
    so a second lookup cannot change it. Anything that is not a CONNECT (a
    plain ``http://`` request) is refused.

    An address can serve many sites (``pypi.org`` and
    ``files.pythonhosted.org`` share a CDN's addresses with other
    customers), so the tunnel's first bytes must be a TLS ClientHello whose
    server name (SNI) is the CONNECT's host (case-insensitive, without a
    trailing dot); only then are they forwarded and the tunnel relayed
    (:func:`_read_client_hello`). Without SNI, with another name, with other
    first bytes or a ClientHello over 16 KB the tunnel is closed. What the
    proxy cannot see is the request inside TLS: a ``Host`` header naming
    another site on the same CDN is the CDN's to refuse (domain fronting).

    The first :data:`_REFUSALS_KEPT` refusals are logged and kept in
    :attr:`refused` as ``(target, reason)`` (each cut to
    :data:`_REFUSAL_TEXT` characters), with a fixed note per refusal in
    :attr:`notes` for the model to read: a note never quotes the CONNECT
    target or TLS server name the client sent. Later refusals are only
    counted (:attr:`unlisted`), so a client opening thousands of refused
    tunnels cannot flood the log or the install's output.

    The allow-list is fixed when the proxy starts, by the harness; nothing
    the sandboxed command sends can add to it. :func:`wrap_argv` mounts the
    socket's directory into the sandbox, where the forwarder relays the
    command's loopback port (:data:`INSTALLER_PROXY_PORT`) to it; :meth:`env`
    gives the proxy variables that point the command there.
    """

    def __init__(self, allowed: Sequence[tuple[str, int]]) -> None:
        self.allowed = frozenset((_host_key(h), int(p)) for h, p in allowed)
        self.refused: list[tuple[str, str]] = []
        self.notes: list[str] = []
        self.unlisted = 0
        self._refused_lock = threading.Lock()
        self.directory = Path(tempfile.mkdtemp(prefix="unify-installer-proxy-"))
        self.path = self.directory / "proxy.sock"
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(64)
        # At most _MAX_TUNNELS connections are handled at once, and close()
        # shuts every open one, so no tunnel outlives the install.
        self._slots = threading.BoundedSemaphore(_MAX_TUNNELS)
        self._open: set[socket.socket] = set()
        self._open_lock = threading.Lock()
        threading.Thread(
            target=self._serve,
            daemon=True,
            name="unify-installer-proxy",
        ).start()

    @property
    def url(self) -> str:
        """The proxy's address as the sandboxed command sees it."""
        return f"http://127.0.0.1:{INSTALLER_PROXY_PORT}"

    def env(self) -> dict[str, str]:
        """The proxy variables for a command behind this proxy."""
        out = {}
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            out[name] = out[name.lower()] = self.url
        return out

    def close(self) -> None:
        try:
            # Wakes the accept() blocked in the serving thread.
            self._server.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._server.close()
        with self._open_lock:
            still_open = list(self._open)
        for s in still_open:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        shutil.rmtree(self.directory, ignore_errors=True)
        if self.unlisted:
            logging.getLogger(__name__).warning(
                "installer proxy: %d more refusals, not logged",
                self.unlisted,
            )

    def _track(self, s: socket.socket) -> socket.socket:
        with self._open_lock:
            self._open.add(s)
        return s

    def _untrack(self, s: socket.socket) -> None:
        with self._open_lock:
            self._open.discard(s)

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            if not self._slots.acquire(blocking=False):
                reason = f"more than {_MAX_TUNNELS} tunnels at once"
                self._refuse(conn, "?", "503 Service Unavailable", reason, reason)
                conn.close()
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _note_refusal(self, target: str, reason: str, note: str) -> None:
        """Keep and log a refusal, or count it past the first few.

        *target* and *reason* may quote what the client sent (they are for
        the harness's log); *note* is fixed text the model may read.
        """
        target, reason = _clip(target), _clip(reason)
        with self._refused_lock:
            kept = len(self.refused) < _REFUSALS_KEPT
            if kept:
                self.refused.append((target, reason))
                self.notes.append(note)
            else:
                self.unlisted += 1
            first_unlisted = not kept and self.unlisted == 1
        if kept:
            # repr(): a target or reason quoting client bytes cannot forge
            # log lines.
            logging.getLogger(__name__).warning(
                "installer proxy: refused %r: %r",
                target,
                reason,
            )
        elif first_unlisted:
            logging.getLogger(__name__).warning(
                "installer proxy: more than %d refusals; the rest are counted",
                _REFUSALS_KEPT,
            )

    def _refuse(
        self,
        conn: socket.socket,
        target: str,
        status: str,
        reason: str,
        note: str,
    ) -> None:
        self._note_refusal(target, reason, note)
        body = f"Refused by the Unify installer proxy: {reason}\n".encode()
        try:
            conn.sendall(
                f"HTTP/1.1 {status}\r\nContent-Type: text/plain\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body,
            )
        except OSError:
            pass

    def _handle(self, conn: socket.socket) -> None:
        up: Optional[socket.socket] = None
        self._track(conn)
        try:
            # One deadline for the whole head: a client trickling a byte at a
            # time cannot hold a tunnel slot past it.
            deadline = time.monotonic() + _PROXY_HANDSHAKE_S
            head = b""
            while b"\r\n\r\n" not in head:
                left = deadline - time.monotonic()
                if left <= 0:
                    reason = f"no complete request head within {_PROXY_HANDSHAKE_S:g} s"
                    self._refuse(conn, "?", "408 Request Timeout", reason, reason)
                    return
                conn.settimeout(left)
                try:
                    data = conn.recv(4096)
                except (socket.timeout, TimeoutError):
                    continue
                if not data:
                    return
                head += data
                if len(head) > _PROXY_HEAD_LIMIT:
                    reason = "request head too large"
                    self._refuse(conn, "?", "431 Too Large", reason, reason)
                    return
            head, _, early = head.partition(b"\r\n\r\n")
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split(" ")
            if len(parts) != 3 or parts[0] != "CONNECT":
                reason = "only CONNECT tunnels to the package index are proxied"
                self._refuse(conn, line, "405 Method Not Allowed", reason, reason)
                return
            target = _authority(parts[1])
            if target is None:
                self._refuse(
                    conn,
                    parts[1],
                    "400 Bad Request",
                    "bad target",
                    "a malformed CONNECT target",
                )
                return
            name = f"{target[0]}:{target[1]}"
            if target not in self.allowed:
                self._refuse(
                    conn,
                    name,
                    "403 Forbidden",
                    f"{name} is not a package index host",
                    "a host off the allow-list (only the package index hosts "
                    "are reached)",
                )
                return
            up, reason = self._connect(*target)
            if up is None:
                self._refuse(
                    conn,
                    name,
                    "403 Forbidden",
                    reason,
                    "a package index host with no reachable public address",
                )
                return
            self._track(up)
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            # Nothing the client sends reaches the index host before its
            # ClientHello names that host; otherwise the tunnel is closed (no
            # HTTP status can follow the 200) and the refusal noted.
            try:
                first, sni = _read_client_hello(
                    conn,
                    early,
                    time.monotonic() + _CLIENT_HELLO_S,
                )
            except _HelloRefused as exc:
                # Its text is fixed, with numbers at most: no client bytes.
                self._note_refusal(
                    name,
                    f"tunnel to {name}: {exc}",
                    f"a tunnel whose TLS ClientHello was refused: {exc}",
                )
                return
            if sni != target[0]:
                self._note_refusal(
                    name,
                    f"tunnel to {name}: the ClientHello names {sni!r}, "
                    f"not {target[0]!r} (SNI must be the CONNECT host)",
                    "a tunnel whose TLS server name was not its CONNECT host "
                    "(SNI must be the CONNECT host)",
                )
                return
            up.sendall(first)
            for s in (conn, up):
                s.settimeout(_TUNNEL_IDLE_S)
            back = threading.Thread(target=_relay, args=(up, conn), daemon=True)
            back.start()
            _relay(conn, up)
            back.join(_TUNNEL_IDLE_S)
        except OSError:
            pass
        except Exception as exc:
            # Whatever the client sent, the handler ends here, so the finally
            # below always closes its sockets and frees its slot. Only the
            # exception's type is recorded: its message may quote client bytes.
            self._note_refusal(
                "?",
                f"the proxy failed on a request ({type(exc).__name__})",
                "a request the proxy could not handle",
            )
        finally:
            for s in (conn, up):
                if s is not None:
                    self._untrack(s)
                    s.close()
            self._slots.release()

    def _connect(self, host: str, port: int) -> tuple[Optional[socket.socket], str]:
        try:
            infos = _resolve(host, port)
        except OSError as exc:
            return None, f"{host} does not resolve ({exc})"
        public = [i for i in infos if public_address(str(i[4][0]))]
        if not public:
            return None, (
                f"{host} resolves only to non-public addresses "
                f"({', '.join(sorted({str(i[4][0]) for i in infos}))}); "
                "loopback, link-local and private addresses are never reached"
            )
        last = ""
        for family, _type, proto, _canon, sockaddr in public:
            s = socket.socket(family, socket.SOCK_STREAM, proto)
            s.settimeout(_PROXY_CONNECT_S)
            try:
                s.connect(sockaddr)
                return s, ""
            except OSError as exc:
                s.close()
                last = str(exc)
        return None, f"could not connect to {host}:{port} ({last})"


def _relay(a: socket.socket, b: socket.socket) -> None:
    """Copy *a* to *b*; at *a*'s end of stream, end *b*'s (half-close)."""
    try:
        while True:
            data = a.recv(65536)
            if not data:
                b.shutdown(socket.SHUT_WR)
                return
            b.sendall(data)
    except OSError:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


@contextmanager
def egress_proxy(allowed: Sequence[tuple[str, int]]) -> Iterator[EgressProxy]:
    """An :class:`EgressProxy` for *allowed*, closed on exit."""
    proxy = EgressProxy(allowed)
    try:
        yield proxy
    finally:
        proxy.close()


# ---------------------------------------------------------------------------
# Dry run: python -m unify.sandbox --print-roots
# ---------------------------------------------------------------------------


def print_roots(out=None) -> None:
    """Each derived root candidate: accepted or refused, its kind, path, reason.

    Paths and fixed reasons only: no environment values, no file contents.
    """
    out = out or sys.stdout
    for d in _derived_candidates():
        state = "accepted" if d.refusal is None else "refused"
        print(f"{state}\t{d.kind}\t{d.path}\t{d.refusal or d.reason}", file=out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m unify.sandbox")
    parser.add_argument(
        "--print-roots",
        action="store_true",
        help="print the derived roots the sandbox would mount or refuse, and exit",
    )
    args = parser.parse_args(argv)
    if args.print_roots:
        print_roots()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    # The package's own module, not a second copy run as __main__.
    from unify import sandbox as _sandbox

    raise SystemExit(_sandbox.main())
