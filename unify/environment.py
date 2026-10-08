"""The workspace environment: one persistent venv every trajectory shares.

Third-party packages the assistant needs are installed into a single virtual
environment under the store home (``<UNIFY_HOME>/venv``). Its ``site-packages``
is appended to ``sys.path`` — after unify's own packages, so unify's
dependencies always win — of the process cells run in. Nothing is ever
removed from it: a package installed during one task is importable in every
later task and session.

Python runs in the sandboxed worker, so that process is the worker, which
puts the environment on its own path
(unify/actor/execution/worker.py). The harness, which holds the provider
credentials, then never imports from it: :func:`activate` refuses, and
:func:`missing` reads the installed distributions' metadata from the
environment's path without putting it on ``sys.path``.

A stored function records the packages it imports as PEP 508 requirement
strings (its ``dependencies``); :func:`ensure` installs whichever of them
are missing right before the function runs.

The packages are the model's choice. So:

* Installs are binary-only (``--only-binary :all:``): no build step (an
  sdist's ``setup.py`` or build backend) ever runs. A package without a
  wheel for this platform, ``pkg @ git+...`` and local source paths fail
  with uv's own error, which the model reads. Installer options among the
  specifiers are refused before anything runs, and ``--no-config`` keeps a
  ``uv.toml`` or ``pyproject.toml`` (one a package dropped into the
  environment, say) from changing any of this.
* ``uv`` never gets the harness's environment, which holds the provider
  credentials: it gets the few variables an install needs
  (:func:`installer_env`).
* It runs inside bubblewrap under the workspace policy (unify/sandbox.py):
  ``/`` read-only, credential locations and ``.env`` files hidden, and only
  this environment and the installer's own cache writable.
* It has no network of its own. Its one route out is the harness's
  allow-listing proxy (:class:`unify.sandbox.EgressProxy`), which opens
  tunnels only to the package index hosts (:func:`index_hosts`: PyPI, or a
  mirror the operator configured with ``UV_INDEX*``) and never to a
  loopback, link-local or private address. The host's loopback services
  and the cloud metadata server (``169.254.169.254``, which hands out the
  attached service account's token) are unreachable from it.

Without bubblewrap nothing is installed: the install raises
:class:`unify.sandbox.SandboxRefusal` (rule ``sandbox-required``).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from packaging.requirements import Requirement

from unify import sandbox
from unify.db import store_home
from unify.sandbox import unconfined

# What an install passes on from the harness's environment: finding uv and
# the home directory, the locale, the index and proxy configuration, and
# the CA bundle. Credential-named variables are dropped even from these.
_INSTALLER_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "UV_INDEX",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_NATIVE_TLS",
    "UV_OFFLINE",
    "UV_HTTP_TIMEOUT",
    "UV_HTTP_RETRIES",
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
)


# The package index hosts the installer's proxy opens tunnels to: PyPI's
# index and the host its files are served from. A mirror the operator
# configured (below) is added to them.
DEFAULT_INDEX_HOSTS: Tuple[Tuple[str, int], ...] = (
    ("pypi.org", 443),
    ("files.pythonhosted.org", 443),
)
# The variables naming an index, each a whitespace-separated list of URLs
# (``UV_INDEX`` entries may be ``name=url``).
_INDEX_URL_VARIABLES = (
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_INDEX",
)


def environment_dir() -> Path:
    """The virtual environment's directory."""
    return store_home() / "venv"


def environment_python() -> Path:
    """The interpreter ``uv pip install`` targets."""
    return environment_dir() / "bin" / "python"


def site_packages() -> Path:
    """Where installed packages land; the directory put on ``sys.path``."""
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return environment_dir() / "lib" / version / "site-packages"


def imports_in_process() -> bool:
    """Whether this process imports the environment's packages: ``False``,
    since Python runs in the sandboxed worker, except for a test of the
    function manager's in-process loaders (tests' ``python_in_process``)."""
    from unify.actor.execution import worker

    return not worker.enabled()


def activate() -> Path | None:
    """Put the environment's packages on ``sys.path`` if it exists.

    Returns the ``site-packages`` path when the environment has been created,
    ``None`` otherwise. Idempotent and cheap, so it runs at the start of every
    trajectory: an environment populated by an earlier session is importable
    before the first cell runs.

    Raises ``RuntimeError`` with Python in the sandboxed worker: what an
    install put there would be importable by the harness (for example
    through ``importlib.metadata.entry_points()`` discovery).
    """
    if not imports_in_process():
        raise RuntimeError(
            "The harness does not import from the workspace environment: with "
            "Python in the sandboxed worker only the worker, confined, imports "
            "what is installed there",
        )
    packages = site_packages()
    if not packages.is_dir():
        return None
    path = str(packages)
    if path not in sys.path:
        sys.path.append(path)
        importlib.invalidate_caches()
    return packages


def installer_cache() -> Path:
    """The confined installer's own uv cache, beside the environment.

    Never the host's uv cache: what a build step writes there would reach
    every later install on the machine, the harness's own included.
    """
    return store_home() / "uv-cache"


# The passed-on variables that hold URLs (whitespace-separated lists, a
# ``UV_INDEX`` entry possibly ``name=url``): their userinfo is removed.
_URL_VARIABLES = frozenset(
    {
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    }
    | {name for name in _INSTALLER_ENV if name.startswith("UV_INDEX")}
    | {name for name in _INSTALLER_ENV if name.endswith("_INDEX_URL")}
    | {"UV_DEFAULT_INDEX"},
)


def _without_userinfo(url: str) -> str:
    """*url* without ``user:password@``; scheme, host, port, path, query kept.

    Only the authority (``netloc``) is checked for ``@``: a path or query
    may hold one legitimately (``/@scope/``, ``?ref=a@b``). An item whose
    userinfo cannot be parsed out is dropped whole (``""``), so a credential
    is never passed on by mistake: one with ``@`` but no authority (no
    scheme), and one whose authority is not a valid ``host[:port]`` after
    the userinfo is removed (a ``/``, ``?`` or ``#`` left unencoded in the
    password ends the authority early, leaving ``user:pass`` as host and
    port).
    """
    if "@" not in url:
        return url
    prefix = ""
    if "=" in url.split("://", 1)[0]:
        prefix, url = url.split("=", 1)
        prefix += "="
    try:
        parts = urlsplit(url)
        if "@" in parts.netloc:
            parts = parts._replace(netloc=parts.netloc.rsplit("@", 1)[1])
            url = parts.geturl()
        parts.port  # noqa: B018 (raises ValueError unless host[:port])
    except ValueError:
        return ""
    if not parts.hostname:
        return ""
    return prefix + url


def _strip_userinfo(env: Dict[str, str]) -> Dict[str, str]:
    """*env* with the userinfo removed from every URL variable's URLs.

    Index and proxy credentials never reach the installer (or anything it
    starts). Authenticated indexes are not supported: the allow-listing index
    proxy (:class:`unify.sandbox.EgressProxy`) only opens CONNECT tunnels and
    adds no credentials, so a mirror that needs them fails.
    Only the variable's name is logged, never its value.
    """
    import logging

    out = dict(env)
    for name, value in env.items():
        if name not in _URL_VARIABLES or "@" not in value:
            continue
        stripped = " ".join(
            kept for kept in (_without_userinfo(item) for item in value.split()) if kept
        )
        if stripped != " ".join(value.split()):
            out[name] = stripped
            logging.getLogger(__name__).warning("userinfo removed from %s", name)
    return out


def installer_env() -> Dict[str, str]:
    """The environment ``uv`` runs with: :data:`_INSTALLER_ENV`, never the
    harness's (which holds the provider credentials), with no userinfo in
    its URLs (:func:`_strip_userinfo`)."""
    return _strip_userinfo(
        sandbox.scrubbed_env(
            {name: os.environ[name] for name in _INSTALLER_ENV if name in os.environ},
        ),
    )


def index_hosts(
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[Tuple[str, int], ...]:
    """The ``(host, port)`` pairs the installer may reach.

    :data:`DEFAULT_INDEX_HOSTS`, plus the host of every ``https`` index
    URL in the ``UV_INDEX*`` variables of *env*, by default the harness's
    own environment as :func:`installer_env` passes it on. Never anything a
    cell chose: :func:`install` takes package specifiers only, and a cell's
    environment variables live in the worker, not here. An ``http://`` index
    is not added (the proxy only opens CONNECT tunnels).
    """
    env = installer_env() if env is None else env
    hosts = list(DEFAULT_INDEX_HOSTS)
    for name in _INDEX_URL_VARIABLES:
        for item in env.get(name, "").split():
            if "=" in item.split("://", 1)[0]:
                item = item.split("=", 1)[1]
            try:
                parts = urlsplit(item)
                port = 443 if parts.port is None else parts.port
            except ValueError:
                continue
            if parts.scheme != "https" or not parts.hostname or not 0 < port < 65536:
                continue
            pair = (parts.hostname.lower().rstrip("."), port)
            if pair not in hosts:
                hosts.append(pair)
    return tuple(hosts)


def _installer(
    argv: List[str],
    egress: sandbox.EgressProxy,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """``(argv, env, cwd)`` for running the installer command *argv*.

    *argv* runs inside bubblewrap under the workspace policy, with this
    environment and the installer's cache bound writable, and no network
    but one loopback port forwarded to *egress*. *env* is
    :func:`installer_env`'s result when the caller already has it.
    """
    env = dict(installer_env() if env is None else env)
    venv = environment_dir()
    cache = installer_cache()
    for path in (venv, cache):
        path.mkdir(parents=True, exist_ok=True)
    policy = sandbox.build_policy()
    if Path(os.path.realpath(venv)) not in policy.readonly_state:
        policy = sandbox.build_policy(fresh=True)
    env.pop("XDG_CACHE_HOME", None)
    env.update(
        {
            "UV_CACHE_DIR": str(cache),
            # The cache and the environment are separate mounts, so uv's
            # hardlinks between them would fail; copy without the warning.
            "UV_LINK_MODE": "copy",
            "TMPDIR": "/tmp",
            # The proxy, whatever the harness's own proxy variables say.
            **egress.env(),
        },
    )
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    # The environment as the working directory: no project configuration
    # (pyproject.toml, uv.toml) a cell wrote in the workspace applies.
    wrapped = sandbox.wrap_argv(
        argv,
        policy,
        cwd=str(venv),
        writable=[venv, cache],
        # uv itself, wherever PATH finds it (~/.local/bin is outside the
        # sandbox's root).
        readonly=[Path(p) for p in (shutil.which(argv[0]),) if p],
        egress=egress,
    )
    return wrapped, env, str(venv)


@contextmanager
def _environment_lock() -> Iterator[None]:
    """Held while the environment is created and an install runs, so no
    installer sandbox (which can write the environment) runs while
    :func:`_create` checks and creates it unconfined. A file lock beside the
    environment, never inside it, so it holds across harness processes
    sharing the store home too."""
    import fcntl

    home = environment_dir().parent
    home.mkdir(parents=True, exist_ok=True)
    with open(home / "venv.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _discard_unless_empty(venv: Path) -> None:
    """Move *venv* aside and remove it, unless it is absent or an empty
    directory (the one the worker makes before it starts, to mount it)."""
    if not os.path.lexists(venv):
        return
    if venv.is_dir() and not venv.is_symlink():
        with os.scandir(venv) as entries:
            if next(entries, None) is None:
                return
    aside = venv.with_name(f"{venv.name}.discarded-{os.getpid()}-{time.time_ns()}")
    os.rename(venv, aside)
    logging.getLogger(__name__).warning(
        "the workspace environment had no interpreter but held files; "
        "creating it afresh (the old one is removed)",
    )
    if aside.is_dir() and not aside.is_symlink():
        # Never follows the links inside (shutil.rmtree.avoids_symlink_attacks).
        shutil.rmtree(aside, ignore_errors=True)
    else:
        aside.unlink(missing_ok=True)


def _create() -> Path:
    """Create the environment with the running interpreter and activate it.

    ``uv venv`` is the harness's own command (no package is named) and
    runs unconfined, so it must never run over what a sandbox wrote. The
    installer's sandbox can write the environment, so code a package runs
    there could remove ``bin/python`` and leave links or files for an
    unconfined ``uv venv`` to write through, or configuration for it to
    read. An environment without its interpreter is therefore created in a
    directory nothing sandboxed has touched: an empty one in place (the
    worker's, which it has mounted), anything else moved aside and removed
    first (a worker already running then sees the new one after it
    restarts). ``--no-config``, and the store home as the working directory
    (never the environment or the workspace), keep any ``uv.toml`` or
    ``pyproject.toml`` from applying. The caller holds
    :func:`_environment_lock`.
    """
    venv = environment_dir()
    if not environment_python().exists():
        venv.parent.mkdir(parents=True, exist_ok=True)
        _discard_unless_empty(venv)
        with unconfined():
            subprocess.run(
                ["uv", "venv", "--no-config", "--python", sys.executable, str(venv)],
                capture_output=True,
                text=True,
                check=True,
                env=installer_env(),
                cwd=str(venv.parent),
            )
        # Only in what uv just made: never through the paths of an
        # environment the installer's sandbox has written.
        site_packages().mkdir(parents=True, exist_ok=True)
    return activate() if imports_in_process() else site_packages()


def _check_specifiers(specifiers: List[str]) -> None:
    """Refuse an installer option among *specifiers*: each names a package."""
    for specifier in specifiers:
        if not isinstance(specifier, str) or specifier.lstrip().startswith("-"):
            raise ValueError(
                f"{specifier!r} is not an installer option the environment "
                "accepts: name packages only, as requirement specifiers "
                "(e.g. 'pandas>=2', 'pandas[sql]==2.1.0').",
            )


def _refusal_note(egress: sandbox.EgressProxy) -> str:
    """What the model reads of *egress*'s refusals: each kept refusal's fixed
    note once, and how many more there were.

    Never the CONNECT target or TLS server name a client sent: the client is
    uv, or code a model-installed package runs inside the installer's
    sandbox, so that text is the model's to choose. The harness's log keeps
    it (:meth:`unify.sandbox.EgressProxy._note_refusal`).
    """
    notes = "; ".join(dict.fromkeys(egress.notes))
    if egress.unlisted:
        notes += f" (+{egress.unlisted} more refusals)"
    return (
        f"Refused by workspace sandbox rule `installer-index-only` "
        f"({sandbox.RULES['installer-index-only']}): {notes}\n"
    )


def install(specifiers: List[str], *, timeout: float = 300) -> Dict[str, Any]:
    """Install *specifiers* into the environment and make them importable.

    Wheels only: no build step runs (see the module docstring). Returns
    ``success``, the installer's ``stdout`` / ``stderr`` and the requested
    ``packages``. Raises ``ValueError`` for an installer option among
    *specifiers* before anything runs, and
    :class:`unify.sandbox.SandboxRefusal` without bubblewrap.
    """
    specifiers = list(specifiers)
    _check_specifiers(specifiers)
    with _environment_lock():
        return _install(specifiers, timeout)


def _install(specifiers: List[str], timeout: float) -> Dict[str, Any]:
    _create()
    # One environment per install: the hosts and the command read the same.
    base_env = installer_env()
    with sandbox.egress_proxy(index_hosts(base_env)) as egress:
        argv, env, cwd = _installer(
            [
                "uv",
                "pip",
                "install",
                "--no-config",
                "--only-binary",
                ":all:",
                "--python",
                str(environment_python()),
                *specifiers,
            ],
            egress,
            base_env,
        )
        # Never wrapped by a cell's subprocess confinement: the command is the
        # harness's, and already wrapped.
        with unconfined():
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
        note = _refusal_note(egress) if egress.refused else ""
    importlib.invalidate_caches()
    stderr = result.stderr
    if note:
        # uv reports only "tunnel error: unsuccessful"; say what was refused.
        stderr = (stderr or "").rstrip("\n") + "\n" + note
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": stderr,
        "packages": list(specifiers),
    }


def parse_requirement(specifier: str) -> Requirement:
    """The PEP 508 requirement a stored dependency string denotes.

    Raises ``packaging.requirements.InvalidRequirement`` for anything else,
    so a malformed dependency is rejected when it is recorded rather than
    when the function first runs.
    """
    return Requirement(specifier)


def _worker_path() -> List[str]:
    """The path the worker imports from: this process's, then the environment."""
    packages = str(site_packages())
    return [p for p in sys.path if p and p != packages] + [packages]


def _installed_version(name: str, path: Optional[List[str]]) -> str:
    """The version of the distribution *name* found on *path* (``sys.path`` if
    ``None``), read from its metadata; nothing is imported."""
    if path is None:
        return importlib.metadata.version(name)
    found = next(iter(importlib.metadata.distributions(name=name, path=path)), None)
    if found is None:
        raise importlib.metadata.PackageNotFoundError(name)
    return found.version


def missing(specifiers: List[str]) -> List[str]:
    """The subset of *specifiers* not satisfied by an installed distribution.

    With Python in the sandboxed worker the distributions are looked up on the
    worker's path by their metadata; the environment never joins this
    process's ``sys.path``.
    """
    path: Optional[List[str]] = None
    if imports_in_process():
        activate()
    else:
        path = _worker_path()
    absent: List[str] = []
    for specifier in specifiers:
        requirement = parse_requirement(specifier)
        try:
            version = _installed_version(requirement.name, path)
        except importlib.metadata.PackageNotFoundError:
            absent.append(specifier)
            continue
        if not requirement.specifier.contains(version, prereleases=True):
            absent.append(specifier)
    return absent


def holds_module(module: str) -> bool:
    """Whether the environment holds the top-level *module*, found by path
    without importing anything or touching ``sys.path``."""
    import importlib.machinery

    try:
        spec = importlib.machinery.PathFinder.find_spec(
            module,
            [str(site_packages())],
        )
    except (ImportError, ValueError):
        return False
    return spec is not None


def ensure(specifiers: List[str]) -> None:
    """Install whichever of *specifiers* are not already importable.

    Raises ``RuntimeError`` carrying the installer's output when an install
    fails, so a function whose dependencies cannot be met fails before its
    body runs rather than on an import deep inside it.
    """
    absent = missing(specifiers)
    if not absent:
        return
    outcome = install(absent)
    if not outcome["success"]:
        raise RuntimeError(
            f"Failed to install {absent}: {outcome['stderr'].strip()}",
        )
