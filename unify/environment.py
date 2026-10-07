"""The workspace environment: one persistent venv every trajectory shares.

Third-party packages the assistant needs are installed into a single virtual
environment under the store home (``<UNIFY_HOME>/venv``). Its ``site-packages``
is appended to ``sys.path`` — after unify's own packages, so unify's
dependencies always win — of the process cells run in. Nothing is ever
removed from it: a package installed during one task is importable in every
later task and session.

With Python in the sandboxed worker (``UNIFY_WORKSPACE_PYTHON=worker``) that
process is the worker, which puts the environment on its own path
(unify/actor/execution/worker.py). The harness, which holds the provider
credentials, then never imports from it: :func:`activate` refuses, and
:func:`missing` reads the installed distributions' metadata from the
environment's path without putting it on ``sys.path``.

A stored function records the packages it imports as PEP 508 requirement
strings (its ``dependencies``); :func:`ensure` installs whichever of them
are missing right before the function runs.

The packages are the model's choice, and installing one can run its build
steps (an sdist's ``setup.py`` or build backend). So ``uv`` never gets the
harness's environment, which holds the provider credentials: it gets the
few variables an install needs (:func:`installer_env`). With
``UNIFY_WORKSPACE=sandboxed`` the install also runs inside bubblewrap under
the workspace policy (unify/sandbox.py): ``/`` read-only, credential
locations and ``.env`` files hidden, and only this environment and the
installer's own cache writable. It keeps the host's network, as it had
before, since it has to reach the package index.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    "UV_CACHE_DIR",
    "XDG_CACHE_HOME",
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
    """Whether cells run in this process, which then imports the environment's
    packages; ``False`` with Python in the sandboxed worker."""
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


def installer_env() -> Dict[str, str]:
    """The environment ``uv`` runs with: :data:`_INSTALLER_ENV`, never the
    harness's (which holds the provider credentials)."""
    return sandbox.scrubbed_env(
        {name: os.environ[name] for name in _INSTALLER_ENV if name in os.environ},
    )


def _installer(argv: List[str]) -> Tuple[List[str], Dict[str, str], Optional[str]]:
    """``(argv, env, cwd)`` for running the installer command *argv*.

    With ``UNIFY_WORKSPACE=sandboxed``, *argv* runs inside bubblewrap under
    the workspace policy, with this environment and the installer's cache
    bound writable and the host's network kept.
    """
    env = installer_env()
    if not sandbox.enabled():
        return argv, env, None
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
        },
    )
    # The environment as the working directory: no project configuration
    # (pyproject.toml, uv.toml) a cell wrote in the workspace applies.
    wrapped = sandbox.wrap_argv(
        argv,
        policy,
        cwd=str(venv),
        writable=[venv, cache],
        share_network=True,
    )
    return wrapped, env, str(venv)


def _create() -> Path:
    """Create the environment with the running interpreter and activate it."""
    if not environment_python().exists():
        environment_dir().parent.mkdir(parents=True, exist_ok=True)
        # The harness's own command (no package is named), so not confined;
        # the worker may already have created the directory, empty.
        with unconfined():
            subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(environment_dir())],
                capture_output=True,
                text=True,
                check=True,
                env=installer_env(),
            )
    site_packages().mkdir(parents=True, exist_ok=True)
    return activate() if imports_in_process() else site_packages()


def _check_specifiers(specifiers: List[str]) -> None:
    """Refuse an installer option among *specifiers*: each names a package."""
    for specifier in specifiers:
        if not isinstance(specifier, str) or specifier.lstrip().startswith("-"):
            raise ValueError(
                f"{specifier!r} is not an installer option the environment "
                "accepts: name packages only, as requirement specifiers "
                "(e.g. 'pandas>=2', 'pkg @ git+https://github.com/user/repo.git').",
            )


def install(specifiers: List[str], *, timeout: float = 300) -> Dict[str, Any]:
    """Install *specifiers* into the environment and make them importable.

    Returns ``success``, the installer's ``stdout`` / ``stderr`` and the
    requested ``packages``. Raises ``ValueError`` for an installer option
    among *specifiers* before anything runs.
    """
    specifiers = list(specifiers)
    _check_specifiers(specifiers)
    _create()
    argv, env, cwd = _installer(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(environment_python()),
            *specifiers,
        ],
    )
    # Never wrapped by a cell's subprocess confinement (UNIFY_WORKSPACE): the
    # command is the harness's, and already wrapped when the sandbox is on.
    with unconfined():
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )
    importlib.invalidate_caches()
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
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
