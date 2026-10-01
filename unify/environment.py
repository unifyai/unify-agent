"""The workspace environment: one persistent venv every trajectory shares.

Third-party packages the assistant needs are installed into a single virtual
environment under the store home (``<UNIFY_HOME>/venv``). Its ``site-packages``
is appended to ``sys.path`` — after unify's own packages, so unify's
dependencies always win — and everything runs in-process. Nothing is ever
removed from it: a package installed during one task is importable in every
later task and session.

A stored function records the packages it imports as PEP 508 requirement
strings (its ``dependencies``); :func:`ensure` installs whichever of them
are missing right before the function runs.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from packaging.requirements import Requirement

from unify.db import store_home
from unify.sandbox import unconfined


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


def activate() -> Path | None:
    """Put the environment's packages on ``sys.path`` if it exists.

    Returns the ``site-packages`` path when the environment has been created,
    ``None`` otherwise. Idempotent and cheap, so it runs at the start of every
    trajectory: an environment populated by an earlier session is importable
    before the first cell runs.
    """
    packages = site_packages()
    if not packages.is_dir():
        return None
    path = str(packages)
    if path not in sys.path:
        sys.path.append(path)
        importlib.invalidate_caches()
    return packages


def _create() -> Path:
    """Create the environment with the running interpreter and activate it."""
    if not environment_python().exists():
        environment_dir().parent.mkdir(parents=True, exist_ok=True)
        with unconfined():
            subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(environment_dir())],
                capture_output=True,
                text=True,
                check=True,
            )
    site_packages().mkdir(parents=True, exist_ok=True)
    return activate()


def install(specifiers: List[str], *, timeout: float = 300) -> Dict[str, Any]:
    """Install *specifiers* into the environment and make them importable.

    Returns ``success``, the installer's ``stdout`` / ``stderr`` and the
    requested ``packages``.
    """
    _create()
    # The harness installs a stored function's declared dependencies even when
    # the function runs from a sandboxed cell (UNIFY_WORKSPACE).
    with unconfined():
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(environment_python()),
                *specifiers,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
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


def missing(specifiers: List[str]) -> List[str]:
    """The subset of *specifiers* not satisfied by an installed distribution."""
    activate()
    absent: List[str] = []
    for specifier in specifiers:
        requirement = parse_requirement(specifier)
        try:
            version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            absent.append(specifier)
            continue
        if not requirement.specifier.contains(version, prereleases=True):
            absent.append(specifier)
    return absent


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
