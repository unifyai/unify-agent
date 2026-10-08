"""The names harness files import from memory v2 (UNIFY_MEMORY_V2); each is inert while the switch is off.

With the switch off every function returns its input unchanged (or nothing), so the harness behaves
byte for byte as shipped. With it on and no request run active, only the storage review and the library
objects are withdrawn.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_LIBRARY_OBJECTS = ("functions", "guidance")
_HELP_ONLY = "Read live docs in-sandbox with\n`help(...)`"

T = TypeVar("T")


def enabled() -> bool:
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_MEMORY_V2", "") == "on"


def _run() -> Any:
    if not enabled():
        return None
    from .request import current

    return current()


def begin_request(request: str) -> Any:
    """This request's memory run (``request.RequestRun``), opened before the actor starts; ``None``
    while the switch is off, when nothing of memory v2 is opened, exported or recorded.
    """
    if not enabled():
        return None
    from .request import RequestRun

    return RequestRun.begin(request)


def can_store(value: T) -> T | bool:
    """No storage review and no library writes under memory v2: memory changes only through Sol's gate."""
    return False if enabled() else value


def system_prompt(text: str) -> str:
    """*text* with the run's memory index appended, last in the cached prefix (spec §G6).

    Under memory v2 the sandbox has no ``functions`` object, so the core prompt's pointer to
    ``functions.search`` is dropped too; the change is the same for every request.
    """
    if not enabled():
        return text
    from unify.actor.prompt_builders import _CORE_SANDBOX_SEARCH_PYTHON

    text = text.replace(_CORE_SANDBOX_SEARCH_PYTHON, _HELP_ONLY)
    run = _run()
    index = getattr(run, "index", "") if run is not None else ""
    return f"{text}\n\n{index}" if index else text


def sandbox_objects(objects: dict) -> dict:
    """The core surface's sandbox objects without ``functions`` and ``guidance`` under memory v2."""
    if not enabled():
        return objects
    return {k: v for k, v in objects.items() if k not in _LIBRARY_OBJECTS}


def worker_paths() -> list[str]:
    """Import paths the worker puts first: the run's memory export."""
    run = _run()
    return [] if run is None else [str(run.paths.checkout)]


def worker_mounts() -> list[Path]:
    """Paths the worker's sandbox binds read-write: the run's memory export, nothing else."""
    run = _run()
    return [] if run is None else [Path(run.paths.checkout)]


def worker_audit() -> dict | None:
    """The ``audit`` entry of the worker's init message, or None (then the key is left out and the child
    installs no hook).

    Only while a memory-v2 request captures the work tree: the workspace root (the sandbox binds it at its
    own path, so the child sees the same path) and the file of the audit hook, which the child loads by path
    (stdlib only; no ``unify`` import there).
    """
    if _run() is None:
        return None
    from .worktree_capture import active

    capture = active()
    if capture is None:
        return None
    from .adapters import audit

    return {"roots": [str(capture.workspace)], "path": str(Path(audit.__file__))}


def worker_cell_done(events: Any) -> None:
    """Hand one cell's drained audit records (the ``done`` message's ``audit``) to the request's work-tree
    capture, stamped now on the harness clock; worker-side times are never used. Inert while off, with no
    run or no capture; never raises."""
    if events is None or _run() is None:
        return
    try:
        from .worktree_capture import active

        capture = active()
        if capture is not None:
            capture.cell_done(time.time(), events)
    except Exception as exc:  # noqa: BLE001 - recording never fails a cell
        logger.warning(
            "memory v2: a cell's audit records were lost (%s)",
            type(exc).__name__,
        )
