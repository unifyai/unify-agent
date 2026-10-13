"""The names harness files import from memory v2 (UNIFY_MEMORY); each is inert while the switch is off.

With the switch off every function returns its input unchanged (or nothing), so the harness behaves
byte for byte as shipped. With it on and no request run active, only the storage review and the library
objects are withdrawn.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_LIBRARY_OBJECTS = ("functions", "guidance")
_HELP_ONLY = "Read live docs in-sandbox with\n`help(...)`"
# A traceback line naming a library function's refusal: ``env.<channel>[.<module>...].MemoryInputError``.
_REFUSAL = re.compile(
    r"^env\.([A-Za-z_][A-Za-z0-9_]*)(?:\.[A-Za-z_][A-Za-z0-9_]*)*\.MemoryInputError\b",
    re.MULTILINE,
)

T = TypeVar("T")


def enabled() -> bool:
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_MEMORY", "") == "on"


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
    """No storage review and no library writes under memory v2: memory changes only through the writer's gate."""
    return False if enabled() else value


def system_prompt(text: str) -> str:
    """*text* with the run's memory section appended, last in the cached prefix: the v2 index,
    or under ``UNIFY_MEMORY_SURFACING=catalogue`` the constant guide (the same bytes for the whole run).

    Under memory v2 the sandbox has no ``functions`` object, so the core prompt's pointer to
    ``functions.search`` is dropped too, and the role and the sandbox section no longer name stored
    functions; the change is the same for every request.
    """
    if not enabled():
        return text
    from unify.actor import prompt_builders as _prompts

    text = text.replace(_prompts._CORE_SANDBOX_SEARCH_PYTHON, _HELP_ONLY)
    # No function store either: the role and the sandbox section say what a
    # cell has instead.
    text = text.replace(
        _prompts._LEAN_ROLE_LIBRARY,
        _prompts._LEAN_ROLE_LIBRARY_MEMORY,
    )
    text = text.replace(
        _prompts._SANDBOX_GLOBALS_INTRO,
        _prompts._SANDBOX_GLOBALS_INTRO_MEMORY,
    )
    run = _run()
    index = getattr(run, "index", "") if run is not None else ""
    return f"{text}\n\n{index}" if index else text


def sandbox_objects(objects: dict) -> dict:
    """The core surface's sandbox objects without ``functions`` and ``guidance`` under memory v2."""
    if not enabled():
        return objects
    return {k: v for k, v in objects.items() if k not in _LIBRARY_OBJECTS}


def _checkout() -> Path | None:
    """The run's memory export, only once it exists (a bind source that does not exist stops the worker)."""
    run = _run()
    if run is None:
        return None
    path = Path(run.paths.checkout)
    return path if path.is_dir() else None


def worker_paths() -> list[str]:
    """Import paths the worker puts first: the run's memory export, then the library test kit beside it when
    the export holds one (:mod:`..testkit`; ``memlab`` for the library's tests)."""
    from ..testkit import EXPORT_DIR

    path = _checkout()
    if path is None:
        return []
    kit = path / EXPORT_DIR
    return [str(path)] + ([str(kit)] if kit.is_dir() else [])


def worker_globals() -> dict[str, list]:
    """Globals the worker binds in every cell, as init specs: in library mode with a library copy, ``memory``,
    imported by the first cell that uses it (the actor's guide says the library is ``memory`` in its cells).
    Nothing of the library runs when the worker starts; empty otherwise."""
    path = _checkout()
    if path is None or not getattr(_run(), "library_mode", False):
        return {}
    try:
        present = (path / "memory" / "__init__.py").is_file()
    except (OSError, ValueError):
        present = False
    if not present:
        return {}
    return {"memory": ["lazy_module", "memory"]}


def worker_mounts() -> list[Path]:
    """Paths the worker's sandbox binds read-write: the run's memory export (v2), nothing else. Under
    UNIFY_MEMORY_LIBRARY there are none: the copy is bound read-only (:func:`worker_readonly_mounts`).
    """
    path = _checkout()
    if path is None or getattr(_run(), "library_mode", False):
        return []
    return [path]


def worker_readonly_mounts() -> list[Path]:
    """Paths the worker's sandbox binds read-only after its masks: the run's library copy (library mode),
    whose root, the parent of ``memory/``, is also first on the import path (:func:`worker_paths`).
    """
    path = _checkout()
    if path is None or not getattr(_run(), "library_mode", False):
        return []
    return [path]


def worker_audit() -> dict | None:
    """The ``audit`` entry of the worker's init message, or None (then the key is left out and the child
    installs no hook).

    While a request captures the work tree: the workspace root, for file events; in library mode without a
    capture, no roots, so only the library's runtime calls are recorded. With a capture: the workspace root as the sandbox
    shows it (``/workspace``) and the file of the audit hook, which the child loads by path
    (stdlib only; no ``unify`` import there).
    """
    if _run() is None:
        return None
    try:
        from .adapters import audit
        from .request import memory_library_on
        from .worktree_capture import active

        capture = active()
        if capture is None:
            # library mode still needs the library's runtime calls (the evidence "used" rests on): the hook goes
            # in with no file roots, so it records calls and no file events
            if not memory_library_on():
                return None
            return {"roots": [], "path": str(Path(audit.__file__))}
        # The sandbox binds the workspace at a fixed path (``unify.sandbox.WORKSPACE_MOUNT``), so that is
        # the root the child sees; the records are mapped back to the host's in :func:`worker_cell_done`.
        from unify.sandbox import WORKSPACE_MOUNT

        return {"roots": [WORKSPACE_MOUNT], "path": str(Path(audit.__file__))}
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the worker always starts; this request then records no file events
        logger.warning("memory v2: no audit for this worker (%s)", type(exc).__name__)
        return None


def result_hook() -> Callable[..., None] | None:
    """The tool loop's hook for a finished tool call (:func:`tool_result`), or None while the switch is
    off or no request run is active: the loop then never enters it (``tools_data``), so with memory v2
    off a tool call runs exactly as shipped. Never raises."""
    try:
        return tool_result if _run() is not None else None
    except Exception as exc:  # noqa: BLE001 - the loop goes on without the hook
        logger.warning("memory v2: no tool result hook (%s)", type(exc).__name__)
        return None


def tool_result(
    name: str,
    call_id: Any,
    raw: Any,
    *,
    raised: BaseException | None = None,
) -> None:
    """Note a finished code cell's structured result for the request's use record (``memory_use``).

    Kept by *call_id*, as names only (``analysis.use``): a code cell's ``ExecutionResult`` (either
    projection; its status, the memory items its traceback left, its session), the code tool's plain
    ``dict`` result (``execute_code`` only: an empty cell, or its session executor raised), or, with
    *raised*, that the ``execute_code`` call raised instead of returning (its outcome is then unknown).
    The use record reads this, never the rendered tool message, so a cell's printed output cannot pose
    as its status. Inert while off or with no run; never raises.
    """
    run = _run()
    if run is None:
        return
    try:
        from unify.actor.execution.types import ExecutionResult

        if raised is not None:
            if name == "execute_code":
                run.note_failure(call_id, raised)
        elif isinstance(raw, ExecutionResult):
            run.note_result(call_id, raw)
        elif (
            name == "execute_code"
            and isinstance(raw, Mapping)
            and "error" in raw
            and "session_id" in raw
        ):
            run.note_result(call_id, raw)
    except Exception as exc:  # noqa: BLE001 - recording never fails a tool call
        logger.warning(
            "memory v2: a cell's result was not noted (%s)",
            type(exc).__name__,
        )


def worker_cell_done(events: Any) -> None:
    """Hand one cell's drained audit records (the ``done`` message's ``audit``) to the request: its library
    calls to the use record's runtime counts (``RequestRun.note_calls``), its path records to the work-tree
    capture, stamped now on the harness clock; worker-side times are never used. Inert while off or with no
    run; never raises."""
    run = _run()
    if events is None or run is None:
        return
    try:
        run.note_calls(events)
    except Exception as exc:  # noqa: BLE001 - recording never fails a cell
        logger.warning(
            "memory v2: a cell's library calls were lost (%s)",
            type(exc).__name__,
        )
    try:
        from .worktree_capture import active

        capture = active()
        if capture is not None:
            capture.cell_done(time.time(), _host_paths(events, capture.workspace))
    except Exception as exc:  # noqa: BLE001 - recording never fails a cell
        logger.warning(
            "memory v2: a cell's audit records were lost (%s)",
            type(exc).__name__,
        )


def _host_paths(events: Any, workspace: Path) -> Any:
    """*events* (a cell's drained audit records) with each record's ``path`` and ``dst`` under the sandbox's
    workspace path (``unify.sandbox.WORKSPACE_MOUNT``) mapped to the host *workspace* the capture reads;
    anything else is returned as it came."""
    from unify.sandbox import WORKSPACE_MOUNT

    if not isinstance(events, dict) or not isinstance(events.get("records"), list):
        return events

    def host(value: Any) -> Any:
        if isinstance(value, str) and (
            value == WORKSPACE_MOUNT or value.startswith(WORKSPACE_MOUNT + "/")
        ):
            return str(workspace) + value[len(WORKSPACE_MOUNT) :]
        return value

    records = []
    for record in events["records"]:
        if isinstance(record, dict):
            record = {**record}
            for key in ("path", "dst"):
                if key in record:
                    record[key] = host(record[key])
        records.append(record)
    return {**events, "records": records}


def cell_error(text: str) -> str:
    """A failed cell's error as the model reads it; under ``UNIFY_MEMORY_SURFACING=catalogue``, when it
    is a ``MemoryInputError`` raised by a channel the harness holds suspect (drift), with a line saying so.

    The prompt carries no drift flag (its guide is constant); ``memory.catalog()`` and
    ``memory.describe()`` show the flag, and so does the refusal itself here. Unchanged while off, under
    ``index``, with no run or no suspect channel; never raises.
    """
    run = _run()
    if run is None or not getattr(getattr(run, "surfacing", None), "catalogue", False):
        return text
    try:
        suspect = set(getattr(getattr(run, "state", None), "suspect", ()) or ())
        hit = sorted({m.group(1) for m in _REFUSAL.finditer(text)} & suspect)
    except Exception as exc:  # noqa: BLE001 - the cell's error is shown either way
        logger.warning(
            "memory v2: refusal not checked for drift (%s)",
            type(exc).__name__,
        )
        return text
    if not hit:
        return text
    notes = "".join(
        f"memory: env.{ch} is suspect: the environment changed since its functions were built, so this "
        "refusal may come from that change; do the work directly.\n"
        for ch in hit
    )
    return (text if text.endswith("\n") else text + "\n") + notes
