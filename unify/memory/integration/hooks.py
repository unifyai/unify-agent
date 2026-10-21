"""The names harness files import from memory v2 (UNIFY_MEMORY); each is inert while the switch is off.

With the switch off every function returns its input unchanged (or nothing), so the harness behaves
byte for byte as shipped. With it on and no request run active, only the storage review and the library
objects are withdrawn.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_LIBRARY_OBJECTS = ("functions", "guidance")
_HELP_ONLY = "Read live docs in-sandbox with\n`help(...)`"
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


def _copy() -> Path | None:
    """The run's library copy as cells read it: the export itself, or under ``UNIFY_MEMORY_IN_EPISODE=on`` its link
    to the current pin (``RequestRun.copy_root``); only once the export exists."""
    path = _checkout()
    if path is None:
        return None
    root = getattr(_run(), "copy_root", None)
    return Path(root) if root is not None else path


def worker_paths() -> list[str]:
    """Import paths the worker puts first: the run's memory export, then the library test kit beside it when
    the export holds one (:mod:`..testkit`; ``memlab`` for the library's tests). Under ``UNIFY_MEMORY_IN_EPISODE=on``
    the current pin's own directory (the link's target): a re-pin gives the worker the new one."""
    from ..testkit import EXPORT_DIR

    path = _copy()
    if path is None:
        return []
    if getattr(_run(), "in_episode", None) is not None:
        # each pin is imported from its own directory, so a function's code names the pin it came from, and a
        # re-pin moves the import path (Worker.repin_modules)
        path = path.resolve()
    kit = path / EXPORT_DIR
    return [str(path)] + ([str(kit)] if kit.is_dir() else [])


def worker_globals() -> dict[str, list]:
    """Globals the worker binds in every cell, as init specs: in library mode with a library copy, ``memory``,
    imported by the first cell that uses it (the actor's guide says the library is ``memory`` in its cells).
    Nothing of the library runs when the worker starts; empty otherwise."""
    path = _copy()
    if path is None:
        return {}
    try:
        present = (path / "memory" / "__init__.py").is_file()
    except (OSError, ValueError):
        present = False
    if not present:
        return {}
    return {"memory": ["lazy_module", "memory"]}


#: The actor's first user message's line on the library in its cells (with the librarian on), filled from the copy.
LIBRARY_LINE = (
    "Your memory library holds {functions} and {notes}; memory.ask() shows the ones that apply, memory.items() "
    "lists them all."
)


def _counted(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def library_counts(copy: Path) -> tuple[int, int]:
    """How many functions and notes the library copy at *copy* holds, as ``memory.items()`` lists them (every one
    but the retired), from its ``.memory/items.json``; ``(0, 0)`` when that cannot be read."""
    from unify.memory.library_helper import _HIDDEN_STATUSES, ITEMS_FILE, _status_of

    try:
        rows = json.loads((Path(copy) / ITEMS_FILE).read_text(encoding="utf-8")).get("items")
    except (OSError, ValueError, UnicodeDecodeError, AttributeError):
        return 0, 0
    kinds = [
        v.get("kind")
        for v in (rows.values() if isinstance(rows, dict) else ())
        if isinstance(v, dict) and _status_of(v) not in _HIDDEN_STATUSES
    ]
    return kinds.count("function"), kinds.count("note")


def library_line(actor: bool = True) -> str | None:
    """The line on the library that opens the actor's first user message of a request (never the system prompt,
    whose cached prefix stays the same for every request): how many functions and notes the run's copy holds when
    the request starts (:func:`library_counts`), generated from the copy. Only for the actor (*actor*), with the
    library bound in its cells and the librarian on (it names ``memory.ask``); None otherwise, and when the copy
    holds no function and no note."""
    if not actor or "memory" not in worker_globals():
        return None
    from unify.memory import librarian

    path = _copy()
    if path is None or not librarian.enabled():
        return None
    functions, notes = library_counts(path)
    if functions + notes == 0:
        return None
    return LIBRARY_LINE.format(functions=_counted(functions, "function"), notes=_counted(notes, "note"))


def worker_mounts() -> list[Path]:
    """Paths the worker's sandbox binds read-write: none (the library copy is bound read-only,
    :func:`worker_readonly_mounts`)."""
    return []


def worker_readonly_mounts() -> list[Path]:
    """Paths the worker's sandbox binds read-only after its masks: the run's library copy (library mode),
    whose root, the parent of ``memory/``, is also first on the import path (:func:`worker_paths`).
    """
    path = _checkout()
    if path is None:
        return []
    return [path]


def worker_audit() -> dict | None:
    """The ``audit`` entry of the worker's init message, or None (then the key is left out and the child
    installs no hook).

    While a request captures the work tree: the workspace root, for file events; in library mode without a
    capture, only the library copy's roots, so the library's runtime calls and the cells' reads of the copy are
    recorded (the use record's ``library_file_lookups``). With a capture: the workspace root as the sandbox
    shows it (``/workspace``), the library copy's roots in library mode, and the file of the audit hook, which the
    child loads by path (stdlib only; no ``unify`` import there).
    """
    if _run() is None:
        return None
    try:
        from .adapters import audit
        from .worktree_capture import active

        capture = active()
        library = _library_roots()
        if capture is None:
            return {"roots": library, "path": str(Path(audit.__file__))}
        # The sandbox binds the workspace at a fixed path (``unify.sandbox.WORKSPACE_MOUNT``), so that is
        # the root the child sees; the records are mapped back to the host's in :func:`worker_cell_done`.
        from unify.sandbox import WORKSPACE_MOUNT

        return {"roots": [WORKSPACE_MOUNT] + library, "path": str(Path(audit.__file__))}
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the worker always starts; this request then records no file events
        logger.warning("memory v2: no audit for this worker (%s)", type(exc).__name__)
        return None


def _library_roots() -> list[str]:
    """The library copy's paths as a cell can see them: the run's export roots (``RequestRun.export_roots``, the
    sandbox's fixed mount among them), else the sandbox's fixed mount alone."""
    from unify.sandbox import LIBRARY_MOUNT

    roots = getattr(_run(), "export_roots", None)
    roots = [r for r in roots if isinstance(r, str) and r] if isinstance(roots, list) else []
    return roots or [LIBRARY_MOUNT]


def _without_library(events: Any, roots: list[str]) -> Any:
    """*events* without the path records under the library copy (*roots*): the work-tree capture records the
    workspace only."""
    if not roots or not isinstance(events, dict) or not isinstance(events.get("records"), list):
        return events
    prefixes = tuple(r.rstrip("/") for r in roots)

    def inside(value: Any) -> bool:
        return isinstance(value, str) and any(value == r or value.startswith(r + "/") for r in prefixes)

    kept = [
        r
        for r in events["records"]
        if not (isinstance(r, dict) and inside(r.get("path")) and not r.get("dst"))
    ]
    return {**events, "records": kept}


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
    ie = getattr(run, "in_episode", None)
    if ie is not None and name == "execute_code":
        # UNIFY_MEMORY_IN_EPISODE=on: the episode's cells are numbered in the order their results come back
        ie.cell_done()
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
            library = _library_roots()
            capture.cell_done(
                time.time(),
                _host_paths(_without_library(events, library), capture.workspace),
            )
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


# --- memory during an episode (UNIFY_MEMORY_IN_EPISODE=on; in_episode.py) ---------------------------------------------


def attach_handle(handle: Any) -> None:
    """The actor's session handle, which a segment fork continues; inert while off or with no run."""
    run = _run()
    if run is not None:
        run.handle = handle


def in_episode_active() -> bool:
    """Whether this request's memory changes during its episode (``UNIFY_MEMORY_IN_EPISODE=on``); never raises."""
    try:
        return getattr(_run(), "in_episode", None) is not None
    except Exception:  # noqa: BLE001 - off
        return False


def before_cell() -> dict | None:
    """Before a worker runs a cell: under ``UNIFY_MEMORY_IN_EPISODE=on``, a re-pin to the commit requests now pin when
    the writer landed one (:meth:`.in_episode.InEpisode.before_cell`): ``{"generation": n, "notice": <line or None>}``,
    and :func:`after_cell` once the cell ends. None while off or with no run (then no :func:`after_cell`); never
    raises."""
    run = _run()
    ie = getattr(run, "in_episode", None) if run is not None else None
    if ie is None:
        return None
    try:
        from ..memory_writer import served_head

        head = served_head(run.stores.memory)
    except Exception as exc:  # noqa: BLE001 - the cell runs on the pin it has
        logger.warning("memory: the served head not read at a cell boundary (%s)", type(exc).__name__)
        head = None
    try:
        return ie.before_cell(head)
    except Exception as exc:  # noqa: BLE001 - a re-pin never stops a cell
        logger.warning("memory: re-pin failed (%s)", type(exc).__name__)
        return None


def after_cell() -> None:
    """The cell :func:`before_cell` began has ended; never raises."""
    run = _run()
    ie = getattr(run, "in_episode", None) if run is not None else None
    if ie is not None:
        try:
            ie.after_cell()
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory: cell end not noted (%s)", type(exc).__name__)


def repin_roots() -> list[str]:
    """The host paths under which a re-pinned worker drops every loaded module before its next cell: the library
    copy's directory (all its pins); [] while off or with no copy."""
    run = _run()
    if getattr(run, "in_episode", None) is None:
        return []
    path = _checkout()
    return [str(path)] if path is not None else []


def turn_boundary(client: Any, *, compacting: bool = False, new_message: bool = False) -> None:
    """Between the actor's turns (the loop, before its next model call; before a compaction; before a new message joins
    the conversation): under ``UNIFY_MEMORY_IN_EPISODE=on`` the pauses due start their segment fork
    (:meth:`.in_episode.InEpisode.at_boundary`). Only the request's own actor (its session's client); inert while off,
    for another loop, or with no run; never raises."""
    run = _run()
    ie = getattr(run, "in_episode", None) if run is not None else None
    if ie is None:
        return
    handle = getattr(run, "handle", None)
    inner = getattr(handle, "_inner", None) or handle
    if handle is None or getattr(inner, "_client", None) is not client:
        return
    try:
        # harness-owned: the segment's record and its fork, which runs in its own box
        event = ie.at_boundary(handle, compacting=compacting, new_message=new_message)
    except Exception as exc:  # noqa: BLE001 - a pause never stops the actor
        logger.warning("memory: a checkpoint failed (%s)", type(exc).__name__)
        return
    if event is not None:
        logger.info("memory: checkpoint %s", event)
