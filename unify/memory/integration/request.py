"""The memory run of the request in progress (one request per CLI process; integration Task 25).

``RequestRun.begin`` opens a run under ``UNIFY_MEMORY=on``: it takes the request lock, exports memory
``main`` into the scratch export the worker mounts, notes its memory functions (for the use record),
renders the memory section the system prompt ends with (``index``: under ``UNIFY_MEMORY_SURFACING=index``,
the default, the v2 per-function index and export line; under ``catalogue``, the constant guide paragraph
once the library has listed anything in this run, after writing the generated catalogue beside the export:
README, ``.memory/catalog.json`` with the suspect flags, and the ``memory`` helper, :mod:`..catalogue`) and
records what it shows in either mode (``shown``: names and a digest, :func:`..analysis.use.record_shown`;
under ``catalogue`` that is the guide alone), takes the work tree's before snapshot, and opens the scope the
actor runs in (its transcript continues the episode id and model costs are recorded). ``finish`` records
the request as one episode (with ``UNIFY_MEMORY_DIALOGUE`` set, its transcript's dialogue actions too:
:mod:`.adapters.dialogue`), with how it used the library (``memory_use.json``, indexed in the evidence
store's ``item_use`` table), and runs the consolidation passes that are due, blocking; it never raises. The
passes' start and end events go to the CLI's ``--jsonl`` output when it has one; the consolidation driver
appends them to the state directory's ``events.jsonl`` (``Paths.events``) either way. ``abort`` cleans up
and records nothing. The harness hooks read the current run (``current()``) for ``index`` and ``paths``.

Only pass/fail of a posted outcome is kept: ``take_outcome`` keeps ``solved`` and drops
everything else at once, so no checker text reaches an episode, the evidence, the writer or a cell.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CURRENT: Any = None

#: The reasoning effort the actor's client gets when nothing sets one (unify/common/llm_client.py
#: ``_build_llm_client``'s ``reasoning_effort`` default).
CLIENT_DEFAULT_EFFORT = "high"
LOCK_TIMEOUT_S = 60.0
#: Library mode: the evidence store's busy wait while a pass worker shares it (async_pass.BUSY_TIMEOUT_S; a
#: constant here keeps the v2 path free of the async module).
_LIBRARY_BUSY_TIMEOUT_S = 60.0
#: The most code-cell statuses a run keeps (use telemetry); every cell of a request fits many times over.
MAX_STATUSES = 10_000
#: The most distinct items a request's runtime call counts keep; calls of more are counted as dropped.
MAX_RUNTIME_ITEMS = 2000
RUNTIME_OUTCOMES = ("ok", "refused", "error")
#: The last line of a cell error that names the library's package as a missing name, or as a failed import.
_NAME_ERROR = re.compile(r"NameError: name 'memory' is not defined")
_IMPORT_ERROR = re.compile(
    r"(?:ImportError|ModuleNotFoundError): .*'memory(?:\.[A-Za-z0-9_.]*)?'",
)
_ERROR_CHARS = 2000


class MemoryUnavailable(RuntimeError):
    """``UNIFY_MEMORY=on`` cannot run here; the message names what to change."""


def current() -> Any:
    return _CURRENT


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def actor_model() -> str:
    """The model the actor runs on for this process (the CLI's actor takes the session default)."""
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[0]


def check_library_switches(settings: Any) -> None:
    """Refuse a library-mode run whose v2 settings contradict its prompts or gate (switch.LIBRARY_CONFLICTS)."""
    from .switch import library_conflicts

    bad = library_conflicts(settings)
    if bad:
        raise MemoryUnavailable(
            "UNIFY_MEMORY_LIBRARY=on cannot run with "
            + ", ".join(bad)
            + ": each is a v2 setting that the library mode's prompts and gate contradict; turn it off",
        )


def writer_effort(actor: str) -> str:
    """The writer's effort for this run: ``UNIFY_MEMORY_WRITER_EFFORT`` when it fixes one (a declared mismatch
    ablation), else the actor's (*actor*; the default ``actor``)."""
    from unify.settings import SETTINGS

    value = (
        str(getattr(SETTINGS, "UNIFY_MEMORY_WRITER_EFFORT", "") or "actor")
        .strip()
        .lower()
    )
    return actor if value == "actor" else value


def actor_effort() -> str:
    """The actor's reasoning effort for this run, read from the setting the actor's client reads.

    Writer passes run at this effort too unless ``UNIFY_MEMORY_WRITER_EFFORT`` fixes another (:func:`writer_effort`).
    """
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[1] or CLIENT_DEFAULT_EFFORT


def build_id() -> str:
    """The harness checkout's commit, or ``unknown`` when it is not a git checkout."""
    from ..gitio import git_base_env

    root = Path(__file__).resolve().parents[3]
    env = git_base_env()
    env.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_TERMINAL_PROMPT="0",
    )
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                "rev-parse",
                "--verify",
                "HEAD",
            ],
            capture_output=True,
            env=env,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.decode("ascii", "replace").strip()
    return sha if out.returncode == 0 and len(sha) == 40 else "unknown"


def new_episode_id(transcripts_dir: Path) -> str:
    """``%Y%m%dT%H%M%S-<8 hex>`` (the transcript session id rule), naming no existing transcript."""
    while True:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        eid = f"{stamp}-{secrets.token_hex(4)}"
        if not (transcripts_dir / f"{eid}.jsonl").exists():
            return eid


def dialogue_counterpart() -> str:
    """The dialogue counterpart ``UNIFY_MEMORY_DIALOGUE`` names (``env``), or ``""`` when it is off."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_MEMORY_DIALOGUE", "") or ""


def memory_library_on() -> bool:
    """Whether ``UNIFY_MEMORY_LIBRARY`` is on: the recorders keep full values and the episode is
    written in record format 2."""
    from unify.settings import SETTINGS

    from .switch import LIBRARY_MODE, parse_memory_library_mode

    return parse_memory_library_mode(getattr(SETTINGS, LIBRARY_MODE, "") or "") == "on"


def checker_visible_on() -> bool:
    """Whether the CLI takes ``{"checker": ...}`` lines: ``UNIFY_MEMORY_LIBRARY`` and
    ``UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE`` both on."""
    from unify.settings import SETTINGS

    from .switch import CHECKER_VISIBLE, parse_checker_visible

    return (
        memory_library_on()
        and parse_checker_visible(getattr(SETTINGS, CHECKER_VISIBLE, "") or "") == "on"
    )


def recorded_dialogue(
    lines: list[dict],
    counterpart: str,
    redactor: Any,
    library_mode: bool = False,
) -> list:
    """The dialogue actions a request's episode records under ``UNIFY_MEMORY_DIALOGUE``: answered replies
    only, so a reply nothing followed (a single-turn request's final reply) adds no action and leaves the
    episode as it is with the setting off. With *library_mode* the observation and the payload are kept whole.
    """
    from .adapters import dialogue

    caps = (
        {
            "max_observation_chars": None,
            "max_payload_chars": None,
            "payload_scan_chars": None,
        }
        if library_mode
        else {}
    )
    return dialogue.dialogue_actions(
        lines,
        counterpart,
        redactor=redactor,
        answered_only=True,
        **caps,
    )


def check_hidden(paths: Any, policy: Any) -> None:
    """Refuse to run while a cell could read any harness-side memory path."""
    readable = [
        str(p) for p in paths.harness_only() if policy.readable_violation(p) is None
    ]
    if readable:
        raise MemoryUnavailable(
            "UNIFY_MEMORY needs its state hidden from cells, but the sandbox would let a cell "
            f"read {readable}; keep UNIFY_HOME outside the workspace and every mount",
        )


def _export_roots(checkout: Path) -> list[str]:
    """The paths cells may see the export at: its host paths (an in-process cell's) and the fixed path the
    worker's sandbox binds it at (``unify.sandbox.LIBRARY_MOUNT``)."""
    from unify.sandbox import LIBRARY_MOUNT

    from ..analysis import use

    return use.roots_of(checkout) + [LIBRARY_MOUNT]


def pinned_items(checkout: Path, library_mode: bool = False) -> list[str]:
    """The ids of the memory functions in the export at *checkout* (listed or not), in id order; with *library_mode* the
    library-mode function items."""
    if library_mode:
        from ..layout import discover

        return sorted(f.item_id for f in discover(Path(checkout)).functions)
    from ..memory_repo import items

    return sorted(
        it.item_id for it in items(Path(checkout)).items if it.kind == "env_function"
    )


def _count(value: Any) -> int:
    return value if type(value) is int and value > 0 else 0


def runtime_calls_empty() -> dict:
    """A request's runtime call evidence before any cell: ``cells`` whose audit records arrived, per-item
    ``counts`` ({ok, refused, error}) and what was lost (:func:`add_runtime_calls`)."""
    return {
        "cells": 0,
        "counts": {},
        "dropped": 0,
        "records_dropped": 0,
        "malformed": 0,
        "helpers_failed": 0,
        "helpers": {},
    }


#: The library package's own helpers whose calls the use record counts (``memory.helper_call`` audit events).
LIBRARY_HELPERS = ("index", "show", "find")


def library_helper_calls(runtime: Any) -> dict:
    """The library-mode use record's ``library_helper_calls``: for each of the package's helpers
    (``memory.index``, ``memory.show``, ``memory.find``), the calls the audit hook saw, by outcome
    ({ok, refused, error}). Counts only; zeros when none was seen."""
    seen = runtime.get("helpers") if isinstance(runtime, dict) else None
    seen = seen if isinstance(seen, dict) else {}
    out = {}
    for name in LIBRARY_HELPERS:
        row = seen.get(name)
        row = row if isinstance(row, dict) else {}
        out[name] = {k: _count(row.get(k)) for k in RUNTIME_OUTCOMES}
    return out


def library_cell_error(error: Any) -> str | None:
    """``name`` when a cell's error (its traceback text) ended in a ``NameError`` on ``memory``, ``import``
    when it ended in an ``ImportError`` or ``ModuleNotFoundError`` naming ``memory`` or one of its modules,
    else None. Only the error's last line is read."""
    if not isinstance(error, str):
        return None
    lines = [line for line in error.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if _NAME_ERROR.match(last):
        return "name"
    if _IMPORT_ERROR.match(last):
        return "import"
    return None


def library_failures(runtime: Any, cell_errors: Any) -> dict:
    """The library-mode use record's failure counts: ``library_calls_failed``, the library calls the audit hook
    saw raise (outcome ``refused`` or ``error``, items and the package's helpers alike);
    ``library_name_errors``, cells whose error was a ``NameError`` on ``memory``; ``library_import_errors``,
    cells whose error was a failed import of ``memory`` or of one of its modules. Counts only."""
    failed = 0
    if isinstance(runtime, dict):
        counts = runtime.get("counts")
        for row in (counts if isinstance(counts, dict) else {}).values():
            if isinstance(row, dict):
                failed += _count(row.get("refused")) + _count(row.get("error"))
        failed += _count(runtime.get("helpers_failed"))
    errors = cell_errors if isinstance(cell_errors, dict) else {}
    return {
        "library_calls_failed": failed,
        "library_name_errors": _count(errors.get("name")),
        "library_import_errors": _count(errors.get("import")),
    }


def add_runtime_calls(acc: dict, drained: Any) -> None:
    """Add one cell's drained audit records (the worker's ``done`` message ``audit``) to *acc*: the cell is
    counted as observed, and its ``calls`` block (``adapters.audit``), when present, adds its per-item counts.
    Malformed parts are skipped; calls of items beyond :data:`MAX_RUNTIME_ITEMS` count as dropped.
    """
    if not isinstance(drained, dict):
        return
    acc["cells"] += 1
    calls = drained.get("calls")
    if not isinstance(calls, dict):
        return
    for key in ("dropped", "records_dropped", "malformed"):
        acc[key] += _count(calls.get(key))
    helpers = calls.get("helpers")
    for helper_id, row in (helpers if isinstance(helpers, dict) else {}).items():
        if isinstance(row, dict):
            acc["helpers_failed"] = (
                _count(acc.get("helpers_failed"))
                + _count(row.get("refused"))
                + _count(row.get("error"))
            )
            # per helper, by outcome: "memory:index" counts as "index"; any other id is not a helper
            name = (
                helper_id.split(":", 1)[1]
                if isinstance(helper_id, str) and helper_id.startswith("memory:")
                else None
            )
            if name in LIBRARY_HELPERS:
                have = acc.setdefault("helpers", {}).setdefault(
                    name,
                    {k: 0 for k in RUNTIME_OUTCOMES},
                )
                for k in RUNTIME_OUTCOMES:
                    have[k] += _count(row.get(k))
            else:
                acc["malformed"] += 1
    counts = calls.get("counts")
    if not isinstance(counts, dict):
        return
    for item, row in counts.items():
        if not isinstance(item, str) or not isinstance(row, dict):
            acc["malformed"] += 1
            continue
        got = {k: _count(row.get(k)) for k in RUNTIME_OUTCOMES}
        have = acc["counts"].get(item)
        if have is None:
            if len(acc["counts"]) >= MAX_RUNTIME_ITEMS:
                acc["dropped"] += sum(got.values())
                continue
            have = acc["counts"][item] = dict.fromkeys(RUNTIME_OUTCOMES, 0)
        for k in RUNTIME_OUTCOMES:
            have[k] += got[k]


def with_runtime_calls(record: dict, runtime: Any) -> dict:
    """*record* (a use record) with the request's runtime call evidence: each called item's row gains
    ``runtime`` = {ok, refused, error} (a row is added for an item the static reading did not see), and the
    record gains ``runtime_calls`` = {cells, items, dropped, records_dropped, malformed}. With no observed
    cell (the audit hook was not installed, or no cell ran) the record is returned unchanged, so readers fall
    back to the static call sites. Every item of a record that has ``runtime_calls`` and no ``runtime`` row
    was observed making no call, unless ``dropped`` is non-zero (some items' calls were not counted).
    """
    if not isinstance(runtime, dict) or _count(runtime.get("cells")) == 0:
        return record
    if not isinstance(record, dict) or "error" in record:
        return record
    rows = record.get("items")
    if not isinstance(rows, dict):
        rows = record["items"] = {}
    counts = runtime.get("counts") if isinstance(runtime.get("counts"), dict) else {}
    for item in sorted(counts):
        row = rows.get(item)
        if not isinstance(row, dict):
            row = rows[item] = {}
        row["runtime"] = dict(counts[item])
    record["runtime_calls"] = {
        "cells": _count(runtime.get("cells")),
        "items": len(counts),
        "dropped": _count(runtime.get("dropped")),
        "records_dropped": _count(runtime.get("records_dropped")),
        "malformed": _count(runtime.get("malformed")),
    }
    return record


def memory_use(
    ep: Any,
    item_ids: list[str],
    *,
    redactor: Any = None,
    export_roots: list[str] | tuple[str, ...] = (),
    surface: dict | None = None,
    shown: dict | None = None,
    shown_text: str = "",
    cell_status: dict | None = None,
    library_mode: bool = False,
    runtime_calls: dict | None = None,
    cell_errors: dict | None = None,
    memory_bound: bool | None = None,
) -> dict:
    """The request's use record (:func:`..analysis.use.request_use`), with *runtime_calls*, the calls the
    actor's cells actually made as the worker's audit hook observed them (:func:`with_runtime_calls`).

    It reads the transcript, actions and ``memory.diff`` as the episode writes them (through *redactor*,
    so the offline analyser recomputes the same record from the written episode), against the items at
    the pin, the export's roots and the library's import surface taken before the actor ran, *shown*,
    what the memory-section renderer recorded (its digest is taken again over *shown_text*, the section
    as rendered, through *redactor*, as the transcript's copy of the prompt is), and *cell_status*, each
    cell's status from the runtime's structured result by tool call id (``RequestRun.note_result``). In
    library mode the record also counts failures (:func:`library_failures`, from *runtime_calls* and
    *cell_errors*, the cells whose error named the library's package). A failure is recorded as its
    exception type, with the items at the pin; it never stops the episode. *memory_bound* says whether the
    worker bound ``memory`` in the cells' globals (None: as *library_mode*; see ``use.request_use``).
    """
    from ..analysis import use

    def clean(value: Any) -> Any:
        return redactor.obj(value) if redactor is not None else value

    try:
        if isinstance(shown, dict) and redactor is not None and shown_text:
            shown = {**shown, **use.section_digest(redactor.text(shown_text))}
        record = use.request_use(
            clean(list(ep.transcript)),
            item_ids,
            [
                clean(
                    {
                        "cell": a.cell,
                        "kind": getattr(a, "kind", "tool"),
                        "channel": a.channel,
                        "status": a.status,
                    },
                )
                for a in ep.actions
            ],
            memory_diff=(
                redactor.text(ep.memory_diff or "")
                if redactor is not None
                else ep.memory_diff or ""
            ),
            export_roots=export_roots,
            surface=surface,
            shown=shown,
            cell_status=cell_status,
            library_mode=library_mode,
            memory_bound=memory_bound,
        )
        record = with_runtime_calls(record, runtime_calls)
        if library_mode and isinstance(record, dict) and "error" not in record:
            record.update(library_failures(runtime_calls, cell_errors))
            record["library_helper_calls"] = library_helper_calls(runtime_calls)
        return record
    except Exception as exc:  # noqa: BLE001 - telemetry never stops recording
        return {
            "version": use.VERSION,
            "error": type(exc).__name__,
            "items_at_pin": list(item_ids)[: use.MAX_ITEMS_AT_PIN],
        }


def _plain(value: Any) -> Any:
    """*value* with every Decimal as a plain decimal string (never exponent notation)."""
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


async def wait_after_finish(
    paths: Any,
    settings: Any,
    progress: Callable[[str], None] | None = None,
    wait: Callable[..., dict] | None = None,
    relay: Callable[[dict], None] | None = None,
    events_from: int = 0,
) -> dict | None:
    """``UNIFY_MEMORY_LIBRARY_WAIT_SLOT=on``: before the CLI exits, wait for the pass slot this request
    spawned (:func:`.async_pass.wait_slot`), so a host whose sandbox ends the controller's processes with it does
    not kill the worker. Off: returns None at once, as built. Never raises."""
    import asyncio

    from .switch import library_pass_wall_s, library_wait_slot

    try:
        if not library_wait_slot(settings):
            return None
        if wait is None:
            from .async_pass import wait_slot as wait
        return await asyncio.to_thread(
            wait,
            paths,
            float(library_pass_wall_s(settings)),
            relay=relay,
            events_from=events_from,
        )
    except Exception as exc:  # noqa: BLE001 - the request's end never fails on the wait
        logger.warning("memory library mode: waiting for the pass slot failed: %s", exc)
        return None


class RequestRun:
    """One request's memory run; create it with :meth:`begin`."""

    def __init__(self, request: str, paths: Any) -> None:
        self.request = request
        self.paths = paths
        self.index = ""  # the memory section of the system prompt (prompt.render_index or render_memory_section)
        # the export's generated files (relative path -> bytes), left out of memory.diff while unchanged
        self.generated: dict[str, bytes] = {}
        self.surfacing: Any = (
            None  # the library-mode switches read at open (switch.SurfacingOptions)
        )
        self.library_mode = False  # UNIFY_MEMORY_LIBRARY, read at open
        # Whether the library copy holds the package the worker binds as ``memory`` in every cell (set at open).
        self.memory_bound = False
        # What the memory section shows (``analysis.use.record_shown``), set where it is rendered (both modes).
        self.shown: dict | None = None
        # The memory functions of the export at the pin, taken before the actor runs (use telemetry).
        self.item_ids: list[str] = []
        self.surface: dict | None = None
        self.export_roots: list[str] = []
        # Each code cell's status from the runtime's structured result, by tool call id (note_result).
        self.cell_status: dict[str, dict] = {}
        # The library calls the cells made, from the worker's audit records (note_calls).
        self.runtime_calls: dict = runtime_calls_empty()
        # Cells whose error named the library's package: ``name`` and ``import`` (library_cell_error).
        self.cell_errors: dict[str, int] = {"name": 0, "import": 0}
        self.episode_id = ""
        self.started_at = ""
        self.pin = ""
        self.model = ""
        self.effort = ""
        self.build = ""
        self.stores: Any = None
        self.state: Any = None
        # The tool adapter is not wired in this build (deferred): ``assemble`` records no tool calls.
        self.observer: Any = None
        self.costs: Any = None
        self.worktree: Any = None
        self._worktree_finished = False
        # Only the checker's verdict of a posted outcome: True, False or None (none posted).
        self.solved: bool | None = None
        self.outcome_at: str | None = None
        # The verdicts the bed showed the actor, one per ``{"checker": ...}`` line, in order:
        # {"label": "pass"|"fail", "obs": observations in the transcript when it came (or None), "at": time}.
        self.checker_seen: list[dict] = []
        self._lock: int | None = None
        self._scope = contextlib.ExitStack()
        self._closed = False
        # library mode (set at open): passes run in a detached worker and the request pins ``served``
        self.library_mode = False
        self._landed: list[dict] = (
            []
        )  # library mode: worker results applied at open, reported at finish

    @property
    def memory_main(self) -> str:
        return self.pin

    # -- begin ------------------------------------------------------------------------------------

    @classmethod
    def begin(cls, request: str) -> "RequestRun":
        global _CURRENT
        from unify import sandbox, transcripts
        from unify.db import store_home

        from .paths import Paths
        from .state import acquire_lock

        # The sandboxed worker, the core surface and transcripts are the only mode at this base (the
        # code freeze removed their switches), so nothing else needs checking before the lock.
        run = cls(request, Paths.under(store_home()))
        run._lock = acquire_lock(run.paths.lock, timeout_s=LOCK_TIMEOUT_S)
        try:
            run._open(sandbox, transcripts)
        except BaseException:
            run._cleanup()
            raise
        _CURRENT = run
        return run

    def _open(self, sandbox: Any, transcripts: Any) -> None:
        from unify.settings import SETTINGS

        from ..analysis import use
        from . import consolidate, cost, worktree_capture
        from .checkout import export_checkout
        from .state import State
        from .switch import surfacing_options, library_enabled

        paths = self.paths
        policy = sandbox.build_policy(fresh=True)
        check_hidden(paths, policy)
        self.library_mode = library_enabled(SETTINGS)
        if (
            self.library_mode
        ):  # a refused run records nothing: begin cleans up and re-raises
            check_library_switches(SETTINGS)
        self.stores = consolidate.open_stores(
            paths,
            busy_timeout_s=_LIBRARY_BUSY_TIMEOUT_S if self.library_mode else None,
        )
        self.state = State.load(paths.state)
        if self.library_mode:  # library mode: the read-only copy and the index view
            from ..memory_writer import served_head
            from .async_pass import apply_results

            # under the request lock, before _open_library_mode copies the pin
            self._landed = apply_results(self.state, paths)
            # the served head, whose item records are already written
            self.pin = served_head(self.stores.memory)
            self.surfacing = surfacing_options(SETTINGS)
            self._open_library_mode(consolidate)
        else:
            self.pin = self.stores.memory.head()
            # the library test kit beside it when the library's tests use it (library mode, stage 5 checks)
            export_checkout(paths.memory, self.pin, paths.checkout, self.stores.blobs)
            # the use record's view of the pin, taken before anything is generated beside the export and
            # before the actor can edit the scratch copy
            self.item_ids = pinned_items(paths.checkout)
            self.surface = use.library_surface(paths.checkout)
            self.export_roots = _export_roots(paths.checkout)
            self.surfacing = surfacing_options(SETTINGS)
            if self.surfacing.catalogue:
                self._surface_catalogue(consolidate)
            else:  # the v2 index and export line, byte for byte; nothing generated in the export
                from .prompt import render_memory

                self.index, self.shown = render_memory(
                    paths.checkout,
                    self.state.suspect,
                    sys.maxsize if self.surfacing.soft_budget else None,
                )
        self.episode_id = new_episode_id(transcripts.transcripts_dir())
        self.started_at = _now()
        self.model, self.effort, self.build = actor_model(), actor_effort(), build_id()
        # Before the actor starts, so the before snapshot precedes the first cell and the worker's
        # audit hook is installed while the capture is active (hooks.worker_audit).
        self.worktree = worktree_capture.WorktreeCapture(
            paths,
            Path(policy.workspace),
            self.redactor,
            # UNIFY_MEMORY_LIBRARY: files stored whole (the keyword only when on: off, the call as at 4675a3c45)
            **({"library_mode": True} if memory_library_on() else {}),
        )
        self.worktree.begin()
        self._scope.enter_context(transcripts.resume_session(self.episode_id))
        self.costs = cost.install()
        self.costs.activate()
        self._scope.callback(self.costs.deactivate)

    def _surface_catalogue(self, consolidate: Any) -> None:
        """``UNIFY_MEMORY_SURFACING=catalogue``: the generated catalogue beside the export (the pinned
        commit's input shapes frozen, the suspect channels flagged) and the constant guide as the prompt's
        memory section, from the first request whose library lists anything (then kept: ``State.guide``).
        """
        from ..catalogue import GENERATED, write_generated
        from ..shape_rows import lookup_from
        from .prompt import render_catalogue

        checkout = self.paths.checkout
        try:
            self.generated = write_generated(
                checkout,
                shapes=lookup_from(self._shape_rows(consolidate)),
                suspect=self.state.suspect,
            )
        except (
            Exception
        ) as exc:  # noqa: BLE001 - the library stays importable without its catalogue
            logger.warning(
                "memory v2: the export's catalogue was not written (%s)",
                type(exc).__name__,
            )
            # a partial write must not reach memory.diff as the request's
            for rel in GENERATED:
                target = checkout / rel
                if target.is_file() and not target.is_symlink():
                    target.unlink()
        # what the prompt shows is the guide alone (names nothing): recorded as such for the use record
        self.index, self.shown = render_catalogue(checkout, self.state.guide)
        if self.index and not self.state.guide:
            self.state.guide = True
            try:  # also saved at finish; saved now so an aborted request still fixes the prefix
                self.state.save()
            except OSError as exc:
                logger.warning(
                    "memory v2: the guide flag was not saved (%s)",
                    type(exc).__name__,
                )

    def _open_library_mode(self, consolidate: Any) -> None:
        """``UNIFY_MEMORY_LIBRARY=on``. Writes the read-only copy of the pin: no tests, plus the
        generated helper, INDEX.md and links.json (:func:`.checkout.export_actor_library_mode`). The memory section is
        the guide and the index view. No memory.diff is recorded, because the copy cannot be written.
        """
        from ..analysis import use
        from ..shape_rows import lookup_from
        from .checkout import export_actor_library_mode
        from .prompt import render_memory_library_mode

        paths = self.paths
        try:
            shapes = lookup_from(self._shape_rows(consolidate, library_mode=True))
        except (
            Exception
        ) as exc:  # noqa: BLE001 - find then matches nothing; the library stays usable
            logger.warning(
                "memory library mode: input shapes not read (%s)",
                type(exc).__name__,
            )
            shapes = None
        from ..gitio import Repo
        from ..item_records import records_at, status_of

        # the item records at the pin: statuses for the index and find, and what memory.show prints
        records, _ = records_at(Repo(paths.memory), self.pin)
        self.generated = export_actor_library_mode(
            paths.memory,
            self.pin,
            paths.checkout,
            shapes=shapes,
            status_of=status_of(records),
            records=records,
        )
        self.item_ids = pinned_items(paths.checkout, library_mode=True)
        # the worker binds ``memory`` in every cell when the copy holds the package (hooks.worker_globals)
        try:
            self.memory_bound = (paths.checkout / "memory" / "__init__.py").is_file()
        except (OSError, ValueError):
            self.memory_bound = False
        self.export_roots = _export_roots(paths.checkout)
        self.surface = use.library_surface(
            paths.checkout,
            library_mode=True,
        )  # re-exports for the use record
        self.index, self.shown = render_memory_library_mode(paths.checkout)

    def _memory_diff(self) -> str:
        """What the request wrote into its export (v2), or ``""`` in library mode, whose copy is read-only and records
        no memory.diff."""
        if self.library_mode:
            return ""
        from .checkout import checkout_diff

        return checkout_diff(
            self.paths.memory,
            self.pin,
            self.paths.checkout,
            self.stores.blobs,
            generated=self.generated,
        )

    def _shape_rows(self, consolidate: Any, library_mode: bool = False) -> dict:
        """The pinned commit's input-shape rows, frozen on first export (:func:`..shape_rows.shapes_at`);
        functions without recorded shapes are backfilled from the evidence store's covers.
        """
        from ..shape_rows import shapes_at

        episodes = getattr(consolidate, "EpisodeLookup", None)
        return shapes_at(
            self.stores.memory,
            self.stores.evidence,
            self.pin,
            self.paths.checkout,
            lookup=episodes(self.stores).action if episodes is not None else None,
            blobs=self.stores.blobs,
            freeze=True,
            library_mode=library_mode,
        )

    def redactor(self) -> Any:
        """The run's redactor as far as it is known before the episode is assembled: the environment's
        secrets. The work-tree capture calls it at finish. (No tool adapter records calls in this build,
        so no credential is learned from a call's response before ``assemble``.)"""
        from ..redact import Redactor

        return Redactor.from_environ(os.environ)

    # -- cell results -----------------------------------------------------------------------------

    def note_result(self, call_id: Any, result: Any) -> None:
        """Keep a code cell's status from its ``ExecutionResult`` (``hooks.tool_result``), reduced at once
        to names (``analysis.use.runtime_status``): the use record's only source of refusals, errors and
        sessions. A cell whose error named the library's package is counted (:func:`library_cell_error`).
        A plain mapping (the code tool's own result for an empty cell, or when its session executor
        raised) is read by ``analysis.use.dict_status``. At most ``MAX_STATUSES`` calls are
        kept; later ones are then unknown.
        """
        from ..analysis import use

        kind = library_cell_error(
            result.get("error")
            if isinstance(result, Mapping)
            else getattr(result, "error", None),
        )
        if kind is not None:
            self.cell_errors[kind] = self.cell_errors.get(kind, 0) + 1
        if call_id is None or len(self.cell_status) >= MAX_STATUSES:
            return
        if isinstance(result, Mapping):
            status = use.dict_status(
                result,
                items=self.item_ids,
                roots=self.export_roots,
                library_mode=self.library_mode,
            )
        else:
            status = use.runtime_status(
                getattr(result, "error", None),
                getattr(result, "session_id", None),
                getattr(result, "session_created", None),
                items=self.item_ids,
                roots=self.export_roots,
                library_mode=self.library_mode,
            )
        self.cell_status[str(call_id)] = status

    def note_calls(self, drained: Any) -> None:
        """Add one cell's drained audit records to the request's runtime call evidence
        (:func:`add_runtime_calls`; ``hooks.worker_cell_done``)."""
        add_runtime_calls(self.runtime_calls, drained)

    def note_failure(self, call_id: Any, exc: BaseException) -> None:
        """Keep that a code cell's tool call raised instead of returning (``hooks.tool_result``): its
        outcome is unknown (``tool_raised``), with the exception's class name when it is a builtin one.
        """
        from ..analysis import use

        if call_id is None or len(self.cell_status) >= MAX_STATUSES:
            return
        self.cell_status[str(call_id)] = use.failed_status(type(exc).__name__)

    # -- the outcome ------------------------------------------------------------------------------

    def take_outcome(self, raw: Any) -> dict:
        """Keep the checker's verdict only; the answer the CLI writes for an ``{"outcome": ...}`` line."""
        from unify import outcome as outcome_mod

        try:
            normalized = outcome_mod.normalize(raw)
        except (
            outcome_mod.OutcomeError
        ) as exc:  # our own text; it never quotes the outcome
            return {"type": "outcome", "accepted": False, "reason": str(exc)}
        solved, checks = normalized["solved"], int(normalized["checks_total"])
        del normalized, raw
        self.solved = solved if isinstance(solved, bool) else None
        self.outcome_at = _now()
        return {
            "type": "outcome",
            "accepted": True,
            "solved": self.solved,
            "checks": checks,
        }

    def take_checker(self, raw: Any) -> dict:
        """Keep one verdict the bed showed the actor; the answer the CLI writes for a
        ``{"checker": {"label": "pass"|"fail"}}`` line. The bed sends it just before the message that shows the
        actor the same verdict, so the transcript's observation count now is that message's ordinal. Refused,
        with nothing kept, unless ``UNIFY_MEMORY_LIBRARY`` and ``UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE`` are on, or when
        the line is anything but exactly ``{"label": "pass"}`` or ``{"label": "fail"}``. The answer never quotes
        the line."""
        if not checker_visible_on():
            return {
                "type": "checker",
                "accepted": False,
                "reason": "this session takes no checker lines (UNIFY_MEMORY_LIBRARY_CHECKER_VISIBLE is off)",
            }
        if (
            not isinstance(raw, dict)
            or set(raw) != {"label"}
            or raw["label"] not in ("pass", "fail")
        ):
            return {
                "type": "checker",
                "accepted": False,
                "reason": 'a checker line is exactly {"label": "pass"} or {"label": "fail"}',
            }
        self.checker_seen.append(
            {"label": raw["label"], "obs": self._observations_now(), "at": _now()},
        )
        return {
            "type": "checker",
            "accepted": True,
            "label": raw["label"],
            "count": len(self.checker_seen),
        }

    def _observations_now(self) -> int | None:
        """How many observations this request's transcript holds now, or None when it cannot be read."""
        from unify import transcripts

        from . import trajectory
        from .adapters import dialogue

        try:
            lines = trajectory.read_jsonl(
                transcripts.transcripts_dir() / f"{self.episode_id}.jsonl",
            )
        except (OSError, ValueError):
            return None
        return dialogue.observation_count(lines)

    # -- finish -----------------------------------------------------------------------------------

    @staticmethod
    def _emitter(emit: Callable[[dict], None] | None) -> Callable[[dict], None] | None:
        """*emit* (the CLI's ``--jsonl`` output) with money as plain decimal strings, or None. The
        consolidation driver also appends every event to the events file (``Paths.events``).
        """
        if emit is None:
            return None
        return lambda event: emit(_plain(dict(event)))

    def _held(
        self,
        eid: str,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        """The episode is recorded; the driver asked that no pass start (it is ending its stream)."""
        from . import consolidate

        event = {
            "type": "consolidation",
            "phase": "held",
            "episode_id": eid,
            "reason_codes": ["quit_without_consolidation"],
        }
        consolidate._deliver(
            self.stores,
            self._emitter(emit),
            event,
        )  # the events file and --jsonl
        progress(
            "memory v2: episode recorded; no consolidation pass started (quit without consolidation)",
        )

    def _error(
        self,
        stage: str,
        exc: BaseException,
        progress: Callable[[str], None],
    ) -> None:
        from ..redact import redact_error

        text = redact_error(f"{type(exc).__name__}: {exc}")[:_ERROR_CHARS]
        row = {
            "ts": _now(),
            "episode_id": self.episode_id,
            "stage": stage,
            "error": text,
        }
        try:
            self.paths.errors.parent.mkdir(parents=True, exist_ok=True)
            with self.paths.errors.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError:
            logger.warning("memory v2: %s failed: %s", stage, text)
        try:
            progress(f"memory v2: {stage} failed: {type(exc).__name__}")
        except Exception:  # noqa: BLE001
            pass

    async def finish(
        self,
        handle: Any,
        *,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None = None,
        consolidate: bool = True,
    ) -> None:
        """Record the request and run the due passes (blocking; under ``UNIFY_MEMORY_LIBRARY=on`` a due pass starts
        in a detached worker and the request returns at once); never raises. The transcript is the record;
        *handle* is read only by the episode-end fork (``UNIFY_MEMORY_LIBRARY_FORK=on``), which starts
        after the episode is recorded and changes nothing in it. With *consolidate* False (a driver's
        ``{"quit": true, "consolidate": false}``) the episode is recorded and no pass starts; due passes stay
        due, and one ``held`` event says so."""
        if self._closed:
            return
        fork = reservation = None
        try:
            self._close_scope()
            # the fork's staging dir and lock exist before the episode is recorded, so admission never sees a
            # recorded episode whose fork has not been reserved
            reservation = self._reserve_fork(progress)
            try:
                eid, sha = self._record_episode()
            except Exception as exc:  # noqa: BLE001
                self._error("episode", exc, progress)
                if reservation is not None:
                    reservation.abandon("refused: the episode was not recorded")
                return
            # the recorded episode's part list, read before the similar index takes the recorded episode
            parts = self._fork_parts(reservation, progress)
            self._index_similar(progress)
            fork = self._start_fork(handle, eid, progress, reservation, parts)
            if consolidate:
                await self._consolidate(eid, sha, progress, emit)
            else:
                self._report_landed(progress, emit)
                self._held(eid, progress, emit)
        except BaseException as exc:
            self._error("finish", exc, progress)
            if not isinstance(exc, Exception):
                raise
        finally:
            if reservation is not None:
                reservation.close()  # a no-op once the fork has started or settled
            self._cleanup()
        if getattr(self, "_spawned", False):
            from unify.settings import SETTINGS

            # after the lock is released: the waiting CLI holds nothing a later request needs
            await wait_after_finish(
                self.paths,
                SETTINGS,
                progress,
                relay=getattr(self, "_relay", None),
                events_from=getattr(self, "_events_from", 0),
            )
        if fork is not None:
            from unify.settings import SETTINGS

            from . import fork as _fork

            # after the reply and the lock: hand over the request, and wait where the host needs it
            await asyncio.to_thread(_fork.finish, fork, SETTINGS, progress)

    def _index_similar(self, progress: Callable[[str], None]) -> None:
        """The ``similar`` tool's index entry for the episode just recorded (library mode only), built here once,
        under its time limit, so no writer pass spends its time on it; never raises."""
        recorded, self._recorded = getattr(self, "_recorded", None), None
        if recorded is None or not memory_library_on():
            return
        try:
            from .. import similar

            similar.record_episode(self.paths.state_dir, *recorded)
        except Exception as exc:  # noqa: BLE001 - the index never stops a request
            self._error("similar index", exc, progress)

    def _reserve_fork(self, progress: Callable[[str], None]) -> Any:
        """The episode-end fork's staging dir and lock, only under ``UNIFY_MEMORY_LIBRARY_FORK=on``; never raises."""
        from unify.settings import SETTINGS

        from .switch import library_enabled, library_fork

        if not (library_enabled(SETTINGS) and library_fork(SETTINGS)):
            return None
        try:
            from . import fork as _fork

            return _fork.reserve(self.paths, self.episode_id, SETTINGS, progress)
        except Exception as exc:  # noqa: BLE001
            self._error("fork", exc, progress)
            return None

    def _fork_parts(self, reservation: Any, progress: Callable[[str], None]) -> str | None:
        """The episode just recorded as the fork's part list (:func:`.fork.part_list`), only when its fork is
        reserved; None otherwise or when it cannot be read (the fork then names parts by its numbering rule alone).
        Never raises."""
        recorded = getattr(self, "_recorded", None)
        if reservation is None or recorded is None:
            return None
        try:
            from . import fork as _fork

            return _fork.part_list(recorded[0])
        except Exception as exc:  # noqa: BLE001 - the part list never stops a request
            self._error("fork part list", exc, progress)
            return None

    def _start_fork(
        self,
        handle: Any,
        eid: str,
        progress: Callable[[str], None],
        reservation: Any = None,
        parts: str | None = None,
    ) -> Any:
        """The episode-end fork, only under ``UNIFY_MEMORY_LIBRARY_FORK=on``; never raises."""
        from unify.settings import SETTINGS

        from .switch import library_enabled, library_fork

        if not (library_enabled(SETTINGS) and library_fork(SETTINGS)):
            return None
        try:
            from . import fork as _fork

            return _fork.start(
                self.paths,
                handle,
                eid,
                SETTINGS,
                progress=progress,
                reservation=reservation,
                parts=parts,
            )
        except Exception as exc:  # noqa: BLE001
            self._error("fork", exc, progress)
            if reservation is not None:
                reservation.abandon(f"error: {type(exc).__name__}")
            return None

    def _record_episode(self) -> tuple[str, str]:
        from unify import transcripts

        from ..episodes import EpisodeWriter
        from . import consolidate, trajectory

        stores, paths = self.stores, self.paths
        lines = trajectory.read_jsonl(
            transcripts.transcripts_dir() / f"{self.episode_id}.jsonl",
        )
        cells = trajectory.timed_cells(trajectory.fold(lines))
        self._worktree_finished = True  # finish ends the capture, whatever it returns
        wt = self.worktree.finish(cells)
        memory_diff = self._memory_diff()
        ended_at = _now()
        extra_actions = wt.actions
        counterpart = dialogue_counterpart()
        if counterpart:
            # UNIFY_MEMORY_DIALOGUE: the transcript's dialogue actions follow the work-tree rows
            extra_actions = [
                *wt.actions,
                *recorded_dialogue(
                    lines,
                    counterpart,
                    self.redactor(),
                    library_mode=memory_library_on(),
                ),
            ]
        ep, redactor = trajectory.assemble(
            self,
            lines,
            memory_diff,
            ended_at,
            extra_actions=extra_actions,
            worktree_before=wt.before,
            worktree_after=wt.after,
            worktree_diff=wt.diff,
        )
        ep.memory_use = memory_use(
            ep,
            self.item_ids,
            redactor=redactor,
            export_roots=self.export_roots,
            surface=self.surface,
            shown=self.shown,
            shown_text=self.index,
            cell_status=dict(self.cell_status),
            library_mode=self.library_mode,
            runtime_calls=self.runtime_calls,
            cell_errors=dict(self.cell_errors),
            memory_bound=self.library_mode and self.memory_bound,
        )
        sha = EpisodeWriter(
            stores.episodes,
            stores.blobs,
            redactor,
            **(
                {"library_mode": True} if memory_library_on() else {}
            ),  # record format 2; off, the call as at 4675a3c45
        ).write(ep)
        stores.evidence.index_episode(ep, sha)
        # the episode and the redactor it was stored with, for the similar index (:meth:`_index_similar`)
        self._recorded = (ep, redactor)
        consolidate.post_checker(
            stores,
            ep.episode_id,
            sha,
            self.solved,
            self.outcome_at or ended_at,
        )
        if self.checker_seen:
            # The verdicts the actor saw, as agent-visible checker signals
            consolidate.post_visible_checkers(
                stores,
                ep,
                sha,
                list(self.checker_seen),
                lines,
                counterpart,
            )
        bumped = self.state.generations.observe(ep.fingerprints or {})
        self.state.drift |= bumped
        self.state.suspect |= bumped
        self.state.save()
        return ep.episode_id, sha

    def _report_landed(
        self,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        """Library mode: one ``landed`` event per worker result applied at open, naming this request (the first
        that ran after it); once."""
        rows, self._landed = self._landed, []
        if not rows:
            return
        try:
            from . import consolidate
            from .async_pass import landed_event

            for row in rows:
                consolidate._deliver(
                    self.stores,
                    self._emitter(emit),
                    landed_event(
                        row,
                        self.episode_id,
                        self.pin,
                        consolidate.library_items_at(self.stores, row.get("commit")),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            self._error("passes", exc, progress)

    async def _consolidate(
        self,
        eid: str,
        sha: str,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        from unify.settings import SETTINGS

        from . import consolidate

        if self.library_mode:  # library mode: never wait for a pass
            self._report_landed(progress, emit)
            try:
                from .async_pass import maybe_spawn
                from .consolidate import events_path

                # WAIT_SLOT's relay starts at the events written from here on (the slot's own rows)
                ev_file = events_path(self.paths)
                self._events_from = ev_file.stat().st_size if ev_file.exists() else 0
                self._relay = self._emitter(emit)
                spawned = maybe_spawn(
                    self.stores,
                    eid,
                    sha,
                    effort=writer_effort(self.effort),
                    settings=SETTINGS,
                    emit=self._emitter(emit),
                )
                self._spawned = spawned.get("phase") == "spawned"
            except Exception as exc:  # noqa: BLE001
                self._error("passes", exc, progress)
            try:
                self.state.save()
            except Exception as exc:  # noqa: BLE001
                self._error("state", exc, progress)
            return
        try:
            await consolidate.run_due_passes(
                self.stores,
                eid,
                sha,
                self.state,
                # The writer's effort matches the actor's unless a mismatch ablation fixes it
                effort=writer_effort(self.effort),
                settings=SETTINGS,
                emit=self._emitter(emit),
                clock=time.monotonic,
            )
        except Exception as exc:  # noqa: BLE001
            self._error("passes", exc, progress)
        finally:
            try:
                self.state.save()
            except Exception as exc:  # noqa: BLE001
                self._error("state", exc, progress)

    # -- abort and cleanup ------------------------------------------------------------------------

    def _close_scope(self) -> None:
        """Leave the actor's scope. Every exit runs; one that cannot (a context variable reset from
        another context, which only a caller outside the CLI's task causes) is logged, never raised.
        """
        try:
            self._scope.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory v2: request scope not closed cleanly: %s", exc)

    def abort(self) -> None:
        """End the run with nothing recorded (the process is closing without a finished request)."""
        self._cleanup()

    def _cleanup(self) -> None:
        global _CURRENT
        if self._closed:
            return
        self._closed = True
        from .checkout import remove_checkout
        from .state import release_lock

        self._close_scope()
        if self.worktree is not None and not self._worktree_finished:
            # A run that never reached the capture's finish (abort, a failed begin or a failed
            # finish before it) ends the capture so the worker installs no audit hook for it.
            self._worktree_finished = True
            try:
                self.worktree.abort()
            except (
                Exception
            ) as exc:  # noqa: BLE001 - abort never raises; belt and braces
                logger.warning("memory v2: work-tree capture not aborted: %s", exc)
        try:
            remove_checkout(self.paths.checkout)
        except OSError as exc:
            logger.warning("memory v2: export not removed: %s", exc)
        if self._lock is not None:
            try:
                release_lock(self._lock)
            except OSError:
                pass
            self._lock = None
        if _CURRENT is self:
            _CURRENT = None
