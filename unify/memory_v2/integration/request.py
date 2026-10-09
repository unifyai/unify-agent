"""The memory run of the request in progress (one request per CLI process; integration Task 25).

``RequestRun.begin`` opens a run under ``UNIFY_MEMORY_V2=on``: it takes the request lock, exports memory
``main`` into the scratch export the worker mounts, notes its memory functions (for the use record),
renders the memory section the system prompt ends with (``index``: under ``UNIFY_MEMORY_V2_SURFACING=index``,
the default, the v2 per-function index and export line; under ``catalogue``, the constant guide paragraph
once the library has listed anything in this run, after writing the generated catalogue beside the export:
README, ``.memory/catalog.json`` with the suspect flags, and the ``memory`` helper, :mod:`..catalogue`) and
records what it shows in either mode (``shown``: names and a digest, :func:`..analysis.use.record_shown`;
under ``catalogue`` that is the guide alone), takes the work tree's before snapshot, and opens the scope the
actor runs in (its transcript continues the episode id and model costs are recorded). ``finish`` records
the request as one episode (with ``UNIFY_MEMORY_V2_DIALOGUE`` set, its transcript's dialogue actions too:
:mod:`.adapters.dialogue`), with how it used the library (``memory_use.json``, indexed in the evidence
store's ``item_use`` table), and runs the consolidation passes that are due, blocking; it never raises. The
passes' start and end events go to the CLI's ``--jsonl`` output when it has one; the consolidation driver
appends them to the state directory's ``events.jsonl`` (``Paths.events``) either way. ``abort`` cleans up
and records nothing. The harness hooks read the current run (``current()``) for ``index`` and ``paths``.

Only pass/fail of a posted outcome is kept (ruling R10): ``take_outcome`` keeps ``solved`` and drops
everything else at once, so no checker text reaches an episode, the evidence, Sol or a cell.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
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
#: The most code-cell statuses a run keeps (use telemetry); every cell of a request fits many times over.
MAX_STATUSES = 10_000
_ERROR_CHARS = 2000


class MemoryV2Unavailable(RuntimeError):
    """``UNIFY_MEMORY_V2=on`` cannot run here; the message names what to change."""


def current() -> Any:
    return _CURRENT


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def actor_model() -> str:
    """The model the actor runs on for this process (the CLI's actor takes the session default)."""
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[0]


def sol_effort(actor: str) -> str:
    """Sol's reasoning effort for this run: ``UNIFY_MEMORY_V2_SOL_EFFORT`` when it fixes one (a declared mismatch
    ablation), else the actor's (*actor*; the default ``actor``)."""
    from unify.settings import SETTINGS

    value = (
        str(getattr(SETTINGS, "UNIFY_MEMORY_V2_SOL_EFFORT", "") or "actor")
        .strip()
        .lower()
    )
    return actor if value == "actor" else value


def actor_effort() -> str:
    """The actor's reasoning effort for this run, read from the setting the actor's client reads.

    Sol's passes run at this effort too unless ``UNIFY_MEMORY_V2_SOL_EFFORT`` fixes another (:func:`sol_effort`).
    """
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[1] or CLIENT_DEFAULT_EFFORT


def build_id() -> str:
    """The harness checkout's commit, or ``unknown`` when it is not a git checkout."""
    root = Path(__file__).resolve().parents[3]
    from ..gitio import git_child_env

    env = git_child_env(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        },
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
    """The dialogue counterpart ``UNIFY_MEMORY_V2_DIALOGUE`` names (``env``), or ``""`` when it is off."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_MEMORY_V2_DIALOGUE", "") or ""


def recorded_dialogue(lines: list[dict], counterpart: str, redactor: Any) -> list:
    """The dialogue actions a request's episode records under ``UNIFY_MEMORY_V2_DIALOGUE``: answered replies
    only, so a reply nothing followed (a single-turn request's final reply) adds no action and leaves the
    episode as it is with the setting off.
    """
    from .adapters import dialogue

    return dialogue.dialogue_actions(
        lines,
        counterpart,
        redactor=redactor,
        answered_only=True,
    )


def check_hidden(paths: Any, policy: Any) -> None:
    """Refuse to run while a cell could read any harness-side memory path."""
    readable = [
        str(p) for p in paths.harness_only() if policy.readable_violation(p) is None
    ]
    if readable:
        raise MemoryV2Unavailable(
            "UNIFY_MEMORY_V2 needs its state hidden from cells, but the sandbox would let a cell "
            f"read {readable}; keep UNIFY_HOME outside the workspace and every mount",
        )


def pinned_items(checkout: Path) -> list[str]:
    """The ids of the memory functions in the export at *checkout* (listed or not), in id order."""
    from ..memory_repo import items

    return sorted(
        it.item_id for it in items(Path(checkout)).items if it.kind == "env_function"
    )


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
) -> dict:
    """The request's use record (:func:`..analysis.use.request_use`).

    It reads the transcript, actions and ``memory.diff`` as the episode writes them (through *redactor*,
    so the offline analyser recomputes the same record from the written episode), against the items at
    the pin, the export's roots and the library's import surface taken before the actor ran, *shown*,
    what the memory-section renderer recorded (its digest is taken again over *shown_text*, the section
    as rendered, through *redactor*, as the transcript's copy of the prompt is), and *cell_status*, each
    cell's status from the runtime's structured result by tool call id (``RequestRun.note_result``). A
    failure is recorded as its exception type, with the items at the pin; it never stops the episode.
    """
    from ..analysis import use

    def clean(value: Any) -> Any:
        return redactor.obj(value) if redactor is not None else value

    try:
        if isinstance(shown, dict) and redactor is not None and shown_text:
            shown = {**shown, **use.section_digest(redactor.text(shown_text))}
        return use.request_use(
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
        )
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


class RequestRun:
    """One request's memory run; create it with :meth:`begin`."""

    def __init__(self, request: str, paths: Any) -> None:
        self.request = request
        self.paths = paths
        self.index = ""  # the memory section of the system prompt (prompt.render_index or render_memory_section)
        # the export's generated files (relative path -> bytes), left out of memory.diff while unchanged
        self.generated: dict[str, bytes] = {}
        self.surfacing: Any = (
            None  # the v2.1 switches read at open (switch.SurfacingOptions)
        )
        # What the memory section shows (``analysis.use.record_shown``), set where it is rendered (both modes).
        self.shown: dict | None = None
        # The memory functions of the export at the pin, taken before the actor runs (use telemetry).
        self.item_ids: list[str] = []
        self.surface: dict | None = None
        self.export_roots: list[str] = []
        # Each code cell's status from the runtime's structured result, by tool call id (note_result).
        self.cell_status: dict[str, dict] = {}
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
        # Only the checker's verdict of a posted outcome (R10): True, False or None (none posted).
        self.solved: bool | None = None
        self.outcome_at: str | None = None
        self._lock: int | None = None
        self._scope = contextlib.ExitStack()
        self._closed = False

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
        from .switch import surfacing_options

        paths = self.paths
        policy = sandbox.build_policy(fresh=True)
        check_hidden(paths, policy)
        self.stores = consolidate.open_stores(paths)
        self.state = State.load(paths.state)
        self.pin = self.stores.memory.head()
        # the library test kit beside it when the library's tests use it (memory v2.1 stage 5)
        export_checkout(paths.memory, self.pin, paths.checkout, self.stores.blobs)
        # the use record's view of the pin, taken before anything is generated beside the export and
        # before the actor can edit the scratch copy
        self.item_ids = pinned_items(paths.checkout)
        self.surface = use.library_surface(paths.checkout)
        self.export_roots = use.roots_of(paths.checkout)
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
        )
        self.worktree.begin()
        self._scope.enter_context(transcripts.resume_session(self.episode_id))
        self.costs = cost.install()
        self.costs.activate()
        self._scope.callback(self.costs.deactivate)

    def _surface_catalogue(self, consolidate: Any) -> None:
        """``UNIFY_MEMORY_V2_SURFACING=catalogue``: the generated catalogue beside the export (the pinned
        commit's input shapes frozen, the suspect channels flagged) and the constant guide as the prompt's
        memory section, from the first request whose library lists anything (then kept: ``State.guide``).
        """
        from ..catalogue import GENERATED, write_generated
        from ..shape_rows import lookup_from
        from .prompt import render_catalogue
        from .switch import observations_on

        checkout = self.paths.checkout
        try:
            self.generated = write_generated(
                checkout,
                shapes=lookup_from(self._shape_rows(consolidate)),
                suspect=self.state.suspect,
                observations=observations_on(),
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

    def _shape_rows(self, consolidate: Any) -> dict:
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
        sessions. A plain mapping (the code tool's own result for an empty cell, or when its session
        executor raised) is read by ``analysis.use.dict_status``. At most ``MAX_STATUSES`` calls are
        kept; later ones are then unknown.
        """
        from ..analysis import use

        if call_id is None or len(self.cell_status) >= MAX_STATUSES:
            return
        if isinstance(result, Mapping):
            status = use.dict_status(
                result,
                items=self.item_ids,
                roots=self.export_roots,
            )
        else:
            status = use.runtime_status(
                getattr(result, "error", None),
                getattr(result, "session_id", None),
                getattr(result, "session_created", None),
                items=self.item_ids,
                roots=self.export_roots,
            )
        self.cell_status[str(call_id)] = status

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
        """Record the request and run the due passes (blocking); never raises. *handle* is unused: the
        transcript is the record. With *consolidate* False (a driver's ``{"quit": true, "consolidate":
        false}``) the episode is recorded and no pass starts; due passes stay due, and one ``held`` event
        says so."""
        if self._closed:
            return
        try:
            self._close_scope()
            try:
                eid, sha = self._record_episode()
            except Exception as exc:  # noqa: BLE001
                self._error("episode", exc, progress)
                return
            if consolidate:
                await self._consolidate(eid, sha, progress, emit)
            else:
                self._held(eid, progress, emit)
        except BaseException as exc:
            self._error("finish", exc, progress)
            if not isinstance(exc, Exception):
                raise
        finally:
            self._cleanup()

    def _record_episode(self) -> tuple[str, str]:
        from unify import transcripts

        from ..episodes import EpisodeWriter
        from . import consolidate, trajectory
        from .checkout import checkout_diff

        stores, paths = self.stores, self.paths
        lines = trajectory.read_jsonl(
            transcripts.transcripts_dir() / f"{self.episode_id}.jsonl",
        )
        cells = trajectory.timed_cells(trajectory.fold(lines))
        self._worktree_finished = True  # finish ends the capture, whatever it returns
        wt = self.worktree.finish(cells)
        memory_diff = checkout_diff(
            paths.memory,
            self.pin,
            paths.checkout,
            stores.blobs,
            generated=self.generated,
        )
        ended_at = _now()
        extra_actions = wt.actions
        counterpart = dialogue_counterpart()
        if counterpart:
            # UNIFY_MEMORY_V2_DIALOGUE: the transcript's dialogue actions follow the work-tree rows
            extra_actions = [
                *wt.actions,
                *recorded_dialogue(lines, counterpart, self.redactor()),
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
        )
        sha = EpisodeWriter(stores.episodes, stores.blobs, redactor).write(ep)
        stores.evidence.index_episode(ep, sha)
        consolidate.post_checker(
            stores,
            ep.episode_id,
            sha,
            self.solved,
            self.outcome_at or ended_at,
        )
        bumped = self.state.generations.observe(ep.fingerprints or {})
        self.state.drift |= bumped
        self.state.suspect |= bumped
        self.state.save()
        return ep.episode_id, sha

    async def _consolidate(
        self,
        eid: str,
        sha: str,
        progress: Callable[[str], None],
        emit: Callable[[dict], None] | None,
    ) -> None:
        from unify.settings import SETTINGS

        from . import consolidate

        try:
            await consolidate.run_due_passes(
                self.stores,
                eid,
                sha,
                self.state,
                # Sol's effort matches the actor's unless a mismatch ablation fixes it (the lead, 8 Oct)
                effort=sol_effort(self.effort),
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
