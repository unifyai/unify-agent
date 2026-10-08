"""The memory run of the request in progress (one request per CLI process; integration Task 25).

``RequestRun.begin`` opens a run under ``UNIFY_MEMORY_V2=on``: it takes the request lock, exports memory
``main`` into the scratch export the worker mounts, renders the memory section the system prompt ends with
(``index``: under ``UNIFY_MEMORY_V2_SURFACING=index``, the default, the v2 per-function index and export
line; under ``catalogue``, the guide paragraph and the channel catalogue, after writing the generated
catalogue beside the export: README, ``.memory/catalog.json`` and the ``memory`` helper,
:mod:`..catalogue`), takes the work tree's before snapshot, and opens the scope the actor runs in (its
transcript continues the episode id and model costs are recorded). ``finish`` records the request as one
episode and runs the consolidation passes that are due, blocking; it never raises. The passes'
start and end events go to the CLI's ``--jsonl`` output when it has one; the consolidation driver
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
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CURRENT: Any = None

#: The reasoning effort the actor's client gets when nothing sets one (unify/common/llm_client.py
#: ``_build_llm_client``'s ``reasoning_effort`` default).
CLIENT_DEFAULT_EFFORT = "high"
LOCK_TIMEOUT_S = 60.0
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


def actor_effort() -> str:
    """The actor's reasoning effort for this run, read from the setting the actor's client reads.

    Sol's passes run at this effort (D23): effort is a fixed condition of a run, never a lever.
    """
    from unify.common.llm_client import resolve_default_model

    return resolve_default_model()[1] or CLIENT_DEFAULT_EFFORT


def build_id() -> str:
    """The harness checkout's commit, or ``unknown`` when it is not a git checkout."""
    root = Path(__file__).resolve().parents[3]
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
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
        export_checkout(paths.memory, self.pin, paths.checkout)
        self.surfacing = surfacing_options(SETTINGS)
        if self.surfacing.catalogue:
            self._surface_catalogue(consolidate)
        else:  # the v2 index and export line, byte for byte; nothing generated in the export
            from .prompt import render_index

            self.index = render_index(
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
        commit's input shapes frozen) and the guide plus channel catalogue as the prompt's memory section.
        """
        from ..catalogue import GENERATED, write_generated
        from ..shape_rows import lookup_from
        from .prompt import render_memory_section

        checkout = self.paths.checkout
        try:
            self.generated = write_generated(
                checkout,
                shapes=lookup_from(self._shape_rows(consolidate)),
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
        self.index = render_memory_section(checkout, self.state.suspect)

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

    def _error(
        self,
        stage: str,
        exc: BaseException,
        progress: Callable[[str], None],
    ) -> None:
        text = f"{type(exc).__name__}: {exc}"[:_ERROR_CHARS]
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
    ) -> None:
        """Record the request and run the due passes (blocking); never raises. *handle* is unused: the
        transcript is the record."""
        if self._closed:
            return
        try:
            self._close_scope()
            try:
                eid, sha = self._record_episode()
            except Exception as exc:  # noqa: BLE001
                self._error("episode", exc, progress)
                return
            await self._consolidate(eid, sha, progress, emit)
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
        ep, redactor = trajectory.assemble(
            self,
            lines,
            memory_diff,
            ended_at,
            extra_actions=wt.actions,
            worktree_before=wt.before,
            worktree_after=wt.after,
            worktree_diff=wt.diff,
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
                effort=self.effort,
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
