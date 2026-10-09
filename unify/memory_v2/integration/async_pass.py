"""Asynchronous consolidation for memory v2.1 (spec §6, D43; RUNTIME 11:24Z).

A request never waits for a pass. When a request ends and the experience recorded since the last pass reaches E
(:func:`.switch.v21_experience_budget`), :func:`maybe_spawn` starts one detached worker process
(:mod:`.pass_worker`) and returns. The worker runs the pass under its wall-clock bound (:class:`Supervisor`).

**Isolation.** Each request pins memory ``main`` when it starts (``RequestRun.pin``) and works on a read-only copy
of that commit (P3). A pass works in its own temporary box and checkout, and its merge moves ``main`` with one
compare-and-swap (``Repo.fast_forward``, ``git update-ref <new> <old>``). A request therefore sees the old commit
or the new one, never a half-written library, and a commit takes effect from the next request that starts after
it lands.

**One pass at a time.** The worker holds ``pass.lock`` (an ``flock`` on a descriptor it inherits from the
spawner) for its whole life. ``pass-inflight.json`` names it: pid, process group, ``/proc`` start time (the
ownership check) and deadline.

**Results.** The worker appends one row to ``pass-results.jsonl`` and never writes the request state
(``state.json``). The next request applies the row under its lock (:func:`apply_results`) and reports the pass as
``landed``, naming itself: the first request that ran on the new commit.

**Shutdown** (:func:`drain`) never starts a pass. A pass in flight gets at most until its own deadline; then it
is sent SIGTERM, which the worker turns into a cancellation through the pass's abort path (the work becomes a
draft, the calls are reconciled from the Sol lane's journal window). A worker that has not ended after the grace
period is killed and its termination verified. Every pass that ended after the last request is reported as such and is never
counted toward the stream.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..reconcile import RECONCILE_S, journal_offset

#: The evidence store's SQLite busy timeout while a worker and a request share it (v2's default is 5 s).
BUSY_TIMEOUT_S = 60.0
#: Seconds of the wall bound kept for the final merge (the whole gate, mutation included) after Sol's deadline.
MERGE_RESERVE_S = 600.0
#: Seconds a worker has, after SIGTERM, to cancel its pass (a cell in flight ends within its own 60 s timeout).
CANCEL_GRACE_S = 30.0
#: The worker's SIGALRM failsafe beyond its bound, its grace and its reconciliation.
FAILSAFE_EXTRA_S = 60.0
WORKER_MODULE = "unify.memory_v2.integration.pass_worker"
_ENDED = ("landed", "refused", "deadline", "cancelled", "error")


def failsafe_s(wall_s: float) -> float:
    return float(wall_s) + CANCEL_GRACE_S + RECONCILE_S + FAILSAFE_EXTRA_S


# --- the supervisor ----------------------------------------------------------------------------------------


@dataclass
class Supervised:
    outcome: Any
    ended: str  # landed | refused | deadline | cancelled | error
    seconds: float


class Supervisor:
    """Runs one pass coroutine under the wall-clock bound *wall_s*; *stop* (set by SIGTERM) cancels it early.
    Cancellation goes through the pass's own abort path: :meth:`..sol_pass.SolPass.run` records the pass as
    failed and, under v2.1, keeps its work as a draft."""

    def __init__(
        self,
        wall_s: float,
        stop: asyncio.Event,
        *,
        journal: str | None = None,
        models: tuple[str, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.wall_s, self.stop, self.clock = float(wall_s), stop, clock
        # P7 Amendment D: the Sol route proxy's journal and Sol's models; the pass's window is the journal's byte
        # range from the pass's start to the end this supervisor records
        self.journal, self.models = journal, tuple(models)
        self.window: tuple[int | None, int | None] = (None, None)
        self.last: Supervised | None = None
        self.reconciled: dict | None = None

    @property
    def pass_deadline_s(self) -> float:
        """Sol's own deadline: the wall bound less the final merge's reserve (at least a minute)."""
        return max(60.0, self.wall_s - MERGE_RESERVE_S)

    async def __call__(self, start: Callable[[], Awaitable[Any]]) -> Supervised:
        t0 = self.clock()
        start_at = journal_offset(self.journal)
        task = asyncio.ensure_future(start())
        stopper = asyncio.ensure_future(self.stop.wait())
        try:
            done, _ = await asyncio.wait(
                {task, stopper},
                timeout=self.wall_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not stopper.done():
                stopper.cancel()
        if task in done:
            if task.exception() is not None:
                result = Supervised(None, "error", self.clock() - t0)
            else:
                out = task.result()
                result = Supervised(
                    out,
                    "landed" if getattr(out, "passed", False) else "refused",
                    self.clock() - t0,
                )
        else:
            ended = "cancelled" if self.stop.is_set() else "deadline"
            task.cancel()
            try:
                await task
            except (
                BaseException
            ):  # noqa: BLE001 - the pass recorded itself; its exception is not ours
                pass
            result = Supervised(None, ended, self.clock() - t0)
        self.window = (start_at, journal_offset(self.journal))
        self.last = result
        return result

    async def reconcile(self, **kw: Any) -> dict:
        """The pass's Sol calls priced from the journal window this supervisor recorded (Amendment D: by lane and
        window, no header and no generation lookup); unpriced calls stay unknown and are booked at the worst
        case."""
        from ..reconcile import reconcile_window

        start, end = self.window
        self.reconciled = await reconcile_window(
            self.journal,
            start,
            end,
            models=self.models,
            **kw,
        )
        return self.reconciled


@dataclass
class StateView:
    """The worker's view of the request state. ``run_due_passes`` may clear drift and suspect channels in it; the
    clears go to the result row, and the next request applies them under its lock. The worker never saves
    ``state.json``."""

    drift: set
    suspect: set
    drift0: frozenset
    suspect0: frozenset

    @classmethod
    def load(cls, path: Path) -> "StateView":
        from .state import State

        s = State.load(path)
        return cls(
            set(s.drift),
            set(s.suspect),
            frozenset(s.drift),
            frozenset(s.suspect),
        )

    def cleared(self) -> tuple[list[str], list[str]]:
        return sorted(self.drift0 - self.drift), sorted(self.suspect0 - self.suspect)


# --- the in-flight record, the lock and the result rows -------------------------------------------------------


@dataclass(frozen=True)
class InFlight:
    pass_id: str
    pid: int
    pgid: int
    proc_start: str  # /proc/<pid>/stat field 22: with pid and pgid, the ownership check before any signal
    started_at: float  # wall clock (time.time())
    deadline_at: float  # started_at + the wall bound
    after_episode: str


def inflight_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass-inflight.json"


def results_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass-results.jsonl"


def cursor_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass-results.cursor"


def lock_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass.lock"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_inflight(paths: Any, rec: InFlight) -> None:
    _atomic_write(inflight_path(paths), json.dumps(asdict(rec), sort_keys=True) + "\n")


def read_inflight(paths: Any) -> InFlight | None:
    try:
        return InFlight(**json.loads(inflight_path(paths).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


def clear_inflight(paths: Any, pass_id: str) -> None:
    rec = read_inflight(paths)
    if rec is not None and rec.pass_id == pass_id:
        inflight_path(paths).unlink(missing_ok=True)


def proc_start(pid: int) -> str | None:
    """The start time of a live, non-zombie process *pid* (``/proc/<pid>/stat`` field 22), else None."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = stat.rsplit(")", 1)[-1].split()
    if len(rest) < 20 or rest[0] in ("Z", "X"):
        return None
    return rest[19]


def owns(rec: InFlight) -> bool:
    """Whether *rec* still names a live process: same pid, same start time, same process group. A record without
    a start time owns nothing (the worker's SIGALRM failsafe ends it): no signal on a pid alone.
    """
    if rec.pid <= 0 or not rec.proc_start:
        return False
    start = proc_start(rec.pid)
    if start is None or start != rec.proc_start:
        return False
    try:
        return os.getpgid(rec.pid) == rec.pgid
    except OSError:
        return False


def try_lock(path: Path) -> int | None:
    """An exclusive non-blocking ``flock`` on *path*, as an inheritable descriptor; None when it is held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.set_inheritable(fd, True)
    return fd


def record_result(paths: Any, row: dict) -> None:
    """Append one worker result row (``pass_id``, ``ended``, ``commit``, the drift and suspect clears, …)."""
    path = results_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
