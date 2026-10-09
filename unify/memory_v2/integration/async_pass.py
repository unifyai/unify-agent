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
import signal
import subprocess
import sys
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


#: The passes one worker slot runs (MAIN, 9 Oct; spec §7): the WRITE pass, then the CURATE pass it made due.
WORKER_PASSES = 2


def slot_s(wall_s: float, passes: int = WORKER_PASSES) -> float:
    """A worker slot's deadline: each pass's wall bound and each item-records step's bisect budget (P5)."""
    from ..item_bisect import BISECT_BUDGET_S

    return passes * (float(wall_s) + BISECT_BUDGET_S)


def failsafe_s(wall_s: float, passes: int = 1) -> float:
    """The worker's failsafe: *passes* passes, each with its wall bound, cancel grace and reconcile, plus a margin;
    a slot of several passes also each item-records step's bisect budget (P5; MAIN, 9 Oct).
    """
    from ..item_bisect import BISECT_BUDGET_S

    bisect = passes * BISECT_BUDGET_S if passes > 1 else 0.0
    return (
        passes * (float(wall_s) + CANCEL_GRACE_S + RECONCILE_S)
        + bisect
        + FAILSAFE_EXTRA_S
    )


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
        # every pass of the slot, in order, with its own reconciliation (WRITE's is never overwritten by CURATE's)
        self.runs: list[tuple[Supervised, dict | None]] = []

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
        self.runs.append((result, None))
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
        if self.runs:
            self.runs[-1] = (self.runs[-1][0], self.reconciled)
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


# --- spawning (the request side) ---------------------------------------------------------------------------


def spawn(
    paths: Any,
    eid: str,
    sha: str,
    effort: str,
    *,
    wall_s: float,
    popen: Callable[..., Any] = subprocess.Popen,
    argv_prefix: list[str] | None = None,
    clock: Callable[[], float] = time.time,
) -> InFlight | None:
    """Start the worker in its own session, holding the pass lock through an inherited descriptor; None when the
    lock is held (a worker is running, starting or ending). Credentials never enter argv.
    """
    fd = try_lock(lock_path(paths))
    if fd is None:
        return None
    log = None
    try:
        log = open(Path(paths.state_dir) / "pass-worker.log", "ab")
        argv = [
            *(argv_prefix or [sys.executable, "-m", WORKER_MODULE]),
            "--home",
            str(paths.home),
            "--episode",
            eid,
            "--sha",
            sha,
            "--effort",
            effort,
            "--lock-fd",
            str(fd),
        ]
        proc = popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            pass_fds=(fd,),
            close_fds=True,
        )
        now = clock()
        rec = InFlight(
            f"{eid}.p0",
            proc.pid,
            proc.pid,
            proc_start(proc.pid) or "",
            now,
            now
            + slot_s(
                wall_s,
            ),  # the slot: WRITE, then the CURATE it made due, each with its records
            eid,
        )
        write_inflight(paths, rec)
        return rec
    finally:
        os.close(fd)  # the worker keeps the lock on the descriptor it inherited
        if log is not None:
            log.close()


def maybe_spawn(
    stores: Any,
    eid: str,
    sha: str,
    *,
    effort: str,
    settings: Any,
    emit: Callable[[dict], None] | None,
    popen: Callable[..., Any] = subprocess.Popen,
    argv_prefix: list[str] | None = None,
) -> dict:
    """At a v2.1 request's end: start a pass if one is due and none is in flight, and return at once (spec §6).
    Delivers one event (``spawned`` or ``busy``) and returns it; ``not_due`` and ``error`` deliver nothing.
    Never raises: a failure is recorded in ``errors.jsonl``, and the request stays due.
    """
    from ..trigger import Trigger
    from .consolidate import _deliver, _error
    from .switch import v21_experience_budget, v21_pass_wall_s

    paths = stores.paths
    try:
        rec = read_inflight(paths)
        if rec is not None and owns(rec):
            event = {
                "type": "consolidation",
                "phase": "busy",
                "episode_id": eid,
                "pass_id": rec.pass_id,
            }
            _deliver(stores, emit, event)
            return event
        budget = v21_experience_budget(settings)
        due = Trigger(
            stores.evidence,
            mode="batched",
            experience_budget=budget,
        ).after_episode(eid)
        if not due:
            return {"phase": "not_due"}
        wall_s = v21_pass_wall_s(settings)
        rec = spawn(
            paths,
            eid,
            sha,
            effort,
            wall_s=wall_s,
            popen=popen,
            argv_prefix=argv_prefix,
        )
        if rec is None:
            event = {
                "type": "consolidation",
                "phase": "busy",
                "episode_id": eid,
                "pass_id": None,
            }
        else:
            event = {
                "type": "consolidation",
                "phase": "spawned",
                "episode_id": eid,
                "pass_id": rec.pass_id,
                "trigger_tokens": due[0].experience_tokens,
                "experience_budget": budget,
                "wall_s": wall_s,
            }
        _deliver(stores, emit, event)
        return event
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a request's end never fails on consolidation
        _error(stores, f"spawn: {type(exc).__name__}: {exc}")
        return {"phase": "error"}


# --- results (the next request) ----------------------------------------------------------------------------


def _cursor(paths: Any) -> int:
    try:
        return int(cursor_path(paths).read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def _rows_from(paths: Any, offset: int) -> tuple[list[dict], int]:
    """Complete result lines from byte *offset* (a line still being written waits), and the offset after them."""
    try:
        with open(results_path(paths), "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1
    rows = []
    for line in data[:end].splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("ended") in _ENDED:
            rows.append(row)
    return rows, offset + end


def unapplied(paths: Any) -> list[dict]:
    return _rows_from(paths, _cursor(paths))[0]


def apply_results(state: Any, paths: Any) -> list[dict]:
    """Apply the worker rows no request has applied yet (drift and suspect clears), save the state, then move the
    cursor. Called under the request lock at a v2.1 request's open."""
    rows, end = _rows_from(paths, _cursor(paths))
    if not rows and end == _cursor(paths):
        return []
    for row in rows:
        state.drift.difference_update(row.get("drift_cleared") or [])
        state.suspect.difference_update(row.get("suspect_cleared") or [])
    state.save()
    _atomic_write(cursor_path(paths), f"{end}\n")
    return rows


def landed_event(row: dict, episode_id: str, pin: str) -> dict:
    """The ``landed`` event: the pass, how it ended, and the first request that ran after it."""
    return {
        "type": "consolidation",
        "phase": "landed",
        "pass_id": row.get("pass_id"),
        "ended": row.get("ended"),
        "commit": row.get("commit"),
        "first_request": episode_id,
        "pinned": pin,
    }


# --- shutdown ------------------------------------------------------------------------------------------------


def _result_for(paths: Any, pass_id: str) -> dict | None:
    rows, _ = _rows_from(paths, 0)
    found = [r for r in rows if r.get("pass_id") == pass_id]
    return found[-1] if found else None


def _event(paths: Any, row: dict) -> None:
    path = Path(paths.state_dir) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def _group_alive(pgid: int) -> bool | None:
    """Whether a live (not zombie) process is in process group *pgid*, from ``/proc``; None when it cannot be
    read (unknown, so termination is not verified)."""
    try:
        names = os.listdir("/proc")
    except OSError:
        return None
    for name in names:
        if not name.isdigit():
            continue
        try:
            stat = Path(f"/proc/{name}/stat").read_text()
        except OSError:
            continue  # ended while we looked
        rest = stat.rsplit(")", 1)[-1].split()  # state, ppid, pgrp, ...
        if len(rest) > 2 and rest[0] not in ("Z", "X") and rest[2] == str(int(pgid)):
            return True
    return False


def wait_slot(
    paths: Any,
    wall_s: float,
    *,
    budget_s: float | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = time.sleep,
    drain_fn: Callable[..., dict] | None = None,
    relay: Callable[[dict], None] | None = None,
    events_from: int = 0,
) -> dict:
    """``UNIFY_MEMORY_V21_WAIT_SLOT=on``: wait for the in-flight worker until it ends, at most its slot's failsafe
    (:func:`failsafe_s` of :data:`WORKER_PASSES` passes, their bisects and reconciliations) plus a margin from its
    start (*budget_s* replaces that bound). A worker still alive then goes through :func:`drain` at once: SIGTERM,
    the grace, a verified end; its late result is reported as not counted. Nothing in flight: returns at once.

    *relay* (MAIN, 10 Oct): the slot's pass events (``start``, ``end`` and ``held`` rows of its passes, WRITE and
    CURATE) appended to ``events.jsonl`` after byte *events_from* are handed to it once each, in order, exactly as
    the synchronous v2 path emitted them (the same dicts :func:`.consolidate._deliver` wrote).
    """
    rec = read_inflight(paths)
    if rec is None or not owns(rec):
        return {"waited": False}
    bound = (
        failsafe_s(wall_s, passes=WORKER_PASSES) + FAILSAFE_EXTRA_S
        if budget_s is None
        else float(budget_s)
    )
    deadline = rec.started_at + bound
    tail = (
        _EventTail(paths, events_from, f"{rec.after_episode}.p")
        if relay is not None
        else None
    )
    while owns(rec) and clock() < deadline:
        if tail is not None:
            tail.relay(relay)
        sleep(0.5)
    out: dict = {"waited": True, "ended": "worker_ended"}
    if owns(rec) or _group_alive(rec.pgid) is True:
        out = {
            "waited": True,
            "ended": "drained",
            "drain": (drain_fn or drain)(paths, wait_s=0),
        }
    if tail is not None:
        tail.relay(relay)  # the rows written as the worker ended (or was drained)
    return out


_RELAYED = ("start", "end", "held")


class _EventTail:
    """Complete ``events.jsonl`` lines after a byte offset; the slot's pass rows, each relayed once."""

    def __init__(self, paths: Any, offset: int, prefix: str) -> None:
        from .consolidate import events_path

        self.path, self.offset, self.prefix = events_path(paths), int(offset), prefix

    def relay(self, out: Callable[[dict], None]) -> None:
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return
        end = data.rfind(b"\n") + 1  # a line still being written waits
        self.offset += end
        for line in data[:end].splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if (
                isinstance(row, dict)
                and row.get("type") == "consolidation"
                and row.get("phase") in _RELAYED
                and str(row.get("pass_id") or "").startswith(self.prefix)
            ):
                try:
                    out(row)
                except Exception:  # noqa: BLE001 - reporting never stops the wait
                    pass


def drain(
    paths: Any,
    *,
    wait_s: float | None = None,
    grace_s: float = CANCEL_GRACE_S + RECONCILE_S,
    settle_s: float = 5.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Any] = time.sleep,
    kill: Callable[[int, int], None] = os.killpg,
    archive_to: Path | None = None,
) -> dict:
    """Shutdown (spec §6): never starts a pass. A pass in flight may run until its own deadline (or *wait_s*, if
    sooner); then SIGTERM cancels it through the abort path; a worker still alive after *grace_s* is killed and its
    end verified within *settle_s*. Every result no request applied is reported under ``after_last_request`` and
    marked ``counted: false``. With *archive_to*, once no worker can write, the memory repo is archived with every
    ref, ``refs/notes/items`` and ``served`` included (:func:`..memory_writer.archive`; a plain clone would drop
    the item records). The report is also appended to ``events.jsonl``.

    Signals go only to the worker's process group, and only after its leader was found alive by :func:`owns`
    (pid, start time, group) in this call; the end is verified when the leader is gone and no live process is
    left in its group. A stale record (its worker already gone) is cleared without any signal.
    """
    report: dict = {
        "type": "consolidation",
        "phase": "shutdown",
        "in_flight": False,
        "pass_id": None,
        "ended": "none",
        "waited_s": 0,
        "terminated": None,
        "after_last_request": [],
        "counted": False,
        "sigkill": False,  # whether the worker's group had to be killed
    }
    rec = read_inflight(paths)
    if rec is not None and not owns(
        rec,
    ):  # the worker ended (or a stale record): nothing to wait for
        row = _result_for(paths, rec.pass_id)
        report.update(pass_id=rec.pass_id, ended=(row or {}).get("ended", "none"))
        clear_inflight(paths, rec.pass_id)
        rec = None
    if rec is not None:
        report.update(in_flight=True, pass_id=rec.pass_id)

        def alive() -> bool:
            return owns(rec) or _group_alive(rec.pgid) is True

        def signal_group(sig: int) -> None:
            try:
                kill(rec.pgid, sig)
            except ProcessLookupError:
                pass  # the group ended in between

        start = clock()
        limit = (
            rec.deadline_at
            if wait_s is None
            else min(rec.deadline_at, start + float(wait_s))
        )
        while owns(rec) and clock() < limit:
            sleep(0.1)
        report["waited_s"] = round(clock() - start, 3)
        killed = False
        if alive():
            signal_group(
                signal.SIGTERM,
            )  # the worker cancels through the abort path and reconciles
            end = clock() + float(grace_s)
            while alive() and clock() < end:
                sleep(0.1)
        if alive():
            signal_group(signal.SIGKILL)
            killed = True
            end = clock() + float(settle_s)
            while alive() and clock() < end:
                sleep(0.05)
        terminated = not owns(rec) and _group_alive(rec.pgid) is False
        if not terminated:
            report.update(ended="not_terminated", terminated=False)
        else:  # the worker's own result row when it wrote one (a straggler killed after it does not change it)
            row = _result_for(paths, rec.pass_id)
            report.update(ended=(row or {}).get("ended", "killed"), terminated=True)
            clear_inflight(paths, rec.pass_id)
        report["sigkill"] = killed
    report["after_last_request"] = [
        {
            "pass_id": r.get("pass_id"),
            "ended": r.get("ended"),
            "commit": r.get("commit"),
        }
        for r in unapplied(paths)
    ]
    # never while a worker may still write
    if archive_to is not None and report["terminated"] is not False:
        from ..gitio import Repo
        from ..memory_writer import archive

        try:
            report["archive"] = archive(Repo(Path(paths.memory)), Path(archive_to))
        except Exception as exc:  # noqa: BLE001
            # reported, never raised: shutdown must finish
            report["archive"] = {"error": type(exc).__name__}
    _event(paths, report)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="async_pass")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "drain",
        help="shutdown: bounded wait, cancel, verified termination; never starts a pass",
    )
    d.add_argument("--home", required=True)
    d.add_argument("--wait-s", type=float, default=None)
    d.add_argument(
        "--archive",
        default=None,
        help="a bundle path: every ref, refs/notes/items included",
    )
    args = p.parse_args(argv)
    from .paths import Paths

    report = drain(
        Paths.under(Path(args.home)),
        wait_s=args.wait_s,
        archive_to=Path(args.archive) if args.archive else None,
    )
    print(json.dumps(report, sort_keys=True))
    return 2 if report["terminated"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
