"""The writer service: one detached writer per memory home, spawned and owned by the harness, that hands the attention
ledger's entries to writer passes until nothing is open.

**Why a service.** A pass worker started at a request's end lives inside that request's process tree: in the paper
adapter's PID namespace the session's 600 s kill took it down with the request, and a third to a half of the writer's
passes were lost at HIGH and MED. The service outlives requests: :func:`ensure_service` starts it (at a request's start
and end, and at a checkpoint) by a double fork and ``setsid`` into its own session, so no request owns it; it is never
started by a runner or an adapter.

**Exactly one.** The service holds ``flock(service.lock)`` for its whole life (on a descriptor no child inherits). A
second :func:`ensure_service` finds the lock held and leaves it alone. A holder whose heartbeat is older than
:data:`STALE_S` and whose ``owns`` check fails (pid, process group, ``/proc`` start time: a reused pid is never taken
for the service) is a leftover of a dead service: the processes the service recorded (``members``, each by pid and
start time, started no later than its last heartbeat) are killed one by one (verified; a group is signalled only while
the service itself is alive, never by its number alone), a ``takeover`` row is written and a new service starts. A service that died with its lock (``kill -9``) is taken over by the next start the
same way: the new service writes ``takeover`` naming the old one.

**The loop** (:func:`service_main`): sync the ledger; stop at the stop marker; at the cap, every entry still owed is
``not_reached: spend_cap`` and the service idles; with entries open, one pass (``hand(k)`` with the capture grouping,
:mod:`.capture_groups`: other kinds at once, else one due group of related captures; the pass, ``verify_reads``, then
the unread ones go back); else idle, and after :data:`IDLE_EXIT_S` with nothing open it exits. A heartbeat thread
rewrites ``service.json`` every :data:`HEARTBEAT_S`. Each turn of the loop is :func:`_iteration`; :func:`step` runs
exactly one such turn in the calling process (deterministic step mode, for the offline replay).

**Bounds, stated, never steering:** the service's lifetime (:data:`LIFETIME_S`), memory's spend cap for the home
(``UNIFY_MEMORY_CAP_USD``: a pass starts only when the room left under the cap holds one writer call at its worst case;
it books that whole room, no call of its writer or helpers starts past it, counting unpriced and in-flight calls at the
worst case, and it settles after it from the writer lane's journal plus its helpers' priced calls, or its own figures,
or, when it ended with no report, its turn files), and each pass's wall clock. A lane relay's cap applies
independently: a call the lane refuses for its cap (HTTP 402, or a 403 whose refusal body says it is the cap) is the
cap reached; a refusal for now (another call in flight, a rate limit, a server error) is tried again, bounded, and
never stops the service.

**Resumable passes.** Before each model call of a pass (the writer's and each helper's), ``turns/<pass>/<n>.json`` is
written ``pending`` with the conversation so far and the pass's inputs; after it, ``done`` with the reply and its
usage. A service that restarts with a pass whose last turn is ``pending`` runs that pass again under the same pass id
with its entries still handed to it: the turns already done are answered from their files (no model call), and the
lost turn is made again and counted once as ``killed`` (billed but lost: its cost is in the writer lane's journal
window, which the pass's settle reads from the pass's first start, so it is never counted twice).

**Stopping.** :func:`stop_service` writes ``service.stop``, lets a pass in flight run to its deadline plus
:data:`STOP_GRACE_S`, then ``SIGTERM`` to the service's process group, :data:`TERM_S`, ``SIGKILL``, and checks in
``/proc`` that no process of the group is left; :class:`Terminated` says how it ended and whether that was verified.
Shutdown (:func:`.integration.async_pass.drain`) stops the service and returns only once it is verified gone, before
the model-call lanes are stopped.

Files, under ``<home>/memory-state/memory-service/`` (harness-only; no box binds them): ``service.lock``,
``service.json``, ``service.stop``, ``service-events.jsonl``, ``spend.jsonl`` (a ``run`` row at each run's start,
written by its controller, then reserve and settle rows; a service restart within the run writes none), and
``turns/<pass_id>/`` (``pass.json`` and one file per model turn).
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Optional

#: Where the service keeps its state, under the memory state directory.
SERVICE_DIR = "memory-service"
LOCK_FILE = "service.lock"
STATE_FILE = "service.json"
STOP_FILE = "service.stop"
EVENTS_FILE = "service-events.jsonl"
SPEND_FILE = "spend.jsonl"
TURNS_DIR = "turns"
PASS_FILE = "pass.json"

#: The heartbeat interval, and the age after which a holder that is not the recorded service is taken over.
HEARTBEAT_S = 30.0
#: How often the service records its processes (``service-members.json``): a stop or takeover after a kill finds
#: every process it started up to this long before it died.
MEMBERS_S = 1.0
MEMBERS_FILE = "service-members.json"
STALE_S = 90.0
#: The service exits after this long idle with nothing open.
IDLE_EXIT_S = 1800.0
#: Its lifetime.
LIFETIME_S = 36 * 3600.0
#: :func:`stop_service`: the grace after a pass's deadline, then SIGTERM, this long, then SIGKILL.
STOP_GRACE_S = 30.0
TERM_S = 10.0
SETTLE_S = 5.0
#: How often the idle loop looks at the ledger (a new entry makes a pass due within this).
POLL_S = 2.0
#: How long :func:`ensure_service` waits for a service it started to say it is up.
START_WAIT_S = 10.0
#: How long :func:`ensure_service` waits for a lock whose holder is exiting (no live process holds it) to free.
EXITING_WAIT_S = 5.0

STATES = ("starting", "running", "idle", "stopping", "stopped")
StopBy = Literal["drain", "suspend", "operator", "cap"]
STOP_BY = ("drain", "suspend", "operator", "cap")
#: What the entries the service could not reach are marked with, by why it stopped. A ``suspend`` (the end of one
#: stage of a run that goes on) marks nothing: every entry still owed stays owed, and the next start hands it again
#: (entries a stopped pass held go back with ``suspend``, no miss).
NOT_REACHED_BY = {
    "drain": "run_ended",
    "operator": "service_stopped",
    "cap": "spend_cap",
}
EVENTS = (
    "kin",
    "start",
    "stop",
    "idle-exit",
    "takeover",
    "resume",
    "killed-turn",
    "cap-reached",
    "lifetime",
    "pass",
    "refused",
    "error",
    "stop-cleared",
)
MODULE = "unify.memory.writer_service"


def service_dir(home: Path | str) -> Path:
    return Path(home) / "memory-state" / SERVICE_DIR


def _now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(at: datetime) -> str:
    return (
        at.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def parse_stamp(text: Any) -> Optional[datetime]:
    if not isinstance(text, str) or not text.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(text[:-1]).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# --- results --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceState:
    """What :func:`ensure_service` found or did: ``off`` (the switch is off), ``refused`` (settings, or the lease),
    ``started``, ``running`` (one is up), ``busy`` (the lock is held by a live process that is not the recorded
    service: left alone), ``error`` (never raised: why is in *detail*)."""

    status: Literal["off", "refused", "started", "running", "busy", "error"]
    pid: Optional[int] = None
    detail: str = ""
    took_over: bool = False


@dataclass(frozen=True)
class Terminated:
    """How :func:`stop_service` ended the service: ``clean`` (it stopped at the marker), ``term`` or ``kill``; and
    whether no process of its group is left (*verified*), with any that is."""

    verified: bool
    how: Literal["clean", "term", "kill"]
    leftover_pids: tuple[int, ...] = ()


# --- files ----------------------------------------------------------------------------------------------------------


def _atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        data = (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
        while data:
            data = data[os.write(fd, data) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _rows(path: Path) -> list[dict]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    out = []
    for line in data[: data.rfind(b"\n") + 1].split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def read_state(home: Path | str) -> Optional[dict]:
    """``service.json``, or None."""
    try:
        doc = json.loads((service_dir(home) / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def events(home: Path | str) -> list[dict]:
    """Every ``service-events.jsonl`` row, in order."""
    return _rows(service_dir(home) / EVENTS_FILE)


def event(home: Path | str, kind: str, **fields: Any) -> dict:
    if kind not in EVENTS:
        raise ValueError(f"unknown service event {kind!r}")
    row = {"v": 1, "event": kind, "at": stamp(_now()), "pid": os.getpid(), **fields}
    _append(service_dir(home) / EVENTS_FILE, row)
    return row


# --- process identity -----------------------------------------------------------------------------------------------


def proc_start(pid: int) -> Optional[str]:
    """The start time of a live, non-zombie process *pid* (``/proc/<pid>/stat`` field 22), else None."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except (OSError, ValueError):
        return None
    rest = stat.rsplit(")", 1)[-1].split()
    if len(rest) < 20 or rest[0] in ("Z", "X"):
        return None
    return rest[19]


def owns(doc: Optional[dict]) -> bool:
    """Whether *doc* (``service.json``) names a live process: the same pid, ``/proc`` start time and process group.
    A doc without a start time owns nothing: no signal on a pid alone."""
    if not isinstance(doc, dict):
        return False
    pid, pgid, start = doc.get("pid"), doc.get("pgid"), doc.get("proc_start")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not start:
        return False
    if proc_start(pid) != start:
        return False
    try:
        return os.getpgid(pid) == pgid
    except OSError:
        return False


def _stat_fields(pid: int) -> Optional[list[str]]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[-1].split()


def group_pids(pgid: int, *, since: Optional[str] = None) -> list[int]:
    """The live, non-zombie processes in process group *pgid* owned by this user; with *since* (a ``/proc`` start time
    in clock ticks), only those started at or after it (a group id a stranger took over is never ours).
    """
    out = []
    uid = os.getuid()
    try:
        names = os.listdir("/proc")
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        fields_ = _stat_fields(int(name))
        if (
            fields_ is None
            or len(fields_) < 20
            or fields_[0] in ("Z", "X")
            or fields_[2] != str(int(pgid))
        ):
            continue
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
        except OSError:
            continue
        if since is not None:
            try:
                if int(fields_[19]) < int(since):
                    continue
            except ValueError:
                continue
        out.append(int(name))
    return sorted(out)


def lock_holders(path: Path) -> list[int]:
    """The live processes of this user that hold a descriptor open on *path* (the lock file): the service, and any
    leftover child of it that kept its descriptor (a lock held through an inherited descriptor shows its first
    owner's pid in ``/proc/locks``, so the descriptors themselves are read). [] when none, or when unknown.
    """
    try:
        st = os.stat(path)
        names = os.listdir("/proc")
    except OSError:
        return []
    want = (st.st_dev, st.st_ino)
    uid = os.getuid()
    out = []
    for name in names:
        if not name.isdigit():
            continue
        fds = f"/proc/{name}/fd"
        try:
            if os.stat(f"/proc/{name}").st_uid != uid:
                continue
            entries = os.listdir(fds)
        except OSError:
            continue
        for fd in entries:
            try:
                target = os.stat(f"{fds}/{fd}")
            except OSError:
                continue
            if (target.st_dev, target.st_ino) == want:
                fields_ = _stat_fields(int(name))
                if fields_ is not None and fields_[0] not in ("Z", "X"):
                    out.append(int(name))
                break
    return sorted(out)


def _boot_time() -> Optional[float]:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def started_at(ticks: Any) -> Optional[datetime]:
    """The wall time a process started, from its ``/proc`` start time in clock ticks since boot; None when unknown."""
    boot = _boot_time()
    try:
        return datetime.fromtimestamp(boot + int(ticks) / os.sysconf("SC_CLK_TCK"), tz=timezone.utc) if boot else None
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def read_members(home: Path | str) -> Optional[dict]:
    """``service-members.json``: the service's processes as it last recorded them (every :data:`MEMBERS_S`)."""
    try:
        doc = json.loads((service_dir(home) / MEMBERS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _same_service(doc: dict, members: Optional[dict]) -> bool:
    return (
        isinstance(members, dict)
        and members.get("pid") == doc.get("pid")
        and members.get("proc_start") == doc.get("proc_start")
    )


def last_alive(doc: Optional[dict], members: Optional[dict] = None) -> Optional[datetime]:
    """The last time the service *doc* records was known alive: its heartbeat, or its last member record."""
    if not isinstance(doc, dict):
        return None
    times = [parse_stamp(doc.get("heartbeat_at"))]
    if _same_service(doc, members):
        times.append(parse_stamp(members.get("at")))
    times = [t for t in times if t is not None]
    return max(times) if times else None


def recorded_members(doc: Optional[dict], members: Optional[dict] = None) -> list[int]:
    """The processes the service recorded as its own (``members``, pid and ``/proc`` start time: in ``service.json``
    at each heartbeat and in ``service-members.json`` every :data:`MEMBERS_S`) that are still those processes, and
    that started no later than it was last known alive plus :data:`HEARTBEAT_S`."""
    if not isinstance(doc, dict):
        return []
    alive = last_alive(doc, members)
    if alive is None:
        return []
    limit = alive + timedelta(seconds=HEARTBEAT_S)
    rows = list(doc.get("members") or ())
    if _same_service(doc, members):
        rows += list(members.get("members") or ())
    out = []
    for row in rows:
        if not (isinstance(row, (list, tuple)) and len(row) == 2):
            continue
        pid, start = row
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or not isinstance(start, str) or not start:
            continue
        if proc_start(pid) != start:
            continue  # gone, or the pid is someone else's now
        at = started_at(start)
        if at is None or at > limit:
            continue
        out.append(pid)
    return sorted(set(out))


def orphaned_members(doc: Optional[dict], members: Optional[dict] = None) -> list[int]:
    """Processes of the dead service's group that it had no time to record: the members of its process group that
    started at or after the service and no later than it was last known alive. While the service was alive its group
    was its own, so a process that joined it then is the service's; a group the number names later (once every
    process of the service's has gone) starts after that. Only for a service that led its group (as every service
    does: ``setsid``), and only while that leader is gone or is still the recorded process (a leader with another
    start time means the number is someone else's)."""
    if not isinstance(doc, dict):
        return []
    pid, pgid, since = doc.get("pid"), doc.get("pgid"), doc.get("proc_start")
    if not isinstance(pgid, int) or isinstance(pgid, bool) or pgid <= 0 or not since or pid != pgid:
        return []
    now = proc_start(pgid)
    if now is not None and now != str(since):
        return []
    alive = last_alive(doc, members)
    if alive is None:
        return []
    out = []
    for p in group_pids(pgid, since=str(since)):
        at = started_at(proc_start(p) or "")
        if at is not None and at <= alive:
            out.append(p)
    return out


def service_group(doc: Optional[dict], members: Optional[dict] = None) -> list[int]:
    """The live processes left of the service *doc* (``service.json``) records. While the service itself is alive
    (:func:`owns`: its pid, start time and group), the members of its process group started at or after it. Once it is
    not, the processes it recorded (:func:`recorded_members`, each checked by its start time) and the members of its
    group it had no time to record (:func:`orphaned_members`): a group number is never trusted on its own, since
    another group may hold it now."""
    if not isinstance(doc, dict):
        return []
    if owns(doc):
        return group_pids(int(doc["pgid"]), since=str(doc["proc_start"]))
    return sorted(set(recorded_members(doc, members)) | set(orphaned_members(doc, members)))


def try_lock(path: Path) -> Optional[int]:
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


# --- settings -------------------------------------------------------------------------------------------------------


def _settings() -> Any:
    from unify.settings import SETTINGS

    return SETTINGS


def lifetime_end(started: datetime) -> datetime:
    """When the service must stop: the end of its own lifetime."""
    return started + timedelta(seconds=LIFETIME_S)


# --- the spend under the service's own cap --------------------------------------------------------------------------


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() and d >= 0 else None


def begin_run(home: Path | str) -> None:
    """Mark the start of a run in ``spend.jsonl`` (a ``run`` row): memory's spend cap counts from here. The run is the
    controller's (one ``unify act``), written once per controller process at its first memory request
    (:meth:`.integration.request.RequestRun`), never by the service: a service restarted within the run (its lifetime
    end, a crash, a takeover, a suspend and start) reads the same rows and keeps the run's total."""
    _append(service_dir(home) / SPEND_FILE, {"v": 1, "op": "run", "pid": os.getpid(), "at": stamp(_now())})


def spend_totals(home: Path | str, *, without: Optional[str] = None) -> tuple[Decimal, Decimal]:
    """``(committed, settled)`` USD of this run (the rows of ``spend.jsonl`` after the last ``run`` row, which the
    run's controller writes when it starts, :func:`begin_run`): a settled pass counts its settled amount, a pass only
    reserved (it never settled: running, or killed) its whole reservation; *without* leaves one pass out. Decimal
    throughout."""
    reserved: dict[str, Decimal] = {}
    settled: dict[str, Decimal] = {}
    for row in _rows(service_dir(home) / SPEND_FILE):
        if row.get("op") == "run":
            # memory's spend cap is per run: what earlier runs spent is not counted
            reserved, settled = {}, {}
            continue
        pid = row.get("pass_id")
        if pid == without:
            continue
        amount = _decimal(row.get("usd"))
        if not isinstance(pid, str) or amount is None:
            continue
        if row.get("op") == "reserve":
            reserved[pid] = amount
        elif row.get("op") == "settle":
            settled[pid] = amount
    committed = sum((settled.get(p, r) for p, r in reserved.items()), Decimal(0))
    committed += sum((v for p, v in settled.items() if p not in reserved), Decimal(0))
    return committed, sum(settled.values(), Decimal(0))


def pass_room(home: Path | str, pass_id: str, settings: Any) -> Decimal:
    """What pass *pass_id* may spend: memory's spend cap less what every other pass of this run committed (never below
    0)."""
    from .integration.switch import memory_cap_usd

    committed, _settled = spend_totals(home, without=pass_id)
    return max(Decimal(0), memory_cap_usd(settings) - committed)


def _turn_files(folder: Path | str, *, since: Optional[str] = None) -> list[tuple[str, dict]]:
    """Every checkpointed model turn of a pass, with its lane (``writer``: ``<n>.json``; ``helpers``:
    ``h<k>/<n>.json``); with *since* (a :func:`stamp`), only the turns written at or after it."""
    out = []
    root = Path(folder)
    for lane, paths in (("writer", sorted(root.glob("*.json"))), ("helpers", sorted(root.glob("h*/*.json")))):
        for path in paths:
            if path.name == PASS_FILE:
                continue
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict) or doc.get("state") not in ("pending", "done", "failed"):
                continue
            if since is not None and str(doc.get("at") or "") < since:
                continue
            out.append((lane, doc))
    return out


def turns_cost(folder: Path | str, *, since: Optional[str] = None) -> dict[str, tuple[Decimal, int]]:
    """What a pass's turns cost by their own files, per lane: ``(known USD, unpriced calls)``, a done call with a
    price counting its USD, a pending (lost) or unpriced one counting as unpriced; plus, at each call, its failed
    attempts that may have cost money (``failed_unpriced``; an attempt the route refused before sending it on cost
    nothing). A call whose last attempt failed counts only those."""
    out = {"writer": (Decimal(0), 0), "helpers": (Decimal(0), 0)}
    for lane, doc in _turn_files(folder, since=since):
        known, unpriced = out[lane]
        unpriced += _failed_unpriced(doc)
        state = doc.get("state")
        if state == "failed":
            out[lane] = (known, unpriced)
            continue
        usd = _decimal(doc.get("usd")) if state == "done" else None
        out[lane] = (known + usd, unpriced) if usd is not None else (known, unpriced + 1)
    return out


def _failed_unpriced(doc: dict) -> int:
    try:
        return max(0, int(doc.get("failed_unpriced") or 0))
    except (TypeError, ValueError):
        return 0


def turns_spend(folder: Path | str, worst: Decimal, *, since: Optional[str] = None) -> Decimal:
    """:func:`turns_cost` over both lanes, each unpriced or lost call at *worst* (one call's worst case)."""
    return sum((k + Decimal(n) * worst for k, n in turns_cost(folder, since=since).values()), Decimal(0))


def pass_call_worst_case(settings: Any) -> Decimal:
    """One model call's worst case for a service pass: the pinned catalogue's bound (:mod:`.call_bounds`) for the
    writer's model, which its helpers run on too. Raises :class:`.call_bounds.NoBound` for a model the catalogue does
    not bound (no pass runs: fail closed)."""
    from .call_bounds import worst_of
    from .integration.consolidate import writer_settings

    return worst_of([writer_settings(settings).model])


def _format(d: Decimal) -> str:
    return format(abs(d) if d.is_zero() else d, "f")


# --- resumable turns ------------------------------------------------------------------------------------------------


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


class ResumableTurn:
    """A model turn whose calls are checkpointed under *folder* (``turns/<pass>`` or a helper's subfolder): each call
    writes ``<n>.json`` ``pending`` (the conversation so far, the pass's inputs) before it is made and ``done`` (the
    reply, its USD and usage) after. Over a folder a killed pass left, the calls whose conversation is the same as
    recorded are answered from their files with no model call; a call recorded ``pending`` is made again and counted
    in :attr:`killed` once (by its turn id); from the first call whose conversation differs, every call is made.
    """

    def __init__(
        self,
        turn: Any,
        folder: Path,
        *,
        inputs: Optional[dict] = None,
        scrub: Callable[[str], str] = str,
    ):
        self.turn, self.folder, self.inputs = turn, Path(folder), dict(inputs or {})
        self._scrub = scrub
        self.n = 0
        self.replayed = 0
        self.killed: list[str] = []
        self.diverged = False
        self.folder.mkdir(parents=True, exist_ok=True)
        self.__dict__.update(
            {
                k: v
                for k, v in getattr(turn, "__dict__", {}).items()
                if k not in self.__dict__
            }
        )

    def _file(self, n: int) -> Path:
        return self.folder / f"{n}.json"

    def _load(self, n: int) -> Optional[dict]:
        try:
            doc = json.loads(self._file(n).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def _write(self, n: int, doc: dict) -> None:
        text = self._scrub(json.dumps(doc, sort_keys=True, default=str))
        path = self._file(n)
        fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    async def __call__(
        self, messages: list[dict], tools: list[dict]
    ) -> tuple[dict, Any]:
        n = self.n
        self.n += 1
        digest = _digest({"messages": messages, "tools": tools})
        saved = self._load(n)
        if saved is not None and not self.diverged:
            if saved.get("digest") != digest:
                self.diverged = True
            elif saved.get("state") == "done" and isinstance(saved.get("reply"), dict):
                # its USD was spent by the run that made it, which the resumed pass takes off its room and settles
                # as it stood (turns_cost at the resume): never counted a second time here
                self.replayed += 1
                return dict(saved["reply"]), "0"
            elif saved.get("state") == "pending":
                self.killed.append(f"{self.folder.name}/{n}")
        # the failed attempts at this call that may have cost money, carried to its file whatever it ends as
        failed = _failed_unpriced(saved) if saved is not None and saved.get("digest") == digest else 0
        self._write(
            n,
            {
                "v": 1,
                "state": "pending",
                "turn": n,
                "digest": digest,
                "messages": messages,
                "inputs": self.inputs,
                "at": stamp(_now()),
                **({"failed_unpriced": failed} if failed else {}),
            },
        )
        try:
            msg, usd = await self.turn(messages, tools)
        except Exception as exc:
            # a failed call (not one cut mid-flight, which stays pending: billed and lost): recorded failed, an
            # unpriced call unless the route refused it before sending it on; a call that may be tried again keeps
            # its number, so the attempt that answers is the one a resumed pass compares
            no_cost = bool(getattr(exc, "no_cost", False))
            self._write(
                n,
                {
                    "v": 1,
                    "state": "failed",
                    "turn": n,
                    "digest": digest,
                    "no_cost": no_cost,
                    "failed_unpriced": failed + (0 if no_cost else 1),
                    "at": stamp(_now()),
                },
            )
            if getattr(exc, "transient", False):
                self.n = n
            raise
        self._write(
            n,
            {
                "v": 1,
                "state": "done",
                "turn": n,
                "digest": digest,
                "reply": msg,
                "usd": usd,
                "usage": msg.get("usage") if isinstance(msg, dict) else None,
                "at": stamp(_now()),
                **({"failed_unpriced": failed} if failed else {}),
            },
        )
        return msg, usd


def pending_passes(home: Path | str) -> list[str]:
    """The passes a stopped service left running (``pass.json`` not ended), oldest first: the next service resumes
    them."""
    root = service_dir(home) / TURNS_DIR
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        try:
            doc = json.loads((root / name / PASS_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("state") == "running":
            out.append((str(doc.get("started_at") or ""), name))
    return [name for _, name in sorted(out)]


# --- ensure ---------------------------------------------------------------------------------------------------------


from unify.lanes.provider_names import LITELLM_PROVIDER_KEY_NAMES  # noqa: E402

#: The provider credential names the model client reads: unillm's (its settings and ``uni_llm``), then the litellm
#: ones it falls back to (:mod:`unify.lanes.provider_names`). With one missing it reads a ``.env`` file or, on a team machine, a
#: Secret Manager (unillm's settings), so a writer process that needs no key gets :data:`NO_PROVIDER_KEY` for each.
PROVIDER_KEY_NAMES = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_MANAGEMENT_API_KEY",
    "TOGETHER_API_KEY",
    "ANTHROPIC_API_KEY",
    "XIAOMI_MIMO_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "UNILLM_LLM_GATEWAY_KEY",
    *LITELLM_PROVIDER_KEY_NAMES,
)
#: What each provider key holds in a writer process whose model calls all go through its lanes: not a key, so a call
#: off the lanes fails at the provider instead of being billed.
NO_PROVIDER_KEY = "none:the-writer-calls-through-its-lanes"


def without_provider_keys(env: dict, settings: Any) -> dict:
    """*env* for a writer process (the service, or a pass worker) whose writer calls go through the writer's lane:
    every provider key replaced by :data:`NO_PROVIDER_KEY`. The writer's calls and its helpers' go through their lanes
    (with no helper lane, helpers are not offered), and the librarian runs in the request, not here. One case keeps the
    keys: an embedding route the user set (``UNIFY_EMBED_URL`` with no lane token), which sends the provider key in its
    own header. With the relay's embedding lane the client sends that lane's token, so the keys are masked."""
    from unify.lanes import embed_token

    try:
        lane_token = embed_token.token()
    except embed_token.EmbedTokenRefused:
        lane_token = ""  # a lane was given, unusable: embeddings are refused, the keys masked all the same
    if lane_token is None and str(getattr(settings, "UNIFY_EMBED_URL", "") or "").strip():
        return env
    out = {k: v for k, v in env.items() if k.upper() not in PROVIDER_KEY_NAMES}
    out.update(dict.fromkeys(PROVIDER_KEY_NAMES, NO_PROVIDER_KEY))
    return out


def _embed_lane_token() -> str | None:
    """The embedding lane's token as this process holds it (None without one, or when it was unusable)."""
    from unify.lanes import embed_token

    try:
        return embed_token.token()
    except embed_token.EmbedTokenRefused:
        return None


from unify.lanes.embed_token import EMBED_TOKEN_FD  # noqa: E402


def _token_pipes() -> tuple[dict, list[int]]:
    """The writer's and the helpers' route tokens as this process holds them, each on a fresh pipe named in the
    child's environment, exactly as the pass worker gets the writer's (never argv, the environment or a file); and
    the read ends to pass. With the writer's route, the child's environment holds no provider key
    (:func:`without_provider_keys`). Without a route, nothing changes."""
    from .integration import switch
    from .integration.async_pass import _route_token

    env = {
        k: v
        for k, v in os.environ.items()
        if k.upper()
        not in (
            switch.WRITER_TOKEN,
            switch.WRITER_TOKEN_FD,
            switch.LIBRARY_HELPER_TOKEN_FD,
            switch.LIBRARY_HELPER_BASE_URL,
            EMBED_TOKEN_FD,
        )
    }
    fds: list[int] = []

    def pipe(value: str) -> int:
        r, w = os.pipe()
        try:
            os.write(w, (value + "\n").encode("latin-1"))
        finally:
            os.close(w)
        os.set_inheritable(r, True)
        fds.append(r)
        return r

    settings = _settings()
    token = _route_token(settings)
    if token:
        env[switch.WRITER_TOKEN_FD] = str(pipe(token))
        env = without_provider_keys(env, settings)
    embed = _embed_lane_token()
    if embed:
        env[EMBED_TOKEN_FD] = str(pipe(embed))
    switch.settle_helper_route_env(os.environ, settings)
    pair, _why = switch.helper_lane_route()
    if pair is not None:
        base, secret = pair
        value = (
            secret.get_secret_value()
            if hasattr(secret, "get_secret_value")
            else str(secret)
        )
        if value:
            env[switch.LIBRARY_HELPER_BASE_URL] = base
            env[switch.LIBRARY_HELPER_TOKEN_FD] = str(pipe(value))
    return env, fds


def _take_over(home: Path, doc: Optional[dict]) -> bool:
    """The lock is held, the recorded service is not alive and its heartbeat is stale: kill the process group of the
    recorded service that still holds the lock (its leftover children), verified. True when the lock's holders are
    gone."""
    lock = service_dir(home) / LOCK_FILE
    if not isinstance(doc, dict):
        return False
    holders = lock_holders(lock)
    ours = set(service_group(doc, read_members(home)))
    if not holders or not set(holders) <= ours:
        return (
            False  # held by someone that is not the old service's group: never touched
        )
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(ours):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        end = time.monotonic() + (TERM_S if sig == signal.SIGTERM else SETTLE_S)
        while time.monotonic() < end and service_group(doc, read_members(home)):
            time.sleep(0.05)
        if not service_group(doc, read_members(home)):
            break
    return not service_group(doc, read_members(home))


def ensure_service(
    home: Path | str,
    *,
    settings: Any = None,
    popen: Callable[..., Any] = subprocess.Popen,
    wait_s: float = START_WAIT_S,
    argv_prefix: Optional[list[str]] = None,
) -> ServiceState:
    """Start the memory home's writer service unless one is up (module docstring); never raises."""
    home = Path(home)
    try:
        settings = settings if settings is not None else _settings()
        from .integration.switch import writer_refusal

        refused = writer_refusal(settings)
        if refused is not None:
            return ServiceState("refused", detail=refused)
        marker = stopping_marker(home)
        if marker is not None:
            # stopped on purpose (shutdown or an operator): no request starts it again
            return ServiceState(
                "refused", detail=f"the service was stopped ({marker.get('by')})"
            )
        folder = service_dir(home)
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / LOCK_FILE
        fd = try_lock(lock)
        took_over = False
        if fd is None and not owns(read_state(home)) and not lock_holders(lock):
            # the holder is exiting: a killed service's leader shows as a zombie while its other threads still close
            # the files they share, so the lock frees within moments; a start never reads that as a live service
            end = time.monotonic() + EXITING_WAIT_S
            while fd is None and time.monotonic() < end:
                time.sleep(0.05)
                fd = try_lock(lock)
        if fd is None:
            doc = read_state(home)
            beat = parse_stamp((doc or {}).get("heartbeat_at"))
            stale = beat is None or (_now() - beat).total_seconds() > STALE_S
            if owns(doc) or not stale:
                return ServiceState("running", pid=(doc or {}).get("pid"))
            if not _take_over(home, doc):
                return ServiceState(
                    "busy",
                    detail="the service lock is held by a process that is not the service",
                )
            event(
                home,
                "takeover",
                previous_pid=(doc or {}).get("pid"),
                by="ensure_service",
                reason="stale heartbeat",
            )
            took_over = True
            fd = try_lock(lock)
            if fd is None:
                return ServiceState("busy", detail="the service lock was taken again")
        # the lock was free: released at once, and the service takes it in its own process (a second service started
        # meanwhile finds it held and exits without a trace)
        os.close(fd)
        return _spawn(
            home,
            popen=popen,
            wait_s=wait_s,
            argv_prefix=argv_prefix,
            took_over=took_over,
        )
    except Exception as exc:  # noqa: BLE001 - a request never fails on the service
        return ServiceState("error", detail=type(exc).__name__)


def _spawn(
    home: Path,
    *,
    popen: Callable[..., Any],
    wait_s: float,
    argv_prefix: Optional[list[str]],
    took_over: bool,
) -> ServiceState:
    """Double fork: the child forks the service and exits, the service ``setsid``s and takes the lock in its own
    process; the route tokens go on their own pipes."""
    asked = _now()
    log = None
    fds: list[int] = []
    try:
        env, fds = _token_pipes()
        log = open(service_dir(home) / "service.log", "ab")
        argv = [
            *(argv_prefix or [sys.executable, "-m", MODULE]),
            "--home",
            str(home),
            "--daemon",
        ]
        proc = popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
            pass_fds=tuple(fds),
            env=env,
        )
        # the child holds its own copies now: this process lets go of the token pipes at once, never while it waits
        for r in fds:
            try:
                os.close(r)
            except OSError:
                pass
        fds = []
        try:
            proc.wait(timeout=30)  # the first child exits at once, leaving the service
        except subprocess.TimeoutExpired:
            pass
    finally:
        for r in fds:
            try:
                os.close(r)
            except OSError:
                pass
        if log is not None:
            log.close()
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        doc = read_state(home)
        started = parse_stamp((doc or {}).get("started_at"))
        if (
            doc
            and started is not None
            and started >= asked - timedelta(seconds=1)
            and owns(doc)
        ):
            return ServiceState("started", pid=doc.get("pid"), took_over=took_over)
        time.sleep(0.05)
    return ServiceState(
        "started", took_over=took_over, detail="the service has not reported yet"
    )


# --- stop -----------------------------------------------------------------------------------------------------------


def stop_service(
    home: Path | str,
    by: StopBy,
    reason: str,
    *,
    grace_s: float = STOP_GRACE_S,
    term_s: float = TERM_S,
    settle_s: float = SETTLE_S,
    kill: Callable[[int, int], None] = os.killpg,
    kill_pid: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], Any] = time.sleep,
    wait_s: Optional[float] = None,
) -> Terminated:
    """Write the stop marker; give a pass in flight its deadline plus *grace_s* (the service stops at the marker once
    the pass ends), or *wait_s* from now if that is sooner; then SIGTERM, *term_s*, SIGKILL; then check in ``/proc``
    that nothing of the service is left (module docstring). A pass cut by the SIGTERM ends through its abort path;
    at a ``suspend`` every entry it held stays owed for the next start. A signal goes to the service's process group
    (*kill*) only while the service itself is alive (:func:`owns`); otherwise only to each process the service
    recorded, checked by its start time (*kill_pid*, :func:`recorded_members`): a group number alone is never
    signalled."""
    if by not in STOP_BY:
        raise ValueError(f"by is one of {', '.join(STOP_BY)}, not {by!r}")
    home = Path(home)
    cap = None if wait_s is None else time.monotonic() + max(0.0, float(wait_s))

    def bounded(limit: float) -> float:
        return limit if cap is None else min(limit, cap)

    _atomic_json(
        service_dir(home) / STOP_FILE,
        {"v": 1, "by": by, "at": stamp(_now()), "reason": str(reason)[:500]},
    )
    _note_writer(home, "stopping")
    doc = read_state(home)
    if not isinstance(doc, dict):
        return Terminated(True, "clean")
    if doc.get("state") == "stopped" and not lock_holders(service_dir(home) / LOCK_FILE):
        # the service wrote its last state and nothing holds the home: nothing is left to stop (a step, or a test,
        # that ran the loop in its own process is never signalled)
        return Terminated(True, "clean")
    pgid = doc.get("pgid")

    def left() -> list[int]:
        return service_group(doc, read_members(home))

    def alive() -> bool:
        return owns(doc) or bool(left())

    if not alive():
        return Terminated(True, "clean")
    pass_end = parse_stamp(doc.get("pass_deadline"))
    limit = bounded(
        time.monotonic()
        + max(0.0, ((pass_end - _now()).total_seconds() if pass_end else 0.0))
        + grace_s
    )
    while owns(doc) and time.monotonic() < limit:
        sleep(0.1)
        fresh = read_state(home)
        if isinstance(fresh, dict) and fresh.get("pid") == doc.get("pid"):
            new_end = parse_stamp(fresh.get("pass_deadline"))
            if new_end is not None and new_end != pass_end:
                pass_end = new_end
                limit = bounded(
                    time.monotonic()
                    + max(0.0, (new_end - _now()).total_seconds())
                    + grace_s
                )
    def signal_all(sig: int) -> None:
        if owns(doc):
            try:
                kill(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            return
        for pid in left():
            try:
                kill_pid(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    how: Literal["clean", "term", "kill"] = "clean"
    if alive():
        how = "term"
        signal_all(signal.SIGTERM)
        end = time.monotonic() + term_s
        while alive() and time.monotonic() < end:
            sleep(0.1)
    if alive():
        how = "kill"
        signal_all(signal.SIGKILL)
        end = time.monotonic() + settle_s
        while alive() and time.monotonic() < end:
            sleep(0.05)
    leftover = tuple(left())
    verified = not owns(doc) and not leftover
    return Terminated(verified, how, leftover)


def _note_writer(home: Path, note: str) -> None:
    """The writer's roster row says *note* (``stopping``: an agent waiting on it sees the service is going away)."""
    try:
        from unify.agents.run_record import WRITER, agents_dir, join_run_record

        if not (agents_dir(home) / "run-record.jsonl").exists():
            return
        record = join_run_record(home)
        row = next((r for r in record.roster_rows(include_ended=True) if r["name"] == WRITER), None)
        if row is not None and row["state"] != "ended":
            record.set_state(WRITER, row["state"], note=note)
    except Exception:  # noqa: BLE001 - a note never stops the stop
        pass


def clear_previous_stop(home: Path | str) -> Optional[dict]:
    """After a ``suspend`` (the end of one stage of a run that goes on, :func:`.integration.async_pass.drain`): the stop
    marker it left is renamed aside (``service.stop.<stamp>``) and recorded (``stop-cleared``, with its ``by`` and
    ``at``), so the next stage's service starts. Returns the old marker, or None. A request never clears a marker."""
    marker = stopping_marker(home)
    if marker is None:
        return None
    path = service_dir(home) / STOP_FILE
    aside = path.with_name(f"{STOP_FILE}.{stamp(_now()).replace(':', '')}")
    try:
        os.replace(path, aside)
    except FileNotFoundError:
        return None
    event(home, "stop-cleared", previous_by=marker.get("by"), previous_at=marker.get("at"), kept=aside.name)
    return marker


def stopping_marker(home: Path | str) -> Optional[dict]:
    try:
        doc = json.loads((service_dir(home) / STOP_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# --- the service ----------------------------------------------------------------------------------------------------

#: ``runner(home, pass_id, handed, resume, ctx) -> PassReport`` (default :func:`.integration.consolidate.run_attention_pass`)
Runner = Callable[..., Awaitable[Any]]


@dataclass
class Service:
    """The running service's own state (one per process)."""

    home: Path
    settings: Any
    lock_fd: int
    started: datetime = field(default_factory=_now)
    state: str = "starting"
    pass_id: Optional[str] = None
    pass_deadline: Optional[datetime] = None
    cap: Optional[Decimal] = None
    capped: bool = False
    stop_reason: Optional[str] = None
    #: set by SIGTERM: a pass in flight is cancelled through its abort path
    stop_flag: Optional[asyncio.Event] = None
    #: the staging folder as the capture step last saw it
    staging_seen: Optional[tuple] = None
    #: the cap's event was written (once per service)
    cap_noted: bool = False
    #: record the group's processes (``members``); False for a step, which runs in its caller's process: nothing of
    #: the caller's is ever recorded as the service's, so nothing of it is ever signalled
    record_members: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def doc(self) -> dict:
        committed, settled = spend_totals(self.home)
        end = lifetime_end(self.started)
        return {
            "v": 1,
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "pgid_start": proc_start(os.getpgid(0)) or "",
            "proc_start": proc_start(os.getpid()) or "",
            "heartbeat_at": stamp(_now()),
            # every process of the service's group (itself, its passes' children) by pid and start time: once the
            # service is gone, only these are ever signalled (recorded_members)
            "members": _members() if self.record_members else [],
            "started_at": stamp(self.started),
            "deadline": stamp(end),
            "state": self.state,
            "pass_id": self.pass_id,
            "pass_deadline": stamp(self.pass_deadline) if self.pass_deadline else None,
            "cap_usd": _format(self.cap) if self.cap is not None else None,
            "committed_usd": _format(committed),
            "settled_usd": _format(settled),
        }

    def beat(self) -> None:
        with self._lock:
            _atomic_json(service_dir(self.home) / STATE_FILE, self.doc())

    def set(self, state: str, **fields: Any) -> None:
        if state not in STATES:
            raise ValueError(state)
        with self._lock:
            self.state = state
            for k, v in fields.items():
                setattr(self, k, v)
            _atomic_json(service_dir(self.home) / STATE_FILE, self.doc())


def _hold_writer_row(home: Path) -> Any:
    """Take back the writer's row on the run record (``rejoin``: the service holds ``writer`` across passes and
    restarts) and return the record."""
    from unify.agents.run_record import WRITER, join_run_record

    record = join_run_record(home)
    record.root = WRITER
    record.join(WRITER, "writer", state="idle", rejoin=True)
    return record


def _end_writer_row(record: Any, why: str) -> None:
    from unify.agents.run_record import WRITER

    if record is None:
        return
    try:
        record.set_state(WRITER, "ended", note=f"service {why}")
    except Exception:  # noqa: BLE001 - the stop is still recorded
        pass


def _members() -> list[list]:
    own = proc_start(os.getpid())
    out = []
    for pid in group_pids(os.getpgid(0), since=own) if own else [os.getpid()]:
        start = proc_start(pid)
        if start:
            out.append([pid, start])
    return out


def _raised_settle(home: Path, pass_id: str, folder: Path, settings: Any, pass_doc: dict) -> Decimal:
    """The settle of a pass that raised: what a killed run of it spent (``prior_usd``, read at the resume) plus the
    turn files this run wrote, each unpriced or lost call at the lanes' worst case; when that cannot be had, the
    pass's whole reservation stays (it is settled at what it reserved)."""
    try:
        prior = pass_doc.get("prior_usd", "0")
        if prior is None:
            raise ValueError("no prior figure")
        since = str(pass_doc.get("run_started_at") or "")
        return Decimal(prior) + turns_spend(folder, pass_call_worst_case(settings), since=since or None)
    except Exception:  # noqa: BLE001 - no worst case: the booking stands
        rows = [r for r in _rows(service_dir(home) / SPEND_FILE) if r.get("pass_id") == pass_id]
        booked = [_decimal(r.get("usd")) for r in rows if r.get("op") == "reserve"]
        return booked[-1] if booked and booked[-1] is not None else Decimal(0)


def check_writer_settings(settings: Any) -> None:
    """Raise when the writer's settings or route refuse (no pass can run); the check the service makes before a pass.
    A pass has no allowance of its own: it books the whole room memory's spend cap leaves (:func:`pass_room`)."""
    from .integration.consolidate import writer_settings

    writer_settings(settings)


def _proxy_refused(report: Any) -> bool:
    """Whether a pass ended because the key proxy refused a writer call for its cap: HTTP 402, or a 403 whose refusal
    body says it is the cap (:data:`.writer_pass.CAP_CATEGORIES`). A 403 for a call in flight, a rate limit or a
    server error is not the cap (the pass tried it again, bounded), and neither is a 403 whose body says nothing."""
    from .writer_pass import CAP_CATEGORIES

    outcome = getattr(report, "outcome", None)
    text = " ".join(str(r) for r in (getattr(outcome, "reasons", None) or ()))
    text += " " + " ".join(str(n) for n in (getattr(report, "notes", None) or ()))
    return any(f"({category})" in text for category in CAP_CATEGORIES)


async def _run_pass(
    service: Service, pass_id: str, handed: list, *, resume: bool, runner: Runner
) -> Any:
    """One pass under the service: the reservation, the pass (under its attribution), the settle, the read checks and
    the returns; returns the runner's report."""
    from unify.common.llm_client import attribution

    from .attention import Ledger
    from .integration.request import actor_effort

    home, settings = service.home, service.settings
    effort = actor_effort()
    folder = service_dir(home) / TURNS_DIR / pass_id
    folder.mkdir(parents=True, exist_ok=True)
    pass_doc_path = folder / PASS_FILE
    try:
        pass_doc = (
            json.loads(pass_doc_path.read_text(encoding="utf-8")) if resume else None
        )
    except (OSError, ValueError):
        pass_doc = None
    if not resume or not isinstance(pass_doc, dict):
        from .integration.switch import writer_journal
        from .reconcile import journal_offset

        # the pass may spend the whole room (its USD stop), so the whole room is booked until it settles
        reservation = pass_room(home, pass_id, settings) or Decimal(0)
        _append(
            service_dir(home) / SPEND_FILE,
            {
                "v": 1,
                "op": "reserve",
                "pass_id": pass_id,
                "usd": _format(reservation),
                "at": stamp(_now()),
            },
        )
        pass_doc = {
            "v": 1,
            "pass_id": pass_id,
            "entries": [e.entry for e in handed],
            "started_at": stamp(_now()),
            "journal_start": journal_offset(writer_journal(settings)),
            "effort": effort,
            "state": "running",
        }
        _atomic_json(pass_doc_path, pass_doc)
    else:
        event(home, "resume", pass_id=pass_id, entries=pass_doc.get("entries"))
        # what the killed run spent, as its files stand before this run writes any (a diverging run overwrites them)
        try:
            pass_doc["prior_usd"] = _format(turns_spend(folder, pass_call_worst_case(settings)))
        except Exception:  # noqa: BLE001 - no worst case: settled at the booking if this run raises
            pass_doc["prior_usd"] = None
    pass_doc["run_started_at"] = stamp(_now())
    _atomic_json(pass_doc_path, pass_doc)
    from .integration.switch import PASS_WALL_S

    wall = float(PASS_WALL_S)
    service.set(
        "running", pass_id=pass_id, pass_deadline=_now() + timedelta(seconds=wall)
    )
    ledger = Ledger(home)
    try:
        with attribution(pass_id, [e.entry for e in handed]):
            report = await runner(
                home,
                pass_id,
                handed,
                resume=resume,
                folder=folder,
                journal_start=pass_doc.get("journal_start"),
                effort=pass_doc.get("effort") or effort,
                settings=settings,
                stop=service.stop_flag,
            )
    except (
        Exception
    ) as exc:  # noqa: BLE001 - a pass that broke: its entries go back, the service goes on
        event(home, "error", pass_id=pass_id, error=type(exc).__name__, where=_where(exc))
        returned = ledger.return_unread(pass_id, stopped=stopping_marker(home) is not None)
        # no report: the pass's own turn files say what it spent (lost and unpriced calls at the worst case)
        settled = _format(_raised_settle(home, pass_id, folder, settings, pass_doc))
        _append(
            service_dir(home) / SPEND_FILE,
            {"v": 1, "op": "settle", "pass_id": pass_id, "usd": settled, "at": stamp(_now())},
        )
        pass_doc.update(state="ended", ended_at=stamp(_now()), error=type(exc).__name__)
        _atomic_json(pass_doc_path, pass_doc)
        _raised_pass_row(home, pass_id, handed, exc, returned, settled, settings)
        return None
    finally:
        service.set("idle", pass_id=None, pass_deadline=None)
    for turn_id in getattr(report, "killed_turns", ()) or ():
        if not any(
            r.get("event") == "killed-turn" and r.get("turn_id") == turn_id
            for r in events(home)
        ):
            event(home, "killed-turn", pass_id=pass_id, turn_id=turn_id)
    settle = getattr(report, "settle_usd", None)
    if settle is not None:
        _append(
            service_dir(home) / SPEND_FILE,
            {
                "v": 1,
                "op": "settle",
                "pass_id": pass_id,
                "usd": settle,
                "at": stamp(_now()),
            },
        )
    from .attention import PassReads

    reads = report.reads if report.reads is not None else PassReads()
    read = ledger.verify_reads(pass_id, reads, list(report.delivered or ()))
    # a pass the service was asked to stop during (a drain or an operator's stop) gives its entries back with no miss
    returned = ledger.return_unread(pass_id, stopped=stopping_marker(home) is not None)
    report.read, report.returned = sorted(read), sorted(returned)
    pass_doc.update(state="ended", ended_at=stamp(_now()))
    _atomic_json(pass_doc_path, pass_doc)
    from .metering import write_pass_row

    write_pass_row(home, report)
    event(
        home,
        "pass",
        pass_id=pass_id,
        ended=getattr(report, "ended", None),
        read=len(read),
        returned=len(returned),
    )
    if _proxy_refused(report):
        service.capped = True
        if not service.cap_noted:
            event(home, "cap-reached", by="proxy", pass_id=pass_id)
            service.cap_noted = True
        ledger.mark_not_reached("spend_cap")
    return report


def _staging_signature(home: Path) -> tuple:
    """The fork staging folder's settled episodes, cheaply (names and whether each has its status file)."""
    root = Path(home) / "memory-state" / "memory-staging"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return ()
    from .integration.fork_worker import FORK_JSON

    return tuple((n, os.path.lexists(root / n / FORK_JSON)) for n in names)


def _capture_if_new(service: "Service") -> None:
    """Capture the forks that settled since the last look (the capture store feeds the ledger's entries); never
    raises."""
    try:
        sig = _staging_signature(service.home)
        if sig == service.staging_seen:
            return
        if not sig:  # nothing staged: nothing to capture
            service.staging_seen = sig
            return
        from .integration.consolidate import capture_settled_forks, open_stores
        from .integration.paths import Paths

        capture_settled_forks(open_stores(Paths.under(service.home), busy_timeout_s=60.0))
        service.staging_seen = sig
    except (
        Exception
    ) as exc:  # noqa: BLE001 - capture never stops the service; the next look tries again
        event(service.home, "error", where="capture", error=type(exc).__name__)


#: Why a turn of the loop ran no pass (:func:`_iteration`, :func:`step`).
IDLE_WHY = (
    "stopping",
    "lifetime",
    "lease",
    "refused",
    "nothing_open",
    "spend_cap",
    "nothing_handed",
    "busy",
)


def _raised_pass_row(
    home: Path, pass_id: str, handed: list, exc: BaseException, returned: Any, settled: str, settings: Any
) -> None:
    """A pass that raised still gets its ``pass-results.jsonl`` row (rev 2: no failed pass is silent): ``ended`` is
    ``error``, with the exception's class and innermost frame (never its message); its USD is unknown (the settle above
    is the spend file's figure). A row that cannot be written is an event, and the service goes on."""
    try:
        from .integration.consolidate import AttentionPassReport
        from .metering import write_pass_row

        innermost = _where(exc, frames=1)
        report = AttentionPassReport(
            pass_id,
            [e.entry for e in handed],
            "error",
            settle_usd=settled,
            returned=sorted(returned or ()),
            error_class=type(exc).__name__,
            error_where=innermost[0] if innermost else "unknown",
        )
        write_pass_row(home, report)
    except Exception as row_exc:  # noqa: BLE001 - the row is evidence; the service goes on without it
        event(home, "pass-row-error", pass_id=pass_id, error=type(row_exc).__name__, where=_where(row_exc))


def _where(exc: BaseException, frames: int = 8) -> list[str]:
    """Where *exc* was raised, as ``file:line:function`` for its innermost frames (file names only: no message, no
    value, so a recorded text never reaches the events)."""
    import traceback

    out = []
    for fs in traceback.extract_tb(exc.__traceback__)[-frames:]:
        out.append(f"{Path(fs.filename).name}:{fs.lineno}:{fs.name}")
    return out


def _turn_result() -> dict:
    return {
        "pass_id": None,
        "resumed": False,
        "handed": [],
        "read": [],
        "closed": [],
        "returned": [],
        "ended": None,
        "usd": None,
        "usd_known": None,
        "why_idle": None,
        "stop": None,
    }


def _passed(out: dict, ledger: Any, pass_id: str, handed: list, report: Any, *, resumed: bool) -> dict:
    """What one pass of the loop did: its entries (handed, verified read, closed by its decisions, gone back), how
    it ended, and its USD (booked, the unpriced calls at their worst case; and known)."""
    ids = [e.entry for e in handed]
    out.update(pass_id=pass_id, resumed=resumed, handed=ids)
    if report is not None:
        out["read"] = list(getattr(report, "read", None) or ())
        out["returned"] = list(getattr(report, "returned", None) or ())
        out["ended"] = getattr(report, "ended", None)
        out["usd"] = getattr(report, "settle_usd", None)
        out["usd_known"] = getattr(report, "usd_known", None)
    else:
        out["ended"] = "error"
    try:
        now = ledger.entries()
        out["closed"] = [i for i in ids if i in now and now[i].state == "closed"]
    except Exception:  # noqa: BLE001 - a ledger that cannot be read again: nothing said closed
        out["closed"] = []
    return out


async def _iteration(
    service: Service,
    ledger: Any,
    resumable: list[str],
    *,
    runner: Runner,
    clock: Callable[[], datetime],
) -> dict:
    """One turn of the service's loop, shared by :func:`service_main` and :func:`step`: the stop marker (and
    SIGTERM), the service's end (its lifetime), a pass left running to resume, else the capture step, ``sync``, the
    cap (every entry still owed is ``not_reached: spend_cap``), the lifetime's margin for a new pass, and one pass
    over up to K handed entries (one due capture group, or the other kinds of entry at once). Returns what it did
    (:func:`_turn_result`): ``stop`` names why the service must stop (None: go on); ``why_idle`` why no pass ran
    (:data:`IDLE_WHY`)."""
    from .attention import K
    from .integration.switch import PASS_WALL_S

    home, settings = service.home, service.settings
    out = _turn_result()
    marker = stopping_marker(home)
    if marker is not None:
        out.update(why_idle="stopping", stop=str(marker.get("by") or "operator"))
        return out
    if service.stop_flag is not None and service.stop_flag.is_set():
        out.update(why_idle="stopping", stop="term")
        return out
    end = lifetime_end(service.started)
    if clock() >= end:
        event(home, "lifetime", why="lifetime")
        out.update(why_idle="lifetime", stop="lifetime")
        return out
    if resumable:
        pass_id = resumable.pop(0)
        handed = ledger.in_pass(pass_id)
        report = await _run_pass(service, pass_id, handed, resume=True, runner=runner)
        return _passed(out, ledger, pass_id, handed, report, resumed=True)
    _capture_if_new(service)
    ledger.sync()
    if not any(e.state == "open" for e in ledger.entries().values()):
        out["why_idle"] = "nothing_open"
        return out
    try:
        # the writer's settings and route as a pass needs them, and one call's worst case for its models (a refused
        # setting or a model the pinned catalogue does not bound: no pass can run)
        check_writer_settings(settings)
        worst = pass_call_worst_case(settings)
    except Exception as exc:  # noqa: BLE001 - a refused writer route or setting: no pass can run
        event(home, "refused", reason=type(exc).__name__)
        out.update(why_idle="refused", stop="refused")
        return out
    committed, _settled = spend_totals(home)
    room = service.cap - committed
    # a pass starts only when its room holds at least one call at the worst case (its USD stop lets no call start
    # past the room), and it books the whole room until it settles
    if service.capped or room < worst:
        if not service.cap_noted:
            event(
                home,
                "cap-reached",
                by="service",
                committed_usd=_format(committed),
                room_usd=_format(room),
                call_worst_case_usd=_format(worst),
                cap_usd=_format(service.cap),
            )
            service.cap_noted = True
        ledger.mark_not_reached("spend_cap")
        out["why_idle"] = "spend_cap"
        return out
    if clock() + timedelta(seconds=float(PASS_WALL_S)) > end:
        # a pass started now could run past the service's end (its lifetime): no new pass; the service stops
        event(home, "lifetime", why="lifetime", refused_pass=True)
        out.update(why_idle="lifetime", stop="lifetime")
        return out
    pass_id = secrets.token_hex(8)
    # an entry is handed only once every episode it names can be read (a running one with no segment record waits,
    # with no miss), so no pass asks for an episode that is not recorded yet
    from .attention import HomeSources, entry_ready

    sources = HomeSources(home, state_dir=ledger.state_dir)
    resolve = sources.resolver()
    # a capture the harness's own relations relate to no other is put to the judge of behaviour once (one call per
    # turn), so its kin can relate it before the hand below
    await _judge_kin(service, ledger, _profiles(service, sources), worst)
    handed = ledger.hand(
        pass_id, K, ready=lambda e: entry_ready(e, resolve), grouping=_grouping(service, sources, clock)
    )
    if not handed:
        out["why_idle"] = "nothing_handed"
        return out
    report = await _run_pass(service, pass_id, handed, resume=False, runner=runner)
    return _passed(out, ledger, pass_id, handed, report, resumed=False)


def _profiles(service: Any, sources: Any) -> Any:
    """The profiles of a home's captures (:class:`.capture_groups.HomeProfiles`), kept on *service* across its turns
    (a capture's files never change once written)."""
    from .capture_groups import HomeProfiles

    profiles = getattr(service, "profiles", None)
    if profiles is None:
        try:
            evidence = sources.stores().evidence
        except Exception:  # noqa: BLE001 - without the evidence store, only the captures' own records relate them
            evidence = None
        profiles = HomeProfiles(sources.state_dir, evidence)
        try:
            service.profiles = profiles
        except AttributeError:
            pass
    return profiles


#: The judge's call's wall clock: a call still running then is cancelled (booked from the writer lane's journal, else
#: at its worst case), as a stop marker or SIGTERM cancels it at once.
KIN_WALL_S = 300.0
#: How often a judge's call in flight checks for the stop marker.
KIN_POLL_S = 1.0


class _KinCut(Exception):
    """The judge's call was cancelled: the service is stopping (``stopping``) or its wall clock ran out (``wall``)."""


def _stopping(service: Any) -> bool:
    """Whether *service* is asked to stop: the stop marker (a drain, a suspend, an operator) or SIGTERM."""
    flag = getattr(service, "stop_flag", None)
    return stopping_marker(service.home) is not None or (flag is not None and flag.is_set())


async def _bounded_call(call: Awaitable[Any], service: Any, wall_s: float) -> Any:
    """*call*'s result; cancelled (raising :class:`_KinCut`) as soon as *service* is asked to stop, or after *wall_s*."""
    task = asyncio.ensure_future(call)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + float(wall_s)
    while True:
        done, _ = await asyncio.wait({task}, timeout=max(0.0, min(KIN_POLL_S, deadline - loop.time())))
        if task in done:
            return task.result()
        why = "stopping" if _stopping(service) else "wall" if loop.time() >= deadline else None
        if why is not None:
            task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - the cancelled call's own end is not ours
                pass
            raise _KinCut(why)


def _writer_lane(settings: Any) -> tuple[str | None, tuple[str, ...], Any]:
    """The writer lane's journal, the writer's model as the journal names it, and the pinned writer prices (each
    None or empty when the settings do not give it)."""
    from .integration import consolidate
    from .integration.switch import writer_journal, writer_prices

    try:
        journal = writer_journal(settings)
    except Exception:  # noqa: BLE001 - no journal: the call's own figures settle it
        journal = None
    try:
        models: tuple[str, ...] = (consolidate.writer_settings(settings).model.split("@", 1)[0],)
    except Exception:  # noqa: BLE001
        models = ()
    try:
        prices = writer_prices(settings)
    except Exception:  # noqa: BLE001
        prices = None
    return journal, models, prices


async def _judge_kin(service: Any, ledger: Any, profiles: Any, worst: Decimal) -> bool:
    """Ask the judge of behaviour (:mod:`.kinship`) about one open capture that no relation of the harness's own
    relates to another capture and that was never asked about, the oldest first; record its answer (a ``kin`` row,
    with or without kin). The call is booked in ``spend.jsonl`` like a pass (its worst case reserved, then settled
    from the writer lane's journal window around the call, as a pass is; with no readable journal, from the call's
    own figures, an unpriced call at its worst case), carries its own id and the entry (``X-Unify-Pass-Id`` /
    ``-Entries``), and its row in ``pass-results.jsonl`` (kind ``kin``) gives its USD to the entry. It sends no tools.
    No call starts once the service is asked to stop (the stop marker of a drain, a suspend or an operator, or
    SIGTERM), or when the room under memory's cap is below one call's worst case; a call in flight is cancelled as
    soon as the service is asked to stop, or after :data:`KIN_WALL_S` (a call cut by a stop is asked again by a later
    service). Returns whether a judge was asked."""
    from unify.common.llm_client import attribution

    from . import capture_groups as cg
    from . import kinship
    from .captures import store_dir
    from .reconcile import journal_offset, reconcile_window

    home, settings = service.home, service.settings
    if _stopping(service):
        return False
    entries = ledger.entries()
    live = ("open", "handed", "read", "closed")
    caps = [e for e in entries.values() if e.kind in cg.GROUPED_KINDS and e.state in live and e.refs.captures]
    if len(caps) < 2:
        return False
    try:
        profs = {e.entry: profiles(e) for e in caps}
    except Exception:  # noqa: BLE001 - unreadable captures: the harness's own relations alone
        return False
    common = cg.common_features(profs.values())
    target = None
    for e in sorted((e for e in caps if e.state == "open" and not e.kin_judged), key=lambda e: (e.created_at, e.entry)):
        if not any(cg.related(profs[e.entry], profs[o.entry], common) for o in caps if o.entry != e.entry):
            target = e
            break
    if target is None:
        return False
    committed, _settled = spend_totals(home)
    if service.capped or service.cap - committed < worst:
        return False
    store = store_dir(ledger.state_dir)
    others = sorted((o for o in caps if o.entry != target.entry), key=lambda o: (o.created_at, o.entry), reverse=True)
    messages, labels = kinship.question(
        (profs[target.entry], store / target.refs.captures[0]),
        [(o.entry, profs[o.entry], store / o.refs.captures[0]) for o in others[: kinship.CANDIDATES]],
    )
    kin_id = secrets.token_hex(8)
    journal, models, prices = _writer_lane(settings)
    start = journal_offset(journal)
    spend = service_dir(home) / SPEND_FILE
    _append(spend, {"v": 1, "op": "reserve", "pass_id": kin_id, "usd": _format(worst), "at": stamp(_now())})
    setter = getattr(service, "set", None)
    if callable(setter):
        # a stop waits for a call in flight until its deadline, as for a pass (the call itself ends at the marker)
        setter("idle", pass_deadline=_now() + timedelta(seconds=KIN_WALL_S))
    usd: Decimal | None = None
    cut: str | None = None
    try:
        turn = _kin_turn(service, kin_id)
        with attribution(kin_id, [target.entry]):
            message, cost = await _bounded_call(turn(messages, None), service, KIN_WALL_S)
        usd = _decimal(cost)
        named, why = kinship.answer(message if isinstance(message, dict) else {}, labels)
    except _KinCut as exc:
        cut = str(exc)
        named, why = [], f"the judge's call was cancelled ({cut})"
    except Exception as exc:  # noqa: BLE001 - a failed call: asked once, nothing named, booked at its worst case
        named, why = [], f"the judge's call failed ({type(exc).__name__})"
    finally:
        if callable(setter):
            setter("idle", pass_deadline=None)
    recon = None
    if journal and models and start is not None:
        try:
            recon = await reconcile_window(
                journal, start, journal_offset(journal), models=models, worst_case_usd=worst, prices=prices
            )
        except Exception:  # noqa: BLE001 - an unreadable journal: the call's own figures
            recon = None
    if recon is not None and recon.get("journal") == "read":
        known, unknown = Decimal(recon["usd"]), int(recon["unknown"] or 0)
    else:
        known, unknown = (usd, 0) if usd is not None else (Decimal(0), 1)
    booked = known + unknown * worst
    _append(spend, {"v": 1, "op": "settle", "pass_id": kin_id, "usd": _format(booked), "at": stamp(_now())})
    if cut != "stopping":
        ledger.record_kin(target.entry, named, why)
    _kin_row(home, kin_id, target.entry, known, unknown, worst)
    event(home, "kin", pass_id=kin_id, entry=target.entry, named=len(named), **({"cut": cut} if cut else {}))
    return True


def _kin_turn(service: Any, kin_id: str) -> Any:
    """The judge's model turn: the writer's model on the writer's lane (*service*'s ``kin_turn`` factory in a test)."""
    factory = getattr(service, "kin_turn", None)
    if factory is not None:
        return factory()
    from .integration import consolidate
    from .integration.request import actor_effort

    cfg = consolidate.writer_settings(service.settings)
    if cfg.route is not None:
        return consolidate.unillm_turn(cfg.model, actor_effort(), route=cfg.route, session=f"kinship.{kin_id}")
    return consolidate.unillm_turn(cfg.model, actor_effort())


def _kin_row(home: Path | str, kin_id: str, entry: str, known: Decimal, unknown: int, worst: Decimal) -> None:
    """The judge's ``pass-results.jsonl`` row (kind ``kin``): the entry it was asked about, read, and its USD (*known*,
    and its *unknown* unpriced calls with their worst case booked), so metering gives the call's cost to that entry."""
    from . import metering

    row = {
        "pass_id": kin_id,
        "kind": "kin",
        "ended": "kin",
        "entries": {"handed": [entry], "taken": [], "read": [entry], "closed": [], "returned": []},
        "items": {},
        "usd": {
            "known": _format(known),
            "unknown_calls": unknown,
            "worst_case_booked": _format(unknown * worst),
            "killed_turns": 0,
            "helper_known": "0",
            "helper_unknown_calls": 0,
        },
        "ended_at": stamp(_now()),
    }
    path = metering.results_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    _append(path, row)
    metering.rebuild_views(home)


def _grouping(service: Any, sources: Any, clock: Callable[[], datetime]) -> Callable[[dict], list]:
    """The capture grouping the pass hands with (:func:`.capture_groups.groups`, at *clock*'s time), over the
    home's capture profiles (:func:`_profiles`)."""
    from .capture_groups import groups

    profiles = _profiles(service, sources)

    def grouping(entries: dict) -> list:
        return groups(entries, profiles, now=clock())

    return grouping


async def step_async(
    home: Path | str,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
    runner: Optional[Runner] = None,
) -> dict:
    """Deterministic step mode (the offline replay drives the service loop in-process): exactly one turn of
    :func:`service_main`'s loop (:func:`_iteration`, the same code) in the calling process, returning when it is done.
    No fork or ``setsid``, no heartbeat thread, no idle sleep, no idle exit, no SIGTERM handler; the model turns are
    the pass's own (``consolidate.unillm_turn``, which the replay routes), or *runner*'s.

    The step holds the home's lock while it runs: when a service (or another step) holds it, nothing is done and
    ``why_idle`` is ``busy``. The step is the service for its duration: the writer's roster row (run record on) is taken back
    (``rejoin``) and left ``idle``; a pass a crash left running is resumed first; at a stop (the stop marker, the
    lease) the step does what the service does when it stops for that reason (the entries still owed are marked
    ``not_reached`` as :data:`NOT_REACHED_BY` says, the writer's row ends). The service's lifetime is a bound on a
    live process and does not apply to a step; the host's lease does, against *now* (UTC; default the clock).

    Returns ``{"pass_id", "resumed", "handed", "read", "closed", "returned", "ended", "usd", "usd_known",
    "why_idle", "stop"}``: the entry ids the pass was handed, verified read, closed and sent back; ``usd`` is what
    the service settled (known plus unpriced calls at their worst case; None when nothing was readable), and
    ``why_idle`` why no pass ran (:data:`IDLE_WHY`), or None."""
    from .attention import Ledger
    from .integration.switch import memory_cap_usd, writer_refusal

    home = Path(home)
    settings = settings if settings is not None else _settings()
    if runner is None:
        from .integration.consolidate import run_attention_pass as runner  # type: ignore[no-redef]
    out = _turn_result()
    refused = writer_refusal(settings)
    if refused is not None:
        event(home, "refused", reason=refused)
        out.update(why_idle="refused", stop="refused")
        return out
    lock_fd = try_lock(service_dir(home) / LOCK_FILE)
    if lock_fd is None:
        out["why_idle"] = "busy"
        return out
    os.set_inheritable(lock_fd, False)
    at = (lambda: now) if now is not None else _now
    service: Optional[Service] = None
    try:
        service = Service(home, settings, lock_fd, started=at(), record_members=False)
        service.cap = memory_cap_usd(settings)
        # a cap an earlier step of this run reached stays as the service keeps it: noted once; a lane's refusal holds
        # every later pass of the run back (the service's own cap is the room, checked afresh each step)
        rows = events(home)
        starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
        reached = [r for r in rows[(starts[-1] + 1) if starts else 0 :] if r.get("event") == "cap-reached"]
        service.cap_noted = bool(reached)
        service.capped = any(r.get("by") == "proxy" for r in reached)
        record = _hold_writer_row(home)
        ledger = Ledger(home)
        resumable = pending_passes(home)
        ledger.return_orphans(live_pass_ids=resumable)
        service.set("idle")
        out = await _iteration(service, ledger, resumable[:1], runner=runner, clock=at)
        if out["stop"] is not None:
            reached = NOT_REACHED_BY.get(out["stop"])
            if reached is not None:
                ledger.mark_not_reached(reached)
            _end_writer_row(record, out["stop"])
        return out
    finally:
        # stopped whatever happened: a later stop or start never takes the caller's process for a running service
        if service is not None:
            try:
                service.set("stopped")
            except OSError:
                pass
        os.close(lock_fd)


def step(
    home: Path | str,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
    runner: Optional[Runner] = None,
) -> dict:
    """:func:`step_async` from code that is not in an event loop."""
    return asyncio.run(step_async(home, settings=settings, now=now, runner=runner))


async def service_main(
    home: Path | str,
    lock_fd: Optional[int] = None,
    *,
    runner: Optional[Runner] = None,
    settings: Any = None,
    poll_s: float = POLL_S,
    idle_exit_s: float = IDLE_EXIT_S,
    heartbeat_s: float = HEARTBEAT_S,
    clock: Callable[[], datetime] = _now,
    monotonic: Callable[[], float] = time.monotonic,
    members_s: float = MEMBERS_S,
) -> str:
    """The service's life (module docstring); returns why it stopped. It takes the home's lock itself (*lock_fd*: one
    already taken, for tests), and returns ``busy`` at once, leaving no trace, when another service holds it. *clock*
    (UTC now) and *monotonic* (the idle clock) are injected by tests."""
    home = Path(home)
    own_lock = lock_fd is None
    if own_lock:
        lock_fd = try_lock(service_dir(home) / LOCK_FILE)
        if lock_fd is None:
            return "busy"  # another service holds the home: this one leaves no trace
    os.set_inheritable(lock_fd, False)
    try:
        return await _serve(home, lock_fd, runner=runner, settings=settings, poll_s=poll_s, idle_exit_s=idle_exit_s,
                            heartbeat_s=heartbeat_s, clock=clock, monotonic=monotonic, members_s=members_s)
    finally:
        if own_lock:
            os.close(lock_fd)


async def _serve(
    home: Path,
    lock_fd: int,
    *,
    runner: Optional[Runner],
    settings: Any,
    poll_s: float,
    idle_exit_s: float,
    heartbeat_s: float,
    clock: Callable[[], datetime],
    monotonic: Callable[[], float],
    members_s: float = MEMBERS_S,
) -> str:
    from .attention import Ledger
    from .integration.switch import memory_cap_usd, writer_refusal

    settings = settings if settings is not None else _settings()
    if runner is None:
        from .integration.consolidate import run_attention_pass as runner  # type: ignore[no-redef]
    refused = writer_refusal(settings)
    if refused is not None:
        event(home, "refused", reason=refused)
        return "refused"
    previous = read_state(home)
    service = Service(home, settings, lock_fd, started=clock())
    service.cap = memory_cap_usd(settings)
    stop_flag = asyncio.Event()
    service.stop_flag = stop_flag
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_flag.set)
    if (
        isinstance(previous, dict)
        and previous.get("state") not in ("stopped", None)
        and previous.get("pid") != os.getpid()
    ):
        event(
            home,
            "takeover",
            previous_pid=previous.get("pid"),
            by="start",
            reason="the previous service ended without stopping",
        )
    service.set("starting")
    record = _hold_writer_row(home)
    beating = threading.Event()

    def heart() -> None:
        while not beating.wait(heartbeat_s):
            try:
                service.beat()
            except OSError:
                pass

    beat_thread = threading.Thread(
        target=heart, name="writer-service-heartbeat", daemon=True
    )
    beat_thread.start()

    def write_members() -> None:
        # the service's processes, often (a stop or takeover after a kill finds what it started up to MEMBERS_S before)
        while not beating.wait(members_s):
            try:
                _atomic_json(
                    service_dir(home) / MEMBERS_FILE,
                    {"pid": os.getpid(), "proc_start": proc_start(os.getpid()) or "", "at": stamp(_now()),
                     "members": _members()},
                )  # fmt: skip
            except OSError:
                pass

    if service.record_members:
        threading.Thread(target=write_members, name="writer-service-members", daemon=True).start()
    event(
        home, "start", cap_usd=_format(service.cap)
    )
    # a service start is no new run: memory's spend cap counts from the run's own start (begin_run), so a restart (its
    # lifetime, a crash, a takeover) keeps the run's total
    ledger = Ledger(home)
    resumable = pending_passes(home)
    ledger.return_orphans(live_pass_ids=resumable)
    why = "stopped"
    idle_since = monotonic()
    try:
        service.set("idle")
        while True:
            turn = await _iteration(service, ledger, resumable, runner=runner, clock=clock)
            if turn["stop"] is not None:
                why = turn["stop"]
                break
            if turn["pass_id"] is not None:
                idle_since = monotonic()
                continue
            if turn["why_idle"] == "nothing_open" and monotonic() - idle_since >= idle_exit_s:
                why = "idle-exit"
                event(home, "idle-exit", idle_s=round(monotonic() - idle_since, 3))
                break
            try:
                await asyncio.wait_for(stop_flag.wait(), timeout=poll_s)
            except asyncio.TimeoutError:
                pass
    except (
        Exception
    ) as exc:  # noqa: BLE001 - the class only: a message could carry recorded text
        event(home, "error", error=type(exc).__name__)
        why = "error"
        raise
    finally:
        beating.set()
        reached = NOT_REACHED_BY.get(why)
        if reached is not None:
            try:
                Ledger(home).mark_not_reached(reached)
            except Exception:  # noqa: BLE001 - the stop is still recorded
                pass
        _end_writer_row(record, why)
        service.set("stopped")
        event(home, "stop", why=why)
    return why


# --- the process --------------------------------------------------------------------------------------------------


def _parse(argv: Optional[list[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="writer_service")
    p.add_argument("--home", required=True)
    p.add_argument("--daemon", action="store_true")
    return p.parse_args(argv)


def release_route_fds(settings: Any = None) -> None:
    """Read the writer's and the helpers' route tokens from the descriptors this process was started with, keep them in
    memory only (:mod:`.integration.switch`), and close those descriptors, before the service forks: no process of the
    service ever holds a token pipe. Called by :func:`main` before :func:`daemonize`. A descriptor the reads left open
    (one the settings did not name) is closed too, when it is still the inherited pipe or file it was started with;
    a descriptor this process opened itself is never inherited (PEP 446) and is left alone."""
    import stat as _stat

    from .integration import switch

    settings = settings if settings is not None else _settings()
    numbers: set[int] = set()
    for name in (switch.WRITER_TOKEN_FD, switch.LIBRARY_HELPER_TOKEN_FD):
        values = [v for k, v in os.environ.items() if k.upper() == name]
        values.append(str(getattr(settings, name, "") or ""))
        for v in values:
            v = v.strip()
            if v.isdigit() and int(v) > 2:
                numbers.add(int(v))
    switch.settle_writer_route_env(os.environ, settings)
    switch.settle_helper_route_env(os.environ, settings)
    for fd in sorted(numbers):
        try:
            mode = os.fstat(fd).st_mode
            if os.get_inheritable(fd) and (_stat.S_ISFIFO(mode) or _stat.S_ISREG(mode)):
                os.close(fd)
        except OSError:
            pass  # read and closed already


def daemonize() -> None:
    """The second fork: the first child exits at once; the service is nobody's child, in a session of its own."""
    if os.fork() != 0:
        os._exit(0)
    os.setsid()


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse(argv)
    # the lane tokens into memory and their pipes closed before the fork: no process of the service holds one
    release_route_fds()
    if args.daemon:
        daemonize()
    try:
        asyncio.run(service_main(args.home))
        return 0
    except Exception as exc:  # noqa: BLE001 - the class only
        print(f"writer service: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
