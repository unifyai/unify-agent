"""The writer service at a request's end, the pass supervisor, the landed results and shutdown.

A request never waits for a pass. At a request's end :func:`request_end` makes sure the writer service
(:mod:`..writer_service`) is up: one detached process per memory home that hands the attention ledger's entries to its
passes, one pass at a time, each under its wall-clock bound (:class:`Supervisor`).

**Isolation.** Each request pins memory ``main`` when it starts (``RequestRun.pin``) and works on a read-only copy of
that commit. A pass works in its own temporary box and checkout, and its merge moves ``main`` with one compare-and-swap
(``Repo.fast_forward``, ``git update-ref <new> <old>``). A request therefore sees the old commit or the new one, never
a half-written library, and a commit takes effect from the next request that starts after it lands.

**Results.** The service appends one row per pass to ``pass-results.jsonl`` and never writes the request state
(``state.json``). The next request applies the rows under its lock (:func:`apply_results`) and reports each pass as
``landed``, naming itself: the first request that ran on the new commit.

**Shutdown** (:func:`drain`) never starts a pass: it stops the writer service (its pass in flight gets until its own
deadline, then SIGTERM cancels it through the abort path, and a process still alive after the grace is killed), and
verifies that no process of it is left. Every pass that ended after the last request is reported as such and is never
counted toward the stream.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..reconcile import journal_offset

#: Seconds of the wall bound kept for the final merge (the whole gate) after the writer's deadline.
MERGE_RESERVE_S = 600.0
#: What a drain ends (:func:`drain`): the run itself, or one stage of a run that goes on.
DRAIN_ENDS = ("final", "suspend")
_ENDED = ("landed", "refused", "deadline", "cancelled", "error")


# --- the supervisor ----------------------------------------------------------------------------------------


@dataclass
class Supervised:
    outcome: Any
    ended: str  # landed | refused | deadline | cancelled | error
    seconds: float
    #: with ``ended == "error"``: what the pass raised, as :func:`error_of` gives it (never a message)
    error: dict | None = None


def error_of(exc: BaseException) -> dict:
    """What a pass raised, for its pass row: ``{"class", "where"}``, the exception's class name and the innermost frame
    as ``file:line:function`` (file names only: no message, no value, so no recorded text reaches the row)."""
    import traceback

    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    where = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}:{frames[-1].name}" if frames else "unknown"
    return {"class": type(exc).__name__, "where": where}


class Supervisor:
    """Runs one pass coroutine under the wall-clock bound *wall_s*; *stop* (set by SIGTERM) cancels it early.
    Cancellation goes through the pass's own abort path: :meth:`..writer_pass.WriterPass.run` records the pass as
    failed and keeps its work as a draft."""

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
        # the writer route proxy's journal and the writer's models; the pass's window is the journal's byte
        # range from the pass's start to the end this supervisor records
        self.journal, self.models = journal, tuple(models)
        self.window: tuple[int | None, int | None] = (None, None)
        self.last: Supervised | None = None
        self.reconciled: dict | None = None
        # every pass of the slot, in order, each with its own reconciliation (a later pass never overwrites an earlier
        # one's)
        self.runs: list[tuple[Supervised, dict | None]] = []

    @property
    def pass_deadline_s(self) -> float:
        """The writer's own deadline: the wall bound less the final merge's reserve (at least a minute)."""
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
                # the pass's exception is kept as its class and frame (rev 2: a failed pass row always says why)
                result = Supervised(None, "error", self.clock() - t0, error=error_of(task.exception()))
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
        """The pass's writer calls priced from the journal window this supervisor recorded (by lane and
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


# --- the in-flight record, the lock and the result rows -------------------------------------------------------


def results_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass-results.jsonl"


def cursor_path(paths: Any) -> Path:
    return Path(paths.state_dir) / "pass-results.cursor"


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


# --- spawning (the request side) ---------------------------------------------------------------------------


def request_end(stores: Any, eid: str, *, settings: Any, emit: Callable[[dict], None] | None) -> dict:
    """At a request's end: make sure the writer service is up (it hands the attention ledger's entries to its passes),
    and return at once. Delivers one ``service`` event and returns it; never raises."""
    from ..writer_service import ensure_service

    try:
        state = ensure_service(stores.paths.home, settings=settings, wait_s=0.0)
    except Exception as exc:  # noqa: BLE001 - a request's end never fails on the writer
        from .consolidate import _error

        _error(stores, f"service: {type(exc).__name__}")
        return {"phase": "error"}
    event = {
        "type": "consolidation",
        "phase": "service",
        "episode_id": eid,
        "status": state.status,
        "pid": state.pid,
        **({"detail": state.detail} if state.detail else {}),
    }
    _deliver_service(stores, emit, event)
    return event


def _deliver_service(stores: Any, emit: Callable[[dict], None] | None, event: dict) -> None:
    from .consolidate import _deliver

    try:
        _deliver(stores, emit, event)
    except Exception:  # noqa: BLE001 - reporting never fails a request's end
        pass


def _route_token(settings: Any) -> str | None:
    """The writer's own route token as this process holds it (read once from its descriptor, or the setting), or
    None when the writer uses no route of its own."""
    from .switch import WRITER_BASE_URL, writer_token

    if not str(getattr(settings, WRITER_BASE_URL, "") or "").strip():
        return None
    token = writer_token(settings)
    value = (
        token.get_secret_value()
        if hasattr(token, "get_secret_value")
        else str(token or "")
    )
    return value.strip() or None


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
    cursor. Called under the request lock at a library-mode request's open."""
    rows, end = _rows_from(paths, _cursor(paths))
    if not rows and end == _cursor(paths):
        return []
    for row in rows:
        state.drift.difference_update(row.get("drift_cleared") or [])
        state.suspect.difference_update(row.get("suspect_cleared") or [])
    state.save()
    _atomic_write(cursor_path(paths), f"{end}\n")
    return rows


def landed_event(
    row: dict,
    episode_id: str,
    pin: str,
    library_items: int | None = None,
) -> dict:
    """The ``landed`` event: the pass, how it ended, the first request that ran after it, and the size of the
    library at the landed commit (*library_items*: functions and notes; None when it could not be counted).
    """
    return {
        "type": "consolidation",
        "phase": "landed",
        "pass_id": row.get("pass_id"),
        "ended": row.get("ended"),
        "commit": row.get("commit"),
        "library_commit": row.get("commit"),
        "library_items": library_items,
        "first_request": episode_id,
        "pinned": pin,
    }


# --- shutdown ------------------------------------------------------------------------------------------------


def _event(paths: Any, row: dict) -> None:
    path = Path(paths.state_dir) / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def drain(
    paths: Any,
    *,
    archive_to: Path | None = None,
    end: str = "final",
    service_wait_s: float | None = None,
) -> dict:
    """Shutdown: never starts a pass. *end* says what ends: ``final`` (the run itself: every attention entry still owed
    is then ``not_reached: run_ended``) or ``suspend`` (one stage of a run that goes on: the service stops the same
    way, and every entry still owed stays owed for the next start to hand again). The writer service is stopped and
    its end verified (:func:`_drain_service`); *service_wait_s* bounds the service's pass in flight: its deadline plus
    the stop's grace, or *service_wait_s* from now if that is sooner. Every result no request applied is reported
    under ``after_last_request`` and marked ``counted: false``. With *archive_to*, once the service is verified gone, the
    memory repo is archived with every ref, ``refs/notes/items`` and ``served`` included
    (:func:`..memory_writer.archive`; a plain clone would drop the item records). The report is also appended to
    ``events.jsonl``.
    """
    if end not in DRAIN_ENDS:
        raise ValueError(f"end is one of {', '.join(DRAIN_ENDS)}, not {end!r}")
    service = _drain_service(paths, end, service_wait_s)
    report: dict = {
        "end": end,
        "type": "consolidation",
        "phase": "shutdown",
        "service": service,
        "terminated": service["verified"],
        "ended": "stopped" if service["verified"] else "not_terminated",
        "counted": False,
        "after_last_request": [
            {"pass_id": r.get("pass_id"), "ended": r.get("ended"), "commit": r.get("commit")}
            for r in unapplied(paths)
        ],
    }
    # never while the service may still write
    if archive_to is not None and report["terminated"]:
        from ..gitio import Repo
        from ..memory_writer import archive

        try:
            report["archive"] = archive(Repo(Path(paths.memory)), Path(archive_to))
        except Exception as exc:  # noqa: BLE001
            # reported, never raised: shutdown must finish
            report["archive"] = {"error": type(exc).__name__}
    _event(paths, report)
    return report


def _drain_service(paths: Any, end: str = "final", wait_s: float | None = None) -> dict:
    """Stop the writer service (the stop marker, its pass's deadline and the grace, SIGTERM, SIGKILL) and verify that
    no process of it is left. At the ``final`` end every attention entry still owed is then ``not_reached:
    run_ended``; at a ``suspend`` nothing is marked (the entries stay owed; the next start hands them again, those a
    stopped pass held going back without a miss). Called before the model-call lanes are torn down."""
    from ..writer_service import stop_service

    final = end == "final"
    done = stop_service(
        paths.home,
        "drain" if final else "suspend",
        "the run ended" if final else "a stage of the run ended",
        wait_s=wait_s,
    )
    out = {"verified": done.verified, "how": done.how, "leftover_pids": list(done.leftover_pids), "end": end}
    if done.verified and not final:
        try:
            from ..attention import Ledger

            out["kept_owed"] = sum(1 for e in Ledger(paths.home).entries().values() if e.active)
        except Exception as exc:  # noqa: BLE001 - reported, never raised: shutdown must finish
            out["kept_owed_error"] = type(exc).__name__
        # the suspend's own stop marker is set aside (recorded ``stop-cleared``), so the next stage's service starts,
        # whether a request or the runner starts it; a marker of the final end stays and refuses every later start
        from ..writer_service import clear_previous_stop

        out["marker_cleared"] = clear_previous_stop(paths.home) is not None
    if done.verified and final:
        try:
            from ..attention import Ledger

            out["not_reached"] = len(Ledger(paths.home).mark_not_reached("run_ended"))
        except Exception as exc:  # noqa: BLE001 - reported, never raised: shutdown must finish
            out["not_reached_error"] = type(exc).__name__
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="async_pass")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser(
        "drain",
        help="shutdown: bounded wait, cancel, verified termination; never starts a pass",
    )
    d.add_argument("--home", required=True)
    d.add_argument(
        "--archive",
        default=None,
        help="a bundle path: every ref, refs/notes/items included",
    )
    d.add_argument(
        "--end",
        choices=DRAIN_ENDS,
        default="final",
        help="final: the run ends (entries still owed are not reached); suspend: one stage of a run that goes on",
    )
    args = p.parse_args(argv)
    from .paths import Paths

    report = drain(
        Paths.under(Path(args.home)),
        archive_to=Path(args.archive) if args.archive else None,
        end=args.end,
    )
    print(json.dumps(report, sort_keys=True))
    return 2 if report["terminated"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
