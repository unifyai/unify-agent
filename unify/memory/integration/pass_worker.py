"""The detached process that runs one library-mode consolidation pass.

Started by :func:`.async_pass.maybe_spawn` in its own session (its own process group), with the pass lock on the
inherited descriptor ``--lock-fd``. It writes ``pass-inflight.json``, runs the due pass through
:func:`.consolidate.run_due_passes` under a :class:`.async_pass.Supervisor` (wall bound; SIGTERM cancels through
the abort path) and reconciles its calls from the writer lane's journal window, and appends one result row. Then,
holding the same lock, it runs each writer pass a verdict made due meanwhile, one slot after the other
(:func:`.consolidate.run_slots`): each slot rewrites the in-flight record, re-arms the failsafe and appends its own
row. It clears its in-flight record and exits. SIGALRM at :func:`.async_pass.failsafe_s` of a slot ends it whatever
happens; the run guard then keeps the pass's whole cap committed. Errors go to the state directory's
``pass-worker.log`` by class name only, never by message.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import signal
import sys
import time
from pathlib import Path


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pass_worker")
    p.add_argument("--home", required=True)
    p.add_argument("--episode", required=True)
    p.add_argument("--sha", required=True)
    p.add_argument("--effort", required=True)
    p.add_argument("--lock-fd", type=int, required=True)
    return p.parse_args(argv)


async def _work(args: argparse.Namespace, stop: asyncio.Event) -> dict | None:
    from unify.settings import SETTINGS

    from . import async_pass as ap
    from . import consolidate
    from .paths import Paths
    from .request import check_library_switches
    from .switch import writer_journal, library_pass_wall_s

    paths = Paths.under(Path(args.home))
    check_library_switches(SETTINGS)
    wall_s = float(library_pass_wall_s(SETTINGS))
    current = {"pass_id": f"{args.episode}.p0"}  # run_due_passes names a session's first pass .p0

    def on_slot(k: int, pass_id: str) -> None:
        current["pass_id"] = pass_id
        if k:
            # each slot has the failsafe of one slot (WRITE, then the CURATE it made due)
            signal.alarm(int(ap.failsafe_s(wall_s, passes=ap.WORKER_PASSES)) + 1)
        now = time.time()
        ap.write_inflight(
            paths,
            ap.InFlight(
                pass_id,
                os.getpid(),
                os.getpgid(0),
                ap.proc_start(os.getpid()) or "",
                now,
                now + ap.slot_s(wall_s),  # the slot: WRITE, then the CURATE it made due, each with its records
                args.episode,
            ),
        )

    on_slot(0, current["pass_id"])
    try:
        stores = consolidate.open_stores(paths, busy_timeout_s=ap.BUSY_TIMEOUT_S)
        view = ap.StateView.load(paths.state)
        writer_model = consolidate.writer_settings(SETTINGS).model.split("@", 1)[0]
        reported: dict[str, set] = {"drift": set(), "suspect": set()}
        rows: list[dict] = []

        def supervisor(k: int) -> ap.Supervisor:
            # the pass's calls are the writer lane's journal rows in the window the supervisor records
            return ap.Supervisor(wall_s, stop, journal=writer_journal(SETTINGS), models=(writer_model,))

        def slot_end(k: int, sup: ap.Supervisor, outcomes: list) -> None:
            # each slot's row, as it ends: the clears since the slot before it
            drift, suspect = view.cleared()
            row = _row(
                args.episode,
                k,
                sup,
                outcomes,
                sorted(set(drift) - reported["drift"]),
                sorted(set(suspect) - reported["suspect"]),
            )
            reported["drift"].update(drift)
            reported["suspect"].update(suspect)
            ap.record_result(paths, row)
            rows.append(row)

        await consolidate.run_slots(
            stores,
            args.episode,
            args.sha,
            view,
            effort=args.effort,
            settings=SETTINGS,
            supervisor=supervisor,
            on_slot=on_slot,
            on_slot_end=slot_end,
        )
        return rows[-1] if rows else None  # none: nothing was due, or the run guard held it back
    finally:
        ap.clear_inflight(paths, current["pass_id"])


def _row(episode: str, k: int, sup, outcomes: list, drift: list, suspect: list) -> dict:
    """Slot *k*'s result row: its WRITE pass, and the CURATE it made due beside it."""
    out = outcomes[0] if outcomes else None
    first, first_recon = sup.runs[0] if sup.runs else (sup.last, sup.reconciled)
    row = {
        "pass_id": f"{episode}.p{2 * k}",
        "ended": first.ended,
        "commit": getattr(out, "commit", None),
        "drift_cleared": drift,
        "suspect_cleared": suspect,
        "seconds": round(first.seconds, 3),
        "reconciled": first_recon,
        "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    if len(sup.runs) > 1:
        curate, curate_recon = sup.runs[1]
        row["curate"] = {
            "pass_id": f"{episode}.p{2 * k + 1}",
            "ended": curate.ended,
            "commit": getattr(curate.outcome, "commit", None),
            "seconds": round(curate.seconds, 3),
            "reconciled": curate_recon,
        }
    return row


async def _main(args: argparse.Namespace) -> dict | None:
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    return await _work(args, stop)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        from unify.settings import SETTINGS

        from .async_pass import WORKER_PASSES, failsafe_s
        from .switch import library_pass_wall_s

        signal.alarm(
            int(failsafe_s(library_pass_wall_s(SETTINGS), passes=WORKER_PASSES)) + 1,
        )
        asyncio.run(_main(args))
        return 0
    except Exception as exc:  # noqa: BLE001
        # the class only: a message could carry recorded text
        print(f"memory library pass worker: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        try:
            os.close(args.lock_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
