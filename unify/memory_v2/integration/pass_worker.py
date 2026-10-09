"""The detached process that runs one memory v2.1 consolidation pass (spec §6, D43).

Started by :func:`.async_pass.maybe_spawn` in its own session (its own process group), with the pass lock on the
inherited descriptor ``--lock-fd``. It writes ``pass-inflight.json``, runs the due pass through
:func:`.consolidate.run_due_passes` under a :class:`.async_pass.Supervisor` (wall bound; SIGTERM cancels through
the abort path) and reconciles its calls from the Sol lane's journal window, appends one result row, clears its in-flight record and
exits. SIGALRM at
:func:`.async_pass.failsafe_s` ends it whatever happens; the run guard then keeps the pass's whole cap committed.
Errors go to the state directory's ``pass-worker.log`` by class name only, never by message.
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
    from .request import check_v21_switches
    from .switch import sol_journal, v21_pass_wall_s

    paths = Paths.under(Path(args.home))
    check_v21_switches(SETTINGS)
    wall_s = float(v21_pass_wall_s(SETTINGS))
    pass_id = f"{args.episode}.p0"  # run_due_passes names a session's first pass .p0
    now = time.time()
    ap.write_inflight(
        paths,
        ap.InFlight(
            pass_id,
            os.getpid(),
            os.getpgid(0),
            ap.proc_start(os.getpid()) or "",
            now,
            now + wall_s,
            args.episode,
        ),
    )
    try:
        stores = consolidate.open_stores(paths, busy_timeout_s=ap.BUSY_TIMEOUT_S)
        view = ap.StateView.load(paths.state)
        sol_model = consolidate.sol_settings(SETTINGS).model.split("@", 1)[0]
        # P7 Amendment D: the pass's calls are the Sol lane's journal rows in the window the supervisor records
        sup = ap.Supervisor(
            wall_s,
            stop,
            journal=sol_journal(SETTINGS),
            models=(sol_model,),
        )
        outcomes = await consolidate.run_due_passes(
            stores,
            args.episode,
            args.sha,
            view,
            effort=args.effort,
            settings=SETTINGS,
            emit=None,
            supervise=sup,
        )
        if sup.last is None:
            return None  # nothing was due, or the run guard held it back: no pass ran
        drift, suspect = view.cleared()
        out = outcomes[0] if outcomes else None
        row = {
            "pass_id": pass_id,
            "ended": sup.last.ended,
            "commit": getattr(out, "commit", None),
            "drift_cleared": drift,
            "suspect_cleared": suspect,
            "seconds": round(sup.last.seconds, 3),
            "reconciled": sup.reconciled,
            "ended_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        ap.record_result(paths, row)
        return row
    finally:
        ap.clear_inflight(paths, pass_id)


async def _main(args: argparse.Namespace) -> dict | None:
    stop = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    return await _work(args, stop)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        from unify.settings import SETTINGS

        from .async_pass import failsafe_s
        from .switch import v21_pass_wall_s

        signal.alarm(int(failsafe_s(v21_pass_wall_s(SETTINGS))) + 1)
        asyncio.run(_main(args))
        return 0
    except Exception as exc:  # noqa: BLE001
        # the class only: a message could carry recorded text
        print(f"memory v2.1 pass worker: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        try:
            os.close(args.lock_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
