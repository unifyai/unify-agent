"""One turn of the writer service's loop in the calling process: the service tests' deterministic step."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from unify.memory import writer_service as ws


async def step_async(
    home: Path | str,
    *,
    settings: Any = None,
    now: Optional[datetime] = None,
    runner: Optional[ws.Runner] = None,
) -> dict:
    """Deterministic step mode for tests: exactly one turn of the service's loop (``writer_service._iteration``, the
    code :func:`unify.memory.writer_service.service_main` runs) in the calling process, returning when it is done.
    No fork or ``setsid``, no heartbeat thread, no idle sleep, no idle exit, no SIGTERM handler; the model turns are
    the pass's own (``consolidate.unillm_turn``), or *runner*'s.

    The step holds the home's lock while it runs: when a service (or another step) holds it, nothing is done and
    ``why_idle`` is ``busy``. The step is the service for its duration: the writer's roster row (run record on) is taken back
    (``rejoin``) and left ``idle``; a pass a crash left running is resumed first; at a stop (the stop marker, the
    lease) the step does what the service does when it stops for that reason (the entries still owed are marked
    ``not_reached`` as ``writer_service.NOT_REACHED_BY`` says, the writer's row ends). The service's lifetime is a bound on a
    live process and does not apply to a step; the host's lease does, against *now* (UTC; default the clock).

    Returns ``{"pass_id", "resumed", "handed", "read", "closed", "returned", "ended", "usd", "usd_known",
    "why_idle", "stop"}``: the entry ids the pass was handed, verified read, closed and sent back; ``usd`` is what
    the service settled (known plus unpriced calls at their worst case; None when nothing was readable), and
    ``why_idle`` why no pass ran, or None."""
    from unify.memory.attention import Ledger
    from unify.memory.integration.switch import memory_cap_usd, writer_refusal

    home = Path(home)
    settings = settings if settings is not None else ws._settings()
    if runner is None:
        from unify.memory.integration.consolidate import run_attention_pass as runner  # type: ignore[no-redef]
    out = ws._turn_result()
    refused = writer_refusal(settings)
    if refused is not None:
        ws.event(home, "refused", reason=refused)
        out.update(why_idle="refused", stop="refused")
        return out
    lock_fd = ws.try_lock(ws.service_dir(home) / ws.LOCK_FILE)
    if lock_fd is None:
        out["why_idle"] = "busy"
        return out
    os.set_inheritable(lock_fd, False)
    at = (lambda: now) if now is not None else ws._now
    service: Optional[ws.Service] = None
    try:
        service = ws.Service(home, settings, lock_fd, started=at(), record_members=False)
        service.cap = memory_cap_usd(settings)
        # a cap an earlier step of this run reached stays as the service keeps it: noted once; a lane's refusal holds
        # every later pass of the run back (the service's own cap is the room, checked afresh each step)
        rows = ws.events(home)
        starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
        reached = [r for r in rows[(starts[-1] + 1) if starts else 0 :] if r.get("event") == "cap-reached"]
        service.cap_noted = bool(reached)
        service.capped = any(r.get("by") == "proxy" for r in reached)
        record = ws._hold_writer_row(home)
        ledger = Ledger(home)
        resumable = ws.pending_passes(home)
        ledger.return_orphans(live_pass_ids=resumable)
        service.set("idle")
        out = await ws._iteration(service, ledger, resumable[:1], runner=runner, clock=at)
        if out["stop"] is not None:
            reached = ws.NOT_REACHED_BY.get(out["stop"])
            if reached is not None:
                ledger.mark_not_reached(reached)
            ws._end_writer_row(record, out["stop"])
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
    runner: Optional[ws.Runner] = None,
) -> dict:
    """:func:`step_async` from code that is not in an event loop."""
    return asyncio.run(step_async(home, settings=settings, now=now, runner=runner))
