"""A stand-in writer service for the service's process tests: the real :func:`unify.memory.writer_service.service_main`
(its lock, heartbeat, stop marker, cap, lifetime, resume and termination) with a scripted pass in place of a writer
pass. :func:`unify.memory.writer_service.ensure_service` starts it through its ``argv_prefix``, exactly as it starts the
real service (the double fork, ``setsid``, the inherited lock).

The pass is a fixed number of scripted model turns through :class:`unify.memory.writer_service.ResumableTurn`; its
conversation is written to ``$DRIVER_OUT/result-<pass>.json`` and each model call to ``$DRIVER_OUT/model-calls.jsonl``.
``DRIVER_BLOCK_AT=n`` makes turn n hang (once: the first time, marked by ``$DRIVER_OUT/blocked``), so a test can kill
the service mid-pass; ``DRIVER_TURN_SLEEP`` makes each turn take that long unless the service is asked to stop.
No model, network or key is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

TURNS = int(os.environ.get("DRIVER_TURNS", "5"))


async def runner(
    home, pass_id, handed, *, resume, folder, journal_start, effort, settings, stop=None
):
    from unify.memory import attention
    from unify.memory.integration.consolidate import AttentionPassReport
    from unify.memory.writer_service import ResumableTurn

    out = Path(os.environ["DRIVER_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    block_at = int(os.environ.get("DRIVER_BLOCK_AT", "-1"))
    turn_sleep = float(os.environ.get("DRIVER_TURN_SLEEP", "0"))

    async def scripted(messages, tools):
        n = sum(1 for m in messages if m.get("role") == "assistant")
        with open(out / "model-calls.jsonl", "a") as fh:
            fh.write(
                json.dumps(
                    {"pass": pass_id, "turn": n, "pid": os.getpid(), "resume": resume}
                )
                + "\n"
            )
        if n == block_at and not (out / "blocked").exists():
            (out / "blocked").write_text(str(os.getpid()))
            await asyncio.sleep(3600)
        if turn_sleep:
            try:
                (
                    await asyncio.wait_for(stop.wait(), turn_sleep)
                    if stop is not None
                    else await asyncio.sleep(turn_sleep)
                )
            except asyncio.TimeoutError:
                pass
        return {
            "role": "assistant",
            "content": f"reply {n} to {messages[-1]['content']}",
        }, "0.001"

    turn = ResumableTurn(scripted, folder, inputs={"pass_id": pass_id})
    messages = [{"role": "user", "content": "go"}]
    for _ in range(TURNS):
        if stop is not None and stop.is_set():
            break
        msg, _usd = await turn(messages, [])
        messages.append(msg)
        messages.append({"role": "user", "content": f"next {len(messages)}"})
    (out / f"result-{pass_id}.json").write_text(json.dumps(messages))
    return AttentionPassReport(
        pass_id,
        [e.entry for e in handed],
        "cancelled" if stop is not None and stop.is_set() else "landed",
        killed_turns=list(turn.killed),
        turns=turn.n,
        replayed_turns=turn.replayed,
        reads=attention.PassReads(),
        usd_known="0.005",
        settle_usd="0.005",
    )


def main(argv=None) -> int:
    from unify.memory import writer_service as ws
    from unify.memory.integration import request

    args = ws._parse(argv)
    if args.daemon:
        ws.daemonize()
    ws.pass_reservation = lambda settings, effort: Decimal(
        os.environ.get("DRIVER_RESERVE", "0.01")
    )
    request.actor_effort = lambda: "low"
    asyncio.run(
        ws.service_main(
            args.home,
            runner=runner,
            poll_s=float(os.environ.get("DRIVER_POLL_S", "0.2")),
            idle_exit_s=float(os.environ.get("DRIVER_IDLE_S", "600")),
            heartbeat_s=float(os.environ.get("DRIVER_BEAT_S", "30")),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
