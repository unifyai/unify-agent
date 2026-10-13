"""Symbolic: the sandboxed Python worker starts every time, promptly.

Each benchmark episode, and each fresh ``SessionExecutor``, starts a worker in
bubblewrap. A start that hangs (the worker's ready line never comes) stalls
the whole episode until ``START_TIMEOUT_S``. This runs 30 starts in a row, a
fresh executor each, and fails on any start that errs, hangs or is slow. No
model is reached; the cell is ``1``.
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest

from tests.actor.code_act.sandbox_world import (  # noqa: F401 (fixture)
    needs_bwrap,
    world,
)
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.worker import START_TIMEOUT_S

STARTS = 30
# Generous: a normal start takes a second or two. What this catches is a hang.
SLOW_S = 15.0
# A run of hangs stops the loop early, so the test stays bounded.
MAX_FAILURES = 3


@needs_bwrap
@pytest.mark.asyncio
# 30 starts under SLOW_S each (450 s) plus MAX_FAILURES hangs of
# START_TIMEOUT_S and a close each (about 210 s), with room to spare.
@pytest.mark.timeout(900)
@pytest.mark.usefixtures("world")
async def test_thirty_consecutive_worker_starts_all_succeed_promptly():
    times: list[float] = []
    failures: list[str] = []
    for i in range(STARTS):
        ex = SessionExecutor()
        t0 = time.perf_counter()
        try:
            res = await asyncio.wait_for(
                ex.execute(code="1", state_mode="stateful", session_id=0),
                timeout=START_TIMEOUT_S,
            )
            elapsed = time.perf_counter() - t0
            if res.get("error") is not None or res.get("result") != 1:
                failures.append(f"start {i}: {res.get('error')!r}")
            else:
                times.append(elapsed)
        except Exception as exc:  # a hang (TimeoutError) or a failed start
            failures.append(
                f"start {i}: {type(exc).__name__} after "
                f"{time.perf_counter() - t0:.1f}s: {exc}",
            )
        finally:
            await asyncio.wait_for(ex.close(), timeout=15)
        if len(failures) >= MAX_FAILURES:
            break
    summary = (
        f"{len(times)} of {STARTS} starts ok; "
        + (
            f"p50 {statistics.median(times):.2f}s, max {max(times):.2f}s"
            if times
            else "none succeeded"
        )
        + (f"; failures: {failures}" if failures else "")
    )
    assert not failures, summary
    assert len(times) == STARTS, summary
    assert max(times) < SLOW_S, summary
