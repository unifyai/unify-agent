"""Loop-safe helpers for bridging sync and async call sites.

Runtime entrypoints already own an event loop via ``asyncio.run``. Nested ``asyncio.run`` then fails with
``RuntimeError: asyncio.run() cannot be called from a running event
loop``. Use :func:`run_coro_sync` from sync façades that need to drive
async work from either context.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

R = TypeVar("R")
T = TypeVar("T")


def run_coro_sync(factory: Callable[[], Awaitable[R]]) -> R:
    """Run an async factory from sync code, including under a running loop.

    When no loop is running, delegates to ``asyncio.run``. When a loop is
    already running, schedules the factory on a private worker thread that
    owns its own loop so nested ``asyncio.run`` is avoided on the caller
    thread. The ContextVars registered with :func:`carry_across_threads`
    keep the caller's values there; no other context crosses, since much of
    it holds objects bound to the caller's loop.

    Prefer ``async def`` + ``await`` end-to-end when authoring stored task
    entrypoints. Use this helper only when a sync façade is required (CLI
    runners, legacy sync libraries, sync symbolic helpers).
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    carried = [(var, var.get()) for var in _CARRIED]

    def _run() -> R:
        for var, value in carried:
            var.set(value)
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(contextvars.Context().run, _run).result()


_CARRIED: list[contextvars.ContextVar[Any]] = []


def carry_across_threads(var: contextvars.ContextVar[T]) -> contextvars.ContextVar[T]:
    """Register *var* to keep its value in :func:`run_coro_sync`'s worker thread.

    For values that must hold wherever the caller's work runs, such as the
    grants an actor run holds (unify/actor/grants.py). *var* must have a
    default. Returns *var*.
    """
    _CARRIED.append(var)
    return var


__all__ = ["carry_across_threads", "run_coro_sync"]
