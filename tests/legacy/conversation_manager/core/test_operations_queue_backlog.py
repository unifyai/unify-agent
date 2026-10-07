"""
tests/conversation_manager/core/test_operations_queue_backlog.py
================================================================

An operations queue with no consumer must say so.

The queue is unbounded and its consumer (``listen_to_operations``) is
started separately from the producers, so an embedder that boots the CM
without spawning the listener loses every queued operation — EventBus
persistence among them — in total silence. The backlog warning is the one
grep-able trace of that state.
"""

from __future__ import annotations

import pytest

from unify.legacy.conversation_manager.domains import managers_utils as mu


async def _noop() -> None:
    return None


async def _drain_queue() -> None:
    while not mu._operations_queue.empty():
        mu._operations_queue.get_nowait()
        mu._operations_queue.task_done()


@pytest.fixture
def recorded_warnings(monkeypatch):
    """Capture the module LOGGER's warnings; it does not propagate to caplog."""
    calls: list[str] = []
    monkeypatch.setattr(
        mu.LOGGER,
        "warning",
        lambda msg, *args, **kwargs: calls.append(str(msg)),
    )
    return calls


def _backlog_warnings(calls: list[str]) -> list[str]:
    return [msg for msg in calls if "listen_to_operations" in msg]


@pytest.mark.asyncio
async def test_backlog_without_consumer_warns_once(recorded_warnings):
    await _drain_queue()
    mu._operations_backlog_warned = False
    try:
        for _ in range(mu._OPERATIONS_QUEUE_BACKLOG_WARN_AT * 2):
            await mu.queue_operation(_noop)

        assert len(_backlog_warnings(recorded_warnings)) == 1, (
            "one backlog episode must produce exactly one warning, "
            f"got {len(_backlog_warnings(recorded_warnings))}"
        )
    finally:
        await _drain_queue()
        mu._operations_backlog_warned = False


@pytest.mark.asyncio
async def test_backlog_warning_rearms_after_drain(recorded_warnings):
    await _drain_queue()
    mu._operations_backlog_warned = False
    try:
        for _ in range(mu._OPERATIONS_QUEUE_BACKLOG_WARN_AT):
            await mu.queue_operation(_noop)
        # The consumer catches up: the next quiet-side enqueue re-arms.
        await _drain_queue()
        await mu.queue_operation(_noop)
        await _drain_queue()
        for _ in range(mu._OPERATIONS_QUEUE_BACKLOG_WARN_AT):
            await mu.queue_operation(_noop)

        assert len(_backlog_warnings(recorded_warnings)) == 2, (
            "a second backlog episode after a drain must warn again, "
            f"got {len(_backlog_warnings(recorded_warnings))}"
        )
    finally:
        await _drain_queue()
        mu._operations_backlog_warned = False
