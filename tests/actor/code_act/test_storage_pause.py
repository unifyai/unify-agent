"""A pause that lands as the task loop finishes still holds the handle.

``_StorageCheckHandle`` forwards steering to whichever loop is active when the
call arrives. A pause that reaches the task loop in the moment it returns its
answer pauses a loop that is already over; the storage review then starts as a
fresh loop. Without carrying the request over, the caller has asked for a
hold, been told it happened, and watches the handle keep working.

Symbolic: the inner loop and the storage loop are stand-ins, and the race is
made deterministic by having the inner loop finish *because* of the pause.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.actor.code_act_actor import _StorageCheckHandle


def _inner_finishing_on(steer: str):
    """An inner loop that returns its answer as *steer* reaches it."""
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    inner = MagicMock()

    async def _await_result():
        return await result_future

    async def _finish(**kwargs):
        if not result_future.done():
            result_future.set_result("task answer")

    inner.result = _await_result
    inner.next_notification = AsyncMock(
        side_effect=lambda: asyncio.Event().wait(),
    )
    for method in ("pause", "resume", "stop"):
        setattr(
            inner,
            method,
            AsyncMock(side_effect=_finish if method == steer else None),
        )
    inner._client = MagicMock(messages=[])
    inner._task = MagicMock(
        get_ask_tools=MagicMock(return_value={}),
        get_completed_tool_metadata=MagicMock(return_value={}),
    )
    return inner


def _actor():
    actor = MagicMock()
    actor.function_manager = None
    actor.guidance_manager = None
    return actor


def _storage_loop():
    finished: asyncio.Future[str] = asyncio.get_event_loop().create_future()
    storage = MagicMock()

    async def _await_result():
        return await finished

    storage.result = _await_result
    storage.pause = AsyncMock()
    storage.resume = AsyncMock(
        side_effect=lambda **_: finished.done() or finished.set_result("reviewed"),
    )
    storage.stop = AsyncMock(
        side_effect=lambda **_: finished.done() or finished.set_result("stopped"),
    )
    return storage


async def _until(predicate, *, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.01)


def _patched(storage):
    return (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
            return_value=storage,
        ),
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_pause_racing_task_completion_holds_the_review():
    storage = _storage_loop()
    start, publish = _patched(storage)
    with start, publish:
        handle = _StorageCheckHandle(inner=_inner_finishing_on("pause"), actor=_actor())

        await handle.pause()
        await _until(lambda: handle._storage_handle is storage)

        storage.pause.assert_awaited_once()
        assert not handle.done()

        await handle.resume()
        storage.resume.assert_awaited_once()
        await _until(handle.done)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_resume_cancels_a_pause_still_waiting_for_the_review():
    storage = _storage_loop()
    start, publish = _patched(storage)
    with start, publish:
        inner = _inner_finishing_on("resume")
        handle = _StorageCheckHandle(inner=inner, actor=_actor())

        await handle.pause()
        await handle.resume()
        await _until(lambda: handle._storage_handle is storage)

        storage.pause.assert_not_awaited()
        await storage.stop()
        await _until(handle.done)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_stop_supersedes_a_pending_pause():
    """A stopped session still gets its review, and it must not start held."""
    storage = _storage_loop()
    start, publish = _patched(storage)
    with start, publish:
        inner = _inner_finishing_on("stop")
        handle = _StorageCheckHandle(inner=inner, actor=_actor())

        await handle.pause()
        await handle.stop(reason="never mind")
        await _until(lambda: handle._storage_handle is storage)

        storage.pause.assert_not_awaited()
        await storage.resume()
        await _until(handle.done)
