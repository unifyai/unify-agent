"""Tests for the StorageCheck-completion wake signal.

``_StorageCheckHandle._run_lifecycle`` Phase 2 enqueues a
``storage_review_complete`` notification right before the handle's
completion event flips (see ``code_act_actor.py``). Two things must hold:

* A watcher that only re-checks ``handle.done()`` between notifications can
  race that flip and drop the notification unless the consumer
  (``actor_watch_notifications``) explicitly drains the queue -- this is
  the "drain race" described in the tech plan.
* The failure path (the storage loop raising) must still emit the signal,
  with ``success=False`` and the error text, never skip it silently.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unify.actor.code_act_actor import _StorageCheckHandle
from unify.legacy.conversation_manager.domains import managers_utils
from unify.legacy.conversation_manager.events import ActorNotification


def _make_inner_handle(result_value: str = "task result") -> MagicMock:
    inner = MagicMock()

    async def _result():
        return result_value

    inner.result = _result
    inner.next_notification = AsyncMock(side_effect=lambda: asyncio.Event().wait())

    mock_client = MagicMock()
    mock_client.messages = [{"role": "user", "content": "split the bill"}]
    inner._client = mock_client

    mock_task = MagicMock()
    mock_task.get_ask_tools = MagicMock(return_value={})
    mock_task.get_completed_tool_metadata = MagicMock(return_value={})
    inner._task = mock_task
    return inner


async def _drive_to_done(handle: _StorageCheckHandle, timeout: float = 10) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not handle.done():
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("_StorageCheckHandle did not complete")
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_storage_review_complete_survives_drain_race_and_failure_path():
    """Drive a ``_StorageCheckHandle`` through both phases twice.

    Scenario 1 (success): by the time a watcher-style consumer starts
    observing, the handle already reports ``done() == True`` -- the worst
    case of the race described in the tech plan, since the emit code puts
    the notification on the queue before flipping completion. The consumer
    must still drain and publish it instead of exiting immediately.

    Scenario 2 (failure): the storage loop raises. The signal must still be
    enqueued, with ``success=False`` and the error text.
    """
    actor = MagicMock()
    actor.function_manager = MagicMock()
    actor.guidance_manager = MagicMock()

    # ── Scenario 1: success path observed via the real watcher ─────────
    success_storage_handle = MagicMock()

    async def _success_result():
        return "Stored a rule, a fact, and the split_dinner_bill skill."

    success_storage_handle.result = _success_result

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
            return_value=success_storage_handle,
        ),
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        handle = _StorageCheckHandle(inner=_make_inner_handle(), actor=actor)
        await _drive_to_done(handle)

    # The handle already looks fully done -- exactly the race window where a
    # naive `while not handle.done()` watcher would exit without ever
    # calling next_notification() again.
    assert handle.done() is True

    with patch.object(
        managers_utils,
        "event_broker",
        new=MagicMock(publish=AsyncMock()),
    ):
        mock_broker = managers_utils.event_broker
        await managers_utils.actor_watch_notifications(0, handle)

    notif_calls = [
        c
        for c in mock_broker.publish.call_args_list
        if c.args[0] == "app:actor:notification"
    ]
    assert (
        notif_calls
    ), "storage_review_complete notification was dropped by the drain race"
    published = ActorNotification.from_json(notif_calls[-1].args[1])
    assert published.kind == "storage_review_complete"
    assert "Stored a rule" in published.response

    # ── Scenario 2: failure path, inspected directly on the raw queue ──
    failing_storage_handle = MagicMock()

    async def _failing_result():
        raise RuntimeError("librarian crashed")

    failing_storage_handle.result = _failing_result

    with (
        patch(
            "unify.actor.code_act_actor._start_storage_check_loop",
            return_value=failing_storage_handle,
        ),
        patch(
            "unify.actor.code_act_actor.publish_manager_method_event",
            new_callable=AsyncMock,
        ),
    ):
        failing_handle = _StorageCheckHandle(inner=_make_inner_handle(), actor=actor)
        await _drive_to_done(failing_handle)

    notif = await asyncio.wait_for(failing_handle.next_notification(), timeout=5)
    assert notif["type"] == "storage_review_complete"
    assert notif["success"] is False
    assert "librarian crashed" in notif["message"]
