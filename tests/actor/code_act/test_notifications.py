import asyncio
import contextlib

import pytest

from unify.actor.code_act_actor import CodeActActor
from unify.events.active_work import ACTIVE_WORK
from tests.baked_defaults import as_shipped  # noqa: F401


# as_shipped: deleted in step 5 (notifications, replaced by the agent record)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@pytest.mark.timeout(120)
async def test_tool_loop_handle_next_notification():
    """
    AsyncToolLoopHandle.next_notification() should surface notifications
    emitted by tools running inside the loop.
    """
    from unify.common.async_tool_loop import start_async_tool_loop
    from unify.common.llm_client import new_llm_client

    # Create a simple tool that emits a notification
    async def notifying_tool(
        message: str,
        *,
        _notification_up_q: asyncio.Queue[dict] | None = None,
    ) -> str:
        """A tool that emits a notification when called."""
        if _notification_up_q is not None:
            await _notification_up_q.put(
                {"type": "test_notification", "message": message},
            )
        return f"Processed: {message}"

    client = new_llm_client()
    client.set_system_message(
        "You are a helpful assistant. When asked to call a tool, call it with the "
        "specified arguments. Do not ask for clarification - just call the tool.",
    )

    handle = start_async_tool_loop(
        client,
        "Call notifying_tool with message='hello'",
        {"notifying_tool": notifying_tool},
        timeout=60,
    )

    try:
        # The inner loop should eventually call the tool and emit a notification.
        # The tool loop wraps notifications with tool_name and call_id, but the original
        # payload (including 'type') is merged in via **event_payload.
        event = await asyncio.wait_for(handle.next_notification(), timeout=60)
        assert event.get("type") == "test_notification"  # From our tool's payload
        assert event.get("tool_name") == "notifying_tool"
        # The message from our tool should be in the event
        assert "hello" in str(event.get("message", ""))
        await handle.result()

    finally:
        if not handle.done():
            try:
                await handle.stop("cleanup")
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_tracks_and_clears_active_work():
    ACTIVE_WORK.clear()
    actor = CodeActActor()
    actor._active_work_heartbeat_interval_s = 0.01

    try:
        execute_code = actor.get_tools("act")["execute_code"]
        task = asyncio.create_task(
            execute_code(
                "run silent async work",
                "import asyncio\nawait asyncio.sleep(0.05)\n'done'",
                state_mode="stateless",
            ),
        )

        deadline = asyncio.get_event_loop().time() + 1.0
        while ACTIVE_WORK.snapshot().active_count == 0:
            if asyncio.get_event_loop().time() >= deadline:
                pytest.fail("active work was not registered")
            await asyncio.sleep(0.005)

        assert ACTIVE_WORK.snapshot().active_count == 1
        await task
        assert ACTIVE_WORK.snapshot().active_count == 0
    finally:
        ACTIVE_WORK.clear()
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_clears_active_work_after_exception_timeout_and_cancellation():
    ACTIVE_WORK.clear()

    actor = CodeActActor(timeout=0.02)
    actor._active_work_heartbeat_interval_s = 0.01
    execute_code = actor.get_tools("act")["execute_code"]

    try:
        error_result = await execute_code(
            "raise an exception",
            "raise RuntimeError('boom')",
            state_mode="stateless",
        )
        assert error_result.error is not None
        assert ACTIVE_WORK.snapshot().active_count == 0

        timeout_result = await execute_code(
            "timeout async work",
            "import asyncio\nawait asyncio.sleep(1)",
            state_mode="stateless",
        )
        assert "timed out" in str(timeout_result.error)
        assert ACTIVE_WORK.snapshot().active_count == 0

        task = asyncio.create_task(
            execute_code(
                "cancel async work",
                "import asyncio\nawait asyncio.sleep(1)",
                state_mode="stateless",
            ),
        )
        deadline = asyncio.get_event_loop().time() + 1.0
        while ACTIVE_WORK.snapshot().active_count == 0:
            if asyncio.get_event_loop().time() >= deadline:
                pytest.fail("active work was not registered before cancellation")
            await asyncio.sleep(0.005)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert ACTIVE_WORK.snapshot().active_count == 0
    finally:
        ACTIVE_WORK.clear()
        await actor.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_execute_code_fallback_progress():
    ACTIVE_WORK.clear()

    actor = CodeActActor()
    actor._active_work_heartbeat_interval_s = 0.01
    actor._active_work_fallback_initial_delay_s = 0.03
    actor._active_work_fallback_repeat_interval_s = 0.05
    execute_code = actor.get_tools("act")["execute_code"]

    try:
        notification_q: asyncio.Queue[dict] = asyncio.Queue()
        await execute_code(
            "run silent work long enough for fallback progress",
            "import asyncio\nawait asyncio.sleep(0.06)",
            state_mode="stateless",
            _notification_up_q=notification_q,
        )
        fallback = await asyncio.wait_for(notification_q.get(), timeout=1.0)
        assert fallback["source"] == "active_work"
        assert "Still working" in fallback["message"]
        assert notification_q.empty()
    finally:
        ACTIVE_WORK.clear()
        await actor.close()
