"""
tests/conversation_manager/core/test_initialization_race.py
===========================================================

Integration tests for manager initialization race conditions.

These tests verify the system correctly handles events that arrive BEFORE or
DURING manager initialization:
- Chat messages can arrive immediately after the session boots
- The chat table is bound part-way through initialization
- Multiple events can arrive in rapid succession

Events arriving before initialization is complete must not fail silently.
These tests verify:
- queue_operation correctly defers work until after initialization
- Chat messages that arrive before the chat table is bound are kept
- Multiple rapid events don't cause race conditions
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from unify.legacy.conversation_manager.in_memory_event_broker import (
    create_in_memory_event_broker,
    reset_in_memory_event_broker,
)


@pytest_asyncio.fixture
async def event_broker():
    """Real in-memory event broker."""
    reset_in_memory_event_broker()
    broker = create_in_memory_event_broker()
    yield broker
    await broker.aclose()
    reset_in_memory_event_broker()


class TestQueueOperationDuringInit:
    """
    Tests for queue_operation pattern that defers work until initialization.

    The queue_operation() function in managers_utils.py queues async operations
    that require managers to be initialized. listen_to_operations() processes
    them after cm.initialized becomes True.
    """

    @pytest.mark.asyncio
    async def test_operation_queued_before_init_executes_after(self, event_broker):
        """
        Test that operations queued before init are executed after init completes.

        This is the core guarantee of the queue_operation pattern.
        """
        from unify.legacy.conversation_manager.domains import managers_utils

        # Reset the queue
        while not managers_utils._operations_queue.empty():
            try:
                managers_utils._operations_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Track execution order
        execution_log = []

        async def tracked_operation(marker: str):
            execution_log.append(f"executed:{marker}")

        # Create a mock CM that is NOT initialized
        mock_cm = MagicMock()
        mock_cm.initialized = False

        # Queue an operation BEFORE initialization
        await managers_utils.queue_operation(tracked_operation, "op1")

        # Verify operation hasn't executed yet
        assert (
            "executed:op1" not in execution_log
        ), "Operation executed before initialization!"

        # Now simulate initialization completing
        mock_cm.initialized = True

        # Start the operations listener (runs until queue is empty)
        async def run_listener_briefly():
            # Wait for init (already done)
            # Then process one item
            try:
                async_func, args, kwargs = await asyncio.wait_for(
                    managers_utils._operations_queue.get(),
                    timeout=1.0,
                )
                await async_func(*args, **kwargs)
            except asyncio.TimeoutError:
                pass

        # Patch wait_for_initialization to return immediately (cm is initialized)
        with patch.object(
            managers_utils,
            "wait_for_initialization",
            new_callable=AsyncMock,
        ):
            await run_listener_briefly()

        # Now the operation should have executed
        assert "executed:op1" in execution_log, (
            "Operation was not executed after initialization. "
            "queue_operation pattern is broken."
        )

    @pytest.mark.asyncio
    async def test_multiple_operations_queued_execute_in_order(self, event_broker):
        """
        Test that multiple queued operations execute in FIFO order.

        Order matters for things like EventBus persistence.
        """
        from unify.legacy.conversation_manager.domains import managers_utils

        # Reset the queue
        while not managers_utils._operations_queue.empty():
            try:
                managers_utils._operations_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        execution_order = []

        async def tracked_op(marker: str):
            execution_order.append(marker)

        # Queue multiple operations
        await managers_utils.queue_operation(tracked_op, "first")
        await managers_utils.queue_operation(tracked_op, "second")
        await managers_utils.queue_operation(tracked_op, "third")

        # Process all queued operations
        with patch.object(
            managers_utils,
            "wait_for_initialization",
            new_callable=AsyncMock,
        ):
            for _ in range(3):
                try:
                    async_func, args, kwargs = await asyncio.wait_for(
                        managers_utils._operations_queue.get(),
                        timeout=1.0,
                    )
                    await async_func(*args, **kwargs)
                except asyncio.TimeoutError:
                    break

        assert execution_order == ["first", "second", "third"], (
            f"Operations executed out of order: {execution_order}. "
            "This could cause message ordering bugs in production."
        )


class TestEventsDuringInitialization:
    """
    Tests for handling events during the initialization window.

    Events can arrive at any time:
    - Before initialization starts
    - During initialization (the chat table not yet bound)
    - After initialization completes

    The system must handle all these cases gracefully.
    """

    @pytest.mark.asyncio
    async def test_message_received_before_init_is_kept(self, event_broker):
        """
        A chat message that arrives before the chat table is bound is kept
        in memory and wakes the brain.

        1. The session boots
        2. A chat message arrives immediately
        3. The chat table is not bound yet
        """
        from unify.legacy.conversation_manager.domains.chat_history import ChatHistory
        from unify.legacy.conversation_manager.domains.event_handlers import (
            EventHandler,
        )
        from unify.legacy.conversation_manager.events import UnifyMessageReceived

        history = ChatHistory()
        mock_cm = MagicMock()
        mock_cm.chat_history = history
        mock_cm.active_ask_handle = None
        mock_cm._session_logger = MagicMock()
        mock_cm.notifications_bar = MagicMock()
        mock_cm.request_llm_run = AsyncMock()

        message = UnifyMessageReceived(content="Hello!")

        await EventHandler.handle_event(message, mock_cm)

        assert not history.is_bound
        assert history.recent()[0].content == "Hello!"
        assert history.recent()[0].role == "user"
        mock_cm.request_llm_run.assert_awaited_once()


class TestRapidEventsRaceCondition:
    """
    Tests for rapid event handling during initialization.

    Several chat messages can be published in rapid succession while the
    session is still booting. This tests that pattern.
    """

    @pytest.mark.asyncio
    async def test_multiple_messages_during_init_window(self, event_broker):
        """
        Test that multiple chat messages during the init window are all handled.

        A flurry of messages shouldn't cause any to be dropped.
        """
        from unify.legacy.conversation_manager.events import UnifyMessageReceived, Event

        messages_received = []

        async with event_broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:comms:*")

            # Send 5 messages rapidly
            tasks = []
            for i in range(5):
                message = UnifyMessageReceived(content=f"Message {i}")
                tasks.append(
                    event_broker.publish(
                        UnifyMessageReceived.topic,
                        message.to_json(),
                    ),
                )

            await asyncio.gather(*tasks)

            # Collect all messages (get_message already has timeout, no extra sleep needed)
            for _ in range(20):
                msg = await pubsub.get_message(
                    timeout=0.2,
                    ignore_subscribe_messages=True,
                )
                if msg:
                    try:
                        event = Event.from_json(msg["data"])
                        if isinstance(event, UnifyMessageReceived):
                            messages_received.append(event.content)
                    except Exception:
                        pass

        # All 5 messages should be received
        assert len(messages_received) >= 5, (
            f"Only received {len(messages_received)}/5 messages. "
            "Messages were lost during rapid-fire scenario."
        )


class TestInitializationTimeout:
    """
    Tests for initialization timeout handling.

    If initialization takes too long, the system should handle it gracefully
    rather than hanging forever.
    """

    @pytest.mark.asyncio
    async def test_wait_for_initialization_polls_until_ready(self):
        """
        Test that wait_for_initialization returns when cm.initialized becomes True.
        """
        from unify.legacy.conversation_manager.domains.managers_utils import (
            wait_for_initialization,
        )

        mock_cm = MagicMock()
        mock_cm.initialized = False

        async def set_initialized_after_delay():
            await asyncio.sleep(0.2)
            mock_cm.initialized = True

        asyncio.create_task(set_initialized_after_delay())

        try:
            await asyncio.wait_for(wait_for_initialization(mock_cm), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "wait_for_initialization never returned even though init completed",
            )
