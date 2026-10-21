"""
tests/conversation_manager/core/test_event_logging.py
=====================================================

Tests that verify ConversationManager publishes events to the EventBus
for observability.

ConversationManager publishes Comms events (UnifyMessageReceived,
UnifyMessageSent) as it processes the chat.

These tests verify that:
1. Inbound events (UnifyMessageReceived) are logged to EventBus
2. Outbound events (UnifyMessageSent) are logged to EventBus
3. Event payloads contain the expected data
"""

from __future__ import annotations

import os
import asyncio

import pytest
import pytest_asyncio

from tests.helpers import _handle_project, capture_events
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)

# =============================================================================
# Helper Functions
# =============================================================================


async def wait_for_operations_queue(timeout: float = 5.0) -> None:
    """
    Wait for all queued operations (including publish_bus_events) to complete.

    The CM uses an async queue for operations like publishing to EventBus.
    This helper waits for that queue to be empty.
    """
    from unify.legacy.conversation_manager.domains import managers_utils

    # Yield to the event loop so the fire-and-forget create_task() in the
    # event handler can execute its Queue.put() (non-blocking on an unbounded
    # queue, so a single yield is sufficient).
    await asyncio.sleep(0)

    try:
        await asyncio.wait_for(
            managers_utils._operations_queue.join(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        pass  # Continue even if timeout - some events may still be captured


# =============================================================================
# Test Fixture: CM with EventBus publishing enabled
# =============================================================================


@pytest_asyncio.fixture
async def cm_with_eventbus():
    """
    Create a ConversationManager with EventBus publishing enabled.

    This is a function-scoped fixture for isolation.
    """
    from unify.actor.simulated import SimulatedActor
    from unify.legacy.conversation_manager.event_broker import reset_event_broker
    from unify.legacy.conversation_manager import start_async, stop_async
    from unify.legacy.conversation_manager.domains import managers_utils

    os.environ["UNIFY_ACTOR_IMPL"] = "simulated"
    os.environ["UNIFY_INCREMENTING_TIMESTAMPS"] = "true"
    os.environ["TEST"] = "true"

    reset_event_broker()

    cm = await start_async()

    # Initialize managers with SimulatedActor. steps=0: this fixture never
    # drives simulate_step()/trigger_completion(); a positive budget would
    # hang any caller that awaits actor.result() without stopping.
    actor = SimulatedActor(steps=0, log_mode="log", emit_notifications=False)
    await managers_utils.init_conv_manager(cm, actor=actor)

    # Start the operations listener that processes EventBus publishing
    asyncio.create_task(managers_utils.listen_to_operations(cm))

    yield cm

    # Cleanup
    await stop_async()
    reset_event_broker()


# =============================================================================
# Event Logging Tests
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_received_logged_to_eventbus(cm_with_eventbus):
    """
    Verify that inbound chat events are published to the EventBus.

    When a chat message is received and processed, a UnifyMessageReceived
    event should be logged with its content.
    """
    from unify.legacy.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus

    unique_content = "💬 Test event logging inbound message"

    unify_msg_event = UnifyMessageReceived(
        content=unique_content,
    )

    async with capture_events("Comms") as events:
        await EventHandler.handle_event(unify_msg_event, cm)

        # Wait for queued operations (publish_bus_events) to complete
        await wait_for_operations_queue()

    unify_received_events = [
        e
        for e in events
        if e.payload_cls == "UnifyMessageReceived"
        and e.payload.get("content") == unique_content
    ]

    assert unify_received_events, (
        f"No UnifyMessageReceived event logged to EventBus. "
        f"Found events: {[e.payload_cls for e in events]}"
    )

    # Verify payload content
    received_evt = unify_received_events[0]
    assert (
        received_evt.payload.get("attachments") == []
    ), "UnifyMessageReceived event should carry its attachment list"


@pytest.mark.asyncio
@_handle_project
async def test_unify_message_sent_logged_to_eventbus(cm_with_eventbus):
    """
    Verify that outbound chat events are published to the EventBus.

    When the assistant's reply is processed, a UnifyMessageSent event should
    be logged with its content.
    """
    from unify.legacy.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus

    unique_content = "💬 Test event logging outbound message"

    sent_event = UnifyMessageSent(
        content=unique_content,
    )

    async with capture_events("Comms") as events:
        await EventHandler.handle_event(sent_event, cm)
        await wait_for_operations_queue()

    unify_sent_events = [
        e
        for e in events
        if e.payload_cls == "UnifyMessageSent"
        and e.payload.get("content") == unique_content
    ]

    assert unify_sent_events, (
        f"No UnifyMessageSent event logged to EventBus. "
        f"Found events: {[e.payload_cls for e in events]}"
    )
    assert (
        unify_sent_events[0].payload.get("attachments") == []
    ), "UnifyMessageSent event should carry its attachment list"


@pytest.mark.asyncio
@_handle_project
async def test_event_bus_event_has_correct_type(cm_with_eventbus):
    """
    Verify that CM events published to EventBus have type="Comms".

    This is important for filtering and observability - all CM events
    should be identifiable as Comms events.
    """
    from unify.legacy.conversation_manager.domains.event_handlers import EventHandler

    cm = cm_with_eventbus

    unify_msg_event = UnifyMessageReceived(
        content="Test event type verification",
    )

    async with capture_events("Comms") as events:
        await EventHandler.handle_event(unify_msg_event, cm)
        await wait_for_operations_queue()

    # All captured events should have type="Comms"
    for event in events:
        assert (
            event.type == "Comms"
        ), f"Expected event type 'Comms', got '{event.type}' for {event.payload_cls}"
