"""
In-memory event broker for async pub/sub communication.

Subscriptions are by exact channel name (``subscribe``) or glob pattern
(``psubscribe``); messages are passed by reference between subscribers in
the same process.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

_log = logging.getLogger("unify")


@dataclass
class _Message:
    """Internal message structure for pub/sub."""

    type: str  # "message", "pmessage", "subscribe", "psubscribe"
    channel: str
    pattern: str | None  # Only set for pmessage
    data: str


@dataclass
class _Subscription:
    """Tracks a single subscription (channel or pattern)."""

    is_pattern: bool
    value: str  # channel name or pattern
    queue: asyncio.Queue[_Message]
    loop: asyncio.AbstractEventLoop


class InMemoryPubSub:
    """
    In-memory pub/sub subscription context.

    Usage:
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:actor:*")
            msg = await pubsub.get_message(timeout=1.0, ignore_subscribe_messages=True)
    """

    def __init__(self, broker: "InMemoryEventBroker"):
        self._broker = broker
        self._subscriptions: list[_Subscription] = []
        self._message_queue: asyncio.Queue[_Message] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._closed = False

    async def subscribe(self, *channels: str) -> None:
        """Subscribe to one or more channels."""
        for channel in channels:
            sub = _Subscription(
                is_pattern=False,
                value=channel,
                queue=self._message_queue,
                loop=self._loop,
            )
            self._subscriptions.append(sub)
            self._broker._add_subscription(sub)

            # Queue a subscribe confirmation message
            await self._message_queue.put(
                _Message(
                    type="subscribe",
                    channel=channel,
                    pattern=None,
                    data=str(len(self._subscriptions)),
                ),
            )

    async def psubscribe(self, *patterns: str) -> None:
        """Subscribe to one or more channel patterns (glob-style)."""
        for pattern in patterns:
            sub = _Subscription(
                is_pattern=True,
                value=pattern,
                queue=self._message_queue,
                loop=self._loop,
            )
            self._subscriptions.append(sub)
            self._broker._add_subscription(sub)

            # Queue a psubscribe confirmation message
            await self._message_queue.put(
                _Message(
                    type="psubscribe",
                    channel=pattern,
                    pattern=None,
                    data=str(len(self._subscriptions)),
                ),
            )

    async def get_message(
        self,
        *,
        timeout: float | None = None,
        ignore_subscribe_messages: bool = False,
    ) -> dict[str, Any] | None:
        """
        Get the next message from subscribed channels.

        Args:
            timeout: Maximum time to wait (seconds). None = no timeout.
            ignore_subscribe_messages: If True, skip subscribe/psubscribe confirmations.

        Returns:
            Message dict with keys: type, channel, pattern (if pmessage), data.
            Returns None on timeout.
        """
        if self._closed:
            return None

        while True:
            try:
                if timeout is not None:
                    msg = await asyncio.wait_for(
                        self._message_queue.get(),
                        timeout=timeout,
                    )
                else:
                    msg = await self._message_queue.get()

                # Skip subscription confirmations if requested
                if ignore_subscribe_messages and msg.type in (
                    "subscribe",
                    "psubscribe",
                ):
                    continue

                # Return message as dict
                result = {
                    "type": msg.type,
                    "channel": msg.channel,
                    "data": msg.data,
                }
                if msg.pattern is not None:
                    result["pattern"] = msg.pattern
                return result

            except asyncio.TimeoutError:
                return None
            except asyncio.CancelledError:
                raise

    async def aclose(self) -> None:
        """Close the pubsub connection and clean up subscriptions."""
        self._closed = True
        for sub in self._subscriptions:
            self._broker._remove_subscription(sub)
        self._subscriptions.clear()


class InMemoryEventBroker:
    """
    In-memory event broker for async publish/subscribe.

    Messages are passed by reference (no serialization) for efficiency.

    Usage:
        broker = InMemoryEventBroker()

        # Publishing
        await broker.publish("app:comms:startup", event.to_json())

        # Subscribing
        async with broker.pubsub() as pubsub:
            await pubsub.psubscribe("app:actor:*")
            while True:
                msg = await pubsub.get_message(timeout=1.0)
                if msg:
                    process(msg)
    """

    def __init__(self):
        self._subscriptions: list[_Subscription] = []
        # Subscriptions may be created/removed from different event loops/threads.
        # Use a thread-safe lock for coordination.
        self._subs_lock = threading.RLock()
        self._closed = False

    def _add_subscription(self, sub: _Subscription) -> None:
        """Register a subscription (called by InMemoryPubSub)."""
        with self._subs_lock:
            self._subscriptions.append(sub)

    def _remove_subscription(self, sub: _Subscription) -> None:
        """Unregister a subscription (called by InMemoryPubSub)."""
        with self._subs_lock:
            if sub in self._subscriptions:
                self._subscriptions.remove(sub)

    async def publish(self, channel: str, message: str) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: The channel name (e.g., "app:comms:startup")
            message: The message payload (typically JSON string)

        Returns:
            Number of subscribers that received the message.
        """
        if self._closed:
            return 0

        receivers = 0
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        # Copy to avoid mutation during iteration (and avoid holding the lock
        # while delivering to subscriber queues).
        with self._subs_lock:
            subs_snapshot = list(self._subscriptions)

        for sub in subs_snapshot:
            matched = False
            msg_type = "message"
            pattern = None

            if sub.is_pattern:
                # Pattern matching (glob-style)
                if fnmatch.fnmatch(channel, sub.value):
                    matched = True
                    msg_type = "pmessage"
                    pattern = sub.value
            else:
                # Exact channel match
                if sub.value == channel:
                    matched = True

            if matched:
                try:
                    msg = _Message(
                        type=msg_type,
                        channel=channel,
                        pattern=pattern,
                        data=message,
                    )
                    # Ensure delivery is safe across threads/event loops:
                    # - If publisher is on the same loop, push directly.
                    # - Otherwise, schedule the put on the subscriber loop.
                    if current_loop is not None and sub.loop is current_loop:
                        sub.queue.put_nowait(msg)
                    else:
                        sub.loop.call_soon_threadsafe(sub.queue.put_nowait, msg)
                    receivers += 1
                except Exception as exc:
                    _log.warning(
                        "EventBroker: failed to deliver message on channel %r "
                        "to subscriber (pattern=%r): %s",
                        channel,
                        sub.value,
                        exc,
                    )

        if receivers == 0 and subs_snapshot:
            _log.debug(
                "EventBroker: published to %r but 0 of %d subscriber(s) matched",
                channel,
                len(subs_snapshot),
            )

        return receivers

    @asynccontextmanager
    async def pubsub(self) -> AsyncIterator[InMemoryPubSub]:
        """
        Create a pub/sub context for subscribing to channels.

        Usage:
            async with broker.pubsub() as pubsub:
                await pubsub.psubscribe("app:*")
                msg = await pubsub.get_message(timeout=1.0)
        """
        ps = InMemoryPubSub(self)
        try:
            yield ps
        finally:
            await ps.aclose()

    async def aclose(self) -> None:
        """Close the broker and all subscriptions."""
        self._closed = True
        with self._subs_lock:
            self._subscriptions.clear()


# Global singleton instance
_broker: InMemoryEventBroker | None = None


def get_in_memory_event_broker() -> InMemoryEventBroker:
    """Get or create the global in-memory event broker singleton."""
    global _broker
    if _broker is None:
        _broker = InMemoryEventBroker()
    return _broker


def create_in_memory_event_broker() -> InMemoryEventBroker:
    """Create a broker that is not the process singleton (for isolated tests)."""
    return InMemoryEventBroker()


def reset_in_memory_event_broker() -> None:
    """Reset the global singleton (useful for testing)."""
    global _broker
    if _broker is not None:
        # Don't await aclose since this might be called from sync context
        _broker._closed = True
        with _broker._subs_lock:
            _broker._subscriptions.clear()
    _broker = None
