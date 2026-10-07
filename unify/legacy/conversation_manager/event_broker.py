"""The process-wide event broker singleton for ConversationManager."""

from __future__ import annotations

from unify.legacy.conversation_manager.in_memory_event_broker import (
    InMemoryEventBroker,
    get_in_memory_event_broker,
    reset_in_memory_event_broker,
)

_broker: InMemoryEventBroker | None = None


def get_event_broker() -> InMemoryEventBroker:
    """Return the shared ``InMemoryEventBroker`` instance, creating it on first use."""
    global _broker

    if _broker is None:
        _broker = get_in_memory_event_broker()

    return _broker


def reset_event_broker() -> None:
    """Drop the singleton so the next ``get_event_broker`` call builds a fresh one."""
    global _broker

    if _broker is not None:
        reset_in_memory_event_broker()

    _broker = None
