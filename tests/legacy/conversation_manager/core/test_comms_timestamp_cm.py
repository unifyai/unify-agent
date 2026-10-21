"""A legacy chat event's timestamp survives conversion to its EventBus event
(moved from tests/event_bus/test_comms_timestamp.py).
"""

import datetime as dt

from unify.legacy.conversation_manager.events import UnifyMessageReceived


def test_unify_message_to_bus_event_keeps_timestamp():
    """UnifyMessageReceived carries its timestamp onto the bus event and payload."""
    original_ts = dt.datetime(2025, 6, 15, 14, 30, 0, tzinfo=dt.UTC)
    original = UnifyMessageReceived(
        timestamp=original_ts,
        content="Message from console",
    )

    bus_event = original.to_bus_event()
    assert bus_event.type == "Comms"
    assert bus_event.payload_cls == "UnifyMessageReceived"
    assert bus_event.timestamp == original_ts
    assert bus_event.payload["timestamp"] == original_ts
