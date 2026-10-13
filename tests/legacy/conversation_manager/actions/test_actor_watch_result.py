from __future__ import annotations

import pytest

from unify.legacy.conversation_manager.domains import managers_utils
from unify.legacy.conversation_manager.events import ActorResult, Event


class _FakeHandle:
    def __init__(self, *, result=None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc

    async def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeBroker:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, data: str) -> None:
        self.published.append((channel, data))


def _single_actor_result(broker: _FakeBroker) -> ActorResult:
    assert len(broker.published) == 1
    channel, payload = broker.published[0]
    assert channel == "app:actor:result"
    event = Event.from_json(payload)
    assert isinstance(event, ActorResult)
    return event


@pytest.mark.asyncio
async def test_actor_watch_result_publishes_success_event(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(managers_utils, "event_broker", broker)

    await managers_utils.actor_watch_result(
        11,
        _FakeHandle(result={"status": "ok"}),
        action_type="act",
    )

    event = _single_actor_result(broker)
    assert event.handle_id == 11
    assert event.success is True
    assert event.result == {"status": "ok"}
    assert event.error is None
    assert event.action_type == "act"


@pytest.mark.asyncio
async def test_actor_watch_result_marks_exception_as_failure(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(managers_utils, "event_broker", broker)

    await managers_utils.actor_watch_result(
        12,
        _FakeHandle(exc=RuntimeError("boom")),
        action_type="act",
    )

    event = _single_actor_result(broker)
    assert event.handle_id == 12
    assert event.success is False
    assert event.result is None
    assert event.error is not None
    assert "boom" in event.error
    assert event.action_type == "act"


@pytest.mark.asyncio
async def test_actor_watch_result_marks_tool_error_payload_as_failure(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(managers_utils, "event_broker", broker)

    await managers_utils.actor_watch_result(
        13,
        _FakeHandle(
            result={
                "error_kind": "permission_denied",
                "message": "Coordinator role required",
            },
        ),
        action_type="act",
    )

    event = _single_actor_result(broker)
    assert event.success is False
    assert event.result["error_kind"] == "permission_denied"
    assert event.error == "Coordinator role required"
    assert event.action_type == "act"


@pytest.mark.asyncio
async def test_actor_watch_result_marks_error_string_prefix_as_failure(monkeypatch):
    broker = _FakeBroker()
    monkeypatch.setattr(managers_utils, "event_broker", broker)

    await managers_utils.actor_watch_result(
        14,
        _FakeHandle(result="error: image exceeds size limit"),
        action_type="act",
    )

    event = _single_actor_result(broker)
    assert event.success is False
    assert event.result == "error: image exceeds size limit"
    assert event.error == "error: image exceeds size limit"
    assert event.action_type == "act"
