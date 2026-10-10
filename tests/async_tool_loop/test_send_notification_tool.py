"""Tests for the first-class send_notification tool.

Verifies that the loop-owned send_notification tool:
  1. Puts notifications onto _notification_up_q (surfaced via handle.next_notification)
  2. Fires the on_notify callback for event emission
  3. Coexists with in-code notify() — both paths produce notifications on the same handle
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.helpers import _handle_project
from tests.async_helpers import first_assistant_tool_call
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (notifications)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]


def make_llm(system_message=None, **llm_kwargs):
    return new_llm_client(**llm_kwargs, system_message=system_message)


# ──────────────────────────────────────────────────────────────────────────
# 1.  Basic: LLM calls send_notification → notification on handle
# ──────────────────────────────────────────────────────────────────────────


async def do_work(task: str) -> str:
    """Stub tool that does some work."""
    return f"Completed: {task}"


@pytest.mark.asyncio
@_handle_project
async def test_send_notification_surfaces_on_handle(llm_config) -> None:
    """send_notification tool call produces a notification on handle.next_notification()."""

    client = make_llm(
        "You have two tools: send_notification and do_work.\n"
        "1) First call send_notification with the message 'Starting task...'.\n"
        "2) Then call do_work with the task 'process data'.\n"
        "3) Finish with a short assistant message containing 'done'.",
        **llm_config,
    )

    handle = start_async_tool_loop(
        client,
        message="Please process data and notify me of progress.",
        tools={
            "do_work": do_work,
        },
        time_awareness=False,
        on_notify=lambda msg: None,
    )

    try:
        event = await asyncio.wait_for(handle.next_notification(), timeout=120)
        assert event["type"] == "notification"
        assert event["tool_name"] == "send_notification"
        assert "starting" in event.get("message", "").lower()

        result = await asyncio.wait_for(handle.result(), timeout=120)
        assert result is not None
    finally:
        try:
            await handle.stop("test cleanup")
        except Exception:
            pass

    _, call = first_assistant_tool_call(client.messages, "send_notification")
    args = json.loads(call["function"]["arguments"])
    assert "starting" in args["message"].lower()


# ──────────────────────────────────────────────────────────────────────────
# 2.  Event emission: on_notify callback fires
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_send_notification_on_notify_callback(llm_config) -> None:
    """The on_notify callback is invoked with the message text."""

    captured: list[str] = []

    async def _capture(msg: str):
        captured.append(msg)

    client = make_llm(
        "Call send_notification with message 'Checkpoint reached' then respond with 'ok'.",
        **llm_config,
    )

    handle = start_async_tool_loop(
        client,
        message="Notify me.",
        tools={"do_work": do_work},
        time_awareness=False,
        on_notify=_capture,
    )

    try:
        await asyncio.wait_for(handle.result(), timeout=120)
    finally:
        try:
            await handle.stop("test cleanup")
        except Exception:
            pass

    assert len(captured) >= 1
    assert any("checkpoint" in m.lower() for m in captured)


# ──────────────────────────────────────────────────────────────────────────
# 3.  Coexistence: send_notification + in-code notify() on same handle
# ──────────────────────────────────────────────────────────────────────────


async def tool_with_inline_notify(
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    """Tool that uses the in-code notification path."""
    if _notification_up_q is not None:
        await _notification_up_q.put(
            {"type": "notification", "message": "inline progress"},
        )
    return "inline done"


@pytest.mark.asyncio
@_handle_project
async def test_send_notification_coexists_with_inline_notify(llm_config) -> None:
    """Both send_notification (tool call) and inline notify (via _notification_up_q)
    produce notifications on the same handle."""

    client = make_llm(
        "You have send_notification and tool_with_inline_notify.\n"
        "1) Call send_notification with message 'tool notification'.\n"
        "2) Call tool_with_inline_notify.\n"
        "3) Respond with 'complete'.",
        **llm_config,
    )

    handle = start_async_tool_loop(
        client,
        message="Run both notification paths.",
        tools={
            "tool_with_inline_notify": tool_with_inline_notify,
        },
        time_awareness=False,
        on_notify=lambda msg: None,
    )

    notifications: list[dict] = []
    try:
        for _ in range(10):
            try:
                evt = await asyncio.wait_for(handle.next_notification(), timeout=30)
                notifications.append(evt)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                break
            if len(notifications) >= 2:
                break

        await asyncio.wait_for(handle.result(), timeout=120)
    finally:
        try:
            await handle.stop("test cleanup")
        except Exception:
            pass

    tool_names = [n.get("tool_name") for n in notifications]
    assert (
        "send_notification" in tool_names
    ), f"Expected send_notification in tool_names, got {tool_names}"
    assert (
        "tool_with_inline_notify" in tool_names
    ), f"Expected tool_with_inline_notify in tool_names, got {tool_names}"
