from __future__ import annotations

import asyncio
from typing import Optional
import json

import pytest
import unillm
from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_assistant_call_prefix,
    first_user_message,
    first_assistant_tool_call,
    last_plain_assistant_message,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (notifications of steerable handles)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]

# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────


def make_llm(
    system_message: Optional[str] = None,
    **llm_kwargs,
) -> unillm.AsyncUnify:
    return new_llm_client(**llm_kwargs, system_message=system_message)


# ──────────────────────────────────────────────────────────────────────────
# 1.  DUMMY TOOLS – send_email emits notifications
# ──────────────────────────────────────────────────────────────────────────
async def send_email(
    address: str,
    description: str,
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    """Send an email, emitting notifications along the way."""
    if _notification_up_q is None:
        raise RuntimeError("notification queue missing")

    # Emit notifications; loop will surface them and allow the assistant to react
    await _notification_up_q.put({"message": "Composing email…"})
    await _notification_up_q.put({"message": "Sending email…"})
    return "Email sent!"


async def send_text(
    number: str,
    description: str,
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    """Send a text message (unused in this test)."""
    # Silently do nothing; this tool is present only to mirror the clarifying test shape
    return "Text queued!"


# Helper tool the assistant can choose to call to actively surface progress upward
async def notify_parent(
    message: str,
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    if _notification_up_q is None:
        raise RuntimeError("notification queue missing")
    await _notification_up_q.put({"message": message})
    return "ack"


# ──────────────────────────────────────────────────────────────────────────
# 2.  The test (two tiers)
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_notification_bubbles_up_two_tiers(llm_config) -> None:
    """
    Verifies that notifications emitted by a running tool are surfaced upstream
    and allow the assistant to react; the tool then completes successfully.
    """

    # Deterministic ordering gates – ensure notify_parent is called while send_email is running
    notify_called_gate = asyncio.Event()

    async def send_email(
        address: str,
        description: str,
        *,
        _notification_up_q: asyncio.Queue | None = None,
    ) -> str:
        """Send an email, emitting notifications along the way (deterministic flow)."""
        if _notification_up_q is None:
            raise RuntimeError("notification queue missing")

        # Emit an early progress update, then block until notify_parent has been requested
        await _notification_up_q.put({"message": "Composing email…"})
        await notify_called_gate.wait()
        await asyncio.sleep(0)  # yield to allow notify_parent to run
        await _notification_up_q.put({"message": "Sending email…"})
        return "Email sent!"

    async def send_text(
        number: str,
        description: str,
        *,
        _notification_up_q: asyncio.Queue | None = None,
    ) -> str:
        return "Text queued!"

    # Helper tool the assistant must call to surface progress upward
    async def notify_parent(
        message: str,
        *,
        _notification_up_q: asyncio.Queue | None = None,
    ) -> str:
        if _notification_up_q is None:
            raise RuntimeError("notification queue missing")
        await _notification_up_q.put({"message": message})
        return "ack"

    outer_client = make_llm(
        "You are coordinating internal tools that may emit progress notifications while running.\n"
        "Follow these rules exactly for this session:\n"
        "1) As soon as a running tool emits a progress update, immediately call notify_parent(message=...) with the exact text (e.g., 'Composing email…').\n"
        "2) Do not produce a normal assistant message while work is pending; use tools only.\n"
        "3) Avoid starting unrelated tools while the original call is in progress.\n"
        "4) Once the email has been sent, produce a single, concise assistant message that includes the word 'sent' (e.g., 'Email sent.').",
        **llm_config,
    )

    outer_tools = {
        "send_email": send_email,
        "send_text": send_text,
        "notify_parent": notify_parent,
    }

    outer_handle = start_async_tool_loop(  # type: ignore[attr-defined]
        outer_client,
        message="Please email jonathan.smith@example.com and politely tell him I (Dan) will be arriving at the BBQ around 5pm.",
        tools=outer_tools,
        time_awareness=False,
    )

    try:
        # Deterministic ordering:
        # 1) Wait until assistant schedules send_email
        await _wait_for_tool_request(outer_client, "send_email", timeout=120.0)
        # 2) Wait until assistant schedules notify_parent in response to the progress update
        await _wait_for_assistant_call_prefix(
            outer_client,
            "notify_parent",
            timeout=120.0,
        )
        # 3) Unblock send_email so it can continue and finish
        notify_called_gate.set()

        # Now assert that a bubbled notification from notify_parent is received
        notification_event = None
        for _ in range(5):
            evt = await asyncio.wait_for(outer_handle.next_notification(), timeout=120)
            if evt.get("tool_name") == "notify_parent":
                notification_event = evt
                break
        assert (
            notification_event is not None
        ), "notify_parent was not called by the assistant"
        assert notification_event["type"] == "notification"
        assert notification_event["tool_name"] == "notify_parent"
        if "message" in notification_event and isinstance(
            notification_event["message"],
            str,
        ):
            assert any(
                k in notification_event["message"].lower()
                for k in ["compos", "sending", "send", "sent", "email", "success"]
            )

        await asyncio.wait_for(outer_handle.result(), timeout=300)
    finally:
        # Ensure loop teardown even if assertions/timeouts fail
        try:
            await outer_handle.stop("test cleanup")
        except Exception:
            pass

    # ─────────────────────────
    # Assertions (chat transcript shape)
    # ─────────────────────────
    msgs = outer_client.messages

    # 1️⃣ original user request – robust to clients that don't persist system messages
    first_user_msg = first_user_message(msgs)
    assert first_user_msg["content"] == (
        "Please email jonathan.smith@example.com and politely tell him I (Dan) "
        "will be arriving at the BBQ around 5pm."
    )

    # 2️⃣ assistant chooses `send_email` --------------------------------------
    # Find the first assistant tool call that requests `send_email` (avoid fixed indices)
    _m1, call1 = first_assistant_tool_call(msgs, "send_email")
    args1 = json.loads(call1["function"]["arguments"])
    assert args1["address"] == "jonathan.smith@example.com"

    # 3️⃣ We no longer require an interim tool message from the base tool; the assistant
    # decides how to surface updates (via notify_parent).

    # 4️⃣ at least one tool message contains the real result ------------------
    tool_contents = [
        (m.get("content") or "").lower() for m in msgs if m.get("role") == "tool"
    ]
    assert any(
        ("email sent" in c) or ("email" in c and "sent" in c) for c in tool_contents
    )

    # 5️⃣ assistant wraps up ---------------------------------------------------
    # Find the last plain assistant message (no tool_calls) and validate it closes correctly
    closing = last_plain_assistant_message(msgs)
    assert closing is not None, "Expected a final assistant message"


# ---------------------------------------------------------------------------
# inner tool  ➟  emits notifications (fire-and-forget) and completes
# ---------------------------------------------------------------------------
async def inner_tool(
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:
    if _notification_up_q is None:
        raise RuntimeError("notification queue missing")

    await _notification_up_q.put({"message": "Inner loop: preparing widget"})
    await _notification_up_q.put({"message": "Inner loop: halfway"})
    return "✅ inner finished"


# ---------------------------------------------------------------------------
# outer tool  ➟  immediately spawns an async-tool loop and RETURNS its handle
#               notification events from the inner loop bubble via the parent's queue
# ---------------------------------------------------------------------------
async def delegating_tool(
    *,
    _notification_up_q: asyncio.Queue | None = None,
) -> str:  # return type misleading on purpose
    inner_llm = make_llm(
        "Surface any internal notifications as they occur; continue to completion.\n"
        "Do not fabricate status; forward only the actual progress messages you receive.",
    )

    # Bridge notifications by closing over the parent notification queue
    async def inner_tool_bridge() -> str:
        return await inner_tool(_notification_up_q=_notification_up_q)

    handle = start_async_tool_loop(  # <-- returns AsyncToolLoopHandle
        inner_llm,
        message="Run inner_tool please.",
        tools={
            "inner_tool": inner_tool_bridge,
        },
    )
    return handle  # outer tool finishes instantly


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_notification_bubbles_through_returned_handle(llm_config) -> None:
    """Notification raised inside the returned handle must still reach the user."""

    outer_llm = make_llm(
        "You are the TOP-LEVEL coordinator. When any delegated or nested tool emits progress notifications, "
        "acknowledge them briefly (non-blocking) and continue until the delegated work completes.\n"
        "Do not wait for any acknowledgement; do not start unrelated tools while work is pending unless necessary.\n"
        "Do not invent status; only reflect actual notifications.\n"
        "When the inner work completes, end your final assistant message by including the word 'finished'. "
        "Keep responses concise.",
        **llm_config,
    )

    handle = start_async_tool_loop(
        outer_llm,
        message="Run delegating_tool please.",
        tools={
            "delegating_tool": delegating_tool,
        },
    )
    try:
        # ── satisfy: we should receive a bubbled notification event from the INNER loop ──
        event = await asyncio.wait_for(handle.next_notification(), timeout=300)
        assert event["type"] == "notification"
        assert event["tool_name"] == "delegating_tool"
        assert "widget" in (event.get("message") or "").lower()

        # ── loop must now complete successfully ───────────────────────────────
        result = await asyncio.wait_for(handle.result(), timeout=300)
        assert result is not None, "Loop should complete with a response"
    finally:
        try:
            await handle.stop("test cleanup")
        except Exception:
            pass
