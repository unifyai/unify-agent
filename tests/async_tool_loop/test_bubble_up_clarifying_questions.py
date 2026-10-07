from __future__ import annotations

import asyncio
from typing import Optional

import pytest
import unillm
from unify.common.async_tool_loop import start_async_tool_loop
from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
    _wait_for_user_message_prefix,
    _wait_for_assistant_steer_action,
    first_user_message,
    first_assistant_tool_call,
    first_assistant_steer_call,
    first_tool_message_by_name_prefix,
    first_tool_message_by_name,
    last_plain_assistant_message,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# as_shipped: deleted in step 5 (clarification channel of steerable handles)
pytestmark = [pytest.mark.llm_call, pytest.mark.usefixtures("as_shipped")]

# ──────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────


def make_llm(
    system_message: Optional[str] = None,
    **llm_kwargs,
) -> unillm.AsyncUnify:
    return new_llm_client(
        **llm_kwargs,
        system_message=system_message,
    )


# ──────────────────────────────────────────────────────────────────────────
# 1.  DUMMY TOOLS – send_email immediately needs clarification
# ──────────────────────────────────────────────────────────────────────────
async def send_email(
    address: str,
    description: str,
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """Send an email, based on the general description provided."""
    if _clarification_up_q is None or _clarification_down_q is None:
        raise RuntimeError("clarification queues missing")

    # Dummy code
    await _clarification_up_q.put(
        "It's best that we also inform him what we're bringing. Will you be bringing anything with you (food/drink)?",
    )
    await _clarification_down_q.get()
    return f"Email sent!"


async def send_text(
    number: str,
    description: str,
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """Send a text message, based on the general description provided."""
    if _clarification_up_q is None or _clarification_down_q is None:
        raise RuntimeError("clarification queues missing")


# ──────────────────────────────────────────────────────────────────────────
# 2.  OUTER get_clarification – simulates the user
# ──────────────────────────────────────────────────────────────────────────
asked_questions: list[str] = []  # for assertions


# ──────────────────────────────────────────────────────────────────────────
# 3.  The test
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@_handle_project
async def test_clarification_bubbles_up_two_tiers(llm_config) -> None:
    """
    Verifies that the clarification travels up & the answer travels down two
    levels of the call stack.
    """

    outer_client = make_llm(
        "You are coordinating internal tools.\n"
        "When any pending tool asks a clarification question, assume the missing detail is a user-specific fact or preference "
        "that is NOT inferable from the current tool context or prior messages in this loop. Do not guess or invent.\n"
        "Therefore: (1) Do not start unrelated base tools until that pending call is unblocked; "
        "(2) Call `request_clarification` to ask the user the exact question and wait for their answer (you may call other tools only to fetch that answer); "
        '(3) As soon as you have the answer, immediately call steer(call_id=<the id from the "[clarification ...]" tail message>, action="clarify", payload=<the answer>) to provide the answer so the pending call can resume.\n'
        "After the original call resumes or completes, you may continue with further tools as needed.\n"
        "When the email has been sent successfully, end your final assistant message with an explicit confirmation using the word 'sent' (e.g., 'Email sent.').\n"
        "Do not hallucinate any details; if unknown, ask. Keep responses concise.",
        **llm_config,
    )

    clar_up_q = asyncio.Queue()
    clar_down_q = asyncio.Queue()

    outer_tools = {
        "send_email": send_email,
        "send_text": send_text,
    }

    outer_handle = start_async_tool_loop(  # type: ignore[attr-defined]
        outer_client,
        message="Please email jonathan.smith@example.com and politely tell him I (Dan) will be arriving at the BBQ around 5pm.",
        tools=outer_tools,
        time_awareness=False,
        clarification_queues=(clar_up_q, clar_down_q),
    )

    # Deterministic ordering using triggers:
    # 1) Wait until assistant schedules send_email
    await _wait_for_tool_request(outer_client, "send_email", timeout=120.0)
    # 2) Wait until the coalesced clarification tail message appears (the
    # pending stub tool message is never rewritten with the question).
    await _wait_for_user_message_prefix(
        outer_client,
        "[clarification ",
        timeout=120.0,
    )

    # 3) The request_clarification tool will bubble the question up – capture it
    await clar_up_q.get()
    # 4) Provide the answer; assistant should then call steer(action="clarify")
    await clar_down_q.put("I'll be bringing sausages and a pack of beer")

    # 5) Ensure the assistant has invoked steer(action="clarify")
    await _wait_for_assistant_steer_action(
        outer_client,
        "clarify",
        timeout=120.0,
    )

    await outer_handle.result()

    # ─────────────────────────
    # Assertions
    # ─────────────────────────

    import json

    # 1️⃣ original user request – robust to clients that don't persist system messages
    _first_user = first_user_message(outer_client.messages)
    assert _first_user["content"] == (
        "Please email jonathan.smith@example.com and politely tell him I (Dan) "
        "will be arriving at the BBQ around 5pm."
    )

    # 2️⃣ assistant chooses `send_email` --------------------------------------
    m1, call1 = first_assistant_tool_call(outer_client.messages, "send_email")
    args1 = json.loads(call1["function"]["arguments"])
    assert args1["address"] == "jonathan.smith@example.com"

    # 3️⃣ tool asks a clarification question — delivered as a coalesced
    # [clarification <call_id>] user-role tail message, not a rewrite of the
    # pending tool-reply stub.
    clar_req = next(
        m
        for m in outer_client.messages
        if m.get("role") == "user"
        and str(m.get("content") or "").startswith("[clarification ")
    )
    assert "Will you be bringing anything" in clar_req["content"]

    # 4️⃣ assistant calls `request_clarification` -----------------------------
    m3, req_call = first_assistant_tool_call(
        outer_client.messages,
        "request_clarification",
    )

    # 5️⃣ tool returns the user’s answer --------------------------------------
    clar_ans = first_tool_message_by_name(
        outer_client.messages,
        "request_clarification",
    )
    assert "sausages" in clar_ans["content"]
    assert "beer" in clar_ans["content"]

    # 6️⃣ assistant forwards the answer via steer(action="clarify") ----------
    _m5, _clar = first_assistant_steer_call(
        outer_client.messages,
        "clarify",
    )

    # 7️⃣ final tool message contains the real result -------------------------
    final_tool = first_tool_message_by_name_prefix(
        outer_client.messages,
        "steer:clarify",
    )
    assert "Email sent" in final_tool["content"]

    # 8️⃣ assistant wraps up ---------------------------------------------------
    closing = last_plain_assistant_message(outer_client.messages)
    assert closing is not None, "Expected a final assistant message"


# ---------------------------------------------------------------------------
# inner tool  ➟  always asks one clarification
# ---------------------------------------------------------------------------
async def inner_tool(
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    if _clarification_up_q is None or _clarification_down_q is None:
        raise RuntimeError("queues missing")

    await _clarification_up_q.put("Inner loop: what colour should the widget be?")
    color = await _clarification_down_q.get()
    return f"✅ inner finished, color: {color}"


# ---------------------------------------------------------------------------
# outer tool  ➟  immediately spawns an async-tool loop and RETURNS its handle
# ---------------------------------------------------------------------------
async def delegating_tool(
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:  # return type misleading on purpose
    inner_llm = make_llm(
        "You are coordinating internal tools in a nested loop.\n"
        "CRITICAL: When any internal tool requests clarification, the missing information is a user-specific preference or fact "
        "that is NOT available to you from tool context or prior messages in this loop. Do not infer, assume, or guess.\n"
        "Therefore, you MUST first call `request_clarification` to ask the user the exact question, wait for the user's answer, "
        'and ONLY THEN call steer(call_id=<the id from the "[clarification ...]" tail message>, action="clarify", payload=<answer>) '
        "to provide that answer so the tool can resume.\n"
        'Never call steer(action="clarify") unless you have just obtained the answer via `request_clarification` in this conversation. '
        "For example, for a question like 'what colour should the widget be?', treat it as a personal preference unknown to you; ask the user. "
        "Keep responses concise.",
    )

    handle = start_async_tool_loop(  # <-- returns AsyncToolLoopHandle
        inner_llm,
        message="Run inner_tool please.",
        tools={
            "inner_tool": inner_tool,
        },
        clarification_queues=(_clarification_up_q, _clarification_down_q),
    )
    return handle  # outer tool finishes instantly


# ---------------------------------------------------------------------------
# regression test
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_clarification_bubbles_through_returned_handle(llm_config) -> None:
    """Clarification raised *inside* the returned handle must still reach the user."""

    clar_up_q: asyncio.Queue[str] = asyncio.Queue()
    clar_down_q: asyncio.Queue[str] = asyncio.Queue()

    outer_llm = make_llm(
        "You are the TOP-LEVEL coordinator. When any pending tool (including nested delegated tools) asks a clarification "
        "question via a [clarification ...] message, you MUST:\n"
        "(1) Call `request_clarification` to ask the user that exact question;\n"
        "(2) Wait for the user's answer;\n"
        '(3) Forward that answer via steer(call_id=<the id from the "[clarification ...]" '
        'tail message>, action="clarify", payload=<answer>).\n'
        "Treat these as user preferences or facts unknown to you; do NOT answer them yourself, guess, or infer from unrelated context.\n"
        "Do NOT start unrelated tools until pending clarifications are resolved.",
        **llm_config,
    )

    handle = start_async_tool_loop(
        outer_llm,
        message="Run delegating_tool please.",
        tools={
            "delegating_tool": delegating_tool,
        },
        clarification_queues=(clar_up_q, clar_down_q),
    )

    # ── satisfy the clarification that should bubble up ──────────────────
    question = await asyncio.wait_for(clar_up_q.get(), timeout=300)
    assert "what colour" in question.lower() or "what color" in question.lower()

    await clar_down_q.put("Blue, please")

    # ── loop must now complete successfully ───────────────────────────────
    await asyncio.wait_for(handle.result(), timeout=300)

    # final sanity-check: assistant ends with the confirmation from inner_tool
    assert "blue" in outer_llm.messages[-1]["content"].lower()


# ---------------------------------------------------------------------------
# Inner tool: always requests clarification and blocks forever without answer
# ---------------------------------------------------------------------------
async def inner_ask(
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """A tool that always asks a clarification question."""
    if _clarification_up_q is None or _clarification_down_q is None:
        raise RuntimeError("clarification queues missing")

    await _clarification_up_q.put("What is the user's favorite color?")
    answer = await _clarification_down_q.get()
    return f"Got answer: {answer}"


# ---------------------------------------------------------------------------
# Outer tool: spawns an inner loop that will request clarification
# ---------------------------------------------------------------------------
async def spawn_inner(
    *,
    _clarification_up_q: asyncio.Queue[str] | None = None,
    _clarification_down_q: asyncio.Queue[str] | None = None,
) -> str:
    """
    Spawns an inner async tool loop.
    The inner loop will call inner_ask, which blocks on clarification.
    """
    inner_llm = make_llm(
        system_message=(
            "You coordinate internal tools. "
            "Call inner_ask to gather user preferences."
        ),
    )

    handle = start_async_tool_loop(
        inner_llm,
        message="Please run inner_ask.",
        tools={
            "inner_ask": inner_ask,
        },
    )
    return handle


# ---------------------------------------------------------------------------
# Test: Outer loop exits cleanly when inner is blocked on unanswered clarification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outer_loop_exits_when_inner_blocked_on_unanswered_clarification(
    llm_config,
) -> None:
    """
    Regression test for deadlock when LLM ends conversation while inner tool
    awaits clarification.

    Scenario:
    1. Outer loop's LLM calls `spawn_inner`
    2. Inner loop's LLM calls `inner_ask`
    3. Inner tool blocks on `down_q.get()` waiting for clarification answer
    4. Clarification bubbles up to outer loop
    5. Outer loop's LLM (without request_clarification tool!) responds with content

    Expected: Outer loop exits with the LLM's response.
    Bug (before fix): Deadlock - outer waits for inner, inner waits forever.
    """
    import time

    outer_llm = make_llm(
        system_message=(
            "You coordinate tools. If a nested tool requests clarification and you "
            "cannot answer it (because you don't have a request_clarification tool), "
            "call the `final_answer` tool with answer='I cannot help with that.' to "
            "end the conversation.\n"
            "IMPORTANT: When you see a [clarification ...] message, use `final_answer` "
            "to respond with 'I cannot help with that.' - do NOT try to answer it "
            "yourself or guess, and do NOT call stop or wait helpers."
        ),
        **llm_config,
    )

    # NOTE: Outer loop does NOT have request_clarification tool!
    handle = start_async_tool_loop(
        outer_llm,
        message="Please run spawn_inner.",
        tools={
            "spawn_inner": spawn_inner,
        },
    )

    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(handle.result(), timeout=60)
    except asyncio.TimeoutError:
        pytest.fail(
            "DEADLOCK: Outer loop timed out waiting for inner loop. "
            "Inner loop is blocked on clarification that will never be answered.",
        )
    elapsed = time.perf_counter() - start

    # Verify we got a real result (not the cancellation placeholder)
    assert isinstance(result, str), f"Expected string result, got {type(result)}"
    assert len(result) > 0, "Expected non-empty result"
    assert result != "processed stopped early, no result", (
        f"Got cancellation placeholder instead of real LLM response. "
        f"This suggests the fix is not working - the loop deadlocked until timeout. "
        f"Elapsed time: {elapsed:.1f}s"
    )

    # Verify we exited quickly (not via 60s timeout deadlock)
    assert elapsed < 45, (
        f"Loop took {elapsed:.1f}s to exit. If close to 60s, this suggests deadlock "
        f"that was only broken by the asyncio.wait_for timeout, not the fix."
    )
