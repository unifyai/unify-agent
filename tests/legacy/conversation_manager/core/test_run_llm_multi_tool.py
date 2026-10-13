"""
tests/conversation_manager/core/test_run_llm_multi_tool.py
=============================================================

Tests for multi-tool handling in ``_run_llm()``.

These tests capture two pre-existing bugs in how ``_run_llm()`` handles
LLM turns where multiple tools are called concurrently:

1. **Lost tool names**: ``_run_llm()`` returns only the first tool name
   via ``result.tool_name``, so callers (including the test driver's
   ``all_tool_calls``) lose visibility of subsequent tool calls.

2. **Missed wait scheduling**: The ``wait(delay=N)`` scheduling logic
   checks only ``result.tool_name`` (first tool). If the LLM calls
   ``wait(delay=N)`` alongside other tools in the same turn and ``wait``
   is not the first tool call, the delayed follow-up turn is never
   scheduled.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.helpers import _handle_project
from unify.common.single_shot import SingleShotResult, ToolExecution
from unify.legacy.conversation_manager.conversation_manager import ConversationManager
from unify.legacy.conversation_manager.domains.event_handlers import (
    OPEN_SLOW_BRAIN_TURN_NOTIFICATION,
)
from unify.legacy.conversation_manager.events import OpenSlowBrainTurn


def _make_multi_tool_result(*tool_pairs: tuple[str, dict, object]) -> SingleShotResult:
    """Build a ``SingleShotResult`` with multiple tool executions.

    Each ``tool_pairs`` element is ``(name, args, result)``.
    """
    return SingleShotResult(
        tools=[
            ToolExecution(name=name, args=args, result=result)
            for name, args, result in tool_pairs
        ],
        text_response=None,
        structured_output=None,
    )


# =============================================================================
# Bug 1: _run_llm() drops tool names beyond the first
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_returns_all_tool_names(initialized_cm):
    """_run_llm() should surface ALL tool names from a multi-tool turn.

    Currently it returns only the first via ``result.tool_name``, silently
    dropping the rest.  This means any caller relying on the return value
    (including the test driver's ``all_tool_calls``) has an incomplete
    picture of what the LLM decided.
    """
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("act", {"query": "Find the sales CSV"}, {"status": "acting"}),
        (
            "act",
            {"query": "Summarize the attached report", "persist": True},
            {"status": "acting"},
        ),
        ("send_unify_message", {"content": "On it."}, None),
    )

    with patch(
        "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
        AsyncMock(return_value=fake_result),
    ):
        returned = await cm._run_llm()

    # The correct behavior: _run_llm() should return all tool names
    # so callers can track the full set of actions taken in this turn.
    assert isinstance(returned, list), (
        f"_run_llm() should return a list of tool names when multiple tools "
        f"are called, but got {type(returned).__name__}: {returned!r}"
    )
    assert returned == [
        "act",
        "act",
        "send_unify_message",
    ], f"Expected all three tool names, got: {returned}"


@pytest.mark.asyncio
@_handle_project
async def test_step_driver_tracks_all_tool_names(initialized_cm):
    """CMStepDriver.all_tool_calls should record EVERY tool called per turn.

    Currently it appends only the single string returned by ``_run_llm()``,
    so when the LLM calls ``[send_unify_message, act]`` in one turn, only
    ``send_unify_message`` appears in ``all_tool_calls``.
    """
    cm_driver = initialized_cm

    fake_result = _make_multi_tool_result(
        ("send_unify_message", {"content": "On it."}, None),
        (
            "act",
            {"query": "Summarize the attached report", "persist": True},
            {"status": "acting"},
        ),
    )

    with patch(
        "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
        AsyncMock(return_value=fake_result),
    ):
        returned = await cm_driver.cm._run_llm()

    # Simulate what the step driver does: append the return value.
    if isinstance(returned, list):
        cm_driver.all_tool_calls.extend(returned)
    elif returned:
        cm_driver.all_tool_calls.append(returned)

    assert "act" in cm_driver.all_tool_calls, (
        f"all_tool_calls should contain 'act' but only has: {cm_driver.all_tool_calls}. "
        f"The second tool call is silently dropped."
    )


# =============================================================================
# OpenSlowBrainTurn: recurring turns until wait
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_opens_follow_on_turn_without_wait(initialized_cm):
    """A turn without wait should schedule an OpenSlowBrainTurn follow-on."""
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("act", {"query": "Summarize files"}, {"status": "acting"}),
        ("send_unify_message", {"content": "Working on it."}, None),
    )

    with (
        patch(
            "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(return_value=fake_result),
        ),
        patch.object(
            cm,
            "_open_slow_brain_follow_on_turn",
            new_callable=AsyncMock,
        ) as mock_follow_on,
    ):
        await cm._run_llm()

    mock_follow_on.assert_awaited_once()
    kwargs = mock_follow_on.await_args.kwargs
    assert kwargs["previous_tools"] == ["act", "send_unify_message"]
    assert kwargs["origin_run_id"]


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_skips_follow_on_when_wait_called(initialized_cm):
    """Calling wait ends the chain — no OpenSlowBrainTurn follow-on."""
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("send_unify_message", {"content": "Done."}, None),
        ("wait", {}, None),
    )

    with (
        patch(
            "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(return_value=fake_result),
        ),
        patch.object(
            cm,
            "_open_slow_brain_follow_on_turn",
            new_callable=AsyncMock,
        ) as mock_follow_on,
    ):
        await cm._run_llm()

    mock_follow_on.assert_not_awaited()


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_skips_follow_on_for_wait_with_delay(initialized_cm):
    """wait(delay=N) ends the chain while still scheduling the delayed run."""
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("act", {"query": "Long task"}, {"status": "acting"}),
        ("wait", {"delay": 5}, None),
    )

    with (
        patch(
            "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(return_value=fake_result),
        ),
        patch.object(cm, "run_llm", new_callable=AsyncMock) as mock_run,
        patch.object(
            cm,
            "_open_slow_brain_follow_on_turn",
            new_callable=AsyncMock,
        ) as mock_follow_on,
    ):
        await cm._run_llm()

    mock_run.assert_awaited_once_with(delay=5)
    mock_follow_on.assert_not_awaited()


@pytest.mark.asyncio
@_handle_project
async def test_open_slow_brain_turn_notification_in_rendered_state(initialized_cm):
    """OpenSlowBrainTurn notification appears in the slow-brain notifications block."""
    from datetime import timedelta

    from unify.legacy.conversation_manager.domains.event_handlers import EventHandler

    cm = initialized_cm.cm
    event = OpenSlowBrainTurn(
        origin_run_id="llmrun-000001",
        previous_tools=["send_unify_message"],
    )
    cm.last_snapshot = event.timestamp - timedelta(seconds=1)
    await EventHandler.handle_event(event, cm)

    rendered = cm.prompt_renderer.render_notification_bar(
        cm.notifications_bar,
        last_snapshot=cm.last_snapshot,
    )
    assert OPEN_SLOW_BRAIN_TURN_NOTIFICATION in rendered
    assert "<notifications>" in rendered


# =============================================================================
# Bug 2: wait(delay=N) scheduling missed when wait is not the first tool
# =============================================================================


@pytest.mark.asyncio
@_handle_project
async def test_wait_delay_scheduled_when_not_first_tool(initialized_cm):
    """wait(delay=N) should schedule a delayed follow-up even when called
    alongside other tools in the same turn.

    Currently the scheduling logic checks only ``result.tool_name`` (first
    tool).  If the LLM calls ``[act, wait(delay=5)]``, the wait is the
    second tool and ``result.tool_name`` is ``"act"``, so the ``delay=5``
    scheduling is silently skipped.
    """
    cm = initialized_cm.cm

    fake_result = _make_multi_tool_result(
        ("act", {"query": "Long task"}, {"status": "acting"}),
        ("wait", {"delay": 5}, None),
    )

    with (
        patch(
            "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(return_value=fake_result),
        ),
        patch.object(cm, "run_llm", new_callable=AsyncMock) as mock_run,
    ):
        await cm._run_llm()

    mock_run.assert_called_once_with(delay=5)


@pytest.mark.asyncio
async def test_run_llm_records_recent_tool_executions_for_follow_up_turns(
    initialized_cm,
):
    cm = initialized_cm.cm
    cm._recent_tool_executions = []
    cm._recent_commissioning_successes = {}

    fake_result = _make_multi_tool_result(
        (
            "act",
            {"query": "Add the Ops HQ row"},
            {"status": "acting", "query": "Add the Ops HQ row"},
        ),
    )
    with patch(
        "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
        AsyncMock(return_value=fake_result),
    ):
        await cm._run_llm(trace_meta={"origin_event_name": "UnifyMessageSent"})

    assert len(cm._recent_tool_executions) >= 1
    last = cm._recent_tool_executions[-1]
    assert last["tool_name"] == "act"
    assert last["origin_event_name"] == "UnifyMessageSent"
    assert "Ops HQ" in last["result_preview"]


def test_run_llm_marks_tool_commit_boundary():
    cm = ConversationManager.__new__(ConversationManager)
    cm._session_logger = MagicMock()
    cm.debouncer = MagicMock()
    cm.debouncer.running_task_trace_meta = {
        "run_id": "llmrun-000123",
        "origin_event_name": "UnifyMessageReceived",
    }
    trace_meta = {"origin_event_name": "UnifyMessageReceived"}

    cm._mark_tool_commit_started(trace_meta, "llmrun-000123")

    assert trace_meta["tool_commit_started"] == "true"
    assert cm.debouncer.running_task_trace_meta["tool_commit_started"] == "true"


@pytest.mark.asyncio
@_handle_project
async def test_run_llm_carries_recent_tool_executions_into_next_turn_prompt(
    initialized_cm,
):
    cm = initialized_cm.cm
    captured_messages = []

    async def fake_single_shot(*args, **kwargs):
        messages = args[1]
        captured_messages.append(messages)
        if len(captured_messages) == 1:
            return _make_multi_tool_result(
                (
                    "act",
                    {"query": "Add the Ops HQ row"},
                    {"status": "acting", "query": "Add the Ops HQ row"},
                ),
            )
        return SingleShotResult(tools=[], text_response="noop", structured_output=None)

    with (
        patch(
            "unify.legacy.conversation_manager.conversation_manager.single_shot_tool_decision",
            AsyncMock(side_effect=fake_single_shot),
        ),
        patch.object(
            cm,
            "_open_slow_brain_follow_on_turn",
            new_callable=AsyncMock,
        ),
    ):
        await cm._run_llm(trace_meta={"origin_event_name": "UnifyMessageSent"})
        await cm._run_llm(trace_meta={"origin_event_name": "UnifyMessageReceived"})

    assert len(captured_messages) == 2
    second_turn_text = "\n".join(
        str(message.get("content")) for message in captured_messages[1]
    )
    assert "<recent_tool_executions>" in second_turn_text
    assert "tool=act" in second_turn_text


def test_duplicate_act_suppression_only_blocks_immediate_followups():
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

    cm = ConversationManager.__new__(ConversationManager)
    cm._llm_gen = 7
    tool_args = {
        "query": "Summarize the attached quarterly report",
        "response_format": None,
        "persist": False,
        "include_conversation_context": True,
    }
    fingerprint = cm._commissioning_tool_fingerprint("act", tool_args)
    cm._recent_commissioning_successes = {fingerprint: 6}
    cm._active_llm_trace_meta = {"origin_event_name": "UnifyMessageSent"}

    suppressed = cm.suppress_duplicate_commissioning_tool(
        tool_name="act",
        tool_args=tool_args,
    )

    assert suppressed is not None
    assert suppressed["error_kind"] == "duplicate_suppressed"
    assert suppressed["details"]["origin_event_name"] == "UnifyMessageSent"

    cm._active_llm_trace_meta = {"origin_event_name": "UnifyMessageReceived"}
    assert (
        cm.suppress_duplicate_commissioning_tool(
            tool_name="act",
            tool_args=tool_args,
        )
        is None
    )


def test_act_duplicate_fingerprint_normalizes_optional_defaults():
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

    cm = ConversationManager.__new__(ConversationManager)
    minimal_args = {
        "query": "Summarize the attached quarterly report",
    }
    expanded_args = {
        **minimal_args,
        "response_format": None,
        "persist": False,
        "include_conversation_context": True,
    }

    assert cm._commissioning_tool_fingerprint(
        "act",
        minimal_args,
    ) == cm._commissioning_tool_fingerprint("act", expanded_args)
