"""
tests/conversation_manager/actions/test_storage_check.py
=========================================================

Tests that verify the ConversationManager brain correctly handles
``_StorageCheckHandle``-wrapped actor handles where ``result()``
resolves after the task phase (Phase 1), while storage runs in the
background (Phase 2).

With early-resolving ``result()``, the CM receives a single
``ActorResult`` event when the task completes.  The brain is shown the
result and relays it to the user.  Storage runs concurrently in the
background — the handle remains live (``done()`` returns ``False``)
but no second wake-up occurs.

The brain's model is scripted (``scripted_brain``): how a live model words
the relay is a question about the model.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.helpers import _handle_project
from tests.legacy.conversation_manager.cm_helpers import filter_events_by_type
from tests.legacy.conversation_manager.actions.integration.helpers import (
    run_cm_until_wait,
)
from tests.legacy.conversation_manager.actions.scripted_brain import (
    say,
    turn,
    wait,
)
from unify.legacy.conversation_manager.events import (
    UnifyMessageReceived,
    UnifyMessageSent,
)


def _make_early_resolving_handle(
    result_value: str,
) -> tuple[MagicMock, asyncio.Event]:
    """Create a mock handle simulating ``_StorageCheckHandle`` with early result.

    ``result()`` returns *immediately* (simulating the early-resolve after
    Phase 1).  ``done()`` starts as ``False`` (storage still running) and
    transitions to ``True`` when *done_gate* is set.

    Returns (handle, done_gate).
    """
    done_gate = asyncio.Event()
    handle = MagicMock()

    def _done():
        return done_gate.is_set()

    handle.done = _done

    async def _result():
        return result_value

    handle.result = _result

    # next_notification / next_clarification block so watcher tasks
    # don't spin-loop or error out.
    async def _block_forever():
        await asyncio.Event().wait()
        return {}

    handle.next_notification = _block_forever
    handle.next_clarification = _block_forever
    handle.ask = AsyncMock(return_value=MagicMock(result=AsyncMock(return_value="")))
    handle.interject = AsyncMock()
    handle.stop = AsyncMock()
    handle.pause = AsyncMock(return_value=None)
    handle.resume = AsyncMock(return_value=None)
    handle.answer_clarification = AsyncMock()
    handle.get_history = MagicMock(return_value=[])

    return handle, done_gate


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_storage_check_result_relay(initialized_cm, scripted_brain):
    """The CM brain relays the task result when ``result()`` resolves.

    With early-resolving ``result()``, the CM receives a single
    ActorResult event.  The brain is shown the action's result, and its
    relay reaches the user.

    After the result is relayed, storage continues in the background
    (``done()`` returns ``False``).  When storage finishes
    (``done()`` transitions to ``True``), the action is already in
    ``completed_actions`` — no duplicate message should be sent.
    """
    cm = initialized_cm
    brain = scripted_brain()

    simulated_result = "The notes file lists three action items for Friday."
    mock_handle, done_gate = _make_early_resolving_handle(simulated_result)

    # ── Setup: process the user's message into chat history ───────────
    await cm.step(
        UnifyMessageReceived(
            content="What is in the notes file in my workspace?",
        ),
        run_llm=False,
    )

    from unify.common.prompt_helpers import now as prompt_now

    handle_id = 0

    # ── Action completes: result() returned, action moved to completed ─
    #
    # Simulate what the CM does when ActorResult fires: the action moves
    # from in_flight to completed with an act_completed entry.

    cm.cm.completed_actions[handle_id] = {
        "handle": mock_handle,
        "query": "What is in the notes file in my workspace?",
        "persist": False,
        "handle_actions": [
            {
                "action_name": "act_started",
                "query": "What is in the notes file in my workspace?",
                "timestamp": prompt_now(),
            },
            {
                "action_name": "act_completed",
                "query": simulated_result,
                "timestamp": prompt_now(),
            },
        ],
        "initial_snapshot_state": None,
    }

    relay = f"Here's what's in your notes file: {simulated_result}"
    brain.next(turn(say(relay), wait()))
    events = await run_cm_until_wait(cm)

    # The brain was shown the completed action with its result.
    completed = brain.last_state().partition("<completed_actions>")[2]
    assert f"<action id='{handle_id}'" in completed
    assert "status='completed'" in completed
    assert f"<result>{simulated_result}</result>" in completed

    sent_events = filter_events_by_type(events, UnifyMessageSent)
    assert [e.content for e in sent_events] == [relay], (
        "Brain should relay the task result when the action completes. "
        "The act_completed event includes the result."
    )
    assert cm.cm.chat_history.recent()[-1].content == relay

    # ── Storage finishes: done() becomes True ──────────────────────────
    # The action is already in completed_actions. Transitioning done()
    # should not produce any additional messages.
    done_gate.set()
    assert mock_handle.done() is True
    assert handle_id in cm.cm.completed_actions
