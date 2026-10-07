"""Focused checks for Console → CM action stop by calling_id."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_stop_in_flight_action_by_calling_id_stops_and_moves():
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

    cm = MagicMock(spec=ConversationManager)
    cm.in_flight_actions = {}
    cm.completed_actions = {}

    handle = AsyncMock()
    handle._manager_call_id = "call-root-1"
    cm.in_flight_actions[7] = {
        "handle": handle,
        "query": "do the thing",
        "calling_id": "call-root-1",
        "handle_actions": [],
    }

    stopped = await ConversationManager.stop_in_flight_action_by_calling_id(
        cm,
        "call-root-1",
        reason="Stopped from Console Actions pane.",
    )

    assert stopped is True
    handle.stop.assert_awaited_once_with(reason="Stopped from Console Actions pane.")
    assert 7 not in cm.in_flight_actions
    assert 7 in cm.completed_actions
    assert cm.completed_actions[7]["handle_actions"][-1]["action_name"] == "stop_7"


@pytest.mark.asyncio
async def test_stop_in_flight_action_by_calling_id_unknown_returns_false():
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )

    cm = MagicMock(spec=ConversationManager)
    cm.in_flight_actions = {}
    cm.completed_actions = {}

    stopped = await ConversationManager.stop_in_flight_action_by_calling_id(
        cm,
        "missing",
        reason="x",
    )
    assert stopped is False
