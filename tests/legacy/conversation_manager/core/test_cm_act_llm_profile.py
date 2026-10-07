from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from unify.common.llm_client import new_llm_client
from unify.common.llm_helpers import method_to_schema
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)

pytestmark = [pytest.mark.eval, pytest.mark.llm_call]


def _make_mock_cm() -> MagicMock:
    cm = MagicMock()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = MagicMock()
    cm.notifications_bar.notifications = []
    cm.chat_history = []
    cm._current_state_snapshot = None
    cm._current_snapshot_state = None
    cm._pending_steering_tasks = set()
    cm._initialized = asyncio.Event()
    cm._initialized.set()
    cm._session_logger = MagicMock()
    cm.request_llm_run = AsyncMock()
    cm.event_broker = MagicMock()
    cm.event_broker.publish = AsyncMock()
    cm.call_manager = MagicMock()
    cm.suppress_duplicate_commissioning_tool.return_value = None
    cm.actor = MagicMock()
    return cm


def _first_tool_args(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    assert tool_calls, f"Expected a tool call, got message: {message}"
    function = tool_calls[0].get("function") or {}
    raw_args = function.get("arguments") or "{}"
    return json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_explicit_max_thinking_request_selects_gpt_5_5_high():
    cm = _make_mock_cm()
    tools = ConversationManagerBrainActionTools(cm)
    schema = method_to_schema(tools.act, tool_name="act", include_class_name=False)

    client = new_llm_client(origin="test.cm.act_llm_profile", stateful=True)
    client.set_system_message(
        "You are a ConversationManager brain. Choose the single available "
        "tool call and arguments that best satisfy the user's request.",
    )
    await client.generate(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use all of your thinking effort to analyze the most robust "
                    "plan for handling a difficult strategic research task."
                ),
            },
        ],
        tools=[schema],
        tool_choice="required",
    )

    args = _first_tool_args(client.messages[-1])
    assert args.get("llm_profile") == "gpt_5_5_high"
