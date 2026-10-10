"""
tests/conversation_manager/actions/test_parent_chat_context.py
========================================================================

Tests for the ``include_conversation_context`` parameter on CM brain tools.

When the LLM sets ``include_conversation_context=False``, the action should
receive no parent chat context, and subsequent steering calls (interject, ask)
on that handle should also skip context forwarding — mirroring the sticky
opt-out semantics of the async tool loop's ``include_parent_chat_context``.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tests.helpers import _handle_project
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)


class TestActIncludeConversationContext:
    """Verify act() respects include_conversation_context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_includes_context_by_default(self, initialized_cm):
        """Default (include_conversation_context=True) passes the snapshot."""
        cm = initialized_cm.cm

        test_snapshot = {
            "role": "user",
            "content": "<notifications>test</notifications>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        captured = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.actor.act = capturing_act

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(query="test query")

            assert len(captured) == 1
            assert captured[0] is not None
            assert isinstance(captured[0], list)
            assert len(captured[0]) == 1
        finally:
            cm.actor.act = original_act
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_skips_context_when_opted_out(self, initialized_cm):
        """include_conversation_context=False passes None as parent context."""
        cm = initialized_cm.cm

        test_snapshot = {
            "role": "user",
            "content": "<notifications>should not appear</notifications>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        captured = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.actor.act = capturing_act

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(
                query="what time is it",
                include_conversation_context=False,
            )

            assert len(captured) == 1
            assert captured[0] is None
        finally:
            cm.actor.act = original_act
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_opt_out_stored_in_flight_actions(self, initialized_cm):
        """The context_opted_in flag is persisted in in_flight_actions."""
        cm = initialized_cm.cm

        brain_tools = ConversationManagerBrainActionTools(cm)
        await brain_tools.act(
            query="simple lookup",
            include_conversation_context=False,
        )

        assert len(cm.in_flight_actions) == 1
        handle_data = next(iter(cm.in_flight_actions.values()))
        assert handle_data["context_opted_in"] is False

    @pytest.mark.asyncio
    @_handle_project
    async def test_opt_in_stored_in_flight_actions(self, initialized_cm):
        """Default opt-in is persisted in in_flight_actions."""
        cm = initialized_cm.cm

        brain_tools = ConversationManagerBrainActionTools(cm)
        await brain_tools.act(query="find the London report")

        assert len(cm.in_flight_actions) == 1
        handle_data = next(iter(cm.in_flight_actions.values()))
        assert handle_data["context_opted_in"] is True


class TestSteeringRespectsOptOut:
    """Verify that steering tools respect the initial context opt-out."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_skips_context_when_opted_out(self, initialized_cm):
        """Interject on an opted-out action sends no context diff."""
        from unify.legacy.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        captured_cont = []

        async def capturing_interject(message, **kwargs):
            captured_cont.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        initial_snapshot = SnapshotState(
            full_render="<initial/>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts,
                    rendered="[User]: Hello",
                ),
            ],
        )

        # Register with context_opted_in=False
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "simple task",
            "handle_actions": [],
            "initial_snapshot_state": initial_snapshot,
            "context_opted_in": False,
        }

        # Create a current snapshot with a new message (would produce a diff)
        new_ts = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)
        cm._current_snapshot_state = SnapshotState(
            full_render="<current/>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts,
                    rendered="[User]: Hello",
                ),
                MessageElement(
                    index_in_conversation=1,
                    timestamp=new_ts,
                    rendered="[User]: New message that would normally be a diff",
                ),
            ],
        )

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["interject_action"](
                handle_id=0,
                message="do more stuff",
            )

            assert len(captured_cont) == 1
            assert captured_cont[0] is None

        finally:
            cm.in_flight_actions.clear()
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_sends_context_when_opted_in(self, initialized_cm):
        """Interject on an opted-in action sends the context diff."""
        from unify.legacy.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        captured_cont = []

        async def capturing_interject(message, **kwargs):
            captured_cont.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        initial_snapshot = SnapshotState(
            full_render="<initial/>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts,
                    rendered="[User]: Hello",
                ),
            ],
        )

        # Register with context_opted_in=True (default)
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "research task",
            "handle_actions": [],
            "initial_snapshot_state": initial_snapshot,
            "context_opted_in": True,
        }

        new_ts = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)
        cm._current_snapshot_state = SnapshotState(
            full_render="<current/>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts,
                    rendered="[User]: Hello",
                ),
                MessageElement(
                    index_in_conversation=1,
                    timestamp=new_ts,
                    rendered="[User]: Please also check the calendar",
                ),
            ],
        )

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["interject_action"](
                handle_id=0,
                message="also check calendar",
            )

            assert len(captured_cont) == 1
            assert captured_cont[0] is not None
            assert len(captured_cont[0]) == 1
            assert captured_cont[0][0].get("_cm_context_diff") is True
            assert "check the calendar" in captured_cont[0][0].get("content", "")

        finally:
            cm.in_flight_actions.clear()
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_ask_skips_context_when_opted_out(self, initialized_cm):
        """Ask on an opted-out action sends no parent context."""
        cm = initialized_cm.cm

        captured_ctx = []

        class MockAskHandle:
            async def result(self):
                return "mock answer"

        async def capturing_ask(question, **kwargs):
            captured_ctx.append(kwargs.get("_parent_chat_context"))
            return MockAskHandle()

        mock_handle = MagicMock()
        mock_handle.ask = capturing_ask

        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "simple task",
            "handle_actions": [],
            "context_opted_in": False,
        }

        cm._current_state_snapshot = {
            "role": "user",
            "content": "<state>should not appear</state>",
            "_cm_state_snapshot": True,
        }

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["ask_action"](
                handle_id=0,
                question="what's the status?",
            )

            pending = set(cm._pending_steering_tasks)
            if pending:
                await asyncio.wait(pending, timeout=300)

            assert len(captured_ctx) == 1
            assert captured_ctx[0] is None

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_ask_sends_context_when_opted_in(self, initialized_cm):
        """Ask on an opted-in action sends the parent context."""
        cm = initialized_cm.cm

        captured_ctx = []

        class MockAskHandle:
            async def result(self):
                return "mock answer"

        async def capturing_ask(question, **kwargs):
            captured_ctx.append(kwargs.get("_parent_chat_context"))
            return MockAskHandle()

        mock_handle = MagicMock()
        mock_handle.ask = capturing_ask

        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "research task",
            "handle_actions": [],
            "context_opted_in": True,
        }

        test_snapshot = {
            "role": "user",
            "content": "<state>conversation context</state>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["ask_action"](
                handle_id=0,
                question="what's the status?",
            )

            pending = set(cm._pending_steering_tasks)
            if pending:
                await asyncio.wait(pending, timeout=300)

            assert len(captured_ctx) == 1
            assert captured_ctx[0] is not None
            assert len(captured_ctx[0]) == 1
            assert captured_ctx[0][0] is test_snapshot

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None
