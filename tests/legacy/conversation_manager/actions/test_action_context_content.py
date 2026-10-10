"""
tests/conversation_manager/actions/test_action_context_content.py
===================================================================

Tests that verify ConversationManager correctly propagates parent chat context
to the Actor when calling `act` and when using steering tools.

These tests verify:
1. `act()` passes the fresh rendered state snapshot as `_parent_chat_context`
2. `ask` steering passes the fresh snapshot as `_parent_chat_context`
3. `interject` steering passes the fresh snapshot as `_parent_chat_context_cont`
4. The snapshot content is correct and would be legible to an LLM

Test Design:
- Symbolic tests: Directly call tools with fixed inputs, capture arguments
- Content tests: Verify the captured context contains expected information

All tests bypass the CM brain LLM to ensure determinism.
"""

import pytest

from tests.helpers import _handle_project
from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
)

pytestmark = pytest.mark.eval


# =============================================================================
# Symbolic Tests: Verify parameters are passed correctly
# =============================================================================


class TestActContextPropagation:
    """Tests that verify `act()` passes the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that act() passes [_current_state_snapshot] as _parent_chat_context.

        Setup:
        - Set _current_state_snapshot to a known message
        - Call act() directly via brain_action_tools
        - Capture the _parent_chat_context argument passed to actor.act()

        Assert:
        - _parent_chat_context is a list with one element
        - That element has content derived from _current_state_snapshot
          (may be a filtered copy to remove CM-internal steering tools)
        """
        cm = initialized_cm.cm

        # Create a known state snapshot (without in_flight_actions, so content unchanged)
        test_snapshot = {
            "role": "user",
            "content": "<notifications></notifications>\n<test>unique_test_content_12345</test>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        # Capture what's passed to actor.act()
        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            # Return a mock handle that won't block
            handle = await original_act(query, **kwargs)
            return handle

        cm.actor.act = capturing_act

        try:
            # Create brain action tools and call act directly
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(query="test query")

            # Verify context was captured
            assert len(captured_context) == 1, "act() should have been called once"
            context = captured_context[0]

            # Verify structure
            assert context is not None, "_parent_chat_context should not be None"
            assert isinstance(context, list), "_parent_chat_context should be a list"
            assert (
                len(context) == 1
            ), "_parent_chat_context should have exactly one element"

            # Verify content matches (filtering may produce a new dict, so compare content)
            # The content should be equivalent since this snapshot has no in_flight_actions
            assert context[0].get("content") == test_snapshot.get(
                "content",
            ), "_parent_chat_context content should match the original snapshot"
            assert (
                context[0].get("_cm_state_snapshot") is True
            ), "_parent_chat_context should preserve _cm_state_snapshot marker"

        finally:
            cm.actor.act = original_act
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_passes_none_when_no_snapshot(self, initialized_cm):
        """
        Verify that act() passes None when _current_state_snapshot is not set.
        """
        cm = initialized_cm.cm

        # Ensure no snapshot is set
        cm._current_state_snapshot = None

        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            handle = await original_act(query, **kwargs)
            return handle

        cm.actor.act = capturing_act

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(query="test query")

            assert len(captured_context) == 1
            assert (
                captured_context[0] is None
            ), "_parent_chat_context should be None when no snapshot is set"

        finally:
            cm.actor.act = original_act


class TestActStartedInHistory:
    """Tests that act() logs a timestamped 'started' event in handle_actions."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_logs_started_event_with_timestamp(self, initialized_cm):
        """
        Calling act() should append an 'act_started' event (with a timestamp)
        to handle_actions so the rendered <history> includes an entry for when
        the action was created.
        """
        cm = initialized_cm.cm

        brain_tools = ConversationManagerBrainActionTools(cm)
        await brain_tools.act(
            query="Find every CSV in the workspace",
        )

        # Exactly one in-flight action should exist.
        assert len(cm.in_flight_actions) == 1
        handle_data = next(iter(cm.in_flight_actions.values()))
        actions = handle_data["handle_actions"]

        # The first entry should record that the action was started.
        assert (
            len(actions) >= 1
        ), "handle_actions should contain at least the 'started' event"
        started = actions[0]
        assert (
            "started" in started.get("action_name", "").lower()
        ), f"First handle_actions entry should be a 'started' event, got: {started}"
        assert started.get(
            "timestamp",
        ), f"'started' event must have a timestamp, got: {started}"

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_started_appears_in_rendered_state(self, initialized_cm):
        """
        The rendered in-flight actions should include a <history> section
        containing the timestamped 'started' event from the act() call.
        """
        from unify.legacy.conversation_manager.domains.renderer import Renderer

        cm = initialized_cm.cm

        brain_tools = ConversationManagerBrainActionTools(cm)
        await brain_tools.act(
            query="Search the web for weather",
        )

        renderer = Renderer()
        snapshot = renderer.render_state(
            cm.chat_history,
            in_flight_actions=cm.in_flight_actions,
        )
        rendered = snapshot.full_render

        assert (
            "<history>" in rendered
        ), "Rendered state should contain <history> for the in-flight action"
        assert (
            "started" in rendered.lower()
        ), "Rendered <history> should contain a 'started' event"
        assert (
            "timestamp=" in rendered
        ), "Rendered 'started' event should include a timestamp attribute"


class TestActCompletionInHistory:
    """Tests that ActorResult logs a timestamped 'completed' event in handle_actions."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_actor_result_logs_completed_event_with_timestamp(
        self,
        initialized_cm,
    ):
        """
        When an ActorResult event is processed, it should append a timestamped
        'completed' event to handle_actions before moving the action to
        completed_actions. This gives the LLM a full lifecycle timeline
        (started → … → completed) in the rendered history.
        """
        from unify.legacy.conversation_manager.domains.event_handlers import (
            EventHandler,
        )
        from unify.legacy.conversation_manager.events import ActorResult

        cm = initialized_cm.cm

        brain_tools = ConversationManagerBrainActionTools(cm)
        await brain_tools.act(
            query="Find every CSV in the workspace",
        )

        assert len(cm.in_flight_actions) == 1
        handle_id = next(iter(cm.in_flight_actions))

        # Simulate actor completion by dispatching an ActorResult event
        # through the real event handler (same path as production).
        result_event = ActorResult(
            handle_id=handle_id,
            success=True,
            result="Found 3 CSV files in the workspace.",
        )
        await EventHandler.handle_event(result_event, cm)

        # The action should now be in completed_actions.
        assert (
            handle_id in cm.completed_actions
        ), "Action should move to completed_actions after ActorResult"
        actions = cm.completed_actions[handle_id]["handle_actions"]

        # Find a completion event in the history.
        completed_events = [
            a
            for a in actions
            if "completed" in a.get("action_name", "").lower()
            or "result" in a.get("action_name", "").lower()
        ]
        assert (
            len(completed_events) >= 1
        ), f"handle_actions should contain a completion event, got: {actions}"
        completed = completed_events[0]
        assert completed.get(
            "timestamp",
        ), f"Completion event must have a timestamp, got: {completed}"


class TestSteeringContextPropagation:
    """Tests that verify steering tools pass the correct parent context."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_passes_incremental_diff_when_snapshot_state_available(
        self,
        initialized_cm,
    ):
        """
        Verify that interject steering passes an incremental diff as
        _parent_chat_context_cont when SnapshotState tracking is available.

        This tests the incremental context propagation feature: when an action
        is started, its initial snapshot state is stored. When interjecting,
        only the DIFF between the initial and current snapshot is sent.

        Setup:
        - Create an in-flight action with a mock handle and initial_snapshot_state
        - Set _current_snapshot_state with some new elements (a new message)
        - Call the interject steering tool directly
        - Capture the _parent_chat_context_cont argument

        Assert:
        - _parent_chat_context_cont contains only the diff (new elements)
        - _parent_chat_context_cont is marked with _cm_context_diff: True
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from unify.legacy.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        # Create a mock handle that captures interject calls
        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Create initial snapshot state (at time of act())
        initial_ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        initial_snapshot_state = SnapshotState(
            full_render="<initial>state</initial>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=initial_ts,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        # Register the mock handle as an in-flight action WITH initial snapshot state
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            "initial_snapshot_state": initial_snapshot_state,
        }

        # Create current snapshot state with a NEW message (simulating state change)
        new_ts = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)
        current_snapshot_state = SnapshotState(
            full_render="<current>state with new message</current>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=initial_ts,
                    rendered="[User @ ...]: Hello",
                ),
                MessageElement(
                    index_in_conversation=1,
                    timestamp=new_ts,
                    rendered="[User @ ...]: Please also check the calendar",
                ),
            ],
        )
        cm._current_snapshot_state = current_snapshot_state

        try:
            # Build the fixed-name steering tools and interject by handle_id
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()
            interject_tool = steering_tools["interject_action"]

            # Call the interject tool
            await interject_tool(handle_id=0, message="additional instruction")

            # Verify context was captured
            assert len(captured_context) == 1, "interject should have been called once"
            context = captured_context[0]

            assert context is not None, "_parent_chat_context_cont should not be None"
            assert isinstance(
                context,
                list,
            ), "_parent_chat_context_cont should be a list"
            assert (
                len(context) == 1
            ), "_parent_chat_context_cont should have one element"

            # Verify it's marked as a diff
            assert (
                context[0].get("_cm_context_diff") is True
            ), "Context should be marked as a diff"

            # Verify the diff contains only the NEW message, not the old one
            diff_content = context[0].get("content", "")
            assert (
                "Please also check the calendar" in diff_content
            ), "Diff should contain the new message"
            assert (
                "<new_messages>" in diff_content
            ), "Diff should have <new_messages> section"
            # The original message should NOT be in the diff
            assert (
                "Hello" not in diff_content
            ), "Diff should NOT contain the original message"

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_returns_empty_diff_when_nothing_changed(
        self,
        initialized_cm,
    ):
        """
        Verify that interject passes None when there's no diff (nothing changed).

        When the initial snapshot and current snapshot are identical, there's
        no incremental update to send.
        """
        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from unify.legacy.conversation_manager.domains.renderer import (
            SnapshotState,
            MessageElement,
        )

        cm = initialized_cm.cm

        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Create identical initial and current snapshots
        ts = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        snapshot_state = SnapshotState(
            full_render="<state>unchanged</state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            "initial_snapshot_state": snapshot_state,
        }
        cm._current_snapshot_state = snapshot_state

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["interject_action"](
                handle_id=0,
                message="additional instruction",
            )

            assert len(captured_context) == 1
            # When nothing changed, context should be None (empty diff)
            assert (
                captured_context[0] is None
            ), "When nothing changed, _parent_chat_context_cont should be None"

        finally:
            cm.in_flight_actions.clear()
            cm._current_snapshot_state = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_interject_falls_back_to_full_snapshot_when_no_tracking(
        self,
        initialized_cm,
    ):
        """
        Verify backward compatibility: when no snapshot tracking is available,
        interject falls back to using the full _current_state_snapshot.
        """
        from unittest.mock import MagicMock

        cm = initialized_cm.cm

        captured_context = []

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))

        mock_handle = MagicMock()
        mock_handle.interject = capturing_interject

        # Register WITHOUT initial_snapshot_state (no tracking)
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "search for test data",
            "handle_actions": [],
            # No initial_snapshot_state
        }

        # Set full snapshot (old behavior)
        test_snapshot = {
            "role": "user",
            "content": "<full_state>test content</full_state>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot
        cm._current_snapshot_state = None  # No tracking

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()

            await steering_tools["interject_action"](
                handle_id=0,
                message="additional instruction",
            )

            assert len(captured_context) == 1
            context = captured_context[0]

            # Should fall back to full snapshot
            assert context is not None
            assert len(context) == 1
            assert context[0] is test_snapshot

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None

    @pytest.mark.asyncio
    @_handle_project
    async def test_ask_passes_current_state_snapshot(self, initialized_cm):
        """
        Verify that ask steering passes [_current_state_snapshot] as
        _parent_chat_context.

        Setup:
        - Create an in-flight action with a mock handle
        - Set _current_state_snapshot to a known message
        - Call the ask steering tool directly
        - Capture the _parent_chat_context argument

        Assert:
        - _parent_chat_context is [_current_state_snapshot]
        """
        import asyncio
        from unittest.mock import MagicMock

        cm = initialized_cm.cm

        # Create a mock handle that captures ask calls
        captured_context = []
        captured_questions = []

        class MockAskHandle:
            async def result(self):
                return "mock answer"

        async def capturing_ask(question, **kwargs):
            captured_questions.append(question)
            captured_context.append(kwargs.get("_parent_chat_context"))
            return MockAskHandle()

        mock_handle = MagicMock()
        mock_handle.ask = capturing_ask

        # Register the mock handle as an in-flight action
        cm.in_flight_actions[0] = {
            "handle": mock_handle,
            "query": "research competitor pricing",
            "handle_actions": [],
        }

        # Set a known state snapshot
        test_snapshot = {
            "role": "user",
            "content": "<active_conversations>test conversation</active_conversations>",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        try:
            brain_tools = ConversationManagerBrainActionTools(cm)
            steering_tools = brain_tools.build_action_steering_tools()
            ask_tool = steering_tools["ask_action"]

            # Call the ask tool (spawns a background steering task)
            await ask_tool(handle_id=0, question="what's the status?")

            # Await the background task so the mock ask() has executed
            pending = set(cm._pending_steering_tasks)
            if pending:
                await asyncio.wait(pending, timeout=300)

            # Verify context was captured
            assert len(captured_context) == 1, "ask should have been called once"
            assert captured_questions == [
                "what's the status?",
            ], "ask should receive the question text from steering tool args"
            context = captured_context[0]

            assert context is not None, "_parent_chat_context should not be None"
            assert isinstance(context, list), "_parent_chat_context should be a list"
            assert len(context) == 1, "_parent_chat_context should have one element"
            assert (
                context[0] is test_snapshot
            ), "_parent_chat_context[0] should be the exact snapshot"

        finally:
            cm.in_flight_actions.clear()
            cm._current_state_snapshot = None


# =============================================================================
# Content Tests: Verify context content is correct
# =============================================================================


class TestContextContent:
    """Tests that verify the context content is correct and legible."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_contains_conversation_from_message(self, initialized_cm):
        """
        Verify that when a message triggers act(), the context passed to the Actor
        contains that message's content.

        This tests that context is fresh and includes the current turn.
        """
        from unify.legacy.conversation_manager.events import UnifyMessageReceived

        cm_driver = initialized_cm
        cm = cm_driver.cm

        # Track what context is passed to actor.act()
        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            return await original_act(query, **kwargs)

        cm.actor.act = capturing_act

        try:
            # Send a message with distinctive content
            await cm_driver.step_until_wait(
                UnifyMessageReceived(
                    content="Please find the XYZZY123 document for me.",
                ),
            )

            # Check if act was called (it might not be depending on LLM decision)
            if captured_context and captured_context[0]:
                context = captured_context[0]
                content = context[0].get("content", "")

                # The context should contain the message that triggered the action
                assert "XYZZY123" in content, (
                    "Context should contain the distinctive message content. "
                    f"Got content: {content[:500]}..."
                )

                # Should be marked as a state snapshot
                assert (
                    context[0].get("_cm_state_snapshot") is True
                ), "Context should be marked as a state snapshot"

        finally:
            cm.actor.act = original_act

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_has_expected_xml_sections(self, initialized_cm):
        """
        Verify that the rendered state snapshot has the expected XML structure.

        This ensures the context is "legible" to an LLM - it has proper structure.
        """
        cm = initialized_cm.cm

        # Create a snapshot via the normal rendering path
        from unify.legacy.conversation_manager.domains.brain import build_brain_spec

        snapshot_state = cm.prompt_renderer.render_state(
            cm.chat_history,
            cm.notifications_bar,
            cm.in_flight_actions,
            cm.completed_actions,
            last_snapshot=cm.last_snapshot,
        )
        brain_spec = build_brain_spec(cm, snapshot_state=snapshot_state)
        state_content = brain_spec.state_prompt

        # Verify expected sections exist
        assert "<notifications>" in state_content, "Should have notifications section"
        assert (
            "<in_flight_actions>" in state_content
        ), "Should have in_flight_actions section"
        assert "<conversation>" in state_content, "Should have conversation section"

    @pytest.mark.asyncio
    @_handle_project
    async def test_context_shows_in_flight_action_when_steering(self, initialized_cm):
        """
        Verify that when steering an in-flight action, the context shows that action.

        With incremental context propagation, the interject receives a diff that may
        include <action_updates> (when action state changed) or the action may be
        part of the initial snapshot (sent with the original act() call).

        This tests that the context is fresh and includes relevant action information.
        """
        from unify.legacy.conversation_manager.events import (
            UnifyMessageReceived,
            ActorHandleStarted,
        )
        from tests.legacy.conversation_manager.cm_helpers import filter_events_by_type

        cm_driver = initialized_cm
        cm = cm_driver.cm

        # Step 1: Start an action
        result1 = await cm_driver.step_until_wait(
            UnifyMessageReceived(
                content="Search my workspace for the engineering notes.",
            ),
        )

        # Check if an action was started
        actor_events = filter_events_by_type(result1.output_events, ActorHandleStarted)
        if not actor_events:
            pytest.skip("LLM did not trigger act - cannot test steering context")

        assert len(cm.in_flight_actions) >= 1, "Should have an in-flight action"

        # Step 2: Capture context when steering
        captured_context = []
        handle_id = list(cm.in_flight_actions.keys())[0]
        handle = cm.in_flight_actions[handle_id]["handle"]

        original_interject = handle.interject

        async def capturing_interject(message, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context_cont"))
            return await original_interject(message, **kwargs)

        handle.interject = capturing_interject

        try:
            # Send a message that should trigger interject
            await cm_driver.step_until_wait(
                UnifyMessageReceived(
                    content="Also include the file sizes.",
                ),
            )

            # If interject was called, verify the context shows relevant information
            if captured_context and captured_context[0]:
                content = captured_context[0][0].get("content", "")

                # With incremental diffs, the action may appear in <action_updates>
                # (if its state changed) or the full snapshot (backward compat mode).
                # Either format is acceptable as long as action info is present.
                has_action_info = (
                    "<action_updates>" in content
                    or "<in_flight_actions>" in content
                    or "action id='0'" in content
                )
                assert (
                    has_action_info
                ), f"Should have action information in context. Got: {content[:500]}..."

                # The action should be visible as executing
                assert (
                    "executing" in content.lower() or "search" in content.lower()
                ), "Context should show the in-flight action"

        finally:
            handle.interject = original_interject

    @pytest.mark.asyncio
    @_handle_project
    async def test_act_context_excludes_cm_steering_tools(self, initialized_cm):
        """
        Verify that context passed to Actor does NOT contain CM-level steering tools.

        The CM state snapshot includes <in_flight_actions> with <steering_tools>
        listing CM-level tools (stop_, pause_, interject_, ask_) for each action.
        These exist only in the CM's tool surface, not in the Actor's scope.

        If passed verbatim, the Actor LLM may interpret these as callable functions
        and generate code like `await stop_search_the_web_for__1()`, which would
        raise NameError.

        This test verifies that in_flight_actions are filtered before passing
        the state snapshot to the Actor as parent context.
        """
        cm = initialized_cm.cm

        # State snapshot containing in_flight_actions with steering tools
        test_snapshot = {
            "role": "user",
            "content": """<notifications>
[PINNED] Previous action completed successfully.
</notifications>

<in_flight_actions>
<action id='1' short_name='search_the_web_for' status='executing'>
<original_request>Search the web for backend engineer job openings at OpenAI.</original_request>
<steering_tools>
  - ask_search_the_web_for__1: Query the status or progress of this running task.
  - stop_search_the_web_for__1: Stop this task immediately, cancelling any pending work.
  - interject_search_the_web_for__1: Provide additional information or instructions.
  - pause_search_the_web_for__1: Pause this task temporarily without cancelling it.
</steering_tools>
</action>
</in_flight_actions>

<conversation>
[User @ Wednesday, February 04, 2026]: Search for frontend engineer roles instead.
</conversation>""",
            "_cm_state_snapshot": True,
        }
        cm._current_state_snapshot = test_snapshot

        # Capture what's passed to actor.act()
        captured_context = []
        original_act = cm.actor.act

        async def capturing_act(query, **kwargs):
            captured_context.append(kwargs.get("_parent_chat_context"))
            handle = await original_act(query, **kwargs)
            return handle

        cm.actor.act = capturing_act

        try:
            # Create brain action tools and call act directly
            brain_tools = ConversationManagerBrainActionTools(cm)
            await brain_tools.act(
                query="Search for frontend engineer roles",
            )

            # Verify context was captured
            assert len(captured_context) == 1, "act() should have been called once"
            context = captured_context[0]

            assert context is not None, "_parent_chat_context should not be None"
            assert len(context) == 1, "_parent_chat_context should have one element"

            content = context[0].get("content", "")

            # Verify CM steering tools are filtered out
            cm_steering_patterns = [
                "stop_search_the_web_for__1",
                "pause_search_the_web_for__1",
                "interject_search_the_web_for__1",
                "ask_search_the_web_for__1",
                "<steering_tools>",
            ]

            for pattern in cm_steering_patterns:
                assert pattern not in content, (
                    f"CM steering tool '{pattern}' should NOT be in Actor context. "
                    f"These are CM-level tools that don't exist in Actor scope. "
                    f"Content excerpt: {content[:1000]}..."
                )

            # Verify useful context IS preserved (conversation content)
            assert (
                "frontend engineer" in content.lower() or "<conversation>" in content
            ), "Useful conversation context should still be present"

        finally:
            cm.actor.act = original_act
            cm._current_state_snapshot = None
