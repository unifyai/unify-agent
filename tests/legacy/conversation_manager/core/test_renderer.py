"""
tests/conversation_manager/core/test_renderer.py
================================================

Unit tests for the Renderer class in `domains/renderer.py`.

These are symbolic tests that verify rendering logic without invoking the LLM.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from unify.legacy.conversation_manager.domains.chat_history import (
    ChatHistory,
    ChatMessage,
)
from unify.legacy.conversation_manager.domains.notifications import (
    NotificationBar,
    Notification,
)
from unify.legacy.conversation_manager.domains.renderer import (
    Renderer,
    SnapshotState,
    MessageElement,
    NotificationElement,
    ActionElement,
    compute_snapshot_diff,
    user_display_name,
)
from unify.session_details import SESSION_DETAILS

pytestmark = pytest.mark.no_unify_context


@pytest.fixture
def renderer():
    """Create a Renderer instance."""
    return Renderer()


@pytest.fixture
def user_name(monkeypatch):
    """Pin the user's name so message lines render deterministically."""
    monkeypatch.setattr(SESSION_DETAILS.user, "first_name", "Dana")
    monkeypatch.setattr(SESSION_DETAILS.user, "surname", "Owner")
    return "Dana Owner"


def _message(role: str, content: str, ts: datetime, attachments=None) -> ChatMessage:
    return ChatMessage(
        role=role,
        content=content,
        timestamp=ts,
        attachments=list(attachments or []),
    )


# =============================================================================
# Tests for message rendering
# =============================================================================


class TestRendererMessage:
    """Tests for one conversation line."""

    def test_render_incoming_message_shows_user_name(self, renderer, user_name):
        """An inbound message is attributed to the user by name."""
        message = _message(
            "user",
            "Please summarise the report.",
            datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[Dana Owner @" in result
        assert "Please summarise the report." in result
        assert "**NEW**" in result

    def test_render_outgoing_message_shows_you(self, renderer, user_name):
        """An assistant message shows 'You' as the sender."""
        message = _message(
            "assistant",
            "Done, I've summarised the report.",
            datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc),
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "[You @" in result
        assert "Done, I've summarised the report." in result

    def test_old_message_has_no_new_marker(self, renderer, user_name):
        """A message older than the last snapshot renders without **NEW**."""
        message = _message(
            "user",
            "Old news.",
            datetime(2025, 6, 13, 10, 0, 0, tzinfo=timezone.utc),
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        assert "**NEW**" not in renderer.render_message(message, last_snapshot)

    def test_render_message_with_attachments(self, renderer, user_name):
        """Attachments render as workspace paths so the Actor can open them."""
        message = _message(
            "user",
            "Here's the document.",
            datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc),
            attachments=["Attachments/report.pdf", "Attachments/data.xlsx"],
        )
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_message(message, last_snapshot)

        assert "Here's the document." in result
        assert "[Attachments: Attachments/report.pdf, Attachments/data.xlsx]" in result

    def test_user_display_name_falls_back_to_placeholder(self, monkeypatch):
        """With no user name configured the placeholder first name is used."""
        monkeypatch.setattr(SESSION_DETAILS.user, "first_name", "")
        monkeypatch.setattr(SESSION_DETAILS.user, "surname", "")
        assert user_display_name() == "Default"


# =============================================================================
# Tests for Incremental Diff
# =============================================================================


class TestComputeSnapshotDiff:
    """Tests for compute_snapshot_diff incremental diff computation."""

    def test_diff_returns_full_render_when_old_is_none(self):
        """When old_snapshot is None, returns full new snapshot."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        new_snapshot = SnapshotState(
            full_render="<full_state>content</full_state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        diff = compute_snapshot_diff(None, new_snapshot)
        assert diff == "<full_state>content</full_state>"

    def test_diff_returns_empty_when_nothing_changed(self):
        """When snapshots are identical, returns empty string."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>same</state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>same</state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert diff == ""

    def test_diff_includes_new_messages(self):
        """New messages are included in <new_messages> section."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            messages=[
                MessageElement(
                    index_in_conversation=0,
                    timestamp=ts1,
                    rendered="[User @ ...]: Hello",
                ),
                MessageElement(
                    index_in_conversation=1,
                    timestamp=ts2,
                    rendered="[User @ ...]: Please help me",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<new_messages>" in diff
        assert "[User @ ...]: Please help me" in diff
        assert "[User @ ...]: Hello" not in diff  # Old message not in diff

    def test_diff_includes_new_notifications(self):
        """New notifications are included in <new_notifications> section."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 6, 13, 12, 5, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Action started"),
                    pinned=False,
                    rendered="[Action Notification] Action started",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Action started"),
                    pinned=False,
                    rendered="[Action Notification] Action started",
                ),
                NotificationElement(
                    timestamp=ts2,
                    content_hash=hash("Action completed"),
                    pinned=False,
                    rendered="[Action Notification] Action completed",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<new_notifications>" in diff
        assert "Action completed" in diff
        assert "Action started" not in diff  # Old notification not in diff

    def test_diff_includes_action_state_changes(self):
        """Action state changes are included in <action_updates> section."""
        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search the workspace",
                    status="executing",
                    history_count=0,
                    rendered="<action id='0' status='executing'>...",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search the workspace",
                    status="executing",
                    history_count=1,  # History count changed (new event)
                    rendered="<action id='0' status='executing'>new history event...",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<action_updates>" in diff
        assert "new history event" in diff

    def test_diff_includes_new_actions(self):
        """New actions (not in old snapshot) are included in diff."""
        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            actions=[],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            actions=[
                ActionElement(
                    handle_id=0,
                    query="search the workspace",
                    status="executing",
                    history_count=0,
                    rendered="<action id='0'>search the workspace...",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        assert "<action_updates>" in diff
        assert "search the workspace" in diff

    def test_diff_tracks_notification_pinned_state_change(self):
        """Notification pinned state change is detected as a new notification."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)

        old_snapshot = SnapshotState(
            full_render="<state>old</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Important reminder"),
                    pinned=False,  # Not pinned
                    rendered="[Notification] Important reminder",
                ),
            ],
        )
        new_snapshot = SnapshotState(
            full_render="<state>new</state>",
            notifications=[
                NotificationElement(
                    timestamp=ts1,
                    content_hash=hash("Important reminder"),
                    pinned=True,  # Now pinned
                    rendered="[PINNED][Notification] Important reminder",
                ),
            ],
        )

        diff = compute_snapshot_diff(old_snapshot, new_snapshot)
        # The pinned=True version has a different identity tuple
        assert "<new_notifications>" in diff
        assert "[PINNED]" in diff


class TestRenderStateWithTracking:
    """Tests for render_state with element tracking."""

    @pytest.fixture
    def chat_history(self):
        """An unbound (in-memory only) chat history."""
        return ChatHistory()

    @pytest.fixture
    def notification_bar(self):
        """Create a NotificationBar."""
        return NotificationBar()

    def test_returns_snapshot_state_with_full_render(
        self,
        renderer,
        chat_history,
        notification_bar,
    ):
        """render_state returns SnapshotState with full_render."""
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert isinstance(result, SnapshotState)
        assert result.full_render is not None
        assert "<notifications>" in result.full_render
        assert "<in_flight_actions>" in result.full_render
        assert "<conversation>" in result.full_render

    def test_snapshot_ends_with_the_current_time_pane(
        self,
        renderer,
        chat_history,
        notification_bar,
    ):
        """The wall clock closes the snapshot, not the system prompt.

        Keeping the clock out of the system prompt keeps that prompt
        byte-stable across minute rollovers (the provider cache is
        all-or-nothing over system+tools); the snapshot tail is the one
        place a per-turn timestamp is cheap.
        """
        from unify.common.prompt_helpers import now

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert result.full_render.split("\n\n")[-1] == f"Current time: {now()}."

    def test_tracks_messages_in_conversation(
        self,
        renderer,
        chat_history,
        notification_bar,
        user_name,
    ):
        """Messages in the conversation are tracked with identity."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        chat_history.append(role="user", content="Hello there!", timestamp=ts1)

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert len(result.messages) == 1
        msg = result.messages[0]
        assert "Hello there!" in msg.rendered
        assert "[Dana Owner @" in msg.rendered
        assert msg.index_in_conversation == 0
        assert msg.timestamp == ts1

    def test_conversation_is_capped_to_the_most_recent_messages(
        self,
        renderer,
        chat_history,
        notification_bar,
        user_name,
    ):
        """Only the last ``max_messages`` messages render, oldest dropped first."""
        base = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            chat_history.append(
                role="user",
                content=f"message_{i}",
                timestamp=base.replace(minute=i),
            )

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            last_snapshot=datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc),
            max_messages=3,
        )

        for i in range(7):
            assert f"message_{i}" not in result.full_render
        for i in range(7, 10):
            assert f"message_{i}" in result.full_render
        assert [m.index_in_conversation for m in result.messages] == [7, 8, 9]

    def test_render_state_includes_recent_tool_executions(
        self,
        renderer,
        chat_history,
        notification_bar,
    ):
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            recent_tool_executions=[
                {
                    "generation": 2,
                    "origin_event_name": "UnifyMessageReceived",
                    "tool_name": "act",
                    "args_preview": '{"query":"Summarise the report"}',
                    "result_preview": '{"status":"acting"}',
                },
            ],
            last_snapshot=last_snapshot,
        )

        assert "<recent_tool_executions>" in result.full_render
        assert "tool=act" in result.full_render
        assert "origin=UnifyMessageReceived" in result.full_render

    def test_tracks_notifications(self, renderer, chat_history, notification_bar):
        """Notifications are tracked with identity."""
        ts1 = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        notification_bar.notifications.append(
            Notification(
                type="action",
                content="Action completed successfully",
                timestamp=ts1,
                pinned=False,
            ),
        )

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            last_snapshot=last_snapshot,
        )

        assert len(result.notifications) == 1
        notif = result.notifications[0]
        assert notif.timestamp == ts1
        assert notif.pinned is False
        assert "Action completed" in notif.rendered

    def test_tracks_in_flight_actions(self, renderer, chat_history, notification_bar):
        """In-flight actions are tracked with identity."""
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True  # Not paused

        in_flight_actions = {
            0: {
                "handle": mock_handle,
                "query": "Summarise the quarterly report",
                "handle_actions": [
                    {"action_name": "interject_0", "query": "also chart it"},
                ],
            },
        }

        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions=in_flight_actions,
            last_snapshot=last_snapshot,
        )

        assert len(result.actions) == 1
        action = result.actions[0]
        assert action.handle_id == 0
        assert action.query == "Summarise the quarterly report"
        assert action.status == "executing"
        assert action.history_count == 1


# =============================================================================
# Tests for Render Caps
# =============================================================================


class TestRenderCaps:
    """Tests verifying that all render caps are respected."""

    def test_pinned_notifications_capped(self, renderer):
        """Only the most recent max_pinned pinned notifications are rendered;
        transient notifications are unaffected by the cap."""
        bar = NotificationBar()
        for i in range(10):
            bar.push_notif(
                "Action",
                f"pinned_{i}",
                datetime(2025, 6, 13, 12, i, 0, tzinfo=timezone.utc),
                pinned=True,
            )
        # Add transient notifications (newer than last_snapshot)
        for i in range(3):
            bar.push_notif(
                "Action",
                f"transient_{i}",
                datetime(2025, 6, 13, 12, 30, i, tzinfo=timezone.utc),
                pinned=False,
            )

        last_snapshot = datetime(2025, 6, 13, 12, 0, 0, tzinfo=timezone.utc)
        result = renderer.render_notification_bar(
            bar,
            last_snapshot=last_snapshot,
            max_pinned=5,
        )

        # 5 most recent pinned present, 5 oldest absent
        for i in range(5, 10):
            assert f"pinned_{i}" in result
        for i in range(5):
            assert f"pinned_{i}" not in result
        # All transient notifications still present
        for i in range(3):
            assert f"transient_{i}" in result

    def test_in_flight_action_history_capped(self, renderer):
        """Only the most recent max_history events are rendered per in-flight action."""
        mock_handle = MagicMock()
        mock_handle._pause_event = MagicMock()
        mock_handle._pause_event.is_set.return_value = True

        handle_actions = [
            {"action_name": f"event_{i}", "query": f"content_{i}"} for i in range(10)
        ]

        in_flight = {
            0: {
                "handle": mock_handle,
                "query": "Test action",
                "handle_actions": handle_actions,
            },
        }

        result = renderer.render_in_flight_actions(in_flight, max_history=3)

        # Only the 3 most recent events should appear
        for i in range(7):
            assert f"content_{i}" not in result
        for i in range(7, 10):
            assert f"content_{i}" in result

    def test_completed_actions_count_capped(self, renderer):
        """Only the most recent max_completed completed actions are rendered."""
        completed = {
            i: {
                "handle": MagicMock(),
                "query": f"action_{i}",
                "handle_actions": [
                    {"action_name": "act_completed", "query": f"result_{i}"},
                ],
            }
            for i in range(10)
        }

        result = renderer.render_completed_actions(completed, max_completed=3)

        # Only the 3 most recent (7, 8, 9) should appear
        for i in range(7):
            assert f"action_{i}" not in result
        for i in range(7, 10):
            assert f"action_{i}" in result

    def test_completed_action_history_capped(self, renderer):
        """Only the most recent max_history events are rendered per completed action."""
        completed = {
            0: {
                "handle": MagicMock(),
                "query": "Test action",
                "handle_actions": [
                    {"action_name": f"step_{i}", "query": f"detail_{i}"}
                    for i in range(10)
                ]
                + [
                    {"action_name": "act_completed", "query": "final result"},
                ],
            },
        }

        result = renderer.render_completed_actions(
            completed,
            max_completed=20,
            max_history=3,
        )

        # Only the 3 most recent history events should appear
        # (the last 2 steps + act_completed)
        for i in range(8):
            assert f"detail_{i}" not in result
        for i in range(8, 10):
            assert f"detail_{i}" in result
        assert "final result" in result


# =============================================================================
# Tests for Completed Actions Rendering
# =============================================================================


class TestRenderCompletedActions:
    """Tests for render_completed_actions method."""

    def test_empty_completed_actions(self, renderer):
        """Empty completed_actions renders placeholder text."""
        result = renderer.render_completed_actions({})
        assert "<completed_actions>" in result
        assert "No completed actions." in result
        assert "</completed_actions>" in result

    def test_none_completed_actions(self, renderer):
        """None completed_actions renders placeholder text."""
        result = renderer.render_completed_actions(None)
        assert "<completed_actions>" in result
        assert "No completed actions." in result
        assert "</completed_actions>" in result

    def test_single_completed_action(self, renderer):
        """Single completed action renders with ask steering tool only."""
        completed_actions = {
            0: {
                "handle": MagicMock(),
                "query": "Find every CSV in the workspace",
                "handle_actions": [],
            },
        }

        result = renderer.render_completed_actions(completed_actions)

        assert "<completed_actions>" in result
        assert "</completed_actions>" in result
        assert "id='0'" in result
        assert "status='completed'" in result
        assert "Find every CSV in the workspace" in result
        # Should have ask steering tool (not close, stop, pause, resume, interject)
        assert "ask_" in result
        assert "close_" not in result
        assert "stop_" not in result
        assert "pause_" not in result
        assert "resume_" not in result
        assert "interject_" not in result

    def test_multiple_completed_actions(self, renderer):
        """Multiple completed actions render correctly."""
        completed_actions = {
            0: {
                "handle": MagicMock(),
                "query": "Search the web for AI news",
                "handle_actions": [],
            },
            1: {
                "handle": MagicMock(),
                "query": "Summarise the attached report",
                "handle_actions": [],
            },
        }

        result = renderer.render_completed_actions(completed_actions)

        assert "<completed_actions>" in result
        assert "</completed_actions>" in result
        assert "id='0'" in result
        assert "id='1'" in result
        assert "Search the web for AI news" in result
        assert "Summarise the attached report" in result

    def test_failed_completed_action_renders_error_state(self, renderer):
        """Failed actions render explicit failed status and error text."""
        completed_actions = {
            0: {
                "handle": MagicMock(),
                "query": "Merge the two spreadsheets",
                "action_type": "act",
                "handle_actions": [
                    {
                        "action_name": "act_failed",
                        "query": "File 7 not found",
                        "success": False,
                        "error": "File 7 not found",
                    },
                ],
            },
        }

        result = renderer.render_completed_actions(completed_actions)

        assert "status='failed'" in result
        assert "<error>File 7 not found</error>" in result

    def test_render_state_includes_completed_actions(self, renderer):
        """render_state includes completed_actions section."""
        chat_history = ChatHistory()
        notification_bar = NotificationBar()
        completed_actions = {
            0: {
                "handle": MagicMock(),
                "query": "Test completed action",
                "handle_actions": [],
            },
        }
        last_snapshot = datetime(2025, 6, 13, 11, 0, 0, tzinfo=timezone.utc)

        result = renderer.render_state(
            chat_history,
            notification_bar,
            in_flight_actions={},
            completed_actions=completed_actions,
            last_snapshot=last_snapshot,
        )

        assert "<completed_actions>" in result.full_render
        assert "Test completed action" in result.full_render
