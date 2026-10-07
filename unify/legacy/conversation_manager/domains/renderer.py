"""
Renderer: Renders conversation state for the ConversationManager LLM.

The conversation comes from ChatHistory; notifications and actions from the
ConversationManager's live state.

SnapshotState: Tracks constituent elements of a rendered snapshot with identity,
enabling incremental diff computation for context propagation to Actor interjections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any

from unify.common._async_tool.utils import get_handle_paused_state
from unify.common.startup_timing import log_startup_timing
from unify.legacy.conversation_manager.domains.chat_history import (
    ChatHistory,
    ChatMessage,
)
from unify.legacy.conversation_manager.domains.notifications import NotificationBar
from unify.legacy.conversation_manager.task_actions import (
    derive_short_name,
    iter_steering_tools_for_action,
    iter_steering_tools_for_completed_action,
)
from unify.logger import LOGGER
from unify.session_details import PLACEHOLDER_USER_FIRST_NAME, SESSION_DETAILS


def user_display_name() -> str:
    """The user's name as it appears on their lines of the conversation."""
    return SESSION_DETAILS.user.name or PLACEHOLDER_USER_FIRST_NAME


# =============================================================================
# Snapshot State Tracking for Incremental Context Propagation
# =============================================================================


@dataclass
class MessageElement:
    """A message element with identity for diff tracking.

    Identity is based on (index_in_conversation, timestamp).
    """

    index_in_conversation: int
    timestamp: datetime
    rendered: str


@dataclass
class NotificationElement:
    """A notification element with identity for diff tracking.

    Identity is based on (timestamp, content_hash, pinned).
    """

    timestamp: datetime
    content_hash: int
    pinned: bool
    rendered: str


@dataclass
class ActionElement:
    """An in-flight action element with identity for diff tracking.

    Identity is based on handle_id. State changes (new history events,
    status changes) are tracked via history_count and status.
    """

    handle_id: int
    query: str
    status: str
    history_count: int
    rendered: str


@dataclass
class SnapshotState:
    """Tracks the constituent elements of a rendered snapshot.

    This enables computing diffs between snapshots for incremental context
    propagation. Each element type has identity tracking:

    - Messages: (index_in_conversation, timestamp)
    - Notifications: (timestamp, content_hash, pinned)
    - Actions: (handle_id, with status/history tracking for state changes)

    The full_render contains the complete rendered snapshot string.
    """

    # The complete rendered snapshot
    full_render: str

    # Constituent elements with identity
    messages: list[MessageElement] = field(default_factory=list)
    notifications: list[NotificationElement] = field(default_factory=list)
    actions: list[ActionElement] = field(default_factory=list)

    def message_ids(self) -> set[tuple[int, datetime]]:
        """Return set of message identity tuples for diff comparison."""
        return {(m.index_in_conversation, m.timestamp) for m in self.messages}

    def notification_ids(self) -> set[tuple[datetime, int, bool]]:
        """Return set of notification identity tuples for diff comparison."""
        return {(n.timestamp, n.content_hash, n.pinned) for n in self.notifications}

    def action_states(self) -> dict[int, tuple[str, int]]:
        """Return dict of action_id -> (status, history_count) for diff comparison."""
        return {a.handle_id: (a.status, a.history_count) for a in self.actions}


def compute_snapshot_diff(
    old_snapshot: SnapshotState | None,
    new_snapshot: SnapshotState,
) -> str:
    """Compute the incremental diff between two snapshots.

    Returns a rendered string containing only the NEW or CHANGED elements:
    - New messages (not present in old snapshot)
    - New or changed notifications (new, or pinned state changed)
    - Action state changes (status changed, new history events)

    If old_snapshot is None, returns the full new snapshot (no diff possible).

    Args:
        old_snapshot: Previous snapshot state (or None for first snapshot)
        new_snapshot: Current snapshot state

    Returns:
        Rendered string containing only incremental updates, or empty string
        if nothing changed.
    """
    if old_snapshot is None:
        return new_snapshot.full_render

    diff_parts: list[str] = []

    # Find new messages
    old_msg_ids = old_snapshot.message_ids()
    new_messages = [
        m
        for m in new_snapshot.messages
        if (m.index_in_conversation, m.timestamp) not in old_msg_ids
    ]
    if new_messages:
        msg_renders = [m.rendered for m in new_messages]
        diff_parts.append(
            "<new_messages>\n" + "\n".join(msg_renders) + "\n</new_messages>",
        )

    # Find new or changed notifications
    old_notif_ids = old_snapshot.notification_ids()
    new_notifications = [
        n
        for n in new_snapshot.notifications
        if (n.timestamp, n.content_hash, n.pinned) not in old_notif_ids
    ]
    if new_notifications:
        notif_renders = [n.rendered for n in new_notifications]
        diff_parts.append(
            "<new_notifications>\n"
            + "\n".join(notif_renders)
            + "\n</new_notifications>",
        )

    # Find action state changes
    old_action_states = old_snapshot.action_states()
    action_changes = []
    for action in new_snapshot.actions:
        old_state = old_action_states.get(action.handle_id)
        if old_state is None:
            # New action
            action_changes.append(action.rendered)
        elif old_state != (action.status, action.history_count):
            # Status or history changed
            action_changes.append(action.rendered)
    if action_changes:
        diff_parts.append(
            "<action_updates>\n" + "\n".join(action_changes) + "\n</action_updates>",
        )

    if not diff_parts:
        return ""

    return "\n\n".join(diff_parts)


class Renderer:

    def render_state(
        self,
        chat_history: ChatHistory,
        notification_bar: NotificationBar = None,
        in_flight_actions: dict = None,
        completed_actions: dict = None,
        last_snapshot: datetime = None,
        recent_tool_executions: list[dict[str, Any]] | None = None,
        max_pinned_notifications: int = 50,
        max_messages: int = 25,
        max_action_history_events: int = 20,
        max_completed_actions: int = 20,
        max_completed_action_history_events: int = 5,
    ) -> SnapshotState:
        """Render the full conversation state.

        Returns a SnapshotState containing the rendered string and constituent
        element tracking for incremental diff computation.
        """
        from unify.common.prompt_helpers import now as prompt_now

        _render_t0 = perf_counter()
        _last_step = _render_t0

        def _mark_step() -> float:
            nonlocal _last_step
            now = perf_counter()
            elapsed_ms = (now - _last_step) * 1000
            _last_step = now
            return elapsed_ms

        message_elements: list[MessageElement] = []
        notification_elements: list[NotificationElement] = []
        action_elements: list[ActionElement] = []

        notif_render = self.render_notification_bar(
            notification_bar,
            last_snapshot=last_snapshot,
            max_pinned=max_pinned_notifications,
            elements_out=notification_elements,
        )
        _notifications_ms = _mark_step()
        actions_render = self.render_in_flight_actions(
            in_flight_actions,
            max_history=max_action_history_events,
            elements_out=action_elements,
        )
        _in_flight_ms = _mark_step()
        completed_render = self.render_completed_actions(
            completed_actions,
            max_completed=max_completed_actions,
            max_history=max_completed_action_history_events,
        )
        _completed_ms = _mark_step()
        recent_tools_render = self.render_recent_tool_executions(
            recent_tool_executions,
        )
        conversation_render = self.render_conversation(
            chat_history,
            last_snapshot=last_snapshot,
            max_messages=max_messages,
            elements_out=message_elements,
        )
        _conversation_ms = _mark_step()

        # The wall clock closes the snapshot rather than living in the system
        # prompt: a minute rollover then only re-tokenizes the snapshot tail
        # instead of invalidating the provider's system+tools cache.
        time_render = f"Current time: {prompt_now()}."

        sections = [
            s
            for s in [
                notif_render,
                actions_render,
                completed_render,
                recent_tools_render,
                conversation_render,
                time_render,
            ]
            if s
        ]
        full_render = "\n\n".join(sections)
        _join_ms = _mark_step()

        snapshot_state = SnapshotState(
            full_render=full_render,
            messages=message_elements,
            notifications=notification_elements,
            actions=action_elements,
        )
        _snapshot_ms = _mark_step()

        log_startup_timing(
            LOGGER,
            (
                "⏱️ [StartupTiming] llm_preamble.render_state.detail "
                "total=%.0fms notifications=%.0fms in_flight=%.0fms "
                "completed=%.0fms conversation=%.0fms join=%.0fms "
                "snapshot=%.0fms chars=%d messages=%d notifications_count=%d "
                "actions=%d sections=%d"
            ),
            (perf_counter() - _render_t0) * 1000,
            _notifications_ms,
            _in_flight_ms,
            _completed_ms,
            _conversation_ms,
            _join_ms,
            _snapshot_ms,
            len(full_render),
            len(message_elements),
            len(notification_elements),
            len(action_elements),
            len(sections),
        )

        return snapshot_state

    def render_notification_bar(
        self,
        notification_bar: NotificationBar,
        last_snapshot: datetime = None,
        max_pinned: int = 50,
        elements_out: list[NotificationElement] | None = None,
    ) -> str:
        """Render the notification bar."""
        if notification_bar is None:
            return "<notifications>\n</notifications>"

        pinned_notifs = sorted(
            (n for n in notification_bar.notifications if n.pinned),
            key=lambda n: n.timestamp,
        )[-max_pinned:]
        new_notifs = [
            n
            for n in notification_bar.notifications
            if not n.pinned and n.timestamp > last_snapshot
        ]
        all_notifs = pinned_notifs + new_notifs

        rendered_lines = []
        for n in all_notifs:
            prefix = "[PINNED]" if n.pinned else ""
            line = f'{prefix}[{n.type.title()} Notification @ {n.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")}] {n.content}'
            rendered_lines.append(line)

            if elements_out is not None:
                elements_out.append(
                    NotificationElement(
                        timestamp=n.timestamp,
                        content_hash=hash(n.content),
                        pinned=n.pinned,
                        rendered=line,
                    ),
                )

        return "<notifications>\n" + "\n".join(rendered_lines) + "\n</notifications>"

    @staticmethod
    def _render_action_history(
        handle_actions: list[dict],
        short_name: str,
        handle_id: int,
        max_history: int,
    ) -> str:
        """Render the event history for an action, capped to the most recent events."""
        displayed = handle_actions[-max_history:]
        if not displayed:
            return ""
        out = "<history>\n"
        for a in displayed:
            action_type = a.get("action_name", "")
            action_query = a.get("query", "")
            action_status = a.get("status", "")
            action_ts = a.get("timestamp", "")

            attrs = f"type='{action_type}'"
            if action_ts:
                attrs += f" timestamp='{action_ts}'"
            if action_status:
                attrs += f" status='{action_status}'"
            out += f"<event {attrs}>\n"

            if action_query:
                out += f"  <content>{action_query}</content>\n"
            if a_res := a.get("response"):
                out += f"  <response>{a_res}</response>\n"

            if action_status == "pending" and action_type.startswith("ask_"):
                out += (
                    "  <note>Result pending - you will receive another "
                    "turn when the answer is ready.</note>\n"
                )

            if action_type == "clarification_request" and not a.get("response"):
                call_id = a.get("call_id", "")
                out += (
                    "  <pending>Use answer_clarification_action("
                    f"handle_id={handle_id}, call_id='{call_id}', "
                    "answer=...) to respond</pending>\n"
                )
            out += "</event>\n"
        out += "</history>\n"
        return out

    def render_in_flight_actions(
        self,
        in_flight_actions: dict,
        max_history: int = 20,
        elements_out: list[ActionElement] | None = None,
    ) -> str:
        """Render in-flight actions with their status and history."""
        out = "<in_flight_actions>\n"
        if not in_flight_actions:
            out += "No actions currently executing.\n"
        else:
            for handle_id, handle_data in in_flight_actions.items():
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle = handle_data.get("handle")
                handle_actions = handle_data.get("handle_actions", [])

                is_paused = get_handle_paused_state(handle)
                status = "paused" if is_paused else "executing"

                pending_clarifications = [
                    a
                    for a in handle_actions
                    if a.get("action_name") == "clarification_request"
                    and not a.get("response")
                ]

                is_persistent = handle_data.get("persist", False)
                mode_attr = " mode='persistent'" if is_persistent else ""
                action_type = handle_data.get("action_type", "act")
                type_attr = f" type='{action_type}'"
                action_render = f"<action id='{handle_id}' short_name='{short_name}' status='{status}'{type_attr}{mode_attr}>\n"
                action_render += f"<original_request>{query}</original_request>\n"
                if is_persistent:
                    action_render += (
                        "<note>Persistent session — will NOT self-complete. "
                        "Use stop_action(handle_id=...) to end it. Responses marked "
                        "'awaiting_input' "
                        "mean the actor finished its turn and needs your next "
                        "interject_action to continue.</note>\n"
                    )

                action_render += "<steering_tools>\n"
                for action_name, description in iter_steering_tools_for_action(
                    handle_id,
                    query,
                    pending_clarifications,
                    is_paused=is_paused,
                ):
                    action_render += f"  - {action_name}: {description}\n"
                action_render += "</steering_tools>\n"

                action_render += self._render_action_history(
                    handle_actions,
                    short_name,
                    handle_id,
                    max_history,
                )

                action_render += "</action>\n"
                out += action_render

                if elements_out is not None:
                    elements_out.append(
                        ActionElement(
                            handle_id=handle_id,
                            query=query,
                            status=status,
                            history_count=len(handle_actions),
                            rendered=action_render,
                        ),
                    )

        out += "</in_flight_actions>"
        return out

    def render_recent_tool_executions(
        self,
        recent_tool_executions: list[dict[str, Any]] | None,
        max_items: int = 12,
    ) -> str:
        """Render a bounded summary of recently executed tools."""
        out = "<recent_tool_executions>\n"
        if not recent_tool_executions:
            out += "No recent tool executions.\n"
            out += "</recent_tool_executions>"
            return out

        for entry in recent_tool_executions[-max_items:]:
            tool_name = str(entry.get("tool_name") or "unknown")
            generation = str(entry.get("generation") or "?")
            origin_event = str(entry.get("origin_event_name") or "-")
            args_preview = str(entry.get("args_preview") or "{}")
            result_preview = str(entry.get("result_preview") or "null")
            out += f"- generation={generation} origin={origin_event} tool={tool_name}\n"
            out += f"  args={args_preview}\n"
            out += f"  result={result_preview}\n"
        out += "</recent_tool_executions>"
        return out

    def render_conversation(
        self,
        chat_history: ChatHistory,
        last_snapshot: datetime = None,
        max_messages: int = 25,
        elements_out: list[MessageElement] | None = None,
    ) -> str:
        """Render the tail of the conversation, most recent messages last."""
        messages = chat_history.recent()
        displayed = messages[-max_messages:]
        start_index = len(messages) - len(displayed)

        rendered_messages = []
        for i, message in enumerate(displayed):
            rendered = self.render_message(message, last_snapshot)
            rendered_messages.append(rendered)

            if elements_out is not None:
                elements_out.append(
                    MessageElement(
                        index_in_conversation=start_index + i,
                        timestamp=message.timestamp,
                        rendered=rendered,
                    ),
                )

        return "<conversation>\n" + "\n".join(rendered_messages) + "\n</conversation>"

    def render_message(
        self,
        message: ChatMessage,
        last_snapshot: datetime = None,
    ) -> str:
        """One conversation line: ``[Name @ time]: content [Attachments: ...]``.

        Messages newer than the last snapshot carry a **NEW** marker.
        """
        is_new = last_snapshot < message.timestamp
        new_marker = "**NEW** " if is_new else ""
        timestamp_str = message.timestamp.strftime("%A, %B %d, %Y at %I:%M %p")
        name = "You" if message.role == "assistant" else user_display_name()

        attachments_line = ""
        if message.attachments:
            attachments_line = f" [Attachments: {', '.join(message.attachments)}]"

        return f"{new_marker}[{name} @ {timestamp_str}]: {message.content}{attachments_line}"

    def render_completed_actions(
        self,
        completed_actions: dict,
        max_completed: int = 20,
        max_history: int = 5,
    ):
        """Render completed actions with their result and a brief history.

        Each entry is self-contained: original query, result, capped history,
        and steering tools for post-completion queries.
        """
        out = "<completed_actions>\n"
        if not completed_actions:
            out += "No completed actions.\n"
        else:
            # Cap to the most recent completed actions by handle_id (monotonic)
            items = list(completed_actions.items())[-max_completed:]

            for handle_id, handle_data in items:
                query = handle_data.get("query", "")
                short_name = derive_short_name(query)
                handle_actions = handle_data.get("handle_actions", [])

                # Extract terminal status from the most recent completion marker.
                terminal_event = None
                for a in reversed(handle_actions):
                    if a.get("action_name") in {"act_completed", "act_failed"}:
                        terminal_event = a
                        break

                action_type = handle_data.get("action_type", "act")
                action_status = (
                    "failed"
                    if terminal_event is not None
                    and terminal_event.get("success") is False
                    else "completed"
                )
                out += f"<action id='{handle_id}' short_name='{short_name}' status='{action_status}' type='{action_type}'>\n"
                out += f"<original_request>{query}</original_request>\n"

                if terminal_event is not None:
                    if terminal_event.get("success") is False:
                        error_text = terminal_event.get("error") or terminal_event.get(
                            "query",
                            "",
                        )
                        if error_text:
                            out += f"<error>{error_text}</error>\n"
                    else:
                        result = terminal_event.get(
                            "result",
                            terminal_event.get("query", ""),
                        )
                        out += f"<result>{result}</result>\n"

                out += self._render_action_history(
                    handle_actions,
                    short_name,
                    handle_id,
                    max_history,
                )

                out += "<steering_tools>\n"
                for (
                    action_name,
                    description,
                ) in iter_steering_tools_for_completed_action(
                    handle_id,
                    query,
                ):
                    out += f"  - {action_name}: {description}\n"
                out += "</steering_tools>\n"

                out += "</action>\n"
        out += "</completed_actions>"
        return out
