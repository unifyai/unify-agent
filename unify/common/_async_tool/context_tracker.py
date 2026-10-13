"""Incremental propagation of parent chat context through nested tool loops:
the initial snapshot is sent once per inner tool, later local messages are
sent incrementally, and each nesting level tracks what it forwarded to whom."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional
from ..context_dump import make_messages_safe_for_context_dump
from .messages import extract_substantive_text, is_loop_authored_message


def _belongs_in_context_snapshot(msg: dict) -> bool:
    """Keep-rule for messages forwarded to an inner tool as parent context.

    A context snapshot carries conversational intent, not payload bulk: it
    keeps genuine user turns and assistant turns whose text a user would
    read. Everything else — tool results, tool-call-bearing assistant
    messages, system messages, and loop-authored notices — is dropped;
    an inner tool rediscovers its own results and receives task specifics
    through its request text. User turns are kept regardless of content
    shape (an image-only or even empty user turn is still a genuine turn
    boundary); only assistant turns must carry substantive text.
    """
    role = msg.get("role")
    if role == "user":
        return not is_loop_authored_message(msg)
    if role == "assistant":
        if msg.get("tool_calls"):
            return False
        return extract_substantive_text(msg.get("content")) is not None
    return False


@dataclass
class ContextForwardingState:
    """What has been forwarded to one inner tool, so only unsent context
    items go out on the next update."""

    initial_context_sent: bool = False
    # Index into the local transcript; earlier messages were sent.
    last_local_msg_idx_forwarded: int = 0


@dataclass
class LoopContextState:
    """Context state of one tool loop: the immutable snapshot it started
    with, and per-inner-tool tracking of what has been forwarded."""

    parent_chat_context: list[dict] = field(default_factory=list)

    # call_id -> ContextForwardingState
    inner_tool_forwarding: dict[str, ContextForwardingState] = field(
        default_factory=dict,
    )

    def get_forwarding_state(self, call_id: str) -> ContextForwardingState:
        """Get or create the forwarding state for an inner tool call."""
        if call_id not in self.inner_tool_forwarding:
            self.inner_tool_forwarding[call_id] = ContextForwardingState()
        return self.inner_tool_forwarding[call_id]

    def compute_context_for_inner_tool(
        self,
        call_id: str,
        current_local_msgs: list[dict],
    ) -> tuple[Optional[list[dict]], Optional[list[dict]]]:
        """Compute the context to pass to an inner tool call: the filtered
        initial snapshot on its first call, then only what has not been sent
        to that call yet.

        Both the initial snapshot and later incremental updates apply the
        snapshot keep-rule (``_belongs_in_context_snapshot``) to this loop's
        own layers — the inherited parent snapshot and the local transcript.
        The filter reads the transcript; it never mutates it or the tracked
        messages.

        ``current_local_msgs`` is this loop's transcript without the context
        header. Returns ``(parent_chat_context, _parent_chat_context_cont)``:
        the filtered initial snapshot (first call only; None when nothing
        survives the keep-rule) and the incremental updates since the last
        call (None when there are none).
        """
        state = self.get_forwarding_state(call_id)

        result_parent_ctx: Optional[list[dict]] = None
        result_cont: Optional[list[dict]] = None

        # Both the inherited parent layer and the local messages pass the
        # snapshot keep-rule, so the snapshot never carries tool payloads
        # or loop notices regardless of which layer they entered at.
        if not state.initial_context_sent:
            parent_ctx_to_send = [
                m for m in self.parent_chat_context if _belongs_in_context_snapshot(m)
            ]
            local_msgs_to_send = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in current_local_msgs
                if _belongs_in_context_snapshot(m)
            ]
            # Local messages nest as children of the last parent message.
            if parent_ctx_to_send:
                result_parent_ctx = copy.deepcopy(parent_ctx_to_send)
                if local_msgs_to_send:
                    result_parent_ctx[-1].setdefault("children", []).extend(
                        local_msgs_to_send,
                    )
            elif local_msgs_to_send:
                result_parent_ctx = local_msgs_to_send

            state.initial_context_sent = True
            state.last_local_msg_idx_forwarded = len(current_local_msgs)

        else:
            incremental_cont: list[dict] = []

            # New local messages since last forward — same snapshot keep-rule
            # as the initial send, so continuations never smuggle in the
            # payload bulk the initial snapshot filtered out. Forwarding
            # indices track the raw transcript, not the filtered view.
            if state.last_local_msg_idx_forwarded < len(current_local_msgs):
                new_local = current_local_msgs[state.last_local_msg_idx_forwarded :]
                new_local_formatted = [
                    {"role": m.get("role"), "content": m.get("content")}
                    for m in new_local
                    if _belongs_in_context_snapshot(m)
                ]
                incremental_cont.extend(new_local_formatted)
                state.last_local_msg_idx_forwarded = len(current_local_msgs)

            if incremental_cont:
                result_cont = incremental_cont

        if result_parent_ctx is not None:
            result_parent_ctx = make_messages_safe_for_context_dump(result_parent_ctx)
        if result_cont is not None:
            result_cont = make_messages_safe_for_context_dump(result_cont)

        return result_parent_ctx, result_cont
