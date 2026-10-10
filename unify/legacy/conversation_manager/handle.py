from __future__ import annotations

import asyncio
from typing import Optional, Type, TypeVar, TYPE_CHECKING
from pydantic import BaseModel
from enum import Enum
from unify.common.async_tool_loop import start_async_tool_loop, SteerableToolHandle
from unify.common.llm_client import new_slow_brain_llm_client
from .base import BaseConversationManagerHandle
from .events import (
    NotificationInjectedEvent,
    DirectMessageEvent,
)
from .prompt_builders import build_ask_handle_prompt

if TYPE_CHECKING:
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )
    from unify.legacy.conversation_manager.in_memory_event_broker import (
        InMemoryEventBroker,
    )

T = TypeVar("T", bound=[BaseModel, Enum])


# How long a question posted to the chat waits for the user before the loop
# is told the question went unanswered. Long enough to cover a person reading
# and thinking; short enough that a user who walked away does not hold the
# loop open indefinitely.
USER_REPLY_TIMEOUT_S = 120

# How much of the conversation the ask loop sees.
RECENT_TRANSCRIPT_MESSAGES = 20


class ConversationManagerHandle(BaseConversationManagerHandle):
    """
    The concrete implementation for steering a live ConversationManager instance.

    The Actor receives this handle so a running plan can reach back into the
    chat: ``ask`` puts a question to the user, ``interject`` drops information
    into the brain's notification bar. Both go over the event broker.
    """

    def __init__(
        self,
        event_broker: "InMemoryEventBroker",
        *,
        conversation_manager: "ConversationManager",
    ):
        self.event_broker = event_broker
        self.conversation_manager = conversation_manager

        self._steering_channel = "app:comms:steering"
        self._stopped = False
        self._final_result = "Handle is active."

    # ─────────────────────────────────────────────────────────────
    # Standard SteerableToolHandle Methods
    # ─────────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        response_format: Optional[Type[T]] = None,
    ) -> SteerableToolHandle:
        """
        Asks a question to the user and returns a handle to the running sub-conversation.

        Args:
            question: The question to ask the user
            response_format: Optional Pydantic model or Enum type for structured responses
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped handle.")

        cm_handle = self

        # The recent conversation gives the loop a chance to answer without
        # asking (PATH 1).
        conversation_turns, _ = self.conversation_manager.get_recent_transcript(
            max_messages=RECENT_TRANSCRIPT_MESSAGES,
        )
        if conversation_turns:
            prompt_lines = [
                f"- {turn['role']}: {turn['content']}" for turn in conversation_turns
            ]
            recent_transcript_for_prompt = (
                f"Recent Transcript (last {RECENT_TRANSCRIPT_MESSAGES} messages):\n"
                + "\n".join(prompt_lines)
            )
        else:
            recent_transcript_for_prompt = "Recent Transcript: (none)"

        # Build prompts using prompt_builders
        prompt_parts = build_ask_handle_prompt(
            question=question,
            recent_transcript=recent_transcript_for_prompt,
        )

        user_reply_future = asyncio.Future()

        # This handles PATH 2 (Ask & Wait).
        async def ask_question(text: str):
            """
            Asks the user a question and WAITS for a reply.
            This tool BLOCKS until the user replies in the chat.
            Use this when you need to ask a clarifying question (PATH 2).
            """
            nonlocal user_reply_future
            # Post to the chat directly (bypasses the main CM brain).
            await self.event_broker.publish(
                "app:comms:direct_speech",
                DirectMessageEvent(content=text).to_json(),
            )

            # An interjection may already have delivered the answer before the
            # question was posted; otherwise wait for the next.
            if not user_reply_future.done():
                try:
                    await asyncio.wait_for(
                        user_reply_future,
                        timeout=USER_REPLY_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    # The wait leaves the future cancelled. Hand the next
                    # question a fresh one: reading a cancelled future raises
                    # CancelledError, which would surface as this whole loop
                    # having been cancelled rather than as one unanswered
                    # question.
                    user_reply_future = asyncio.Future()
                    return "Timed out waiting for user reply."

            user_msg = user_reply_future.result()
            # Reset for potential future questions
            user_reply_future = asyncio.Future()
            return f"User replied: {user_msg}"

        tools = {"ask_question": ask_question}

        # ──────────────────────────────────────────────────────────────────
        # 3. START THE LOOP
        # ──────────────────────────────────────────────────────────────────
        llm = new_slow_brain_llm_client(
            return_full_completion=False,
        )
        llm.set_system_message(prompt_parts.to_list())

        # Get the parent lineage from the ConversationManager's session logger
        parent_lineage: list[str] = []
        if hasattr(self.conversation_manager, "_session_logger"):
            parent_lineage = self.conversation_manager._session_logger.child_lineage()

        # Pass response_format directly - the async tool loop handles
        # final_answer tool injection automatically
        inner_handle = start_async_tool_loop(
            client=llm,
            message=f"Answer the question: '{question}'",
            tools=tools,
            response_format=response_format,
            interrupt_llm_with_interjections=True,
            loop_id="ConversationManager.ask",
            parent_lineage=parent_lineage,
        )

        # ──────────────────────────────────────────────────────────────────
        # 4. THE WRAPPER (The Bridge)
        # ──────────────────────────────────────────────────────────────────
        class InterceptingHandle(SteerableToolHandle):
            def __init__(self):
                pass

            # Delegate standard lifecycle methods
            async def stop(self, reason: Optional[str] = None, **kwargs):
                await inner_handle.stop(reason, **kwargs)

            async def pause(self):
                return await inner_handle.pause()

            async def resume(self):
                return await inner_handle.resume()

            def done(self):
                return inner_handle.done()

            # Delegate event APIs
            async def next_clarification(self) -> dict:
                return await inner_handle.next_clarification()

            async def next_notification(self) -> dict:
                return await inner_handle.next_notification()

            async def answer_clarification(self, call_id: str, answer: str) -> None:
                return await inner_handle.answer_clarification(call_id, answer)

            async def ask(self, question: str, **kwargs) -> SteerableToolHandle:
                return await inner_handle.ask(question, **kwargs)

            # INTERJECT HANDLER (Triggered by ConversationManager)
            async def interject(self, message: str, **kwargs):
                if not user_reply_future.done():
                    # ask_question is blocking — deliver the reply via the
                    # future so it surfaces as a tool result, not as a
                    # duplicate user message in the conversation.
                    user_reply_future.set_result(message)
                else:
                    # No ask_question pending — forward as a regular
                    # interjection into the tool loop.
                    await inner_handle.interject(
                        message,
                        trigger_immediate_llm_turn=False,
                        **kwargs,
                    )

            async def result(self):
                try:
                    raw_result = await inner_handle.result()

                    # Handle the standard stop notice from async tool loop
                    if raw_result == "processed stopped early, no result":
                        return None

                    # Handle null/None results
                    if raw_result is None:
                        return None

                    return raw_result

                finally:
                    if cm_handle.conversation_manager.active_ask_handle == self:
                        cm_handle.conversation_manager.active_ask_handle = None

        # Register with CM
        wrapped_handle = InterceptingHandle()
        self.conversation_manager.active_ask_handle = wrapped_handle

        return wrapped_handle

    async def interject(self, message: str, **kwargs) -> str:
        """Provide additional information or instructions to the conversation.

        Publishes a ``NotificationInjectedEvent`` to the steering channel so
        the CM brain can incorporate the message.  Plumbing kwargs (e.g.
        ``_parent_chat_context_cont``) are accepted but unused -- the CM handle
        does not maintain an LLM loop to inject context into.

        Returns
        -------
        str
            The ``interjection_id`` assigned to this interjection.
        """
        if self._stopped:
            return ""
        event = NotificationInjectedEvent(
            content=message,
            source="interjection",
        )
        await self.event_broker.publish(self._steering_channel, event.to_json())
        return event.interjection_id

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        """Stops the handle."""
        if self._stopped:
            return
        self._stopped = True
        self._final_result = (
            f"Handle stopped. Reason: {reason or 'No reason provided.'}"
        )

    def done(self) -> bool:
        return self._stopped

    async def result(self) -> str:
        while not self._stopped:
            await asyncio.sleep(0.1)
        return self._final_result

    # --- Other SteerableToolHandle methods (no-op for this handle) ---

    async def pause(self) -> str:
        return "ConversationManagerHandle does not support pausing."

    async def resume(self) -> str:
        return "ConversationManagerHandle does not support resuming."

    async def next_clarification(self) -> dict:
        return {}

    async def next_notification(self) -> dict:
        return {}

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass
