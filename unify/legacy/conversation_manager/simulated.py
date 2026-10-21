from __future__ import annotations

import asyncio
from typing import Optional, Type, Any
from pydantic import BaseModel
import unillm

from ...common.llm_client import new_slow_brain_llm_client
from ...common.simulated import (
    SimulatedLineage,
    SimulatedLog,
    simulated_llm_roundtrip,
    SimulatedHandleMixin,
)
from .base import BaseConversationManagerHandle
from ...common import SteerableToolHandle


class SimulatedConversationManagerHandle(
    SimulatedHandleMixin,
    BaseConversationManagerHandle,
):
    """
    Simulated conversation manager handle for testing and demos.

    Uses a stateful LLM to simulate conversation steering without
    actual pub/sub or real conversation state.
    """

    def __init__(
        self,
        assistant_id: str = "",
        *,
        description: str = "A simulated conversation between an AI assistant and a user.",
        simulation_guidance: Optional[str] = None,
        **kwargs: Any,
    ):
        self.assistant_id = assistant_id
        self._description = description
        self._simulation_guidance = simulation_guidance

        # A shared, stateful LLM for maintaining conversation context
        self._llm = new_slow_brain_llm_client(
            stateful=True,
            origin="SimulatedConversationManager",
        )

        # Initialize the system message for the stateful LLM
        system_msg = self._build_system_message()
        self._llm.set_system_message(system_msg)

        # Internal state management
        self._stopped = False
        self._paused = False
        self._final_result = "Conversation is active."

        # Human-friendly log label for consistent hierarchical logging
        self._log_label = SimulatedLineage.make_label("SimulatedConversationManager")

    def _build_system_message(self) -> str:
        """Builds a detailed system message for the stateful LLM, instructing it on how to simulate the conversation."""
        return f"""You are a simulated ConversationManager. Your role is to maintain the state of a conversation and respond to steering commands.

### Conversation Scenario
**Description:** {self._description}
**Assistant ID:** {self.assistant_id}

### Core Responsibilities
1.  **Maintain Internal State:** You must remember the conversation history, the user's mood, and any notifications you receive.
2.  **Respond to `ask`:** When you receive a question via `ask`, provide a plausible, in-character response from the simulated user. The response should be concise and directly answer the question.
3.  **Acknowledge `interject`:** When an interjection is received, incorporate its content into your internal state. For example, if you receive "Task 'X' is complete," your subsequent `ask` responses should reflect this knowledge.
4.  **Adhere to Simulation Guidance:** {self._simulation_guidance or "No specific guidance provided."}

### Response Formats
- For `ask` calls, provide a direct, first-person answer as the simulated user.
- For `interject` calls, acknowledge and incorporate the information into your state.
"""

    # ─────────────────────────────────────────────────────────────
    # Standard SteerableToolHandle Methods
    # ─────────────────────────────────────────────────────────────

    async def ask(
        self,
        question: str,
        *,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> "SteerableToolHandle":
        """
        Asks a question to the simulated user in the conversation.
        Supports both simple string responses and structured Pydantic models.
        """
        if self._stopped:
            raise RuntimeError("Cannot ask a stopped conversation.")

        # Log the ask request
        ask_label = SimulatedLineage.make_label("SimulatedConversationManager.ask")
        SimulatedLog.log_request("ask", ask_label, question)

        prompt = f"""The external process is asking the user a question. Based on your persona and the conversation history, provide a direct and plausible answer.
**Question:** "{question}"
"""
        if response_format:
            prompt += "\n**FORMAT INSTRUCTIONS:** Your response MUST be a JSON object that strictly conforms to the provided Pydantic model schema."

        class _AnswerHandle(SimulatedHandleMixin, SteerableToolHandle):
            def __init__(
                inner_self,
                stateful_llm: unillm.AsyncUnify,
                prompt_str: str,
                pydantic_model: Optional[Type[BaseModel]],
                log_label: str,
            ):
                inner_self._llm = stateful_llm
                inner_self._prompt = prompt_str
                inner_self._model = pydantic_model
                inner_self._result_cache: Optional[Any] = None
                inner_self._done = False
                inner_self._log_label = log_label

            async def result(inner_self) -> Any:
                if inner_self._result_cache is None:
                    inner_self._result_cache = await simulated_llm_roundtrip(
                        inner_self._llm,
                        label=inner_self._log_label,
                        prompt=inner_self._prompt,
                        response_format=inner_self._model,
                    )
                    inner_self._done = True
                return inner_self._result_cache

            def done(inner_self) -> bool:
                return inner_self._done

            async def stop(inner_self, reason: str | None = None, *args, **kwargs):
                inner_self._log_stop(reason)

            async def pause(inner_self):
                inner_self._log_pause()

            async def resume(inner_self):
                inner_self._log_resume()

            async def interject(inner_self, message: str, *args, **kwargs):
                inner_self._log_interject(message)

            async def ask(inner_self, *args, **kwargs):
                return inner_self

            async def answer_clarification(
                inner_self,
                call_id: str,
                answer: str,
            ) -> None:
                pass

        return _AnswerHandle(self._llm, prompt, response_format, ask_label)

    async def interject(self, message: str, **kwargs) -> str:
        """Provide additional information or instructions to the conversation.

        Feeds ``message`` into the stateful LLM so subsequent ``ask`` calls
        reflect the new information.  Plumbing kwargs (e.g.
        ``_parent_chat_context_cont``) are accepted but unused.

        Returns
        -------
        str
            A synthetic ``interjection_id`` for parity with the real handle.
        """
        if self._stopped:
            return ""
        self._log_interject(message)
        prompt = (
            "An interjection has been received. Incorporate the following "
            "information into your internal state for future questions.\n"
            f"- **Content:** {message}"
        )
        await self._llm.generate(prompt)
        import uuid as _uuid

        return _uuid.uuid4().hex[:12]

    async def pause(self) -> str:
        """Pauses the simulated conversation."""
        self._log_pause()
        self._paused = True
        return "Simulated conversation is paused."

    async def resume(self) -> str:
        """Resumes a paused conversation."""
        self._log_resume()
        self._paused = False
        return "Simulated conversation is resumed."

    async def stop(self, reason: Optional[str] = None, **kwargs) -> None:
        """Stops the simulated conversation."""
        self._log_stop(reason)
        self._stopped = True
        self._final_result = (
            f"Conversation stopped. Reason: {reason or 'No reason provided.'}"
        )

    def done(self) -> bool:
        """Checks if the conversation is stopped."""
        return self._stopped

    async def result(self) -> str:
        """Returns the final state of the conversation."""
        while not self._stopped:
            await asyncio.sleep(0.1)
        return self._final_result

    async def answer_clarification(self, call_id: str, answer: str) -> None:
        pass
