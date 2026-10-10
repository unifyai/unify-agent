from __future__ import annotations

from unify.common.async_tool_loop import SteerableToolHandle


class BaseConversationManagerHandle(SteerableToolHandle):
    """
    Abstract interface for steering a live ConversationManager instance.

    This handle enables external processes (like the Actor)
    to inject information into the conversation flow in real-time. The primary
    methods for bidirectional communication are:

    - `ask()`: For the external process to get information FROM the user. (Bottom-up)
    - `interject()`: For an external system to send information TO the user. (Top-down)
    """
