"""Typed payload models for known EventBus event types.

Each event type has a corresponding Pydantic model that defines its schema.
The EventBus validates every payload against its model at publish time and
hands consumers the validated fields as a plain dict.
"""

from __future__ import annotations


from pydantic import BaseModel

from .comms import CommsPayload
from .llm import LLMPayload
from .manager_method import ManagerMethodPayload
from .tool_loop import ToolLoopCancelledTurnPayload, ToolLoopPayload

__all__ = [
    "ManagerMethodPayload",
    "ToolLoopPayload",
    "ToolLoopCancelledTurnPayload",
    "CommsPayload",
    "LLMPayload",
    "PAYLOAD_REGISTRY",
]

# Registry: event type string → Pydantic payload model
PAYLOAD_REGISTRY: dict[str, type[BaseModel]] = {
    "ManagerMethod": ManagerMethodPayload,
    "ToolLoop": ToolLoopPayload,
    "ToolLoopCancelledTurn": ToolLoopCancelledTurnPayload,
    "Comms": CommsPayload,
    "LLM": LLMPayload,
}
