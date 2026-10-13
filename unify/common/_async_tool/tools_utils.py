from typing import TypedDict, Literal
from dataclasses import dataclass, field
import asyncio
from typing import Any

from . import time_context


@dataclass
class ToolCallMetadata:
    name: str
    call_id: str
    call_dict: dict
    call_idx: int
    chat_context: Any
    assistant_msg: dict
    tool_schema: dict
    llm_arguments: dict
    raw_arguments_json: str
    clar_up_queue: asyncio.Queue[str] | None = None
    clar_down_queue: asyncio.Queue[str] | None = None
    # Optional notification stream emitted by tools; payload is a dict with arbitrary fields
    notification_queue: asyncio.Queue[dict] | None = None
    # Monotonic time when tool was scheduled (uses perf_counter for monkey-patchability)
    scheduled_time: float = field(default_factory=lambda: time_context.perf_counter())
    # Whether the LLM opted in to receive parent chat context for this tool.
    # When False, context continuations should NOT be forwarded to this tool.
    context_opted_in: bool = True


class ToolCallMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


def create_tool_call_message(name: str, call_id: str, content: str) -> ToolCallMessage:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }
