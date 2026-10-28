import time
from typing import TypedDict, Literal
from dataclasses import dataclass, field


@dataclass
class ToolCallMetadata:
    name: str
    call_id: str
    call_dict: dict
    call_idx: int
    assistant_msg: dict
    tool_schema: dict
    llm_arguments: dict
    raw_arguments_json: str
    # Monotonic time when the tool was scheduled
    scheduled_time: float = field(default_factory=time.perf_counter)


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
