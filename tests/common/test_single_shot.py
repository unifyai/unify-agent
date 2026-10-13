"""Tests for single_shot_tool_decision."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from tests.helpers import _handle_project
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common.llm_helpers import inject_tool_call_thoughts, method_to_schema
from unify.common.single_shot import (
    single_shot_tool_decision,
    SingleShotResult,
    ToolExecution,
)
from unify.common.llm_client import new_llm_client
from unify.logger import LOGGER

# --------------------------------------------------------------------------- #
#  Test fixtures: response format models                                       #
# --------------------------------------------------------------------------- #


class ThoughtsResponse(BaseModel):
    """Simple structured output with just thoughts."""

    thoughts: str = Field(..., description="Your reasoning about the request")


class DecisionResponse(BaseModel):
    """Structured output for decisions."""

    thoughts: str = Field(..., description="Your reasoning about the decision")
    decision: str = Field(..., description="Your decision: 'approve' or 'reject'")
    confidence: int = Field(..., description="Confidence level from 1-10")


# --------------------------------------------------------------------------- #
#  Test fixtures: simple tools                                                #
# --------------------------------------------------------------------------- #


def greet(*, name: str) -> str:
    """Greet a person by name.

    Parameters
    ----------
    name : str
        The name of the person to greet.

    Returns
    -------
    str
        A greeting message.
    """
    return f"Hello, {name}!"


def add_numbers(*, a: int, b: int) -> int:
    """Add two numbers together.

    Parameters
    ----------
    a : int
        First number.
    b : int
        Second number.

    Returns
    -------
    int
        The sum of a and b.
    """
    return a + b


async def async_multiply(*, x: int, y: int) -> int:
    """Multiply two numbers (async version).

    Parameters
    ----------
    x : int
        First factor.
    y : int
        Second factor.

    Returns
    -------
    int
        The product of x and y.
    """
    return x * y


def do_nothing() -> str:
    """Do nothing and return a confirmation.

    Use this when no action is needed.

    Returns
    -------
    str
        Confirmation that nothing was done.
    """
    return "Nothing done."


def send_notification(*, message: str) -> str:
    """Send a notification message.

    Parameters
    ----------
    message : str
        The notification message to send.

    Returns
    -------
    str
        Confirmation that the notification was sent.
    """
    return f"Notification sent: {message}"


def start_background_task(*, task_name: str) -> str:
    """Start a background task.

    Parameters
    ----------
    task_name : str
        Name of the task to start.

    Returns
    -------
    str
        Confirmation that the task was started.
    """
    return f"Task '{task_name}' started"


# --------------------------------------------------------------------------- #
#  Unit tests: result structure                                               #
# --------------------------------------------------------------------------- #


def test_tool_execution_dataclass():
    """ToolExecution has expected fields."""
    execution = ToolExecution(
        name="greet",
        args={"name": "Alice"},
        result="Hello, Alice!",
    )
    assert execution.name == "greet"
    assert execution.args == {"name": "Alice"}
    assert execution.result == "Hello, Alice!"


def test_single_shot_result_single_tool():
    """SingleShotResult with a single tool provides backward-compatible properties."""
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
        ],
        text_response=None,
    )
    # New API
    assert len(result.tools) == 1
    assert result.tools[0].name == "greet"
    assert result.tools[0].args == {"name": "Alice"}
    assert result.tools[0].result == "Hello, Alice!"
    # Backward-compatible properties
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"
    assert result.text_response is None


def test_single_shot_result_multiple_tools():
    """SingleShotResult can hold multiple tool executions."""
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
            ToolExecution(name="add_numbers", args={"a": 2, "b": 3}, result=5),
        ],
        text_response=None,
    )
    # New API - all tools accessible
    assert len(result.tools) == 2
    assert result.tools[0].name == "greet"
    assert result.tools[1].name == "add_numbers"
    assert result.tools[1].result == 5
    # Backward-compatible properties return first tool
    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"


def test_single_shot_result_no_tool():
    """SingleShotResult can represent no-tool case."""
    result = SingleShotResult(
        tools=[],
        text_response="I don't need to use any tools.",
    )
    assert len(result.tools) == 0
    # Backward-compatible properties return None when no tools
    assert result.tool_name is None
    assert result.tool_args is None
    assert result.tool_result is None
    assert result.text_response is not None


def test_single_shot_result_with_structured_output():
    """SingleShotResult can include structured_output."""
    structured = ThoughtsResponse(thoughts="This is my reasoning")
    result = SingleShotResult(
        tools=[
            ToolExecution(name="greet", args={"name": "Alice"}, result="Hello, Alice!"),
        ],
        text_response='{"thoughts": "This is my reasoning"}',
        structured_output=structured,
    )
    assert result.tool_name == "greet"
    assert result.structured_output is not None
    assert result.structured_output.thoughts == "This is my reasoning"


def test_inject_tool_call_thoughts_schema():
    """Injected tool schemas expose optional thoughts, not required."""
    schema = inject_tool_call_thoughts(
        method_to_schema(greet, tool_name="greet", include_class_name=False),
    )
    params = schema["function"]["parameters"]
    assert "thoughts" in params["properties"]
    assert "thoughts" not in params["required"]


@pytest.mark.asyncio
async def test_single_shot_strips_tool_thoughts_before_execution():
    """Per-tool thoughts are stripped before invocation and stored separately."""

    def strict_greet(*, name: str) -> str:
        return f"Hello, {name}!"

    async def fake_generate(**_kwargs):
        client.messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "greet",
                            "arguments": (
                                '{"name": "Alice", "thoughts": "User asked for a greeting"}'
                            ),
                        },
                    },
                ],
            },
        ]

    client = SimpleNamespace(messages=[], generate=fake_generate)

    result = await single_shot_tool_decision(
        client,
        "hello",
        {"greet": strict_greet},
        inject_tool_thoughts=True,
    )

    assert result.tools[0].args == {"name": "Alice"}
    assert result.tools[0].thoughts == "User asked for a greeting"
    assert result.tools[0].result == "Hello, Alice!"


@pytest.mark.asyncio
async def test_debug_level_logs_single_shot_response_shape(caplog):

    async def fake_generate(**_kwargs):
        client.messages = [
            {
                "role": "assistant",
                "content": '{"thoughts": "Greet the user."}',
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "greet",
                            "arguments": '{"name": "Alice"}',
                        },
                    },
                ],
            },
        ]

    client = SimpleNamespace(messages=[], generate=fake_generate)
    LOGGER.addHandler(caplog.handler)
    caplog.set_level(logging.DEBUG, logger="unify")

    try:
        await single_shot_tool_decision(
            client,
            "hello",
            {"greet": lambda name: f"Hello, {name}!"},
        )
    finally:
        LOGGER.removeHandler(caplog.handler)

    assert "Single-shot response shape" in caplog.text
    assert "message_type=dict" in caplog.text
    assert "content_type=str" in caplog.text
    assert "tool_calls_type=list" in caplog.text
    assert "tool_call_names=['greet']" in caplog.text


# --------------------------------------------------------------------------- #
#  Cancellation: a fake client stands in for the model                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_single_shot_cancels_during_model_generation():
    """Cancellation before the model returns should stop the stale turn."""
    generate_started = asyncio.Event()
    tool_called = False

    async def fake_generate(**_kwargs):
        generate_started.set()
        await asyncio.Event().wait()

    async def slow_tool() -> str:
        """Run a slow side-effecting tool."""
        nonlocal tool_called
        tool_called = True
        return "done"

    client = SimpleNamespace(messages=[], generate=fake_generate)
    task = asyncio.create_task(
        single_shot_tool_decision(
            client,
            "hello",
            {"slow_tool": slow_tool},
        ),
    )

    await asyncio.wait_for(generate_started.wait(), timeout=2.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert tool_called is False


@pytest.mark.asyncio
async def test_single_shot_defers_cancellation_after_tool_execution_starts():
    """Once tools begin executing, cancellation waits for them to complete."""
    tool_started = asyncio.Event()
    tool_can_finish = asyncio.Event()
    tool_finished = asyncio.Event()
    callback_calls = 0

    async def fake_generate(**_kwargs):
        client.messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "slow_tool",
                            "arguments": '{"value": 7}',
                        },
                    },
                ],
            },
        ]

    async def slow_tool(*, value: int) -> int:
        """Return a value after the test allows the tool to finish."""
        tool_started.set()
        await tool_can_finish.wait()
        tool_finished.set()
        return value + 1

    def on_tool_execution_start() -> None:
        nonlocal callback_calls
        callback_calls += 1

    client = SimpleNamespace(messages=[], generate=fake_generate)
    task = asyncio.create_task(
        single_shot_tool_decision(
            client,
            "hello",
            {"slow_tool": slow_tool},
            on_tool_execution_start=on_tool_execution_start,
        ),
    )

    await asyncio.wait_for(tool_started.wait(), timeout=2.0)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False

    tool_can_finish.set()
    result = await asyncio.wait_for(task, timeout=2.0)

    assert callback_calls == 1
    assert tool_finished.is_set()
    assert result.tool_name == "slow_tool"
    assert result.tool_result == 8


# --------------------------------------------------------------------------- #
#  A scripted model: what single_shot_tool_decision makes of a reply          #
# --------------------------------------------------------------------------- #
#
# The model is the scripted transport (tests/scripted_model.py): each test
# scripts the reply a model would give and checks what the request offered
# (tools, tool_choice, response_format, the messages) and what the decision
# did with the reply (which callable ran, with which arguments, and what came
# back). A request that offers tools is an ``actor`` request there, one
# without tools a ``query_llm`` request.


def _scripted_client(system: str):
    """A client built inside the scripted block, so it never reads a recording."""
    client = new_llm_client()
    client.set_system_message(system)
    return client


def _says(message: dict, text: str) -> bool:
    return text in json.dumps(message.get("content"))


def _sent(call, system: str, user: str) -> None:
    """The request carried the client's system message and the user message."""
    assert any(
        m.get("role") == "system" and _says(m, system) for m in call.messages
    ), call.messages
    assert call.messages[-1]["role"] == "user"
    assert _says(call.messages[-1], user), call.messages[-1]


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_calls_tool():
    """The tool the model calls runs with the model's arguments; every tool is
    offered, with tool_choice auto."""
    system = "You are a helpful assistant. Use the available tools to respond to requests."
    model = ScriptedModel(actor=[reply(calls=[("greet", {"name": "Alice"})])])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(system),
            "Please greet Alice.",
            {"greet": greet, "add_numbers": add_numbers},
        )

    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Alice"}
    assert result.tool_result == "Hello, Alice!"
    (call,) = model.calls
    assert sorted(call.tool_names) == ["add_numbers", "greet"]
    assert call.request["tool_choice"] == "auto"
    _sent(call, system, "Please greet Alice.")
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_chooses_correct_tool():
    """Of several tools, the one the model names runs (and only that one)."""
    model = ScriptedModel(actor=[reply(calls=[("add_numbers", {"a": 5, "b": 7})])])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are a calculator assistant. Use tools to perform calculations.",
            ),
            "What is 5 + 7?",
            {"greet": greet, "add_numbers": add_numbers},
        )

    assert result.tool_name == "add_numbers"
    assert result.tool_args == {"a": 5, "b": 7}
    assert result.tool_result == 12
    assert len(result.tools) == 1
    (call,) = model.calls
    assert sorted(call.tool_names) == ["add_numbers", "greet"]
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_async_tool():
    """An async tool is awaited: its result is the value, not a coroutine."""
    model = ScriptedModel(actor=[reply(calls=[("multiply", {"x": 6, "y": 8})])])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are a calculator assistant. Use tools to perform calculations.",
            ),
            "What is 6 times 8?",
            {"multiply": async_multiply},
        )

    assert result.tool_name == "multiply"
    assert result.tool_args == {"x": 6, "y": 8}
    assert result.tool_result == 48
    (call,) = model.calls
    assert call.tool_names == ["multiply"]
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_tool_choice_required():
    """tool_choice='required' reaches the request, and the called tool runs."""
    model = ScriptedModel(actor=[reply(calls=[("do_nothing", {})])])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client("You must use a tool to respond."),
            "Hello, how are you?",
            {"do_nothing": do_nothing},
            tool_choice="required",
        )

    assert result.tool_name == "do_nothing"
    assert result.tool_args == {}
    assert result.tool_result == "Nothing done."
    (call,) = model.calls
    assert call.request["tool_choice"] == "required"
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_no_tools():
    """With no tools the request offers none (and no tool_choice), and the
    model's text is the response."""
    system = "You are a helpful assistant."
    model = ScriptedModel(query_llm=[reply("2 + 2 = 4.")])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(system),
            "What is 2+2?",
            {},
        )

    assert result.tools == []
    assert result.tool_name is None
    assert result.tool_result is None
    assert result.text_response == "2 + 2 = 4."
    assert result.structured_output is None
    (call,) = model.calls
    assert not call.request["tools"]
    assert call.request["tool_choice"] is None
    _sent(call, system, "What is 2+2?")
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  response_format (structured output), scripted                              #
# --------------------------------------------------------------------------- #


def _schema_of(call) -> dict:
    """The JSON schema the request's response_format carries."""
    response_format = call.request["response_format"]
    assert response_format["type"] == "json_schema", response_format
    return response_format["json_schema"]["schema"]


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_no_tools():
    """With response_format and no tools, the request carries the schema and
    the model's JSON text is parsed into the model class."""
    decision = {
        "thoughts": "A $50 expense is within the usual limit.",
        "decision": "approve",
        "confidence": 8,
    }
    model = ScriptedModel(query_llm=[reply(json.dumps(decision))])
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are a decision assistant. Always provide your reasoning.",
            ),
            "Should I approve this expense report for $50?",
            {},
            response_format=DecisionResponse,
        )

    assert result.tool_name is None
    assert result.tool_result is None
    assert isinstance(result.structured_output, DecisionResponse)
    assert result.structured_output.model_dump() == decision
    assert result.text_response == json.dumps(decision)
    (call,) = model.calls
    assert not call.request["tools"]
    assert set(_schema_of(call)["properties"]) == {"thoughts", "decision", "confidence"}
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_with_tools():
    """With response_format AND tools, one reply carries both: the tool runs
    and the JSON content is parsed."""
    model = ScriptedModel(
        actor=[
            reply(
                json.dumps({"thoughts": "Bob asked for a greeting."}),
                calls=[("greet", {"name": "Bob"})],
            ),
        ],
    )
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are a helpful assistant. First think about the request, then use "
                "the appropriate tool. Your response must include your thoughts as JSON.",
            ),
            "Please greet Bob. Think about why this is a good greeting.",
            {"greet": greet, "add_numbers": add_numbers},
            response_format=ThoughtsResponse,
        )

    assert result.tool_name == "greet"
    assert result.tool_args == {"name": "Bob"}
    assert result.tool_result == "Hello, Bob!"
    assert result.structured_output == ThoughtsResponse(
        thoughts="Bob asked for a greeting.",
    )
    (call,) = model.calls
    assert sorted(call.tool_names) == ["add_numbers", "greet"]
    assert set(_schema_of(call)["properties"]) == {"thoughts"}
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_structured_output_complex():
    """A multi-field model: every field, its type and its description reach
    the schema, and a reply is parsed with its types."""
    model = ScriptedModel(
        query_llm=[
            reply(
                '{"thoughts": "Strongly positive wording.", '
                '"decision": "approve", "confidence": 9}',
            ),
        ],
    )
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are a sentiment analyzer. Analyze the sentiment and provide your "
                "decision with confidence.",
            ),
            "The product is absolutely amazing and I love it!",
            {},
            response_format=DecisionResponse,
        )

    assert isinstance(result.structured_output, DecisionResponse)
    assert result.structured_output.decision == "approve"
    assert result.structured_output.confidence == 9
    assert result.structured_output.thoughts == "Strongly positive wording."
    (call,) = model.calls
    schema = _schema_of(call)
    assert set(schema["required"]) == {"thoughts", "decision", "confidence"}
    assert schema["properties"]["confidence"]["type"] == "integer"
    for name, field in DecisionResponse.model_fields.items():
        assert schema["properties"][name]["description"] == field.description
    model.assert_used_up()


# --------------------------------------------------------------------------- #
#  Several tool calls in one decision, scripted                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_multiple_concurrent_tools():
    """One decision with several tool calls runs every one and returns all
    their results, in the order the model called them."""
    model = ScriptedModel(
        actor=[
            reply(
                calls=[
                    ("start_background_task", {"task_name": "data_sync"}),
                    ("send_notification", {"message": "Sync started"}),
                ],
            ),
        ],
    )
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are an efficient assistant. When asked to do multiple things, "
                "call ALL relevant tools in parallel in a single response. Do not call "
                "them one at a time - use parallel tool calls.",
            ),
            "Start a background task called 'data_sync' AND send a notification "
            "saying 'Sync started'. Do both in parallel.",
            {
                "send_notification": send_notification,
                "start_background_task": start_background_task,
            },
            tool_choice="required",
        )

    assert [(t.name, t.args, t.result) for t in result.tools] == [
        (
            "start_background_task",
            {"task_name": "data_sync"},
            "Task 'data_sync' started",
        ),
        (
            "send_notification",
            {"message": "Sync started"},
            "Notification sent: Sync started",
        ),
    ]
    # The first call answers the backward-compatible properties.
    assert result.tool_name == "start_background_task"
    (call,) = model.calls
    assert call.request["tool_choice"] == "required"
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_single_shot_multiple_tools_same_type():
    """The same tool called twice in one decision runs twice, once per call's
    arguments."""
    model = ScriptedModel(
        actor=[
            reply(calls=[("greet", {"name": "Alice"}), ("greet", {"name": "Bob"})]),
        ],
    )
    with scripted(model):
        result = await single_shot_tool_decision(
            _scripted_client(
                "You are an efficient assistant. When asked to do multiple things, "
                "call ALL relevant tools in parallel in a single response.",
            ),
            "Greet both Alice and Bob. Call the greet tool twice in parallel - "
            "once for Alice and once for Bob.",
            {"greet": greet},
            tool_choice="required",
        )

    assert [t.name for t in result.tools] == ["greet", "greet"]
    assert [t.args for t in result.tools] == [{"name": "Alice"}, {"name": "Bob"}]
    assert [t.result for t in result.tools] == ["Hello, Alice!", "Hello, Bob!"]
    model.assert_used_up()
