"""
pytest tests for the async-tool loop helpers.

The model is scripted (``tests/scripted_model.py``): each test writes the
model's replies, every request is still built as in production for the
``llm_config`` model, and nothing leaves the process. The behaviours covered:

* single-tool “happy path”
* a turn's calls run in order, each to completion, before the next model call
* recovery after a tool error and counter reset
* aborting when too many consecutive tool failures occur
* a realistic mix of async + sync tools
* duplicate tool calls kept or pruned, no-tool flows and the concurrency limit
"""

from __future__ import annotations

import asyncio
import time
from unify.common.llm_client import new_llm_client
from tests.async_helpers import (
    _wait_for_tool_request,
)
from tests.scripted_model import ScriptedModel, reply, scripted

import pytest

# --------------------------------------------------------------------------- #
#  MODULE UNDER TEST                                                          #
# --------------------------------------------------------------------------- #
from unify.common.async_tool_loop import start_async_tool_loop


# --------------------------------------------------------------------------- #
#  TOOL IMPLEMENTATIONS (sync + async)                                        #
# --------------------------------------------------------------------------- #
def add(x: int, y: int) -> int:
    return x + y


def divide(a: int, b: int) -> float:  # may raise
    return a / b


def launch() -> None:
    raise Exception


async def fast_tool(res: str = "fast") -> str:
    await asyncio.sleep(0.05)
    return res


async def slow_tool(res: str = "slow") -> str:
    await asyncio.sleep(0.3)
    return res


def count_tool_messages(client) -> int:
    return sum(1 for m in client.messages if m["role"] == "tool")


# --------------------------------------------------------------------------- #
#  HAPPY PATH – single synchronous tool                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_happy_path_single_sync_tool(llm_config):
    model = ScriptedModel(
        actor=[reply(calls=[("add", {"x": 2, "y": 3})]), reply("5")],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        answer = await start_async_tool_loop(
            client,
            message="Add 2 and 3 using the `add` tool and answer with the result only.",
            tools={"add": add},
        ).result()

    assert answer.strip().startswith("5")
    assert count_tool_messages(client) == 1
    assert [m["content"] for m in client.messages if m["role"] == "tool"] == ["5"]
    assert model.kinds() == ["actor", "actor"]


# --------------------------------------------------------------------------- #
#  CONCURRENT sync/async tools                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_turns_calls_run_in_order_before_the_next_model_call():
    """
    One turn asks for `fast` and `slow`: the loop runs them in call order,
    each to completion (no overlap), and calls the model again only once
    both have finished. The model is scripted: nothing leaves the process.
    """
    from tests import cache_discipline_helpers as h

    events: list[tuple[str, float]] = []

    async def slow():
        """Take a while."""
        events.append(("slow_start", time.monotonic()))
        await asyncio.sleep(0.30)
        events.append(("slow_end", time.monotonic()))
        return "slow"

    async def fast():
        """Return at once."""
        events.append(("fast_start", time.monotonic()))
        await asyncio.sleep(0.05)
        events.append(("fast_end", time.monotonic()))
        return "fast"

    def _turn(calls):
        def reply():
            events.append(("generate", time.monotonic()))
            return h.completion(calls=calls) if calls else h.completion(content="ok")

        return reply

    with h.scripted([_turn([("slow", {}), ("fast", {})]), _turn([])]):
        client = h.new_client()
        answer = await start_async_tool_loop(
            client,
            message="Call both tools, then reply with 'ok'.",
            tools={"fast": fast, "slow": slow},
            log_steps=False,
        ).result()

    assert answer == "ok"
    names = [e for e, _ in events]
    assert names == [
        "generate",
        "slow_start",
        "slow_end",
        "fast_start",
        "fast_end",
        "generate",
    ]
    results = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert results == ["slow", "fast"]


# --------------------------------------------------------------------------- #
#  RECOVERY AFTER A FAILURE & COUNTER RESET                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_recovers_after_failure(llm_config):
    model = ScriptedModel(
        actor=[
            reply(calls=[("divide", {"a": 4, "b": 0})]),
            reply(calls=[("divide", {"a": 4, "b": 2})]),
            reply("2"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        answer = await start_async_tool_loop(
            client,
            message=(
                "Call `divide` with a=4 and b=0, then with a=4 and b=2, and "
                "reply with the result only."
            ),
            tools={"divide": divide},
        ).result()

    assert "2" in answer.strip()

    tool_msgs = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert "ZeroDivisionError" in (tool_msgs[0] or "")
    assert "2" in tool_msgs[1]
    assert model.kinds() == ["actor", "actor", "actor"]


# --------------------------------------------------------------------------- #
#  ABORT AFTER MAX CONSECUTIVE FAILURES                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_aborts_after_too_many_failures(llm_config):
    from unify.common._async_tool.loop import MAX_CONSECUTIVE_FAILURES

    model = ScriptedModel(
        actor=[reply(calls=[("launch", {})])] * MAX_CONSECUTIVE_FAILURES,
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        with pytest.raises(RuntimeError):
            await start_async_tool_loop(
                client,
                message=("Please run the launch tool."),
                tools={"launch": launch},
            ).result()

    # The last allowed failure ends the loop: the model is not asked again.
    assert model.kinds() == ["actor"] * MAX_CONSECUTIVE_FAILURES


# --------------------------------------------------------------------------- #
#  REALISTIC MIX – first async, then sync                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_mixed_sync_async_tools(llm_config):
    model = ScriptedModel(
        actor=[
            reply(calls=[("fast_tool", {})]),
            reply(calls=[("add", {"x": 6, "y": 7})]),
            reply("13"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        answer = await start_async_tool_loop(
            client,
            message=(
                "Call the async tool `fast_tool` (which just returns a token),   "
                "then call the sync `add` tool with 6 and 7, and finally reply "
                "with the result only."
            ),
            tools={"fast_tool": fast_tool, "add": add},
        ).result()

    assert "13" in answer.strip()
    tool_msgs = [m["content"] for m in client.messages if m["role"] == "tool"]
    assert tool_msgs == ["fast", "13"]
    assert model.kinds() == ["actor", "actor", "actor"]


# --------------------------------------------------------------------------- #
#  PRETTY PRINTING – tool returns pure JSON string                            #
# --------------------------------------------------------------------------- #
def emit_json() -> str:
    # Compact JSON string (no spaces/newlines). The loop should pretty‑print it.
    return '{"foo":1,"bar":[2,3],"baz":{"ok":true}}'


@pytest.mark.asyncio
async def test_pretty_prints_json_string_tool_result(llm_config):
    model = ScriptedModel(actor=[reply(calls=[("emit_json", {})]), reply("ok")])
    with scripted(model):
        client = new_llm_client(**llm_config)

        # The model calls the tool once, then replies. The result should be
        # pretty‑printed in the transcript.
        _ = await start_async_tool_loop(
            client,
            message=(
                "Call `emit_json` once with empty arguments {}; after receiving "
                "the tool reply, answer with the single word 'ok'."
            ),
            tools={"emit_json": emit_json},
        ).result()

    # Find the tool message for emit_json
    tool_msgs = [
        m
        for m in client.messages
        if m.get("role") == "tool" and m.get("name") == "emit_json"
    ]
    assert tool_msgs, "Expected a tool reply from emit_json"
    content = tool_msgs[0].get("content") or ""

    # It should now be a pretty‑printed JSON object string (not a quoted JSON string)
    assert isinstance(content, str)
    assert content.strip().startswith(
        "{",
    ), "Content should start with a JSON object, not a quoted string"
    assert not content.strip().startswith(
        '"{',
    ), "Content should not be a double‑encoded JSON string"
    assert "\n" in content, "Pretty‑printed JSON should contain newlines"
    assert (
        '"foo": 1' in content
    ), "Pretty‑printed JSON should include spaces after colons"
    assert model.kinds() == ["actor", "actor"]


@pytest.mark.asyncio
async def test_duplicate_tool_calls_are_pruned(
    llm_config,
) -> None:  # noqa: D401
    """Verify that a turn's duplicate tool calls run once."""

    log: list[str] = []

    async def echo(text: str) -> str:
        """Minimal echo tool used only to count invocations."""
        log.append(text)
        return text.upper()

    echo.__name__ = "echo"
    echo.__qualname__ = "echo"

    twice = [("echo", {"text": "hello"}), ("echo", {"text": "hello"})]
    pruned = ScriptedModel(actor=[reply(calls=twice), reply("done")])
    with scripted(pruned):
        client = new_llm_client(**llm_config)
        await start_async_tool_loop(
            client=client,
            message="Call echo twice with 'hello', then reply with a short sentence.",
            tools={"echo": echo},
        ).result()
    # Only one call executes from the turn (the duplicate is pruned)
    assert log == ["hello"], "Only the first of the duplicate calls should run"

    # The first assistant turn with tool calls, then the tool results before the next assistant turn
    first_assistant_idx = next(
        i
        for i, m in enumerate(client.messages)
        if m.get("role") == "assistant" and m.get("tool_calls")
    )
    tool_results_after_first = []
    for m in client.messages[first_assistant_idx + 1 :]:
        if m.get("role") == "tool" and m.get("name") == "echo":
            tool_results_after_first.append(m)
        elif m.get("role") == "assistant":
            break
    assert (
        len(tool_results_after_first) == 1
    ), "Only 1 tool result should follow the assistant turn"
    assert pruned.kinds() == ["actor", "actor"]


# --------------------------------------------------------------------------- #
#  NO-TOOLS FLOWS                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_no_tools_with_system_message(llm_config) -> None:
    """
    Verify that the loop completes correctly when **no** tools are available
    and the conversation starts with a system prompt.

    The assistant must answer directly without any tool calls.
    """
    model = ScriptedModel(query_llm=[reply("Hello there!")])
    with scripted(model):
        client = new_llm_client(**llm_config).set_system_message(
            "You are a helpful assistant.",
        )

        answer = await start_async_tool_loop(
            client,
            message="Just reply with a friendly greeting – no tools are available.",
            tools={},  # ← empty tool-kit
        ).result()

    # The assistant must answer directly and never insert any tool messages.
    assert answer.strip() == "Hello there!"
    assert count_tool_messages(client) == 0
    assert any(
        m["role"] == "system" and "You are a helpful assistant." in str(m["content"])
        for m in model.calls[0].messages
    )
    # A loop with no tools sends a request with none, which the scripted model
    # classifies as a query_llm call.
    assert model.kinds() == ["query_llm"]


@pytest.mark.asyncio
async def test_no_tools_without_explicit_system_message(llm_config) -> None:
    """
    No tools, no explicit system message provided by the caller.

    The assistant must answer directly without any tool calls.
    """
    model = ScriptedModel(query_llm=[reply("Hello back!")])
    with scripted(model):
        client = new_llm_client(**llm_config)

        answer = await start_async_tool_loop(
            client,
            message="Say hello back to me – there are no tools at all.",
            tools={},  # ← still an empty tool-kit
        ).result()

    assert answer.strip() == "Hello back!"
    assert count_tool_messages(client) == 0
    assert model.kinds() == ["query_llm"]


# --------------------------------------------------------------------------- #
#  ONE CALL AT A TIME                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_turns_calls_never_overlap(llm_config) -> None:  # noqa: D401
    """The model asks for the same tool twice in one turn: the second call
    starts only once the first ends (**no overlap > 1**)."""

    events: list[tuple[str, float]] = []

    async def limited(label: str) -> str:
        events.append(("start", time.monotonic()))
        await asyncio.sleep(0.15)
        events.append(("end", time.monotonic()))
        return label.upper()

    limited.__name__ = "limited"
    limited.__qualname__ = "limited"

    tools = {"limited": limited}

    model = ScriptedModel(
        actor=[
            reply(
                calls=[
                    ("limited", {"label": "one"}),
                    ("limited", {"label": "two"}),
                ],
            ),
            reply("done"),
        ],
    )
    with scripted(model):
        client = new_llm_client(**llm_config)

        # Kick off the interactive loop *without* awaiting the final result
        # yet so that we can synchronise on the **first** tool request.
        handle = start_async_tool_loop(
            client=client,
            message=(
                "Invoke `limited` twice *concurrently* – once with 'one', once "
                "with 'two' – then reply 'done'."
            ),
            tools=tools,
        )

        await _wait_for_tool_request(client, "limited", timeout=30)

        # Now wait for the whole loop to finish.
        await handle.result()

    starts = [t for e, t in events if e == "start"]
    ends = [t for e, t in events if e == "end"]

    # Sanity: any start must be paired with an end
    assert len(starts) == len(
        ends,
    ), "Mismatched start/end counts – tool never returned?"

    assert len(starts) == 2, "Both calls of the turn should run."

    # Verify the core requirement: that the peak concurrency never exceeded 1.
    timeline = sorted(events, key=lambda p: p[1])
    running = peak = 0
    for kind, _ in timeline:
        running += 1 if kind == "start" else -1
        peak = max(peak, running)

    assert (
        peak == 1
    ), "More than one call of a turn ran at once."
    assert model.kinds() == ["actor", "actor"]
