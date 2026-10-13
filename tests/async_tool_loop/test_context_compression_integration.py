"""Integration tests for context compression wiring in the async tool loop.

The tests that run a whole compression script the model (``loop_model``,
tests/async_tool_loop/conftest.py) and force the threshold with a trigger
the first ``add`` call fires. A compression has two paths, and each test
names the one it takes:

* as shipped (``FORK_SUMMARY``): the summary comes from a fork of the last
  request (a ``compression_fork`` request), and the compactor is never
  called. The fork's scripted answer resets the trigger.
* the fallback: the fork returns no text, so ``compress_and_rebuild`` runs
  the (patched) compactor, which resets the trigger and archives the raw
  transcript in ``raw_archives``.

The turn that must compress is an ``actor`` request: the session's tool
list stays the same, and the loop refuses any call but ``compress_context``
on that turn.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import MagicMock

import unify.common._async_tool.loop as _loop_mod
import unify.common._async_tool.context_compression as _cc_mod
import unify.common.async_tool_loop as _atl_mod
from unify.common._async_tool.context_compression import (
    CompressedMessage,
    CompressedMessages,
)
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from unify.common.tool_spec import ToolSpec
from tests.helpers import _handle_project
from tests.scripted_model import reply

_SYS = (
    "You are in a test. Follow the steps exactly:\n"
    "1. Call `add` with a=2, b=3.\n"
    "2. Call `add` with a=10, b=20.\n"
    "3. Report both results."
)

SUMMARY = "Summary: added 2 + 3 = 5 and 10 + 20 = 30; report both."
FINAL = "5 and 30"
STOPPED = "processed stopped early, no result"
THRESHOLD_NOTICE = "Context window is nearly full."

# The loop's turns up to the compression: add, add (its reply is the first
# over the threshold, since the first add fired the trigger), then the turn
# that must compress.
ADD_FIRST = reply(calls=[("add", {"a": 2, "b": 3})])
ADD_SECOND = reply(calls=[("add", {"a": 10, "b": 20})])
COMPRESS = reply(calls=[("compress_context", {})])


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_threshold_trigger():
    _fire = False

    def trigger():
        nonlocal _fire
        _fire = True

    def reset():
        nonlocal _fire
        _fire = False

    def check(n_tokens, threshold, max_input_tokens):
        return _fire

    return trigger, reset, check


async def _mock_compress(messages, endpoint, **kwargs):
    return CompressedMessages(
        messages=[
            CompressedMessage(
                content=json.dumps(
                    {
                        "role": m.get("role", "user"),
                        "content": f"[c] {str(m.get('content', ''))[:40]}",
                    },
                ),
            )
            for m in messages
        ],
    )


async def _compactor_not_called(messages, endpoint, **kwargs):
    raise AssertionError("the shipped path never calls the compactor")


def _make_add(trigger, calls=None):
    _triggered = False

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        nonlocal _triggered
        if calls is not None:
            calls.append((a, b))
        if not _triggered:
            _triggered = True
            trigger()
        return str(a + b)

    return add


def _summary(then=None):
    """The fork's answer on the shipped path: run *then* (sync or async),
    then return the summary."""

    async def answer(call):
        if then is not None:
            done = then()
            if asyncio.iscoroutine(done):
                await done
        return reply(SUMMARY)

    return answer


def _text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return "" if content is None else str(content)


def _assert_forced_turn(call) -> None:
    """*call* is the turn that must compress: the loop's notice is the last
    message and a tool call is required."""
    assert THRESHOLD_NOTICE in _text(call.messages[-1])
    assert call.request["tool_choice"] == "required"


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_handle_state_preserved_after_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """Events and queues on the handle must be the same objects after compression.

    Path: as shipped (the fork's summary)."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        COMPRESS,
        reply(FINAL),
        compression_fork=[_summary(reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    orig_queue = handle._queue
    orig_cancel = handle._cancel_event
    orig_stop = handle._stop_event

    result = await handle.result()

    assert handle._compression.count == 1
    assert handle._queue is orig_queue
    assert handle._cancel_event is orig_cancel
    assert handle._stop_event is orig_stop
    assert result == FINAL
    _assert_forced_turn(model.calls[2])
    # The restarted loop continues from the summary.
    assert SUMMARY in "\n".join(_text(m) for m in model.calls[-1].messages)
    assert model.kinds() == ["actor", "actor", "actor", "compression_fork", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_nested_inner_compression_outer_unaffected(
    llm_config,
    monkeypatch,
    loop_model,
):
    """Inner loop compresses; outer handle finishes normally.

    Path: as shipped (the fork's summary), in the inner loop."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    inner_refs: dict = {}
    add = _make_add(trigger)

    # Each loop's turns, told apart by the tools the request offers.
    outer_turns = [
        reply(calls=[("spawn_inner", {})]),
        reply(f"The inner loop returned: {FINAL}"),
    ]
    inner_turns = [ADD_FIRST, ADD_SECOND, COMPRESS, reply(FINAL)]

    def route(call):
        turns = outer_turns if "spawn_inner" in call.tool_names else inner_turns
        return turns.pop(0)

    model = loop_model(
        *[route] * (len(outer_turns) + len(inner_turns)),
        compression_fork=[_summary(reset)],
    )

    async def spawn_inner():
        """Spawn an inner async tool loop and return its result."""
        inner_client = new_llm_client(**llm_config)
        inner_client.set_system_message(_SYS)
        h = start_async_tool_loop(
            inner_client,
            message="Add the numbers",
            tools={"add": add},
            timeout=120,
            max_parallel_tool_calls=1,
        )
        inner_refs["handle"] = h
        return await h.result()

    outer_client = new_llm_client(**llm_config)
    outer_client.set_system_message(
        "You are in a test. Call `spawn_inner`. Report what it returns.",
    )

    handle = start_async_tool_loop(
        outer_client,
        message="start",
        tools={"spawn_inner": spawn_inner},
        timeout=240,
    )

    result = await handle.result()
    assert result == f"The inner loop returned: {FINAL}"

    inner_h = inner_refs.get("handle")
    assert inner_h is not None
    assert inner_h._compression.count == 1
    assert handle._compression.count == 0
    assert outer_turns == [] and inner_turns == []
    # The outer loop's second turn reads the inner loop's result.
    assert FINAL in "\n".join(_text(m) for m in model.calls[-1].messages)
    assert model.kinds() == [
        "actor",  # outer: spawn_inner
        "actor",  # inner: add
        "actor",  # inner: add
        "actor",  # inner: compress_context
        "compression_fork",
        "actor",  # inner, restarted: its answer
        "actor",  # outer: its answer
    ]


@pytest.mark.asyncio
@_handle_project
async def test_no_new_tools_when_threshold_triggered(
    llm_config,
    monkeypatch,
    loop_model,
):
    """When the threshold fires with no pending tools, only compress_context
    can be called: a call to another tool on that turn is refused, not run.

    Path: the fallback (the fork returns no text), so the compactor runs and
    the raw transcript is archived."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        reply(calls=[("add", {"a": 1, "b": 1})]),  # refused: must compress
        COMPRESS,
        reply(FINAL),
        compression_fork=[reply("")],
    )
    calls: list[tuple[int, int]] = []
    add = _make_add(trigger, calls)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )

    result = await handle.result()
    assert result == FINAL
    assert handle._compression.count == 1
    assert calls == [(2, 3), (10, 20)]
    _assert_forced_turn(model.calls[2])
    _assert_forced_turn(model.calls[3])

    archived = handle._compression.raw_archives[0]
    threshold_idx = next(
        (
            i
            for i, m in enumerate(archived)
            if "must call" in str(m.get("content", ""))
            and "compress_context" in str(m.get("content", ""))
        ),
        None,
    )
    assert threshold_idx is not None, "Expected 'must call compress_context' in archive"

    # After the notice, the only call that ran is compress_context: the add
    # call was answered with the refusal.
    called = {
        tc["id"]: tc["function"]["name"]
        for m in archived[threshold_idx:]
        if m.get("role") == "assistant"
        for tc in m.get("tool_calls") or []
    }
    assert sorted(called.values()) == ["add", "compress_context"]
    answers = {
        m.get("tool_call_id"): str(m.get("content", ""))
        for m in archived[threshold_idx:]
        if m.get("role") == "tool"
    }
    (add_id,) = [i for i, name in called.items() if name == "add"]
    assert "Available now: `compress_context`." in answers[add_id]
    assert model.kinds() == [
        "actor",
        "actor",
        "actor",
        "actor",
        "compression_fork",
        "actor",
    ]


@pytest.mark.asyncio
@_handle_project
async def test_stop_carries_over_during_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """A stop during the compression window ends the restarted loop before
    it asks the model anything.

    Path: as shipped; the stop comes while the fork writes its summary."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    handle_ref: dict = {}

    async def _stop_and_reset():
        await handle_ref["handle"].stop()
        reset()

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        COMPRESS,
        compression_fork=[_summary(_stop_and_reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )
    handle_ref["handle"] = handle

    result = await handle.result()
    assert result == STOPPED
    assert handle._compression.count == 1
    assert handle._stop_event.is_set()
    # No turn after the fork.
    assert model.kinds() == ["actor", "actor", "actor", "compression_fork"]


@pytest.mark.asyncio
@_handle_project
async def test_interjection_carries_over_during_compression(
    llm_config,
    monkeypatch,
    loop_model,
):
    """Interjection queued during compression is received by the new loop.

    Path: as shipped; the interjection comes while the fork writes its
    summary."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)
    monkeypatch.setattr(_cc_mod, "compress_messages", _compactor_not_called)

    handle_ref: dict = {}
    interjection_text = "URGENT: also compute 10 + 20"

    async def _interject_and_reset():
        await handle_ref["handle"]._queue.put(interjection_text)
        reset()

    model = loop_model(
        ADD_FIRST,
        ADD_SECOND,
        COMPRESS,
        reply(FINAL),
        compression_fork=[_summary(_interject_and_reset)],
    )
    add = _make_add(trigger)
    client = new_llm_client(**llm_config)
    client.set_system_message(
        "You are in a test. Follow the steps exactly:\n"
        "1. Call `add` with a=2, b=3.\n"
        "2. Call `add` with a=10, b=20.\n"
        "3. Report both results.\n"
        "If you receive additional instructions, follow them.",
    )

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
    )
    handle_ref["handle"] = handle

    result = await handle.result()
    assert result == FINAL
    assert handle._compression.count == 1
    # The restarted loop's first request carries the summary and then the
    # interjection.
    restarted = "\n".join(_text(m) for m in model.calls[-1].messages)
    assert SUMMARY in restarted
    assert interjection_text in restarted
    assert restarted.index(SUMMARY) < restarted.index(interjection_text)
    assert all(
        interjection_text not in _text(m) for c in model.calls[:-1] for m in c.messages
    )
    assert model.kinds() == ["actor", "actor", "actor", "compression_fork", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_enable_compression_false(llm_config, monkeypatch, loop_model):
    """enable_compression=False suppresses all compression machinery."""
    monkeypatch.setattr(_loop_mod, "context_over_threshold", lambda *a, **kw: True)

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    model = loop_model(ADD_FIRST, ADD_SECOND, reply(FINAL))
    client = new_llm_client(**llm_config)
    client.set_system_message(_SYS)

    handle = start_async_tool_loop(
        client=client,
        message="Add the numbers",
        tools={"add": add},
        timeout=120,
        max_parallel_tool_calls=1,
        enable_compression=False,
    )

    result = await handle.result()
    assert result == FINAL
    assert handle._compression.count == 0
    assert all("compress_context" not in c.tool_names for c in model.calls)
    assert all(THRESHOLD_NOTICE not in _text(c.messages[-1]) for c in model.calls)
    assert model.kinds() == ["actor", "actor", "actor"]


@pytest.mark.asyncio
@_handle_project
async def test_tool_quota_survives_compression_restart(llm_config, monkeypatch):
    """A quota-exhausted tool must stay hidden after compression restarts the loop."""
    monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress)

    counter = {"n": 0}
    phase = {"n": 0}

    async def short_tool():
        counter["n"] += 1
        return "ok"

    def _assistant_tool_call(name: str, call_id: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }

    async def _fake_generate_with_preprocess(_client, preprocess_msgs, **gen_kwargs):
        tool_names = {
            tool.get("function", {}).get("name") for tool in gen_kwargs.get("tools", [])
        }

        if phase["n"] == 0 and "short_tool" in tool_names:
            phase["n"] = 1
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("short_tool", "call_initial_1"),
                        _assistant_tool_call("short_tool", "call_initial_2"),
                    ],
                },
            )
            return {"ok": True}

        if phase["n"] == 1 and "compress_context" in tool_names:
            phase["n"] = 2
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("compress_context", "call_compress"),
                    ],
                },
            )
            return {"ok": True}

        if phase["n"] == 2 and "short_tool" in tool_names:
            phase["n"] = 3
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("short_tool", "call_after_compression"),
                    ],
                },
            )
            return {"ok": True}

        phase["n"] = 4
        _client.messages.append(
            {"role": "assistant", "content": "done", "tool_calls": None},
        )
        return {"ok": True}

    monkeypatch.setattr(
        _loop_mod,
        "generate_with_preprocess",
        _fake_generate_with_preprocess,
        raising=True,
    )

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client=client,
        message="Start by calling short_tool twice, then compress context.",
        tools={"short_tool": ToolSpec(fn=short_tool, max_total_calls=2)},
        prune_tool_duplicates=False,
        timeout=30,
        max_steps=20,
    )

    await handle.result()

    assert handle._compression.count == 1
    assert phase["n"] == 4
    assert counter["n"] == 2


@pytest.mark.asyncio
@_handle_project
async def test_tool_policy_history_survives_compression_restart(
    llm_config,
    monkeypatch,
):
    """3-arg tool policies should see calls made before a compression restart."""
    monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress)

    phase = {"n": 0}
    call_log: list[str] = []
    policy_log: list[tuple[int, list[str], int]] = []

    async def tool_a():
        call_log.append("a")
        return "a"

    async def tool_b():
        call_log.append("b")
        return "b"

    def policy(step, tools, called_tools):
        policy_log.append((step, list(called_tools), phase["n"]))
        return "auto", tools

    def _assistant_tool_call(name: str, call_id: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }

    async def _fake_generate_with_preprocess(_client, preprocess_msgs, **gen_kwargs):
        tool_names = {
            tool.get("function", {}).get("name") for tool in gen_kwargs.get("tools", [])
        }

        if phase["n"] == 0 and "tool_a" in tool_names:
            phase["n"] = 1
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_assistant_tool_call("tool_a", "call_a")],
                },
            )
            return {"ok": True}

        if phase["n"] == 1 and "compress_context" in tool_names:
            phase["n"] = 2
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("compress_context", "call_compress"),
                    ],
                },
            )
            return {"ok": True}

        phase["n"] = 3
        _client.messages.append(
            {"role": "assistant", "content": "done", "tool_calls": None},
        )
        return {"ok": True}

    monkeypatch.setattr(
        _loop_mod,
        "generate_with_preprocess",
        _fake_generate_with_preprocess,
        raising=True,
    )

    client = new_llm_client(**llm_config)
    handle = start_async_tool_loop(
        client=client,
        message="Call tool_a, compress, then finish.",
        tools={"tool_a": tool_a, "tool_b": tool_b},
        tool_policy=policy,
        timeout=30,
        max_steps=20,
    )

    await handle.result()

    assert handle._compression.count == 1
    assert call_log == ["a"]
    after_restart_entries = [entry for entry in policy_log if entry[2] == 2]
    assert after_restart_entries
    assert "tool_a" in after_restart_entries[0][1]
    assert "compress_context" in after_restart_entries[0][1]


@pytest.mark.asyncio
@_handle_project
async def test_max_steps_survives_compression_restart(llm_config, monkeypatch):
    """max_steps should apply to the logical loop, not just the compressed transcript."""
    monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress)

    phase = {"n": 0}
    post_restart_calls = {"n": 0}

    async def starter_tool():
        return "started"

    async def post_restart_tool():
        post_restart_calls["n"] += 1
        return "should not run"

    def _assistant_tool_call(name: str, call_id: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }

    async def _fake_generate_with_preprocess(_client, preprocess_msgs, **gen_kwargs):
        tool_names = {
            tool.get("function", {}).get("name") for tool in gen_kwargs.get("tools", [])
        }

        if phase["n"] == 0 and "starter_tool" in tool_names:
            phase["n"] = 1
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_assistant_tool_call("starter_tool", "call_start")],
                },
            )
            return {"ok": True}

        if phase["n"] == 1 and "compress_context" in tool_names:
            phase["n"] = 2
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("compress_context", "call_compress"),
                    ],
                },
            )
            return {"ok": True}

        if phase["n"] == 2 and "post_restart_tool" in tool_names:
            phase["n"] = 3
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("post_restart_tool", "call_after_restart"),
                    ],
                },
            )
            return {"ok": True}

        phase["n"] = 4
        _client.messages.append(
            {"role": "assistant", "content": "done", "tool_calls": None},
        )
        return {"ok": True}

    monkeypatch.setattr(
        _loop_mod,
        "generate_with_preprocess",
        _fake_generate_with_preprocess,
        raising=True,
    )

    client = new_llm_client(**llm_config)
    result = await start_async_tool_loop(
        client=client,
        message="Start, compress, then continue.",
        tools={
            "starter_tool": starter_tool,
            "post_restart_tool": post_restart_tool,
        },
        timeout=30,
        max_steps=7,
        raise_on_limit=False,
    ).result()

    assert "max_steps" in result
    assert post_restart_calls["n"] == 0


@pytest.mark.asyncio
@_handle_project
async def test_consecutive_failures_survive_compression_restart(
    llm_config,
    monkeypatch,
):
    """Consecutive tool-failure counts should not reset after compression."""
    monkeypatch.setattr(_cc_mod, "compress_messages", _mock_compress)

    phase = {"n": 0}
    failing_calls = {"n": 0}

    async def failing_tool():
        failing_calls["n"] += 1
        raise RuntimeError("synthetic tool failure")

    def _assistant_tool_call(name: str, call_id: str) -> dict:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }

    async def _fake_generate_with_preprocess(_client, preprocess_msgs, **gen_kwargs):
        tool_names = {
            tool.get("function", {}).get("name") for tool in gen_kwargs.get("tools", [])
        }

        if phase["n"] == 0 and "failing_tool" in tool_names:
            phase["n"] = 1
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_assistant_tool_call("failing_tool", "call_fail_1")],
                },
            )
            return {"ok": True}

        if phase["n"] == 1 and "compress_context" in tool_names:
            phase["n"] = 2
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        _assistant_tool_call("compress_context", "call_compress"),
                    ],
                },
            )
            return {"ok": True}

        if phase["n"] == 2 and "failing_tool" in tool_names:
            phase["n"] = 3
            _client.messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_assistant_tool_call("failing_tool", "call_fail_2")],
                },
            )
            return {"ok": True}

        phase["n"] = 4
        _client.messages.append(
            {"role": "assistant", "content": "done", "tool_calls": None},
        )
        return {"ok": True}

    monkeypatch.setattr(
        _loop_mod,
        "generate_with_preprocess",
        _fake_generate_with_preprocess,
        raising=True,
    )

    client = new_llm_client(**llm_config)
    with pytest.raises(RuntimeError, match="too many consecutive tool failures"):
        await start_async_tool_loop(
            client=client,
            message="Fail once, compress, fail again.",
            tools={"failing_tool": failing_tool},
            timeout=30,
            max_steps=20,
            max_consecutive_failures=2,
        ).result()

    assert failing_calls["n"] == 2


@pytest.mark.asyncio
@_handle_project
async def test_compression_failure_returns_gracefully(monkeypatch):
    """When compression fails, result() returns gracefully.

    Compression first
    forks the conversation for a summary and uses ``compress_and_rebuild``
    only when the fork yields none, so both are made to fail. The model is
    scripted: the first turn fills the context, the next must compress.
    """
    from tests import cache_discipline_helpers as h

    async def _no_fork(self, cfg):
        return None

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(_atl_mod.AsyncToolLoopHandle, "_summarise_as_fork", _no_fork)
    monkeypatch.setattr(_atl_mod, "compress_and_rebuild", _boom)

    async def add(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: The first.
            b: The second.
        """
        return a + b

    replies = [
        lambda: h.completion(
            calls=[("add", {"a": 1, "b": 2})],
            prompt_tokens=900_000,
        ),
        lambda: h.completion(calls=[("compress_context", {})]),
    ]
    with h.scripted(replies):
        handle = start_async_tool_loop(
            h.new_client(),
            "Add the numbers",
            {"add": add},
            log_steps=False,
            timeout=30,
        )
        result = await asyncio.wait_for(handle.result(), 30)
    assert result == "processed stopped early, no result"
    assert handle._compression.count == 0


@pytest.mark.asyncio
async def test_compression_restart_omits_parent_chat_context(monkeypatch):
    """Compression restarts must not re-inject the original parent context blob."""

    from unify.common._async_tool.context_compression import RebuildResult
    from unify.common.async_tool_loop import AsyncToolLoopHandle

    captured: dict[str, object] = {}

    async def _fake_inner(_client, _message, _tools, **kwargs):
        captured["parent_chat_context"] = kwargs.get("parent_chat_context")
        return "done"

    async def _fake_compress(_compression, _messages, _endpoint, _tools):
        return RebuildResult(
            system_msgs=[{"role": "system", "content": "compressed"}],
            tools={},
        )

    monkeypatch.setattr(_atl_mod, "async_tool_loop_inner", _fake_inner)
    monkeypatch.setattr(_atl_mod, "compress_and_rebuild", _fake_compress)

    parent_ctx = [{"role": "user", "content": "large parent transcript"}]
    handle = AsyncToolLoopHandle(
        task=asyncio.create_task(asyncio.sleep(0)),
        interject_queue=asyncio.Queue(),
        cancel_event=asyncio.Event(),
        stop_event=asyncio.Event(),
    )
    handle._client = MagicMock()
    handle._client.messages = [{"role": "user", "content": "loop"}]
    handle._client.endpoint = "openai/gpt-4o@openrouter"
    handle._loop_config = {
        "loop_id": "test-loop",
        "parent_lineage": [],
        "tools": {},
        "parent_chat_context": parent_ctx,
        "max_consecutive_failures": 3,
        "prune_tool_duplicates": True,
        "propagate_chat_context": True,
        "caller_description": None,
        "log_steps": False,
        "max_steps": None,
        "timeout": None,
        "raise_on_limit": False,
        "tool_policy": None,
        "preprocess_msgs": None,
        "response_format": None,
        "max_parallel_tool_calls": None,
        "persist": False,
        "prompt_caching": False,
        "time_awareness": False,
        "enable_compression": True,
        "extra_compression_tools": None,
        "clarification_queues": None,
        "on_clarification_request": None,
        "on_clarification_answer": None,
        "on_notify": None,
        "runtime_state": handle._runtime_state,
    }

    await handle._restart_with_compressed_context()
    await asyncio.sleep(0)

    assert captured["parent_chat_context"] is None
