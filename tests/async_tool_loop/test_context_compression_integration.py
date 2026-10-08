"""Integration tests for context compression wiring in the async tool loop."""

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
from tests.async_helpers import _wait_for_condition

pytestmark = pytest.mark.llm_call

_SYS = (
    "You are in a test. Follow the steps exactly:\n"
    "1. Call `add` with a=2, b=3.\n"
    "2. Call `add` with a=10, b=20.\n"
    "3. Report both results."
)


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


def _make_add(trigger):
    _triggered = False

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        nonlocal _triggered
        if not _triggered:
            _triggered = True
            trigger()
        return str(a + b)

    return add


def _msg_contains(client, snippet):
    async def _check():
        for m in client.messages or []:
            if snippet in str(m.get("content", "")):
                return True
        return False

    return _check


async def _wait_compression(handle, timeout=60):
    async def _done():
        return handle._compression.count >= 1

    await _wait_for_condition(_done, poll=0.1, timeout=timeout)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@_handle_project
async def test_handle_state_preserved_after_compression(llm_config, monkeypatch):
    """Events and queues on the handle must be the same objects after compression."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

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

    assert handle._compression.count >= 1
    assert handle._queue is orig_queue
    assert handle._cancel_event is orig_cancel
    assert handle._stop_event is orig_stop
    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_nested_inner_compression_outer_unaffected(llm_config, monkeypatch):
    """Inner loop compresses; outer handle finishes normally."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

    inner_refs: dict = {}
    add = _make_add(trigger)

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
    assert result is not None

    inner_h = inner_refs.get("handle")
    assert inner_h is not None
    assert inner_h._compression.count >= 1


@pytest.mark.asyncio
@_handle_project
async def test_no_new_tools_when_threshold_triggered(llm_config, monkeypatch):
    """When threshold fires with no pending tools, only compress_context is callable."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    async def _compress_and_reset(messages, endpoint, **kwargs):
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_and_reset)

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

    result = await handle.result()
    assert handle._compression.count >= 1

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

    for m in archived[threshold_idx:]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                assert (
                    name == "compress_context"
                ), f"Expected only compress_context after threshold, got {name}"

    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_stop_carries_over_during_compression(llm_config, monkeypatch):
    """Stop set during the compression window causes the new loop to exit."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    handle_ref: dict = {}

    async def _compress_with_stop(messages, endpoint, **kwargs):
        h = handle_ref.get("handle")
        if h:
            h._stop_event.set()
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_with_stop)

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
    assert result is not None
    assert handle._compression.count >= 1


@pytest.mark.asyncio
@_handle_project
async def test_interjection_carries_over_during_compression(llm_config, monkeypatch):
    """Interjection queued during compression is received by the new loop."""
    trigger, reset, check = _make_threshold_trigger()
    monkeypatch.setattr(_loop_mod, "context_over_threshold", check)

    handle_ref: dict = {}
    interjection_text = "URGENT: also compute 10 + 20"

    async def _compress_with_interjection(messages, endpoint, **kwargs):
        h = handle_ref.get("handle")
        if h:
            await h._queue.put(interjection_text)
        reset()
        return await _mock_compress(messages, endpoint)

    monkeypatch.setattr(_cc_mod, "compress_messages", _compress_with_interjection)

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

    result_task = asyncio.create_task(handle.result())

    await _wait_compression(handle)

    await _wait_for_condition(
        _msg_contains(client, interjection_text),
        poll=0.1,
        timeout=60,
    )

    result = await result_task
    assert result is not None


@pytest.mark.asyncio
@_handle_project
async def test_enable_compression_false(llm_config, monkeypatch):
    """enable_compression=False suppresses all compression machinery."""
    monkeypatch.setattr(_loop_mod, "context_over_threshold", lambda *a, **kw: True)

    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

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
    assert result is not None
    assert handle._compression.count == 0


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

    Under UNIFY_CACHE_DISCIPLINE (the baked default) compression first
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
