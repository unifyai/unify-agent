"""
Stopping an async tool loop.

stop() queues a mirror of itself before it signals cancellation. The loop
records the mirror (one ``steer(stop)`` per running child, forwarded to each)
and then ends without sending the LLM another request.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unify.common._async_tool import loop as _loop
from unify.common._async_tool.utils import get_handle_paused_state
from unify.common.async_tool_loop import AsyncToolLoopHandle, start_async_tool_loop
from unify.common.llm_client import new_llm_client

_STOPPED = "processed stopped early, no result"


def _count_llm_requests(monkeypatch) -> dict[str, int]:
    """Count the requests the loop dispatches; each still reaches the client."""
    sent = {"n": 0}
    dispatch = _loop.generate_with_preprocess

    def counting_dispatch(*args, **kwargs):
        sent["n"] += 1
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(_loop, "generate_with_preprocess", counting_dispatch)
    return sent


def echo(text: str) -> str:
    """Return *text* unchanged."""
    return text


def _start_holding(client) -> tuple[AsyncToolLoopHandle, asyncio.Event]:
    """Start a loop whose one tool call runs until the loop is stopped.

    The call is seeded, so the tool starts without a model turn. The event is
    set once the tool runs.
    """
    started = asyncio.Event()

    async def hold() -> str:
        """Run until stopped."""
        started.set()
        await asyncio.Event().wait()
        return "released"

    handle = start_async_tool_loop(
        client,
        [
            {"role": "user", "content": "Hold until told otherwise."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_hold",
                        "type": "function",
                        "function": {"name": "hold", "arguments": "{}"},
                    },
                ],
            },
        ],
        {"hold": hold},
    )
    return handle, started


def _last_steer(client) -> dict:
    """The arguments of the one steer call in the last assistant turn."""
    last_turn = next(m for m in reversed(client.messages) if m["role"] == "assistant")
    (steer,) = last_turn["tool_calls"]
    assert steer["function"]["name"] == "steer"
    return json.loads(steer["function"]["arguments"])


@pytest.mark.asyncio
async def test_stop_before_first_turn_sends_no_request(
    llm_config,
    monkeypatch,
    unify_logs,
):
    """A loop stopped before its first LLM step ends without a request, and
    still logs the stop that step would otherwise have flushed."""
    requests = _count_llm_requests(monkeypatch)
    handle = start_async_tool_loop(
        new_llm_client(**llm_config),
        "Echo something, then say 'ok'.",
        {"echo": echo},
    )

    await handle.stop("done early")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0
    assert "Stop requested – reason: done early" in unify_logs.text


@pytest.mark.asyncio
async def test_stop_while_a_turn_is_built_sends_no_request(
    llm_config,
    monkeypatch,
):
    """A stop that lands while the loop builds a turn is drained, not sent."""
    requests = _count_llm_requests(monkeypatch)
    build = _loop.ensure_placeholders_for_pending

    async def stop_mid_build(*args, **kwargs):
        await handle.stop("stopped mid-build")
        return await build(*args, **kwargs)

    monkeypatch.setattr(_loop, "ensure_placeholders_for_pending", stop_mid_build)
    handle = start_async_tool_loop(
        new_llm_client(**llm_config),
        "Echo something, then say 'ok'.",
        {"echo": echo},
    )

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0


@pytest.mark.asyncio
async def test_stop_with_a_tool_running_sends_no_further_request(
    llm_config,
    monkeypatch,
):
    """Stopping mid-tool records and forwards the steer(stop), then ends."""
    requests = _count_llm_requests(monkeypatch)
    client = new_llm_client(**llm_config)
    handle, started = _start_holding(client)
    await asyncio.wait_for(started.wait(), timeout=30)

    await handle.stop("done")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0
    assert _last_steer(client) == {
        "call_id": "call_hold",
        "action": "stop",
        "payload": "done",
    }


@pytest.mark.asyncio
async def test_stop_while_paused_with_a_tool_running_records_the_steer(
    llm_config,
    monkeypatch,
):
    """A stop that reaches a paused loop waiting on its running tool still
    records and forwards the steer(stop), then ends without a request."""
    parked = asyncio.Event()
    find_unreplied = _loop.find_unreplied_assistant_entries

    def spot_the_pause_gate(client):
        # While the loop is paused only the pause gate looks for unreplied
        # calls, and it looks just before it waits on the running tool.
        if get_handle_paused_state(handle):
            parked.set()
        return find_unreplied(client)

    monkeypatch.setattr(
        _loop,
        "find_unreplied_assistant_entries",
        spot_the_pause_gate,
    )
    requests = _count_llm_requests(monkeypatch)
    client = new_llm_client(**llm_config)
    handle, started = _start_holding(client)
    await asyncio.wait_for(started.wait(), timeout=30)
    await handle.pause()
    await asyncio.wait_for(parked.wait(), timeout=30)

    await handle.stop("done")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0
    assert _last_steer(client) == {
        "call_id": "call_hold",
        "action": "stop",
        "payload": "done",
    }


@pytest.mark.llm_call
@pytest.mark.asyncio
async def test_stop_while_persist_waits_logs_the_stop(
    llm_config,
    monkeypatch,
    unify_logs,
):
    """A persist loop stopped while it waits for the next interjection logs
    the stop and ends without another request."""
    requests = _count_llm_requests(monkeypatch)
    handle = start_async_tool_loop(
        new_llm_client(**llm_config),
        "Say 'ready' and nothing else.",
        {},
        persist=True,
    )
    # The loop reports its turn as a response when it starts to wait.
    notification = await asyncio.wait_for(handle.next_notification(), timeout=120)
    assert notification["type"] == "response"

    await handle.stop("done for now")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 1
    assert "Stop requested – reason: done for now" in unify_logs.text
