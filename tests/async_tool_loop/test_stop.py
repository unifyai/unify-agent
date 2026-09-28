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
from unify.common.async_tool_loop import start_async_tool_loop
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
    started = asyncio.Event()

    async def hold() -> str:
        """Run until stopped."""
        started.set()
        await asyncio.Event().wait()
        return "released"

    requests = _count_llm_requests(monkeypatch)
    client = new_llm_client(**llm_config)
    # A seeded call starts `hold` without a model turn.
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
    await asyncio.wait_for(started.wait(), timeout=30)

    await handle.stop("done")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0
    last_turn = next(m for m in reversed(client.messages) if m["role"] == "assistant")
    (steer,) = last_turn["tool_calls"]
    assert steer["function"]["name"] == "steer"
    assert json.loads(steer["function"]["arguments"]) == {
        "call_id": "call_hold",
        "action": "stop",
        "payload": "done",
    }
