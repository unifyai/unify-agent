"""
Stopping an async tool loop.

stop() signals cancellation: the loop cancels the model call or tool call in
flight and ends without sending the LLM another request.
"""

from __future__ import annotations

import asyncio

import pytest

from unify.common._async_tool import loop as _loop
from unify.common.async_tool_loop import AsyncToolLoopHandle, start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.baked_defaults import as_shipped  # noqa: F401

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
    cancelled = asyncio.Event()

    async def hold() -> str:
        """Run until stopped."""
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
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
    handle.hold_cancelled = cancelled  # type: ignore[attr-defined]
    return handle, started


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
    build = _loop.method_to_schema
    stopped: list = []

    def stop_mid_build(*args, **kwargs):
        if not stopped:
            # stop() signals before it first awaits, so it runs to its end here.
            stopping = handle.stop("stopped mid-build")
            stopped.append(stopping)
            try:
                stopping.send(None)
            except StopIteration:
                pass
        return build(*args, **kwargs)

    monkeypatch.setattr(_loop, "method_to_schema", stop_mid_build)
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
    """Stopping mid-tool cancels the running call, then ends."""
    requests = _count_llm_requests(monkeypatch)
    client = new_llm_client(**llm_config)
    handle, started = _start_holding(client)
    await asyncio.wait_for(started.wait(), timeout=30)

    await handle.stop("done")

    assert await handle.result() == _STOPPED
    assert requests["n"] == 0
    assert handle.hold_cancelled.is_set()


# as_shipped: deleted in step 5 (steer(stop) of steerable handles)
@pytest.mark.usefixtures("as_shipped")
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
