"""The request a turn sends must not depend on when a running tool finished.

A cached replay reproduces a run's tool results but not its timing: tools
finish milliseconds earlier or later relative to the loop's own steps than
they did when the run was recorded. Whatever the loop sends next must
therefore be the same whichever side of a dispatch, or of the building of a
turn, a tool happened to finish on, or the replay sends a request that was
never recorded.

Each test scripts the model's turns by replacing the client's ``generate``
(no LLM calls, no network), so the real ``generate_with_preprocess`` still
advances the sent watermark and undoes it for an unanswered dispatch.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from unify.common.async_tool_loop import start_async_tool_loop
from unify.common.llm_client import new_llm_client
from tests.helpers import _handle_project
from tests.async_helpers import _wait_for_condition, _wait_for_tool_request


def _tool_call(call_id: str, name: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _scripted_client(llm_config, monkeypatch):
    """A client whose turns come from a queue. Every dispatch records the
    tool_choice it was sent with and its serialized transcript."""
    client = new_llm_client(**llm_config)
    client.set_system_message("This conversation is fully scripted by the test.")
    turns: asyncio.Queue = asyncio.Queue()
    requests: list[dict] = []

    async def _generate(**gen_kwargs):
        requests.append(
            {
                "tool_choice": gen_kwargs.get("tool_choice"),
                "messages": [
                    json.dumps(m, sort_keys=True, default=str) for m in client.messages
                ],
            },
        )
        client.messages.append(await turns.get())
        return {"ok": True}

    monkeypatch.setattr(client, "generate", _generate)
    return client, turns, requests


def _gated_tool(result: str):
    gate = asyncio.Event()

    async def tool() -> str:
        """Finish once the test opens the gate."""
        await gate.wait()
        return result

    return gate, tool


async def _dispatches(requests: list[dict], n: int) -> None:
    async def _reached() -> bool:
        return len(requests) >= n

    await _wait_for_condition(_reached, poll=0.02, timeout=10.0)


async def _final_turn(llm_config, monkeypatch, *, superseded: bool) -> list[dict]:
    """Run two parallel tools and return every request sent. With
    *superseded*, the second tool finishes while the turn the first one
    earned is in flight; otherwise both finish together."""
    client, turns, requests = _scripted_client(llm_config, monkeypatch)
    quick_gate, quick = _gated_tool("quick-done")
    slow_gate, slow = _gated_tool("slow-done")

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"quick": quick, "slow": slow},
        timeout=30,
    )
    await turns.put(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("call_quick", "quick"),
                _tool_call("call_slow", "slow"),
            ],
        },
    )
    await _wait_for_tool_request(client, "slow")

    if superseded:
        quick_gate.set()
        await _dispatches(requests, 2)  # slow is still running
        slow_gate.set()
        await _dispatches(requests, 3)
    else:
        quick_gate.set()
        slow_gate.set()
        await _dispatches(requests, 2)

    await turns.put({"role": "assistant", "content": "done", "tool_calls": []})
    assert await asyncio.wait_for(handle.result(), timeout=30) == "done"
    return requests


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_superseded_turn_leaves_the_next_request_unchanged(
    llm_config,
    monkeypatch,
) -> None:
    """A tool that finishes while a turn is in flight supersedes that turn.
    The turn issued next must be the one that would have been sent had the
    tool finished before the superseded dispatch: its result written into
    its placeholder, never a check_status pair."""
    direct = await _final_turn(llm_config, monkeypatch, superseded=False)
    superseded = await _final_turn(llm_config, monkeypatch, superseded=True)

    assert len(direct) == 2 and len(superseded) == 3
    assert superseded[1]["tool_choice"] == "required"
    assert superseded[2] == direct[1]
    assert direct[1]["tool_choice"] == "auto"
    messages = direct[1]["messages"]
    assert not any("check_status_" in m for m in messages)
    assert any("slow-done" in m for m in messages)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_turn_built_after_a_tool_finished_does_not_force_required(
    llm_config,
    monkeypatch,
) -> None:
    """A tool that finished but has not been ingested when the next turn is
    built is ingested first: its result is in the request, and with nothing
    left running the turn does not force tool_choice="required"."""
    client, turns, requests = _scripted_client(llm_config, monkeypatch)
    gate, slow = _gated_tool("slow-done")

    handle = start_async_tool_loop(
        client=client,
        message="start",
        tools={"slow": slow},
        timeout=30,
    )
    await turns.put(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_slow", "slow")],
        },
    )
    await _wait_for_tool_request(client, "slow")

    # The tool finishes in the same tick as an interjection arrives. The
    # interjection is handled first, so the turn it earns is built while the
    # finished tool is still waiting to be ingested.
    gate.set()
    await handle.interject("continue")
    await _dispatches(requests, 2)

    await turns.put({"role": "assistant", "content": "done", "tool_calls": []})
    assert await asyncio.wait_for(handle.result(), timeout=30) == "done"

    assert requests[1]["tool_choice"] == "auto"
    assert any("slow-done" in m for m in requests[1]["messages"])
