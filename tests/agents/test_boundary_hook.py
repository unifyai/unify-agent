"""Symbolic: the loop's on_turn_boundary callback runs only just before a model call,
after every tool result, and what it returns is appended once, at the end."""

import asyncio

import pytest

from tests import cache_discipline_helpers as h


async def _loop(replies, tools, on_turn_boundary):
    from unify.common.async_tool_loop import start_async_tool_loop

    with h.scripted(replies) as provider:
        handle = start_async_tool_loop(
            h.new_client(),
            "Do the task.",
            tools,
            log_steps=False,
            timeout=60,
            max_steps=30,
            interrupt_llm_with_interjections=False,
            steering_tools=False,
            on_turn_boundary=on_turn_boundary,
        )
        result = await asyncio.wait_for(handle.result(), timeout=60)
    return result, provider.requests


def _feed():
    pending: list[str] = []
    calls = {"n": 0}

    async def on_turn_boundary():
        calls["n"] += 1
        if not pending:
            return None
        text = "[record]\n" + "\n".join(pending)
        pending.clear()
        return text

    return pending, calls, on_turn_boundary


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_what_arrives_during_a_call_and_a_cell_is_appended_after_the_tool_result():
    pending, calls, hook = _feed()

    async def execute_code(code: str) -> str:
        """Run Python code.

        Args:
            code: The code to run.
        """
        pending.append("posted during the cell")
        await asyncio.sleep(0.05)
        return f"ran {code}"

    def first():
        pending.append("posted during the model call")
        return h.completion(calls=[("execute_code", {"code": "x"})])

    result, requests = await _loop(
        [first, lambda: h.completion(content="done")],
        {"execute_code": execute_code},
        hook,
    )
    assert (
        result == "done" and len(requests) == 2
    )  # no extra, discarded or cancelled call
    assert "posted during" not in str(requests[0])
    msgs = requests[1]["messages"]
    assert msgs[-2]["role"] == "tool" and msgs[-1]["role"] == "user"
    assert msgs[-1]["content"].endswith(
        "[record]\nposted during the model call\nposted during the cell",
    )
    # Append-only: the second request starts with the first one's messages, unchanged.
    first = requests[0]["messages"]
    assert requests[1]["messages"][: len(first)] == first
    assert calls["n"] == 2  # once before each model call, never after the turn ended


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_nothing_new_appends_nothing():
    _, calls, hook = _feed()
    tools = h.make_tools({})
    result, requests = await _loop(
        [
            lambda: h.completion(calls=[("execute_code", {"code": "1+1"})]),
            lambda: h.completion(content="done"),
        ],
        {"execute_code": tools["execute_code"]},
        hook,
    )
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert calls["n"] == 2


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_entry_posted_during_the_last_call_is_not_appended_to_any_request():
    pending, calls, hook = _feed()

    def only():
        pending.append("too late")
        return h.completion(content="done")

    result, requests = await _loop([only], {}, hook)
    assert result == "done" and len(requests) == 1 and calls["n"] == 1
    assert pending == ["too late"]
