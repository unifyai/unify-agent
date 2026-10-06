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


@pytest.mark.asyncio
@pytest.mark.timeout(300)
@pytest.mark.parametrize("seed", range(12))
async def test_randomised_schedules_deliver_each_post_once_at_the_next_boundary(seed):
    """Posts land at random points (during a model call or a tool call); each one
    appears exactly once, at the first boundary after it, and every scripted
    reply is consumed by exactly one model call (none discarded or cancelled)."""
    import random

    rnd = random.Random(seed)
    pending, calls, hook = _feed()
    posted_at: dict[str, int] = {}  # text -> index of the request that must carry it
    steps = rnd.randint(1, 4)
    step = {"i": 0}

    async def execute_code(code: str) -> str:
        """Run Python code.

        Args:
            code: The code to run.
        """
        for k in range(rnd.randint(0, 2)):
            text = f"cell {code} post {k}"
            pending.append(text)
            posted_at[text] = int(code) + 1
            await asyncio.sleep(rnd.random() / 50)
        return f"ran {code}"

    def reply_for(i):
        def reply():
            for k in range(rnd.randint(0, 2)):
                text = f"call {i} post {k}"
                pending.append(text)
                posted_at[text] = i + 1 if i < steps else None
            if i < steps:
                return h.completion(calls=[("execute_code", {"code": str(i)})])
            return h.completion(content="done")

        return reply

    result, requests = await _loop(
        [reply_for(i) for i in range(steps + 1)],
        {"execute_code": execute_code},
        hook,
    )
    assert result == "done" and len(requests) == steps + 1
    for text, at in posted_at.items():
        carriers = [j for j, r in enumerate(requests) if text in str(r["messages"])]
        if at is None:  # posted during the last call: that turn ends there
            assert carriers == []
        else:
            assert carriers and carriers[0] == at
            last = requests[at]["messages"][-1]
            assert last["role"] == "user" and text in last["content"]
            assert requests[at]["messages"][-2]["role"] == "tool"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_record_block_comes_after_the_budget_footer(monkeypatch):
    from unify.common.async_tool_loop import start_async_tool_loop
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_BUDGET_FOOTER", True)
    pending, _, hook = _feed()
    seen_footer = {"at": None}
    requests = []

    async def look() -> str:
        """Look again."""
        pending.append(f"post {len(requests)}")
        return "Nothing new."

    async def model(*, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        requests.append(messages)
        tools = [m for m in messages if m.get("role") == "tool"]
        if tools and "[step budget]" in str(tools[-1].get("content")):
            seen_footer["at"] = len(requests) - 1
            return h.completion(content="done")
        return h.completion(calls=[("look", {})])

    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = start_async_tool_loop(
            h.new_client(),
            "Do the task.",
            {"look": look},
            log_steps=False,
            timeout=60,
            # The cap counts messages and a record block is one, so a round is
            # three steps; the footer's window (the last tenth, 4 of 40) is wider.
            max_steps=40,
            interrupt_llm_with_interjections=False,
            steering_tools=False,
            on_turn_boundary=hook,
        )
        await asyncio.wait_for(handle.result(), 60)
    at = seen_footer["at"]
    assert at is not None, "the footer never appeared"
    msgs = requests[at]
    assert msgs[-2]["role"] == "tool" and "[step budget]" in str(msgs[-2]["content"])
    assert msgs[-1]["role"] == "user" and msgs[-1]["content"].startswith("[record]")
