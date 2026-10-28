"""Symbolic: ``max_steps`` ends a request, not the loop.

``max_steps`` counts every message of a loop, so a persistent loop that has
handled many requests would reach it and end, and no later message would be
answered. Instead the limit ends only the request: the reply says so and
quotes the latest text drafted for the request, a pending call is cancelled
and answered as such, and the next request is counted from its own message.
A loop that is not persistent still ends at the limit, its notice followed by
the draft. The transport is scripted, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h

TASK = "Find the answer and reply with it."
DRAFT = "Checking again; best so far: 42."
CONTINUE = "Please continue and give your answer."
FINAL = "The answer is 42."
MAX_STEPS = 6


def _last_request(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not message.get("_loop_authored"):
            return str(message.get("content") or "")
    return ""


class _Model:
    """Calls ``look`` (with *draft* as its text) until the request is
    *answer_to*, which it answers with ``FINAL``."""

    def __init__(self, *, draft: str | None = DRAFT, answer_to: str = CONTINUE):
        self.draft = draft
        self.answer_to = answer_to
        self.requests: list[list[dict]] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _last_request(messages) == self.answer_to:
            return h.completion(content=FINAL)
        return h.completion(content=self.draft, calls=[("look", {})])


def _start(tools: dict, *, persist: bool, max_steps: int = MAX_STEPS):
    from unify.common.async_tool_loop import start_async_tool_loop

    return start_async_tool_loop(
        h.new_client(),
        TASK,
        tools,
        log_steps=False,
        timeout=30,
        persist=persist,
        max_steps=max_steps,
    )


async def look() -> str:
    """Look again."""
    return "Nothing new."


def _assert_every_call_answered(messages: list) -> None:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["id"] in answered, call


@pytest.mark.asyncio
async def test_the_limit_ends_the_request_and_the_next_one_is_answered():
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        capped = (await h._next_response(handle))["content"]
        first_request_calls = len(model.requests)
        await handle.submit(CONTINUE)
        answered = (await h._next_response(handle))["content"]
        assert not handle.done()
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)

    assert capped == (
        f"🔚 Stopped at the step limit: max_steps ({MAX_STEPS}) exceeded, so "
        "this request ended before it was finished. The session is still "
        "open: the next message starts a new request.\n\n"
        f"Best current answer:\n{DRAFT}"
    )
    assert answered == FINAL
    # The second request was answered by one call, which saw the reply.
    assert len(model.requests) == first_request_calls + 1
    last = model.requests[-1]
    assert _last_request(last) == CONTINUE
    assert any(
        m.get("role") == "assistant" and m.get("content") == capped for m in last
    )
    _assert_every_call_answered(last)


@pytest.mark.asyncio
async def test_every_request_has_its_own_limit():
    """A request that runs away again stops again, after as many steps."""
    model = _Model(answer_to="never")
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        first = (await h._next_response(handle))["content"]
        calls_first = len(model.requests)
        await handle.submit(CONTINUE)
        second = (await h._next_response(handle))["content"]
        calls_second = len(model.requests) - calls_first
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)

    assert first.startswith("🔚 Stopped at the step limit")
    assert second.startswith("🔚 Stopped at the step limit")
    assert 1 <= calls_second <= calls_first + 1


@pytest.mark.asyncio
async def test_without_a_draft_the_reply_says_so():
    model = _Model(draft=None)
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        capped = (await h._next_response(handle))["content"]
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)
    assert capped.endswith("\n\nNo reply text was drafted for this request.")


@pytest.mark.asyncio
async def test_a_call_after_the_limit_is_answered_and_not_run_again():
    """A turn's calls run in order; once the limit is reached the calls
    after it are answered as not run, and none is run for the next request."""
    started: list[int] = []

    async def slow() -> str:
        """Take a long look."""
        started.append(1)
        await asyncio.Event().wait()
        return "done"

    class _SlowModel(_Model):
        async def __call__(self, *, shared_session=None, client=None, **kw):
            messages = kw.get("messages") or []
            self.requests.append(messages)
            if _last_request(messages) == CONTINUE:
                return h.completion(content=FINAL)
            # The first call's result reaches the limit (the loop's header,
            # the request, this turn's message, that result).
            return h.completion(content=DRAFT, calls=[("look", {}), ("slow", {})])

    model = _SlowModel()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look, "slow": slow}, persist=True, max_steps=4)
        capped = (await h._next_response(handle))["content"]
        await handle.submit(CONTINUE)
        answered = (await h._next_response(handle))["content"]
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)

    assert capped.startswith("🔚 Stopped at the step limit")
    assert answered == FINAL
    assert started == []
    last = model.requests[-1]
    _assert_every_call_answered(last)
    not_run = [
        m
        for m in last
        if m.get("role") == "tool"
        and str(m.get("content")).startswith("Not run: the step limit")
    ]
    assert [m["name"] for m in not_run] == ["slow"]
    assert any(
        m.get("role") == "tool" and m.get("content") == "Nothing new." for m in last
    )


@pytest.mark.asyncio
async def test_a_loop_that_is_not_persistent_ends_with_the_draft():
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=False)
        result = await asyncio.wait_for(handle.result(), 20)
    assert result == (
        f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded\n\n"
        f"Best current answer:\n{DRAFT}"
    )
