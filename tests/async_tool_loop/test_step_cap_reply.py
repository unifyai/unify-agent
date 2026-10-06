"""Symbolic: ``UNIFY_STEP_CAP_REPLY`` ends a request at ``max_steps``, not the loop.

``max_steps`` counts every message of a loop, so a persistent loop that has
handled many requests reaches it and ends: its result is the stop notice and
no later message is answered (the Continual-ARC hang of 5 October). With the
switch, the limit ends only the request: the reply says so and quotes the
latest text drafted for the request, a pending call is cancelled and answered
as such, and the next request is counted from its own message. The limit
itself is unchanged. A loop that is not persistent still ends at the limit,
its notice followed by the draft. The transport is scripted, so nothing
leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

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


@pytest.fixture
def cap_reply(monkeypatch):
    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", on)

    return set_


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
async def test_off_a_persistent_loop_ends_at_the_limit_as_shipped(cap_reply):
    cap_reply(False)
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        result = await asyncio.wait_for(handle.result(), 20)
        assert result == f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded"
        assert handle.done()
        # Nothing is left to answer a later message.
        await handle.interject(CONTINUE)
        await asyncio.sleep(0.5)
    assert not any(_last_request(r) == CONTINUE for r in model.requests)


@pytest.mark.asyncio
async def test_on_the_limit_ends_the_request_and_the_next_one_is_answered(cap_reply):
    cap_reply(True)
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        capped = (await h._next_response(handle))["content"]
        first_request_calls = len(model.requests)
        await handle.interject(CONTINUE)
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
async def test_on_every_request_has_its_own_limit(cap_reply):
    """A request that runs away again stops again, after as many steps."""
    cap_reply(True)
    model = _Model(answer_to="never")
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True)
        first = (await h._next_response(handle))["content"]
        calls_first = len(model.requests)
        await handle.interject(CONTINUE)
        second = (await h._next_response(handle))["content"]
        calls_second = len(model.requests) - calls_first
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)

    assert first.startswith("🔚 Stopped at the step limit")
    assert second.startswith("🔚 Stopped at the step limit")
    assert 1 <= calls_second <= calls_first + 1


@pytest.mark.asyncio
async def test_on_without_a_draft_the_reply_says_so(cap_reply):
    cap_reply(True)
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
async def test_on_a_call_in_flight_is_cancelled_answered_and_not_run_again(cap_reply):
    cap_reply(True)
    started: list[int] = []
    release = asyncio.Event()

    async def slow() -> str:
        """Take a long look."""
        started.append(1)
        await release.wait()
        return "done"

    class _SlowModel(_Model):
        async def __call__(self, *, shared_session=None, client=None, **kw):
            messages = kw.get("messages") or []
            self.requests.append(messages)
            if _last_request(messages) == CONTINUE:
                return h.completion(content=FINAL)
            # Two calls at once, which reach the limit while still pending.
            return h.completion(content=DRAFT, calls=[("look", {}), ("slow", {})])

    model = _SlowModel()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look, "slow": slow}, persist=True, max_steps=5)
        capped = (await h._next_response(handle))["content"]
        await handle.interject(CONTINUE)
        answered = (await h._next_response(handle))["content"]
        await handle.stop()
        await asyncio.wait_for(handle.result(), 20)

    assert capped.startswith("🔚 Stopped at the step limit")
    assert answered == FINAL
    # Cancelled at the limit, and not scheduled again for the next request.
    assert len(started) <= 1
    last = model.requests[-1]
    _assert_every_call_answered(last)
    cancelled = [
        m
        for m in last
        if m.get("role") == "tool"
        and str(m.get("content")).startswith("Cancelled: the step limit")
    ]
    assert len(cancelled) == 2


@pytest.mark.asyncio
async def test_on_a_loop_that_is_not_persistent_ends_with_the_draft(cap_reply):
    cap_reply(True)
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


@pytest.mark.asyncio
async def test_off_a_loop_that_is_not_persistent_ends_as_shipped(cap_reply):
    cap_reply(False)
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=False)
        result = await asyncio.wait_for(handle.result(), 20)
    assert result == f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded"
