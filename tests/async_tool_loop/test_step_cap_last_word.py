"""Symbolic: ``UNIFY_STEP_CAP_REPLY=last_word`` gives the model one tool-less turn at the limit.

With ``draft`` a request stopped at ``max_steps`` quotes the latest text the
model drafted, which is often a progress note ("Checking again...") rather
than an answer. With ``last_word`` the pending calls are answered as
cancelled, a loop-authored notice says the step limit is reached, and the
model is called once with no tools offered; the stop reply quotes that
answer. If the call fails, is stopped or gives no text, the draft is quoted
instead, so the request never ends silent. The switch was a bool, so the
bool spellings keep meaning ``draft``. The transport is scripted, so nothing
leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS, ProductionSettings

TASK = "Find the answer and reply with it."
DRAFT = "Checking again; best so far: 42."
LAST_WORD = "My best answer: 42."
CONTINUE = "Please continue and give your answer."
FINAL = "The answer is 42."
MAX_STEPS = 6
BOUND = 5  # seconds; a scripted run takes well under one
STOPPED = (
    f"🔚 Stopped at the step limit: max_steps ({MAX_STEPS}) exceeded, so this "
    "request ended before it was finished. The session is still open: the "
    "next message starts a new request."
)
NOTICE_START = "The step limit for this request is reached"


def _last_request(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and NOTICE_START not in str(
            message.get("content"),
        ):
            if not message.get("_loop_authored"):
                return str(message.get("content") or "")
    return ""


class _Model:
    """Calls ``look`` until the limit; answers a tool-less request with
    *last_word* (a reply, or an exception to raise, or a callable)."""

    def __init__(self, last_word=LAST_WORD):
        self.last_word = last_word
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(
            {"messages": messages, "tools": kw.get("tools"), "kw": kw},
        )
        if not kw.get("tools"):
            reply = self.last_word
            if callable(reply):
                reply = await reply()
            if isinstance(reply, BaseException):
                raise reply
            return reply if not isinstance(reply, str) else h.completion(reply)
        if _last_request(messages) == CONTINUE:
            return h.completion(content=FINAL)
        return h.completion(content=DRAFT, calls=[("look", {})])

    @property
    def toolless(self) -> list[dict]:
        return [r for r in self.requests if not r["tools"]]


async def look() -> str:
    """Look again."""
    return "Nothing new."


def _start(*, persist: bool):
    from unify.common.async_tool_loop import start_async_tool_loop

    return start_async_tool_loop(
        h.new_client(),
        TASK,
        {"look": look},
        log_steps=False,
        timeout=30,
        persist=persist,
        max_steps=MAX_STEPS,
    )


@pytest.fixture
def last_word(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "last_word")


def _install(model) -> None:
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model


def _assert_every_call_answered(messages: list) -> None:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["id"] in answered, call


@pytest.mark.asyncio
async def test_the_request_ends_with_the_models_last_word(last_word):
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(persist=True)
        capped = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        calls_at_cap = len(model.requests)
        await handle.interject(CONTINUE)
        answered = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        stats = handle._runtime_state
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)

    assert capped == f"{STOPPED}\n\nBest current answer:\n{LAST_WORD}"
    assert answered == FINAL
    # One tool-less call, the last before the reply: no tools, no tool_choice.
    assert len(model.toolless) == 1
    toolless = model.toolless[0]
    assert model.requests[calls_at_cap - 1] is toolless
    assert toolless["kw"].get("tool_choice") in (None, "auto")
    assert NOTICE_START in str(toolless["messages"][-1]["content"])
    _assert_every_call_answered(toolless["messages"])
    # The next request sees the notice and the answer, not a second reply.
    last = model.requests[-1]["messages"]
    assert any(m.get("content") == LAST_WORD for m in last)
    assert not any(str(m.get("content")).startswith("🔚") for m in last)
    _assert_every_call_answered(last)
    assert (stats.step_cap_last_word_turns, stats.step_cap_last_word_fallbacks) == (
        1,
        0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("provider 502"),
        h.completion(content=""),
        h.completion(content=None, calls=[("look", {})]),
    ],
    ids=["call-fails", "empty-reply", "tool-call-only"],
)
async def test_without_a_last_word_the_draft_is_quoted(last_word, reply):
    model = _Model(last_word=reply)
    looks_before: list[int] = []
    with h.scripted(()):
        _install(model)
        handle = _start(persist=True)
        capped = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        looks_before.append(len(model.requests))
        await handle.interject(CONTINUE)
        answered = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        stats = handle._runtime_state
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)

    assert capped == f"{STOPPED}\n\nBest current answer:\n{DRAFT}"
    assert answered == FINAL
    assert (stats.step_cap_last_word_turns, stats.step_cap_last_word_fallbacks) == (
        1,
        1,
    )
    # A tool call named by a tool-less reply is not run after the limit.
    last = model.requests[-1]["messages"]
    _assert_every_call_answered(last)
    assert len(model.requests) == looks_before[0] + 1


@pytest.mark.asyncio
async def test_a_stop_during_the_last_word_ends_the_loop(last_word):
    entered = asyncio.Event()

    async def never():
        entered.set()
        await asyncio.sleep(3600)

    model = _Model(last_word=never)
    with h.scripted(()):
        _install(model)
        handle = _start(persist=True)
        await asyncio.wait_for(entered.wait(), BOUND)
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)
    assert handle.done()


@pytest.mark.asyncio
async def test_a_loop_that_is_not_persistent_ends_with_the_last_word(last_word):
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await asyncio.wait_for(handle.result(), BOUND)
    assert result == (
        f"🔚 Terminating early: max_steps ({MAX_STEPS}) exceeded\n\n"
        f"Best current answer:\n{LAST_WORD}"
    )
    assert len(model.toolless) == 1
    _assert_every_call_answered(model.toolless[0]["messages"])


@pytest.mark.asyncio
async def test_draft_mode_makes_no_tool_less_call(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "draft")
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(persist=True)
        capped = (await asyncio.wait_for(h._next_response(handle), BOUND))["content"]
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)
    assert capped == f"{STOPPED}\n\nBest current answer:\n{DRAFT}"
    assert model.toolless == []


@pytest.mark.parametrize(
    ("raw", "mode"),
    [
        ("", ""),
        ("false", ""),
        ("0", ""),
        ("no", ""),
        ("true", "draft"),
        ("1", "draft"),
        ("yes", "draft"),
        ("draft", "draft"),
        (" Last_Word ", "last_word"),
    ],
)
def test_the_switch_keeps_its_bool_spellings(monkeypatch, raw, mode):
    monkeypatch.setenv("UNIFY_STEP_CAP_REPLY", raw)
    settings = ProductionSettings(_env_file=None)
    assert settings.UNIFY_STEP_CAP_REPLY == mode
    assert settings.step_cap_reply() == mode


def test_the_switch_refuses_an_unknown_mode(monkeypatch):
    monkeypatch.setenv("UNIFY_STEP_CAP_REPLY", "lastword")
    with pytest.raises(ValueError, match="UNIFY_STEP_CAP_REPLY"):
        ProductionSettings(_env_file=None)


def test_a_bool_set_in_code_reads_as_before(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", True)
    assert SETTINGS.step_cap_reply() == "draft"
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", False)
    assert SETTINGS.step_cap_reply() == ""
