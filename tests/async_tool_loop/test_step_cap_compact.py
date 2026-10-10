"""Symbolic: ``UNIFY_STEP_CAP_COMPACT`` compacts the conversation at ``max_steps``.

As shipped, ``max_steps`` (a count of every message of the loop, or of the
request under ``UNIFY_STEP_CAP_REPLY``) ends the loop with a stop notice, or
ends the request with a reply that says so. A persistent ScienceWorld
session that reached it was restarted by its host without its conversation
(3.5% of the overhauled HIGH episodes, 34% of a research build's LOW).

With the switch on, a loop that can compress its context compacts it at the
limit with that same compression, and the same request goes on in the same
loop, its steps counted from the compacted conversation. A request may be
compacted twice; at its third limit the shipped behaviour applies. A reply
the request has already given is not held back for a compaction. When the
compaction fails, runs past the loop's timeout or is cancelled, the shipped
behaviour applies too, or the cancel does. The transport is scripted, so
nothing leaves the process.

Under the baked defaults (the cache discipline) the compression is
the fork summary: the session's last request plus one instruction, answered
by the session's own model. The compactor runs only when the fork yields no
summary.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import unify.common._async_tool.context_compression as _cc
from tests import cache_discipline_helpers as h
from unify.common._async_tool.cache_discipline import COMPRESSION_FORK_INSTRUCTION
from unify.common._async_tool.context_compression import _COMPRESSED_HEADER
from unify.settings import SETTINGS

TASK = "Find the answer and reply with it."
DRAFT = "Checking again; best so far: 42."
FINAL = "The answer is 42."
CONTINUE = "Please continue and give your answer."
FINAL_2 = "Still 42."
LOOP_AGAIN = "Look into it once more."
REPLY = "Replied from a cell: 42."
RESTART = "Context was compressed. Continue from where you left off."
SUMMARY = "Summary: looked several times; best so far 42."
TERMINATED = "🔚 Terminating early: max_steps ({}) exceeded"
STOPPED = "🔚 Stopped at the step limit: max_steps ({}) exceeded"
MAX_STEPS = 17
WAIT = 2  # seconds any one wait of a session may take


def _is_compactor(messages: list) -> bool:
    return any(
        m.get("role") == "system"
        and "You are a context compactor" in str(m.get("content"))
        for m in messages
    )


def _is_fork(messages: list) -> bool:
    """A compression fork: the session's request plus the summary instruction."""
    return bool(messages) and messages[-1].get("content") == (
        COMPRESSION_FORK_INSTRUCTION
    )


def _last_request(messages: list) -> str:
    """The latest requester message (the loop's ``[steerable ...]`` notices
    reach the model without their ``_loop_authored`` mark)."""
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if (
            message.get("role") == "user"
            and not message.get("_loop_authored")
            and not content.startswith("[")
        ):
            return content
    return ""


def _after_restart(messages: list) -> list | None:
    """The messages after the latest restart notice, if there is one."""
    starts = [
        i
        for i, m in enumerate(messages)
        if m.get("role") == "user" and RESTART in str(m.get("content"))
    ]
    return messages[starts[-1] + 1 :] if starts else None


def _text(messages: list) -> str:
    return json.dumps(messages, default=str)


class _Model:
    """The scripted model of the session, its compression fork and its compactor.

    The session calls ``look`` (with *DRAFT* as its text) until the request
    is ``CONTINUE`` (answered with ``FINAL_2``) or, with *after_restart*,
    until the conversation was compacted: ``"answer"`` then answers with
    ``FINAL`` at once, ``"look"`` first looks once more. A compression fork
    answers with *SUMMARY*, or as *fork* says: ``"fail"`` raises, ``"block"``
    answers its first *block_calls* calls only once :attr:`release` is set
    (:attr:`fork_started` is set when one starts). unillm shields a model
    call from a caller that stops waiting (the provider is already charging;
    the abandoned call is billed when it ends), so a blocked fork is
    abandoned by the loop, never cancelled. The compactor returns at once,
    leaving every entry as it is.
    """

    def __init__(
        self,
        *,
        after_restart: str | None = "answer",
        fork: str = "summary",
        block_calls: int = 1,
    ):
        self.after_restart = after_restart
        self.fork = fork
        self.block_calls = block_calls
        self.requests: list[list[dict]] = []
        self.forks: list[list[dict]] = []
        self.compactor_calls = 0
        self.fork_started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def fork_calls(self) -> int:
        return len(self.forks)

    @property
    def session_requests(self) -> list[list[dict]]:
        return [r for r in self.requests if not _is_compactor(r) and not _is_fork(r)]

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        if _is_compactor(messages):
            self.compactor_calls += 1
            self.requests.append(messages)
            return h.completion(content="Compacted.")
        self.requests.append(messages)
        if _is_fork(messages):
            self.forks.append(messages)
            if self.fork == "fail":
                raise RuntimeError("provider unavailable")
            if self.fork == "block" and len(self.forks) <= self.block_calls:
                self.fork_started.set()
                await self.release.wait()
            return h.completion(content=SUMMARY)
        last = _last_request(messages)
        if last == CONTINUE:
            return h.completion(content=FINAL_2)
        after = _after_restart(messages) if last == TASK or RESTART in last else None
        if after is not None and self.after_restart == "answer":
            return h.completion(content=FINAL)
        if after is not None and self.after_restart == "look":
            if any(m.get("role") == "tool" for m in after):
                return h.completion(content=FINAL)
        return h.completion(content=DRAFT, calls=[("look", {})])


async def look() -> str:
    """Look again."""
    return "Nothing new."


@pytest.fixture
def switches(monkeypatch):
    def set_(*, compact: str = "on", cap_reply: bool = False, **others) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_COMPACT", compact)
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", cap_reply)
        for name, value in others.items():
            monkeypatch.setattr(SETTINGS, name, value)

    return set_


def _start(
    tools=None,
    *,
    persist: bool = True,
    timeout: float = 30,
    drive: bool = False,
    **kwargs,
):
    """Start the loop; with *drive*, await its result in the background, as
    ``unify act --persist`` does for the whole session: the handle restarts
    a compacted loop from its ``result()``."""
    from unify.common.async_tool_loop import start_async_tool_loop

    handle = start_async_tool_loop(
        h.new_client(),
        TASK,
        tools or {"look": look},
        log_steps=False,
        timeout=timeout,
        persist=persist,
        max_steps=MAX_STEPS,
        **kwargs,
    )
    if drive:
        handle._test_result = asyncio.ensure_future(handle.result())
    return handle


async def _response(handle) -> dict:
    return await asyncio.wait_for(h._next_response(handle), WAIT)


async def _close(handle) -> None:
    await handle.stop()
    await asyncio.wait_for(handle._test_result, WAIT)


def _install(model) -> None:
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model


def _assert_every_call_answered(messages: list) -> None:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["id"] in answered, call


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_STEP_CAP_COMPACT == ""
    assert ProductionSettings(UNIFY_STEP_CAP_COMPACT="off").UNIFY_STEP_CAP_COMPACT == ""
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT=" On ").UNIFY_STEP_CAP_COMPACT == "on"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="yes")


@pytest.mark.asyncio
async def test_off_the_limit_ends_the_loop_as_shipped(switches):
    switches(compact="")
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start()
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == TERMINATED.format(MAX_STEPS)
    assert (model.fork_calls, model.compactor_calls) == (0, 0)
    assert handle._runtime_state.step_cap_compactions == 0


# ── whole-conversation count (UNIFY_STEP_CAP_REPLY off) ─────────────────


@pytest.mark.asyncio
async def test_on_the_session_is_compacted_and_goes_on(switches):
    """The capped request goes on after the compaction and is answered.
    Later requests are answered in the same session, until one is answered
    at the limit: the answer is given, and the request after it is read
    after a compaction that ran while it waited in the queue."""
    switches()
    model = _Model()
    answers = []
    with h.scripted(()):
        _install(model)
        handle = _start(drive=True)
        answers.append((await _response(handle))["content"])
        for _ in range(10):
            await handle.submit(CONTINUE)
            answers.append((await _response(handle))["content"])
            if model.fork_calls == 2:
                break
        assert not handle.done()
        state = handle._runtime_state
        await _close(handle)

    assert answers[0] == FINAL
    assert set(answers[1:]) == {FINAL_2}
    assert model.fork_calls == state.step_cap_compactions == 2
    assert model.compactor_calls == 0
    # The step count restarted from the compacted conversation.
    assert state.message_count_offset == 0
    # The first answer came after the compaction, from its summary of the
    # conversation, which held the task.
    answered = next(r for r in model.session_requests if RESTART in _last_request(r))
    assert _COMPRESSED_HEADER.strip() in _text(answered)
    assert SUMMARY in _text(answered)
    assert TASK in _text(model.forks[0])
    assert len(answered) < MAX_STEPS
    # The second compaction summarised a conversation at the limit.
    assert len(model.forks[1]) >= MAX_STEPS
    # The last request was compacted before it was read, and read after it.
    last = model.session_requests[-1]
    assert _last_request(last) == CONTINUE
    assert any(RESTART in str(m.get("content")) for m in last)
    assert _COMPRESSED_HEADER.strip() in _text(last)
    assert "Terminating early" not in _text(model.requests)
    assert "Stopped at the step limit" not in _text(model.requests)
    _assert_every_call_answered(last)


@pytest.mark.asyncio
async def test_on_a_request_is_compacted_twice_then_ends_as_shipped(switches):
    switches()
    model = _Model(after_restart=None)
    with h.scripted(()):
        _install(model)
        handle = _start()
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == TERMINATED.format(MAX_STEPS)
    assert (model.fork_calls, model.compactor_calls) == (2, 0)
    assert handle._runtime_state.step_cap_compactions == 2


@pytest.mark.asyncio
async def test_on_a_loop_that_is_not_persistent_is_compacted_and_answers(switches):
    switches()
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == FINAL
    assert (model.fork_calls, model.compactor_calls) == (1, 0)


@pytest.mark.asyncio
async def test_on_without_compression_the_limit_ends_the_loop_as_shipped(switches):
    switches()
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(enable_compression=False)
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == TERMINATED.format(MAX_STEPS)
    assert (model.fork_calls, model.compactor_calls) == (0, 0)


@pytest.mark.asyncio
async def test_on_the_fork_summary_compacts_and_the_request_goes_on_from_it(
    switches,
):
    """The compaction is the fork summary a full context gets: the session's
    last request plus the instruction; the request goes on from it."""
    switches()
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == FINAL
    assert (model.fork_calls, model.compactor_calls) == (1, 0)
    # the fork continues the session's last request
    fork = model.forks[0]
    assert fork[:-1] == model.session_requests[-2]
    answered = model.session_requests[-1]
    assert RESTART in _last_request(answered)
    assert SUMMARY in _last_request(answered)
    assert handle._runtime_state.step_cap_compactions == 1


# ── per-request count (UNIFY_STEP_CAP_REPLY on) ─────────────────────────


@pytest.mark.asyncio
async def test_on_with_cap_reply_a_long_request_is_compacted_and_answered(switches):
    switches(cap_reply=True)
    model = _Model()
    with h.scripted(()):
        _install(model)
        handle = _start(drive=True)
        first = (await _response(handle))["content"]
        await handle.submit(CONTINUE)
        second = (await _response(handle))["content"]
        await _close(handle)
    assert (first, second) == (FINAL, FINAL_2)
    assert (model.fork_calls, model.compactor_calls) == (1, 0)


@pytest.mark.asyncio
async def test_on_with_cap_reply_the_third_limit_replies_and_the_next_request_has_its_own(
    switches,
):
    switches(cap_reply=True)
    model = _Model(after_restart=None)
    with h.scripted(()):
        _install(model)
        handle = _start(drive=True)
        capped = (await _response(handle))["content"]
        after_first = model.fork_calls
        await handle.submit(CONTINUE)
        answered = (await _response(handle))["content"]
        await handle.submit(LOOP_AGAIN)
        capped_again = (await _response(handle))["content"]
        await _close(handle)
    assert capped.startswith(STOPPED.format(MAX_STEPS))
    assert capped.endswith("Best current answer:\n" + DRAFT)
    assert after_first == 2
    assert answered == FINAL_2
    assert capped_again.startswith(STOPPED.format(MAX_STEPS))
    assert (model.fork_calls, model.compactor_calls) == (4, 0)


# ── failure, timeout and cancel ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_a_failed_compaction_ends_the_loop_as_shipped(switches, monkeypatch):
    switches()

    async def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    # The fork fails, and so does the compactor it falls back to.
    monkeypatch.setattr(_cc, "compress_messages", fail)
    model = _Model(fork="fail")
    with h.scripted(()):
        _install(model)
        handle = _start()
        result = await asyncio.wait_for(handle.result(), WAIT)
    assert result == TERMINATED.format(MAX_STEPS)
    assert model.fork_calls == 1
    state = handle._runtime_state
    assert (state.step_cap_compactions, state.step_cap_compaction_failures) == (0, 1)


@pytest.mark.asyncio
async def test_on_with_cap_reply_a_failed_compaction_replies_and_the_session_goes_on(
    switches,
    monkeypatch,
):
    switches(cap_reply=True)

    async def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(_cc, "compress_messages", fail)
    model = _Model(fork="fail")
    with h.scripted(()):
        _install(model)
        handle = _start(drive=True)
        capped = (await _response(handle))["content"]
        await handle.submit(CONTINUE)
        answered = (await _response(handle))["content"]
        await _close(handle)
    assert capped.startswith(STOPPED.format(MAX_STEPS))
    assert answered == FINAL_2


@pytest.mark.asyncio
async def test_on_stop_during_a_compaction_ends_it_at_once(switches):
    switches()
    model = _Model(fork="block")
    with h.scripted(()):
        _install(model)
        handle = _start()
        await asyncio.wait_for(model.fork_started.wait(), WAIT)
        await handle.stop()
        await asyncio.wait_for(handle.result(), WAIT)
        # The loop ended while the fork's call was still out.
        assert handle.done() and not model.release.is_set()
        model.release.set()
    assert model.fork_calls == 1
    assert handle._runtime_state.step_cap_compactions == 0


@pytest.mark.asyncio
async def test_on_a_cancelled_request_during_a_compaction_ends_the_request(
    switches,
    monkeypatch,
):
    switches()
    model = _Model(fork="block")
    with h.scripted(()):
        _install(model)
        handle = _start(drive=True)
        await asyncio.wait_for(model.fork_started.wait(), WAIT)
        assert await handle.cancel_request("enough")
        cancelled = await _response(handle)
        await handle.submit(CONTINUE)
        answered = (await _response(handle))["content"]
        state = handle._runtime_state
        await _close(handle)
        model.release.set()
    assert cancelled.get("cancelled") is True
    # The abandoned fork's summary was never used; the next request's was.
    assert model.fork_calls == 2
    # The next request was compacted before it was read, and answered.
    assert answered == FINAL_2
    assert state.step_cap_compactions == 1


@pytest.mark.asyncio
async def test_on_a_compaction_is_bounded_by_the_loop_timeout(switches):
    switches()
    model = _Model(fork="block")
    with h.scripted(()):
        _install(model)
        handle = _start(timeout=0.5)
        result = await asyncio.wait_for(handle.result(), WAIT)
        model.release.set()
    assert result == TERMINATED.format(MAX_STEPS)
    assert model.fork_calls == 1
    assert handle._runtime_state.step_cap_compaction_failures == 1


# ── what survives the compaction ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_a_reply_from_a_cell_at_the_limit_ends_the_turn(switches):
    """A reply a cell gave at the limit is the turn's reply: nothing is
    compacted for it. The next request is compacted before it is read."""
    from unify.common._async_tool import cell_reply

    switches(UNIFY_REPLY_CHANNEL="code+text")

    async def answer() -> str:
        """Reply with the answer."""
        assert cell_reply.deliver(REPLY, False) is None
        return "Replied."

    class _Replier(_Model):
        async def __call__(self, **kw):
            messages = kw.get("messages") or []
            if (
                not _is_compactor(messages)
                and not _is_fork(messages)
                and _last_request(messages) == TASK
            ):
                self.requests.append(messages)
                # Six looks, then the reply lands at the limit (the 17th message).
                if sum(m.get("role") == "tool" for m in messages) < 6:
                    return h.completion(content=DRAFT, calls=[("look", {})])
                return h.completion(content=DRAFT, calls=[("answer", {})])
            return await super().__call__(**kw)

    model = _Replier()
    with h.scripted(()):
        _install(model)
        handle = _start(
            {"look": look, "answer": answer},
            reply_channel=True,
            drive=True,
        )
        replied = (await _response(handle))["content"]
        compacted_before = model.fork_calls
        at_reply = len(handle.get_history())
        await handle.submit(CONTINUE)
        answered = (await _response(handle))["content"]
        await _close(handle)
    assert replied == REPLY
    assert at_reply >= MAX_STEPS
    assert compacted_before == 0
    assert answered == FINAL_2
    assert (model.fork_calls, model.compactor_calls) == (1, 0)


@pytest.mark.asyncio
async def test_on_the_turn_boundary_hook_goes_on_after_the_compaction(switches):
    """A record block delivered before the compaction is in the compacted
    context; the hook is called again after it, and a new block reaches
    the model."""
    switches()
    model = _Model(after_restart="look")
    calls = {"after": 0}
    holder: list = []

    async def on_turn_boundary():
        state = holder[0]._runtime_state if holder else None
        if state is not None and state.step_cap_compactions:
            calls["after"] += 1
            return (
                "[record] RECORD-2: a helper reported." if calls["after"] == 1 else None
            )
        if not calls.get("first"):
            calls["first"] = True
            return "[record] RECORD-1: the run started."
        return None

    with h.scripted(()):
        _install(model)
        handle = _start(on_turn_boundary=on_turn_boundary, drive=True)
        holder.append(handle)
        answer = (await _response(handle))["content"]
        await _close(handle)
    assert answer == FINAL
    assert calls["after"] >= 1
    after = [r for r in model.session_requests if RESTART in _last_request(r)]
    # The block delivered before the compaction was in what the fork summarised.
    assert model.fork_calls == 1 and "RECORD-1" in _text(model.forks[0])
    assert any(
        m.get("role") == "user" and "RECORD-2" in str(m.get("content"))
        for m in after[-1]
    )


@pytest.mark.asyncio
async def test_on_the_bound_request_survives_the_compaction(switches):
    from unify.common._async_tool import bound_request

    switches(UNIFY_BIND_REQUEST="on")
    seen: list = []

    async def look_and_read() -> str:
        """Look again."""
        slot = bound_request.current()
        seen.append(slot.text if slot is not None else None)
        return "Nothing new."

    model = _Model(after_restart="look")
    with h.scripted(()):
        _install(model)
        handle = _start({"look": look_and_read}, bind_request=True, drive=True)
        answer = (await _response(handle))["content"]
        await _close(handle)
    assert answer == FINAL
    assert (model.fork_calls, model.compactor_calls) == (1, 0)
    # Every call, the one after the compaction too, read the request.
    assert len(seen) >= 3 and set(seen) == {TASK}
