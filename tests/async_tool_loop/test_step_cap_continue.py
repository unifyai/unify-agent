"""Symbolic: ``UNIFY_STEP_CAP_COMPACT=continue``, one long-horizon mode.

In the Crafter census of 8 October, 253 of 286 episodes hit the loop's step
limit: ``max_steps`` (a count of messages) runs over a persistent session's
whole conversation, so a session of many short requests reaches it, ends,
and its host restarts it cold. ``continue`` makes the actor's task loop (the
one that answers a requester; every other loop runs as shipped):

* count the step budget per request, from the requester's message, and
  afresh after each compaction;
* compact at the limit as ``on`` does, with no bound per request, and go
  on; the compaction's summary is loop-authored (the marker never reaches
  the provider), so it starts no request and ``UNIFY_LOOP_STOP``'s count
  carries across it;
* end a request whose second compaction in a row did not make the context
  smaller (serialised message characters);
* end every request that a stop ends (a failed compaction, an ineffective
  one, a loop stop, the loop's timeout) through ``UNIFY_STEP_CAP_REPLY``'s
  reply path, ``draft`` when that switch is empty, so the requester always
  gets an answer and a persistent session goes on to its next request.

Off and ``on`` send the bytes they sent before. The transport is scripted,
so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import io
import json

import pytest

import unify.common._async_tool.context_compression as _cc
from tests import cache_discipline_helpers as h
from unify.common._async_tool import loop_stop as _loop_stop
from unify.common._async_tool.cache_discipline import COMPRESSION_FORK_INSTRUCTION
from unify.common._async_tool.context_compression import _COMPRESSED_HEADER
from unify.common._async_tool.loop import continue_mode_active
from unify.settings import SETTINGS, ProductionSettings

TASK = "Find the answer and reply with it."
NEXT = "Next request: check it once more."
DRAFT = "Checking again; best so far: 42."
FINAL = "The answer is 42."
REPLY = "Replied from a cell: 42."
SUMMARY = "Summary: looked several times; best so far 42."
RESTART = "Context was compressed. Continue from where you left off."
RESTARTED = "<restarted>"  # the request as the model reads it after a compaction
TERMINATED = "🔚 Terminating early: max_steps ({}) exceeded"
STOPPED = (
    "🔚 Stopped at the step limit: max_steps ({}) exceeded, so this request "
    "ended before it was finished. The session is still open: the next "
    "message starts a new request."
)
BEST = "\n\nBest current answer:\n"
WAIT = 5  # seconds any one wait of a session may take


def _is_compactor(messages: list) -> bool:
    return any(
        m.get("role") == "system"
        and "You are a context compactor" in str(m.get("content"))
        for m in messages
    )


def _is_fork(messages: list) -> bool:
    return bool(messages) and messages[-1].get("content") == (
        COMPRESSION_FORK_INSTRUCTION
    )


def _requester(messages: list) -> tuple[str, int]:
    """The request the model is answering and its tool-calling turns so far.

    The loop's own notices reach the transport without their marker, so they
    are told by their text; a compaction's summary reads as *RESTARTED*.
    """
    turns = 0
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("role") == "user":
            if RESTART in content:
                return RESTARTED, turns
            if not content.startswith("["):
                return content, turns
        if message.get("role") == "assistant" and message.get("tool_calls"):
            turns += 1
    return "", turns


class _Model:
    """The session's model, its compression fork and its compactor.

    *plan(request, turns)* gives the calls of a session turn (with *DRAFT*
    as its text), or ``None`` to reply ``FINAL``; *fork(n)* answers the n-th
    compression fork (a completion, or a coroutine giving one)."""

    def __init__(self, plan, *, fork=None):
        self.plan = plan
        self.fork = fork or (lambda n: h.completion(content=SUMMARY))
        self.requests: list[dict] = []
        self.forks: list[list] = []
        self.compactor_calls = 0

    @property
    def session_requests(self) -> list[dict]:
        return [
            r
            for r in self.requests
            if not _is_compactor(r["messages"]) and not _is_fork(r["messages"])
        ]

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(
            copy.deepcopy(
                {
                    k: kw.get(k)
                    for k in ("messages", "tools", "tool_choice", "reasoning_effort")
                },
            ),
        )
        if _is_compactor(messages):
            self.compactor_calls += 1
            return h.completion(content="Compacted.")
        if _is_fork(messages):
            self.forks.append(messages)
            result = self.fork(len(self.forks))
            if asyncio.iscoroutine(result):
                result = await result
            return result
        if not kw.get("tools"):
            return h.completion(content=FINAL)
        request, turns = _requester(messages)
        calls = self.plan(request, turns)
        if calls is None:
            return h.completion(content=FINAL)
        return h.completion(content=DRAFT, calls=calls)


async def look() -> str:
    """Look again."""
    return "Nothing new."


async def slow() -> str:
    """Wait for the slow service."""
    await asyncio.sleep(30)
    return "Done."


async def execute_code(thought: str = "", code: str = "") -> str:
    """Run Python code.

    Args:
        thought: Why the code is run.
        code: The code to run.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # the scripted model's own constant cells
    return out.getvalue()


LOOK = [("look", {})]


def _install(model) -> None:
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model


@pytest.fixture
def mode(monkeypatch):
    def set_(compact: str, *, cap_reply: str = "", **others) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_COMPACT", compact)
        monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", cap_reply)
        monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "")
        for name, value in others.items():
            monkeypatch.setattr(SETTINGS, name, value)

    return set_


def _start(tools=None, *, persist: bool = True, max_steps=7, **kwargs):
    """The actor's task loop: it answers a requester, and no loop started it.
    The handle's ``result()`` is awaited in the background, as ``unify act
    --persist`` does, since it restarts a compacted loop."""
    from unify.common.async_tool_loop import start_async_tool_loop

    kwargs.setdefault("reply_channel", True)
    kwargs.setdefault("timeout", 30)
    handle = start_async_tool_loop(
        h.new_client(),
        TASK,
        tools or {"look": look},
        log_steps=False,
        persist=persist,
        max_steps=max_steps,
        **kwargs,
    )
    handle._test_result = asyncio.ensure_future(handle.result())
    return handle


async def _response(handle) -> str:
    return (await asyncio.wait_for(h._next_response(handle), WAIT))["content"]


async def _ended(handle) -> str:
    return await asyncio.wait_for(handle._test_result, WAIT)


async def _close(handle) -> None:
    await handle.stop()
    await _ended(handle)


async def _session(model, *requests: str, **kwargs):
    """A persistent session: the task, then *requests*; the responses and
    the session's runtime state."""
    with h.scripted(()):
        _install(model)
        handle = _start(**kwargs)
        responses = [await _response(handle)]
        for request in requests:
            await handle.submit(request)
            responses.append(await _response(handle))
        state = handle._runtime_state
        history = list(handle.get_history())
        await _close(handle)
    return responses, state, history


def _assert_every_call_answered(messages: list) -> None:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["id"] in answered, call


def _digest(requests: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([h.request_bytes(r) for r in requests]).encode(),
    ).hexdigest()


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_takes_continue():
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT=" Continue ").UNIFY_STEP_CAP_COMPACT
        == "continue"
    )
    assert (
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="on").UNIFY_STEP_CAP_COMPACT == "on"
    )
    assert ProductionSettings(UNIFY_STEP_CAP_COMPACT="off").UNIFY_STEP_CAP_COMPACT == ""
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_STEP_CAP_COMPACT="forever")


def test_the_mode_hook(mode):
    from unify.common._async_tool.loop import ToolLoopRuntimeState

    state = ToolLoopRuntimeState()
    mode("on")
    assert not continue_mode_active()
    mode("continue")
    assert continue_mode_active()
    # A loop's state says whether that loop runs in the mode.
    assert not continue_mode_active(state)
    state.step_cap_continue = True
    assert continue_mode_active(state)
    mode("")
    assert not continue_mode_active(state)


# ── off and on are as shipped ────────────────────────────────────────────


def _answer_task_then_loop(request: str, turns: int):
    """The task is answered after one look; the next request loops on it."""
    if request == TASK:
        return LOOK if turns < 1 else None
    return LOOK


# The requests the scripted session below sent before this mode existed
# (6170a5194, the switch off): the task answered, then a request that loops
# to the limit. Under ``on`` the fork's summary request differs from run to
# run (the transcript pointer), so ``on`` is pinned by its behaviour only.
SHIPPED_OFF_DIGEST = "9c3724e0afcdd5d517af3018f9fd1373a23c8136e3dfdafab464cfea23d83864"  # pragma: allowlist secret


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", ["", "on"])
async def test_off_and_on_send_the_bytes_they_sent_before(mode, compact):
    mode(compact)
    model = _Model(_answer_task_then_loop)
    with h.scripted(()):
        _install(model)
        handle = _start(max_steps=9)
        answered = await _response(handle)
        await handle.submit(NEXT)
        ended = await _ended(handle)
        state = handle._runtime_state
    assert (answered, ended) == (FINAL, TERMINATED.format(9))
    assert not state.step_cap_continue
    assert len(model.forks) == (2 if compact == "on" else 0)
    if compact == "":
        assert _digest(model.requests) == SHIPPED_OFF_DIGEST


# ── the per-request budget ───────────────────────────────────────────────


def _one_look(request: str, turns: int):
    return LOOK if turns < 1 else None


@pytest.mark.asyncio
async def test_continue_counts_the_budget_per_request(mode):
    """Six requests of two model calls (four messages) each: under a budget
    of six messages none reaches it, nothing is compacted."""
    mode("continue")
    model = _Model(_one_look)
    responses, state, _ = await _session(model, *[NEXT] * 5, max_steps=6)
    assert responses == [FINAL] * 6
    assert (len(model.forks), model.compactor_calls) == (0, 0)
    assert state.step_cap_compactions == 0 and state.step_cap_continue


@pytest.mark.asyncio
async def test_off_the_same_session_reaches_the_limit_on_its_second_request(mode):
    mode("")
    model = _Model(_one_look)
    with h.scripted(()):
        _install(model)
        handle = _start(max_steps=6)
        first = await _response(handle)
        await handle.submit(NEXT)
        ended = await _ended(handle)
    assert (first, ended) == (FINAL, TERMINATED.format(6))


# ── compactions ──────────────────────────────────────────────────────────


def _answer_after_forks(model_ref: list, forks: int):
    """Look on the task, and after a compaction until *forks* compactions
    have run; then answer. A later request is answered at once."""

    def plan(request: str, turns: int):
        if request == TASK:
            return LOOK
        if request == RESTARTED:
            return LOOK if len(model_ref[0].forks) < forks else None
        return None

    return plan


def _model_answering_after(forks: int, **kwargs) -> _Model:
    ref: list = []
    model = _Model(_answer_after_forks(ref, forks), **kwargs)
    ref.append(model)
    return model


@pytest.mark.asyncio
async def test_continue_compacts_a_request_any_number_of_times(mode):
    mode("continue")
    model = _model_answering_after(3)
    (answered, next_answer), state, history = await _session(model, NEXT)
    assert (answered, next_answer) == (FINAL, FINAL)
    assert (len(model.forks), model.compactor_calls) == (3, 0)
    assert state.step_cap_compactions == 3
    assert state.step_cap_ineffective_compactions == 0
    # Each compaction gave the request a fresh budget: the session requests
    # after each one read the summary with no stop notice.
    assert "Terminating early" not in json.dumps(model.requests)
    _assert_every_call_answered(history)


@pytest.mark.asyncio
async def test_on_the_same_request_stops_at_its_third_limit(mode):
    mode("on")
    model = _model_answering_after(3)
    with h.scripted(()):
        _install(model)
        handle = _start()
        ended = await _ended(handle)
    assert ended == TERMINATED.format(7)
    assert len(model.forks) == 2


@pytest.mark.asyncio
async def test_continue_marks_the_summary_loop_authored_and_sends_it_unmarked(mode):
    mode("continue")
    model = _model_answering_after(1)
    _, _, history = await _session(model)
    restart = [m for m in history if RESTART in str(m.get("content"))]
    assert len(restart) == 1 and restart[0].get("_loop_authored") is True
    assert restart[0]["content"].startswith(_COMPRESSED_HEADER + SUMMARY)
    after = next(
        r for r in model.session_requests if _requester(r["messages"])[0] == RESTARTED
    )
    sent = [m for m in after["messages"] if RESTART in str(m.get("content"))]
    assert sent == [{"role": "user", "content": restart[0]["content"]}]


@pytest.mark.asyncio
async def test_on_leaves_the_summary_unmarked(mode):
    mode("on")
    model = _model_answering_after(1)
    _, _, history = await _session(model)
    restart = [m for m in history if RESTART in str(m.get("content"))]
    assert len(restart) == 1 and "_loop_authored" not in restart[0]


@pytest.mark.asyncio
async def test_a_one_shot_request_is_compacted_and_answered(mode):
    mode("continue")
    model = _model_answering_after(2)
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await _ended(handle)
    assert result == FINAL
    assert len(model.forks) == 2


@pytest.mark.asyncio
async def test_calls_past_the_limit_are_answered_before_the_compaction(mode):
    """A turn whose calls run past the limit: the ones after it are answered
    as not run before the compaction, and the restarted request has no
    call left unanswered."""
    mode("continue")

    async def look_at(place: str) -> str:
        """Look at one place.

        Args:
            place: Where to look.
        """
        return f"Nothing at {place}."

    def plan(request: str, turns: int):
        if request == TASK:
            # Distinct calls: identical ones in one turn are pruned.
            return LOOK if turns < 1 else [("look_at", {"place": p}) for p in "abcd"]
        return None

    model = _Model(plan)
    (answered,), _, history = await _session(
        model,
        tools={"look": look, "look_at": look_at},
    )
    assert answered == FINAL
    assert len(model.forks) == 1
    # The fork is the last request sent plus its instruction (a cache hit),
    # so it ends before that turn; the conversation it summarises has every
    # call of the turn answered, the last as not run.
    _assert_every_call_answered(model.forks[0][:-1])
    restarted_at = next(
        i for i, m in enumerate(history) if RESTART in str(m.get("content"))
    )
    assert restarted_at > 0
    assert [
        m["content"]
        for m in model.session_requests[-1]["messages"]
        if m.get("role") == "tool"
    ] == []
    _assert_every_call_answered(history)


@pytest.mark.asyncio
async def test_a_reply_from_a_cell_at_the_limit_ends_the_turn(mode):
    """A reply a cell gave at the limit is the turn's reply: nothing is
    compacted for it, and the next request starts with its own budget."""
    from unify.common._async_tool import cell_reply

    mode("continue", UNIFY_REPLY_CHANNEL="code+text")

    async def answer() -> str:
        """Reply with the answer."""
        assert cell_reply.deliver(REPLY, False) is None
        return "Replied."

    def plan(request: str, turns: int):
        if request == TASK:
            return LOOK if turns < 2 else [("answer", {})]
        return None

    model = _Model(plan)
    (replied, answered), _, _ = await _session(
        model,
        NEXT,
        tools={"look": look, "answer": answer},
    )
    assert (replied, answered) == (REPLY, FINAL)
    assert len(model.forks) == 0


@pytest.mark.asyncio
async def test_a_message_queued_at_the_limit_is_not_folded_into_the_summary(mode):
    """A requester message sent while the compaction runs stays queued: the
    summary is made without it, and it is read once, after the summary."""
    mode("continue")
    started, release = asyncio.Event(), asyncio.Event()

    async def fork(n: int):
        started.set()
        await release.wait()
        return h.completion(content=SUMMARY)

    def plan(request: str, turns: int):
        return LOOK if request in (TASK, RESTARTED) else None

    model = _Model(plan, fork=fork)
    with h.scripted(()):
        _install(model)
        handle = _start()
        await asyncio.wait_for(started.wait(), WAIT)
        await handle.submit(NEXT)
        release.set()
        answered = await _response(handle)
        await _close(handle)
    assert answered == FINAL
    assert len(model.forks) == 1 and NEXT not in json.dumps(model.forks[0])
    last = model.session_requests[-1]["messages"]
    contents = [str(m.get("content")) for m in last]
    summary_at = next(i for i, c in enumerate(contents) if RESTART in c)
    assert contents.count(NEXT) == 1 and contents.index(NEXT) > summary_at


# ── ineffective compactions ──────────────────────────────────────────────


def _looks(request: str, turns: int):
    return LOOK if request in (TASK, RESTARTED) else None


def _summary_of(chars: int):
    return lambda n: h.completion(content="x" * chars)


@pytest.mark.asyncio
async def test_two_ineffective_compactions_in_a_row_end_the_request(mode):
    """Each summary is larger than the context it replaces: the first is
    used, the second ends the request with its draft, and the session
    takes its next request."""
    mode("continue")
    model = _Model(_looks, fork=lambda n: h.completion(content="x" * 40_000 * n))
    (capped, answered), state, history = await _session(model, NEXT)
    assert capped == STOPPED.format(7) + BEST + DRAFT
    assert answered == FINAL
    assert len(model.forks) == 2
    assert state.step_cap_compactions == 1
    assert state.step_cap_ineffective_compactions == 2
    assert state.step_cap_ineffective_in_a_row == 0  # the next request's
    _assert_every_call_answered(history)


@pytest.mark.asyncio
async def test_an_effective_compaction_between_resets_the_run(mode):
    mode("continue")
    sizes = {1: 40_000, 2: 10, 3: 40_000}
    model = _model_answering_after(
        3,
        fork=lambda n: h.completion(content="x" * sizes[n]),
    )
    (answered,), state, _ = await _session(model)
    assert answered == FINAL
    assert len(model.forks) == state.step_cap_compactions == 3
    assert state.step_cap_ineffective_compactions == 2


# ── failures and stops end through the reply path ───────────────────────


def _fail_fork(n: int):
    raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_a_failed_fork_falls_back_to_the_compactor(mode):
    mode("continue")
    model = _model_answering_after(0, fork=_fail_fork)
    (answered,), state, _ = await _session(model)
    assert answered == FINAL
    assert len(model.forks) == 1 and model.compactor_calls >= 1
    assert state.step_cap_compactions == 1


@pytest.fixture
def compactor_fails(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(_cc, "compress_messages", fail)


@pytest.mark.asyncio
async def test_a_failed_compaction_ends_the_request_with_its_draft(
    mode,
    compactor_fails,
):
    mode("continue")
    model = _Model(_looks, fork=_fail_fork)
    (capped, answered), state, _ = await _session(model, NEXT)
    assert capped == STOPPED.format(7) + BEST + DRAFT
    assert answered == FINAL
    assert (state.step_cap_compactions, state.step_cap_compaction_failures) == (0, 1)


@pytest.mark.asyncio
async def test_a_failed_compaction_ends_a_one_shot_request_with_its_draft(
    mode,
    compactor_fails,
):
    mode("continue")
    model = _Model(_looks, fork=_fail_fork)
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await _ended(handle)
    assert result == TERMINATED.format(7) + BEST + DRAFT


@pytest.mark.asyncio
async def test_a_compaction_that_runs_past_the_timeout_ends_the_request(mode):
    mode("continue")
    release = asyncio.Event()

    async def fork(n: int):
        await release.wait()
        return h.completion(content=SUMMARY)

    model = _Model(_looks, fork=fork)
    try:
        (capped, answered), state, _ = await _session(model, NEXT, timeout=1)
    finally:
        release.set()
    assert capped == STOPPED.format(7) + BEST + DRAFT
    assert answered == FINAL
    assert state.step_cap_compaction_failures == 1


@pytest.mark.asyncio
async def test_stop_during_a_compaction_ends_it_at_once(mode):
    mode("continue")
    started, release = asyncio.Event(), asyncio.Event()

    async def fork(n: int):
        started.set()
        await release.wait()
        return h.completion(content=SUMMARY)

    model = _Model(_looks, fork=fork)
    with h.scripted(()):
        _install(model)
        handle = _start()
        await asyncio.wait_for(started.wait(), WAIT)
        await handle.stop()
        await _ended(handle)
        assert handle.done() and not release.is_set()
        release.set()
    assert len(model.forks) == 1
    assert handle._runtime_state.step_cap_compactions == 0


def _slow_then_answer(request: str, turns: int):
    return [("slow", {})] if request == TASK and turns < 1 else None


@pytest.mark.asyncio
async def test_the_timeout_ends_the_request_with_its_draft(mode):
    mode("continue")
    model = _Model(_slow_then_answer)
    (stopped, answered), _, history = await _session(
        model,
        NEXT,
        tools={"slow": slow},
        timeout=1,
    )
    assert stopped == (
        "🔚 Stopped: timeout (1s) exceeded, so this request ended before it "
        "was finished. The session is still open: the next message starts a "
        "new request." + BEST + DRAFT
    )
    assert answered == FINAL
    assert any(
        m.get("role") == "tool"
        and "Cancelled: the timeout (1s) was reached before this call finished."
        == m.get("content")
        for m in history
    )
    _assert_every_call_answered(history)


@pytest.mark.asyncio
async def test_off_the_timeout_ends_the_loop_as_shipped(mode):
    mode("")
    model = _Model(_slow_then_answer)
    with h.scripted(()):
        _install(model)
        handle = _start({"slow": slow}, timeout=1)
        ended = await _ended(handle)
    assert ended == "🔚 Terminating early: timeout (1s) exceeded"


# ── the loop stop across a compaction ───────────────────────────────────


@pytest.mark.asyncio
async def test_the_loop_stop_counts_across_a_compaction(mode):
    """K=4 no-op cells: two before the compaction (at five messages), two
    after it. The count carries over, so the stop fires at the fourth, with
    the draft, instead of a second compaction."""
    mode("continue", UNIFY_LOOP_STOP="on", UNIFY_LOOP_STOP_K=4)

    def plan(request: str, turns: int):
        if request in (TASK, RESTARTED):
            return [("execute_code", {"thought": "Run.", "code": "print('')"})]
        return None

    model = _Model(plan)
    (stopped, answered), state, _ = await _session(
        model,
        NEXT,
        tools={"execute_code": execute_code},
        max_steps=5,
    )
    stop = _loop_stop.Stop(k=4, last_word=False)
    assert stopped == stop.headline + BEST + DRAFT
    assert answered == FINAL
    assert len(model.forks) == state.step_cap_compactions == 1
    assert state.loop_stops == 1
    # Two no-op turns before the compaction, two after it.
    assert [
        _requester(r["messages"])
        for r in model.session_requests
        if _requester(r["messages"])[0] in (TASK, RESTARTED)
    ] == [(TASK, 0), (TASK, 1), (RESTARTED, 0), (RESTARTED, 1)]


# ── scope ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_loop_that_answers_no_requester_runs_as_shipped(mode):
    mode("continue")
    model = _Model(_looks)
    with h.scripted(()):
        _install(model)
        handle = _start(reply_channel=False)
        ended = await _ended(handle)
        state = handle._runtime_state
    assert ended == TERMINATED.format(7)
    assert len(model.forks) == 0
    assert not state.step_cap_continue and not continue_mode_active(state)
