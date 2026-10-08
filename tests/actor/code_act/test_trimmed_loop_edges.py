"""Symbolic: the edge cases of the trimmed actor loop (no steering).

The loop runs one model turn at a time and a turn's tool calls in call order,
each to completion; nothing the requester sends interrupts a model call or a
cell (it is appended at the next turn boundary), and only ``stop()`` cancels
what is in flight. Each case of the design note
(``DESIGN-trim-the-loop.md``, cases 1-12) has a test here:

1. several ``execute_code`` calls in one turn run in order;
2. a plain text reply is the final answer (and a persistent session's
   response);
3. malformed arguments and an unknown tool are answered and the loop goes on;
4. a cell that raises (a), hangs past the cell timeout (b) or kills its
   worker (c) leaves the session usable, with no worker left behind;
5. provider errors are retried (a), an empty reply is nudged (b) and a reply
   cut at the token limit is taken as it is (c);
6. the step cap and the loop's timeout end the request cleanly;
7. compression mid-loop keeps every tool result;
8. a cell's ``reply()`` ends the turn; a second one is refused;
9. ``stop()`` during a model call and during a hung cell;
10. record blocks posted during a cell arrive once, at the next boundary;
11. a persistent session keeps its state across requests and errors;
12. ``UNIFY_LOOP_STOP`` ends a request of no-progress cells.

The model is a scripted transport replacing unillm's provider call (or, for
case 5a, ``litellm.acompletion`` below unillm's retry policy): nothing leaves
the process. Cells run in the sandboxed Python worker (bubblewrap), the
default workspace.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from typing import Any, Optional

import pytest
from openai.types.chat import ChatCompletion

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(120)]

DONE = "All done."
_IDS = itertools.count()


# ── the scripted model ──────────────────────────────────────────────────────


def _reply(
    content: Optional[str] = None,
    calls: list = (),
    *,
    finish_reason: Optional[str] = None,
    prompt_tokens: int = 100,
) -> ChatCompletion:
    """One provider reply. *calls* are ``(name, args)`` with *args* a dict, or
    a raw string sent as the arguments as it is (malformed JSON)."""
    tool_calls = [
        {
            "id": f"call_{next(_IDS)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": args if isinstance(args, str) else json.dumps(args),
            },
        }
        for name, args in calls
    ]
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "openai/gpt-5.6-sol",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason
                    or ("tool_calls" if tool_calls else "stop"),
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls or None,
                    },
                },
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 5,
                "total_tokens": prompt_tokens + 5,
            },
        },
    )


def _cell(code: str, thought: str = "Running a step.") -> tuple[str, dict]:
    return ("execute_code", {"thought": thought, "code": code})


class _Model:
    """Plays *script* in order. An entry is a reply, or a callable taking the
    request's messages and returning a reply (or an awaitable of one)."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list[list[dict]] = []
        self.in_flight = asyncio.Event()

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = json.loads(json.dumps(kw.get("messages") or [], default=str))
        self.requests.append(messages)
        if not self.script:
            raise AssertionError(f"no scripted reply for request {len(self.requests)}")
        entry = self.script.pop(0)
        if callable(entry):
            entry = entry(messages)
            if asyncio.iscoroutine(entry):
                entry = await entry
        return entry


@contextlib.contextmanager
def _scripted(model: _Model, *, below_retry: bool = False):
    """Install *model* as the provider transport (``below_retry``: as
    ``litellm.acompletion``, under unillm's transient-retry policy)."""
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    old_cache = os.environ.get("UNILLM_CACHE")
    old_default = unillm_settings.UNILLM_CACHE
    os.environ["UNILLM_CACHE"] = "false"
    unillm_settings.UNILLM_CACHE = False
    if below_retry:
        original = uni_llm.litellm.acompletion
        uni_llm.litellm.acompletion = model
    else:
        original = uni_llm._acompletion_with_transient_retry
        uni_llm._acompletion_with_transient_retry = model
    try:
        yield model
    finally:
        if below_retry:
            uni_llm.litellm.acompletion = original
        else:
            uni_llm._acompletion_with_transient_retry = original
        unillm_settings.UNILLM_CACHE = old_default
        if old_cache is None:
            os.environ.pop("UNILLM_CACHE", None)
        else:
            os.environ["UNILLM_CACHE"] = old_cache


# ── worker processes ────────────────────────────────────────────────────────


@pytest.fixture
def worker_pids(monkeypatch) -> list[int]:
    """The pid of every sandboxed Python worker started during the test."""
    from unify.actor.execution import worker as worker_mod

    assert worker_mod.enabled(), "these tests run cells in the sandboxed worker"
    pids: list[int] = []
    original = worker_mod.PythonWorker._start

    async def _start(self):
        await original(self)
        pids.append(self.pid)

    monkeypatch.setattr(worker_mod.PythonWorker, "_start", _start)
    return pids


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with contextlib.suppress(OSError):
        with open(f"/proc/{pid}/stat") as f:
            if f.read().split(") ", 1)[1].startswith("Z"):
                return False  # a zombie: exited, waiting to be reaped
    return True


async def _assert_gone(pids: list[int], bound: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + bound
    while any(_alive(p) for p in pids):
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(
                f"worker processes still alive: {[p for p in pids if _alive(p)]}",
            )
        await asyncio.sleep(0.05)


def _loop_tasks() -> list[str]:
    """The loop's own tasks still alive (model calls, tool calls, the loop)."""
    names = ("ToolUseLoop", "LLMGenerate", "ToolCall_")
    return [
        t.get_name()
        for t in asyncio.all_tasks()
        if not t.done() and t.get_name().startswith(names)
    ]


# ── transcript helpers ──────────────────────────────────────────────────────


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _stdout_lines(content: Any) -> list[str]:
    """The lines of a tool result's text, however the result is rendered."""
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    else:
        text = str(content)
    return [ln.strip() for ln in text.replace("\\n", "\n").splitlines()]


def _results_of(messages: list[dict], assistant: dict) -> list[dict]:
    """The tool messages answering *assistant*'s calls, in transcript order."""
    ids = {c["id"] for c in assistant.get("tool_calls") or []}
    start = messages.index(assistant)
    out = []
    for m in messages[start + 1 :]:
        if m.get("role") != "tool":
            break
        if m.get("tool_call_id") in ids:
            out.append(m)
    return out


def _append_only(requests: list[list[dict]]) -> bool:
    """Each request's messages extend the previous request's, byte for byte."""
    for before, after in zip(requests, requests[1:]):
        prev = [json.dumps(m, sort_keys=True, default=str) for m in before]
        nxt = [json.dumps(m, sort_keys=True, default=str) for m in after]
        if nxt[: len(prev)] != prev:
            return False
    return True


def _every_call_answered(messages: list[dict]) -> bool:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    return all(
        call["id"] in answered for m in messages for call in (m.get("tool_calls") or [])
    )


def _turns_with_calls(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]


async def _act(actor, request: str, **kw):
    return await actor.act(request, can_store=False, **kw)


@contextlib.asynccontextmanager
async def _actor(**kw):
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor(**kw)
    try:
        yield actor
    finally:
        await actor.close()


async def _next_response(handle, bound: float = 30.0) -> dict:
    while True:
        event = await asyncio.wait_for(handle.next_notification(), bound)
        if isinstance(event, dict) and event.get("type") == "response":
            return event


# ── 1. several calls in one turn ───────────────────────────────────────────


async def test_1_calls_of_one_turn_run_in_call_order(worker_pids):
    model = _Model(
        [
            _reply(
                calls=[
                    _cell("x = 1\nprint(f'x={x}')"),
                    _cell("x += 1\nprint(f'x={x}')"),
                    _cell("x *= 10\nprint(f'x={x}')"),
                ],
            ),
            _reply(DONE),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Compute x.")
            result = await asyncio.wait_for(handle.result(), 60)
            messages = list(handle.get_history())
    assert result == DONE
    (turn,) = _turns_with_calls(messages)
    results = _results_of(messages, turn)
    assert [r["tool_call_id"] for r in results] == [c["id"] for c in turn["tool_calls"]]
    printed = [
        [ln for ln in _stdout_lines(r["content"]) if ln.startswith("x=")]
        for r in results
    ]
    assert printed == [["x=1"], ["x=2"], ["x=20"]]
    # The second request carries all three results, in call order.
    sent = [m for m in model.requests[1] if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in sent] == [c["id"] for c in turn["tool_calls"]]
    await _assert_gone(worker_pids)


async def test_1_the_actors_requests_extend_each_other_byte_for_byte():
    """The whole actor (the baked defaults' first scenario): every request of
    the session extends the one before it, and no loop notice reaches it."""
    result, _, requests = await h.scenario_actor()
    assert result
    session = [r["messages"] for r in h.session_requests(requests)]
    assert len(session) >= 2
    assert _append_only(session)
    for messages in session:
        for m in messages:
            text = str(m.get("content") or "")
            assert not text.startswith(("[steerable ", "[askable ", "[progress "))
            assert "User Visibility Context" not in text


# ── 2. a plain text reply ──────────────────────────────────────────────────


async def test_2_a_text_reply_is_the_final_answer():
    from unify.common._async_tool import cell_reply

    model = _Model([_reply("The answer is 4.")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "What is 2 + 2?")
            result = await asyncio.wait_for(handle.result(), 60)
            final = handle.get_history()[-1]
    assert result == "The answer is 4."
    assert final["role"] == "assistant" and not final.get("tool_calls")
    if cell_reply.enabled():
        assert final[cell_reply.SOURCE_KEY] == "text"
    assert len(model.requests) == 1


async def test_2_a_persistent_sessions_text_reply_is_its_response():
    model = _Model([_reply("First answer."), _reply("Second answer.")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "First request.", persist=True)
            first = await _next_response(handle)
            await handle.submit("Second request.")
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert (first["content"], second["content"]) == ("First answer.", "Second answer.")


# ── 3. malformed or unknown tool calls ─────────────────────────────────────


async def test_3_malformed_and_unknown_calls_are_answered_and_the_loop_goes_on(
    worker_pids,
):
    """unillm asks again once for malformed arguments; a reply that is still
    malformed reaches the loop, which answers each bad call and goes on."""

    def _bad():
        return _reply(
            calls=[
                ("execute_code", '{"thought": "cut off", "code": "print(1'),
                ("no_such_tool", {"x": 1}),
            ],
        )

    model = _Model(
        [
            _bad(),
            _bad(),
            _reply(calls=[_cell("print('recovered')")]),
            _reply(DONE),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Print something.")
            result = await asyncio.wait_for(handle.result(), 60)
            messages = list(handle.get_history())
    assert result == DONE
    # unillm's own retry (as shipped) sent the turn again with its nudge.
    assert "were not valid JSON" in _text(model.requests[1][-1].get("content"))
    turns = _turns_with_calls(messages)
    assert len(turns) == 2, json.dumps(messages, default=str)[-3000:]
    bad, good = turns
    malformed, unknown = _results_of(messages, bad)
    assert "were not valid JSON" in _text(malformed["content"])
    # UNIFY_CACHE_DISCIPLINE (on for the actor) refuses it by the session's
    # fixed tool list; without it the loop says the tool is not available.
    assert "no_such_tool" in _text(unknown["content"])
    assert any(
        phrase in _text(unknown["content"])
        for phrase in ("is not in this session's tool list", "is not available")
    )
    (ok,) = _results_of(messages, good)
    assert "recovered" in _text(ok["content"])
    await _assert_gone(worker_pids)


# ── 4. cells that fail ─────────────────────────────────────────────────────


async def test_4a_a_cell_that_raises_leaves_the_session_usable(worker_pids):
    model = _Model(
        [
            _reply(calls=[_cell("kept = 5")]),
            _reply(calls=[_cell("raise ValueError('boom')")]),
            _reply(calls=[_cell("print(f'kept={kept}')")]),
            _reply(DONE),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            assert await asyncio.wait_for(handle.result(), 60) == DONE
            messages = list(handle.get_history())
    _, raised, read = _turns_with_calls(messages)
    assert "ValueError: boom" in _text(_results_of(messages, raised)[0]["content"])
    assert "kept=5" in _text(_results_of(messages, read)[0]["content"])
    await _assert_gone(worker_pids)


async def test_4b_a_hung_cell_is_interrupted_at_the_cell_timeout(worker_pids):
    model = _Model(
        [
            _reply(calls=[_cell("while True:\n    pass")]),
            _reply(calls=[_cell("print('alive')")]),
            _reply(DONE),
        ],
    )
    async with _actor(timeout=2) as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            assert await asyncio.wait_for(handle.result(), 60) == DONE
            messages = list(handle.get_history())
    hung, after = _turns_with_calls(messages)
    hung_text = _text(_results_of(messages, hung)[0]["content"])
    assert "timed out after 2" in hung_text
    assert "worker was killed" in hung_text
    after_text = _text(_results_of(messages, after)[0]["content"])
    assert "alive" in after_text and "A fresh Python worker started" in after_text
    assert len(worker_pids) == 2  # the hung one, then a fresh one
    await _assert_gone(worker_pids)


async def test_4c_a_worker_that_dies_is_replaced_for_the_next_cell(worker_pids):
    model = _Model(
        [
            _reply(
                calls=[
                    _cell("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)"),
                ],
            ),
            _reply(calls=[_cell("print('alive')")]),
            _reply(DONE),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            assert await asyncio.wait_for(handle.result(), 60) == DONE
            messages = list(handle.get_history())
    died, after = _turns_with_calls(messages)
    assert "worker exited during the cell" in _text(
        _results_of(messages, died)[0]["content"],
    )
    assert "alive" in _text(_results_of(messages, after)[0]["content"])
    assert len(worker_pids) == 2
    await _assert_gone(worker_pids)


# ── 5. provider errors, empty and truncated replies ────────────────────────


async def test_5a_transient_provider_errors_are_retried(monkeypatch):
    import litellm
    from unillm import helpers as unillm_helpers

    monkeypatch.setattr(unillm_helpers, "_BACKOFF_BASE_SECONDS", 0.01)

    def _raise(exc):
        async def _call(_messages):
            raise exc

        return _call

    model = _Model(
        [
            _raise(
                litellm.RateLimitError(
                    "rate limited",
                    llm_provider="openrouter",
                    model="openai/gpt-5.6-sol",
                ),
            ),
            _raise(
                litellm.InternalServerError(
                    "upstream failed",
                    llm_provider="openrouter",
                    model="openai/gpt-5.6-sol",
                ),
            ),
            _reply("Recovered answer."),
        ],
    )
    async with _actor() as actor:
        with _scripted(model, below_retry=True):
            handle = await _act(actor, "Answer.")
            result = await asyncio.wait_for(handle.result(), 60)
    assert result == "Recovered answer."
    assert len(model.requests) == 3


async def test_5b_an_empty_reply_is_nudged_once():
    model = _Model([_reply(None), _reply("Now with text.")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Answer.")
            result = await asyncio.wait_for(handle.result(), 60)
    assert result == "Now with text."
    assert model.requests[1][-1]["content"] == "Produce your final answer as text."


async def test_5c_a_reply_cut_at_the_token_limit_is_taken_as_it_is():
    model = _Model([_reply("A partial answer that was cut", finish_reason="length")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Answer.")
            result = await asyncio.wait_for(handle.result(), 60)
    assert result == "A partial answer that was cut"


# ── 6. caps ────────────────────────────────────────────────────────────────


async def test_6_the_step_cap_ends_the_request_and_leaves_nothing_running(
    monkeypatch,
    worker_pids,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MAX_TOOL_LOOP_STEPS", 6)
    model = _Model([_reply(calls=[_cell(f"print({i})")]) for i in range(10)])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Loop forever.")
            result = await asyncio.wait_for(handle.result(), 60)
            assert _loop_tasks() == []
    assert result.startswith("🔚 Terminating early: max_steps (6) exceeded")
    await _assert_gone(worker_pids)


async def test_6_the_step_cap_quotes_the_draft_under_step_cap_reply(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MAX_TOOL_LOOP_STEPS", 6)
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "draft")
    model = _Model(
        [_reply("Draft: 41.", calls=[_cell(f"print({i})")]) for i in range(10)],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Loop forever.")
            result = await asyncio.wait_for(handle.result(), 60)
    assert "Best current answer:\nDraft: 41." in result


async def test_6_the_loop_timeout_cancels_and_answers_the_running_call():
    from unify.common.async_tool_loop import start_async_tool_loop

    cancelled = asyncio.Event()

    async def slow(seconds: int) -> str:
        """Sleep.

        Args:
            seconds: How long.
        """
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "slept"

    model = _Model(
        [_reply(calls=[("slow", {"seconds": 30}), ("slow", {"seconds": 1})])],
    )
    with _scripted(model):
        handle = start_async_tool_loop(
            h.new_client(),
            "Sleep.",
            {"slow": slow},
            log_steps=False,
            timeout=1,
        )
        result = await asyncio.wait_for(handle.result(), 30)
        messages = list(handle.get_history())
    assert result.startswith("🔚 Terminating early: timeout (1s) exceeded")
    assert cancelled.is_set()
    (turn,) = _turns_with_calls(messages)
    first, second = _results_of(messages, turn)
    assert first["content"].startswith("Cancelled: the timeout (1s) was reached")
    assert second["content"].startswith("Not run: the timeout (1s) was reached")
    assert _loop_tasks() == []


# ── 7. compression mid-loop ────────────────────────────────────────────────


@pytest.mark.parametrize("full_on_turn", [0, 1])
async def test_7_compression_keeps_every_tool_result(full_on_turn):
    """The context fills on the first or on the second call's turn; both
    results are in the request that compresses, and the session goes on from
    the summary."""
    from unify.common.async_tool_loop import start_async_tool_loop

    ran: list[str] = []

    async def execute_code(code: str) -> str:
        """Run code.

        Args:
            code: The code.
        """
        ran.append(code)
        return f"result of {code}"

    tokens = [100, 100]
    tokens[full_on_turn] = 900_000
    summary = "Summary: both steps ran; answer done."
    model = _Model(
        [
            _reply(calls=[("execute_code", {"code": "one"})], prompt_tokens=tokens[0]),
            *(
                [
                    _reply(
                        calls=[("execute_code", {"code": "two"})],
                        prompt_tokens=tokens[1],
                    ),
                ]
                if full_on_turn == 1
                else []
            ),
            _reply(calls=[("compress_context", {})]),
            _reply(summary),
            *(
                [_reply(calls=[("execute_code", {"code": "two"})])]
                if full_on_turn == 0
                else []
            ),
            _reply(DONE),
        ],
    )
    with _scripted(model):
        handle = start_async_tool_loop(
            h.new_client(),
            "Run one, then two.",
            {"execute_code": execute_code},
            log_steps=False,
            timeout=60,
        )
        result = await asyncio.wait_for(handle.result(), 60)
    assert result == DONE
    assert ran == ["one", "two"]
    # The request that was told to compress carries every result so far.
    compress_request = next(
        r
        for r in model.requests
        if any(
            "You must call `compress_context` now." in _text(m.get("content"))
            for m in r
        )
    )
    carried = [_text(m["content"]) for m in compress_request if m.get("role") == "tool"]
    expected = ["result of one"] + (["result of two"] if full_on_turn == 1 else [])
    assert [c for c in carried if c.startswith("result of")] == expected
    # The session went on from the summary.
    assert any(summary in _text(m.get("content")) for m in model.requests[-1])
    # Continuity: up to the compression every request extends the one
    # before it, the summary request extends the compressing one (a fork),
    # the requests after the restart extend each other from the summary, and
    # no request leaves a call unanswered.
    at = model.requests.index(compress_request)
    before, summarising, after = (
        model.requests[: at + 1],
        model.requests[at + 1],
        model.requests[at + 2 :],
    )
    assert _append_only(before)
    if SETTINGS.UNIFY_CACHE_DISCIPLINE:
        assert _append_only([compress_request, summarising])
    assert _append_only(after)
    assert any(summary in _text(m.get("content")) for m in after[0])
    assert all(_every_call_answered(r) for r in before + after)


# ── 8. reply() from a cell ─────────────────────────────────────────────────


@pytest.fixture
def reply_channel(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")


async def test_8_a_cells_reply_ends_the_turn_and_a_second_is_ignored(reply_channel):
    """Two reply() calls in one cell: the first ends the cell."""
    model = _Model(
        [_reply(calls=[_cell("reply('first')\nreply('second')\nprint('unreached')")])],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Answer from code.")
            result = await asyncio.wait_for(handle.result(), 60)
            messages = list(handle.get_history())
    assert result == "first"
    assert len(model.requests) == 1  # no model call after the reply
    (turn,) = _turns_with_calls(messages)
    assert "unreached" not in _text(_results_of(messages, turn)[0]["content"])


async def test_8_a_second_cell_reply_in_the_same_turn_is_refused(reply_channel):
    from unify.common._async_tool import cell_reply

    model = _Model(
        [_reply(calls=[_cell("reply('first')"), _cell("reply('second')")])],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Answer from code.")
            result = await asyncio.wait_for(handle.result(), 60)
            messages = list(handle.get_history())
    assert result == "first"
    (turn,) = _turns_with_calls(messages)
    first, second = _results_of(messages, turn)
    assert cell_reply.ALREADY_REPLIED in _text(second["content"])


async def test_8_a_reply_in_a_later_request_is_that_requests_answer(reply_channel):
    """The reply slot starts empty for each request: a persistent session's
    later request answers with its own cell's reply()."""
    model = _Model(
        [
            _reply(calls=[_cell("reply('first')")]),
            _reply(calls=[_cell("reply('second')")]),
            _reply("unused"),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Answer from code.", persist=True)
            first = await _next_response(handle)
            await handle.submit("Again.")
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert (first["content"], second["content"]) == ("first", "second")
    assert len(model.requests) == 2  # no model call after either reply


# ── 9. stop ────────────────────────────────────────────────────────────────


async def test_9_stop_during_a_model_call(worker_pids):
    entered = asyncio.Event()

    async def _hang(_messages):
        entered.set()
        await asyncio.Event().wait()

    model = _Model([_reply(calls=[_cell("print('ran')")]), _hang])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            await asyncio.wait_for(entered.wait(), 30)
            await handle.stop("budget spent")
            result = await asyncio.wait_for(handle.result(), 30)
            assert _loop_tasks() == []
    assert result == "processed stopped early, no result"
    await _assert_gone(worker_pids)


async def test_9_a_stopped_model_calls_cost_still_reaches_the_run_meter(monkeypatch):
    """The provider bills a call it received even when the loop stops
    waiting for it: unillm lets it finish and reports the charge, which the
    run's meter counts, and the loop notes the cancelled turn."""
    import unillm.clients.uni_llm as uni_llm
    from decimal import Decimal

    monkeypatch.setattr(
        uni_llm,
        "compute_cost_from_response",
        lambda *_a, **_k: 0.0125,
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def _slow(_messages):
        entered.set()
        await release.wait()
        return _reply("An answer nobody waited for.")

    model = _Model([_reply(calls=[_cell("print('ran')")]), _slow])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            await asyncio.wait_for(entered.wait(), 30)
            await handle.stop("budget spent")
            assert await asyncio.wait_for(handle.result(), 30) == (
                "processed stopped early, no result"
            )
            meter = handle.run_meter
            # unillm reports the cancelled call at once, with no charge
            # (an unknown cost, never a zero one).
            assert meter.known_cost_usd("planning") == Decimal("0.0125")
            assert meter.cost_usd("planning") is None
            before = meter.calls["planning"]
            release.set()
            deadline = asyncio.get_running_loop().time() + 10
            while meter.calls["planning"] < before + 1:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.02)
    # The provider's answer arrives later and its charge is counted.
    assert meter.known_cost_usd("planning") == Decimal("0.0250")
    assert meter.cost_usd("planning") is None  # still never understated
    assert handle._runtime_state.cancelled_turns_by_cause == {"stop": 1}


async def test_9_a_callers_own_timeout_propagates_and_ends_the_loop():
    """result() reports a loop that died on its own as the stopped notice;
    a caller's own timeout around it surfaces as TimeoutError instead, and
    the unshielded cancellation ends the loop with it."""
    from unify.common.async_tool_loop import start_async_tool_loop

    async def _hang(_messages):
        await asyncio.Event().wait()

    model = _Model([_hang])
    with _scripted(model):
        handle = start_async_tool_loop(
            h.new_client(),
            "start",
            {},
            log_steps=False,
            timeout=120,
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(handle.result(), timeout=1)
        assert handle.done()
        assert _loop_tasks() == []


async def test_9_stop_during_a_hung_cell_kills_its_worker(worker_pids):
    model = _Model([_reply(calls=[_cell("import time\ntime.sleep(60)")])])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Go.")
            deadline = asyncio.get_running_loop().time() + 30
            while not worker_pids:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.5)  # the cell is running
            await handle.stop("wall cap")
            result = await asyncio.wait_for(handle.result(), 30)
            assert _loop_tasks() == []
            await _assert_gone(worker_pids)
    assert result == "processed stopped early, no result"
    assert len(model.requests) == 1


# ── 10. record blocks during a cell ────────────────────────────────────────


async def test_10_record_posts_during_a_cell_arrive_once_at_the_next_boundary():
    from unify.agents.cli_bridge import attach_bridge

    def _first(_messages):
        return _reply(calls=[_cell("import time\ntime.sleep(1.5)\nprint('slept')")])

    async def _second(messages):
        return _reply(DONE)

    model = _Model([_first, _second, _reply("unused")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Wait a little.")
            assert (
                getattr(handle, "agents_pool", None) is not None
            ), "UNIFY_AGENTS=record is the default"
            bridge = attach_bridge(handle, lambda **_: None)
            # Wait until the cell runs (the first model call has answered).
            deadline = asyncio.get_running_loop().time() + 30
            while len(model.requests) < 1:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            await bridge.user_message("POST-ONE: use base 10")
            await bridge.user_message("POST-TWO: be brief")
            # Nothing reaches the model while the cell runs.
            assert len(model.requests) == 1
            result = await asyncio.wait_for(handle.result(), 60)
    assert result == DONE
    assert len(model.requests) == 2
    second = model.requests[1]
    texts = [_text(m.get("content")) for m in second]
    # Each post is in the request exactly once, after the cell's result.
    for post in ("POST-ONE", "POST-TWO"):
        assert sum(post in t for t in texts) == 1, texts
    tool_at = max(i for i, m in enumerate(second) if m.get("role") == "tool")
    block_at = next(i for i, t in enumerate(texts) if "POST-ONE" in t)
    assert block_at > tool_at
    assert "slept" in texts[tool_at]


async def test_10_a_running_calls_progress_reaches_the_handle_not_the_model():
    """A running call's progress notifications go to the handle
    (``next_notification``) as they come; none enters the transcript, and
    the model is not called until the call has finished."""
    from unify.common.async_tool_loop import start_async_tool_loop

    progressed, release = asyncio.Event(), asyncio.Event()

    async def work(*, _notification_up_q: asyncio.Queue | None = None) -> str:
        """Do some work, reporting progress."""
        await _notification_up_q.put({"message": "halfway"})
        progressed.set()
        await release.wait()
        await _notification_up_q.put({"message": "finishing"})
        return "worked"

    model = _Model([_reply(calls=[("work", {})]), _reply(DONE)])
    with _scripted(model):
        handle = start_async_tool_loop(
            h.new_client(),
            "Work.",
            {"work": work},
            log_steps=False,
            timeout=30,
        )
        first = await asyncio.wait_for(handle.next_notification(), 10)
        await asyncio.wait_for(progressed.wait(), 10)
        assert len(model.requests) == 1  # nothing woke the model mid-call
        release.set()
        result = await asyncio.wait_for(handle.result(), 30)
        second = await asyncio.wait_for(handle.next_notification(), 10)
    assert result == DONE
    assert (first["message"], second["message"]) == ("halfway", "finishing")
    assert first["type"] == "notification" and first["tool_name"] == "work"
    sent = json.dumps(model.requests, default=str)
    assert "halfway" not in sent and "finishing" not in sent


# ── 11. a persistent session ───────────────────────────────────────────────


async def test_11_a_persistent_session_keeps_state_across_requests_and_errors(
    worker_pids,
):
    model = _Model(
        [
            _reply(calls=[_cell("x = 41")]),
            _reply("Set x."),
            _reply(calls=[_cell("raise RuntimeError('request two fails')")]),
            _reply("That failed."),
            _reply(calls=[_cell("print(f'x+1={x + 1}')")]),
            _reply("Read x."),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Set x to 41.", persist=True)
            assert (await _next_response(handle))["content"] == "Set x."
            await handle.submit("Now fail.")
            assert (await _next_response(handle))["content"] == "That failed."
            await handle.submit("Read x.")
            assert (await _next_response(handle))["content"] == "Read x."
            messages = list(handle.get_history())
            await handle.stop("session ended")
            await asyncio.wait_for(handle.result(), 30)
    turns = _turns_with_calls(messages)
    assert "request two fails" in _text(_results_of(messages, turns[1])[0]["content"])
    assert "x+1=42" in _text(_results_of(messages, turns[2])[0]["content"])
    await _assert_gone(worker_pids)


async def test_11_a_message_submitted_during_a_cell_waits_for_the_boundary():
    """While a cell runs, submit() is queued; the model is not called until
    the cell's result is in, and the message follows that result."""
    model = _Model(
        [
            _reply(calls=[_cell("import time\ntime.sleep(1.5)\nprint('slept')")]),
            _reply("Answered both."),
            _reply("unused"),
        ],
    )
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "First.", persist=True)
            deadline = asyncio.get_running_loop().time() + 30
            while len(model.requests) < 1:
                assert asyncio.get_running_loop().time() < deadline
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            await handle.submit("ALSO: second message")
            await asyncio.sleep(0.3)
            assert len(model.requests) == 1  # nothing interrupted the cell
            response = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert response["content"] == "Answered both."
    second = model.requests[1]
    tool_at = max(i for i, m in enumerate(second) if m.get("role") == "tool")
    message_at = next(
        i
        for i, m in enumerate(second)
        if "ALSO: second message" in _text(m.get("content"))
    )
    assert message_at > tool_at
    assert "slept" in _text(second[tool_at]["content"])


async def test_11_after_a_request_that_failed_at_the_step_limit_the_next_starts_cleanly(
    monkeypatch,
):
    """A request that ends at its step limit (a failed request, not just a
    failed cell) leaves the session's state and the next request intact."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "draft")
    monkeypatch.setattr(SETTINGS, "UNIFY_MAX_TOOL_LOOP_STEPS", 9)
    model = _Model(
        [
            _reply(calls=[_cell("x = 41")]),
            _reply("Set x."),
            *[_reply(calls=[_cell(f"print({i})")]) for i in range(8)],
        ],
    )

    def _read_x(messages):
        return _reply(calls=[_cell("print(f'x+1={x + 1}')")])

    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Set x to 41.", persist=True)
            assert (await _next_response(handle))["content"] == "Set x."
            await handle.submit("Loop for ever.")
            failed = await _next_response(handle)
            # The rest of the script answers the third request.
            model.script[:] = [_read_x, _reply("Read x.")]
            await handle.submit("Read x.")
            third = await _next_response(handle)
            messages = list(handle.get_history())
            await handle.stop("session ended")
            await asyncio.wait_for(handle.result(), 30)
    assert failed["content"].startswith("🔚 Stopped at the step limit")
    assert third["content"] == "Read x."
    last_turn = _turns_with_calls(messages)[-1]
    assert "x+1=42" in _text(_results_of(messages, last_turn)[0]["content"])
    assert _every_call_answered(messages)


def _drive(coro) -> None:
    """Run a handle method that never suspends (submit, cancel_request) now,
    without yielding to the loop."""
    try:
        coro.send(None)
    except StopIteration:
        pass
    else:  # pragma: no cover - the method suspended
        raise AssertionError("expected the call to complete without awaiting")


def _user_texts(messages: list[dict]) -> list[str]:
    return [
        _text(m.get("content"))
        for m in messages
        if m.get("role") == "user" and not m.get("_loop_authored")
    ]


async def test_11_messages_sent_during_the_final_call_are_the_next_request_in_order():
    """A message sent while the request's final model call runs is the
    session's next request (the current one is answered first); two such
    messages keep the order they were sent in."""
    holder: list = []

    def _answer_and_send(_messages):
        _drive(holder[0].submit("MESSAGE-A"))
        _drive(holder[0].submit("MESSAGE-B"))
        return _reply("First answer.")

    model = _Model([_answer_and_send, _reply("Second answer."), _reply("unused")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "First request.", persist=True)
            holder.append(handle)
            first = await _next_response(handle)
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert (first["content"], second["content"]) == ("First answer.", "Second answer.")
    texts = _user_texts(model.requests[1])
    assert [t for t in texts if t.startswith("MESSAGE-")] == ["MESSAGE-A", "MESSAGE-B"]
    assert not first.get("cancelled") and not second.get("cancelled")


@pytest.mark.parametrize("cancel_first", [False, True])
async def test_11_a_cancel_queued_as_a_request_ends_is_dropped(cancel_first):
    """A cancel still queued when a request ends was sent for that request:
    it is dropped, and a message queued beside it is the next request,
    answered and not cancelled."""
    model = _Model([_reply("First answer."), _reply("Answer to A."), _reply("unused")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "First request.", persist=True)
            put = handle._notification_q.put
            sent: list = []

            async def _put(item):
                # Just as the first request ends in its response, before
                # the loop parks.
                if item.get("type") == "response" and not sent:
                    sent.append(True)
                    if cancel_first:
                        _drive(handle.cancel_request("too late"))
                    _drive(handle.submit("MESSAGE-A"))
                    if not cancel_first:
                        _drive(handle.cancel_request("too late"))
                await put(item)

            handle._notification_q.put = _put
            first = await _next_response(handle)
            second = await _next_response(handle)
            await handle.stop("done")
            await asyncio.wait_for(handle.result(), 30)
    assert first["content"] == "First answer."
    assert second == {"type": "response", "content": "Answer to A."}
    assert "MESSAGE-A" in _user_texts(model.requests[1])
    assert len(model.requests) == 2


async def test_11_requeue_at_front_keeps_the_queue_order():
    """What a parked session takes off its queue goes back ahead of the rest."""
    from unify.common._async_tool.loop import _requeue_at_front

    queue: asyncio.Queue = asyncio.Queue()
    for item in ("first", "second", "third"):
        queue.put_nowait(item)
    _requeue_at_front(queue, queue.get_nowait())
    assert [queue.get_nowait() for _ in range(queue.qsize())] == [
        "first",
        "second",
        "third",
    ]


# ── 12. loop stop ──────────────────────────────────────────────────────────


async def test_12_loop_stop_ends_a_request_of_no_progress_cells(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP_K", 3)

    def _turn(messages):
        return _reply(calls=[_cell("print('thinking')")])

    def _last_word(messages):
        return _reply("Best answer: 7.")

    model = _Model([_turn, _turn, _turn, _last_word, _reply("unused")])
    async with _actor() as actor:
        with _scripted(model):
            handle = await _act(actor, "Find the number.")
            result = await asyncio.wait_for(handle.result(), 60)
    assert "Best answer: 7." in result
    assert len(model.requests) == 4
    # The loop stop fired (not the step limit or the model ending it).
    assert handle._runtime_state.loop_stops == 1
    assert "no-progress" in result or "Terminating early" in result
    assert model.requests[-1][-1]["content"].startswith("The last ")
