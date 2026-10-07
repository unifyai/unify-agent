"""Symbolic: ``UNIFY_LOOP_STOP`` ends a request whose tool calls stop making progress.

The loop census of 7 October (87,147 recorded requests) found that most
loops inside one request are no-op narration. The model runs
``print('Request another demo.')``, ``print('')`` or ``pass`` instead of
replying. The worst case is ARC LOW S r1, instance 15: 150 such prints with
45 distinct strings, over 343 s, until the per-request step cap. A rule that
looks only for exact repeats misses it.

With the switch, a model call is "no progress" when every tool call it makes
either runs a cell that does nothing (only ``pass``, comments, prints of
constant text, bare constants) or repeats one of the two calls before it,
exactly or with only its literals changed, and gets the same result. A
result that is not known yet never counts. K such calls in a row
(``UNIFY_LOOP_STOP_K``, default 10) end the request through the
per-request step-limit path: the model is asked for its best answer in one
tool-less turn (or the draft is quoted, under ``UNIFY_STEP_CAP_REPLY=draft``),
and a persistent session takes the next request as usual. Any other call, a
text reply or a new requester message resets the count. Off: as shipped.

The transport is scripted, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS, ProductionSettings

TASK = "Work out the total and reply with it."
CONTINUE = "Please continue."
LAST_WORD = "My best answer: 45."
DRAFT = "Still working; best so far: 45."
K = 10
BOUND = 5  # seconds; a scripted run takes well under one


def _headline(k: int = K) -> str:
    return (
        f"🔚 Stopped: the last {k} tool calls made no progress (each ran a "
        "cell that does nothing, or repeated a recent call and got the same "
        "result), so this request ended before it was finished. The session "
        "is still open: the next message starts a new request."
    )


NOTICE_HEAD = "The last "  # the loop stop's notice

# The loop's own notices (call lifecycle, record blocks) reach the provider
# without their ``_loop_authored`` marker; none is a requester message.
_NOTICE = re.compile(r"^\[(?:steerable|askable|h\d+|record|note)[ :\]]")


def _is_notice(message: dict) -> bool:
    return (
        bool(message.get("_loop_authored"))
        or bool(
            _NOTICE.match(str(message.get("content") or "")),
        )
        or str(message.get("content") or "").startswith(NOTICE_HEAD)
    )


def _request(messages: list) -> tuple[str, int]:
    """The current request's text and the tool-calling turns made for it."""
    turns = 0
    for message in reversed(messages):
        if message.get("role") == "user" and not _is_notice(message):
            return str(message.get("content") or ""), turns
        if message.get("role") == "assistant" and message.get("tool_calls"):
            turns += 1
    return "", turns


def _code(code: str) -> list:
    return [("execute_code", {"thought": "Running it.", "code": code})]


class _Model:
    """Plays *plan(request, n)*: the calls of the request's turn *n*, or
    ``None`` to reply. A tool-less request (the last word) gets *last_word*."""

    def __init__(self, plan, *, draft=None, last_word=LAST_WORD):
        self.plan = plan
        self.draft = draft
        self.last_word = last_word
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append({"messages": messages, "tools": kw.get("tools")})
        if not kw.get("tools"):
            return h.completion(content=self.last_word)
        request, n = _request(messages)
        calls = self.plan(request, n)
        if calls is None:
            return h.completion(content=f"done: {request}")
        return h.completion(content=self.draft, calls=calls)

    @property
    def toolless(self) -> list[dict]:
        return [r for r in self.requests if not r["tools"]]

    def tool_turns(self) -> int:
        """Replies with tool calls (every request offering tools but the last)."""
        return sum(1 for r in self.requests if r["tools"])


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


_POLLS = {"n": 0}


async def status() -> str:
    """How far the job has got."""
    _POLLS["n"] += 1
    return f"job at {_POLLS['n'] * 5}%"


async def poke() -> str:
    """Poke the service (its answer comes later)."""
    return ""


async def pending() -> str:
    """Start a job."""
    return json.dumps({"_placeholder": "pending"})


TOOLS = {
    "execute_code": execute_code,
    "status": status,
    "poke": poke,
    "pending": pending,
}


@pytest.fixture
def loop_stop(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP_K", K)
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "")
    _POLLS["n"] = 0


def _install(model) -> None:
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model


def _start(*, persist: bool = True, max_steps: int = 300, **kwargs):
    """The actor's task loop: it answers a requester, and no loop started it."""
    from unify.common.async_tool_loop import start_async_tool_loop

    kwargs.setdefault("reply_channel", True)
    kwargs.setdefault("bind_request", True)
    return start_async_tool_loop(
        h.new_client(),
        TASK,
        TOOLS,
        log_steps=False,
        timeout=30,
        persist=persist,
        max_steps=max_steps,
        **kwargs,
    )


async def _session(model, *requests: str, **kwargs) -> tuple[list[str], object]:
    """Run one persistent session: the task, then *requests*; the responses."""
    with h.scripted(()):
        _install(model)
        handle = _start(**kwargs)
        responses = [
            (await asyncio.wait_for(h._next_response(handle), BOUND))["content"],
        ]
        for request in requests:
            await handle.interject(request)
            responses.append(
                (await asyncio.wait_for(h._next_response(handle), BOUND))["content"],
            )
        stats = handle._runtime_state
        await handle.stop()
        await asyncio.wait_for(handle.result(), BOUND)
    return responses, stats


def _then_reply(calls_for, limit: int):
    """Turns 0..limit-1 make *calls_for(n)*; then the model replies."""

    def plan(request: str, n: int):
        if request == TASK and n < limit:
            return calls_for(n)
        return None

    return plan


def _assert_every_call_answered(messages: list) -> None:
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in messages:
        for call in message.get("tool_calls") or []:
            assert call["id"] in answered, call


# ── stops ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_prints_stop_at_k_with_the_models_last_word(loop_stop):
    model = _Model(_then_reply(lambda n: _code("print('')"), 50))
    (stopped, answered), stats = await _session(model, CONTINUE)

    assert stopped == f"{_headline()}\n\nBest current answer:\n{LAST_WORD}"
    assert answered == f"done: {CONTINUE}"
    # K turns with tools, then one tool-less turn; then the next request.
    (toolless,) = model.toolless
    assert model.requests.index(toolless) == K
    notice = toolless["messages"][-1]
    assert notice.get("role") == "user"
    assert str(notice["content"]).startswith(f"The last {K} tool calls")
    _assert_every_call_answered(toolless["messages"])
    assert stats.loop_stops == 1
    # The next request sees the notice and the answer, not a second reply.
    last = model.requests[-1]["messages"]
    assert any(m.get("content") == LAST_WORD for m in last)
    assert not any(str(m.get("content")).startswith("🔚") for m in last)


@pytest.mark.asyncio
async def test_distinct_constant_prints_stop_at_k(loop_stop):
    """The S r1 pattern: no two prints alike, so no exact repeat."""
    lines = [f"Request another example ({i})." for i in range(45)]
    model = _Model(
        _then_reply(lambda n: _code(f"# narrate\nprint({lines[n % 45]!r})"), 150),
    )
    (stopped,), stats = await _session(model)

    assert stopped.startswith(_headline())
    assert model.tool_turns() == K
    assert stats.loop_stops == 1


@pytest.mark.asyncio
async def test_identical_computation_with_identical_results_stops(loop_stop):
    code = "x = sum(range(10))\nprint(x)"
    model = _Model(_then_reply(lambda n: _code(code), 50))
    (stopped,), stats = await _session(model)

    assert stopped.startswith(_headline())
    # The first call is new; the next K repeat it.
    assert model.tool_turns() == K + 1
    assert stats.loop_stops == 1


@pytest.mark.asyncio
async def test_near_identical_calls_with_the_same_result_stop(loop_stop):
    """Only literals change, and so does nothing else: the result repeats."""
    cells = ["print(sum([1, 2, 3]))", "print(sum([3, 2, 1]))  # again"]
    model = _Model(_then_reply(lambda n: _code(cells[n % 2]), 50))
    (stopped,), _ = await _session(model)

    assert stopped.startswith(_headline())
    assert model.tool_turns() == K + 1


@pytest.mark.asyncio
async def test_k_is_configurable(loop_stop, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP_K", 3)
    model = _Model(_then_reply(lambda n: _code("pass"), 50))
    (stopped,), _ = await _session(model)

    assert stopped.startswith(_headline(3))
    assert model.tool_turns() == 3


@pytest.mark.asyncio
async def test_under_draft_the_stop_quotes_the_draft(loop_stop, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "draft")
    model = _Model(_then_reply(lambda n: _code("print('')"), 50), draft=DRAFT)
    (stopped, answered), stats = await _session(model, CONTINUE)

    assert stopped == f"{_headline()}\n\nBest current answer:\n{DRAFT}"
    assert answered == f"done: {CONTINUE}"
    assert model.toolless == []
    assert model.tool_turns() == K + 1  # K, then the next request's reply
    assert stats.loop_stops == 1


@pytest.mark.asyncio
async def test_under_last_word_the_stop_asks_for_the_answer(loop_stop, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "last_word")
    model = _Model(_then_reply(lambda n: _code("pass"), 50), draft=DRAFT)
    (stopped,), stats = await _session(model)

    assert stopped == f"{_headline()}\n\nBest current answer:\n{LAST_WORD}"
    assert len(model.toolless) == 1
    assert (stats.step_cap_last_word_turns, stats.loop_stops) == (1, 1)


@pytest.mark.asyncio
async def test_a_loop_that_is_not_persistent_ends_with_the_last_word(loop_stop):
    model = _Model(_then_reply(lambda n: _code("print('')"), 50))
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False)
        result = await asyncio.wait_for(handle.result(), BOUND)

    assert result == (
        f"🔚 Terminating early: the last {K} tool calls made no progress"
        f"\n\nBest current answer:\n{LAST_WORD}"
    )
    assert model.tool_turns() == K


@pytest.mark.asyncio
async def test_record_blocks_at_the_boundary_do_not_reset_the_count(loop_stop):
    """UNIFY_AGENTS=record: a loop-authored block is not a requester message."""
    blocks = []

    async def boundary():
        blocks.append(1)
        return f"[record] entry {len(blocks)}"

    model = _Model(_then_reply(lambda n: _code("pass"), 50))
    (stopped,), _ = await _session(model, on_turn_boundary=boundary)

    assert stopped.startswith(_headline())
    assert model.tool_turns() == K
    # The stop came before the boundary of the turn it replaced.
    assert len(blocks) == K


# ── no stop ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_similar_code_over_different_data_does_not_stop(loop_stop):
    model = _Model(_then_reply(lambda n: _code(f"print(sum(range({n})))"), 25))
    (reply,), stats = await _session(model)

    assert reply == f"done: {TASK}"
    assert model.tool_turns() == 26
    assert stats.loop_stops == 0


@pytest.mark.asyncio
async def test_polling_with_changing_results_does_not_stop(loop_stop):
    model = _Model(_then_reply(lambda n: [("status", {})], 25))
    (reply,), stats = await _session(model)

    assert reply == f"done: {TASK}"
    assert stats.loop_stops == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["poke", "pending"])
async def test_unknown_results_never_count_as_repeated(loop_stop, tool):
    model = _Model(_then_reply(lambda n: [(tool, {})], 25))
    (reply,), stats = await _session(model)

    assert reply == f"done: {TASK}"
    assert model.tool_turns() == 26
    assert stats.loop_stops == 0


@pytest.mark.asyncio
async def test_a_real_call_between_no_ops_resets_the_count(loop_stop):
    def calls(n: int):
        if n % K == K - 1:
            return _code(f"print(sum(range({n})))")
        return _code("print('')")

    model = _Model(_then_reply(calls, 3 * K))
    (reply,), stats = await _session(model)

    assert reply == f"done: {TASK}"
    assert stats.loop_stops == 0


@pytest.mark.asyncio
async def test_the_count_starts_again_with_each_request(loop_stop):
    def plan(request: str, n: int):
        return _code("pass") if n < K - 1 else None

    model = _Model(plan)
    replies, stats = await _session(model, CONTINUE, "And again.")

    assert replies == [f"done: {TASK}", f"done: {CONTINUE}", "done: And again."]
    assert stats.loop_stops == 0


async def _one_shot(model, **kwargs) -> str:
    with h.scripted(()):
        _install(model)
        handle = _start(persist=False, **kwargs)
        return await asyncio.wait_for(handle.result(), BOUND)


@pytest.mark.asyncio
async def test_a_sub_agents_loop_never_stops(loop_stop):
    """A loop started inside another loop's call (a sub-agent): its lineage
    names its parent, and the actor hands it the parent's conversation."""
    from unify.common._async_tool.loop_config import TOOL_LOOP_LINEAGE

    model = _Model(_then_reply(lambda n: _code("print('')"), K + 5))
    token = TOOL_LOOP_LINEAGE.set(["CodeActActor.act(ab12)"])
    try:
        assert await _one_shot(model) == f"done: {TASK}"
    finally:
        TOOL_LOOP_LINEAGE.reset(token)
    assert model.tool_turns() == K + 6
    assert model.toolless == []

    model = _Model(_then_reply(lambda n: _code("print('')"), K + 5))
    assert await _one_shot(model, parent_chat_context=[]) == f"done: {TASK}"
    assert model.tool_turns() == K + 6


@pytest.mark.asyncio
async def test_a_loop_that_answers_no_requester_never_stops(loop_stop):
    """A review, its fork, a routing question: no reply channel."""
    model = _Model(_then_reply(lambda n: _code("print('')"), K + 5))
    result = await _one_shot(model, reply_channel=False, bind_request=False)

    assert result == f"done: {TASK}"
    assert model.tool_turns() == K + 6
    assert model.toolless == []


@pytest.mark.asyncio
async def test_off_the_loop_runs_as_shipped(loop_stop, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "")
    model = _Model(_then_reply(lambda n: _code("print('')"), K + 5))
    (reply,), stats = await _session(model)

    assert reply == f"done: {TASK}"
    assert model.tool_turns() == K + 6
    assert model.toolless == []
    assert stats.loop_stops == 0


# ── the rule ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "code",
    [
        "",
        "pass",
        "# just a note",
        "print()",
        "print('')",
        "print('Requesting the next item now.')",
        "print(f'Done.')",
        "print('a', 'b', sep='')",
        "...",
        "None",
        "'''A note.'''",
        "%matplotlib inline\nprint('ready')",
        "pass\n# then\npass",
    ],
)
def test_cells_that_do_nothing(code):
    from unify.common._async_tool import loop_stop

    assert loop_stop.is_noop_cell(code)


@pytest.mark.parametrize(
    "code",
    [
        "print(x)",
        "x = 1",
        "print(f'{x}')",
        "import os",
        "!ls",
        "print('a' + b)",
        "print(len('abc'))",
        "def f():\n    pass",
    ],
)
def test_cells_that_do_something(code):
    from unify.common._async_tool import loop_stop

    assert not loop_stop.is_noop_cell(code)


def test_only_python_cells_can_be_no_ops():
    from unify.common._async_tool import loop_stop

    python = loop_stop.call_record("execute_code", {"code": "pass"})
    shell = loop_stop.call_record(
        "execute_code",
        {"code": "pass", "language": "bash"},
    )
    assert python.noop and not shell.noop


def test_the_thought_comments_and_whitespace_are_not_part_of_the_action():
    from unify.common._async_tool import loop_stop

    a = loop_stop.call_record(
        "execute_code",
        {"thought": "First.", "code": "x = f(1)\nprint(x)"},
    )
    b = loop_stop.call_record(
        "execute_code",
        json.dumps({"thought": "Again.", "code": "x = f(1)  # once more\n\nprint(x)"}),
    )
    c = loop_stop.call_record("execute_code", {"code": "x = f(2)\nprint(x)"})
    assert a.near == b.near == c.near
    assert a.exact == b.exact != c.exact


def test_results_are_compared_without_times_ids_durations_or_the_footer():
    from unify.common._async_tool import loop_stop

    a = loop_stop.normalise_result(
        "done at 2026-10-07T10:00:00Z id 3f2a9c1e-1111-4222-8333-444455556666 "
        "obj 0x7f00ab in 1.5s",
    )
    b = loop_stop.normalise_result(
        "done at 2026-10-07T11:22:33.5+00:00 id 0aa2b9c1-9999-4888-8777-666655554444 "
        "obj 0x7f99cd in 20 ms\n\n[step budget] 3 of 30 steps left before the "
        "step limit stops this request (each message is a step: a tool call and "
        "its result take two).",
    )
    assert a == b
    assert loop_stop.normalise_result("done in 2 steps") != a


def _turn(i: int, code: str, result, *, name: str = "execute_code") -> list[dict]:
    call_id = f"c{i}"
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"code": code})},
            },
        ],
    }
    if result is None:
        return [assistant]
    return [assistant, {"role": "tool", "tool_call_id": call_id, "content": result}]


def test_the_tracker_counts_resets_and_fires():
    from unify.common._async_tool import loop_stop

    tracker = loop_stop.Tracker(k=3)
    messages: list[dict] = [{"role": "user", "content": TASK}]
    fired = []
    for i, (code, result) in enumerate(
        [
            ("pass", ""),  # 1
            ("print(f(1))", "7"),  # new: 0
            ("print(f(1))", "7"),  # repeat: 1
            ("print(f(2))", "7"),  # near, same result: 2
            ("print(f(3))", "8"),  # new result: 0
            ("pass", ""),  # 1
            ("print('x')", "x"),  # 2
            ("print(f(3))", "8"),  # three calls back, outside W=2: 0
            ("pass", ""),  # 1
            ("print('')", "\n"),  # 2
            ("...", ""),  # 3: fires
        ],
    ):
        messages += _turn(i, code, result)
        fired.append(tracker.observe(messages))
    assert fired == [False] * 10 + [True]


def test_in_flight_calls_unknown_results_and_new_requests_reset():
    from unify.common._async_tool import loop_stop

    tracker = loop_stop.Tracker(k=2)
    messages: list[dict] = [{"role": "user", "content": TASK}]
    messages += _turn(0, "pass", "")
    assert not tracker.observe(messages)
    # A call made while others run: forced turns are not counted.
    messages += _turn(1, "pass", "")
    assert not tracker.observe(messages, in_flight=True)
    # A loop-authored notice does not start a request.
    messages.append({"role": "user", "content": "[note]", "_loop_authored": True})
    messages += _turn(2, "pass", "")
    messages += _turn(3, "pass", "")
    assert tracker.observe(messages)

    tracker = loop_stop.Tracker(k=2)
    messages = [{"role": "user", "content": TASK}]
    messages += _turn(0, "print(f())", "1")  # new
    messages += _turn(1, "print(f())", None)  # no result yet
    messages += _turn(2, "print(f())", '{"_placeholder": "pending"}')
    messages += _turn(3, "print(f())", "1")  # the two before it are unknown
    assert not tracker.observe(messages)
    messages += _turn(4, "print(f())", "1")  # a repeat: 1
    assert not tracker.observe(messages)
    messages.append({"role": "user", "content": "A new request."})
    messages += _turn(5, "pass", "")
    assert not tracker.observe(messages)  # 1, counted from the new request


def test_an_async_result_is_read_from_its_completion_pair():
    from unify.common._async_tool import loop_stop

    tracker = loop_stop.Tracker(k=2)
    messages: list[dict] = [{"role": "user", "content": TASK}]
    for i in range(3):
        call_id = f"c{i}"
        messages += _turn(i, "print(f())", '{"_placeholder": "pending"}')
        messages += [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"{call_id}_completed",
                        "type": "function",
                        "function": {
                            "name": f"check_status_{call_id}",
                            "arguments": "{}",
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": f"{call_id}_completed", "content": "1"},
        ]
    # The completion stubs are the loop's, not model calls: three model calls,
    # two of them repeats with the same result.
    assert tracker.observe(messages)


# ── settings and stats ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, mode",
    [("", ""), ("off", ""), ("on", "on"), ("ON", "on")],
)
def test_the_switch_takes_on_or_off(value, mode):
    assert ProductionSettings(UNIFY_LOOP_STOP=value).UNIFY_LOOP_STOP == mode


@pytest.mark.parametrize("value", ["yes", "true", "10"])
def test_the_switch_refuses_anything_else(value):
    with pytest.raises(ValueError, match="UNIFY_LOOP_STOP"):
        ProductionSettings(UNIFY_LOOP_STOP=value)


@pytest.mark.parametrize("value, k", [("4", 4), (25, 25), ("", 10), (None, 10)])
def test_k_is_a_positive_whole_number(value, k):
    assert ProductionSettings(UNIFY_LOOP_STOP_K=value).UNIFY_LOOP_STOP_K == k


@pytest.mark.parametrize("value", ["0", -1, "2.5", 2.5, "ten", True])
def test_k_refuses_anything_else(value):
    with pytest.raises(ValueError, match="UNIFY_LOOP_STOP_K"):
        ProductionSettings(UNIFY_LOOP_STOP_K=value)


def test_the_sessions_run_stats_count_loop_stops_only_when_on(monkeypatch):
    from types import SimpleNamespace

    from unify.actor.code_act_actor import _StorageCheckHandle

    handle = SimpleNamespace(
        _meter=None,
        _inner=SimpleNamespace(_runtime_state=SimpleNamespace(loop_stops=2)),
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "")
    assert _StorageCheckHandle.run_stats.fget(handle) == {}
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "on")
    assert _StorageCheckHandle.run_stats.fget(handle) == {"loop_stops": 2}


# ── beside the step-cap compaction and the reply receipt ─────────────────


def _is_compactor(messages: list) -> bool:
    return any(
        m.get("role") == "system"
        and "You are a context compactor" in str(m.get("content"))
        for m in messages
    )


@pytest.mark.asyncio
async def test_a_loop_stop_never_compacts(loop_stop, monkeypatch):
    """UNIFY_STEP_CAP_COMPACT compacts at the step limit only: a loop stop
    is not one, so it ends the request with its own reply."""
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_COMPACT", "on")
    model = _Model(_then_reply(lambda n: _code("print('')"), 50))
    (stopped, answered), stats = await _session(model, CONTINUE, max_steps=40)

    assert stopped == f"{_headline()}\n\nBest current answer:\n{LAST_WORD}"
    assert answered == f"done: {CONTINUE}"
    assert not any(_is_compactor(r["messages"]) for r in model.requests)
    assert (stats.step_cap_compactions, stats.loop_stops) == (0, 1)
