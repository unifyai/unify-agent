"""Symbolic: ``UNIFY_REPLY_RECEIPT=on`` shows facts about a drafted reply once.

On the office stream all 7 failed replies to the "meal" request answered
0.00 after a filter matched nothing, and the model ended its turn on that
draft. The offline gate (memo §E2) found two checks precise enough to show
before a reply ends a turn: C2_nc (a degenerate answer: 0, NaN, None, empty,
every item the same, or a JSON value identical to one in the request) and
C4_last (the last computing cell raised, or a cell caught and printed an
error, and the reply mentions no error); pooled they fired on 4.0% of 2,126
replies with precision 0.80.

With the switch on, a text reply (or a cell's ``reply()``) that would end the
turn and on which a check fires does not end it: the loop appends the facts
as one loop-authored message and makes one more model call, and whatever the
model replies next ends the turn, unchanged (it may repeat or revise). At
most one receipt per request; none at the step limit, on a cancel, or when
fewer than three steps remain. A persistent session emits only the final
reply as its response. Off: as shipped.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process. Every wait is bounded.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

REQUEST = "What did the team spend on meals in March? Reply with the total."
DRAFT = "The meals in March cost **0.00**."
REVISED = "The meals in March cost **31.20**."
RECEIPT_ZERO = "The answer is zero (0.00)."
RECEIPT_RAISED = (
    "Code cell 1 since the request raised `ValueError: boom`; the reply does "
    "not mention an error."
)
TRACEBACK = (
    'Traceback (most recent call last):\n  File "<cell>", line 1, in <module>\n'
    "ValueError: boom"
)
FINE = "The meals in March cost **31.20**."
WAIT_S = 2.0


@pytest.fixture
def receipt(monkeypatch):
    def set_(value: str = "on") -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_RECEIPT", value, raising=False)

    for name, value in (
        ("UNIFY_REPLY_CHANNEL", ""),
        ("UNIFY_STEP_CAP_REPLY", ""),
        ("UNIFY_BUDGET_FOOTER", False),
        ("UNIFY_REPEAT_GUARD", False),
        ("UNIFY_DISCOVERY_GATE", False),
        ("UNIFY_WORKSPACE_PYTHON", ""),
        ("UNIFY_REVIEW_LAST_REPLY", False),
    ):
        monkeypatch.setattr(SETTINGS, name, value)
    set_()
    return set_


def _cells(outputs: dict[str, str]):
    async def execute_code(code: str) -> str:
        """Run Python code.

        Args:
            code: The code to run.
        """
        return outputs.get(code, "")

    return execute_code


def _start(tools: dict, *, persist: bool = False, max_steps: int = 100, **kwargs):
    from unify.common.async_tool_loop import start_async_tool_loop

    return start_async_tool_loop(
        h.new_client(),
        REQUEST,
        tools,
        log_steps=False,
        timeout=30,
        persist=persist,
        max_steps=max_steps,
        reply_channel=True,
        **kwargs,
    )


async def _one_shot(replies, tools=None, **kwargs):
    with h.scripted(replies) as provider:
        handle = _start(tools or {}, **kwargs)
        result = await asyncio.wait_for(handle.result(), WAIT_S)
    return result, provider.requests, handle


def _receipts(messages: list) -> list[dict]:
    return [m for m in messages if m.get("_reply_receipt")]


def _state(handle):
    return handle._runtime_state


# ── off ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_off_the_draft_ends_the_turn_as_shipped(receipt):
    receipt("")
    result, requests, handle = await _one_shot([h.completion(content=DRAFT)])
    assert result == DRAFT
    assert len(requests) == 1
    assert _receipts(handle._client.messages) == []
    assert (_state(handle).receipts_shown, _state(handle).receipts_revised) == (0, 0)


# ── shown once; the next reply ends the turn ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_a_confirmed_draft_ends_the_turn_after_one_receipt(receipt):
    result, requests, handle = await _one_shot(
        [h.completion(content=DRAFT), h.completion(content=DRAFT)],
    )
    assert result == DRAFT
    assert len(requests) == 2
    second = requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["content"] == DRAFT
    # One loop-authored message holding the facts only; no marker reaches the
    # provider.
    assert second[-1] == {"role": "user", "content": RECEIPT_ZERO}
    (shown,) = _receipts(handle._client.messages)
    assert shown["_loop_authored"] is True and shown["content"] == RECEIPT_ZERO
    assert (_state(handle).receipts_shown, _state(handle).receipts_revised) == (1, 0)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_a_revised_reply_ends_the_turn_and_is_counted(receipt):
    result, requests, handle = await _one_shot(
        [h.completion(content=DRAFT), h.completion(content=REVISED)],
    )
    assert result == REVISED
    assert len(requests) == 2
    assert (_state(handle).receipts_shown, _state(handle).receipts_revised) == (1, 1)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_after_the_receipt_the_model_may_run_a_cell_and_no_second_receipt_follows(
    receipt,
):
    tools = {"execute_code": _cells({"print(total)": "0.00"})}
    result, requests, handle = await _one_shot(
        [
            h.completion(content=DRAFT),
            h.completion(calls=[("execute_code", {"code": "print(total)"})]),
            h.completion(content=DRAFT),
        ],
        tools,
    )
    assert result == DRAFT
    assert len(requests) == 3
    assert len(_receipts(handle._client.messages)) == 1
    assert (_state(handle).receipts_shown, _state(handle).receipts_revised) == (1, 0)


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_no_receipt_when_no_check_fires(receipt):
    result, requests, handle = await _one_shot([h.completion(content=FINE)])
    assert result == FINE and len(requests) == 1
    assert _receipts(handle._client.messages) == []


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_an_unmentioned_error_in_the_last_cell_is_shown(receipt):
    tools = {"execute_code": _cells({"d['k']": TRACEBACK})}
    result, requests, handle = await _one_shot(
        [
            h.completion(calls=[("execute_code", {"code": "d['k']"})]),
            h.completion(content="The answer is **7**."),
            h.completion(content="The lookup raised an error, so **7** is a guess."),
        ],
        tools,
    )
    assert result == "The lookup raised an error, so **7** is a guess."
    assert requests[2]["messages"][-1] == {"role": "user", "content": RECEIPT_RAISED}


# ── the step limit ───────────────────────────────────────────────────────


async def _messages_at_the_draft(receipt) -> int:
    """How many messages the loop holds when the one-tool-round draft lands."""
    receipt("")
    tools = {"execute_code": _cells({"x": "0.00"})}
    _, _, handle = await _one_shot(
        [
            h.completion(calls=[("execute_code", {"code": "x"})]),
            h.completion(content=DRAFT),
        ],
        tools,
        max_steps=1000,
    )
    receipt("on")
    return len(handle._client.messages)


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_no_receipt_when_it_and_the_next_reply_would_reach_the_step_limit(
    receipt,
):
    n = await _messages_at_the_draft(receipt)
    tools = {"execute_code": _cells({"x": "0.00"})}
    script = [
        h.completion(calls=[("execute_code", {"code": "x"})]),
        h.completion(content=DRAFT),
    ]
    # Two steps left: the receipt and the reply would end the request at the
    # limit, so the draft ends it as shipped.
    result, requests, handle = await _one_shot(list(script), tools, max_steps=n + 2)
    assert result == DRAFT and len(requests) == 2
    assert _receipts(handle._client.messages) == []
    # Three left: the receipt is shown and the reply after it ends the turn.
    result, requests, handle = await _one_shot(
        [*script, h.completion(content=REVISED)],
        tools,
        max_steps=n + 3,
    )
    assert result == REVISED and len(requests) == 3
    assert len(_receipts(handle._client.messages)) == 1


class _Model:
    """A scripted model whose replies are chosen by a function of the call."""

    def __init__(self, choose):
        self.choose = choose
        self.requests: list[list[dict]] = []
        self.in_flight = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        reply = self.choose(len(self.requests), messages)
        if reply is None:
            self.in_flight.set()
            await self.release.wait()
            return h.completion(content="a late reply")
        return reply


def _last_request(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not message.get("_loop_authored"):
            return str(message.get("content") or "")
    return ""


def _has_receipt(messages: list) -> bool:
    return any(m.get("content") == RECEIPT_ZERO for m in messages)


async def look() -> str:
    """Look again."""
    return "Nothing new."


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_no_receipt_at_the_step_cap_end_of_a_request(receipt, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "draft")
    model = _Model(lambda n, m: h.completion(content=DRAFT, calls=[("look", {})]))
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"look": look}, persist=True, max_steps=8)
        capped = (await asyncio.wait_for(h._next_response(handle), WAIT_S))["content"]
        await handle.stop()
    assert capped.startswith("🔚 Stopped at the step limit")
    assert DRAFT in capped
    assert not any(_has_receipt(r) for r in model.requests)
    assert _state(handle).receipts_shown == 0


# ── cancel ───────────────────────────────────────────────────────────────


async def study() -> str:
    """Study the data."""
    await asyncio.sleep(30)
    return "studied"


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_a_cancelled_request_ends_without_a_receipt(receipt):
    model = _Model(
        lambda n, m: (
            h.completion(content=DRAFT, calls=[("study", {})]) if n == 1 else None
        ),
    )
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({"study": study}, persist=True)
        await asyncio.sleep(0.2)
        await handle.cancel_request()
        cancelled = await asyncio.wait_for(h._next_response(handle), WAIT_S)
        await handle.stop()
    assert cancelled == {"type": "response", "content": DRAFT, "cancelled": True}
    assert not any(_has_receipt(r) for r in model.requests)
    assert _receipts(handle._client.messages) == []


@pytest.mark.asyncio
@pytest.mark.timeout(20)
async def test_a_cancel_during_the_receipts_call_ends_the_request_and_the_next_gets_its_own(
    receipt,
):
    follow_up = "And in April?"

    def choose(n, messages):
        # The provider never sees the loop's markers, so the receipt reads as
        # the latest user message.
        after_receipt = _has_receipt(messages[-1:])
        if any(_has_receipt([m]) for m in messages[:-1]) or (
            _last_request(messages[:-1] if after_receipt else messages) == follow_up
        ):
            return h.completion(content=REVISED if after_receipt else DRAFT)
        if n == 1:
            return h.completion(content=DRAFT)
        return None  # the first request's receipt call: held until the cancel

    model = _Model(choose)
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        handle = _start({}, persist=True)
        await asyncio.wait_for(model.in_flight.wait(), WAIT_S)
        await handle.cancel_request()
        cancelled = await asyncio.wait_for(h._next_response(handle), WAIT_S)
        await handle.interject(follow_up)
        answered = await asyncio.wait_for(h._next_response(handle), WAIT_S)
        await handle.stop()
    # The cancel's response carries the draft; no reply after the receipt
    # was made, so none is counted as a revision.
    assert cancelled == {"type": "response", "content": DRAFT, "cancelled": True}
    # The next request is a new one: its own draft gets its own receipt.
    assert answered["content"] == REVISED
    assert len(_receipts(handle._client.messages)) == 2
    assert (_state(handle).receipts_shown, _state(handle).receipts_revised) == (2, 1)


# ── neighbours: the budget footer and the turn-boundary hook ─────────────


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_the_receipt_carries_no_budget_footer(receipt, monkeypatch):
    rounds = 45
    tools = {"execute_code": _cells({"x": "0.00"})}
    script = [h.completion(calls=[("execute_code", {"code": "x"})])] * rounds
    receipt("")
    _, _, handle = await _one_shot(
        [*script, h.completion(content=DRAFT)],
        tools,
        max_steps=1000,
    )
    n = len(handle._client.messages)
    receipt("on")
    monkeypatch.setattr(SETTINGS, "UNIFY_BUDGET_FOOTER", True)
    result, requests, handle = await _one_shot(
        [*script, h.completion(content=DRAFT), h.completion(content=REVISED)],
        tools,
        max_steps=n + 3,
    )
    assert result == REVISED
    last = requests[-1]["messages"]
    tool_results = [m for m in last if m.get("role") == "tool"]
    assert "[step budget]" in str(tool_results[-1]["content"])
    assert last[-1] == {"role": "user", "content": RECEIPT_ZERO}


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_the_record_block_follows_the_receipt_before_the_extra_call(receipt):
    pending: list[str] = []
    calls = {"n": 0}

    async def on_turn_boundary():
        calls["n"] += 1
        if not pending:
            return None
        text = "[record]\n" + "\n".join(pending)
        pending.clear()
        return text

    def draft():
        pending.append("posted during the draft's call")
        return h.completion(content=DRAFT)

    result, requests, handle = await _one_shot(
        [draft, h.completion(content=DRAFT)],
        on_turn_boundary=on_turn_boundary,
    )
    assert result == DRAFT and len(requests) == 2
    tail = requests[1]["messages"][-3:]
    assert [m["role"] for m in tail] == ["assistant", "user", "user"]
    assert tail[1]["content"] == RECEIPT_ZERO
    assert tail[2]["content"].endswith("[record]\nposted during the draft's call")
    # Once before each model call; never after the turn ended.
    assert calls["n"] == 2


# ── a cell's reply() (UNIFY_REPLY_CHANNEL=code+text) ─────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_a_cells_reply_gets_the_receipt_too(receipt, monkeypatch):
    from unify.actor.code_act_actor import CodeActActor
    from unify.common._async_tool import cell_reply

    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")
    actor = CodeActActor()
    try:
        replies = [
            h.completion(
                calls=[("execute_code", {"code": "total = 0\nreply(str(total))"})],
            ),
            h.completion(content="12"),
        ]
        with h.scripted(replies) as provider:
            handle = await actor.act(
                REQUEST,
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            result = await asyncio.wait_for(handle.result(), WAIT_S)
        messages = list(handle._client.messages)
        state = handle._runtime_state
    finally:
        await actor.close()
    assert result == "12"
    assert len(provider.requests) == 2
    assert provider.requests[1]["messages"][-1] == {
        "role": "user",
        "content": "The answer is zero (0).",
    }
    draft, shown, final = messages[-3:]
    assert draft[cell_reply.SOURCE_KEY] == "cell" and draft["content"] == "0"
    assert shown.get("_reply_receipt") is True
    assert final[cell_reply.SOURCE_KEY] == "text" and final["content"] == "12"
    assert (state.receipts_shown, state.receipts_revised) == (1, 1)


# ── unify act --persist --jsonl ──────────────────────────────────────────

TASK = "Save the March report."
SAVED = "Done: the report is saved."
SUMMARY = "Nothing worth storing."
HOST_ENDED = "(The host then ended the session; no outcome was posted.)"


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    # The storage review as shipped, or framed as the agent's own curation
    # step (UNIFY_REVIEW_FRAMING=unified, the default since the code freeze).
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


class _SessionModel:
    """The task is answered at once; the follow-up first with the zero draft,
    then, after the receipt, with the revision."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        # The provider never sees the loop's markers: the receipt reads as
        # the latest user message.
        if messages[-1].get("content") == RECEIPT_ZERO:
            return h.completion(content=REVISED)
        if REQUEST in _last_request(messages):
            return h.completion(content=DRAFT)
        return h.completion(content=SAVED)


class _Actor:
    """The actor ``unify act`` starts: its own execute_code on a persistent loop."""

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client(TASK),
            request,
            {"execute_code": self._actor.get_tools("act")["execute_code"]},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=300,
            persist=persist,
            reply_channel=True,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch, receipt):
    """``unify act --persist --jsonl`` on the scripted actor; stdin is a pipe."""
    from unify.cli import Act

    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_LAST_REPLY", True)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    async def start(self) -> None:
        self._actor = _Actor()

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=True,
            no_compose=False,
            no_store=False,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send
    os.close(write_fd)


async def _until(predicate, timeout: float = WAIT_S) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _responses(lines: list[dict]) -> list[dict]:
    return [line for line in lines if line["type"] == "response"]


@pytest.mark.asyncio
@pytest.mark.timeout(15)
async def test_the_cli_response_line_carries_only_the_final_reply(jsonl_session):
    session, lines, send = jsonl_session
    model = _SessionModel()
    started = time.monotonic()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: len(_responses(lines)) == 1)
        send({"message": REQUEST})
        await _until(lambda: len(_responses(lines)) == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, 2 * WAIT_S)
    assert time.monotonic() - started < 3 * WAIT_S
    assert code == 0
    # The draft was never a response: the host sees the reply after the receipt.
    assert _responses(lines) == [
        {"type": "response", "content": SAVED},
        {"type": "response", "content": REVISED},
    ]
    assert DRAFT not in json.dumps(lines)
    (result,) = [line for line in lines if line["type"] == "result"]
    assert result["run_stats"] == {"receipts_shown": 1, "receipts_revised": 1}
    # UNIFY_REVIEW_LAST_REPLY: the storage review reads the final reply.
    reviews = [r for r in model.requests if _is_review(r)]
    assert reviews
    review_text = json.dumps(reviews[-1], default=str)
    assert json.dumps(f"{REVISED}\n\n{HOST_ENDED}")[1:-1] in review_text
    assert json.dumps(f"{DRAFT}\n\n{HOST_ENDED}")[1:-1] not in review_text
