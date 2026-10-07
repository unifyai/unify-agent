"""Symbolic: a model call that ends with ``finish_reason: "error"``.

The Continual-ARC cell ``arc-pm2-up592-h0-medium-ws0`` (5 October, instance
19) was restarted after 300 s of silence, and its diagnosis named a LiteLLM
warning a minute earlier, ``Unmapped finish_reason 'error', defaulting to
'stop'``, as the cause. OpenRouter sends that reply with HTTP 200 when the
upstream provider fails part-way. The recorded body has
``finish_reason: "error"``, ``error: {code: 502, message: "Stream ended
before a terminal response event", metadata: {error_type:
"provider_unavailable"}}``, ``content: null`` (or the start of the text) and
a reasoning summary. LiteLLM maps the finish reason to ``stop`` and keeps the
original only under the choice's ``provider_specific_fields``.

That run had 17 such replies in 806 calls. Every one answered a request the
tool loop had already abandoned: a tool finished during the LLM race, the
loop cancelled that call 20 ms after sending it and sent a new one, and
unillm let the abandoned request finish in the background so it could bill
it. No error reply reached the loop. Unify sent five more requests after
the warning; the silence began at 19:52:31Z, after a delegated sub-actor asked
for clarification and then called ``wait``.

These tests drive the real ``CodeActActor`` under ``unify act --persist
--jsonl`` with such a reply as the one the loop waits on. The session always
answers and takes the next message; it is never silent.

The reply is built by LiteLLM's own response conversion from the recorded
body. The transport is scripted (``tests/cache_discipline_helpers.py``), so
nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import needs_bwrap

TASK = (
    "Solve the puzzle. Reply with one action as a JSON object on the last "
    'line: {"action": "request_demos"} or {"action": "submit", "grid": [[...]]}.'
)
FOLLOW_UP = "Feedback since your last action:\nDemos received (request 1)."
FINAL = '{"action": "submit", "grid": [[4]]}'
AFTER = '{"action":"request_demos"}'
PARTIAL = '{"action": "request_demos"}'
REASONING = "**Requesting action demos**\n\nI need to request demos first."
SUMMARY = "Nothing worth storing."
# Every wait is bounded; the recorded host gave up after 300 s.
WAIT_S = 30


def error_finish(content=None):
    """The reply LiteLLM hands unillm for OpenRouter's recorded error body."""
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )
    from litellm.types.utils import ModelResponse

    reasoning = REASONING
    raw = {
        "id": "gen-1791229867-error",
        "object": "chat.completion",
        "created": 1791229867,
        "model": "openai/gpt-6-luna",
        "provider": "OpenAI",
        "choices": [
            {
                "index": 0,
                "finish_reason": "error",
                "native_finish_reason": None,
                "logprobs": None,
                "error": {
                    "code": 502,
                    "message": "Stream ended before a terminal response event",
                    "metadata": {"error_type": "provider_unavailable"},
                },
                "message": {
                    "role": "assistant",
                    "content": content,
                    "refusal": None,
                    "reasoning": reasoning,
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "summary": reasoning,
                            "format": "openai-responses-v1",
                            "index": 0,
                        },
                    ],
                },
            },
        ],
        "usage": {
            "prompt_tokens": 18234,
            "completion_tokens": 225,
            "total_tokens": 18459,
            "completion_tokens_details": {"reasoning_tokens": 225},
        },
    }
    return convert_to_model_response_object(
        response_object=raw,
        model_response_object=ModelResponse(),
    )


def test_the_recorded_reply_reaches_unillm_as_a_stop():
    """LiteLLM's mapping, which is what the loop sees: no error, no text."""
    reply = error_finish()
    choice = reply.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content is None
    assert not choice.message.tool_calls
    assert choice.provider_specific_fields["native_finish_reason"] == "error"
    assert choice.provider_specific_fields["error"]["code"] == 502


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    # The storage review as shipped, or framed as the agent's own curation
    # step (UNIFY_REVIEW_FRAMING=unified, the default since the code freeze).
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


class _Model:
    """The session's first calls play ``first`` in order; later calls answer:
    ``AFTER`` before the follow-up, ``FINAL`` once it has arrived."""

    def __init__(self, first) -> None:
        self.first = list(first)
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(kw)
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        if self.first:
            return self.first.pop(0)
        if "Demos received (request 1)" in json.dumps(messages, default=str):
            return h.completion(content=FINAL)
        return h.completion(content=AFTER)


@pytest.fixture
def jsonl_session(monkeypatch):
    """``unify act --persist --jsonl`` on the real ``CodeActActor``."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    async def start(self) -> None:
        self._actor = CodeActActor()

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


async def _until(predicate, what: str, lines: list, model: "_Model") -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(poll(), WAIT_S)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"silent for {WAIT_S}s waiting for {what}; lines so far: {lines}; "
            f"model calls: {len(model.requests)}",
        ) from None


def _responses(lines: list[dict]) -> list[str]:
    return [line["content"] for line in lines if line["type"] == "response"]


# What the session answers first, by what its first calls return, and the
# session calls made before that answer. The first step is not gated
# (``tool_choice="auto"``: the discovery gate went with the library's JSON
# tools), so unillm makes no tool-choice retry and hands the loop the failed
# reply (its partial text, or, with no text, its reasoning summary promoted
# to content), still marked with the provider's error. The loop drops such a
# reply and sends the turn again, at most twice; only when every attempt
# fails is the failed reply kept, as shipped.
ERRORS = 3  # three loop attempts
CASES = {
    "no-text": ([None], AFTER, 2),
    "partial-text": ([PARTIAL], AFTER, 2),
    "no-text-twice": ([None, None], AFTER, 3),
    "partial-text-twice": ([PARTIAL, PARTIAL], AFTER, 3),
    "no-text-always": ([None] * ERRORS, REASONING, ERRORS),
    "partial-text-always": ([PARTIAL] * ERRORS, PARTIAL, ERRORS),
}


@pytest.fixture(autouse=True)
def quick_backoff(monkeypatch):
    from unify.common._async_tool import loop

    monkeypatch.setattr(loop, "_PROVIDER_ERROR_BACKOFF_S", 0.01)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(CASES))
async def test_a_session_answers_after_an_error_finish(jsonl_session, case):
    contents, expected_first, calls_before_answer = CASES[case]
    session, lines, send = jsonl_session
    model = _Model([error_finish(c) for c in contents])
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        try:
            await _until(
                lambda: len(_responses(lines)) >= 1,
                "the first response",
                lines,
                model,
            )
            send({"message": FOLLOW_UP})
            await _until(
                lambda: FINAL in _responses(lines),
                "the follow-up's answer",
                lines,
                model,
            )
            send({"quit": True})
            code = await asyncio.wait_for(run, WAIT_S)
        finally:
            if not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)

    assert code == 0
    assert [line["type"] for line in lines if line["type"] != "storage"] == [
        "response",
        "response",
        "result",
        "ended",
    ]
    first, answered = _responses(lines)
    assert first == expected_first
    assert answered == FINAL
    # A dropped reply's turn is sent again unchanged: same messages, tools
    # and tool choice as the turn whose reply failed.
    session_calls = [r for r in model.requests if not _is_review(r["messages"])]
    turn = session_calls[0]
    for retry in session_calls[1:calls_before_answer]:
        for key in ("messages", "tools", "tool_choice"):
            assert retry.get(key) == turn.get(key), key
    # The follow-up's step takes one call.
    assert len(session_calls) == calls_before_answer + 1


def test_only_a_reply_marked_by_the_provider_counts_as_failed():
    from unify.common._async_tool.loop import _completion_provider_error

    assert _completion_provider_error(error_finish()) == (
        "502: Stream ended before a terminal response event"
    )
    assert _completion_provider_error(error_finish(PARTIAL)) is not None
    # A normal reply, with text or tool calls, is not.
    assert _completion_provider_error(h.completion(content=FINAL)) is None
    assert _completion_provider_error(h.completion(calls=[("wait", {})])) is None
    assert _completion_provider_error(None) is None
