"""The memory agents' model turn keeps each call's usage report. unillm keeps the assistant reply in the client's
history without ``usage`` and reports it only on the call's event response; the turn attaches that report to the
message it returns, the pass and helper loops take it off before the message joins the conversation (no request
sends it back to the provider), and the compaction trigger and the pass's token counts read it.

The first tests go through unillm's real call path with a fake transport below it (no network, no key): the fake
response carries usage, as a provider's does. A scripted message that already holds usage still works.
"""

from __future__ import annotations

import asyncio
import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from unify.common._async_tool import cache_discipline
from unify.memory import writer_pass
from unify.memory.integration import consolidate
from unify.memory.writer_pass import WriterRoute, split_usage, unillm_turn
from tests.memory.test_helpers import _ctx as _helper_ctx
from tests.memory.test_helpers import _msg, _run as _run_helper, _tc
from tests.memory.test_writer_compaction import _events, _reads, _run
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_route import (  # noqa: F401 - fixtures
    REAL_MODEL,
    WRITER_BASE,
    WRITER_TOKEN,
    _otel_off_unless_a_test_turns_it_on,
    actor_gateway,
    restore_unillm,
)

INSTRUCTION = cache_discipline.COMPRESSION_FORK_INSTRUCTION
SUMMARY = "SUMMARY: read e1 cell 0"
PROMPT, CACHED, OUTPUT = 900, 600, 5
WINDOW = 1_000  # PROMPT is past the compaction point of this window


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return "".join(str(p.get("text", "")) for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


def _completion(reply: dict):
    from openai.types.chat import ChatCompletion

    calls = reply.get("tool_calls")
    return ChatCompletion.model_validate(
        {
            "id": "cmpl-usage",
            "object": "chat.completion",
            "created": 0,
            "model": REAL_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if calls else "stop",
                    "message": {
                        "role": "assistant",
                        "content": reply.get("content"),
                        "tool_calls": calls,
                    },
                },
            ],
            "usage": {
                "prompt_tokens": PROMPT,
                "completion_tokens": OUTPUT,
                "total_tokens": PROMPT + OUTPUT,
                "prompt_tokens_details": {"cached_tokens": CACHED},
                "cost": 0.001,
            },
        },
    )


@pytest.fixture
def transport(monkeypatch, actor_gateway, restore_unillm):
    """``litellm.acompletion`` replaced by a fake that keeps each request's messages and answers from *replies*
    (a request ending with the compaction instruction gets the summary; an empty queue, finish). unillm's response
    cache off; everything above the transport is unillm's own path."""
    import unillm.clients.uni_llm as uni_llm
    from unillm.settings import SETTINGS as unillm_settings

    monkeypatch.setenv("UNILLM_CACHE", "false")
    monkeypatch.setattr(unillm_settings, "UNILLM_CACHE", False)
    state = SimpleNamespace(sent=[], replies=[])

    async def fake_acompletion(*, shared_session=None, client=None, **kw):
        sent = copy.deepcopy(list(kw.get("messages") or []))
        state.sent.append(sent)
        if sent and _text(sent[-1]) == INSTRUCTION:
            reply = {"role": "assistant", "content": SUMMARY}
        elif state.replies:
            reply = state.replies.pop(0)
        else:
            reply = _call("fin", "finish", {"summary": "done"})
        # unillm streams every call of the harness's client: the reply as LiteLLM's mock stream of it
        from tests.scripted_model import as_requested

        return await as_requested(kw, _completion(reply))

    monkeypatch.setattr(uni_llm.litellm, "acompletion", fake_acompletion)
    return state


def _turn(origin: str = "memory_writer"):
    """The real turn on the writer's route, and every message list the loop handed it; the window is WINDOW."""
    turn = unillm_turn(
        REAL_MODEL,
        "low",
        origin=origin,
        route=WriterRoute(WRITER_BASE, SecretStr(WRITER_TOKEN)),
    )
    handed: list[list[dict]] = []

    async def recorded(messages, tools):
        handed.append(copy.deepcopy(messages))
        return await turn(messages, tools)

    recorded.max_input_tokens = WINDOW
    return recorded, handed


def _no_usage(requests: list[list[dict]]) -> bool:
    return not any(isinstance(m, dict) and "usage" in m for r in requests for m in r)


def test_the_real_turn_returns_the_calls_usage(transport):
    turn, _ = _turn()
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msg, usd = asyncio.run(turn(messages, []))
    assert msg["usage"]["prompt_tokens"] == PROMPT
    stripped, reported = split_usage(msg)
    assert "usage" not in stripped and "usage" in msg  # the turn's own dict is left as it is
    usage = writer_pass._usage_of(stripped, usd, reported)
    assert usage["input"] == PROMPT
    assert usage["cache_read"] == CACHED and usage["output"] == OUTPUT
    assert usage["usd"] == usd == "0.001"


def test_a_pass_on_the_real_turn_counts_tokens_compacts_and_never_sends_usage(
    transport,
    tmp_path,
    monkeypatch,
):
    transport.replies = _reads(3)
    turn, handed = _turn()
    out, writer = _run(tmp_path, turn)
    # every request the loop built, and every request the transport received, is free of the usage report
    assert len(transport.sent) == len(handed) > 2
    assert _no_usage(handed) and _no_usage(transport.sent)
    # the first call reported PROMPT, past the compaction point: the next call is the summary
    assert writer_pass.compaction_due(PROMPT, WINDOW)
    summaries = [r for r in handed if r and _text(r[-1]) == INSTRUCTION]
    assert summaries and handed.index(summaries[0]) == 1
    assert out.compactions >= 1
    # the pass's token counts: every call reported PROMPT and CACHED
    assert out.input_tokens == PROMPT * len(handed)
    assert out.cache_read_tokens == CACHED * len(handed)
    assert out.max_prompt_tokens == PROMPT and out.context_window == WINDOW
    # a tool call's row carries its model call's usage
    rows = [
        r
        for r in _events(tmp_path)
        if r.get("usage") and r.get("event") != "compaction"
    ]
    assert rows and rows[0]["usage"]["input"] == PROMPT
    assert rows[0]["usage"]["cache_read"] == CACHED
    # the pass's end event (its library part is read from the outcome)
    monkeypatch.setattr(consolidate, "_library_after", lambda stores: (None, None))
    end = consolidate._end_event(None, "p1", out, [], 1.0, None)
    assert end["input_tokens"] == PROMPT * len(handed)
    assert end["cache_read_tokens"] == CACHED * len(handed)
    assert end["max_prompt_tokens"] == PROMPT


def test_a_helper_on_the_real_turn_counts_tokens_compacts_and_never_sends_usage(
    transport,
    tmp_path,
):
    from unify.memory.integration.cost import HELPER_ORIGIN

    transport.replies = [
        _msg(_tc("o", "overview", {"episode": "e1"})),
        _msg(_tc("d", "done", {"summary": "s", "claims": []})),
    ]
    turn, handed = _turn(HELPER_ORIGIN)
    ctx = _helper_ctx(
        tmp_path,
        new_turn=lambda: turn,
        step_guard=6,
    )
    reply = _run_helper({"task": "t", "episodes": ["e1"]}, ctx)
    assert reply.startswith("helper call-1: ok"), reply
    assert _no_usage(handed) and _no_usage(transport.sent)
    summaries = [r for r in handed if r and _text(r[-1]) == INSTRUCTION]
    assert summaries and handed.index(summaries[0]) == 1
    status = json.loads((ctx.out_root / "call-1" / "helper.json").read_text())
    assert status["max_prompt_tokens"] == PROMPT
    rows = [json.loads(x) for x in ctx.events.read_text().splitlines()]
    tools = [r for r in rows if r.get("event") != "compaction" and r.get("usage")]
    assert tools and tools[0]["usage"]["input"] == PROMPT
    assert tools[0]["usage"]["cache_read"] == CACHED


def test_a_scripted_message_holding_usage_still_counts_and_is_not_changed(tmp_path):
    returned: list[dict] = []
    requests: list[list[dict]] = []
    queue = _reads(2)

    async def turn(messages, tools):
        requests.append(copy.deepcopy(messages))
        msg = queue.pop(0) if queue else _call("fin", "finish", {"summary": "done"})
        msg = {
            **msg,
            "usage": {
                "prompt_tokens": 400,
                "prompt_tokens_details": {"cached_tokens": 100},
            },
        }
        returned.append(msg)
        return msg, "0.001"

    turn.max_input_tokens = WINDOW
    out, _ = _run(tmp_path, turn)
    assert _no_usage(requests)
    assert all("usage" in m for m in returned)  # the loop copies; the turn's dicts keep their report
    assert out.input_tokens == 400 * len(requests)
    assert out.cache_read_tokens == 100 * len(requests)
    assert out.max_prompt_tokens == 400 and out.compactions == 0
    assert Decimal(out.usd) == Decimal("0.001") * len(requests)
