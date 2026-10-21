"""The agent loop with the tools in code (the actor's and its teammates'): ``execute_code`` is the request's
only model-API tool; over the compaction point the loop compacts by itself before the next model call, with the same
fork summary and restart message; and ``tools.answer(answer)`` from a cell
answers the request (validated against its response format) with no further model call.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tests import cache_discipline_helpers as h
from tests.scripted_model import Always, ScriptedModel, reply, scripted
from unify.actor import tool_handlers
from unify.common import tool_registry
from unify.common._async_tool import cell_reply
from unify.common._async_tool import context_compression as cc
from unify.common._async_tool.response_format import normalize_response_format

SUMMARY = "Summary: ran one cell; the answer is 42."
WAIT = 30


def _names(call) -> list[str]:
    return [t["function"]["name"] for t in call.request.get("tools") or []]


async def _run_loop(tools: dict, answer, *, response_format=None, fork=None):
    from unify.common.async_tool_loop import start_async_tool_loop

    model = ScriptedModel(actor=Always(answer), **({"compression_fork": fork} if fork else {}))
    client = h.new_client()
    with scripted(model):
        handle = start_async_tool_loop(
            client,
            "Run the cell, then answer.",
            tools,
            log_steps=False,
            timeout=WAIT,
            tools_in_code=True,
            response_format=response_format,
        )
        result = await asyncio.wait_for(handle.result(), WAIT)
    return result, model


@pytest.mark.asyncio
async def test_the_loop_compacts_before_the_next_call_with_no_forced_turn():
    ran = []

    async def execute_code(code: str) -> str:
        """Run a cell."""
        ran.append(code)
        return f"cell output: {code}"

    def answer(call):
        if any(SUMMARY in str(m.get("content")) for m in call.messages):
            return reply("done")
        return reply(calls=[("execute_code", {"code": "print(6 * 7)"})], prompt_tokens=cc.COMPACTION_AT_TOKENS)

    result, model = await _run_loop({"execute_code": execute_code}, answer, fork=[reply(SUMMARY)])
    assert result == "done" and ran == ["print(6 * 7)"]
    assert model.kinds() == ["actor", "compression_fork", "actor"]
    first, fork, restarted = model.calls
    assert _names(first) == ["execute_code"] and _names(restarted) == ["execute_code"]
    assert first.request["tools"] == tool_registry.json_tools("actor")
    # the summary request is the last request, then its reply and the cell's result, then the instruction
    def plain(messages):
        return [(m.get("role"), str(m.get("content"))) for m in messages]

    sent = fork.messages
    assert plain(sent[: len(first.messages)]) == plain(first.messages)
    assert any(m.get("role") == "tool" and "cell output: print(6 * 7)" in str(m.get("content")) for m in sent)
    restart = [m for m in restarted.messages if m.get("role") == "user"][-1]["content"]
    assert SUMMARY in str(restart) and "Context was compressed. Continue from where you left off." in str(restart)


@pytest.mark.asyncio
async def test_below_the_compaction_point_nothing_is_compacted():
    async def execute_code(code: str) -> str:
        """Run a cell."""
        return "ok"

    def answer(call):
        if any(m.get("role") == "tool" for m in call.messages):
            return reply("done")
        return reply(calls=[("execute_code", {"code": "1"})], prompt_tokens=cc.COMPACTION_AT_TOKENS - 1)

    result, model = await _run_loop({"execute_code": execute_code}, answer)
    assert result == "done" and model.kinds() == ["actor", "actor"]


def _answering_cell(validate=None):
    """An ``execute_code`` whose cell calls ``tools.answer``: the host handler, with this request's state."""

    async def execute_code(code: str) -> str:
        """Run a cell."""
        state = tool_handlers.ActorRequestState(reply_slot=cell_reply.current(), validate=validate)
        ctx = SimpleNamespace(agent="actor", name="root", state=state)
        out = await tool_handlers.answer(ctx, {"answer": json.loads(code)})
        return f"{out.status}: {out.text}"

    return execute_code


@pytest.mark.asyncio
async def test_answer_from_a_cell_answers_the_request_with_no_further_model_call():
    def answer(call):
        return reply(calls=[("execute_code", {"code": json.dumps("the answer is 42")})])

    result, model = await _run_loop({"execute_code": _answering_cell()}, answer)
    assert result == "the answer is 42"
    assert model.kinds() == ["actor"]


class Answer(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_answer_validates_against_the_response_format_and_a_refusal_lets_the_request_go_on():
    validate = tool_handlers.response_validator(normalize_response_format(Answer))

    def answer(call):
        tool_results = [str(m.get("content")) for m in call.messages if m.get("role") == "tool"]
        if not tool_results:
            return reply(calls=[("execute_code", {"code": json.dumps({"value": "not a number"})})])
        assert "not answered: the answer does not match the required response format" in tool_results[-1]
        return reply(calls=[("execute_code", {"code": json.dumps({"value": 42})})])

    result, model = await _run_loop(
        {"execute_code": _answering_cell(validate)},
        answer,
        response_format=Answer,
    )
    assert getattr(result, "value", None) == 42 or json.loads(result) == {"value": 42}
    assert model.kinds() == ["actor", "actor"]
    first = model.calls[0]
    # no response tool and no forced tool choice: the answer comes from tools.answer
    assert _names(first) == ["execute_code"] and first.request.get("tool_choice") != "required"
    assert any("tools.answer(answer)" in str(m.get("content")) for m in first.messages)


@pytest.mark.asyncio
async def test_a_second_answer_in_the_same_turn_is_refused():
    async def execute_code(code: str) -> str:
        """Run a cell."""
        state = tool_handlers.ActorRequestState(reply_slot=cell_reply.current())
        ctx = SimpleNamespace(agent="actor", name="root", state=state)
        first = await tool_handlers.answer(ctx, {"answer": "one"})
        second = await tool_handlers.answer(ctx, {"answer": "two"})
        return f"{first.status} {second.status} {second.form}"

    def answer(call):
        return reply(calls=[("execute_code", {"code": "x"})])

    result, model = await _run_loop({"execute_code": execute_code}, answer)
    assert result == "one" and model.kinds() == ["actor"]


@pytest.mark.asyncio
async def test_one_state_answers_each_request_of_a_session_once():
    # a persistent session keeps one request state; the loop takes each request's answer from the slot, so the
    # next request starts with it empty and may answer again, while a second answer before the take is refused
    slot = cell_reply.ReplySlot()
    state = tool_handlers.ActorRequestState(reply_slot=slot)
    ctx = SimpleNamespace(agent="actor", name="root", state=state)
    first = await tool_handlers.answer(ctx, {"answer": "one"})
    again = await tool_handlers.answer(ctx, {"answer": "two"})
    assert first.status == "ok" and again.status != "ok"
    assert slot.take()["content"] == "one"
    second = await tool_handlers.answer(ctx, {"answer": {"action": "submit"}})
    assert second.status == "ok"
    assert slot.take()["content"] == json.dumps({"action": "submit"})


