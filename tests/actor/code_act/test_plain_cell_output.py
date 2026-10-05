"""Symbolic: ``UNIFY_PLAIN_CELL_OUTPUT``: a cell's result reads as a notebook cell's.

As shipped every result opens with a JSON envelope (``result``, ``error``,
``state_mode``, ``session_id``, ``session_name``, ``session_created``,
``duration_ms``), then ``--- stdout ---`` and ``--- stderr ---`` sections. With
the switch on: stdout, then stderr after ``[stderr]``, then ``Out: <repr>``,
then the traceback. A result holding a steerable handle keeps the envelope,
so the loop adopts the handle exactly as shipped; what steered a block is a
last line.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.helpers import patch_actor_act
from tests.helpers import _handle_project
from unify.actor.execution.types import ExecutionResult, ImagePart, TextPart
from unify.common._async_tool.tools_data import _extract_nested_handle
from unify.settings import SETTINGS

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", True)


def _text(blocks):
    return "".join(b["text"] for b in blocks if b["type"] == "text")


def _result(**kwargs):
    base = dict(
        state_mode="stateful",
        session_id=0,
        session_created=False,
        duration_ms=12,
    )
    return ExecutionResult(**{**base, **kwargs})


def test_off_the_envelope_is_as_shipped():
    r = _result(stdout=[TextPart(text="hi\n")], result=[1, 2])
    assert r.to_llm_content() == [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "result": [1, 2],
                    "state_mode": "stateful",
                    "session_id": 0,
                    "session_created": False,
                    "duration_ms": 12,
                },
                indent=2,
            ),
        },
        {"type": "text", "text": "\n--- stdout ---\n"},
        {"type": "text", "text": "hi\n"},
    ]


def test_stdout_then_stderr_then_the_value_then_the_traceback(plain):
    r = _result(
        stdout=[TextPart(text="first\nsecond")],
        stderr=[TextPart(text="careful\n")],
        result={"total": 3.5, "name": "x"},
        error="Traceback (most recent call last):\nValueError: bad\n",
    )
    assert r.to_llm_content() == [
        {
            "type": "text",
            "text": "first\nsecond\n[stderr]\ncareful\n"
            "Out: {'total': 3.5, 'name': 'x'}\n"
            "Traceback (most recent call last):\nValueError: bad",
        },
    ]


def test_the_value_is_its_repr_and_none_is_not_shown(plain):
    assert _text(_result(result="abc").to_llm_content()) == "Out: 'abc'"
    assert _text(_result(result=0).to_llm_content()) == "Out: 0"
    assert _text(_result(stdout=[TextPart(text="x\n")]).to_llm_content()) == "x"
    assert _result().to_llm_content() == [{"type": "text", "text": "(no output)"}]

    class Broken:
        def __repr__(self):
            raise RuntimeError("no")

    assert "repr failed" in _text(_result(result=Broken()).to_llm_content())


def test_no_session_or_timing_metadata(plain):
    text = _text(_result(stdout=[TextPart(text="ok")], result=1).to_llm_content())
    for word in ("state_mode", "session_id", "session_created", "duration_ms", "{"):
        assert word not in text


def test_images_keep_their_place(plain):
    r = _result(
        stdout=[
            TextPart(text="before\n"),
            ImagePart(data=PNG),
            TextPart(text="after\n"),
        ],
        result=2,
    )
    blocks = r.to_llm_content()
    assert [b["type"] for b in blocks] == ["text", "image_url", "text"]
    assert blocks[0]["text"] == "before\n"
    assert blocks[2]["text"] == "after\nOut: 2"


def test_a_note_comes_first(plain):
    r = _result(note="access_token received a placeholder", result=1)
    assert (
        _text(r.to_llm_content())
        == "[note] access_token received a placeholder\nOut: 1"
    )


def test_steering_counters_are_left_out_and_what_steered_is_shown(plain):
    counters = {"steps": 3, "retries": 0, "replayed": 0, "executed": 1}
    failed = _result(error="Traceback\nKeyError: 'a'\n", steering=counters)
    assert _text(failed.to_llm_content()) == "Traceback\nKeyError: 'a'"
    steered = {**counters, "interjections_received": ["use the other file"]}
    text = _text(_result(result=1, steering=steered).to_llm_content())
    assert text.startswith("Out: 1\n[steering] ")
    assert json.loads(text.split("[steering] ", 1)[1]) == steered


@pytest.mark.parametrize(
    "result",
    ["<steerable handle — now in-flight>", "[h0: steerable]", {"a": "[h1: steerable]"}],
)
def test_a_handle_result_keeps_the_envelope(plain, monkeypatch, result):
    on = _result(result=result).to_llm_content()
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", False)
    assert on == _result(result=result).to_llm_content()


def test_the_loop_finds_and_labels_a_handle_as_shipped(plain, monkeypatch):
    """The cleaned result the loop shows while it steers the handle is the
    shipped envelope, with the label in place of the handle."""
    from unify.actor.simulated import _StaticAnswerHandle

    raw = _result(
        stdout=[TextPart(text="started\n")],
        result={"h": _StaticAnswerHandle("x")},
    )
    handles, cleaned = _extract_nested_handle(raw)
    assert [label for _, label in handles] == ["h0"]
    on = cleaned.to_llm_content()
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", False)
    assert on == cleaned.to_llm_content()
    assert "[h0: steerable]" in _text(on)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_handle_as_the_last_expression_is_adopted_as_shipped(monkeypatch):
    """A cell's last expression is a sub-actor's handle: with the switch on the
    loop adopts it as shipped, and once the handle finishes its answer is
    the cell's ``Out:`` where the envelope's ``result`` held it."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.environments.actor import ActorEnvironment
    from unify.actor.simulated import _StaticAnswerHandle

    calls: list = []

    async def act(request, **kwargs):
        calls.append(request)
        return _StaticAnswerHandle(f"done: {request}")

    patch_actor_act(monkeypatch, act)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)

    async def run():
        actor = CodeActActor(environments=[ActorEnvironment()], can_store=False)
        replies = (
            lambda: h.completion(
                calls=[
                    (
                        "execute_code",
                        {
                            "thought": "Hand it on.",
                            "code": "await primitives.actor.act(request='summarise')",
                        },
                    ),
                ],
                call_ids=["call_h"],
            ),
            lambda: h.completion(content="finished"),
        )
        try:
            with h.scripted(replies) as provider:
                handle = await actor.act("Summarise.", persist=False)
                result = await asyncio.wait_for(handle.result(), 90)
        finally:
            await actor.close()
        tool = [
            m["content"]
            for m in provider.requests[-1]["messages"]
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_h"
        ]
        return result, tool

    off_result, off_tool = await run()
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", True)
    on_result, on_tool = await run()
    assert calls == ["summarise", "summarise"]
    assert on_result == off_result == "finished"
    (off_content,) = off_tool
    (on_content,) = on_tool
    shipped = json.loads(_text(off_content).split("\n--- stdout ---\n")[0])
    assert shipped["result"] == "done: summarise"
    assert on_content == [{"type": "text", "text": "Out: 'done: summarise'"}]
