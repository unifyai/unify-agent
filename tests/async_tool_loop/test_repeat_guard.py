"""Symbolic: ``UNIFY_REPEAT_GUARD`` holds back a reply the requester already answered.

In persistent sessions (``unify act --persist --jsonl``, where each verdict
from the environment arrives as the next user message), the low-effort actor
re-sent final replies that had already been answered with a rejection: 73,
106 and 79 identical resubmissions in the three ARC LOW arms of 2 Oct, 40-69
of them immediate resubmits with zero reasoning tokens. With the switch, such
a reply is held back once and a note quoting the requester's answer is
appended at the tail; sending it again surfaces it. The transport is
scripted, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.common._async_tool import repeat_guard
from unify.settings import ProductionSettings, SETTINGS

REJECTION = "Incorrect: the output grid does not match. Try again."


async def _session(replies, follow_ups) -> tuple[list[str], list[dict]]:
    """A persistent loop: the first request, then one follow-up per turn."""
    from unify.common.async_tool_loop import start_async_tool_loop

    with h.scripted([(lambda r=r: h.completion(content=r)) for r in replies]) as p:
        handle = start_async_tool_loop(
            h.new_client(),
            "Solve the puzzle and reply with the answer.",
            {},
            log_steps=False,
            timeout=60,
            persist=True,
        )
        surfaced = [(await h._next_response(handle))["content"]]
        for message in follow_ups:
            await handle.interject(message)
            surfaced.append((await h._next_response(handle))["content"])
        await handle.stop()
        await asyncio.wait_for(handle.result(), 30)
    return surfaced, p.requests


def _notes(request: dict) -> list[str]:
    return [
        str(m.get("content"))
        for m in request["messages"]
        if m.get("role") == "user"
        and "is identical to your reply" in str(m.get("content"))
    ]


def _assert_append_only(requests: list[dict]) -> None:
    for earlier, later in zip(requests, requests[1:]):
        sent = json.dumps(earlier["messages"], default=str)
        prefix = later["messages"][: len(earlier["messages"])]
        assert json.dumps(prefix, default=str) == sent


@pytest.fixture
def guard_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPEAT_GUARD", True)


@pytest.mark.asyncio
async def test_a_repeat_of_an_answered_reply_gets_one_note_then_is_surfaced(guard_on):
    first = '{"grid": [[1, 2]], "id": "x"}'
    # Same JSON, other key order and spacing.
    repeat = '{"id": "x",   "grid": [[1,2]]}'
    surfaced, requests = await _session([first, repeat, repeat], [REJECTION])
    assert surfaced == [first, repeat]
    assert len(requests) == 3
    assert _notes(requests[1]) == []
    (note,) = _notes(requests[2])
    assert note == (
        "Your reply is identical to your reply in turn 1, after which the "
        f"requester said: '{REJECTION}'. If you still intend it, send it "
        "again; otherwise revise."
    )
    # The note is the last message: nothing already sent was edited.
    assert requests[2]["messages"][-1]["content"] == note
    _assert_append_only(requests)


@pytest.mark.asyncio
async def test_a_revised_reply_after_the_note_is_surfaced(guard_on):
    surfaced, requests = await _session(
        ["answer A", "answer   A", "answer B", "answer A"],
        [REJECTION, "Still incorrect."],
    )
    # Turn 2 repeats A, is held, revised to B; turn 3 repeats A, which was
    # already held back once, and is surfaced without a second note.
    assert surfaced == ["answer A", "answer B", "answer A"]
    assert len(requests) == 4
    assert [len(_notes(r)) for r in requests] == [0, 0, 1, 1]
    _assert_append_only(requests)


@pytest.mark.asyncio
async def test_a_different_reply_gets_no_note(guard_on):
    surfaced, requests = await _session(["answer A", "answer B"], [REJECTION])
    assert surfaced == ["answer A", "answer B"]
    assert len(requests) == 2
    assert not any(_notes(r) for r in requests)


@pytest.mark.asyncio
async def test_off_a_repeat_is_surfaced_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_REPEAT_GUARD", False)
    surfaced, requests = await _session(["answer A", "answer A"], [REJECTION])
    assert surfaced == ["answer A", "answer A"]
    assert len(requests) == 2
    assert not any(_notes(r) for r in requests)


@pytest.mark.asyncio
async def test_on_the_persist_scenario_with_distinct_replies_is_upstreams(guard_on):
    golden = json.loads(h.GOLDEN.read_text())["persist"]
    _result, _counter, requests = await h.scenario_persist()
    assert [h.request_bytes(r) for r in requests] == golden


@pytest.mark.asyncio
async def test_on_a_one_shot_run_is_upstreams(guard_on):
    golden = json.loads(h.GOLDEN.read_text())["interrupt"]
    _result, _counter, requests = await h.scenario_interrupt()
    assert [h.request_bytes(r) for r in requests] == golden


def test_normalisation():
    n = repeat_guard.normalise
    assert n("  a \n\t b ") == "a b"
    assert n('{"b": 1, "a": [1, 2]}') == n('{ "a":[1,2],"b":1 }')
    assert n("[1, 2]") == n("[1,2]")
    assert n("{not json   at all") == "{not json at all"
    assert n(None) == ""


def test_an_unanswered_or_empty_reply_is_never_held():
    guard = repeat_guard.RepeatGuard()
    guard.surfaced("x")
    assert guard.check("x") is None  # nothing answered it yet
    guard.requester_said("no")
    guard.surfaced("")
    guard.requester_said("say something")
    assert guard.check("") is None
    assert guard.check(" x ") is not None
    assert guard.check("x") is None  # held once already


def test_the_setting_defaults_off():
    assert ProductionSettings().UNIFY_REPEAT_GUARD is False
    assert ProductionSettings(UNIFY_REPEAT_GUARD="true").UNIFY_REPEAT_GUARD is True
