"""``wait`` and ``ask`` return at once when no message can arrive: every other running agent on the record is itself
blocked in ``wait`` or ``ask`` with nothing unread. The last agent to start waiting releases every waiter; a waiter is
never released while another agent runs and is not waiting. Keyless: every model is a scripted fake; the one pass in a
real box needs bubblewrap and prlimit, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from unify.memory import helpers, messaging
from unify.memory.trigger import PassRequest
from tests.memory.test_helper_messaging import Steps, _done
from tests.memory.test_helpers import _ctx, _msg, _tc, fake_box  # noqa: F401 (fixture)
from tests.memory.test_tool_channel_box import HELPER_CELL, _cell, _descendants, _rows, _writer, needs_box

#: what ``wait`` answers when every other running agent is waiting too
ALL_WAITING = f"{messaging.NO_NEW}; {messaging.ALL_WAITING}"


def _record(*names: str) -> messaging.PassRecord:
    """A writer's record with the helpers *names* running (started on the record, no model behind them)."""
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    for name in names:
        pr.add_helper(name, "a task")
        pr.record.add_agent(name, spawner="writer")
    return pr


def test_two_agents_that_both_wait_both_return_at_once():
    async def go():
        pr = _record("call-0")
        infos: dict[str, dict] = {"writer": {}, "call-0": {}}
        started = time.monotonic()
        writer = asyncio.ensure_future(messaging.wait(pr, "writer", infos["writer"]))
        await asyncio.sleep(0.1)
        assert not writer.done()  # the helper is running and not waiting: a message can still come from it
        helper = await asyncio.wait_for(messaging.wait(pr, "call-0", infos["call-0"]), 5)
        assert helper == ALL_WAITING
        assert await asyncio.wait_for(writer, 5) == ALL_WAITING
        assert time.monotonic() - started < 2
        assert {i["outcome"] for i in infos.values()} == {"nothing_can_arrive"}
        assert pr.waiting == {}

    asyncio.run(go())


def test_a_helper_whose_first_call_is_wait_releases_the_writers_wait(tmp_path, fake_box):
    model = Steps({"t": [_msg(_tc("w", "wait", {})), _done(summary="waited")]})
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        started = time.monotonic()
        assert await asyncio.wait_for(messaging.wait(pr, "writer"), 5) == ALL_WAITING
        assert time.monotonic() - started < 2
        reply = await asyncio.wait_for(helpers.result(ctx, "call-0"), 5)
        assert reply.startswith("helper call-0: ok")
        await messaging.close(pr, "the writer finished")

    asyncio.run(go())
    assert model.outputs("t")["w"] == ALL_WAITING


def test_ask_returns_without_a_reply_when_the_one_it_asked_waits_too():
    async def go():
        pr = _record("call-0")
        writer = asyncio.ensure_future(messaging.wait(pr, "writer"))
        await asyncio.sleep(0.05)
        info: dict = {}
        ask = asyncio.ensure_future(
            messaging.call(pr, "call-0", "ask", {"question": "which part shows the area?"}, info),
        )
        # the question is unread by the writer while the helper's ask starts waiting: the writer is not stuck, so
        # the ask keeps waiting and the writer's wait returns the question
        question = await asyncio.wait_for(writer, 5)
        assert "call-0: which part shows the area?" in question
        await asyncio.sleep(0.1)
        assert not ask.done()
        # the writer waits again without answering: now nothing can arrive for either
        started = time.monotonic()
        assert await asyncio.wait_for(messaging.wait(pr, "writer"), 5) == ALL_WAITING
        assert await asyncio.wait_for(ask, 5) == f"no reply yet ({ALL_WAITING})"
        assert time.monotonic() - started < 2
        assert info["outcome"] == "nothing_can_arrive" and info["answered"] is False
        assert pr.waiting == {}

    asyncio.run(go())


def test_a_waiter_is_not_released_while_another_agent_runs_and_a_real_message_wakes_it():
    async def go():
        pr = _record("call-0", "call-1")
        infos: dict[str, dict] = {"writer": {}, "call-0": {}}
        writer = asyncio.ensure_future(messaging.wait(pr, "writer", infos["writer"]))
        helper = asyncio.ensure_future(messaging.wait(pr, "call-0", infos["call-0"]))
        await asyncio.sleep(0.5)
        # call-1 is running and not waiting: it can still send, so neither wait ends
        assert not writer.done() and not helper.done()
        assert sorted(pr.waiting.values()) == ["call-0", "writer"]
        sent = messaging.post(pr, "call-1", "@call-0 e2/cell:0 prints two", messaging.SEND_SPEC)[0]
        assert sent.startswith("sent to call-0")
        got = await asyncio.wait_for(helper, 5)
        assert "call-1: @call-0 e2/cell:0 prints two" in got and infos["call-0"]["outcome"] == "message"
        await asyncio.sleep(0.2)
        assert not writer.done()  # call-1 still runs and is not waiting
        messaging.post(pr, "call-1", "found nothing more", messaging.SEND_SPEC)
        got = await asyncio.wait_for(writer, 5)
        assert "call-1: found nothing more" in got and infos["writer"]["outcome"] == "message"
        assert pr.waiting == {}

    asyncio.run(go())


def test_the_last_of_three_agents_to_wait_releases_them_all():
    async def go():
        pr = _record("call-0", "call-1")
        names = ("writer", "call-0", "call-1")
        infos: dict[str, dict] = {n: {} for n in names}
        first = [asyncio.ensure_future(messaging.wait(pr, n, infos[n])) for n in names[:2]]
        await asyncio.sleep(0.3)
        assert not any(t.done() for t in first)
        started = time.monotonic()
        last = await asyncio.wait_for(messaging.wait(pr, "call-1", infos["call-1"]), 5)
        outs = await asyncio.wait_for(asyncio.gather(*first), 5)
        assert time.monotonic() - started < 2
        assert [*outs, last] == [ALL_WAITING] * 3
        assert {i["outcome"] for i in infos.values()} == {"nothing_can_arrive"}
        assert pr.waiting == {}

    asyncio.run(go())


@pytest.mark.parametrize("similar", [False, True])
@pytest.mark.parametrize("in_code", [False, True])
def test_the_helper_brief_shows_sending_before_reading_replies(similar, in_code):
    brief = helpers.helper_brief(similar, in_code=in_code)
    role = brief[brief.index("You are a helper") :]
    if in_code:
        calls = ("tools.send(", "tools.ask(", "tools.messages()", "tools.wait()")
    else:
        calls = ("`send(", "`ask(", "`messages`", "`wait`")
    first = [role.index(c) for c in calls]
    assert first == sorted(first), dict(zip(calls, first))


WRITER_CELL = """
print("DELEGATE", tools.delegate("draft an area function", episodes=[]))
print("WAIT", tools.wait()[:300])
"""

REPLY_CELL = 'print("REPLY", tools.wait()[:300])'

HELPER_WAIT_CELL = 'print("HELPER WAIT", tools.wait())'


@needs_box
def test_a_writer_and_its_helper_that_both_wait_first_do_not_hold_the_pass_to_its_wall_clock(tmp_path):
    deadline = 300.0
    turns = [
        _cell("c1", WRITER_CELL),
        _cell("c2", REPLY_CELL),
        _cell("c3", "tools.finish('drafted nothing to keep')"),
    ]
    writer, model, _ = _writer(tmp_path, turns, edit_tools=True, deadline_s=deadline, max_calls=6)
    helper_outputs: dict[str, str] = {}

    def factory(label=None):
        cells = [_cell("h1", HELPER_WAIT_CELL), _cell("h2", HELPER_CELL)]

        async def turn(messages, tools):
            helper_outputs.update({m["tool_call_id"]: _text(m["content"]) for m in messages if m.get("role") == "tool"})
            return (cells.pop(0) if cells else _call_done()), "0.001"

        return turn

    writer.helper_turn = factory
    started = time.monotonic()
    asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    took = time.monotonic() - started
    rows = _rows(tmp_path)
    summary = [(r.get("agent"), r.get("tool"), r.get("status"), r.get("messaging")) for r in rows]
    assert took < deadline / 2, (took, summary)
    c1, c2 = _text(model.outputs["c1"]), _text(model.outputs["c2"])
    assert "DELEGATE helper call-0 started" in c1 and f"WAIT {ALL_WAITING}" in c1, c1
    assert f"HELPER WAIT {ALL_WAITING}" in helper_outputs["h1"], helper_outputs
    # after the release the pass goes on as usual: the helper's done reaches the writer's next wait
    assert "REPLY" in c2 and "call-0 (done)" in c2, c2
    waits = [r for r in rows if r["event"] == "tool_call" and r["tool"] == "wait"]
    assert [(r.get("messaging") or {}).get("outcome") for r in waits[:2]] == ["nothing_can_arrive"] * 2, summary
    assert not writer._in_code_threads.alive()
    assert _descendants(os.getpid()) == []


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _call_done() -> dict:
    return _cell("h-done", "tools.done('nothing more', [])")
