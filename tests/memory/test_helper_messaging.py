"""The pass's record: delegate returns at once, helpers and the writer message each other (any agent to any other),
stop and finish end helpers and keep what they staged, and every agent gets the record's tools. Keyless: every
model is a scripted fake; nothing here opens a socket or a network connection.
"""

import asyncio
import inspect
import json
import time

import pytest

from unify.memory import agent_profiles, helpers, messaging, tool_spec
from unify.memory.writer_pass import writer_tools
from tests.memory.test_helpers import EPS, _ctx, _msg, _tc, fake_box  # noqa: F401 (fixture)
from tests.memory.test_writer_pass import needs_bwrap

RECORD_TOOLS = {"send", "messages", "wait", "who"}


class Steps:
    """A fresh scripted helper model per helper, chosen by the task line of its first message. Each step is a
    message, or a callable taking the messages so far and returning a message (or an awaitable of one)."""

    def __init__(self, steps):
        self.steps = steps
        self.calls: list[tuple[str, int]] = []
        self.seen: dict[str, list] = {}
        self.in_flight = self.max_in_flight = 0

    def factory(self, session=None):
        state = {"i": 0}

        async def turn(messages, tools):
            key = messages[1]["content"].split("\n")[1]
            i = state["i"]
            state["i"] += 1
            self.calls.append((key, i))
            self.seen[key] = messages
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                step = self.steps[key][i]
                out = step(messages) if callable(step) else step
                if inspect.isawaitable(out):
                    out = await out
            finally:
                self.in_flight -= 1
            return out, "0.001"

        return turn

    def outputs(self, key):
        return {
            m["tool_call_id"]: m["content"]
            for m in self.seen.get(key, [])
            if m.get("role") == "tool"
        }


def _done(cid="d", summary="s"):
    return _msg(_tc(cid, "done", {"summary": summary, "claims": []}))


def _after(seconds, msg, flag=None):
    async def step(messages):
        if flag is not None:
            flag.set()
        await asyncio.sleep(seconds)
        return msg

    return step


def _never(flag=None):
    async def step(messages):
        if flag is not None:
            flag.set()
        await asyncio.sleep(3600)
        raise AssertionError("a helper's call ran to the end of the test")

    return step


def _status(ctx, call="call-0"):
    return json.loads((ctx.out_root / call / "helper.json").read_text())


def test_delegate_returns_at_once_while_the_helper_is_still_running(tmp_path, fake_box):
    entered = asyncio.Event()
    model = Steps({"t": [_after(0.3, _done(), entered)]})
    ctx = _ctx(tmp_path, model)

    async def go():
        started = time.monotonic()
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        assert time.monotonic() - started < 0.25
        assert out.startswith(
            "helper call-0 started; its messages arrive in the record: use wait or messages",
        ), out
        assert ctx.pass_record.record.participants["call-0"].state == "running"
        await asyncio.wait_for(entered.wait(), 5)
        assert not (ctx.out_root / "call-0" / "helper.json").exists()
        reply = await helpers.result(ctx, "call-0")
        assert reply.startswith("helper call-0: ok")
        assert await messaging.close(ctx.pass_record, "the writer finished") == {"call-0": "ended"}

    asyncio.run(go())


def test_a_helpers_send_wakes_the_writers_wait(tmp_path, fake_box):
    found = "e1 computes w * h; source e1/cell:0"
    model = Steps(
        {"t": [_after(0.1, _msg(_tc("s", "send", {"text": found}))), _done(summary="sent it")]},
    )
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        got = await asyncio.wait_for(messaging.wait(pr, "writer"), 5)
        assert f"call-0: {found}" in got
        assert model.outputs("t")["s"].startswith("sent to writer")
        while "(done)" not in got:
            got += await asyncio.wait_for(messaging.wait(pr, "writer"), 5)
        assert "call-0 (done): helper call-0: ok" in got and "summary: sent it" in got
        # nothing more can arrive: wait returns at once, and messages never blocks
        started = time.monotonic()
        assert await messaging.wait(pr, "writer") == messaging.NOTHING
        assert messaging.messages(pr, "writer").startswith(messaging.NO_NEW)
        assert time.monotonic() - started < 0.5
        await messaging.close(pr, "the writer finished")

    asyncio.run(go())


def test_ask_returns_the_writers_reply(tmp_path, fake_box):
    model = Steps(
        {
            "t": [
                _msg(_tc("q", "ask", {"question": "which episode shows the area?"})),
                lambda messages: _done(summary=messages[-1]["content"][:200]),
            ],
        },
    )
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        question = await asyncio.wait_for(messaging.wait(pr, "writer"), 5)
        assert "call-0: which episode shows the area?" in question
        out = await messaging.call(pr, "writer", "reply", {"agent": "call-0", "text": "e1/cell:0 shows it"})
        assert out.startswith("sent to call-0")
        await asyncio.wait_for(helpers.result(ctx, "call-0"), 5)
        await messaging.close(pr, "the writer finished")

    asyncio.run(go())
    answer = model.outputs("t")["q"]
    assert "writer: @call-0 e1/cell:0 shows it" in answer
    assert _status(ctx)["summary"].startswith("#")


def test_stop_cancels_at_once_and_keeps_what_the_helper_staged(tmp_path, fake_box):
    busy = asyncio.Event()
    model = Steps(
        {
            "t": [
                _msg(_tc("s", "stage", {"path": "notes.md", "text": "a lesson"})),
                _never(busy),
            ],
        },
    )
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await asyncio.wait_for(busy.wait(), 5)
        started = time.monotonic()
        info = {}
        out = await messaging.call(pr, "writer", "stop", {"helper": "call-0"}, info)
        assert time.monotonic() - started < 5
        assert out.startswith("helper call-0 stopped"), out
        assert info == {"helper": "call-0", "in_flight": True, "ended": True}
        assert pr.record.participants["call-0"].state == "stopped"
        again = await messaging.call(pr, "writer", "stop", {"helper": "call-0"})
        assert tool_spec.form_of(again) == "not_running"
        unknown = await messaging.call(pr, "writer", "stop", {"helper": "call-9"})
        assert tool_spec.form_of(unknown) == "unknown_helper"
        assert await messaging.close(pr, "the writer finished") == {"call-0": "ended"}

    asyncio.run(go())
    status = _status(ctx)
    assert status["status"] == "stopped: the writer stopped it" and status["turns"] == 1
    assert status["stopped_by"] == "stop" and status["calls_cut"] == 1
    assert (ctx.out_root / "call-0" / "notes.md").read_text() == "a lesson"
    assert (ctx.inputs / "helpers" / "call-0" / "notes.md").read_text() == "a lesson"
    assert "may be lost" in messaging.STOP_SPEC.description


def test_finishing_lets_a_call_in_flight_finish_and_starts_no_new_one(tmp_path, fake_box):
    flags = {"A": asyncio.Event(), "B": asyncio.Event()}

    def steps(key):
        return [
            _msg(_tc(f"{key}s", "stage", {"path": "notes.md", "text": f"lesson {key}"})),
            _after(0.3, _msg(_tc(f"{key}o", "overview", {"episode": "e1"})), flags[key]),
            _never(),  # never reached: no new call starts once the writer finished
        ]

    model = Steps({"A": steps("A"), "B": steps("B")})
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "A", "episodes": ["e1"]}, ctx)
        await helpers.handle({"task": "B", "episodes": ["e1"]}, ctx)
        await asyncio.wait_for(asyncio.gather(flags["A"].wait(), flags["B"].wait()), 5)
        closed = await messaging.close(pr, "the writer finished")
        assert closed == {"call-0": "ended", "call-1": "ended"}

    asyncio.run(go())
    assert sorted(model.calls) == [("A", 0), ("A", 1), ("B", 0), ("B", 1)]
    for call, key in (("call-0", "A"), ("call-1", "B")):
        status = _status(ctx, call)
        # the call in flight finished and was recorded as a turn; its tool calls were answered
        assert status["status"] == "stopped: the writer finished" and status["turns"] == 2
        assert status["stopped_by"] == "finish" and status["calls_cut"] == 0
        assert model.outputs(key)[f"{key}o"].strip()
        assert (ctx.out_root / call / "notes.md").read_text() == f"lesson {key}"
    # each helper was told on the record why it stopped
    texts = [e.text for e in pr.record.entries if e.kind == "cancel"]
    assert texts == ["@call-0 stop: the writer finished", "@call-1 stop: the writer finished"]


def test_finishing_cancels_what_is_still_running_at_the_wall_clock_and_checks_it_ended(
    tmp_path,
    fake_box,
):
    busy = asyncio.Event()
    model = Steps({"t": [_never(busy)]})
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await asyncio.wait_for(busy.wait(), 5)
        pr.deadline = time.monotonic() + 0.2  # the outer backstop: the pass's wall clock
        assert await messaging.close(pr, "the writer finished") == {"call-0": "ended"}

    asyncio.run(go())
    status = _status(ctx)
    assert status["status"] == "stopped: the pass's wall clock was reached"
    assert status["stopped_by"] == "wall_clock" and status["calls_cut"] == 1
    assert (ctx.inputs / "helpers" / "call-0" / "helper.json").is_file()


def test_two_helpers_model_calls_are_in_flight_at_the_same_time(tmp_path, fake_box):
    model = Steps({"A": [_after(0.2, _done())], "B": [_after(0.2, _done())]})
    ctx = _ctx(tmp_path, model)

    async def go():
        await helpers.handle({"task": "A", "episodes": ["e1"]}, ctx)
        await helpers.handle({"task": "B", "episodes": ["e2"]}, ctx)
        await asyncio.gather(helpers.result(ctx, "call-0"), helpers.result(ctx, "call-1"))
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    assert model.max_in_flight == 2


def test_wait_with_nothing_running_and_nothing_unseen_returns_at_once():
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    started = time.monotonic()
    assert asyncio.run(messaging.wait(pr, "writer")) == messaging.NOTHING
    assert time.monotonic() - started < 0.5


def test_a_helper_mentions_another_helper_and_its_wait_returns_it(tmp_path, fake_box):
    model = Steps(
        {
            "A": [
                _after(0.1, _msg(_tc("s", "send", {"text": "@call-1 e2/cell:0 prints two"}))),
                _done(),
            ],
            "B": [_msg(_tc("w", "wait", {})), _done()],
        },
    )
    ctx = _ctx(tmp_path, model)

    async def go():
        await helpers.handle({"task": "A", "episodes": ["e1"]}, ctx)
        await helpers.handle({"task": "B", "episodes": ["e2"]}, ctx)
        await asyncio.wait_for(
            asyncio.gather(helpers.result(ctx, "call-0"), helpers.result(ctx, "call-1")),
            5,
        )
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    assert model.outputs("A")["s"].startswith("sent to call-1")
    assert "call-0: @call-1 e2/cell:0 prints two" in model.outputs("B")["w"]
    entry = next(e for e in ctx.pass_record.record.entries if e.author == "call-0" and e.kind == "post")
    assert entry.mentions == ("call-1",)


def test_the_roster_lists_every_agent_and_follows_spawns_and_ends(tmp_path, fake_box):
    long_task = "read e1 and draft the area function " * 10
    model = Steps({long_task.split("\n")[0]: [_after(0.1, _done())]})
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        before = messaging.who(pr)
        assert "- writer (writer, running): keeps the library" in before and "call-0" not in before
        await helpers.handle({"task": long_task, "episodes": ["e1"]}, ctx)
        during = messaging.who(pr)
        line = next(ln for ln in during.splitlines() if ln.startswith("- call-0 "))
        assert line.startswith("- call-0 (helper, running): read e1 and draft the area function")
        assert len(line.split(": ", 1)[1]) <= messaging.HELP_CHARS + 1
        await helpers.result(ctx, "call-0")
        after = messaging.who(pr)
        assert "- call-0 (helper, replied): read e1" in after
        assert "@writer" in after and "@helpers" in after
        await messaging.close(pr, "the writer finished")

    asyncio.run(go())


def test_role_names_resolve_and_an_unknown_name_is_refused():
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    for name in ("call-0", "call-1"):
        pr.add_helper(name, "a task")
        pr.record.add_agent(name, spawner="writer")
    out, to = messaging.post(pr, "call-0", "@writer done reading e1", messaging.SEND_SPEC)
    assert to == ["writer"] and out.startswith("sent to writer")
    out, to = messaging.post(pr, "writer", "@helpers stop drafting", messaging.SEND_SPEC)
    assert to == ["call-0", "call-1"]
    out, to = messaging.post(pr, "call-0", "@helpers e1 is done", messaging.SEND_SPEC)
    assert to == ["call-1"]  # never to itself
    out, to = messaging.post(pr, "call-1", "no address", messaging.SEND_SPEC)
    assert to == ["writer"]  # a helper's message that names no one goes to the writer
    refused, to = messaging.post(pr, "call-0", "@nobody look", messaging.SEND_SPEC)
    assert to == [] and tool_spec.form_of(refused) == "unknown_name"
    assert "@nobody" in refused and "@call-1" in refused and "@helpers" in refused
    refused, _ = messaging.post(pr, "writer", "to whom?", messaging.SEND_SPEC)
    assert tool_spec.form_of(refused) == "no_recipient"
    # nothing refused reached the record
    assert not any("nobody" in e.text or "whom" in e.text for e in pr.record.entries)
    # in a curate pass @writer and @curator both name the curator
    cur = messaging.PassRecord(root="curator", deadline=time.monotonic() + 600)
    cur.add_helper("call-0", "a task")
    cur.record.add_agent("call-0", spawner="curator")
    for alias in ("@writer", "@curator"):
        _, to = messaging.post(cur, "call-0", f"{alias} hello", messaging.SEND_SPEC)
        assert to == ["curator"]


def test_the_curators_delegate_wait_and_messages_work_the_same_way(tmp_path, fake_box):
    model = Steps({"t": [_msg(_tc("s", "send", {"text": "@curator e1 overlaps e2"})), _done()]})
    pr = messaging.PassRecord(root="curator", deadline=time.monotonic() + 600)
    ctx = _ctx(tmp_path, model, pass_record=pr)

    async def go():
        out = await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        assert out.startswith("helper call-0 started")
        got = await asyncio.wait_for(messaging.wait(pr, "curator"), 5)
        assert "call-0: @curator e1 overlaps e2" in got
        await helpers.result(ctx, "call-0")
        # the helper's end arrives on the record too, whether the wait above already took it or not
        got += messaging.messages(pr, "curator")
        assert "call-0 (done)" in got
        await messaging.close(pr, "the curator finished")

    asyncio.run(go())
    # the helper was told who runs its pass
    assert "the agent that runs this pass: curator" in model.seen["t"][1]["content"]


def test_every_agent_gets_the_record_tools_from_the_registry_and_its_profile():
    specs = {s.name: s for s in tool_spec.registry()}
    for name in RECORD_TOOLS:
        assert specs[name].roles == frozenset({"write", "curate", "helper"}) and specs[name].requires is None
    for name in ("reply", "stop"):
        assert specs[name].roles == frozenset({"write", "curate"})
    assert specs["ask"].roles == frozenset({"helper"})
    for agent, flags in (("writer", frozenset({"helper"})), ("curator", frozenset()), ("helper", frozenset())):
        row = agent_profiles.profile(agent)
        assert RECORD_TOOLS <= set(row["tools"]), agent
        assert RECORD_TOOLS <= set(agent_profiles.tool_names(agent, flags)), agent
        assert any(c["artifact"] == "record" for c in row["consumes"]), agent
        assert any(p["artifact"] == "record" for p in row["produces"]), agent
    names = lambda tools: {t["function"]["name"] for t in tools}  # noqa: E731
    assert RECORD_TOOLS | {"reply", "stop"} <= names(writer_tools(library_mode=True))
    assert RECORD_TOOLS | {"reply", "stop"} <= names(writer_tools(library_mode=True, role="curate"))
    assert RECORD_TOOLS | {"ask"} <= names(helpers.helper_tools())
    assert not {"reply", "stop", "delegate"} & names(helpers.helper_tools())
    # each schema round-trips as JSON and its refusals render
    for spec in messaging.SPECS:
        tool = spec.as_tool()
        assert json.loads(json.dumps(tool)) == tool and tool["function"]["parameters"]["type"] == "object"
        for form in spec.refusals:
            fields = {k: "x" for k in ("names", "valid", "why", "name", "state")}
            assert tool_spec.form_of(tool_spec.refusal(spec.name, form, **fields)) == form
    # no time allowance is left on delegate
    assert "seconds" not in helpers.SPEC.parameters["properties"]
    assert "finishing window" not in helpers.SPEC.description


def test_the_briefs_state_the_record_as_facts():
    from unify.memory import library_prompts as lp

    for brief in (lp.write_brief_now(helper=True), lp.write_brief_now(), lp.curate_brief_now()):
        assert messaging.RECORD_TEXT in brief
    for text in (messaging.RECORD_TEXT, helpers.helper_brief(True), helpers.helper_brief(False)):
        assert "`who`" in text and "@writer" in text and "@helpers" in text
        assert "<episode>/<part>" in text and "untrusted input" in text
    brief = helpers.helper_brief(False)
    for fact in (
        "The library under /memory may be empty at the start of a run.",
        "A part you already read in this task is the same text if you read it again: re-reading is allowed and "
        "costs time.",
        "What other agents found arrives as messages (`messages` / `wait`).",
    ):
        assert fact in brief, fact
    with_helpers = lp.write_brief_now(helper=True)
    assert "`delegate`. It returns at once" in with_helpers and "`stop` ends a helper" in with_helpers
    assert "finishing window" not in with_helpers and "last quarter" not in with_helpers


def test_a_helper_reads_the_writers_inputs_read_only(tmp_path, fake_box):
    model = Steps(
        {
            "t": [
                _msg(
                    _tc("st", "read", {"path": "/inputs/staging/e1/candidates/f.py"}),
                    _tc("pool", "read", {"path": "/inputs/pool/INDEX.md"}),
                    _tc("task", "read", {"path": "/task/episodes/e1.json"}),
                    _tc("x", "execute_code", {"code": "print(1)"}),
                ),
                _done(),
            ],
        },
    )
    ctx = _ctx(tmp_path, model)
    staged = ctx.inputs / "staging" / "e1" / "candidates"
    staged.mkdir(parents=True)
    (staged / "f.py").write_text("def f():\n    return 1\n")

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-0")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")
    assert "def f()" in out["st"] and "pool" in out["pool"] and not out["pool"].startswith("refused")
    assert '"e1"' in out["task"] or "e1" in out["task"]
    (call,) = fake_box
    # the writer's own /inputs, read-only; nothing writable but /work
    assert call["ro"][ctx.inputs] == "/inputs" and list(call["rw"].values()) == ["/work"]
    assert sorted(call["ro"].values()) == ["/cell", "/inputs", "/memory", "/task"]


@needs_bwrap
def test_in_the_real_box_the_writers_inputs_read_and_refuse_writes(tmp_path):
    code = (
        "import pathlib\n"
        "print(open('/inputs/pool/INDEX.md').read().strip())\n"
        "for p in ('/inputs/pool/x.txt', '/inputs/staging.txt', '/task/y.txt'):\n"
        "    try:\n"
        "        pathlib.Path(p).write_text('no')\n"
        "        print('WROTE', p)\n"
        "    except OSError:\n"
        "        print('DENIED', p)\n"
    )
    model = Steps({"t": [_msg(_tc("x", "execute_code", {"code": code})), _done()]})
    ctx = _ctx(tmp_path, model)

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-0")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")["x"]
    assert out.splitlines()[0] == "pool" and "WROTE" not in out
    assert out.count("DENIED") == 3
    assert not (ctx.inputs / "pool" / "x.txt").exists() and not (ctx.inputs / "staging.txt").exists()


@pytest.mark.parametrize("tool", ["send", "reply"])
def test_an_empty_message_is_refused(tool):
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600)
    args = {"text": " "} if tool == "send" else {"agent": "writer", "text": " "}
    out = asyncio.run(messaging.call(pr, "writer", tool, args))
    assert tool_spec.form_of(out) == "no_text"


def test_every_record_tool_call_of_a_helper_is_one_event_row_with_its_messaging_fields(tmp_path, fake_box):
    model = Steps(
        {
            "t": [
                _msg(_tc("who", "who", {})),
                _msg(_tc("s", "send", {"text": "e1/cell:0 holds the area"})),
                _msg(_tc("m", "messages", {})),
                _msg(_tc("q", "ask", {"question": "keep going?"})),
                _done(),
            ],
        },
    )
    ctx = _ctx(tmp_path, model)
    pr = ctx.pass_record

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        while True:
            got = await asyncio.wait_for(messaging.wait(pr, "writer"), 5)
            if "keep going?" in got:
                break
        await messaging.call(pr, "writer", "reply", {"agent": "call-0", "text": "yes"})
        await helpers.result(ctx, "call-0")
        await messaging.close(pr, "the writer finished")

    asyncio.run(go())
    rows = {r["tool"]: r for r in map(json.loads, ctx.events.read_text().splitlines())}
    assert all(r["name"] == r["helper_call"] == "call-0" for r in rows.values())
    who = rows["who"]["messaging"]["roster"]
    assert {"name": "writer", "role": "writer", "state": "running"} == {
        k: who[0][k] for k in ("name", "role", "state")
    }
    assert who[1]["name"] == "call-0" and who[1]["role"] == "helper" and who[1]["help"] == "t"
    send = rows["send"]["messaging"]
    assert send["to"] == ["writer"] and pr.record.entries[send["seq"] - 1].text == "e1/cell:0 holds the area"
    assert rows["messages"]["messaging"] == {"entries": 0, "from": []}
    ask = rows["ask"]["messaging"]
    assert ask["outcome"] == "message" and ask["answered"] is True and ask["seconds"] >= 0
    assert ask["to"] == ["writer"] and isinstance(ask["seq"], int)
    assert "messaging" not in rows["done"]
    status = _status(ctx)
    assert status["stopped_by"] == "done" and status["calls_cut"] == 0
    # one identity: the record name, the event name and the output folder
    assert "call-0" in pr.record.participants and (ctx.out_root / "call-0").is_dir()


def _readable_inputs(ctx):
    """The writer's /inputs as a pass stages it: its batch map, a batch episode record and a shared block."""
    (ctx.inputs / "batch_map.json").write_text(
        json.dumps({"episodes": [{"episode_id": "e1", "observations": 1, "cells": 1, "actions": 0}]}),
    )
    (ctx.inputs / "episodes").mkdir()
    (ctx.inputs / "episodes" / "e1.json").write_text('{"episode": "e1"}')
    (ctx.inputs / "shared").mkdir()
    (ctx.inputs / "shared" / "a1b2c3d4e5f6.txt").write_text("the shared rules")


def test_the_recorded_helper_calls_that_were_refused_now_work(tmp_path, fake_box):
    """Calls reconstructed from recorded runs, each refused before: every valid intent is now answered."""
    calls = [
        _tc("bm", "read", {"path": "/inputs/batch_map.json"}),
        _tc("ep", "read", {"path": "/inputs/episodes/e1.json"}),
        _tc("sh", "read", {"path": "/inputs/shared/a1b2c3d4e5f6.txt"}),
        _tc("ov", "overview", {"episode": "", "offset": 0}),
        _tc("toc", "read_episode", {"episode": "e1", "parts": [], "part": "", "offset": 0}),
        _tc("one", "read_episode", {"episode": "e1", "parts": [{"part": "cell:0", "offset": 0}]}),
        _tc("mix", "read_episode", {"episode": "e1", "parts": ["cell:0", "cell:7"]}),
        _tc("shp", "read_episode", {"episode": "e1", "parts": ["shared:b1"]}),
    ]
    model = Steps({"t": [_msg(*calls), _done()]})
    ctx = _ctx(tmp_path, model, shared={"b1": "shared block"})
    _readable_inputs(ctx)

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-0")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")
    rows = [json.loads(x) for x in ctx.events.read_text().splitlines()]
    assert [r["tool"] for r in rows if r["status"] != "ok"] == []
    assert '"episode_id": "e1"' in out["bm"] and "the shared rules" in out["sh"]
    assert out["ov"].startswith("e1: 1 observation(s), 1 cell(s)")
    assert "cell:0" in out["toc"] and "request" in out["toc"]
    assert "== cell:0 ==" in out["one"]
    assert "== cell:0 ==" in out["mix"] and "unknown part(s) in e1: 'cell:7'" in out["mix"]
    assert "shared block" in out["shp"]
    assert helpers.episode_paths("read_episode", {"parts": [], "part": ""}, "e1") == [
        {"path": "/episodes/e1", "mode": "read"},
    ]


def test_a_missing_path_lists_its_directory_and_the_box_stays_closed(tmp_path, fake_box):
    calls = [
        _tc("noext", "read", {"path": "/inputs/shared/a1b2c3d4e5f6"}),
        _tc("out", "read", {"path": "/outputs"}),
        _tc("up", "read", {"path": "/inputs/../../etc/passwd"}),
        _tc("bad", "read_episode", {"episode": "e1", "parts": ["cell:9"]}),
    ]
    model = Steps({"t": [_msg(*calls), _done()]})
    ctx = _ctx(tmp_path, model)
    _readable_inputs(ctx)

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-0")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")
    forms = {r["call"]: (r["refusal"] or {}).get("form") for r in map(json.loads, ctx.events.read_text().splitlines())}
    assert forms[0] == "not_found" and "/inputs/shared holds: a1b2c3d4e5f6.txt" in out["noext"]
    assert forms[1] == forms[2] == "outside_box"
    assert out["out"] == "refused: /outputs is outside /inputs, /task, /memory and /work"
    assert forms[3] == "unknown_part" and "cell:0" in out["bad"]
    # the helper's own read and grep never name /outputs
    tools = {t["function"]["name"]: t["function"]["description"] for t in helpers.helper_tools()}
    assert "/outputs" not in tools["read"] and "/outputs" not in tools["grep"]
    assert "/task" in tools["read"]


def test_in_a_curate_pass_without_helpers_reply_and_stop_are_refused_with_stable_codes():
    pr = messaging.PassRecord(root="curator", deadline=time.monotonic() + 600)
    reply = asyncio.run(messaging.call(pr, "curator", "reply", {"agent": "call-0", "text": "hello"}))
    stop = asyncio.run(messaging.call(pr, "curator", "stop", {"helper": "call-0"}))
    send = asyncio.run(messaging.call(pr, "curator", "send", {"text": "@helpers hello"}))
    assert tool_spec.form_of(reply) == "unknown_name" and tool_spec.form_of(stop) == "unknown_helper"
    assert tool_spec.form_of(send) == "no_recipient"  # @helpers names no running helper
    codes = tool_spec.refusal_codes()
    assert "unknown_name" in codes["reply"] and "unknown_helper" in codes["stop"]
    assert "no_recipient" in codes["send"] and {"unknown_name", "not_posted"} <= set(codes["ask"])
    assert pr.record.entries == []  # nothing refused reached the record
    # the reply's seq is recorded as send's is
    pr.add_helper("call-0", "a task")
    pr.record.add_agent("call-0", spawner="curator")
    info = {}
    asyncio.run(messaging.call(pr, "curator", "reply", {"agent": "call-0", "text": "hello"}, info))
    assert info == {"to": ["call-0"], "seq": 1}


@needs_bwrap
def test_a_helper_cell_imports_the_library_as_the_writer_sees_it(tmp_path):
    code = "import memory, os\nprint(memory.__file__)\nprint(os.getcwd())\n"
    model = Steps({"t": [_msg(_tc("x", "execute_code", {"code": code})), _done()]})
    ctx = _ctx(tmp_path, model)
    (ctx.library / "memory").mkdir()
    (ctx.library / "memory" / "__init__.py").write_text("ITEMS = []\n")

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-0")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")["x"].splitlines()
    assert out[:2] == ["/memory/memory/__init__.py", "/work"], out


def test_a_helper_turn_records_its_message_keys_and_reasoning_summary():
    from unify.memory.writer_pass import transcript_lines

    turn = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
        "provider_specific_fields": {"reasoning": "the area is w * h, so read cell 0"},
    }
    messages = [{"role": "system", "content": "brief"}, {"role": "user", "content": "task"}, turn]
    label = {"agent": "helper", "helper": "call-0"}
    lines = [json.loads(x) for x in transcript_lines(messages, "p1", label=label)]
    (line,) = [x for x in lines if x.get("i") == 2]
    assert line["helper"] == "call-0" and line["pass_id"] == "p1"
    assert line["message_keys"] == ["content", "provider_specific_fields", "role", "tool_calls"]
    assert line["reasoning_summary"] == "the area is w * h, so read cell 0"
    plain = [json.loads(x) for x in transcript_lines(messages, "p1")]
    assert all("message_keys" not in x and "reasoning_summary" not in x for x in plain)
