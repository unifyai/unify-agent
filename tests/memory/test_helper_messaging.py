"""The pass's record: delegate returns at once, helpers and the writer message each other (any agent to any other),
stop and finish end helpers and keep what they staged, and every agent gets the record's tools. Keyless: every
model is a scripted fake; nothing here opens a socket or a network connection.
"""

import asyncio
import inspect
import json
import time

import pytest

from unify.memory import helpers, messaging, tool_spec
from tests.memory.test_helpers import EPS, _ctx, _msg, _tc, helper_boxes  # noqa: F401 (fixture)
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


def _status(ctx, call="call-1"):
    return json.loads((ctx.out_root / call / "helper.json").read_text())


def test_two_helpers_model_calls_are_in_flight_at_the_same_time(tmp_path, helper_boxes):
    model = Steps({"A": [_after(0.2, _done())], "B": [_after(0.2, _done())]})
    ctx = _ctx(tmp_path, model)

    async def go():
        await helpers.handle({"task": "A", "episodes": ["e1"]}, ctx)
        await helpers.handle({"task": "B", "episodes": ["e2"]}, ctx)
        await asyncio.gather(helpers.result(ctx, "call-1"), helpers.result(ctx, "call-2"))
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    assert model.max_in_flight == 2


def test_a_helper_mentions_another_helper_and_its_wait_returns_it(tmp_path, helper_boxes):
    model = Steps(
        {
            "A": [
                _after(0.1, _msg(_tc("s", "send", {"text": "@call-2 e2/cell:0 prints two"}))),
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
            asyncio.gather(helpers.result(ctx, "call-1"), helpers.result(ctx, "call-2")),
            5,
        )
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    assert model.outputs("A")["s"].startswith("sent to call-2")
    assert "call-1: @call-2 e2/cell:0 prints two" in model.outputs("B")["w"]
    entry = next(e for e in ctx.pass_record.record.entries if e.author == "call-1" and e.kind == "post")
    assert entry.mentions == ("call-2",)


def test_role_names_resolve_and_an_unknown_name_is_refused(tmp_path):
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600, run_home=str(tmp_path / "run-home"))
    for name in ("call-1", "call-2"):
        pr.add_helper(name, "a task")
        pr.record.add_agent(name, spawner="writer")
    out, to = messaging.post(pr, "call-1", "@writer done reading e1", messaging.SEND_SPEC)
    assert to == ["writer"] and out.startswith("sent to writer")
    out, to = messaging.post(pr, "writer", "@helpers stop drafting", messaging.SEND_SPEC)
    assert to == ["call-1", "call-2"]
    out, to = messaging.post(pr, "call-1", "@helpers e1 is done", messaging.SEND_SPEC)
    assert to == ["call-2"]  # never to itself
    out, to = messaging.post(pr, "call-2", "no address", messaging.SEND_SPEC)
    assert to == ["writer"]  # a helper's message that names no one goes to the writer
    refused, to = messaging.post(pr, "call-1", "@nobody look", messaging.SEND_SPEC)
    assert to == [] and tool_spec.form_of(refused) == "unknown_name"
    assert "@nobody" in refused and "@call-2" in refused and "@helpers" in refused
    refused, _ = messaging.post(pr, "writer", "to whom?", messaging.SEND_SPEC)
    assert tool_spec.form_of(refused) == "no_recipient"
    # nothing refused reached the record
    assert not any("nobody" in e.text or "whom" in e.text for e in pr.record.entries)


def test_a_helper_reads_the_writers_inputs_read_only(tmp_path, helper_boxes):
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
        await helpers.result(ctx, "call-1")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")
    assert "def f()" in out["st"] and "pool" in out["pool"] and not out["pool"].startswith("refused")
    assert '"e1"' in out["task"] or "e1" in out["task"]
    (call,) = helper_boxes
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
        await helpers.result(ctx, "call-1")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")["x"]
    assert out.splitlines()[0] == "pool" and "WROTE" not in out
    assert out.count("DENIED") == 3
    assert not (ctx.inputs / "pool" / "x.txt").exists() and not (ctx.inputs / "staging.txt").exists()


@pytest.mark.parametrize("tool", ["send", "reply"])
def test_an_empty_message_is_refused(tool, tmp_path):
    pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 600, run_home=str(tmp_path / "run-home"))
    args = {"text": " "} if tool == "send" else {"agent": "writer", "text": " "}
    out = asyncio.run(messaging.call(pr, "writer", tool, args))
    assert tool_spec.form_of(out) == "no_text"


def _readable_inputs(ctx):
    """The writer's /inputs as a pass stages it: its batch map, a batch episode record and a shared block."""
    (ctx.inputs / "batch_map.json").write_text(
        json.dumps({"episodes": [{"episode_id": "e1", "observations": 1, "cells": 1, "actions": 0}]}),
    )
    (ctx.inputs / "episodes").mkdir()
    (ctx.inputs / "episodes" / "e1.json").write_text('{"episode": "e1"}')
    (ctx.inputs / "shared").mkdir()
    (ctx.inputs / "shared" / "a1b2c3d4e5f6.txt").write_text("the shared rules")


def test_the_recorded_helper_calls_that_were_refused_now_work(tmp_path, helper_boxes):
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
        await helpers.result(ctx, "call-1")
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


@needs_bwrap
def test_a_helper_cell_imports_the_library_as_the_writer_sees_it(tmp_path):
    code = "import memory, os\nprint(memory.__file__)\nprint(os.getcwd())\n"
    model = Steps({"t": [_msg(_tc("x", "execute_code", {"code": code})), _done()]})
    ctx = _ctx(tmp_path, model)
    (ctx.library / "memory").mkdir()
    (ctx.library / "memory" / "__init__.py").write_text("ITEMS = []\n")

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-1")
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    out = model.outputs("t")["x"].splitlines()
    assert out[:2] == ["/memory/memory/__init__.py", "/work"], out
