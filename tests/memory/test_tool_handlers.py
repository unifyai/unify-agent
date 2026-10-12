"""The memory agents' tools in code (``PassConfig.tools_in_code``): every request carries only ``execute_code``; the
writer's, the curator's and a helper's other tools are calls from cells through the tool channel, answered by
``tool_handlers`` with today's texts and forms. Keyless and boxless: the interpreter is an in-process stand-in whose
cells are JSON scripts of tool calls made through the real box-side client over the real channel pipes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.common import tool_channel as tc
from unify.memory import helpers, tool_handlers, tool_spec
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import writer_tools
from unify.memory.writer_tools import Reply
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import Turns, _call, _writer

CLIENT = Path(tc.__file__).with_name("tool_channel_client.py")
PLANTED_ENV = "planted-value-for-the-channel-test"  # pragma: allowlist secret
PLANTED_TOKEN = "planted-helper-token-for-the-channel-test"  # pragma: allowlist secret
BASE_FIELDS = {
    "event",
    "agent",
    "name",
    "pass_id",
    "call",
    "utc",
    "tool",
    "status",
    "refusal",
    "paths",
    "args_bytes",
    "result_bytes",
    "result_truncated",
    "usage",
}
ADDED_FIELDS = {"via", "cell", "args_sha256", "box_gone"}


def _client():
    spec = importlib.util.spec_from_file_location(f"box_tools_{time.monotonic_ns()}", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeBox:
    """Stands in for the persistent interpreter: a cell is a JSON list of ``[tool, args]`` calls, made in a thread
    through the box-side client; each call prints one line (its text, or the exception the cell would see)."""

    started: list["FakeBox"] = []

    @classmethod
    async def start(cls, *, agent, profile, binds, env, channel, stub_source, cell_limit_s):
        box = cls()
        box.agent, box.env = agent, dict(env)
        box.binds = binds() if callable(binds) else binds
        box.client = _client()
        req, rep = channel.box_fds()
        box.client._STATE["fds"] = (os.dup(req), os.dup(rep))  # as a spawn inherits them
        box.stopped = asyncio.Event()
        cls.started.append(box)
        return box

    def host_paths(self):
        return [str(p) for group in ("ro", "rw") for p in (self.binds.get(group) or {})]

    async def run_cell(self, code):
        lines: list[str] = []
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def go():
            for tool, args in json.loads(code):
                try:
                    out = self.client._call(tool, args)
                    lines.append(f"{tool} -> {out}")
                except Exception as exc:  # noqa: BLE001 - printed as the cell would show it
                    lines.append(f"{tool} !! {type(exc).__name__}: {exc}")
            loop.call_soon_threadsafe(lambda: done.done() or done.set_result(None))

        threading.Thread(target=go, daemon=True).start()
        stop = asyncio.ensure_future(self.stopped.wait())
        await asyncio.wait({done, stop}, return_when=asyncio.FIRST_COMPLETED)
        stop.cancel()
        return SimpleNamespace(
            output="\n".join(list(lines)),
            error="",
            timed_out=False,
            restarted=False,
            opens=None,
        )

    async def stop(self):
        self.stopped.set()


class ToolsSeen(Turns):
    def __init__(self, messages, usd="0.001"):
        super().__init__(messages, usd)
        self.tool_lists: list[list[str]] = []

    async def __call__(self, messages, tools):
        self.tool_lists.append([t["function"]["name"] for t in tools])
        return await super().__call__(messages, tools)


def _cell(cid, *calls):
    return _call(cid, "execute_code", {"code": json.dumps([list(c) for c in calls])})


def _helper_factory(script, seen):
    """A helper whose first turn runs *script* as one cell; the factory holds a planted route token."""

    def factory(label=None):
        state = {"step": 0}

        async def turn(messages, tools):
            seen.append([t["function"]["name"] for t in tools])
            state["step"] += 1
            if state["step"] == 1:
                return _cell("h1", *script), "0.001"
            return _cell(f"h{state['step']}", ("done", {"summary": "again", "claims": []})), "0.001"

        turn.route_token = PLANTED_TOKEN
        return turn

    factory.route_token = PLANTED_TOKEN
    return factory


def _in_code_pass(tmp_path, turns, *, helper=None, **cfg):
    FakeBox.started = []
    model = ToolsSeen(turns)
    _, _, writer = _writer(
        tmp_path,
        model,
        library_mode=True,
        tools_in_code=True,
        max_calls=cfg.pop("max_calls", 4),
        pass_record_root=str(tmp_path / "out"),
        **cfg,
    )
    writer.load = EPS.__getitem__
    writer.interpreter_start = FakeBox.start
    if helper is not None:
        writer.helper_turn = helper
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text().splitlines()
    ]
    return out, model, writer, rows


# --- the surface ---------------------------------------------------------------------------------------------------


def test_every_request_carries_only_execute_code_and_the_tools_answer_from_cells(tmp_path):
    turns = [
        _cell(
            "c1",
            ("read", {"path": "/inputs"}),
            ("who", {}),
            ("messages", {}),
            ("read", {"path": "/memory/nothing-here.py"}),
            ("finish", {"summary": "done"}),
        ),
        _call("j1", "read", {"path": "/inputs"}),
    ]
    out, model, writer, rows = _in_code_pass(tmp_path, turns)
    assert model.tool_lists and all(t == ["execute_code"] for t in model.tool_lists)
    text = model.outputs["c1"]
    assert "read -> " in text and "who -> " in text
    assert "read !! Refused: refused: " in text  # the missing file, with today's form
    # finish killed the box first; its answer is the cell's last line (no manifest: refused, today's text)
    assert text.splitlines()[-1].startswith("not finished: ")
    # a JSON call of an in-code tool is an unknown tool
    assert model.outputs["j1"].startswith("refused: unknown tool 'read'")
    assert out.finished is False


def test_the_event_file_has_the_base_fields_plus_the_additive_ones(tmp_path):
    turns = [
        _cell(
            "c1",
            ("read", {"path": "/inputs"}),
            ("grep", {"pattern": "(", "path": "/inputs"}),
            ("who", {}),
        ),
    ]
    _, _, writer, rows = _in_code_pass(tmp_path, turns, max_calls=2)
    calls = [r for r in rows if r["event"] == "tool_call"]
    assert calls and all(BASE_FIELDS | ADDED_FIELDS <= set(r) for r in calls)
    cell_row = next(r for r in calls if r["tool"] == "execute_code")
    in_code = [r for r in calls if r["via"] == "channel"]
    assert cell_row["via"] == "model" and cell_row["cell"] is None
    assert [r["tool"] for r in in_code] == ["read", "grep", "who"]
    assert all(r["cell"] == cell_row["call"] and r["call"] > cell_row["call"] for r in in_code)
    assert all(r["usage"] is None for r in in_code) and cell_row["usage"] is not None
    assert sorted(r["call"] for r in calls) == list(range(len(calls)))
    grep_row = next(r for r in in_code if r["tool"] == "grep")
    assert grep_row["status"] == "refused" and grep_row["refusal"]["form"] == "bad_pattern"
    codes = tool_spec.refusal_codes()
    for r in calls:
        if r["refusal"] is not None:
            assert r["refusal"]["form"] in codes.get(r["tool"], []) + codes["*"], r
    # the rows kept on the pass are the file's rows
    assert [r["call"] for r in writer.tool_events] == [r["call"] for r in calls]


def test_off_the_requests_and_rows_are_as_before(tmp_path):
    model = ToolsSeen([_call("a", "read", {"path": "/inputs"})])
    _, _, writer = _writer(
        tmp_path,
        model,
        library_mode=True,
        max_calls=2,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    names = [t["function"]["name"] for t in writer_tools(library_mode=True)]
    assert model.tool_lists[0] == names
    assert all(not (ADDED_FIELDS & set(r)) for r in writer.tool_events)


# --- the allow-lists ---------------------------------------------------------------------------------------------------


def _allow(names):
    return tuple(n for n in names if n != "execute_code")


def _channel_for(agent, allow, state):
    binding = tc.CallContext(
        agent=agent,
        name=agent,
        role=agent,
        pass_id="p1",
        request_id=None,
        roots={},
        limits=None,
        counters=state,
        record=None,
        episodes=None,
        deadline=time.monotonic() + 60,
        state=state,
    )
    rows: list[dict] = []
    ch = tc.Channel.open(binding, allow, rows.append, specs=tool_handlers.channel_specs())
    box = _client()
    req, rep = ch.box_fds()
    box._STATE["fds"] = (os.dup(req), os.dup(rep))
    ch.after_spawn()
    ch.cell_started(0)
    return ch, box, rows


def _thread_call(box, tool, args):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def go():
        try:
            out = box._call(tool, args)
        except Exception as exc:  # noqa: BLE001
            out = exc
        loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(out))

    threading.Thread(target=go, daemon=True).start()
    return fut


def test_a_helper_cannot_call_delegate_finish_or_check_and_a_curator_cannot_call_pool():
    helper_allow = _allow(t["function"]["name"] for t in helpers.helper_tools())
    curator_allow = _allow(
        t["function"]["name"]
        for t in writer_tools(library_mode=True, role="curate", pool=True, helper=True, similar=True)
    )
    assert "delegate" not in helper_allow and "pool" not in curator_allow

    async def main():
        out = {}
        ch, box, rows = _channel_for("helper", helper_allow, SimpleNamespace())
        task = asyncio.ensure_future(ch.serve())
        for tool, args in (("delegate", {"task": "x"}), ("finish", {"summary": "s"}), ("check", {})):
            out[tool] = await _thread_call(box, tool, args)
        await ch.close("ended", record=False)
        await asyncio.gather(task, return_exceptions=True)
        ch2, box2, rows2 = _channel_for("curator", curator_allow, SimpleNamespace())
        task2 = asyncio.ensure_future(ch2.serve())
        out["pool"] = await _thread_call(box2, "pool", {"op": "list"})
        await ch2.close("ended", record=False)
        await asyncio.gather(task2, return_exceptions=True)
        return out, rows + rows2

    out, rows = asyncio.run(main())
    for tool, exc in out.items():
        assert getattr(exc, "form", None) == "unknown_tool", (tool, exc)
        assert f"unknown tool {tool!r}" in str(exc)
    assert [r["refusal"]["form"] for r in rows] == ["unknown_tool"] * 4


# --- criterion 5: no secret, host path or route value in any reply or row ------------------------------------------------


def test_no_reply_or_row_carries_an_environment_value_a_host_path_or_a_route_token(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_PLANTED_FOR_THE_CHANNEL_TEST", PLANTED_ENV)
    monkeypatch.setenv("UNIFY_MEMORY_LIBRARY_HELPER_TOKEN", PLANTED_TOKEN)
    seen_helper_tools: list[list[str]] = []
    helper_script = [
        ("delegate", {"task": "more", "episodes": []}),
        ("finish", {"summary": "s"}),
        ("check", {}),
        ("read", {"path": "/etc/passwd"}),
        ("read", {"path": "/work/missing.txt"}),
        ("grep", {"pattern": "(", "path": "/work"}),
        ("overview", {"episode": "e999"}),
        ("read_episode", {"episode": "../../etc"}),
        ("stage", {"path": "../escape.py", "text": "x"}),
        ("send", {"text": ""}),
        ("ask", {"question": ""}),
        ("done", {"summary": "s", "claims": [{"claim": "", "sources": []}]}),
        ("done", {"summary": "s", "claims": [], "extra": 1}),
        ("done", {"summary": "handing back", "claims": []}),
    ]
    writer_script = [
        ("read", {"path": "/etc/passwd"}),
        ("read", {"path": "/memory/../../etc/passwd"}),
        ("read", {"path": "/memory/missing.py"}),
        ("grep", {"pattern": "(", "path": "/inputs"}),
        ("grep", {"pattern": "x", "path": "/elsewhere"}),
        ("overview", {"episode": "e999"}),
        ("read_episode", {"episode": "e999"}),
        ("read_episode", {"episode": "e1", "parts": ["nope"]}),
        ("read_episode", {"episode": "e1", "nope": 1}),
        ("fixture", {"episode": "e999", "dest": "memory/x/tests/fixtures/a.json"}),
        ("replay", {"item": "nothing"}),
        ("check", {"manifest": "{not json"}),
        ("write", {"path": "../escape.py", "content": "x"}),
        ("write", {"path": ".git/config", "content": "x"}),
        ("write", {"path": "memory/ok.py", "content": "x = 1\n"}),
        ("edit", {"path": "memory/missing.py", "old": "a", "new": "b"}),
        ("edit", {"path": "memory/ok.py", "old": "zzz", "new": "y"}),
        ("send", {"text": ""}),
        ("reply", {"agent": "nobody", "text": "hi"}),
        ("stop", {"helper": "nobody"}),
        ("delegate", {"task": "", "episodes": []}),
        ("delegate", {"task": "draft one", "episodes": []}),
        ("wait", {}),
        ("similar", {"episode": "e1"}),
        ("pool", {"op": "bogus"}),
        ("bogus_tool", {}),
        ("finish", {"summary": "done"}),
    ]
    out, model, writer, rows = _in_code_pass(
        tmp_path,
        [_cell("c1", *writer_script)],
        helper=_helper_factory(helper_script, seen_helper_tools),
        edit_tools=True,
        max_calls=3,
        deadline_s=120.0,
    )
    texts = [str(v) for v in model.outputs.values()]
    events_text = (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text()
    forbidden = {PLANTED_ENV, PLANTED_TOKEN, str(tmp_path), os.path.expanduser("~")}
    for box in FakeBox.started:
        forbidden.update(box.host_paths())
    forbidden.discard("")
    for blob in (*texts, events_text):
        for bad in forbidden:
            assert bad not in blob, bad
    # every call answered: one line per scripted call until finish ended the cell
    cell = model.outputs["c1"]
    assert cell.count("\n") >= len(writer_script) - 1
    assert "bogus_tool !! Refused: refused: unknown tool 'bogus_tool'" in cell
    assert "read_episode !! Refused: refused: unknown argument 'nope'" in cell
    # the helper ran in its own box with only execute_code as its JSON tool, and was refused the writer's tools
    assert seen_helper_tools and all(t == ["execute_code"] for t in seen_helper_tools)
    helper_rows = [r for r in rows if r.get("agent") == "helper" and r.get("via") == "channel"]
    forms = {r["tool"]: r["refusal"]["form"] for r in helper_rows if r["refusal"]}
    assert forms.get("delegate") == forms.get("finish") == forms.get("check") == "unknown_tool"
    assert any(r["tool"] == "done" and r["status"] == "ok" for r in helper_rows)
    assert all(r["helper_call"] == r["name"] for r in helper_rows)


# --- liveness and C4 ---------------------------------------------------------------------------------------------------


def test_a_helper_polling_during_a_slow_check_is_stopped_by_a_finish_requested_mid_check():
    from unify.memory import messaging

    def slow_check(ctx, args):
        time.sleep(2.0)  # a synchronous check that takes its time, as the gate's does
        return Reply("ok")

    specs = {"check": tool_handlers.SpecView("check", {"type": "object", "properties": {}}, "action", True, None, slow_check)}

    async def main():
        pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 60)
        state = SimpleNamespace()
        binding = tc.CallContext(
            agent="writer", name="writer", role="write", pass_id="p1", request_id=None, roots={},
            limits=None, counters=state, record=pr, episodes=None, deadline=time.monotonic() + 60, state=state,
        )
        threads = tc.HandlerThreads()
        ch = tc.Channel.open(binding, ("check",), None, specs=specs, threads=threads)
        box = _client()
        req, rep = ch.box_fds()
        box._STATE["fds"] = (os.dup(req), os.dup(rep))
        ch.after_spawn()
        ch.cell_started(0)
        serve = asyncio.ensure_future(ch.serve())
        polls = {"n": 0, "stopped_at": None}
        started = time.monotonic()

        async def helper_loop():
            # a helper's loop reads its stop flag before each model call
            while pr.stopping.get("helper-1") is None:
                polls["n"] += 1
                await asyncio.sleep(0.02)
            polls["stopped_at"] = time.monotonic() - started

        helper = asyncio.ensure_future(helper_loop())
        check = _thread_call(box, "check", {})
        await asyncio.sleep(0.3)
        pr.stopping["helper-1"] = "the writer finished"  # finish, requested while the check runs
        await asyncio.wait_for(helper, 1.0)
        gate_may_start_mid_check = not threads.alive()
        result = await check
        idle = await threads.wait_idle(5)
        await ch.close("ended", record=False)
        await asyncio.gather(serve, return_exceptions=True)
        return polls, gate_may_start_mid_check, str(result), idle

    polls, gate_mid, result, idle = asyncio.run(main())
    assert polls["stopped_at"] is not None and polls["stopped_at"] < 1.0
    assert polls["n"] >= 10  # the loop ran throughout
    assert gate_mid is False and idle is True  # C4
    assert result == "ok"


def test_pool_decide_writes_the_draft_manifest_without_following_a_link(tmp_path):
    box = tmp_path / "memory"
    (box / ".pass").mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("untouched\n")
    (box / ".pass" / "manifest.json").symlink_to(target)
    with pytest.raises(OSError):
        tool_handlers.write_draft_manifest(box, {"items": []})
    assert target.read_text() == "untouched\n"
    (box / ".pass" / "manifest.json").unlink()
    tool_handlers.write_draft_manifest(box, {"items": []})
    assert json.loads((box / ".pass" / "manifest.json").read_text()) == {"items": []}


def test_every_memory_tool_has_a_handler_named_as_the_tool():
    for name in tool_handlers.TOOL_FIELDS:
        fn = getattr(tool_handlers, name)
        assert callable(fn) and fn.__name__ == name
    specs = tool_handlers.channel_specs()
    for name, spec in specs.items():
        if isinstance(spec, tool_handlers.SpecView):
            assert spec.handler == f"unify.memory.tool_handlers:{name}"
            assert tc.resolve_handler(spec.handler) is getattr(tool_handlers, name)


def test_the_worker_pool_reads_the_evidence_store_through_its_own_connection(tmp_path):
    from unify.memory.evidence import EvidenceStore

    ev = EvidenceStore(tmp_path / "e.sqlite")
    undo = tool_handlers.thread_reads(ev)
    try:
        own = ev.db.execute("SELECT count(*) FROM passes").fetchone()
        other = {}

        def read():
            other["n"] = ev.db.execute("SELECT count(*) FROM passes").fetchone()

        t = threading.Thread(target=read)
        t.start()
        t.join()
        assert own == other["n"]
    finally:
        undo()
    assert type(ev.db).__name__ == "Connection"

