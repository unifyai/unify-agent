"""The memory agents' tools in code (``PassConfig.tools_in_code``): every request carries only ``execute_code``; the
writer's, the curator's and a helper's other tools are calls from cells through the tool channel, answered by
``tool_handlers`` with today's texts and forms. Keyless and boxless: the interpreter is an in-process stand-in whose
cells are JSON scripts of tool calls made through the real box-side client over the real channel pipes.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.common import tool_channel as tc
from unify.memory import tool_handlers, tool_spec
from unify.memory.trigger import PassRequest
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


#: One tool call as a scripted model's cell makes it (:func:`tests.memory.in_code_calls.as_cell`).
_AS_CELL = re.compile(r"try:\n    print\(tools\.(\w+)\(\*\*(\{.*\})\)\)\nexcept Exception as _exc:\n    print\(_exc\)\n\Z", re.S)


def _calls_of(code: str) -> list:
    """The calls a FakeBox cell makes: a JSON list of ``[tool, args]``, or one call in the form
    :func:`tests.memory.in_code_calls.as_cell` writes (a scripted model's own calls, such as its closing finish)."""
    m = _AS_CELL.match(code)
    if m is not None:
        return [(m.group(1), ast.literal_eval(m.group(2)))]
    return json.loads(code)


class FakeBox:
    """Stands in for ``BoxInterpreter``: it opens a channel from the factory it is given, serves it, brackets each
    cell, and runs a cell (a JSON list of ``[tool, args]`` calls) in a thread through the box-side client; each call
    prints one line (its text, or the exception the cell would see)."""

    started: list["FakeBox"] = []

    @classmethod
    async def start(cls, *, agent, profile, binds, env, channel, stub_source, cell_limit_s):
        box = cls()
        box.agent, box.env, box.binds, box.profile = agent, dict(env), binds, profile
        box.channel = channel() if callable(channel) and not hasattr(channel, "box_fds") else channel
        box.client = _client()
        req, rep = box.channel.box_fds()
        box.client._STATE["fds"] = (os.dup(req), os.dup(rep))  # as a spawn inherits them
        box.channel.after_spawn()
        box.serve = asyncio.ensure_future(box.channel.serve())
        box.stopped = asyncio.Event()
        box.cells = 0
        cls.started.append(box)
        return box

    def host_paths(self):
        return [str(p) for group in ("ro", "rw") for p in (self.binds.get(group) or {})]

    def box_alive(self):
        return not self.stopped.is_set()

    async def measure_quota(self):
        measure = getattr(self.profile, "measure", None)
        return await asyncio.to_thread(measure) if measure is not None else None

    def _root(self, group, box_path=None):
        for host, mounted in (self.binds.get(group) or {}).items():
            if box_path is None or mounted == box_path:
                return Path(host)
        raise KeyError(box_path)

    def _pseudo(self, tool, args):
        """Test set-up between calls, as a cell could do it (or, for /inputs, as the host stages it)."""
        if tool == "__sleep__":
            time.sleep(float(args))
            return
        if tool == "__until_left__":  # sleep until this many seconds of the pass's wall clock are left
            time.sleep(max(0.0, self.channel.binding.deadline - time.monotonic() - float(args)))
            return
        for kind, (rel, value) in args.items():
            if kind == "input":
                path = self._root("ro", "/inputs") / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
                continue
            path = self._root("rw") / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "link":
                path.symlink_to(value)
            elif kind == "bytes":
                path.write_bytes(bytes.fromhex(value))

    async def run_cell(self, code, *, call=None):
        self.cells += 1
        cell = self.cells
        lines: list[str] = []
        loop = asyncio.get_running_loop()
        done = loop.create_future()

        def go():
            for tool, args in _calls_of(code):
                if tool in ("__sleep__", "__setup__", "__until_left__"):
                    self._pseudo(tool, args)
                    continue
                try:
                    out = self.client._call(tool, args)
                    lines.append(f"{tool} -> {out}")
                except Exception as exc:  # noqa: BLE001 - printed as the cell would show it
                    lines.append(f"{tool} !! {type(exc).__name__}: {exc}")
            loop.call_soon_threadsafe(lambda: done.done() or done.set_result(None))

        self.channel.cell_started(cell)
        try:
            threading.Thread(target=go, daemon=True).start()
            stop = asyncio.ensure_future(self.stopped.wait())
            await asyncio.wait({done, stop}, return_when=asyncio.FIRST_COMPLETED)
            stop.cancel()
        finally:
            self.channel.cell_ended(cell)
        return SimpleNamespace(
            output="\n".join(list(lines)),
            error=None,
            timed_out=False,
            restarted=False,
            opens=None,
            quota=None,
            notes_before=[],
            notes_after=(["(the interpreter exited during the cell)"] if self.stopped.is_set() else []),
        )

    async def stop(self):
        self.stopped.set()
        await self.channel.close("stopped")
        if self.serve is not asyncio.current_task() and not self.serve.done():
            self.serve.cancel()


class ToolsSeen(Turns):
    def __init__(self, messages, usd="0.001"):
        super().__init__(messages, usd)
        self.tool_lists: list[list[str]] = []

    async def __call__(self, messages, tools):
        self.tool_lists.append([t["function"]["name"] for t in tools])
        return await super().__call__(messages, tools)


def _around(blob, bad):
    i = blob.find(bad)
    return blob[max(0, i - 300) : i + 100]


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
        # a model-API call of an in-code tool, as a model would send it (not through a cell)
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "j1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "/inputs"})}},
            ],
        },
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
    # c1's finish was refused (no manifest); the scripted model's closing finish is the second refused finish, and
    # rev 2 ends the pass there (FINISH_G1_REFUSALS), unfinished in substance: nothing landed
    assert out.finished is True and not out.passed and "no manifest" in out.reasons


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
    # the calls of cell c1 (the scripted model's own closing finish is a cell of its own after it)
    in_code = [r for r in calls if r["via"] == "channel" and r["cell"] == cell_row["call"]]
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
    assert sorted(r["call"] for r in writer.tool_events) == sorted(r["call"] for r in calls)
    # every handler thread had returned when the gate would start
    (end,) = [r for r in rows if r["event"] == "pass_end"]
    assert end["handlers_alive_at_gate"] == 0 and "temp_kept" not in end


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
    ch = tc.Channel.open(binding, allow, rows.append)  # the shared registry's specs
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


# --- no secret, host path or route value in any reply or row --------------------------------------------------------------


def _summary(out, rows):
    return {
        "reasons": getattr(out, "reasons", None),
        "rows": [
            (r.get("agent"), r.get("tool"), r.get("status"), (r.get("refusal") or {}).get("form"), r.get("reason"))
            for r in rows
            if r.get("event") in ("tool_call", "channel_closed")
        ],
    }


def _check_no_leak(tmp_path, model, rows):
    texts = [str(v) for v in model.outputs.values()]
    events_text = (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text()
    call_rows = [json.dumps(r) for r in rows if r.get("event") in ("tool_call", "channel_closed")]
    host = {str(tmp_path), os.path.expanduser("~")}
    for box in FakeBox.started:
        host.update(box.host_paths())
    host.discard("")
    for blob in (*texts, events_text):  # every reply and every record the pass wrote
        for bad in (PLANTED_ENV, PLANTED_TOKEN):
            assert bad not in blob, (bad, _around(blob, bad))
    for blob in (*texts, *call_rows):  # every reply and every call's row
        for bad in host:
            assert bad not in blob, (bad, _around(blob, bad))


WRITER_REFUSALS = [
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
    ("send", {"text": ""}),
    ("reply", {"agent": "nobody", "text": "hi"}),
    ("stop", {"helper": "nobody"}),
    ("delegate", {"task": "", "episodes": []}),
    ("similar", {"episode": "e1"}),
    ("pool", {"op": "bogus"}),
    ("bogus_tool", {}),
    ("finish", {"summary": "done"}),
]


def test_no_writer_reply_or_row_carries_an_environment_value_a_host_path_or_a_route_token(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_PLANTED_FOR_THE_CHANNEL_TEST", PLANTED_ENV)
    monkeypatch.setenv("UNIFY_MEMORY_LIBRARY_HELPER_TOKEN", PLANTED_TOKEN)
    out, model, writer, rows = _in_code_pass(
        tmp_path,
        [_cell("c1", *WRITER_REFUSALS)],
        helper=_helper_factory([("done", {"summary": "s", "claims": []})], []),
        max_calls=12,  # check, replay and fixture each count against it as today (rev 2 has no write or edit tool)
        deadline_s=90.0,
    )
    assert "c1" in model.outputs, _summary(out, rows)
    _check_no_leak(tmp_path, model, rows)
    cell = model.outputs["c1"]
    lines = cell.splitlines()
    # one line per call before finish, then how the cell ended, then finish's answer
    printed = sum(1 for x in lines if " -> " in x or " !! " in x)
    assert len(WRITER_REFUSALS) - 1 <= printed <= len(WRITER_REFUSALS), cell
    assert "bogus_tool !! Refused: refused: unknown tool 'bogus_tool'" in cell
    assert (
        "read_episode !! Refused: refused: tools.read_episode(episode='e1', nope=1): there is no argument 'nope'"
    ) in cell
    assert lines[-1].startswith("not finished: "), cell
    # the calls of cell c1 (the scripted model's own closing cells come after it)
    c1_row = next(r for r in rows if r.get("via") == "model" and r.get("tool") == "execute_code")
    in_code = [r for r in rows if r.get("via") == "channel" and r.get("cell") == c1_row["call"]]
    assert [r["tool"] for r in in_code] == [t for t, _ in WRITER_REFUSALS], _summary(out, rows)


def test_a_helper_runs_in_its_own_box_and_is_refused_the_writers_tools(tmp_path, monkeypatch):
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
    out, model, writer, rows = _in_code_pass(
        tmp_path,
        [
            _cell("c1", ("delegate", {"task": "draft one", "episodes": []})),
            _cell("c2", ("wait", {})),
            _cell("c3", ("who", {})),
        ],
        helper=_helper_factory(helper_script, seen_helper_tools),
        max_calls=4,
        deadline_s=90.0,
    )
    assert {"c1", "c2", "c3"} <= set(model.outputs), _summary(out, rows)
    _check_no_leak(tmp_path, model, rows)
    assert seen_helper_tools and all(t == ["execute_code"] for t in seen_helper_tools)
    helper_rows = [r for r in rows if r.get("agent") == "helper" and r.get("via") == "channel"]
    assert [r["tool"] for r in helper_rows] == [t for t, _ in helper_script], _summary(out, rows)
    forms = {r["tool"]: (r["refusal"] or {}).get("form") for r in helper_rows[:3]}
    assert forms == {"delegate": "unknown_tool", "finish": "unknown_tool", "check": "unknown_tool"}
    assert helper_rows[-1]["status"] == "ok" and helper_rows[-1]["tool"] == "done"
    assert all(r["helper_call"] == r["name"] for r in helper_rows)
    cell_rows = [r for r in rows if r.get("agent") == "helper" and r.get("tool") == "execute_code"]
    assert cell_rows and all(r["cell"] == cell_rows[0]["call"] for r in helper_rows)


# --- liveness and the gate ---------------------------------------------------------------------------------------------------


def test_a_helper_polling_during_a_slow_check_is_stopped_by_a_finish_requested_mid_check():
    from unify.memory import messaging

    def slow_check(ctx, args):
        time.sleep(2.0)  # a synchronous check that takes its time, as the gate's does
        return Reply("ok")

    specs = {"check": SimpleNamespace(parameters={"type": "object", "properties": {}}, kind="action", blocking=True,
                                      deadline_s=None, handler=slow_check)}

    async def main():
        pr = messaging.PassRecord(root="writer", deadline=time.monotonic() + 60, run_home=__import__("tempfile").mkdtemp())
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
    assert gate_mid is False and idle is True  # no gate while a handler thread lives
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
    import inspect

    from unify.common import tool_registry

    mine = {
        name: spec
        for name, spec in tool_registry.registry().items()
        if spec.handler.startswith("unify.memory.tool_handlers:")
    }
    # rev 2: every agent is on the run record, whose own tools (send, messages, wait, who) the run record's handlers
    # answer (unify.agents.run_record_tools); every other tool tool_handlers handles is its own
    record_tools = {"send", "messages", "wait", "who"}
    assert set(mine) == set(tool_handlers.HANDLED) and not set(tool_handlers.HANDLED) & record_tools
    assert not any(hasattr(tool_handlers, name) for name in record_tools)
    for name in record_tools:
        assert tool_registry.registry()[name].handler == f"unify.agents.run_record_tools:{name}"
    for name, spec in mine.items():
        assert spec.handler == f"unify.memory.tool_handlers:{name}"
        fn = tc.resolve_handler(spec.handler)
        assert fn is getattr(tool_handlers, name) and fn.__name__ == name
        # synchronous work never runs on the event loop: a synchronous handler is dispatched to the worker pool
        # (the registry says blocking), a coroutine handler sends its blocking parts there itself
        assert inspect.iscoroutinefunction(fn) != spec.blocking, name
        for agent in ("writer", "curator", "helper"):
            if agent in spec.roles:
                assert tool_registry.handler_for(agent, name) == (spec.handler, spec.blocking)


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


def test_a_pass_that_ends_with_a_handler_thread_alive_keeps_its_folder_and_says_why(tmp_path, monkeypatch):
    release = threading.Event()
    seen = {}

    def stuck_read(ctx, args):
        seen["box"] = str(ctx.state.box)
        release.wait(30)
        return Reply("late")

    monkeypatch.setattr(tool_handlers, "read", stuck_read)
    import dataclasses

    from unify.common import tool_registry

    monkeypatch.setattr(
        tool_registry,
        "_REGISTRY",
        tuple(
            dataclasses.replace(s, deadline_s=0.2) if s.name == "read" else s
            for s in tool_registry._REGISTRY
        ),
    )
    monkeypatch.setattr(tc, "HANDLER_GRACE_S", 0.1)
    FakeBox.started = []
    model = ToolsSeen([_cell("c1", ("read", {"path": "/inputs"}))])
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=3,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    writer.interpreter_start = FakeBox.start
    writer.in_code_idle_s = 0.2
    try:
        out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
        rows = [
            json.loads(line)
            for line in (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text().splitlines()
        ]
        box = Path(seen["box"])
        # fail closed: no gate, and the folder the thread may still use is left in place
        assert not out.passed and out.commit is None
        assert any("the gate did not start" in r for r in out.reasons), out.reasons
        assert any("temporary folder was kept" in r for r in out.reasons), out.reasons
        assert box.is_dir()
        (end,) = [r for r in rows if r["event"] == "pass_end"]
        assert end["temp_kept"]["reason"] == "a tool handler of this pass is still running"
        assert Path(end["temp_kept"]["path"]) == box.parent
        # the stuck handler's thread was alive when the gate would start
        assert end["handlers_alive_at_gate"] == 1
        closed = [r for r in rows if r["event"] == "channel_closed"]
        assert [r["reason"] for r in closed] == ["handler_stuck"]
    finally:
        release.set()
    shutil.rmtree(Path(seen["box"]).parent, ignore_errors=True)


class CrashBox(FakeBox):
    """A box whose second cell raises out of the interpreter instead of returning a result."""

    async def run_cell(self, code, *, call=None):
        if self.cells >= 1:
            self.cells += 1
            raise RuntimeError("the interpreter broke during the cell")
        return await super().run_cell(code, call=call)


def test_a_cell_that_raises_still_writes_its_row_at_its_reserved_number(tmp_path):
    turns = [
        _cell("c1", ("who", {})),
        _cell("c2", ("who", {})),
    ]
    model = ToolsSeen(turns)
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=4,
        pass_record_root=str(tmp_path / "out"),
    )
    writer.load = EPS.__getitem__
    writer.interpreter_start = CrashBox.start
    with pytest.raises(RuntimeError, match="the interpreter broke"):
        asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text().splitlines()
    ]
    calls = [r for r in rows if r["event"] == "tool_call"]
    cells = sorted((r for r in calls if r["tool"] == "execute_code"), key=lambda r: r["call"])
    assert [r["status"] for r in cells] == ["ok", "error"], cells
    crashed = cells[1]
    assert crashed["exception"] == "RuntimeError" and crashed["via"] == "model" and crashed["cell"] is None
    assert BASE_FIELDS | ADDED_FIELDS <= set(crashed)
    # its number is the one reserved before the cell ran: after the first cell's calls, before any of its own
    assert crashed["call"] == max(r["call"] for r in calls)
    assert sorted(r["call"] for r in calls) == list(range(len(calls)))
    # the message text is not in the row, and no placeholder is left on the pass
    assert "the interpreter broke" not in json.dumps(rows)
    assert not any(r.get("event") == "reserved" for r in writer.tool_events)
    assert sorted(r["call"] for r in writer.tool_events) == list(range(len(writer.tool_events)))
