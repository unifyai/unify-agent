"""The box-to-host tool channel: framing and its order of checks, the allow-list and the schema, one call in flight,
deadlines, a box that goes away, frames between cells, the counters under a parallel flood, a stalled frame and the
handler threads the gate waits for. Keyless and boxless: the "box" is the stdlib client run in threads of the test.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from unify.common import tool_channel as tc
from unify.memory.writer_tools import Reply, general

CLIENT = Path(tc.__file__).with_name("tool_channel_client.py")


@dataclass(frozen=True)
class Spec:
    name: str
    parameters: dict
    handler: Any
    kind: str = "reader"
    blocking: bool = True
    deadline_s: float | None = None


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required)}


def echo(ctx, args):
    return Reply(f"echo {args.get('text', '')}", paths=({"path": "/memory/x", "mode": "read"},))


def refuse(ctx, args):
    return general("readers_per_turn")


def boom(ctx, args):
    raise RuntimeError("/host/secret/path should never be shown")


SPECS = {
    "echo": Spec("echo", _obj({"text": {"type": "string"}}, ["text"]), echo),
    "refuse": Spec("refuse", _obj({}), refuse),
    "boom": Spec("boom", _obj({}), boom),
}


def _binding(counters=None, deadline_s=60.0, name="writer"):
    return tc.CallContext(
        agent="writer",
        name=name,
        role="write",
        pass_id="p1",
        request_id=None,
        roots={},
        limits=None,
        counters=counters,
        record=None,
        episodes=None,
        deadline=time.monotonic() + deadline_s,
        state=None,
    )


def _client(channel):
    """A fresh copy of the box's client module, wired to the channel's box ends (duplicated, as a spawn would)."""
    spec = importlib.util.spec_from_file_location(f"box_tools_{id(channel)}_{time.monotonic_ns()}", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    req, rep = channel.box_fds()
    mod._STATE["fds"] = (os.dup(req), os.dup(rep))
    return mod


class Harness:
    def __init__(self, specs=SPECS, allow=None, counters=None, threads=None, deadline_s=60.0):
        self.rows: list[dict] = []
        self.channel = tc.Channel.open(
            _binding(counters, deadline_s),
            tuple(allow if allow is not None else specs),
            self.rows.append,
            specs=specs,
            threads=threads,
        )
        self.box = _client(self.channel)
        self.channel.after_spawn()
        self.channel.cell_started(0)
        self.task = asyncio.ensure_future(self.channel.serve())

    async def call(self, tool, args):
        """One call from its own box thread (never the loop's executor, which the handlers' work must not wait on)."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def go():
            out = self._call(tool, args)
            loop.call_soon_threadsafe(lambda: fut.done() or fut.set_result(out))

        threading.Thread(target=go, daemon=True).start()
        return await fut

    def _call(self, tool, args):
        try:
            return self.box._call(tool, args)
        except Exception as exc:  # noqa: BLE001 - the test inspects what the cell would see
            return exc

    def raw(self, data: bytes):
        os.write(self.box._STATE["fds"][0], data)

    def read_reply(self):
        fd = self.box._STATE["fds"][1]
        n = int.from_bytes(os.read(fd, 4), "big")
        body = b""
        while len(body) < n:
            body += os.read(fd, n - len(body))
        return json.loads(body)

    async def ended(self, timeout=5.0):
        await asyncio.wait_for(self.task, timeout)
        return self.channel.close_reason

    async def close(self):
        await self.channel.close("ended", record=False)
        for fd in self.box._STATE["fds"]:
            try:
                os.close(fd)
            except OSError:
                pass


def run(coro):
    return asyncio.run(coro)


def _frame(obj) -> bytes:
    data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return len(data).to_bytes(4, "big") + data


# --- calls -------------------------------------------------------------------------------------------------------


def test_a_call_returns_its_text_and_paths_and_writes_one_row_with_the_additive_fields():
    async def main():
        h = Harness()
        out = await h.call("echo", {"text": "hi"})
        await h.close()
        return h, out

    h, out = run(main())
    assert isinstance(out, str) and out == "echo hi"
    assert out.paths == [{"path": "/memory/x", "mode": "read"}]
    (row,) = h.rows
    assert row["event"] == "tool_call" and row["tool"] == "echo" and row["status"] == "ok"
    assert row["via"] == "channel" and row["cell"] == 0 and row["box_gone"] is False
    canon = tc.canonical_args({"text": "hi"})
    assert row["args_bytes"] == len(canon) and len(row["args_sha256"]) == 64
    assert row["agent"] == "writer" and row["pass_id"] == "p1" and row["usage"] is None
    for key in ("refusal", "paths", "result_bytes", "result_truncated", "utc", "name"):
        assert key in row


def test_refusals_errors_and_not_run_raise_their_exceptions_with_the_frames_hidden():
    async def main():
        h = Harness()
        a = await h.call("refuse", {})
        b = await h.call("boom", {})
        await h.close()
        return h, a, b

    h, not_run, err = run(main())
    assert type(not_run).__name__ == "NotRun" and isinstance(not_run, h.box.Refused)
    assert str(not_run).startswith("not run: ") and not_run.form == "readers_per_turn"
    assert not_run.__cause__ is None and not_run.__suppress_context__
    assert type(err).__name__ == "ToolError" and "RuntimeError" in str(err)
    assert "/host/secret" not in str(err)
    assert [r["status"] for r in h.rows] == ["refused", "error"]


def test_an_unknown_tool_and_an_unknown_argument_are_refusals_and_the_channel_stays_open():
    async def main():
        h = Harness(allow=("echo",))
        a = await h.call("boom", {})
        b = await h.call("echo", {"text": "x", "role": "writer"})
        c = await h.call("echo", {})
        d = await h.call("echo", {"text": 3})
        e = await h.call("echo", {"text": "still open"})
        await h.close()
        return h, a, b, c, d, e

    h, unknown, extra, missing, wrong, ok = run(main())
    assert isinstance(unknown, h.box.Refused) and unknown.form == "unknown_tool"
    assert "your tools: echo" in str(unknown)
    for exc in (extra, missing, wrong):
        assert isinstance(exc, h.box.Refused) and exc.form == "bad_arguments"
    assert "'role'" in str(extra)
    assert ok == "echo still open" and h.channel.close_reason == "ended"


# --- malformed frames ----------------------------------------------------------------------------------------------

GOOD = {"v": 1, "id": 1, "tool": "echo", "args": {"text": "a"}}


@pytest.mark.parametrize(
    "payload, reason",
    [
        (b"\xff\xfe", "not_utf8"),
        (b"{not json", "not_json"),
        (b"[1, 2]", "bad_envelope"),
        ({**GOOD, "role": "writer"}, "bad_envelope"),
        ({k: v for k, v in GOOD.items() if k != "args"}, "bad_envelope"),
        ({**GOOD, "v": 2}, "bad_version"),
        ({**GOOD, "v": True}, "bad_version"),
        ({**GOOD, "id": True}, "bad_envelope"),
        ({**GOOD, "id": "1"}, "bad_envelope"),
        ({**GOOD, "tool": 7}, "bad_envelope"),
        ({**GOOD, "args": []}, "bad_envelope"),
    ],
)
def test_each_malformed_frame_closes_the_channel_with_one_row(payload, reason):
    async def main():
        h = Harness()
        h.raw(_frame(payload))
        closed = await h.ended()
        notice = await asyncio.to_thread(h.read_reply)
        after = await h.call("echo", {"text": "x"})
        await h.close()
        return h, closed, notice, after

    h, closed, notice, after = run(main())
    assert closed == reason
    assert notice["status"] == "closed" and notice["form"] == reason
    assert [r["event"] for r in h.rows] == ["channel_closed"] and h.rows[0]["reason"] == reason
    assert type(after).__name__ == "ChannelClosed"


def test_an_envelope_with_a_role_field_never_reaches_a_handler():
    seen = []

    def record(ctx, args):
        seen.append(args)
        return "ok"

    async def main():
        h = Harness(specs={"echo": Spec("echo", _obj({}), record)})
        h.raw(_frame({"v": 1, "id": 1, "tool": "echo", "args": {}, "role": "writer", "agent": "writer"}))
        closed = await h.ended()
        await h.close()
        return closed

    assert run(main()) == "bad_envelope" and seen == []


def test_oversize_and_too_deep_frames_are_refused_before_any_parsing(monkeypatch):
    def no_parse(*a, **k):
        raise AssertionError("parsed")

    monkeypatch.setattr(tc.json, "loads", no_parse)

    async def main():
        big = Harness()
        # only the length prefix is sent: the frame is refused on its length, its body is never read
        big.raw((tc.MAX_REQUEST_BYTES + 1).to_bytes(4, "big"))
        r1 = await big.ended()
        await big.close()
        deep = Harness()
        nested = "[" * (tc.MAX_DEPTH + 1) + "]" * (tc.MAX_DEPTH + 1)
        deep.raw(_frame(f'{{"v":1,"id":1,"tool":"echo","args":{{"x":{nested}}}}}'.encode()))
        r2 = await deep.ended()
        await deep.close()
        return r1, r2

    assert run(main()) == ("oversize", "too_deep")


def test_brackets_inside_strings_do_not_count_as_depth():
    text = json.dumps({"v": 1, "id": 1, "tool": "echo", "args": {"text": "[[[[" * 40}})
    assert not tc.too_deep(text)
    assert tc.parse_request(text.encode(), None)["args"]["text"].startswith("[[[[")


def test_ids_must_strictly_increase():
    async def main():
        h = Harness()
        h.raw(_frame({**GOOD, "id": 5}))
        first = await asyncio.to_thread(h.read_reply)
        h.raw(_frame({**GOOD, "id": 5}))
        closed = await h.ended()
        await h.close()
        return first, closed, h.rows

    first, closed, rows = run(main())
    assert first["id"] == 5 and first["status"] == "ok"
    assert closed == "id_not_increasing"
    assert [r["event"] for r in rows] == ["tool_call", "channel_closed"]


def test_a_stalled_frame_closes_with_frame_timeout(monkeypatch):
    monkeypatch.setattr(tc, "FRAME_READ_TIMEOUT_S", 0.3)

    async def main():
        h = Harness()
        h.raw(b"\x00\x00")  # half a length prefix, then nothing
        started = time.monotonic()
        closed = await h.ended()
        took = time.monotonic() - started
        await h.close()
        return closed, took, h.rows

    closed, took, rows = run(main())
    assert closed == "frame_timeout" and took < 3.0
    assert rows[-1]["event"] == "channel_closed" and rows[-1]["reason"] == "frame_timeout"


def test_a_frame_between_cells_closes_the_channel():
    async def main():
        h = Harness()
        ok = await h.call("echo", {"text": "in the cell"})
        h.channel.cell_ended(0)
        late = await h.call("echo", {"text": "after the cell"})
        closed = await h.ended()
        await h.close()
        return ok, late, closed

    ok, late, closed = run(main())
    assert ok == "echo in the cell"
    assert type(late).__name__ == "ChannelClosed" and closed == "between_cells"


# --- one call in flight, deadlines, a box that goes away ---------------------------------------------------------------


def test_ten_threads_calling_at_once_get_serialised_answers():
    state = {"now": 0, "most": 0}
    lock = threading.Lock()

    def slow(ctx, args):
        with lock:
            state["now"] += 1
            state["most"] = max(state["most"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return f"answer {args['text']}"

    async def main():
        h = Harness(specs={"echo": Spec("echo", _obj({"text": {"type": "string"}}), slow)})
        outs = await asyncio.gather(*(h.call("echo", {"text": str(i)}) for i in range(10)))
        await h.close()
        return outs, h.rows

    outs, rows = run(main())
    assert sorted(outs) == sorted(f"answer {i}" for i in range(10))
    assert all(o == f"answer {i}" for i, o in enumerate(outs))
    assert state["most"] == 1 and len(rows) == 10


def test_a_slow_coroutine_handler_is_answered_not_run_at_its_deadline():
    async def slow(ctx, args):
        await asyncio.sleep(5)
        return "late"

    async def main():
        h = Harness(specs={"echo": Spec("echo", _obj({}), slow, blocking=False, deadline_s=0.2)})
        out = await h.call("echo", {})
        again = await h.call("echo", {})
        await h.close()
        return h, out, again

    h, out, again = run(main())
    assert isinstance(out, h.box.NotRun) and str(out) == "not run: timed out"
    assert isinstance(again, h.box.NotRun)  # the channel stays open for a coroutine handler


def test_a_handler_thread_past_its_deadline_closes_the_channel_and_is_tracked_until_it_ends(monkeypatch):
    monkeypatch.setattr(tc, "HANDLER_GRACE_S", 0.1)
    release = threading.Event()

    def stuck(ctx, args):
        release.wait(10)
        return "done"

    async def main():
        pass_threads = tc.HandlerThreads()
        h = Harness(
            specs={"echo": Spec("echo", _obj({}), stuck, deadline_s=0.2)},
            threads=pass_threads,
        )
        out = await h.call("echo", {})
        closed = await h.ended()
        # C4: the gate may not start while the thread lives
        alive = (h.channel.handler_threads_alive(), pass_threads.alive())
        idle_early = await pass_threads.wait_idle(0.05)
        release.set()
        idle = await pass_threads.wait_idle(5)
        await h.close()
        return h, out, closed, alive, idle_early, idle

    h, out, closed, alive, idle_early, idle = run(main())
    assert str(out) == "not run: timed out"
    assert closed == "handler_stuck" and h.channel.stuck
    assert alive == (True, True) and idle_early is False and idle is True


def test_the_box_gone_mid_call_is_recorded_and_the_completed_action_stands():
    effects = []

    async def main():
        holder = {}

        def act(ctx, args):
            # the box dies while the handler runs: its ends of both pipes close
            for fd in holder["h"].box._STATE["fds"]:
                os.close(fd)
            holder["h"].box._STATE["fds"] = (-1, -1)
            effects.append("written")
            return "ok: written"

        h = Harness(specs={"act": Spec("act", _obj({}), act, kind="action")})
        holder["h"] = h
        h.raw(_frame({"v": 1, "id": 1, "tool": "act", "args": {}}))
        for _ in range(100):
            if h.rows:
                break
            await asyncio.sleep(0.02)
        await h.close()
        return h

    h = run(main())
    assert effects == ["written"]
    (row,) = [r for r in h.rows if r["event"] == "tool_call"]
    assert row["status"] == "ok" and row["box_gone"] is True


def test_box_gone_from_the_host_drops_the_reply():
    async def main():
        holder = {}

        def kill_then_answer(ctx, args):
            holder["h"].channel.box_gone()
            return "ok"

        h = Harness(specs={"k": Spec("k", _obj({}), kill_then_answer, blocking=False)})
        holder["h"] = h
        t = asyncio.ensure_future(h.call("k", {}))
        await asyncio.sleep(0.3)
        rows = list(h.rows)
        await h.close()  # the closing notice ends the box's wait
        await t
        return rows

    rows = run(main())
    assert rows and rows[0]["box_gone"] is True


# --- C1: the counters under a parallel flood ---------------------------------------------------------------------------


class Counter:
    """A limit checked and incremented in admit (the event loop), as the pass's counters are."""

    def __init__(self, limit):
        self.limit, self.used, self.loop_thread = limit, 0, threading.get_ident()
        self.off_loop = 0

    def admit(self, tool, kind, args):
        if threading.get_ident() != self.loop_thread:
            self.off_loop += 1
        if self.used >= self.limit:
            return general("reads_limit", limit=self.limit)
        self.used += 1
        return None


def test_a_parallel_flood_across_channels_never_passes_a_counter_limit():
    ran = []
    lock = threading.Lock()

    def work(ctx, args):
        time.sleep(0.005)
        with lock:
            ran.append(1)
        return "ok"

    async def main():
        counter = Counter(7)
        specs = {"echo": Spec("echo", _obj({"text": {"type": "string"}}), work)}
        hs = [Harness(specs=specs, counters=counter) for _ in range(8)]
        outs = await asyncio.gather(
            *(h.call("echo", {"text": str(i)}) for h in hs for i in range(12)),
        )
        for h in hs:
            await h.close()
        return counter, outs

    counter, outs = run(main())
    oks = [o for o in outs if o == "ok"]
    assert len(oks) == 7 and len(ran) == 7 and counter.used == 7
    assert counter.off_loop == 0
    assert all(getattr(o, "form", None) == "reads_limit" for o in outs if o != "ok")


# --- validation and client --------------------------------------------------------------------------------------------


def test_validation_refuses_unknown_names_at_every_depth():
    schema = _obj(
        {
            "a": {"type": "array", "items": _obj({"x": {"type": "integer"}})},
            "b": {"type": ["integer", "null"]},
            "c": {"enum": ["one", "two"]},
        },
    )
    assert tc.validate(schema, {"a": [{"x": 1}], "b": None, "c": "one"}) is None
    assert "unknown argument" in tc.validate(schema, {"a": [{"x": 1, "y": 2}]})
    assert tc.validate(schema, {"b": True}) is not None  # a boolean is not an integer
    assert tc.validate(schema, {"c": "three"}) is not None
    assert tc.validate(schema, {"zz": 1}).startswith("unknown argument")


def test_the_client_is_standard_library_only():
    import ast

    tree = ast.parse(CLIENT.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names <= {"json", "os", "threading"}


def test_pipes_are_not_inheritable_and_the_env_value_names_the_box_ends():
    async def main():
        ch = tc.Channel.open(_binding(), ("echo",), None, specs=SPECS)
        fds = (ch._req_r, ch._req_w, ch._rep_r, ch._rep_w)
        inheritable = [os.get_inheritable(fd) for fd in fds]
        value = ch.env_value()
        req, rep = ch.box_fds()
        await ch.close("ended", record=False)
        return inheritable, value, req, rep

    inheritable, value, req, rep = run(main())
    assert inheritable == [False] * 4
    assert value == f"fd:{req},{rep}"
