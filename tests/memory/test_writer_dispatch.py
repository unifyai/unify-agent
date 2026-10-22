"""The writer's ``delegate`` and ``similar`` tools: offered only when their lane or index exists, named in the brief's
Tools section, and dispatched from the pass loop (delegate returns at once; the helpers' messages arrive on the
pass's record). Every way out of the pass ends its helpers. Keyless: the writer and its helpers are scripted fakes.
"""

import asyncio
import json
from pathlib import Path

import pytest

from unify.memory import similar
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import Turns, _call, _writer

CANDIDATE = 'def area(w, h):\n    """Area of a w by h rectangle."""\n    return w * h\n'
HELPER_TOOLS = {
    "read",
    "grep",
    "overview",
    "read_episode",
    "similar",
    "stage",
    "done",
    "execute_code",
    "send",
    "ask",
    "messages",
    "wait",
    "who",
}


def names(tools):
    return {t["function"]["name"] for t in tools}


def _tc(cid, name, args):
    from tests.memory.in_code_calls import as_cell

    name, args = as_cell(name, args)
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _msg(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class Helpers:
    """A fresh scripted helper per delegate call: it stages ``candidates/<task>.py`` and hands back."""

    def __init__(self):
        self.labels: list[str] = []
        self.tools: list[set[str]] = []

    def factory(self, label=None):
        self.labels.append(label)
        state = {"step": 0}

        async def turn(messages, tools):
            self.tools.append(names(tools))
            task = messages[1]["content"].split("\n")[1]
            state["step"] += 1
            if state["step"] == 1:
                return (
                    _msg(
                        _tc(
                            "s",
                            "stage",
                            {"path": f"candidates/{task}.py", "text": CANDIDATE},
                        ),
                    ),
                    "0.001",
                )
            return (
                _msg(_tc("d", "done", {"summary": f"drafted {task}", "claims": []})),
                "0.001",
            )

        return turn


def _pass(tmp_path, turns, *, helper=None, **cfg):
    model = Turns(turns)
    _, _, writer = _writer(tmp_path, model, **cfg)
    writer.load = EPS.__getitem__
    if helper is not None:
        writer.helper_turn = helper.factory
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"),
    )
    return out, model, writer


def test_a_writer_that_delegates_gets_the_helpers_messages_and_staged_files(tmp_path):
    lane = Helpers()
    turns = [
        # delegate returns at once; a later turn's call gets the next call id
        _msg(
            _tc("d0", "delegate", {"task": "area", "episodes": []}),
            _tc("d1", "delegate", {"task": "perimeter", "episodes": []}),
        ),
        _call("d2", "delegate", {"task": "volume", "episodes": []}),
        _call("w0", "wait", {}),
        _call("w1", "wait", {}),
        _call("w2", "wait", {}),
        _call("w3", "wait", {}),
        _call("who", "who", {}),
    ]
    out, model, writer = _pass(
        tmp_path,
        turns,
        helper=lane,
        pass_record_root=str(tmp_path / "records"),
    )
    # rev 2: the writer's one model-API tool is execute_code; its other tools are calls from its cells, by the
    # registry's allow-list for the pass's configuration (a helper lane, no similar index)
    from unify.common import tool_registry as R
    from unify.memory.library_prompts import tool_flags

    assert names(writer._tools_json) == {"execute_code"}
    allowed = set(R.allow_list("writer", tool_flags(helper=True, similar=False)))
    assert allowed >= {
        "delegate",
        "wait",
        "messages",
        "send",
        "who",
        "reply",
        "stop",
    }
    assert "similar" not in allowed
    # rev 2: the brief names the in-code calls it relies on (no Tools section), and the run record's rules open the
    # first message
    brief = writer.messages[0]["content"]
    assert "`tools.delegate`" in brief and "tools.similar" not in brief
    assert R.COMMON_SECTION in writer.messages[1]["content"]
    for cid, call in (("d0", "call-1"), ("d1", "call-2"), ("d2", "call-3")):
        assert model.outputs[cid].startswith(f"helper {call} started"), model.outputs[
            cid
        ]
    # each helper's reply reaches the writer once it is on the record: in the output of a wait, or, when it lands
    # between the writer's turns, in the record's block at the next turn boundary (which one depends on timing only)
    received = "\n".join(
        str(m.get("content")) for m in model.seen if m.get("role") in ("tool", "user")
    )
    for call, task in (
        ("call-1", "area"),
        ("call-2", "perimeter"),
        ("call-3", "volume"),
    ):
        assert f"{call} (reply): helper {call}: ok" in received, received  # rev 2 tags a reply "(reply)"
        assert (
            f"files under /inputs/helpers/{call}/" in received
            and f"- candidates/{task}.py" in received
        )
        assert f"summary: drafted {task}" in received
    # rev 2: who lists the agents on the record now; the three helpers had replied and ended, so they are counted
    assert "(3 agents of this run have ended)" in model.outputs["who"]
    assert "- writer (writer, running)" in model.outputs["who"] and "- call-" not in model.outputs["who"]
    assert [h["call"] for h in out.helpers] == ["call-1", "call-2", "call-3"]
    assert all(h["files"] == 1 and h["status"] == "ok" for h in out.helpers)
    # no similar index: the helpers are not offered similar either
    # rev 2: a helper's one model-API tool is execute_code; its in-code allow-list (the pass's flags: no index) has
    # every helper tool but similar, with the shell and the examples check rev 2 adds
    from unify.memory.helpers import helper_flags

    assert lane.tools and all(t == {"execute_code"} for t in lane.tools)
    assert set(R.allow_list("helper", helper_flags(False))) == (HELPER_TOOLS - {"similar", "execute_code"}) | {
        "sh",
        "check_examples",
    }
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "records" / "p1" / "tool-events.jsonl")
        .read_text()
        .splitlines()
    ]
    writer_rows = [
        r for r in rows if r["agent"] == "writer" and r.get("tool") == "delegate"
    ]
    helper_rows = [r for r in rows if r["agent"] == "helper"]
    assert len(writer_rows) == 3 and all(r["status"] == "ok" for r in writer_rows)
    # one helper identity everywhere: delegate's row, the helper's rows, the record and the output folder
    assert all(r["name"] == "writer" for r in rows if r["agent"] == "writer")
    assert [r["messaging"] for r in writer_rows] == [
        {"helper": f"call-{i}"} for i in range(1, 4)
    ]
    assert all(r["name"] == r["helper_call"] for r in helper_rows)
    waits = [r for r in rows if r.get("tool") == "wait"]
    assert len(waits) == 4 and all(
        r["messaging"]["outcome"] in ("message", "nothing_can_arrive")
        and r["messaging"]["seconds"] >= 0
        for r in waits
    )
    (who,) = [r for r in rows if r.get("tool") == "who"]
    # the roster who showed: the agents on the record then (the helpers had ended, as who said)
    assert [a["name"] for a in who["messaging"]["roster"]] == ["writer"]
    # the pass's end: every helper ended, and where its record is
    (end,) = [r for r in rows if r["event"] == "pass_end"]
    assert rows[-1] is end or rows[-1] == end
    assert end["helpers_closed"] == {
        "call-1": "ended",
        "call-2": "ended",
        "call-3": "ended",
    }
    # rev 2: the pass is on the run's one record, in the run home's harness-only state (not beside the pass's events)
    from unify.agents import run_record as rr

    record = rr.record_path(tmp_path / "run-home")
    # (the run record is written as each entry is posted, so the end event carries no separate write error)
    assert end["record"] == str(record) and "record_error" not in end
    entries = [json.loads(ln) for ln in record.read_text().splitlines()]
    assert (
        len(entries) == end["record_entries"] == len(writer._pass_record.record.entries)
    )
    assert [e["seq"] for e in entries] == list(range(1, len(entries) + 1))
    assert any(e["author"] == "call-3" and e["kind"] == "reply" for e in entries)
    cursors = [
        json.loads(ln)
        for ln in rr.deliveries_path(tmp_path / "run-home")
        .read_text()
        .splitlines()
    ]
    assert {"agent": "writer"}.items() <= cursors[-1].items() or any(
        c["agent"] == "writer" for c in cursors
    )
    assert {r["helper_call"] for r in helper_rows} == {"call-1", "call-2", "call-3"}
    # in code each cell is an execute_code row and each call it makes has its own row
    assert {r["tool"] for r in helper_rows} == {"execute_code", "stage", "done"}
    # each helper's transcript is recorded with the writer's, every line naming the pass and the helper; the
    # pass's record follows
    lines = [json.loads(ln) for ln in writer.transcript("p1")]
    assert all(ln["pass_id"] == "p1" for ln in lines)
    turns_of = [ln for ln in lines if ln.get("helper") == "call-2" and "message" in ln]
    assert turns_of and all(ln["agent"] == "helper" for ln in turns_of)
    assert any(
        "candidates/perimeter.py" in json.dumps(ln["message"]) for ln in turns_of
    )
    record = [ln["record"] for ln in lines if "record" in ln]
    assert any(r["author"] == "call-2" and r["kind"] == "reply" for r in record)
    # and each helper's transcript is stored beside the pass's events, named in the end event
    (end,) = [r for r in rows if r["event"] == "pass_end"]
    stored = {m["agent"]: m for m in end["helper_transcripts"]}
    assert set(stored) == {"call-1", "call-2", "call-3"} and not any(
        "error" in m for m in stored.values()
    )
    # whole, in sha-checked chunk files (transcripts.record_transcript, the one way every agent stores its own)
    from unify.memory import transcripts

    kept = transcripts.read_transcript(tmp_path / "records", "p1", "call-2")
    assert len(kept) == stored["call-2"]["lines"] and all(
        json.loads(x)["helper"] == "call-2" for x in kept
    )
    assert stored["call-2"]["manifest"] == "transcript-call-2.json"
    assert all(
        c["name"].startswith("transcript-call-2-") for c in stored["call-2"]["chunks"]
    )


class Slow:
    """A helper that stages a note, then keeps reading, each model call taking *gap* seconds."""

    def __init__(self, gap=0.05):
        self.gap, self.calls = gap, 0

    def factory(self, label=None):
        async def turn(messages, tools):
            self.calls += 1
            first = self.calls == 1
            await asyncio.sleep(self.gap)
            if first:
                return (
                    _msg(_tc("s", "stage", {"path": "notes.md", "text": "a lesson"})),
                    "0.001",
                )
            return _msg(_tc(f"o{self.calls}", "overview", {"episode": "e1"})), "0.001"

        return turn


def _ended_with(writer, out, status):
    assert writer._pass_record.closed == {"call-1": "ended"}
    assert writer._pass_record.record.participants["call-1"].state != "running"
    assert [h["status"] for h in out.helpers] == [status]
    assert out.helpers[0]["files"] == 1


def test_finishing_ends_a_helper_still_running_and_keeps_what_it_staged(tmp_path):
    lane = Slow()
    turns = [
        _call("d", "delegate", {"task": "area", "episodes": []}),
        _call("x", "who", {}),
        _call("y", "who", {}),
        _call("f", "finish", {"summary": "done"}),
    ]
    out, model, writer = _pass(tmp_path, turns, helper=lane)
    _ended_with(writer, out, "stopped: the writer finished")
    calls = lane.calls
    assert calls >= 1
    stop = [e for e in writer._pass_record.record.entries if e.kind == "cancel"]
    assert [e.text for e in stop] == ["@call-1 stop: the writer finished"]


def test_a_writer_error_ends_its_helpers(tmp_path):
    lane = Slow()

    class Failing(Turns):
        async def __call__(self, messages, tools):
            if not self.queue:
                await asyncio.sleep(0.2)
                raise RuntimeError("the writer's model failed")
            return await super().__call__(messages, tools)

    model = Failing([_call("d", "delegate", {"task": "area", "episodes": []})])
    _, _, writer = _writer(tmp_path, model)
    writer.load = EPS.__getitem__
    writer.helper_turn = lane.factory
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"),
    )
    assert not out.passed
    _ended_with(writer, out, "stopped: the writer finished")


def test_a_cancelled_pass_ends_its_helpers_before_it_goes(tmp_path):
    lane = Slow()
    entered = []

    class Hangs(Turns):
        async def __call__(self, messages, tools):
            if not self.queue:
                entered.append(True)
                await asyncio.sleep(3600)
            return await super().__call__(messages, tools)

    model = Hangs([_call("d", "delegate", {"task": "area", "episodes": []})])
    _, _, writer = _writer(tmp_path, model)
    writer.load = EPS.__getitem__
    writer.helper_turn = lane.factory

    async def go():
        task = asyncio.ensure_future(
            writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"),
        )
        while not entered or lane.calls < 2:
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # no helper task outlives the pass
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert not [t for t in others if t.get_name().startswith("unify-agents:")]

    asyncio.run(go())
    assert writer._pass_record.closed == {"call-1": "ended"}
    assert writer._helper_rows[0]["status"] == "stopped: the writer finished"
    assert writer._helper_rows[0]["stopped_by"] == "finish"


def test_without_a_lane_or_index_neither_tool_is_offered_or_run(tmp_path):
    turns = [
        _msg(
            _tc("d", "delegate", {"task": "area", "episodes": []}),
            _tc("s", "similar", {"code": "x = 1"}),
        ),
    ]
    from unify.common import tool_registry as R
    from unify.memory.library_prompts import tool_flags

    out, model, writer = _pass(tmp_path, turns)
    # rev 2: neither is on the writer's in-code allow-list without its lane or index, and a cell's call of either is
    # answered by its tools module (the stub has no such function)
    offered = set(R.allow_list("writer", tool_flags(helper=False, similar=False)))
    assert names(writer._tools_json) == {"execute_code"}
    assert "delegate" not in offered and "similar" not in offered
    assert model.outputs["d"].startswith("tools.delegate isn't available in your role.")
    assert model.outputs["s"].startswith("tools.similar isn't available in your role.")
    assert out.helpers == []
    brief = writer.messages[0]["content"]
    assert "delegate" not in brief and "similar" not in brief


def test_similar_dispatch_returns_the_handlers_answer(tmp_path, monkeypatch):
    seen = []

    def handle(args, ctx):
        seen.append((args, ctx))
        if args.get("episode") == "e9":
            return similar._refuse("unknown_episode", episode="e9")
        return "whole cell: e2 cell 0 (1 time)"

    monkeypatch.setattr(similar, "handle", handle)
    index = tmp_path / "similar-index"
    turns = [
        _call("s", "similar", {"code": "x = 1"}),
        _call("u", "similar", {"episode": "e9", "cell": 0}),
    ]
    from unify.common import tool_registry as R
    from unify.memory.library_prompts import tool_flags

    _, model, writer = _pass(tmp_path, turns, similar_index=str(index))
    # rev 2: offered in code with the index, and the brief names the call
    assert "similar" in R.allow_list("writer", tool_flags(helper=False, similar=True))
    assert "`tools.similar`" in writer.messages[0]["content"]
    assert model.outputs["s"] == "whole cell: e2 cell 0 (1 time)"
    assert model.outputs["u"] == "refused: no recorded episode e9."
    assert seen[0][0] == {"code": "x = 1"} and seen[0][1].index_dir == Path(index)
    rows = [r for r in writer.tool_events if r["tool"] == "similar"]
    assert rows[0]["status"] == "ok" and rows[0]["refusal"] is None
    assert (
        rows[1]["status"] == "refused"
        and rows[1]["refusal"]["form"] == "unknown_episode"
    )


def test_a_refused_similar_call_leaves_messages_a_model_call_can_copy(
    tmp_path,
    monkeypatch,
):
    """A refused lookup used to leave its form-carrying text in the messages, which the model call's deep copy
    could not rebuild (TypeError); tool messages now hold plain text."""
    import copy

    turns = [_call("u", "similar", {"cell": 0})]  # no episode: refused
    _, model, writer = _pass(
        tmp_path,
        turns,
        similar_index=str(tmp_path / "similar-index"),
    )
    assert model.outputs["u"].startswith("refused: ")
    tool_messages = [m for m in writer.messages if m.get("role") == "tool"]
    assert tool_messages and all(type(m["content"]) is str for m in tool_messages)
    copy.deepcopy(writer.messages)
