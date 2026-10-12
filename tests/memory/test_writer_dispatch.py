"""The writer's ``delegate`` and ``similar`` tools: offered only when their lane or index exists, named in the brief's
Tools section, and dispatched from the pass loop (delegate returns at once; the helpers' messages arrive on the
pass's record). Every way out of the pass ends its helpers. Keyless: the writer and its helpers are scripted fakes.
"""

import asyncio
import json
from pathlib import Path

import pytest

from unify.memory import helpers
from unify.memory import library_prompts as lp
from unify.memory import similar
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import writer_tools
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
    _, _, writer = _writer(tmp_path, model, library_mode=True, **cfg)
    writer.load = EPS.__getitem__
    if helper is not None:
        writer.helper_turn = helper.factory
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"),
    )
    return out, model, writer


def test_delegate_and_similar_are_offered_only_with_their_lane_and_index():
    plain = names(writer_tools(library_mode=True))
    assert "delegate" not in plain and "similar" not in plain
    both = names(writer_tools(library_mode=True, helper=True, similar=True))
    assert {"delegate", "similar"} <= both and both - plain == {"delegate", "similar"}
    assert "delegate" not in names(
        writer_tools(library_mode=True, helper=True, role="curate"),
    )
    assert "similar" not in names(
        writer_tools(library_mode=True, similar=True, role="curate"),
    )
    # the brief's Tools section names exactly the tools offered
    brief = lp.write_brief_now(helper=True, similar=True)
    assert (
        "- `delegate`:" in brief
        and "- `similar`:" in brief
        and "with `delegate`" in brief
    )
    bare = lp.write_brief_now()
    assert "`delegate`" not in bare and "`similar`" not in bare


def test_the_helper_gets_its_readers_from_the_registry_and_never_delegate():
    assert names(helpers.helper_tools()) == HELPER_TOOLS
    assert names(helpers.helper_tools(similar=False)) == HELPER_TOOLS - {"similar"}


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
    assert names(writer._tools_json) >= {"delegate", "wait", "messages", "send", "who", "reply", "stop"}
    assert "similar" not in names(writer._tools_json)
    brief = writer.messages[0]["content"]
    assert "- `delegate`:" in brief and "`similar`" not in brief and "# The record\n" in brief
    for cid, call in (("d0", "call-0"), ("d1", "call-1"), ("d2", "call-2")):
        assert model.outputs[cid].startswith(f"helper {call} started"), model.outputs[cid]
    waited = "\n".join(model.outputs[w] for w in ("w0", "w1", "w2", "w3"))
    for call, task in (("call-0", "area"), ("call-1", "perimeter"), ("call-2", "volume")):
        assert f"{call} (done): helper {call}: ok" in waited, waited
        assert f"files under /inputs/helpers/{call}/" in waited and f"- candidates/{task}.py" in waited
        assert f"summary: drafted {task}" in waited
    assert "- call-0 (helper, replied): area" in model.outputs["who"]
    assert [h["call"] for h in out.helpers] == ["call-0", "call-1", "call-2"]
    assert all(h["files"] == 1 and h["status"] == "ok" for h in out.helpers)
    # no similar index: the helpers are not offered similar either
    assert all(t == HELPER_TOOLS - {"similar"} for t in lane.tools)
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
    assert [r["messaging"] for r in writer_rows] == [{"helper": f"call-{i}"} for i in range(3)]
    assert all(r["name"] == r["helper_call"] for r in helper_rows)
    waits = [r for r in rows if r.get("tool") == "wait"]
    assert len(waits) == 4 and all(
        r["messaging"]["outcome"] in ("message", "nothing_can_arrive") and r["messaging"]["seconds"] >= 0
        for r in waits
    )
    (who,) = [r for r in rows if r.get("tool") == "who"]
    assert [a["name"] for a in who["messaging"]["roster"]] == ["writer", "call-0", "call-1", "call-2"]
    # the pass's end: every helper ended, and where its record is
    (end,) = [r for r in rows if r["event"] == "pass_end"]
    assert rows[-1] is end or rows[-1] == end
    assert end["helpers_closed"] == {"call-0": "ended", "call-1": "ended", "call-2": "ended"}
    record = tmp_path / "records" / "p1" / "record.jsonl"
    assert end["record"] == str(record) and end["record_error"] is None
    entries = [json.loads(ln) for ln in record.read_text().splitlines()]
    assert len(entries) == end["record_entries"] == len(writer._pass_record.record.entries)
    assert [e["seq"] for e in entries] == list(range(1, len(entries) + 1))
    assert any(e["author"] == "call-2" and e["kind"] == "reply" for e in entries)
    cursors = [json.loads(ln) for ln in (tmp_path / "records" / "p1" / "record.cursors.jsonl").read_text().splitlines()]
    assert {"agent": "writer"}.items() <= cursors[-1].items() or any(c["agent"] == "writer" for c in cursors)
    assert {r["helper_call"] for r in helper_rows} == {"call-0", "call-1", "call-2"}
    assert {r["tool"] for r in helper_rows} == {"stage", "done"}
    # each helper's transcript is recorded with the writer's, every line naming the pass and the helper; the
    # pass's record follows
    lines = [json.loads(ln) for ln in writer.transcript("p1")]
    assert all(ln["pass_id"] == "p1" for ln in lines)
    turns_of = [ln for ln in lines if ln.get("helper") == "call-1" and "message" in ln]
    assert turns_of and all(ln["agent"] == "helper" for ln in turns_of)
    assert any("candidates/perimeter.py" in json.dumps(ln["message"]) for ln in turns_of)
    record = [ln["record"] for ln in lines if "record" in ln]
    assert any(r["author"] == "call-1" and r["kind"] == "reply" for r in record)


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
                return _msg(_tc("s", "stage", {"path": "notes.md", "text": "a lesson"})), "0.001"
            return _msg(_tc(f"o{self.calls}", "overview", {"episode": "e1"})), "0.001"

        return turn


def _ended_with(writer, out, status):
    assert writer._pass_record.closed == {"call-0": "ended"}
    assert writer._pass_record.record.participants["call-0"].state != "running"
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
    assert [e.text for e in stop] == ["@call-0 stop: the writer finished"]


def test_a_writer_error_ends_its_helpers(tmp_path):
    lane = Slow()

    class Failing(Turns):
        async def __call__(self, messages, tools):
            if not self.queue:
                await asyncio.sleep(0.2)
                raise RuntimeError("the writer's model failed")
            return await super().__call__(messages, tools)

    model = Failing([_call("d", "delegate", {"task": "area", "episodes": []})])
    _, _, writer = _writer(tmp_path, model, library_mode=True)
    writer.load = EPS.__getitem__
    writer.helper_turn = lane.factory
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1", "e2"], False), "p1"))
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
    _, _, writer = _writer(tmp_path, model, library_mode=True)
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
    assert writer._pass_record.closed == {"call-0": "ended"}
    assert writer._helper_rows[0]["status"] == "stopped: the writer finished"
    assert writer._helper_rows[0]["stopped_by"] == "finish"


def test_without_a_lane_or_index_neither_tool_is_offered_or_run(tmp_path):
    turns = [
        _msg(
            _tc("d", "delegate", {"task": "area", "episodes": []}),
            _tc("s", "similar", {"code": "x = 1"}),
        ),
    ]
    out, model, writer = _pass(tmp_path, turns)
    offered = names(writer._tools_json)
    assert "delegate" not in offered and "similar" not in offered
    assert model.outputs["d"].startswith("refused: unknown tool 'delegate'")
    assert model.outputs["s"].startswith("refused: unknown tool 'similar'")
    assert out.helpers == []
    brief = writer.messages[0]["content"]
    assert "`delegate`" not in brief and "`similar`" not in brief


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
    _, model, writer = _pass(tmp_path, turns, similar_index=str(index))
    assert "similar" in names(writer._tools_json)
    assert "- `similar`:" in writer.messages[0]["content"]
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
