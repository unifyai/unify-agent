"""The writer's ``delegate`` and ``similar`` tools: offered only when their lane or index exists, named in the brief's
Tools section, and dispatched from the pass loop (delegate calls of one turn run together through helpers.py).
Keyless: the writer and its helpers are scripted fakes.
"""

import asyncio
import json
from pathlib import Path

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
            return _msg(_tc("d", "done", {"summary": f"drafted {task}"})), "0.001"

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


def test_a_writer_turn_that_delegates_gets_the_helpers_staged_files(tmp_path):
    lane = Helpers()
    turns = [
        # two delegate calls in one turn run together; a later turn's call gets the next call id
        _msg(
            _tc("d0", "delegate", {"task": "area", "episodes": []}),
            _tc("d1", "delegate", {"task": "perimeter", "episodes": []}),
        ),
        _call("d2", "delegate", {"task": "volume", "episodes": []}),
    ]
    out, model, writer = _pass(
        tmp_path,
        turns,
        helper=lane,
        pass_record_root=str(tmp_path / "records"),
    )
    assert names(writer._tools_json) >= {"delegate"} and "similar" not in names(
        writer._tools_json,
    )
    brief = writer.messages[0]["content"]
    assert "- `delegate`:" in brief and "`similar`" not in brief
    for cid, call, task in (
        ("d0", "call-0", "area"),
        ("d1", "call-1", "perimeter"),
        ("d2", "call-2", "volume"),
    ):
        reply = model.outputs[cid]
        assert reply.startswith(f"helper {call}: ok"), reply
        assert (
            f"files under /inputs/helpers/{call}/" in reply
            and f"- candidates/{task}.py" in reply
        )
        assert f"summary: drafted {task}" in reply
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
        r for r in rows if r["agent"] == "writer" and r["tool"] == "delegate"
    ]
    helper_rows = [r for r in rows if r["agent"] == "helper"]
    assert len(writer_rows) == 3 and all(r["status"] == "ok" for r in writer_rows)
    assert {r["helper_call"] for r in helper_rows} == {"call-0", "call-1", "call-2"}
    assert {r["tool"] for r in helper_rows} == {"stage", "done"}


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
            return "refused: " + similar.SPEC.refusals["unknown_episode"].format(
                episode="e9",
            )
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
