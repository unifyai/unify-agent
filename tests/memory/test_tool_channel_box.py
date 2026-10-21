"""Tools in code end to end in real boxes: a scripted writer pass whose cells run in the persistent confined
interpreter (``BoxInterpreter``) and call ``tools.<name>(...)`` through the real tool channel, with a real evidence
store, gate and blob store; a helper started with ``tools.delegate`` in its own box; and the cell clock, which pauses
while the host answers a call. Keyless; the boxes need bubblewrap and prlimit, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tests.memory.test_gate import ITEM, KIT, MAN, MOD, TEST
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME
from unify.memory import sandbox_run as sr
from unify.memory import tool_handlers
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap and prlimit required",
)


def _descendants(pid: int) -> list[int]:
    """Every live process below *pid* (from /proc)."""
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] == "Z":
            continue
        kids.setdefault(int(fields[1]), []).append(int(d))
    out, stack = [], [pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def _cell(cid: str, code: str) -> dict:
    return _call(cid, "execute_code", {"code": code})


def _writer(tmp_path: Path, turns: list, **cfg) -> tuple[WriterPass, Turns, EvidenceStore]:
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup, library_mode=LibraryConfig())
    model = Turns(turns)
    cfg.setdefault("max_calls", 30)
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(
            pass_record_root=str(tmp_path / "out"),
            **cfg,
        ),
    )
    return writer, model, ev


def _rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "out" / "p1" / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


READERS = """
toc = tools.read_episode("e1", parts=["request"])
print("EPISODE", len(toc) > 0, type(toc).__name__)
print("LISTING", "batch_map.json" in tools.read("/inputs"))
print("GREP", type(tools.grep("venmo", "/inputs")).__name__)
try:
    tools.read("/etc/passwd")
except tools.Refused as exc:
    print("REFUSED", exc.form)
"""


def _files_cell() -> str:
    manifest = json.dumps({**MAN, "items": [ITEM], "summary": "s"})
    return "\n".join(
        [
            f"print(tools.write('unify_memory_testkit.py', {KIT!r}))",
            f"print(tools.write('env/venmo/tests/test_me.py', {TEST!r}))",
            f"print(tools.write('env/venmo/__init__.py', {MOD!r}))",
            f"print(tools.write('.pass/manifest.json', {manifest!r}))",
            "try:",
            "    tools.fixture('e1', 'env/venmo/tests/data/none.json', source='no-such-source')",
            "except (tools.Refused, tools.ToolError) as exc:",
            "    print('FIXTURE', type(exc).__name__, exc.form)",
            "print('CHECK', tools.check()[:400])",
        ],
    )


DELEGATE = """
print("DELEGATE", tools.delegate("draft an area function", episodes=[]))
print("WAIT", tools.wait()[:300])
"""

HELPER_CELL = """
print("STAGED", tools.stage("candidates/area.py", "def area(w, h):\\n    return w * h\\n"))
tools.done("drafted area", [])
print("NOT REACHED")
"""


def _helper_factory(seen: list):
    def factory(label=None):
        async def turn(messages, tools):
            seen.append([t["function"]["name"] for t in tools])
            return _cell(f"h{len(seen)}", HELPER_CELL), "0.001"

        return turn

    return factory


@needs_box
def test_a_scripted_writer_pass_runs_its_tools_from_cells_in_a_real_box(tmp_path):
    turns = [
        _cell("c1", READERS),
        _cell("c2", _files_cell()),
        _cell("c3", DELEGATE),
        _cell("c4", "tools.finish('stored the me function')\nprint('NOT REACHED')"),
    ]
    writer, model, ev = _writer(tmp_path, turns, edit_tools=True)
    helper_tools: list = []
    writer.helper_turn = _helper_factory(helper_tools)
    started = time.monotonic()
    out = asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    took = time.monotonic() - started
    rows = _rows(tmp_path)
    summary = [(r.get("agent"), r.get("tool"), r.get("status"), (r.get("refusal") or {}).get("form")) for r in rows]
    assert out.finished, (out.reasons, model.outputs, summary)
    assert out.passed, (out.reasons, model.outputs, summary)
    assert took < 600
    # the tool results the conversation holds (the last one is never sent back to the model)
    outputs = {m["tool_call_id"]: m["content"] for m in writer.messages if m.get("role") == "tool"}
    c1, c2, c3, c4 = (outputs[k] for k in ("c1", "c2", "c3", "c4"))
    assert "EPISODE True Result" in c1 and "LISTING True" in c1 and "GREP Result" in c1, c1
    assert "REFUSED outside_roots" in c1, c1
    assert "ok: created env/venmo/__init__.py" in c2 and "FIXTURE" in c2, c2
    assert "CHECK ok" in c2, c2
    assert "DELEGATE helper call-0 started" in c3 and "WAIT" in c3, c3
    assert c4.splitlines()[-1] == "ok" and "NOT REACHED" not in c4, c4
    # the event rows: one per cell from the model, one per in-code call from the channel, each naming its cell
    calls = [r for r in rows if r["event"] == "tool_call"]
    in_code = [r for r in calls if r.get("via") == "channel" and r["agent"] == "writer"]
    cells = {r["call"] for r in calls if r["tool"] == "execute_code" and r["agent"] == "writer"}
    assert {r["tool"] for r in in_code} >= {
        "read_episode",
        "read",
        "grep",
        "write",
        "fixture",
        "check",
        "delegate",
        "wait",
        "finish",
    }, summary
    assert all(r["cell"] in cells and r["call"] > r["cell"] for r in in_code)
    assert next(r for r in in_code if r["tool"] == "check")["status"] == "ok"
    # the helper ran in its own box: only execute_code on its requests, its staged file and its done
    assert helper_tools and all(t == ["execute_code"] for t in helper_tools)
    helper = [r for r in calls if r["agent"] == "helper" and r.get("via") == "channel"]
    assert [(r["tool"], r["status"]) for r in helper] == [("stage", "ok"), ("done", "ok")], summary
    # nothing outlives the pass: no box and no handler thread
    assert not writer._in_code_threads.alive()
    assert _descendants(os.getpid()) == []


def _waiting(monkeypatch, seconds: float):
    """``wait`` as a host call of *seconds* (it stands for any call the host takes that long to answer)."""

    async def wait(ctx, args):
        await asyncio.sleep(seconds)
        return "no new messages"

    monkeypatch.setattr(tool_handlers, "wait", wait)


CLOCK_CELL = """
import time
started = time.monotonic()
try:
    print("WAIT", tools.wait())
except tools.NotRun as exc:
    print("NOTRUN", exc)
print("SLEPT", round(time.monotonic() - started))
"""


@needs_box
def test_the_cell_clock_pauses_while_the_host_answers_a_call(tmp_path, monkeypatch):
    _waiting(monkeypatch, 6.0)
    writer, model, _ = _writer(tmp_path, [_cell("c1", CLOCK_CELL)], cell_timeout_s=3.0, max_calls=2)
    asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    text = model.outputs["c1"]
    # six seconds of host time in a cell limited to three: the cell is not killed for it
    assert "WAIT no new messages" in text and "SLEPT 6" in text, text
    assert "timed out" not in text, text


@needs_box
def test_host_time_in_a_cell_is_bounded_by_the_pass_wall_clock(tmp_path, monkeypatch):
    _waiting(monkeypatch, 1000.0)
    deadline = 20.0
    writer, model, _ = _writer(
        tmp_path,
        [_cell("c1", CLOCK_CELL)],
        cell_timeout_s=3.0,
        max_calls=2,
        deadline_s=deadline,
    )
    started = time.monotonic()
    asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    took = time.monotonic() - started
    text = model.outputs.get("c1", "")
    rows = [r for r in _rows(tmp_path) if r["event"] == "tool_call" and r["tool"] == "wait"]
    assert rows and rows[0]["status"] == "refused" and rows[0]["refusal"]["form"] == "stopped", rows
    assert took < deadline + 30, took
    if text:  # the model saw the cell only when the pass had a call left after it
        assert "NOTRUN not run: timed out" in text, text
    assert not writer._in_code_threads.alive()
    assert _descendants(os.getpid()) == []
