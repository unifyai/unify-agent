"""The writer pass's ``pass_end`` event with tools in code (``PassConfig.tools_in_code``) says how many tool handler
threads were still alive when the pass reached its gate (``handlers_alive_at_gate``): 0 when the gate ran, and
more than 0, together with ``temp_kept``, when a handler thread was alive and the pass failed closed.

Keyless and boxless: the interpreter is the in-process stand-in of ``tests/memory/test_tool_handlers.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import threading
from pathlib import Path

from tests.memory.test_tool_handlers import EPS, FakeBox, ToolsSeen, _cell, _in_code_pass
from tests.memory.test_writer_pass import _writer
from unify.common import tool_channel as tc
from unify.memory import tool_handlers
from unify.memory.trigger import PassRequest
from unify.memory.writer_tools import Reply

# ── ASSUMED NAMES (the integrator adapts them here, in one place) ────────────────────────────────────────────────
HANDLERS_FIELD = "handlers_alive_at_gate"
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


def _end(rows: list[dict]) -> dict:
    (end,) = [r for r in rows if r["event"] == "pass_end"]
    return end


def test_a_pass_whose_gate_ran_says_no_handler_was_alive(tmp_path):
    turns = [_cell("c1", ("read", {"path": "/inputs"}), ("who", {}))]
    out, _model, _writer_, rows = _in_code_pass(tmp_path, turns, max_calls=2)
    assert not any("the gate did not start" in r for r in out.reasons), out.reasons
    end = _end(rows)
    assert end[HANDLERS_FIELD] == 0
    assert isinstance(end[HANDLERS_FIELD], int) and not isinstance(end[HANDLERS_FIELD], bool)
    assert "temp_kept" not in end


def test_a_pass_with_a_handler_alive_counts_it_beside_temp_kept(tmp_path, monkeypatch):
    release = threading.Event()
    seen = {}

    def stuck_read(ctx, args):
        seen["box"] = str(ctx.state.box)
        release.wait(30)
        return Reply("late")

    monkeypatch.setattr(tool_handlers, "read", stuck_read)
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
        library_mode=True,
        tools_in_code=True,
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
        assert not out.passed and any("the gate did not start" in r for r in out.reasons), out.reasons
        end = _end(rows)
        assert end["temp_kept"]["reason"] == "a tool handler of this pass is still running"
        assert isinstance(end[HANDLERS_FIELD], int) and end[HANDLERS_FIELD] > 0, end
        # a count only: no thread names, paths or arguments
        assert not isinstance(end[HANDLERS_FIELD], bool)
    finally:
        release.set()
    if "box" in seen:
        shutil.rmtree(Path(seen["box"]).parent, ignore_errors=True)
