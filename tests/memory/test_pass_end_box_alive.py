"""A writer pass fails closed when a box of its interpreter is still alive at its end: no gate runs, the pass's
temporary folder is kept with the reason, and the pass's code is ``writer_error``. With tools in code on and off.

The interpreter is a stand-in, started through the pass's interpreter hook (``WriterPass.interpreter_start``, the
``start`` of its ``InterpreterSession``): its ``stop()`` returns, but its box may stay alive. It writes a manifest
into the pass's /memory when it starts, so a pass whose box did stop reaches the gate (the control). Keyless and
boxless.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.test_gate import MAN
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import CODE_WRITER_ERROR, PassConfig, WriterPass
from unify.settings import SETTINGS

STILL_RUNNING = "a box of this pass is still running"


class StandIn:
    """An interpreter whose cells print one line and whose box survives its stop when *survives*."""

    def __init__(self, survives: bool) -> None:
        self.survives = survives
        self.stopped = False
        self.cells = 0

    async def run_cell(self, code: str, *, call=None):
        self.cells += 1
        return SimpleNamespace(
            output="ran",
            error=None,
            notes_before=[],
            notes_after=[],
            opens=None,
            restarted=False,
            timed_out=False,
            quota=None,
        )

    async def stop(self) -> None:
        self.stopped = True

    def box_alive(self) -> bool:
        return self.survives or not self.stopped

    async def measure_quota(self):
        return None


def _starter(survives: bool, started: list):
    async def start(**kw):
        # the pass's /memory on the host: a manifest there lets a pass whose box stopped reach the gate
        (box,) = [host for host, inside in dict(kw["binds"]["rw"]).items() if inside == "/memory"]
        manifest = Path(box) / ".pass" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({**MAN, "items": [], "summary": "s"}))
        interp = StandIn(survives)
        started.append(interp)
        return interp

    return start


def _pass(tmp_path: Path, monkeypatch, turns: list, *, in_code: bool, survives: bool):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on" if in_code else "off")
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup, library_mode=LibraryConfig())
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=Turns(turns),
        config=PassConfig(
            tools_in_code=in_code,
            pass_record_root=str(tmp_path / "out"),
            max_calls=len(turns),
        ),
    )
    started: list = []
    writer.interpreter_start = _starter(survives, started)
    merged: list = []
    real_merge = gate.merge

    def merge(*args, **kwargs):
        merged.append(args)
        return real_merge(*args, **kwargs)

    gate.merge = merge
    try:
        out = asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    finally:
        kept = getattr(writer, "_temp_kept", None)
        if kept:
            shutil.rmtree(kept["path"], ignore_errors=True)
    return writer, out, merged, started, kept


def _cell(cid: str, code: str) -> dict:
    return _call(cid, "execute_code", {"code": code})


TWO_CELLS = [_cell("c1", "x = 1"), _cell("c2", "print(x)")]


def _assert_failed_closed(out, merged, started, kept) -> None:
    assert started and all(i.stopped for i in started), started
    assert merged == [], "the gate ran while a box of the pass was alive"
    assert kept and kept["reason"] == STILL_RUNNING, kept
    assert CODE_WRITER_ERROR in out.codes, out.codes
    assert not out.passed
    assert any(STILL_RUNNING in r and "the gate did not start" in r for r in out.reasons), out.reasons


@pytest.mark.parametrize("in_code", [True, False], ids=["tools-in-code", "tools-off"])
def test_a_box_alive_after_the_pass_s_stop_keeps_the_gate_from_running(tmp_path, monkeypatch, in_code):
    _, out, merged, started, kept = _pass(tmp_path, monkeypatch, TWO_CELLS, in_code=in_code, survives=True)
    assert started[0].cells == 2  # one interpreter for both cells
    _assert_failed_closed(out, merged, started, kept)


def test_a_box_alive_after_an_end_call_stopped_it_keeps_the_gate_from_running(tmp_path, monkeypatch):
    """Tools in code off: the accepted finish stops the interpreter at once; its box surviving that stop is seen at
    the pass's end, though no interpreter is running any more."""
    turns = [_cell("c1", "x = 1"), _call("f", "finish", {"summary": "done"})]
    _, out, merged, started, kept = _pass(tmp_path, monkeypatch, turns, in_code=False, survives=True)
    _assert_failed_closed(out, merged, started, kept)


@pytest.mark.parametrize("in_code", [True, False], ids=["tools-in-code", "tools-off"])
def test_a_box_that_stopped_lets_the_gate_run(tmp_path, monkeypatch, in_code):
    _, out, merged, started, kept = _pass(tmp_path, monkeypatch, TWO_CELLS, in_code=in_code, survives=False)
    assert started and all(i.stopped for i in started), started
    assert len(merged) == 1, merged
    assert not kept, kept
    assert CODE_WRITER_ERROR not in out.codes, out.codes
