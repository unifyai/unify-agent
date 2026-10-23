"""The memory agents' interpreters keep their state between cells, and ``memory`` and ``help`` are bound in every cell.
The end calls stop the interpreter before anything is judged, the restart cases say so in the cell's output, a quota
breach stops it, and a library module changed on disk is imported anew. Keyless; the boxes need bubblewrap and
prlimit, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from tests.memory.test_helpers import EPS, _ctx, _msg, _tc
from tests.memory.test_helper_messaging import Steps
from tests.memory.test_tool_channel_box import _descendants
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME
from unify.memory import helpers, messaging
from unify.memory import sandbox_run as sr
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.tool_handlers import RESTARTED
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap and prlimit required",
)

_ = EPS  # the helper context reads the shared recorded episodes


def _cell(cid: str, code: str) -> dict:
    return _call(cid, "execute_code", {"code": code})


class Watched(Turns):
    """The scripted writer, which also records how many processes run below the test before each model call (a
    live interpreter box is one or more)."""

    def __init__(self, messages, **kw):
        super().__init__(messages, **kw)
        self.boxes: list[int] = []

    async def __call__(self, messages, tools):
        self.boxes.append(len(_descendants(os.getpid())))
        return await super().__call__(messages, tools)


def _writer(tmp_path: Path, turns: list, **cfg) -> tuple[WriterPass, Watched]:
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup, library_mode=LibraryConfig())
    model = Watched(turns)
    cfg.setdefault("max_calls", 12)
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), 
            pass_record_root=str(tmp_path / "out"),
            **cfg,
        ),
    )
    return writer, model


def _run(writer: WriterPass):
    return asyncio.run(writer.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))


def _end() -> list:
    """The pass ends from a cell."""
    return [_cell("end", "tools.finish('done')")]


# --- state between cells, per role and per switch ------------------------------------------------------------------


@needs_box
def test_a_writer_cell_sees_the_last_cells_variables(tmp_path, monkeypatch):
    turns = [_cell("c1", "x = 41"), _cell("c2", "print(x + 1)"), *_end()]
    writer, model = _writer(tmp_path, turns)
    _run(writer)
    out = model.outputs["c2"]
    assert out.splitlines()[0] == "42", out
    assert _descendants(os.getpid()) == []


def _helper_steps(cells: list[tuple[str, str]]) -> list:
    steps = [_msg(_tc(cid, "execute_code", {"code": code})) for cid, code in cells]
    steps.append(_msg(_tc("d", "execute_code", {"code": "tools.done('s', [])"})))
    return steps


def _helper(tmp_path: Path, cells: list[tuple[str, str]], *, library: str | None = None) -> dict:
    """One helper task running *cells*; its tool outputs by call id."""
    model = Steps({"t": _helper_steps(cells)})
    ctx = _ctx(tmp_path, model)
    if library is not None:
        (ctx.library / "memory").mkdir()
        (ctx.library / "memory" / "__init__.py").write_text(library)

    async def go():
        await helpers.handle({"task": "t", "episodes": ["e1"]}, ctx)
        await helpers.result(ctx, "call-1")  # rev 2 names helpers call-1, call-2, ...
        await messaging.close(ctx.pass_record, "the writer finished")

    asyncio.run(go())
    return model.outputs("t")


@needs_box
def test_a_helper_cell_sees_the_last_cells_variables(tmp_path, monkeypatch):
    out = _helper(tmp_path, [("c1", "x = 41"), ("c2", "print(x + 1)")])["c2"]
    assert out.splitlines()[0] == "42", out
    assert _descendants(os.getpid()) == []


# --- the names bound in every cell ---------------------------------------------------------------------------------

BOUND = (
    "print('TYPE', type(memory).__name__)\n"
    "print('INDEX', memory.index() is not None)\n"
    "help(len)\n"
)


@needs_box
def test_memory_and_help_are_bound_in_the_writers_cells_with_no_import(tmp_path, monkeypatch):
    writer, model = _writer(tmp_path, [_cell("c1", BOUND), *_end()])
    _run(writer)
    out = model.outputs["c1"]
    # imported by the first cell that names it, inside that cell, before its first line runs
    assert "TYPE module" in out and "INDEX True" in out, out
    assert "number of items" in out and "Error" not in out, out


@needs_box
def test_memory_and_help_are_bound_in_a_helpers_cells_with_no_import(tmp_path, monkeypatch):
    library = "def index():\n    return 'helper index\\n'\n"
    out = _helper(tmp_path, [("c1", BOUND)], library=library)["c1"]
    assert "TYPE module" in out and "INDEX True" in out, out
    assert "number of items" in out and "Error" not in out, out


@needs_box
def test_a_helper_without_the_library_package_has_no_memory_name(tmp_path, monkeypatch):
    out = _helper(tmp_path, [("c1", "print('memory' in dir())")])["c1"]
    assert out.splitlines()[0] == "False", out


# --- the end call stops the interpreter (tools in code off) ---------------------------------------------------------


@needs_box
def test_finish_stops_the_interpreter_before_anything_is_judged_and_a_repair_cell_starts_anew(tmp_path, monkeypatch):
    turns = [
        _cell("c1", "x = 41"),
        _call("f1", "finish", {"summary": "s"}),  # no manifest yet: refused (G1), the pass goes on
        _cell("c2", "print(x)"),
    ]
    writer, model = _writer(tmp_path, turns, )
    judged: list[int] = []
    problem = writer._finish_manifest_problem
    close_record = writer._close_record

    def finish_problem(box):
        judged.append(len(_descendants(os.getpid())))
        return problem(box)

    async def closing():
        # an accepted finish ends the record after the box was stopped (and the loop's end does once more)
        judged.append(len(_descendants(os.getpid())))
        await close_record()

    writer._finish_manifest_problem = finish_problem
    writer._close_record = closing
    _run(writer)
    # the box was alive between the cells, and never while a finish was judged
    assert model.boxes[1] > 0, model.boxes
    assert judged and all(n == 0 for n in judged), judged
    # refused for its missing manifest (G1); the pass goes on
    assert "G1" in model.outputs["f1"] and "manifest" in model.outputs["f1"], model.outputs["f1"]
    out = model.outputs["c2"]
    assert out.splitlines()[0] == RESTARTED and "NameError" in out, out
    assert _descendants(os.getpid()) == []


# --- the restart cases say so ------------------------------------------------------------------------------------


@needs_box
def test_a_timed_out_cell_and_a_thread_left_running_each_restart_with_a_note(tmp_path, monkeypatch):
    turns = [
        _cell("c1", "x = 41"),
        _cell("c2", "import time\ntime.sleep(60)"),
        _cell("c3", "print(x)"),
        _cell("c4", "x = 41"),
        _cell("c5", "import threading, time\nthreading.Thread(target=time.sleep, args=(60,), daemon=True).start()"),
        _cell("c6", "print(x)"),
    ]
    writer, model = _writer(tmp_path, turns, cell_timeout_s=3.0)
    _run(writer)
    assert "(the cell timed out after 3 s; the interpreter was restarted: variables are gone, files remain)" in (
        model.outputs["c2"]
    ), model.outputs["c2"]
    assert "NameError" in model.outputs["c3"], model.outputs["c3"]
    assert "this cell started was still running after it ended" in model.outputs["c5"], model.outputs["c5"]
    assert "NameError" in model.outputs["c6"], model.outputs["c6"]
    assert _descendants(os.getpid()) == []


# --- a changed library module is imported anew ----------------------------------------------------------------------


@needs_box
def test_a_module_changed_on_disk_is_imported_anew_in_the_same_interpreter(tmp_path, monkeypatch):
    write = "import pathlib, time\npathlib.Path('/memory/sweepmod.py').write_text({!r})\n"
    turns = [
        # older than the next cell's start by more than the sweep's margin, so the import there is recorded as
        # matching its file (not dropped only because it was new)
        _cell("c1", write.format("V = 1\n") + "time.sleep(3)\nprint('written')"),
        _cell("c2", "import sweepmod\nprint('V', sweepmod.V)"),
        _cell("c3", write.format("V = 22\n") + "print('rewritten')"),
        _cell("c4", "import importlib\nprint('V', importlib.import_module('sweepmod').V)\nprint('x', 'x' in dir())"),
    ]
    writer, model = _writer(tmp_path, [_cell("c0", "x = 1"), *turns])
    _run(writer)
    assert model.outputs["c2"].splitlines()[0] == "V 1", model.outputs["c2"]
    out = model.outputs["c4"].splitlines()
    # the new code, in the same interpreter (its variables are still there)
    assert out[:2] == ["V 22", "x True"], out
    assert _descendants(os.getpid()) == []
