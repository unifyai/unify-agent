"""Symbolic: model-run code reads an empty stdin, never the harness's own.

In the Continual-ARC cell ``arc-pm2-up592-h-low-ws0`` (5 October, instance
19) a delegated sub-actor called ``execute_function("help")`` with no
arguments. That is ``help()``: pydoc's interactive help, which reads its
next request from stdin. In process, a cell runs on the event loop's thread
and the harness's stdin is ``unify act --jsonl``'s message channel, so the
whole process waited on a line that only the host could send, and the host
was waiting for Unify: no model call for 300 s, then an idle timeout and a
restart. A line that did arrive would have gone to pydoc instead of the
session.

A cell's stdin is now empty, in process as in the sandboxed worker (whose
descriptor 0 is ``/dev/null``): ``help()`` ends at once, ``input()`` raises
``EOFError`` in the cell, where the model sees it, and ``sys.stdin.read()``
returns ``""``. Outside a cell ``sys.stdin`` is the process's own.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.settings import SETTINGS

HOST_LINE = '{"message": "the host\'s next message"}\n'
# The watchdog writes the host's line only this late: a cell still reading
# stdin by then has hung, and the line it then takes was not its own.
WATCHDOG_S = 5.0


@pytest.fixture
def harness_stdin(monkeypatch):
    """The harness's stdin is a pipe with nothing in it yet, as under
    ``unify act --jsonl`` while the host waits for an answer."""
    read_fd, write_fd = os.pipe()
    stdin = os.fdopen(read_fd, "r")
    monkeypatch.setattr(sys, "stdin", stdin)

    closed = threading.Event()

    def unblock() -> None:
        # The host's line, then the end of input: a cell that reads the pipe
        # (pydoc reads until end of input) is freed, so a failing run ends.
        os.write(write_fd, HOST_LINE.encode())
        os.close(write_fd)
        closed.set()

    watchdog = threading.Timer(WATCHDOG_S, unblock)
    # Started by the test right before the cell runs.
    stdin.arm = watchdog.start
    yield stdin
    watchdog.cancel()
    if not closed.is_set():
        os.close(write_fd)
    stdin.close()


async def _run(code: str) -> tuple[str, dict, float]:
    ex = SessionExecutor(environments={}, timeout=None)
    try:
        started = time.monotonic()
        res = await ex.execute(code=code, state_mode="stateful", session_id=0)
        return parts_to_text(res["stdout"]), res, time.monotonic() - started
    finally:
        await ex.close()


CELLS = {
    "help": ("help()", None),
    # ``input`` is not among a cell's builtins; the module still has it.
    "input": ("import builtins\nbuiltins.input('next? ')", "EOFError"),
    "read": ("import sys\nsys.stdin.read()", None),
    "readline": ("import sys\nsys.stdin.readline()", None),
    "iterate": ("import sys\n[line for line in sys.stdin]", None),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("cell", sorted(CELLS))
async def test_a_cell_in_process_reads_an_empty_stdin(cell, harness_stdin, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    code, error = CELLS[cell]
    harness_stdin.arm()
    out, res, elapsed = await _run(code)

    assert (
        elapsed < WATCHDOG_S
    ), f"the cell waited on the harness's stdin ({elapsed:.1f}s)"
    if error is None:
        assert res["error"] is None, res["error"]
    else:
        assert error in res["error"]
    if cell in ("read", "readline"):
        assert res["result"] == ""
    if cell == "iterate":
        assert res["result"] == []
    if cell == "help":
        assert "help>" in out
    # The host's line is still there for the harness to read.
    time.sleep(WATCHDOG_S - elapsed + 0.5)
    assert harness_stdin.readline() == HOST_LINE


@pytest.mark.asyncio
async def test_outside_a_cell_stdin_is_the_processs_own(harness_stdin, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    await _run("x = 1")
    # The cell's empty stdin ended with the cell.
    assert sys.stdin.fileno() == harness_stdin.fileno()
    assert not sys.stdin.isatty()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.parametrize("cell", sorted(CELLS))
async def test_a_cell_in_the_worker_reads_an_empty_stdin(cell, world, monkeypatch):
    """Unchanged: the worker's descriptor 0 was already ``/dev/null``, and its
    restricted builtins leave out ``help``."""
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    code, _error = CELLS[cell]
    out, res, elapsed = await _run(code)
    assert elapsed < 30
    if cell == "help":
        assert "NameError: name 'help' is not defined" in res["error"]
    elif cell == "input":
        assert "EOFError" in res["error"]
    else:
        assert res["error"] is None, res["error"]
        assert res["result"] == ([] if cell == "iterate" else "")


@pytest.mark.asyncio
async def test_execute_function_help_with_no_arguments_returns(
    harness_stdin,
    monkeypatch,
):
    """The recorded call: ``execute_function("help")``, no arguments."""
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    actor = CodeActActor()
    try:
        fn = actor.get_tools("act")["execute_function"]
        fn = getattr(fn, "fn", fn)
        harness_stdin.arm()
        started = time.monotonic()
        out = await fn(thought="Reading the help.", function_name="help")
        elapsed = time.monotonic() - started
    finally:
        await actor.close()

    assert elapsed < WATCHDOG_S, f"execute_function waited on stdin ({elapsed:.1f}s)"
    assert "help>" in json.dumps(out, default=str)
    time.sleep(WATCHDOG_S - elapsed + 0.5)
    assert harness_stdin.readline() == HOST_LINE


# ── what a cell starts ─────────────────────────────────────────────────────
# An empty ``sys.stdin`` covers code that reads it on the cell's own thread.
# A subprocess the cell starts inherits descriptor 0 instead, and a thread
# it starts does not inherit the cell's context, so both would still reach
# the driver's channel. While ``unify act`` reads a channel that is not a
# terminal, descriptor 0 is ``/dev/null`` and the channel is read from a
# private copy.

STARTED = {
    "subprocess": (
        "import subprocess\n"
        "subprocess.run(['cat'], capture_output=True, text=True, timeout=30).stdout"
    ),
    "thread": (
        "import sys, threading\n"
        "got = []\n"
        "t = threading.Thread(target=lambda: got.append(sys.stdin.readline()))\n"
        "t.start(); t.join(30)\n"
        "got"
    ),
}


@pytest.fixture
def driver_channel(monkeypatch):
    """Descriptor 0 is a pipe holding the driver's next line, as under
    ``unify act --jsonl``; the pipe closes after the watchdog, so a cell that
    reads it still ends."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(0)
    os.dup2(read_fd, 0)
    os.close(read_fd)
    monkeypatch.setattr(sys, "stdin", open(0, closefd=False))
    os.write(write_fd, HOST_LINE.encode())
    watchdog = threading.Timer(WATCHDOG_S, os.close, (write_fd,))
    watchdog.start()
    yield
    watchdog.join()
    os.dup2(saved, 0)
    os.close(saved)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", sorted(STARTED))
async def test_what_a_cell_starts_reads_end_of_input(
    started,
    driver_channel,
    monkeypatch,
):
    import asyncio

    from unify.cli import _stdin_reader

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    with _stdin_reader() as reader:
        out, res, elapsed = await _run(STARTED[started])
        assert res["error"] is None, res["error"]
        assert (
            elapsed < WATCHDOG_S
        ), f"the {started} waited on the driver's channel ({elapsed:.1f}s)"
        assert res["result"] in ("", [""]), res["result"]
        # The driver's line is still the CLI's to read.
        line = await asyncio.wait_for(reader.readline(), WATCHDOG_S * 2)
        assert line.decode() == HOST_LINE
    # Descriptor 0 is the channel again once the reader is done.
    assert os.fstat(0).st_ino != os.stat(os.devnull).st_ino
