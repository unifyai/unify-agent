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

import os
import sys
import threading
import time
from typing import Callable

import pytest

from tests.actor.code_act.helpers import WorkerStarts, worker_starts  # noqa: F401
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text

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


async def _run(
    code: str,
    starts: WorkerStarts | None = None,
    *,
    warmed: Callable[[], None] | None = None,
) -> tuple[str, dict, float]:
    """Run *code* in a fresh session. The elapsed time leaves out the
    sandboxed worker's start when *starts* times it. With *warmed*, a
    throwaway first cell runs before the clock starts (the worker's start,
    the session's first policy build and first execution), and *warmed* is
    called once it has."""
    ex = SessionExecutor(environments={}, timeout=None)
    try:
        if warmed is not None:
            await ex.execute(code="1", state_mode="stateful", session_id=0)
            warmed()
        started = time.monotonic()
        res = await ex.execute(code=code, state_mode="stateful", session_id=0)
        elapsed = time.monotonic() - started
        if starts is not None:
            elapsed -= starts.seconds(since=started)
        return parts_to_text(res["stdout"]), res, elapsed
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
async def test_outside_a_cell_stdin_is_the_processs_own(harness_stdin, monkeypatch):
    await _run("x = 1")
    # The cell's empty stdin ended with the cell.
    assert sys.stdin.fileno() == harness_stdin.fileno()
    assert not sys.stdin.isatty()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.parametrize("cell", sorted(CELLS))
async def test_a_cell_in_the_worker_reads_an_empty_stdin(
    cell,
    world,
    monkeypatch,
    worker_starts,  # noqa: F811
):
    """Unchanged: the worker's descriptor 0 was already ``/dev/null``. Under
    the core tool surface (the default) its ``help`` is the worker's own,
    whose page comes from the harness: ``help()`` prints the objects' index
    and reads nothing."""
    code, _error = CELLS[cell]
    out, res, elapsed = await _run(code, worker_starts)
    assert elapsed < 30
    if cell == "help":
        assert res["error"] is None, res["error"]
    elif cell == "input":
        assert "EOFError" in res["error"]
    else:
        assert res["error"] is None, res["error"]
        assert res["result"] == ([] if cell == "iterate" else "")


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
    reads it still ends. The fixture gives the watchdog's start: a test arms
    it once its cold work is done, so the channel is still open when the
    timed cell runs."""
    read_fd, write_fd = os.pipe()
    saved = os.dup(0)
    os.dup2(read_fd, 0)
    os.close(read_fd)
    monkeypatch.setattr(sys, "stdin", open(0, closefd=False))
    os.write(write_fd, HOST_LINE.encode())
    closed = threading.Event()

    def close_channel() -> None:
        os.close(write_fd)
        closed.set()

    watchdog = threading.Timer(WATCHDOG_S, close_channel)
    yield watchdog.start
    # A test that ended before the watchdog has nothing left to free.
    watchdog.cancel()
    if watchdog.ident is not None:
        watchdog.join()
    if not closed.is_set():
        os.close(write_fd)
    os.dup2(saved, 0)
    os.close(saved)


@pytest.mark.asyncio
@pytest.mark.parametrize("started", sorted(STARTED))
async def test_what_a_cell_starts_reads_end_of_input(
    started,
    driver_channel,
    monkeypatch,
    worker_starts,  # noqa: F811
):
    import asyncio

    from unify.cli import _stdin_reader

    with _stdin_reader() as reader:
        # The worker's start and the session's first policy build are not
        # the cell's time; the watchdog starts once they are done.
        out, res, elapsed = await _run(
            STARTED[started],
            worker_starts,
            warmed=driver_channel,
        )
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
