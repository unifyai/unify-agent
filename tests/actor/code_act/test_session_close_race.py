"""A cell whose box is still starting when its session closes: close() finds no interpreter to stop, so the box that
the start returns afterwards is stopped by the cell itself; a cell of a session already closed is refused and starts
no box. Without this a
teammate's last cell, begun as its request ended, left a sandboxed worker running past the session (and the test that
started it)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from unify.actor import in_code_tools
from unify.settings import SETTINGS


def _session(tmp_path):
    return in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events",
    )


class _Box:
    """Stands in for a started interpreter: records its stop."""

    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    async def run_cell(self, code, call=None):  # the cell runs cleanly, so only the session close can stop the box
        return SimpleNamespace(notes_before=[], parts=[], output="", error=None, result=None, notes_after=[])

    def box_alive(self) -> bool:
        return not self.stopped


@pytest.mark.asyncio
async def test_a_box_that_finishes_starting_after_its_session_closed_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    release, starting, boxes = asyncio.Event(), asyncio.Event(), []

    async def start(**kw):
        starting.set()
        await release.wait()
        boxes.append(_Box())
        return boxes[-1]

    session = _session(tmp_path)
    session._start = start
    cell = asyncio.create_task(session.run_cell("x = 1 + 1\nprint(x)"))
    await asyncio.wait_for(starting.wait(), 10)
    await session.close()  # the box is still starting: there is no interpreter to stop yet
    release.set()
    out = await asyncio.wait_for(cell, 10)
    assert [b.stopped for b in boxes] == [True], "the box that started after the close is still running"
    assert session.interp is None and not session.box_alive()
    assert out.error and "could not run" in out.error


@pytest.mark.asyncio
async def test_a_closed_session_starts_no_box(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    started = []

    async def start(**kw):
        started.append(kw)
        return _Box()

    session = _session(tmp_path)
    session._start = start
    await session.close()
    with pytest.raises(RuntimeError, match="tools have ended"):
        await session.run_cell("y = 2 + 2\nprint(y)")
    assert started == [] and session.interp is None


@pytest.mark.asyncio
async def test_a_close_that_is_cancelled_midway_still_stops_the_box(tmp_path, monkeypatch):
    """A helper's request is cancelled as its run ends, while its session closes in a ``finally``: the box's stop
    still runs to its end, and the cancellation is raised again afterwards."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    stopping, release = asyncio.Event(), asyncio.Event()

    class SlowBox(_Box):
        async def stop(self) -> None:
            stopping.set()
            await release.wait()
            self.stopped = True

    box = SlowBox()

    async def start(**kw):
        return box

    session = _session(tmp_path)
    session._start = start
    await session.run_cell("x = 1 + 1\nprint(x)")
    closing = asyncio.create_task(session.close())
    await asyncio.wait_for(stopping.wait(), 10)
    closing.cancel()  # the closing task is cancelled while the box is stopping
    await asyncio.sleep(0.05)
    assert not closing.done(), "the close gave up on the box at the cancellation"
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, 10)
    assert box.stopped and session.interp is None and not session.box_alive()
