"""Symbolic: with ``UNIFY_TOOLS_IN_CODE=on`` a cell that raises out of the in-code cell path
(``InCodeTools.run_cell``: the cell's task cancelled, or a control exception that is not the cell's reply) still
writes its ``execute_code`` row in ``tool-events.jsonl``, with an error status, at the call number reserved when
it started; no placeholder row for that reservation is left in the file, and the next cell takes the next number.

Keyless and boxless: the interpreter is a stand-in whose ``run_cell`` raises (or waits to be cancelled).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.actor.code_act.test_in_code_tools_wiring import _session
from unify.settings import SETTINGS

# ── ASSUMED NAMES (the integrator adapts them here, in one place) ────────────────────────────────────────────────
#: The status a cell's row has when it did not end normally ("ok" is a normal end).
OK_STATUS = "ok"
#: The event of a placeholder row a reservation might write first; none may remain once the cell's row is written.
PLACEHOLDER_EVENT = "reserved"
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────


class Halt(BaseException):
    """A control exception (not an ``Exception``, not the cell's reply) that ends the cell path."""


def _ok_result() -> SimpleNamespace:
    return SimpleNamespace(
        output="fine\n",
        parts=[],
        error=None,
        result=None,
        notes_before=[],
        notes_after=[],
    )


class StandIn:
    """Stands in for the persistent interpreter: each ``run_cell`` takes the next scripted outcome (an exception
    to raise, ``"wait"`` to wait until cancelled, or a result to return)."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list = []
        self.started = asyncio.Event()
        self.stopped = False

    async def run_cell(self, code, *, call=None):
        self.calls.append(call)
        outcome = self.outcomes.pop(0)
        self.started.set()
        if outcome == "wait":
            await asyncio.sleep(3600)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stop(self):
        self.stopped = True

    def box_alive(self):
        return False


def _tools(tmp_path, monkeypatch, interp: StandIn):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")

    async def start(**_kw):
        return interp

    return _session(tmp_path, start=start)


def _rows(tmp_path) -> list[dict]:
    path = tmp_path / "events" / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _assert_error_row(rows: list[dict], call: int) -> dict:
    assert not [r for r in rows if r.get("event") == PLACEHOLDER_EVENT], rows
    mine = [r for r in rows if r.get("call") == call]
    assert len(mine) == 1, rows
    (row,) = mine
    assert row["event"] == "tool_call" and row["tool"] == "execute_code" and row["via"] == "model"
    assert row["status"] != OK_STATUS, row
    assert row["error"] is True, row
    return row


@pytest.mark.asyncio
async def test_a_cancelled_cell_still_writes_its_row_at_its_number(tmp_path, monkeypatch):
    interp = StandIn([asyncio.CancelledError()])
    tools = _tools(tmp_path, monkeypatch, interp)
    try:
        with pytest.raises(asyncio.CancelledError):
            await tools.run_cell("print('never')")
    finally:
        await tools.close(completed=False)
    assert interp.calls == [0]
    _assert_error_row(_rows(tmp_path), 0)


@pytest.mark.asyncio
async def test_a_cell_whose_task_is_cancelled_mid_cell_still_writes_its_row(tmp_path, monkeypatch):
    interp = StandIn(["wait"])
    tools = _tools(tmp_path, monkeypatch, interp)
    try:
        task = asyncio.ensure_future(tools.run_cell("import time\ntime.sleep(3600)"))
        await asyncio.wait_for(interp.started.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await tools.close(completed=False)
    _assert_error_row(_rows(tmp_path), 0)


@pytest.mark.asyncio
async def test_a_control_exception_out_of_the_cell_writes_an_error_row(tmp_path, monkeypatch):
    interp = StandIn([Halt("stop here")])
    tools = _tools(tmp_path, monkeypatch, interp)
    try:
        with pytest.raises(Halt):
            await tools.run_cell("print('x')")
    finally:
        await tools.close(completed=False)
    _assert_error_row(_rows(tmp_path), 0)


@pytest.mark.asyncio
async def test_the_next_cell_takes_the_next_number_and_ends_normally(tmp_path, monkeypatch):
    interp = StandIn([asyncio.CancelledError(), _ok_result()])
    tools = _tools(tmp_path, monkeypatch, interp)
    try:
        with pytest.raises(asyncio.CancelledError):
            await tools.run_cell("print('first')")
        out = await tools.run_cell("print('second')")
        assert out.error is None
    finally:
        await tools.close(completed=True)
    rows = _rows(tmp_path)
    assert interp.calls == [0, 1]
    _assert_error_row(rows, 0)
    (second,) = [r for r in rows if r.get("call") == 1]
    assert second["status"] == OK_STATUS and second["error"] is False
    cells = [r for r in rows if r.get("tool") == "execute_code"]
    assert sorted(r["call"] for r in cells) == [0, 1]
    # every row in the file is a complete row (no placeholder left by a reservation)
    assert all(r.get("event") == "tool_call" for r in rows), rows


@pytest.mark.asyncio
async def test_an_interpreter_failure_is_still_written_as_before(tmp_path, monkeypatch):
    # the handled path (an Exception from the interpreter) keeps writing its row; it is not a raise out of the path
    interp = StandIn([RuntimeError("the box went away")])
    tools = _tools(tmp_path, monkeypatch, interp)
    try:
        out = await tools.run_cell("print('x')")
    finally:
        await tools.close(completed=False)
    assert out.error and "could not run" in out.error
    rows = _rows(tmp_path)
    assert not [r for r in rows if r.get("event") == PLACEHOLDER_EVENT]
    (row,) = [r for r in rows if r.get("call") == 0]
    assert row["error"] is True
