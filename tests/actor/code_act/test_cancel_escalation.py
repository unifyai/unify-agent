"""Symbolic: a host's cancel is answered even when running work does not stop.

``{"cancel": true}`` ends the running request of a persistent ``unify act
--jsonl`` session: the loop cancels the model call and the tool calls in
flight and ends the request in its ``response`` line (``"cancelled":
true``), the acknowledgement. As first built, that line waited, with no
limit, for every cancelled call to stop, so two kinds of work held it:

- a tool that ignores its cancellation (swallows ``CancelledError``): the
  loop now waits a bounded grace, then abandons the call (answered as
  cancelled and abandoned; whatever it still does is not reported, and what
  it costs stays unknown);
- an in-process cell that holds the event loop's thread (``time.sleep``):
  the loop cannot even read the cancel. The channel is read on its own
  thread, which sees the cancel, and when no response has come after the
  grace it interrupts the cell running on the main thread.

Code blocked in C that never returns to the interpreter cannot be
interrupted in process. The transport is scripted; every wait is bounded,
and a failing run ends when the work does (4-6 s).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act import test_cancel_request_session as base
from tests.actor.code_act.test_cancel_request_session import (  # noqa: F401
    DRAFT,
    FINAL,
    _finish,
    _follow_up_request,
    _Model,
    _types,
    _until,
    jsonl_session,
)

GRACE_S = 0.3
#: The cancelled response must arrive within this of the cancel.
BOUND_S = 2.0
#: How long the stubborn work runs if nothing stops it.
STUBBORN_S = 5.0


@pytest.fixture(autouse=True)
def short_grace(monkeypatch):
    from unify import cli
    from unify.common._async_tool import loop

    monkeypatch.setattr(loop, "_CANCEL_GRACE_S", GRACE_S)
    monkeypatch.setattr(cli, "CANCEL_INTERRUPT_GRACE_S", GRACE_S)


def _install(model) -> None:
    import unillm.clients.uni_llm as uni_llm

    uni_llm._acompletion_with_transient_retry = model


@pytest.mark.asyncio
async def test_a_tool_that_ignores_its_cancel_is_abandoned(jsonl_session, monkeypatch):
    session, lines, send, tools = jsonl_session
    ignored: list[float] = []

    async def study(self) -> str:
        """Study the demonstrations at length."""
        self.calls += 1
        self.running.set()
        deadline = time.monotonic() + STUBBORN_S
        while time.monotonic() < deadline:
            try:
                await asyncio.sleep(deadline - time.monotonic())
            except asyncio.CancelledError:
                ignored.append(time.monotonic())
        return "done"

    monkeypatch.setattr(base._Tools, "study", study)
    model = _Model(
        lambda n: (
            h.completion(content=DRAFT, calls=[("study", {})]) if n == 1 else None
        ),
    )
    with h.scripted(()):
        _install(model)
        run = asyncio.create_task(session.run(base.TASK))
        await asyncio.wait_for(tools.running.wait(), BOUND_S * 5)
        started = time.monotonic()
        send({"cancel": True})
        await _until(lambda: "response" in _types(lines), STUBBORN_S + 2)
        elapsed = time.monotonic() - started
        code = await _finish(run, send, lines)

    assert ignored, "the tool was never cancelled"
    assert elapsed < BOUND_S, f"the cancelled response took {elapsed:.1f}s"
    assert code == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    cancelled = next(line for line in lines if line["type"] == "response")
    assert cancelled == {"type": "response", "content": DRAFT, "cancelled": True}
    (reply,) = [
        m
        for m in _follow_up_request(model)
        if m.get("role") == "tool" and m.get("name") == "study"
    ]
    assert str(reply["content"]).startswith("Cancelled")
    assert "abandoned" in str(reply["content"])
    assert tools.calls == 1


@pytest.mark.asyncio
async def test_a_cell_holding_the_loop_is_interrupted(
    jsonl_session,
    tmp_path,
    monkeypatch,
):
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    session, lines, send, tools = jsonl_session
    tools.real_cells = True
    marker = tmp_path / "cell-started"
    code = (
        "import time\n"
        f"open({str(marker)!r}, 'w').write('started')\n"
        f"time.sleep({STUBBORN_S})\n"
        "print('the cell finished')"
    )
    model = _Model(
        lambda n: (
            h.completion(
                content=DRAFT,
                calls=[("execute_code", {"thought": "Studying.", "code": code})],
            )
            if n == 1
            else None
        ),
    )
    sent_at: list[float] = []

    def host() -> None:
        # The event loop is held by the cell, so the host's side runs on a
        # thread: it sends the cancel once the cell has started.
        deadline = time.monotonic() + 20
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        sent_at.append(time.monotonic())
        send({"cancel": True})

    with h.scripted(()):
        _install(model)
        run = asyncio.create_task(session.run(base.TASK))
        threading.Thread(target=host, daemon=True).start()
        await _until(lambda: "response" in _types(lines), 20)
        answered_at = time.monotonic()
        model.release.set()
        code_ = await _finish(run, send, lines)

    elapsed = answered_at - sent_at[0]
    assert elapsed < BOUND_S, f"the cancelled response took {elapsed:.1f}s"
    assert code_ == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    cancelled = next(line for line in lines if line["type"] == "response")
    assert cancelled["cancelled"] is True
    (reply,) = [
        m
        for m in _follow_up_request(model)
        if m.get("role") == "tool" and m.get("name") == "execute_code"
    ]
    # Interrupted (it would have slept 5 s), then answered as cancelled.
    assert str(reply["content"]).startswith("Cancelled")
    assert "the cell finished" not in str(reply["content"])


def test_a_signal_outside_a_cell_interrupts_nothing():
    """The handler raises only inside a cell's frame, and only while a
    cancel is unanswered."""
    from types import SimpleNamespace

    from unify.cli import Act

    act = Act(SimpleNamespace(jsonl=True, persist=True))
    act._cancel_answered.clear()
    frame = __import__("sys")._getframe()
    act._interrupt_cell(None, frame)  # no __exec_wrapper on this stack

    namespace: dict = {}
    exec(
        "def __exec_wrapper(handler, sys):\n    handler(None, sys._getframe())\n",
        namespace,
    )
    from unify.cli import CellInterrupted

    with pytest.raises(CellInterrupted):
        namespace["__exec_wrapper"](act._interrupt_cell, __import__("sys"))
    act._cancel_answered.set()
    namespace["__exec_wrapper"](act._interrupt_cell, __import__("sys"))
