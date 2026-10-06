"""Symbolic: a host cancels the running request of a persistent ``unify act --jsonl`` session.

In the Continual-ARC runs of 5 October at HIGH effort
(``arc-pm2-up592-h-high-fresh0``, instances 6 and 15, and
``arc-pm2-up592-h0-high-fresh0``, instances 6 and 14) a turn ran past the
paper's per-action budget (600 s). The host keeps the session and gives the
turn 60 s to end, as it does for the ACP harnesses after ``session/cancel``,
but ``unify act --jsonl`` had no line that ends a request: a stdin line other
than ``{"message": ...}``, ``{"quit": true}`` or ``{"outcome": ...}`` was
dropped. So nothing could stop the turn, which was still waiting on a model
call (and, in instance 6, on a cell blocked in ``query_llm``); the host
killed and restarted the session 60 s later.

``{"cancel": true}`` now ends the running request and keeps the session: the
model call in flight and the running tool calls are cancelled, each pending
call is answered as cancelled, the transcript says the request was cancelled,
and the request ends in its ``response`` line (``"cancelled": true``, the text
drafted so far as its content). With no request running the line is ignored.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h

PROTOCOL = (
    "You are solving a grid puzzle. Reply with one action as a JSON object on "
    'the last line: {"action": "submit", "grid": [[...]]}.'
)
TASK = "Solve the puzzle. " + PROTOCOL
DRAFT = 'Checking the demos first. Best so far: {"action": "submit", "grid": [[4]]}'
FOLLOW_UP = (
    "Feedback since your last action:\nYou ran out of time on your last action. "
    "Reply with one action."
)
FINAL = '{"action": "submit", "grid": [[4]]}'
LATE = "A late answer to the cancelled request."
SUMMARY = "Nothing worth storing."
# Well under the host's 60 s grace: a cancelled request must end at once.
CANCEL_BOUND_S = 5.0


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    return "## Storage Review" in text or "You are a skill librarian" in text


def _last_request_text(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not message.get("_loop_authored"):
            return str(message.get("content") or "")
    return ""


class _Model:
    """Scripted replies. ``first`` answers the task's first call; a call it
    returns ``None`` for hangs until cancelled. The follow-up is answered
    with an action and the review summarises."""

    def __init__(self, first) -> None:
        self._first = first
        self.requests: list[list[dict]] = []
        self.in_flight = asyncio.Event()
        self.release = asyncio.Event()
        self._task_calls = 0

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append(messages)
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        if FOLLOW_UP in _last_request_text(messages):
            return h.completion(content=FINAL)
        self._task_calls += 1
        reply = self._first(self._task_calls)
        if reply is not None:
            return reply
        # A long reasoning call that has not answered yet. UniLLM shields
        # the provider request from a caller that gives up (the provider
        # bills it anyway), so this call answers late, once released.
        self.in_flight.set()
        await self.release.wait()
        return h.completion(content=LATE)


class _Tools:
    """``study`` runs until cancelled, as a cell blocked in a nested model
    call does."""

    def __init__(self) -> None:
        self.running = asyncio.Event()
        self.real_cells = False
        self.calls = 0
        self.cancelled = 0

    async def study(self) -> str:
        """Study the demonstrations at length."""
        self.calls += 1
        self.running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return "done"


class _Actor:
    """The actor ``unify act`` starts, on a scripted persistent loop."""

    def __init__(self, tools: _Tools) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()
        self._tools = tools
        self.real_cells = False

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client(PROTOCOL),
            request,
            (
                {"execute_code": self._actor.get_tools("act")["execute_code"]}
                if self.real_cells
                else {"study": self._tools.study}
            ),
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=300,
            persist=persist,
        )
        return _StorageCheckHandle(inner=inner, actor=self._actor, persist=persist)

    async def close(self) -> None:
        await self._actor.close()


@pytest.fixture
def jsonl_session(monkeypatch):
    """``unify act --persist --jsonl`` on the scripted actor; stdin is a pipe."""
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    tools = _Tools()

    async def start(self) -> None:
        self._actor = _Actor(tools)
        self._actor.real_cells = tools.real_cells

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=True,
            no_compose=False,
            no_store=False,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send, tools
    os.close(write_fd)


async def _until(predicate, timeout: float = 20) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout)


def _types(lines: list[dict]) -> list[str]:
    return [line["type"] for line in lines if line["type"] != "storage"]


async def _cancel(send, lines) -> float:
    """Send the cancel and wait, bounded, for the request's response line."""
    before = _types(lines).count("response")
    started = time.monotonic()
    send({"cancel": True})
    try:
        await _until(
            lambda: _types(lines).count("response") > before,
            timeout=CANCEL_BOUND_S,
        )
    except asyncio.TimeoutError:
        raise AssertionError(
            f"the cancelled request did not end within {CANCEL_BOUND_S:g}s; "
            f"lines so far: {_types(lines)}",
        ) from None
    return time.monotonic() - started


async def _finish(run, send, lines) -> int:
    """The host's follow-up, answered as usual, then the end of the session."""
    send({"message": FOLLOW_UP})
    await _until(lambda: FINAL in [line.get("content") for line in lines])
    send({"quit": True})
    return await asyncio.wait_for(run, 20)


def _follow_up_request(model: _Model) -> list[dict]:
    (turn,) = [r for r in model.requests if _last_request_text(r) == FOLLOW_UP]
    return turn


@pytest.mark.asyncio
async def test_a_cancel_ends_a_request_waiting_on_a_model_call(jsonl_session):
    """The recorded shape: the request's model call has not answered."""
    session, lines, send, _tools = jsonl_session
    model = _Model(lambda n: None)
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await asyncio.wait_for(model.in_flight.wait(), 20)
        elapsed = await _cancel(send, lines)
        stats = session._handle._inner._runtime_state.cancelled_turns_by_cause
        # The late answer arrives while the next request runs.
        model.release.set()
        code = await _finish(run, send, lines)

    assert code == 0
    assert elapsed < CANCEL_BOUND_S
    assert stats == {"cancel": 1}
    # The cancelled call's late answer reaches neither the host nor the model.
    assert LATE not in json.dumps(lines)
    assert not any(LATE in json.dumps(r, default=str) for r in model.requests)
    assert _types(lines) == ["response", "response", "result", "ended"]
    cancelled, answered = (line for line in lines if line["type"] == "response")
    assert cancelled == {"type": "response", "content": "", "cancelled": True}
    assert answered == {"type": "response", "content": FINAL}
    # The next request reads that the last one was cancelled.
    assert any(
        m.get("role") == "assistant"
        and str(m.get("content") or "").startswith("🔚 Cancelled")
        for m in _follow_up_request(model)
    )


@pytest.mark.asyncio
async def test_a_cancel_ends_a_request_waiting_on_a_running_tool(jsonl_session):
    """A call that never finishes is cancelled and answered as cancelled."""
    session, lines, send, tools = jsonl_session
    model = _Model(
        lambda n: (
            h.completion(content=DRAFT, calls=[("study", {})]) if n == 1 else None
        ),
    )
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await asyncio.wait_for(tools.running.wait(), 20)
        elapsed = await _cancel(send, lines)
        code = await _finish(run, send, lines)

    assert code == 0
    assert elapsed < CANCEL_BOUND_S
    assert tools.cancelled == 1
    assert _types(lines) == ["response", "response", "result", "ended"]
    cancelled = next(line for line in lines if line["type"] == "response")
    # The text drafted for the request is its content.
    assert cancelled == {"type": "response", "content": DRAFT, "cancelled": True}
    turn = _follow_up_request(model)
    (reply,) = [m for m in turn if m.get("role") == "tool" and m.get("name") == "study"]
    assert str(reply["content"]).startswith("Cancelled")
    # The cancelled call is not run again.
    assert tools.calls == 1


@pytest.mark.asyncio
async def test_a_cancel_ends_a_request_waiting_on_a_running_cell(
    jsonl_session,
    tmp_path,
    monkeypatch,
):
    """The actor's own ``execute_code``, in process, on a cell that awaits
    for an hour (as the recorded cell awaited its ``query_llm`` call)."""
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "")
    session, lines, send, tools = jsonl_session
    tools.real_cells = True
    marker = tmp_path / "cell-started"
    code = (
        "import asyncio\n"
        f"open({str(marker)!r}, 'w').write('started')\n"
        "await asyncio.sleep(3600)\n"
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
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(marker.exists)
        elapsed = await _cancel(send, lines)
        model.release.set()
        code_ = await _finish(run, send, lines)

    assert code_ == 0
    assert elapsed < CANCEL_BOUND_S
    assert _types(lines) == ["response", "response", "result", "ended"]
    cancelled = next(line for line in lines if line["type"] == "response")
    assert cancelled == {"type": "response", "content": DRAFT, "cancelled": True}
    turn = _follow_up_request(model)
    (reply,) = [
        m for m in turn if m.get("role") == "tool" and m.get("name") == "execute_code"
    ]
    assert str(reply["content"]).startswith("Cancelled")


@pytest.mark.asyncio
async def test_a_cancel_with_no_request_running_is_ignored(jsonl_session):
    """Each request still ends in exactly one response line."""
    session, lines, send, _tools = jsonl_session
    model = _Model(lambda n: h.completion(content=DRAFT))
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: "response" in _types(lines))
        send({"cancel": True})
        code = await _finish(run, send, lines)

    assert code == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    first, answered = (line for line in lines if line["type"] == "response")
    assert first == {"type": "response", "content": DRAFT}
    assert answered == {"type": "response", "content": FINAL}
    assert not any(
        str(m.get("content") or "").startswith("🔚 Cancelled")
        for m in _follow_up_request(model)
    )


def test_the_cancel_is_not_a_steering_action():
    """The model's steering surface is unchanged: ``cancel_request`` is for
    the host, never offered as a method a ``steer`` call can name."""
    from unify.actor.code_act_actor import _StorageCheckHandle
    from unify.common._async_tool.dynamic_tools_factory import DynamicToolFactory

    handle = _StorageCheckHandle.__new__(_StorageCheckHandle)
    assert "cancel_request" not in DynamicToolFactory._discover_custom_public_methods(
        handle,
    )
