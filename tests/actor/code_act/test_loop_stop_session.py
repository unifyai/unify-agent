"""Symbolic: a persistent ``unify act --jsonl`` session whose model narrates in no-op cells.

In the overhauled ARC LOW screen of 6 October (``arc-ovscs-e8c4fdb-h-low-s0-r1``,
instance 15) the model ran 150 cells that only printed constant text, 45
distinct strings among them, over 343 s, until the per-request step cap
(300 messages) ended the request. No two consecutive cells were alike, so an
exact-repeat rule never fires.

With ``UNIFY_LOOP_STOP=on`` the tenth such cell in a row ends the request:
the model is asked for its best answer in one tool-less turn, the CLI writes
that as the request's ``response`` line, and the session takes the host's
next message as usual. ``UNIFY_STEP_CAP_REPLY`` stays off here, as in the
paper rows: the stop still answers the request.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import re
import os
import sys
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

PROTOCOL = "Reply with one action as a JSON object on the last line."
TASK = "Solve the puzzle. " + PROTOCOL
FOLLOW_UP = "Feedback since your last action: none. Reply with one action."
LAST_WORD = '{"action": "submit", "answer": 4}'
FINAL = '{"action": "submit", "answer": 5}'
SUMMARY = "Nothing worth storing."
LINES = [f"Requesting the next example now ({i})." for i in range(45)]
K = 10
BOUND = 5  # seconds; a scripted run takes well under one


def _is_review(messages: list) -> bool:
    text = json.dumps(messages, default=str)
    # The storage review as shipped, or framed as the agent's own curation
    # step (UNIFY_REVIEW_FRAMING=unified, the default since the code freeze).
    return (
        "## Storage Review" in text
        or "You are a skill librarian" in text
        or "This is the curation step that follows" in text
    )


NOTICE_HEAD = "The last "  # the loop stop's notice

# The loop's own notices (call lifecycle, record blocks) reach the provider
# without their ``_loop_authored`` marker; none is a requester message.
_NOTICE = re.compile(r"^\[(?:steerable|askable|h\d+|record|note)[ :\]]")


def _is_notice(message: dict) -> bool:
    return (
        bool(message.get("_loop_authored"))
        or bool(
            _NOTICE.match(str(message.get("content") or "")),
        )
        or str(message.get("content") or "").startswith(NOTICE_HEAD)
    )


def _last_request_text(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and not _is_notice(message):
            return str(message.get("content") or "")
    return ""


class _Model:
    """Narrates in print cells for the task; answers the follow-up."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.cells = 0

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        self.requests.append({"messages": messages, "tools": kw.get("tools")})
        if _is_review(messages):
            return h.completion(content=SUMMARY)
        if not kw.get("tools"):
            return h.completion(content=LAST_WORD)
        if FOLLOW_UP in _last_request_text(messages):
            return h.completion(content=FINAL)
        line = LINES[self.cells % len(LINES)]
        self.cells += 1
        return h.completion(
            calls=[
                (
                    "execute_code",
                    {"thought": "Asking for more.", "code": f"print({line!r})"},
                ),
            ],
        )


async def execute_code(thought: str = "", code: str = "") -> str:
    """Run Python code.

    Args:
        thought: Why the code is run.
        code: The code to run.
    """
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(code, {})  # the scripted model's own constant cells
    return out.getvalue()


class _Actor:
    """The actor ``unify act`` starts, on a scripted persistent loop."""

    def __init__(self) -> None:
        from unify.actor.code_act_actor import CodeActActor

        self._actor = CodeActActor()

    async def act(self, request: str, *, persist: bool, **_kwargs):
        from unify.actor.code_act_actor import _StorageCheckHandle
        from unify.common.async_tool_loop import start_async_tool_loop

        inner = start_async_tool_loop(
            h.new_client(PROTOCOL),
            request,
            {"execute_code": execute_code},
            loop_id="CodeActActor.act",
            log_steps=False,
            timeout=30,
            persist=persist,
            max_steps=300,
            # As the actor starts its task loop.
            reply_channel=True,
            bind_request=True,
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

    async def start(self) -> None:
        self._actor = _Actor()

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_compose=False,
            no_store=False,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send
    os.close(write_fd)


async def _until(predicate, timeout: float = BOUND) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout)


def _types(lines: list[dict]) -> list[str]:
    return [line["type"] for line in lines if line["type"] != "storage"]


@pytest.mark.asyncio
async def test_narration_cells_end_the_request_and_the_session_goes_on(
    jsonl_session,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_LOOP_STOP_K", K)
    monkeypatch.setattr(SETTINGS, "UNIFY_STEP_CAP_REPLY", "")
    session, lines, send = jsonl_session
    model = _Model()
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        await _until(lambda: "response" in _types(lines))
        cells_at_stop = model.cells
        send({"message": FOLLOW_UP})
        await _until(lambda: _types(lines).count("response") == 2)
        send({"quit": True})
        code = await asyncio.wait_for(run, BOUND)

    assert code == 0
    assert _types(lines) == ["response", "response", "result", "ended"]
    stopped, answered = (
        line["content"] for line in lines if line["type"] == "response"
    )
    assert stopped.startswith(f"🔚 Stopped: the last {K} tool calls made no progress")
    # The host reads the action on the last line.
    assert stopped.endswith("Best current answer:\n" + LAST_WORD)
    assert answered == FINAL
    (result,) = [line for line in lines if line["type"] == "result"]
    assert result["run_stats"]["loop_stops"] == 1
    # Ten cells, not the 150 of the recorded run.
    assert cells_at_stop == K
    # The follow-up reached the session after the model's last word.
    (turn,) = [
        r
        for r in model.requests
        if r["tools"] and _last_request_text(r["messages"]) == FOLLOW_UP
    ]
    assert any(
        m.get("role") == "assistant" and m.get("content") == LAST_WORD
        for m in turn["messages"]
    )
