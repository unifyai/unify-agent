"""Symbolic: a request's answer reaches the persistent JSONL session of ``unify act`` with the tools in code.

A program driving the harness runs ``unify act --persist --jsonl --no-clarify --quiet --home <dir> "<request>"``,
reads JSON lines from stdout until one whose ``type`` is ``response``, ``result`` or ``ended``, takes
``str(event["content"])`` and parses the last JSON object in it as its next action. Later requests go in as stdin
lines ``{"message": ...}``; ``{"quit": true}`` ends the session.

With ``UNIFY_TOOLS_IN_CODE=on`` the actor's one model-API tool is ``execute_code(code)`` and it ends a turn either with
a plain text reply or with ``tools.answer(answer=...)`` from a cell (a dict answer is sent as its JSON text, and the
loop ends the turn without another model call). Each test drives the CLI's ``Act`` session in process on the real
``CodeActActor`` and the real sandboxed interpreter; the model is the scripted transport (tests/scripted_model.py):
no key, no network. Every line the session emits is serialised as ``Act._emit`` serialises it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

from tests.actor.code_act.core_world import core_world, new_actor  # noqa: F401 (fixture)
from tests.actor.code_act.helpers import WORKER_START_BOUND_S
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.scripted_model import ScriptedModel, cell, reply, scripted
from unify.actor import in_code_tools
from unify.common import tool_registry as R
from unify.memory.integration import request as request_mod
from unify.settings import SETTINGS, ProductionSettings

REQUEST = "Look at the task and reply with your next action as one JSON object."
FOLLOW_UP = "The action was taken. Reply with your next action as one JSON object."
SUBMIT = {"action": "submit", "grid": [[1, 2], [3, 4]]}
SUBMIT_TEXT = json.dumps(SUBMIT)
DEMO_TEXT = '{"action": "request_demo"}'
# The reply a test scripts after the cell that should have ended the turn: seen only when it did not.
NOT_ENDED = "the cell did not end the turn"
TEST_BOUND_S = 300
# A request's response comes once its cells and model calls are done: the first may start the interpreter's box,
# a later one runs on the box already started. Neither waits for any idle period.
FIRST_RESPONSE_BOUND_S = WORKER_START_BOUND_S + 60
NEXT_RESPONSE_BOUND_S = 30.0
# From the quit line to the end of the session (its episode is recorded; no pass is due).
END_BOUND_S = 60.0
ANSWER_TYPES = ("response", "result", "ended")


def code(text: str):
    """An ``execute_code`` call as the tools in code take it: one argument."""
    return ("execute_code", {"code": text})


def last_object(text: str) -> Optional[Any]:
    """The last top-level JSON object in *text*, as a driver parses its action from the content; None when there is
    none."""
    decoder = json.JSONDecoder()
    found = None
    at = 0
    while True:
        at = text.find("{", at)
        if at < 0:
            return found
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at += 1
            continue
        if isinstance(value, dict):
            found = value
        at = end


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _tool_results(call) -> str:
    return "\n".join(_text(m.get("content")) for m in call.messages if m.get("role") == "tool")


def _why(model: ScriptedModel) -> str:
    """The last request's tool results (the cells' output and errors), for a failing assertion."""
    return "\n" + "\n".join(_tool_results(model.calls[-1]).splitlines()[-40:]) if model.calls else "\n(no model call)"


class _Collect:
    """Stands in for :data:`in_code_tools.SESSIONS`: keeps every in-code session a test opens."""

    def __init__(self) -> None:
        self.opened: list = []

    def add(self, session) -> None:
        self.opened.append(session)


@pytest.fixture
def session_world(core_world, monkeypatch):  # noqa: F811
    """The sandbox world as the in-code tests set it up: a memory run (no stored-function library, no storage
    review) whose consolidation is never due, the default cell state, and every in-code session the test opens.
    Each test sets ``UNIFY_TOOLS_IN_CODE`` itself."""
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", ProductionSettings.model_fields["UNIFY_CELL_STATE"].default)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_E", 10**9)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"world": core_world, "opened": collect.opened}


async def _drive(
    world: dict,
    monkeypatch,
    model: ScriptedModel,
    first: str,
    follow_ups: list[str],
) -> dict:
    """One ``unify act --persist --jsonl --no-clarify --quiet --home <dir> <first>`` session, in process. Each
    ``response`` line is answered by the next follow-up as a ``{"message": ...}`` line, and the last one by
    ``{"quit": true}``. Returns the emitted lines, each with the seconds it took (from the session's start for the
    first response, from the message it answers for a later one, from the quit for the lines after it), and the exit
    code."""
    from unify.cli import Act, _parse_args

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    monkeypatch.chdir(world["workspace"])
    args = _parse_args(
        ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "--home", str(world["state"]), first],
    )
    session = Act(args)
    lines: list[dict] = []
    waited: list[float] = []
    queue = list(follow_ups)
    state = {"since": time.monotonic(), "closed": False}

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())
        state["since"] = time.monotonic()

    def emit(**payload: Any) -> None:
        # the bytes Act._emit prints, read back as the driver reads them
        lines.append(json.loads(json.dumps(payload, default=str)))
        waited.append(time.monotonic() - state["since"])
        if payload.get("type") != "response" or state["closed"]:
            return
        if queue:
            send({"message": queue.pop(0)})
        else:
            state["closed"] = True
            send({"quit": True})
            os.close(write_fd)

    async def start() -> None:
        session._actor = new_actor()

    monkeypatch.setattr(session, "_emit", emit)
    monkeypatch.setattr(session, "start", start)
    try:
        with scripted(model):
            code_ = await asyncio.wait_for(session.run(first), TEST_BOUND_S - 60)
    finally:
        await session.close()
        if not state["closed"]:
            os.close(write_fd)
    return {"code": code_, "lines": lines, "waited": waited}


def _responses(lines: list[dict]) -> list[dict]:
    return [line for line in lines if line["type"] == "response"]


def _check_session(run: dict, world: dict, requests: int, model: ScriptedModel) -> list[str]:
    """The session ended cleanly with one ``response`` per request, each in time; no line's content holds a host
    path of the test's home. Returns the responses' contents as a driver reads them."""
    lines = run["lines"]
    assert run["code"] == 0, (lines, _why(model))
    types = [line["type"] for line in lines if line["type"] not in ("storage", "record")]
    assert types == ["response"] * requests + ["result", "ended"], (lines, _why(model))
    for line, seconds in zip(lines, run["waited"]):
        if line["type"] == "response":
            first = line is _responses(lines)[0]
            bound = FIRST_RESPONSE_BOUND_S if first else NEXT_RESPONSE_BOUND_S
            assert seconds < bound, (line, seconds)
        else:
            assert seconds < END_BOUND_S, (line, seconds)
    root = Path(world["state"]).parent
    hosts = {str(root), os.path.realpath(root), str(world["home"]), str(world["state"])}
    for line in lines:
        if line["type"] in ANSWER_TYPES and "content" in line:
            content = str(line["content"])
            for host in hosts:
                assert host not in content, (line["type"], content[:500])
    return [str(line["content"]) for line in _responses(lines)]


def _assert_in_code_requests(model: ScriptedModel, opened: list) -> None:
    """Every actor request carried ``execute_code`` alone; the request's in-code session ended with its box."""
    for call in model.calls:
        assert call.request["tools"] == R.json_tools("actor")
    assert [s.agent for s in opened] == ["actor"]
    for session in opened:
        assert session.closed and not session.box_alive() and not session.threads_alive()


# ── one request ──────────────────────────────────────────────────────────────

CASES = {
    # the cell runs, the next turn answers in plain text
    "text": (
        [reply(calls=[code("print(6*7)")]), reply(SUBMIT_TEXT)],
        SUBMIT,
    ),
    # the cell's tools.answer with a dict: its JSON text is the answer, no model call after the cell
    "answer-dict": (
        [reply(calls=[code(f"tools.answer(answer={SUBMIT!r})")])],
        SUBMIT,
    ),
    # the cell's tools.answer with a string: the string is the answer as it is
    "answer-string": (
        [reply(calls=[code(f"tools.answer(answer={DEMO_TEXT!r})")])],
        {"action": "request_demo"},
    ),
}


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(TEST_BOUND_S)
@pytest.mark.parametrize("case", sorted(CASES))
async def test_the_answer_reaches_the_session(session_world, monkeypatch, case):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    script, expected = CASES[case]
    model = ScriptedModel(actor=list(script))
    world = session_world["world"]
    run = await _drive(world, monkeypatch, model, REQUEST, [])

    (content,) = _check_session(run, world, 1, model)
    assert last_object(content) == expected, (content, _why(model))
    if case == "text":
        assert content == SUBMIT_TEXT
        assert "42" in _tool_results(model.calls[1]), _why(model)
    elif case == "answer-dict":
        assert content == json.dumps(SUBMIT)
    else:
        assert content == DEMO_TEXT
    # no model call after the cell that ended the turn, and none of any other kind
    model.assert_used_up()
    assert model.kinds() == ["actor"] * len(script)
    _assert_in_code_requests(model, session_world["opened"])


# ── two requests in one persistent session ───────────────────────────────────

# The first request's cell binds ``kept``; the second request's cell reads it, so its answer is right only when the
# interpreter kept the variable between the requests (UNIFY_CELL_STATE's default, ``persistent``).
SECOND_GRID = "[[kept - 40, 2], [3, 4]]"


def _two_requests(mode: str) -> list:
    if mode == "text":
        return [
            reply(calls=[code("kept = 41\nprint(6*7)")]),
            reply(SUBMIT_TEXT),
            reply(calls=[code(f"grid = {SECOND_GRID}\nprint('KEPT', kept + 1)")]),
            reply(json.dumps({"action": "submit", "grid": [[1, 2], [3, 4]]})),
            reply(NOT_ENDED),
        ]
    return [
        reply(calls=[code(f"kept = 41\ntools.answer(answer={SUBMIT!r})")]),
        reply(
            calls=[
                code(
                    f"print('KEPT', kept + 1)\n"
                    f"tools.answer(answer={{'action': 'submit', 'grid': {SECOND_GRID}}})",
                ),
            ],
        ),
        reply(NOT_ENDED),
    ]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(TEST_BOUND_S)
@pytest.mark.parametrize("mode", ["text", "answer"])
async def test_a_second_request_in_the_session_is_answered_the_same_way(session_world, monkeypatch, mode):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    model = ScriptedModel(actor=_two_requests(mode))
    world = session_world["world"]
    run = await _drive(world, monkeypatch, model, REQUEST, [FOLLOW_UP])

    first, second = _check_session(run, world, 2, model)
    assert last_object(first) == SUBMIT, (first, _why(model))
    assert last_object(second) == SUBMIT, (second, _why(model))
    assert NOT_ENDED not in (first, second)
    # the follow-up reached the model as the second request
    assert any(FOLLOW_UP in json.dumps(c.messages) for c in model.calls[1:])
    if mode == "text":
        # the second request's cell read the variable the first request's cell bound
        assert "KEPT 42" in _tool_results(model.calls[3]), _why(model)
        assert model.kinds() == ["actor"] * 4
    else:
        assert model.kinds() == ["actor"] * 2
    assert model.unused() == {"actor": 1}
    _assert_in_code_requests(model, session_world["opened"])


# ── the tools off ────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(TEST_BOUND_S)
async def test_off_a_text_answer_reaches_the_session_the_same(session_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    model = ScriptedModel(actor=[reply(calls=[cell("print(6*7)")]), reply(SUBMIT_TEXT)])
    world = session_world["world"]
    run = await _drive(world, monkeypatch, model, REQUEST, [])

    (content,) = _check_session(run, world, 1, model)
    assert content == SUBMIT_TEXT
    assert last_object(content) == SUBMIT
    assert "42" in _tool_results(model.calls[1]), _why(model)
    model.assert_used_up()
    assert model.kinds() == ["actor", "actor"]
    # the switch off: the actor's own tools, no in-code session
    assert model.calls[0].request["tools"] != R.json_tools("actor")
    assert session_world["opened"] == []
