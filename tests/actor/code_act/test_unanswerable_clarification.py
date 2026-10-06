"""Symbolic: a sub-actor of a ``unify act --no-clarify`` session asks a question.

In the Continual-ARC runs of 6 October on Unify @ main 592685713
(``arc-pm2-up592-h0-medium-ws0`` and ``arc-pm2-up592-h0-high-fresh0``,
instance 24 of each) the session's actor delegated to a sub-actor with
``execute_function(primitives.actor.act)``. ``--no-clarify`` had taken
``request_clarification`` away from that actor, but the sub-actor was
started with it, and its question went to the only reader it had: its own
model, told to answer it with ``steer(clarify)``. In one run the sub-actor
called ``wait()``, which the loop refused (a clarification was pending) and
then waited anyway; in the other it stopped its latest question with
``steer(stop)`` and three earlier ones were still blocked. Either way its loop
parked on calls only its own model could end, the actor that started it was
parked on its result, nothing more was written, and the host restarted the
session after its idle timeout (300 s).

Under ``UNIFY_PROMPT_ACCURACY`` a sub-actor may ask only when the actor that
started it can, so in such a session it decides on its own and answers. The
transport is scripted (``tests/cache_discipline_helpers.py``) and the model
is a script that asks whenever ``request_clarification`` is offered: nothing
leaves the process, and every wait is bounded.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS

TASK = "Solve the grid puzzle and reply with your answer."
CHILD_REQUEST = "Please request one more demonstration pair for this task."
QUESTION = "Could you provide one more demonstration pair?"
CHILD_ANSWER = "No one can provide a demonstration pair here; none was obtained."
PARENT_ANSWER = "Answer: [[4]]"
#: How long a session that does not hang needs, with room to spare.
BOUND = 20


def _first_user_text(messages: list) -> str:
    for message in messages:
        if message.get("role") == "user":
            return json.dumps(message.get("content"), default=str)
    return ""


def _tool_names(tools) -> set[str]:
    return {t["function"]["name"] for t in tools or []}


def _clarification_calls(messages: list) -> list[str]:
    return [
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
        if call["function"]["name"] == "request_clarification"
    ]


class _Model:
    """The parent delegates; the sub-actor asks when it can, then gives up
    on the answer as in the recorded runs (``wait`` or ``steer(stop)``)."""

    def __init__(self, give_up: str) -> None:
        self.give_up = give_up
        self.parent: list[dict] = []
        self.child: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        messages = kw.get("messages") or []
        tools = _tool_names(kw.get("tools"))
        if CHILD_REQUEST in _first_user_text(messages):
            self.child.append(kw)
            return self._child(messages, tools)
        self.parent.append(kw)
        return self._parent(messages)

    def _parent(self, messages: list):
        text = json.dumps(messages, default=str)
        if not any(m.get("role") == "assistant" for m in messages):
            return h.completion(
                calls=[
                    (
                        "execute_function",
                        {
                            "thought": "Delegating the request.",
                            "function_name": "primitives.actor.act",
                            "call_kwargs": {"request": CHILD_REQUEST},
                        },
                    ),
                ],
            )
        if CHILD_ANSWER in text:
            return h.completion(content=PARENT_ANSWER)
        return h.completion(calls=[("wait", {})])

    def _child(self, messages: list, tools: set[str]):
        asked = _clarification_calls(messages)
        if "request_clarification" not in tools:
            return h.completion(content=CHILD_ANSWER)
        if self.give_up == "wait":
            if not asked:
                return h.completion(
                    calls=[("request_clarification", {"question": QUESTION})],
                )
            return h.completion(calls=[("wait", {})])
        # give_up == "stop": two questions, then the latest is stopped.
        if len(asked) < 2:
            return h.completion(
                calls=[("request_clarification", {"question": QUESTION})],
            )
        return h.completion(
            calls=[
                (
                    "steer",
                    {
                        "call_id": asked[-1],
                        "action": "stop",
                        "payload": "No answer is coming.",
                    },
                ),
            ],
        )


@pytest.fixture
def jsonl_session(monkeypatch):
    """``unify act --persist --jsonl --no-clarify`` on a real actor that can
    delegate; stdin is a pipe."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.actor.environments import ActorEnvironment
    from unify.cli import Act

    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))

    async def start(self) -> None:
        self._actor = CodeActActor(environments=[ActorEnvironment()], can_store=False)

    monkeypatch.setattr(Act, "start", start)
    session = Act(
        SimpleNamespace(
            persist=True,
            jsonl=True,
            quiet=True,
            no_clarify=True,
            no_compose=False,
            no_store=True,
            timeout=None,
        ),
    )
    lines: list[dict] = []
    session._emit = lambda **payload: lines.append(payload)

    def send(payload: dict) -> None:
        os.write(write_fd, (json.dumps(payload) + "\n").encode())

    yield session, lines, send
    os.close(write_fd)


async def _until(predicate, timeout: float) -> bool:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.05)

    try:
        await asyncio.wait_for(poll(), timeout)
    except asyncio.TimeoutError:
        return False
    return True


async def _session(
    jsonl_session,
    model,
    bound: float = BOUND,
) -> tuple[bool, list[dict]]:
    """Run the session until its first response (or ``bound`` seconds), then
    end it; every wait is bounded."""
    session, lines, send = jsonl_session
    with h.scripted(()):
        import unillm.clients.uni_llm as uni_llm

        uni_llm._acompletion_with_transient_retry = model
        run = asyncio.create_task(session.run(TASK))
        answered = await _until(
            lambda: any(line["type"] == "response" for line in lines),
            bound,
        )
        send({"quit": True})
        try:
            await asyncio.wait_for(run, BOUND)
        except asyncio.TimeoutError:
            # A hung session ends here, not at the test timeout.
            await session.close()
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)
    return answered, lines


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("give_up", ["wait", "stop"])
async def test_off_a_sub_actor_question_hangs_the_session(
    jsonl_session,
    monkeypatch,
    give_up,
):
    """As shipped: the recorded hang. Nothing answers within the bound."""
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    model = _Model(give_up)
    # The recorded sessions sat for 300 s; ten show the same.
    answered, lines = await _session(jsonl_session, model, bound=10)

    # The session's own actor never had the tool; its sub-actor did.
    assert all(
        "request_clarification" not in _tool_names(r["tools"]) for r in model.parent
    )
    assert "request_clarification" in _tool_names(model.child[0]["tools"])
    assert not answered
    assert [line["type"] for line in lines if line["type"] == "response"] == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("give_up", ["wait", "stop"])
@pytest.mark.parametrize(
    "accuracy, profile",
    [(True, ""), (False, "lean")],
    ids=["switch", "lean-profile"],
)
async def test_on_a_sub_actor_of_a_session_that_cannot_ask_answers(
    jsonl_session,
    monkeypatch,
    give_up,
    accuracy,
    profile,
):
    """The switch, or the lean profile that implies it (lean-all)."""
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", accuracy)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", profile)
    model = _Model(give_up)
    answered, lines = await _session(jsonl_session, model)

    assert answered
    responses = [line["content"] for line in lines if line["type"] == "response"]
    assert responses == [PARENT_ANSWER]
    assert [line["type"] for line in lines][-2:] == ["result", "ended"]
    # Neither loop was offered a question no one could answer.
    for request in model.parent + model.child:
        assert "request_clarification" not in _tool_names(request["tools"])
    assert len(model.child) == 1


class _TopLevelWaits:
    """The session's own actor calls ``wait()`` with nothing running."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw):
        self.requests.append(kw)
        messages = kw.get("messages") or []
        if not any(m.get("role") == "assistant" for m in messages):
            return h.completion(calls=[("wait", {})])
        return h.completion(content=PARENT_ANSWER)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("accuracy", [False, True])
async def test_a_top_level_wait_with_nothing_running_is_answered(
    jsonl_session,
    monkeypatch,
    accuracy,
):
    """The session's own actor cannot hang this way: with nothing running,
    ``wait`` is answered at once and the model takes the next turn."""
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", accuracy)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")
    model = _TopLevelWaits()
    answered, lines = await _session(jsonl_session, model)

    assert answered
    assert [line["content"] for line in lines if line["type"] == "response"] == [
        PARENT_ANSWER,
    ]
    assert len(model.requests) == 2
    replies = [
        m
        for m in model.requests[1]["messages"]
        if m.get("role") == "tool" and m.get("name") == "wait"
    ]
    assert [m["content"] for m in replies] == ["No tasks are currently running."]
    assert all(
        "request_clarification" not in _tool_names(r["tools"]) for r in model.requests
    )
