"""Symbolic: ``UNIFY_PENDING_TIMEOUT_S`` ends a wait on a question nobody can answer.

The hang of ``test_unanswerable_clarification.py`` (Continual-ARC, 5-6
October, instance 24): a sub-actor of a ``unify act --no-clarify`` session
asks a question that only its own model reads, then calls ``wait()`` (which
the loop refuses, and then waits anyway) or stops its latest question while
earlier ones stay blocked. Its loop parks on calls only an answer can end,
and the session says nothing until the host's idle timeout.

With the switch, once every call a loop waits on is a question and none has
been answered for that long, the model gets a plain notice that no answer
has arrived and takes its turn; nothing is cancelled and no tool is added.
Here the sub-actor then answers without it, and the session replies. The
transport is scripted; every wait is bounded.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.test_unanswerable_clarification import (  # noqa: F401
    CHILD_ANSWER,
    PARENT_ANSWER,
    _Model,
    _session,
    _tool_names,
    jsonl_session,
)
from unify.settings import ProductionSettings, SETTINGS

TIMEOUT_S = 0.3
#: A session that does not hang answers well within this.
BOUND = 2.0
NOTICE = "No answer has arrived to the question of"


class _GivesUpWhenTold(_Model):
    """The recorded sub-actor, which answers once told no answer came."""

    def _child(self, messages: list, tools: set[str]):
        if NOTICE in json.dumps(messages, default=str):
            return h.completion(content=CHILD_ANSWER)
        return super()._child(messages, tools)


@pytest.fixture
def as_shipped_prompts(monkeypatch):
    # The sub-actor is offered the question, as in the recorded runs.
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_ACCURACY", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "")


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("give_up", ["wait", "stop"])
async def test_off_the_session_still_waits(
    jsonl_session,
    as_shipped_prompts,
    monkeypatch,
    give_up,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PENDING_TIMEOUT_S", 0.0)
    model = _GivesUpWhenTold(give_up)
    answered, lines = await _session(jsonl_session, model, bound=BOUND)
    assert not answered
    assert "request_clarification" in _tool_names(model.child[0]["tools"])


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("give_up", ["wait", "stop"])
async def test_on_the_sub_actor_is_told_and_the_session_answers(
    jsonl_session,
    as_shipped_prompts,
    monkeypatch,
    give_up,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_PENDING_TIMEOUT_S", TIMEOUT_S)
    model = _GivesUpWhenTold(give_up)
    answered, lines = await _session(jsonl_session, model, bound=BOUND)

    assert answered
    assert [line["content"] for line in lines if line["type"] == "response"] == [
        PARENT_ANSWER,
    ]
    assert [line["type"] for line in lines][-2:] == ["result", "ended"]
    # The notice reached the sub-actor's model once, as a plain message;
    # its tools are the ones it had (nothing added, nothing forced).
    told = [r for r in model.child if NOTICE in json.dumps(r["messages"], default=str)]
    # One notice; the turn it starts is sent as any turn with calls pending
    # is (tool_choice "required" under the shipped UNIFY_PENDING_REQUIRED, so
    # a text reply costs unillm's one tool-choice retry, as it always has).
    assert 1 <= len(told) <= 2
    assert _tool_names(told[0]["tools"]) == _tool_names(model.child[0]["tools"])
    assert told[0].get("tool_choice") == model.child[1].get("tool_choice")
    notices = [
        m
        for m in told[0]["messages"]
        if m.get("role") == "user" and NOTICE in str(m.get("content"))
    ]
    assert len(notices) == 1
    assert f"after {TIMEOUT_S:g}s" in notices[0]["content"]


@pytest.mark.parametrize(("raw", "value"), [("", 0.0), ("0", 0.0), ("45", 45.0)])
def test_the_switch_parses_seconds(monkeypatch, raw, value):
    monkeypatch.setenv("UNIFY_PENDING_TIMEOUT_S", raw)
    assert ProductionSettings(_env_file=None).UNIFY_PENDING_TIMEOUT_S == value


def test_the_switch_refuses_a_negative_time(monkeypatch):
    monkeypatch.setenv("UNIFY_PENDING_TIMEOUT_S", "-1")
    with pytest.raises(ValueError, match="UNIFY_PENDING_TIMEOUT_S"):
        ProductionSettings(_env_file=None)
