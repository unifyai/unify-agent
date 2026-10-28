"""Symbolic: with the tools in code the actor ends a request with ``tools.answer(answer)`` and has no ``finish``.

A program driving the harness runs ``unify act --persist --jsonl --no-clarify --quiet --home <dir> "<request>"``,
reads JSON lines from stdout until one whose ``type`` is ``response``, ``result`` or ``ended``, and sends later
requests as stdin lines ``{"message": ...}``; ``{"quit": true}`` ends the session. The session here is driven in
process on the real ``CodeActActor`` and the real sandboxed interpreter, with the scripted model
(tests/scripted_model.py): no key, no network.

* Two requests of one session are each answered by ``tools.answer({...})`` from a cell, and each response carries
  its answer; the second answer is a benchmark's own ``{"action": "finish"}``, which reaches the host unchanged.
* The actor's system prompt and its ``tools.list()`` name no ``finish``; the writer's and the librarian's tool lists
  still do.
* The actor's system prompt states the answer rule once, in its role, and says one line about the record, pointing
  to ``tools.help("record")``, which shows how the record works.
* ``tools.finish(...)`` in an actor cell fails as a tool the actor does not have (the box module has no such
  function, and the host refuses the name as an unknown tool), and the request goes on to be answered by
  ``tools.answer``.
"""

from __future__ import annotations

import json
import re

import pytest

from tests.actor.code_act.core_world import core_world  # noqa: F401 (fixture)
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from tests.actor.code_act.test_answer_reaches_the_session_under_tools_in_code import (  # noqa: F401 (fixture)
    FOLLOW_UP,
    NOT_ENDED,
    REQUEST,
    SUBMIT,
    TEST_BOUND_S,
    _assert_in_code_requests,
    _check_session,
    _drive,
    _text,
    _tool_results,
    _why,
    code,
    last_object,
    session_world,
)
from tests.actor.code_act.test_in_code_tools_wiring import _channel_rows, _rows, _system_texts
from tests.scripted_model import ScriptedModel, reply
from unify.common import tool_registry as R

#: A benchmark whose own protocol ends an instance with a "finish" action (the ARC adapter's end="finish") gets it
#: as the request's answer, given by tools.answer like any other answer.
SECOND = {"action": "finish"}
#: The cell that tries the end call the actor no longer has: through the host by name (refused there as an unknown
#: tool), then as a function of the box module (which has none).
FINISH_CELL = (
    "try:\n"
    "    tools._call('finish', {'answer': 'early'})\n"
    "except tools.Refused as exc:\n"
    "    print('HOST', exc.form)\n"
    f"tools.finish(answer={SUBMIT!r})\n"
)
#: The role's one answer rule (whitespace collapsed: the role is wrapped), and the opening's one line about the record.
ROLE_ANSWER = (
    "Your answer is your final reply: a message without a tool call. `tools.answer(answer)` gives it from a cell."
)


def _last_tool_result(call) -> str:
    tools = [_text(m.get("content")) for m in call.messages if m.get("role") == "tool"]
    return tools[-1] if tools else ""


def _names_finish(text: str) -> bool:
    return "tools.finish" in text or re.search(r"\bfinish\b", text) is not None


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(TEST_BOUND_S)
async def test_each_request_is_answered_by_tools_answer_and_finish_is_unknown(session_world, monkeypatch):
    model = ScriptedModel(
        actor=[
            # the first request: the tool list and the record's help, the end call the actor does not have, then its
            # answer
            reply(calls=[code('print(tools.list())\nprint(tools.help("record"))\n')]),
            reply(calls=[code(FINISH_CELL)]),
            reply(calls=[code(f"tools.answer({SUBMIT!r})\n")]),
            # the second request: its answer
            reply(calls=[code(f"tools.answer({SECOND!r})\n")]),
            reply(NOT_ENDED),
        ],
    )
    world = session_world["world"]
    run = await _drive(world, monkeypatch, model, REQUEST, [FOLLOW_UP])

    # each request was answered by its tools.answer, and its response carries the answer
    first, second = _check_session(run, world, 2, model)
    assert first == json.dumps(SUBMIT) and last_object(first) == SUBMIT, (first, _why(model))
    assert second == json.dumps(SECOND) and last_object(second) == SECOND, (second, _why(model))
    assert NOT_ENDED not in (first, second)
    # no model call after a cell that answered: three for the first request, one for the second
    assert model.kinds() == ["actor"] * 4
    assert model.unused() == {"actor": 1}
    assert FOLLOW_UP not in json.dumps(model.calls[2].messages)
    assert FOLLOW_UP in json.dumps(model.calls[3].messages)
    _assert_in_code_requests(model, session_world["opened"])

    # the actor's system prompt and its tool list name answer, never finish
    (session,) = session_world["opened"]
    system = "\n".join(_system_texts(model.calls[0]))
    assert "tools.answer(answer)" in system, system[:2000]
    assert not _names_finish(system), system
    # one answer rule, in the role; the opening names no record (the run record's section comes with the first message)
    assert ROLE_ANSWER in " ".join(system.split()), system[:2000]
    assert "You end your work with" not in system and "Messages: other agents" not in system, system[:3000]
    assert R.COMMON_SECTION not in system, system[:3000]
    first_user = next(m for m in model.calls[0].messages if m.get("role") == "user")
    assert R.COMMON_SECTION in str(first_user.get("content")), first_user
    listing = _tool_results(model.calls[1])
    for line in R.list_text(session.allow, "actor").splitlines():
        assert line in listing, (line, listing)
    assert "answer(answer) - " in listing and "read(" in listing, listing
    assert "finish" not in listing, listing
    # print(tools.help("record")) showed how the record works: the run record's common section
    assert R.COMMON_SECTION in listing, listing
    assert "answer" in session.allow and "finish" not in session.allow
    assert "answer" in session.specs and "finish" not in session.specs

    # tools.finish failed in the cell, by name on the host and as a function of the box module; the request went on
    tried = _last_tool_result(model.calls[2])
    assert "HOST unknown_tool" in tried, tried
    assert "AttributeError" in tried and "tools.finish isn't available in your role" in tried, tried
    calls = _channel_rows(_rows(session))
    assert [(r["tool"], r["status"]) for r in calls] == [("finish", "refused"), ("answer", "ok"), ("answer", "ok")]
    assert calls[0]["refusal"]["form"] == "unknown_tool"


def test_the_writer_and_the_librarian_still_list_finish():
    writer = R.list_text(R.allow_list("writer"), "writer")
    librarian = R.list_text(R.allow_list("librarian"), "librarian")
    assert "finish(summary) - " in writer and "answer(" not in writer
    assert "finish(answer) - " in librarian and "answer(answer) - " not in librarian
    actor = R.list_text(R.allow_list("actor", frozenset({"helper"})), "actor")
    assert "answer(answer) - " in actor and "finish" not in actor
