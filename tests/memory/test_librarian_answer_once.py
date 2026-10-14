"""One librarian answer per question (``UNIFY_MEMORY_LIBRARIAN=on``): the request's session keeps the first answer the
librarian gives for a question about the current request with the given facts against the library copy's commit, and
returns it for every later ``memory.ask()`` of the same, marked as already given, with no model call. Other facts, or
another question, are asked anew; nothing is refused for asking again. The use record counts the asks, the answers
given from the cache and the model calls apart.

The host-side tests drive ``ask_library`` with a model turn of their own and the librarian's ``finish`` run on the host
in place of its box, so no box and no network is needed; the last test runs ``memory.ask()`` twice in a real actor
box against a real librarian box, with the scripted model (``tests/scripted_model.py``), and needs bubblewrap."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.memory import test_librarian as T
from tests.scripted_model import reply, scripted
from unify.actor import tool_handlers as H
from unify.memory import librarian as L

TOTAL = "memory.tables.procedures:total"
READ_ROWS = "memory.tables.readers:read_total"
COUNT = "memory.tables.count:count_rows"
CATALOGUE = f"## Functions\n\n### {COUNT}\nHow many rows.\n\n### {TOTAL}\nSum every number.\n\n### {READ_ROWS}\nRead.\n"
REQUEST = "4096 8191 3\n5 7 11"
PARSED = [{"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"}]
NOT_PARSED = [{"id": TOTAL, "reader": READ_ROWS, "read": "did-not-parse", "applies": "not-run"}]
FITS = {"fits": [{"id": TOTAL, "call": f"memory.run({TOTAL!r}, text)", "why": "its reader parsed the text"}]}
NEAREST = {"fits": [], "nearest": [{"id": COUNT, "differs": "it counts rows; the reader did not parse the text"}]}


def _copy(tmp_path: Path) -> Path:
    """A library copy as the librarian reads it: its dispatch file names the procedure and its reader."""
    root = tmp_path / "copy"
    (root / ".memory").mkdir(parents=True)
    rows = [{"id": TOTAL, "reader": READ_ROWS, "applies": True, "input": "{}"}]
    (root / ".memory" / "dispatch.json").write_text(json.dumps({"procedures": rows, "applies_of": {}}))
    return root


class Turn:
    """The librarian's model, one answer per call: each turn is a cell that finishes with the next answer."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list, tools: list) -> tuple[dict, Any]:
        self.calls.append([dict(m) for m in messages])
        answer = self.answers.pop(0)
        call = {
            "id": f"call_{len(self.calls)}",
            "type": "function",
            "function": {"name": "execute_code", "arguments": json.dumps({"code": json.dumps(answer)})},
        }
        usage = {"prompt_tokens": 100, "completion_tokens": 5}
        return {"role": "assistant", "content": None, "tool_calls": [call], "usage": usage}, "0.001"


def _session(tmp_path: Path, turn: Turn, library: str | None = "c0ffee") -> L.LibrarianSession:
    """A session whose cells run ``tools.finish(answer)`` on the host (its box is never started)."""
    session = L.LibrarianSession(
        copy=_copy(tmp_path), catalogue=CATALOGUE, turn=turn, compact=False, library=library,
    )

    async def tool_call(tc: dict) -> str:
        answer = json.loads(json.loads(tc["function"]["arguments"])["code"])
        return (await L.finish(SimpleNamespace(state=session.state), {"answer": answer})).text

    session._tool_call = tool_call
    return session


def _ctx(session: L.LibrarianSession, request: str | None = REQUEST) -> SimpleNamespace:
    async def open_librarian() -> L.LibrarianSession:
        return session

    return SimpleNamespace(
        agent="actor", state=H.ActorRequestState(open_librarian=open_librarian, request_text=request),
    )


async def _ask(ctx: SimpleNamespace, facts: list, question: str = REQUEST) -> tuple[str, dict]:
    got = await H.ask_library(ctx, {"question": question, "facts": facts})
    assert got.status == "ok", got
    return got.text, json.loads(got.text)


@pytest.mark.asyncio
async def test_two_asks_in_a_request_get_one_answer_the_second_from_the_cache(tmp_path):
    turn = Turn([FITS])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    try:
        first_text, first = await _ask(ctx, PARSED)
        second_text, second = await _ask(ctx, PARSED)
    finally:
        await session.close()
    # the first answer is exactly as the librarian gave it, with no mark
    assert first == FITS and L.ALREADY_GIVEN not in first
    assert first_text == json.dumps(FITS, ensure_ascii=False, sort_keys=True)
    # the second is the same answer, marked as already given, and no model call was made for it
    assert second == {**FITS, L.ALREADY_GIVEN: True}
    assert len(turn.calls) == 1 and session.usage.calls == 1
    assert sum(1 for m in session.messages if m.get("role") == "user") == 1  # one question reached the librarian
    assert session.questions == 2 and session.from_cache == 1
    fields = L.usage_fields([session])
    assert (fields["questions"], fields["answers_from_cache"], fields["calls"]) == (2, 1, 1)
    assert fields["input_tokens"] == 100 and fields["usd"] == "0.001"


@pytest.mark.asyncio
async def test_other_facts_or_another_question_are_asked_again_one_answer_per_key(tmp_path):
    turn = Turn([FITS, NEAREST, NEAREST])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    try:
        _, one = await _ask(ctx, PARSED)
        _, two = await _ask(ctx, NOT_PARSED)  # the readers' outcomes differ: asked again
        _, three = await _ask(ctx, PARSED)  # the first facts again: the first answer
        _, four = await _ask(ctx, NOT_PARSED)  # the second facts again: the second answer
        _, five = await _ask(ctx, PARSED, question="And is there one that counts rows?")  # another question
    finally:
        await session.close()
    assert one == FITS and two == NEAREST and five == NEAREST
    assert three == {**FITS, L.ALREADY_GIVEN: True} and four == {**NEAREST, L.ALREADY_GIVEN: True}
    assert len(turn.calls) == 3 and session.usage.calls == 3
    # the second question carried its own facts
    asked = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert len(asked) == 3 and "did not parse the text" in asked[1] and "counts rows" in asked[2]
    fields = L.usage_fields([session])
    assert (fields["questions"], fields["answers_from_cache"], fields["calls"]) == (5, 2, 3)


def test_the_key_is_the_question_the_request_the_checked_facts_and_the_library():
    key = L.answer_key(REQUEST, REQUEST, PARSED, 0, "c0ffee")
    assert key == L.answer_key(REQUEST, REQUEST, [dict(PARSED[0])], 0, "c0ffee")
    for other in (
        L.answer_key("another question", REQUEST, PARSED, 0, "c0ffee"),
        L.answer_key(REQUEST, "another request", PARSED, 0, "c0ffee"),
        L.answer_key(REQUEST, REQUEST, NOT_PARSED, 0, "c0ffee"),
        L.answer_key(REQUEST, REQUEST, PARSED, 1, "c0ffee"),
        L.answer_key(REQUEST, REQUEST, PARSED, 0, "beef"),
    ):
        assert other != key


@pytest.mark.asyncio
async def test_a_question_without_an_answer_is_kept_for_nothing_and_asked_again(tmp_path):
    turn = Turn([{"fit": []}, FITS])  # a malformed answer, then the right one

    session = _session(tmp_path, turn)
    session.step_guard = 1  # the first ask ends at the guard, without an accepted answer
    ctx = _ctx(session)
    try:
        refused = await H.ask_library(ctx, {"question": REQUEST, "facts": PARSED})
        assert refused.status == "refused" and refused.form == "no_answer"
        _, again = await _ask(ctx, PARSED)
    finally:
        await session.close()
    assert again == FITS and len(turn.calls) == 2 and session.from_cache == 0


# --- in real boxes ------------------------------------------------------------------------------------------------


@T.needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_memory_ask_twice_in_a_cell_makes_one_librarian_call_and_marks_the_second_answer(tmp_path):
    from unify.common.tool_channel import HandlerThreads
    from unify.memory import library_export as lx
    from unify.memory import tool_handlers as th
    from unify.memory.writer_pass import unillm_turn

    copy = T._copy(tmp_path)
    catalogue = (copy / lx.CATALOGUE_FILE).read_text()
    opened: list[L.LibrarianSession] = []

    async def open_librarian() -> L.LibrarianSession:
        session = L.LibrarianSession(
            copy=copy,
            catalogue=catalogue,
            turn=unillm_turn("openai/gpt-5.6-sol@openrouter", "low", origin=L.ORIGIN),
            request_id="r1",
            library="c0ffee",
        )
        opened.append(session)
        return session

    state = H.ActorRequestState(open_librarian=open_librarian, request_text=T.REQUEST)
    actor = th.InterpreterSession(
        agent="actor",
        name="root",
        role="actor",
        pass_id="r1",
        state=state,
        allow=("ask_library", "current_request"),
        binds={"ro": {copy: "/library"}, "rw": {}},
        env={},
        roots={"/library": copy},
        deadline=time.monotonic() + 3600,
        cell_limit_s=200.0,
        on_event=lambda row: None,
        threads=HandlerThreads(),
        profile=th.memory_profile(
            cwd="/library", sys_path=("/library",), audit_roots=("/library",), open_areas=("/library",),
        ),
    )
    cell = (
        "import json, memory\n"
        "a = memory.ask()\n"
        "b = memory.ask()\n"
        "print('ASKED', json.dumps([[[i.id, i.reason] for i in a], a.already_given, [[i.id, i.reason] for i in b], "
        "b.already_given]))\n"
        "print(str(b))\n"
    )
    model = T.HeaderModel(actor=[reply(calls=[T.code(T.LIBRARIAN_SECOND)])])
    try:
        with scripted(model):
            result = await actor.cell(cell, 0)
    finally:
        await actor.close()
        if state.librarian is not None:
            await state.librarian.close()
    shown = result.output + (result.error or "")
    assert "ASKED" in shown, shown
    a, a_again, b, b_again = json.loads(shown.split("ASKED ", 1)[1].splitlines()[0])
    assert a == b == [[T.TOTAL, "reader read_rows parsed the text"]]
    assert a_again is False and b_again is True
    assert "The librarian already gave this answer in this request." in shown
    (session,) = opened
    assert len(model.calls) == 1 and session.usage.calls == 1
    assert session.questions == 2 and session.from_cache == 1


# --- when a kept answer no longer answers: the library, the facts, what the request received, the question ----------


@pytest.mark.asyncio
async def test_a_library_that_changed_since_the_answer_is_asked_again(tmp_path):
    turn = Turn([FITS, NEAREST])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    try:
        await _ask(ctx, PARSED)
        # the copy changes after the first ask (its dispatch file is read at every ask, whatever the commit says)
        dispatch = session.copy / ".memory" / "dispatch.json"
        data = json.loads(dispatch.read_text())
        data["procedures"].append({"id": COUNT, "reader": None, "applies": True, "input": ""})
        dispatch.write_text(json.dumps(data))
        _, second = await _ask(ctx, PARSED)
    finally:
        await session.close()
    assert second == NEAREST and len(turn.calls) == 2


@pytest.mark.asyncio
async def test_a_new_observation_asks_again_and_nothing_new_keeps_the_answer(tmp_path):
    turn = Turn([FITS, NEAREST])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    seen = {"n": 0}
    ctx.state.observed = lambda: f"calls:{seen['n']}"
    try:
        await _ask(ctx, PARSED)
        _, again = await _ask(ctx, PARSED)  # nothing received since: the kept answer
        seen["n"] += 1  # a tool's result or a user message since the last ask
        _, third = await _ask(ctx, PARSED)
    finally:
        await session.close()
    assert again == {**FITS, L.ALREADY_GIVEN: True}
    assert third == NEAREST and len(turn.calls) == 2


def test_the_runs_observed_changes_with_each_user_message_and_nothing_else():
    import hashlib

    from unify.memory.integration.request import RequestRun

    run = SimpleNamespace(_observed=hashlib.sha256(), observed_messages=0, journal=None)

    def line(role: str, content: str = "x") -> str:
        return json.dumps({"seq": 1, "type": "message", "message": {"role": role, "content": content}})

    before = RequestRun.observed(run)
    for role in ("assistant", "tool", "system"):
        RequestRun._on_transcript_line(run, line(role))
    assert RequestRun.observed(run) == before  # what the agent wrote, or its own cells, is no observation
    RequestRun._on_transcript_line(run, line("user", "the environment's next observation"))
    after = RequestRun.observed(run)
    assert after != before
    RequestRun._on_transcript_line(run, line("user", "another"))
    assert RequestRun.observed(run) not in (before, after)


@pytest.mark.asyncio
async def test_the_agents_observed_changes_with_each_tool_call_but_the_memory_packages_own(tmp_path, monkeypatch):
    from unify.actor import in_code_tools
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    session = in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events",
    )
    try:
        before = session.state.observed()
        for own in ("ask_library", "current_request"):
            session._on_event({"event": "tool_call", "tool": own, "status": "ok", "via": "channel"})
        assert session.state.observed() == before
        session._on_event({"event": "tool_call", "tool": "read", "status": "ok", "via": "channel", "result_bytes": 10})
        assert session.state.observed() != before
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_answers_never_cross_requests(tmp_path):
    # each request opens its own session (the actor's request state holds it; it ends with the request), so a
    # second request asks afresh what the first was answered
    first_turn, second_turn = Turn([FITS]), Turn([FITS])
    first, second = _session(tmp_path / "a", first_turn), _session(tmp_path / "b", second_turn)
    try:
        await _ask(_ctx(first), PARSED)
        _, got = await _ask(_ctx(second), PARSED)
    finally:
        await first.close()
        await second.close()
    assert got == FITS and len(second_turn.calls) == 1  # no mark: not from the other request's answers
