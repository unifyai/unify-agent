"""One librarian answer per question (``UNIFY_MEMORY_LIBRARIAN=on``): the request's session keeps the first answer the
librarian gives for a question about the current request against the library copy's commit and catalogue, and
returns it for every later ``memory.ask()`` of the same, marked as already given, with no model call. Another
question is asked anew (rev 2: no reader runs, so facts a cell may still send change nothing); nothing is refused for asking again. The use record counts the asks, the answers
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
REQUEST = "4096 8191 3\n5 7 11"
PARSED = [{"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"}]
NOT_PARSED = [{"id": TOTAL, "reader": READ_ROWS, "read": "did-not-parse", "applies": "not-run"}]
FITS = {"fits": [{"id": TOTAL, "call": f"memory.run({TOTAL!r}, text)", "why": "its reader parsed the text"}]}
NEAREST = {"fits": [], "nearest": [{"id": COUNT, "differs": "it counts rows; the reader did not parse the text"}]}


def _copy(tmp_path: Path) -> Path:
    """A library copy as the librarian reads it: its items, and its dispatch file naming the procedure and its
    reader."""
    root = tmp_path / "copy"
    (root / ".memory").mkdir(parents=True)
    rows = [{"id": TOTAL, "reader": READ_ROWS, "applies": True, "input": "{}"}]
    (root / ".memory" / "dispatch.json").write_text(json.dumps({"procedures": rows, "applies_of": {}}))
    items = {i: {"kind": "function"} for i in (COUNT, TOTAL, READ_ROWS)}
    (root / ".memory" / "items.json").write_text(json.dumps({"version": 1, "items": items}))
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
        copy=_copy(tmp_path), turn=turn, library=library,
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
async def test_another_question_is_asked_again_one_answer_per_key(tmp_path):
    """Rev 2: no reader runs, so the facts a cell may still send change nothing; the question does."""
    turn = Turn([FITS, NEAREST])
    session = _session(tmp_path, turn)
    ctx = _ctx(session)
    other = "And is there one that counts rows?"
    try:
        _, one = await _ask(ctx, PARSED)
        _, two = await _ask(ctx, NOT_PARSED)  # other facts, the same question: the first answer
        _, three = await _ask(ctx, PARSED, question=other)  # another question: asked
        _, four = await _ask(ctx, [], question=other)  # that question again: its answer
    finally:
        await session.close()
    assert one == FITS and three == NEAREST
    assert two == {**FITS, L.ALREADY_GIVEN: True} and four == {**NEAREST, L.ALREADY_GIVEN: True}
    assert len(turn.calls) == 2 and session.usage.calls == 2
    asked = [m["content"] for m in session.messages if m.get("role") == "user"]
    assert len(asked) == 2 and "counts rows" in asked[1] and all("parse" not in a for a in asked)
    fields = L.usage_fields([session])
    assert (fields["questions"], fields["answers_from_cache"], fields["calls"]) == (4, 2, 2)


def test_the_key_is_the_digest_of_exactly_what_the_librarian_is_sent():
    library = {"commit": "c0ffee", "dispatch": "d1"}
    sent = L.sent_to_librarian(REQUEST, library, rows=1)
    # the message in the structure is the one the session sends
    assert sent["message"] == L.question_message(REQUEST, 1)
    key = L.answer_key(sent)
    assert key == L.answer_key(L.sent_to_librarian(REQUEST, dict(library), rows=1))
    for other in (
        L.sent_to_librarian("another question", library, rows=1),
        L.sent_to_librarian(REQUEST, library, rows=2),
        L.sent_to_librarian(REQUEST, {"commit": "beef", "dispatch": "d1"}, rows=1),
        L.sent_to_librarian(REQUEST, {"commit": "c0ffee", "dispatch": "d2"}, rows=1),
    ):
        assert L.answer_key(other) != key


@pytest.mark.asyncio
async def test_the_session_sends_the_message_its_key_was_made_from(tmp_path):
    turn = Turn([FITS])
    session = _session(tmp_path, turn)
    try:
        await _ask(_ctx(session), PARSED)
    finally:
        await session.close()
    (sent_message,) = [m["content"] for m in session.messages if m.get("role") == "user"]
    library = {"commit": "c0ffee", "dispatch": L._digest_of(L.read_dispatch(session.copy))}
    sent = L.sent_to_librarian(REQUEST, library, rows=len(session.dispatch))
    assert sent_message == sent["message"] and list(session.answers) == [L.answer_key(sent)]


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
async def test_answers_never_cross_requests(tmp_path):
    # each request opens its own session (held on that request's state; it ends with the request), so a second
    # request asks afresh what the first was answered
    first_turn, second_turn = Turn([FITS]), Turn([FITS])
    first, second = _session(tmp_path / "a", first_turn), _session(tmp_path / "b", second_turn)
    try:
        await _ask(_ctx(first), PARSED)
        _, got = await _ask(_ctx(second), PARSED)
    finally:
        await first.close()
        await second.close()
    assert got == FITS and len(second_turn.calls) == 1  # no mark: not from the other request's answers


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
    from unify.memory import tool_handlers as th
    from unify.memory.writer_pass import unillm_turn

    copy = T._copy(tmp_path)
    opened: list[L.LibrarianSession] = []

    async def open_librarian() -> L.LibrarianSession:
        session = L.LibrarianSession(
            copy=copy,
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
