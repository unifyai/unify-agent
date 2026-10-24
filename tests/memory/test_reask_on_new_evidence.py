"""Re-ask on new evidence (rev 2.3).

As a new message of the request arrives, the harness asks the request's librarian again, in its live session, when new
evidence came since the last check: a message whose value-free shape is new for this request, a verdict, or a bad use
of an item (a bad use). It asks only while no fit stands, or the standing fit had a bad use; then the answer goes above the
message as one line in the one-primary form, with no count. The guide no longer tells the actor to ask again after a
new message.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from unify.actor import tool_handlers as H
from unify.memory import librarian as L
from unify.memory import library_prompts
from unify.memory.integration import request as R

A, B = "capture:00112233445566aa", "memory.tables.count:count_rows"
FIT_A = {"fits": [{"id": A, "call": f"memory.get({A!r})()", "why": "it re-runs the same work"}]}
NONE = {"fits": [], "nearest": []}


def _run(**kw):
    """A stand-in for the request's run with the real event and gate logic."""
    run = SimpleNamespace(
        request="Sort the rows by id.",
        standing_fit=None,
        message_shapes=set(),
        _reask_seen=(0, 0),
        checker_seen=[],
        bad_here={},
        held_here={},
        reasks=[],
        reask_state=None,
        reask_pending=False,
        _deferred_events=[],
        _observations_now=lambda: 2,
    )
    run.__dict__.update(kw)
    run.reask_events = lambda message: R.RequestRun.reask_events(run, message)
    run.note_answer = lambda answer: R.RequestRun.note_answer(run, answer)
    return run


# --- the events and the gate ------------------------------------------------------------------------------------


def test_a_message_of_a_shape_already_seen_is_no_event():
    run = _run()
    assert run.reask_events("Sort the rows by name.") == []  # the request's own shape


def test_a_message_of_a_new_shape_is_an_event_once():
    run = _run()
    assert run.reask_events('{"rows": [[1, 2]], "note": "x"}') == ["message"]
    assert run.reask_events('{"rows": [[3, 4]], "note": "y"}') == []


def test_a_verdict_and_a_bad_use_are_events_once_each():
    run = _run()
    run.checker_seen.append({"label": "fail"})
    run.bad_here[A] = {"how": "value", "at": 1}
    assert run.reask_events("Sort the rows by name.") == ["outcome", "failed_use"]
    assert run.reask_events("Sort the rows by name.") == []


def test_while_a_fit_stands_and_has_not_failed_nothing_is_asked():
    run = _run()
    run.note_answer(FIT_A)
    run.checker_seen.append({"label": "pass"})
    assert run.reask_events('{"new": "shape"}') == []
    # the events were taken: a later failure of the standing fit opens the gate on its own event
    run.bad_here[A] = {"how": "failed", "at": 2}
    assert run.reask_events("Sort the rows by name.") == ["failed_use"]


def test_an_answer_that_names_no_fit_leaves_no_fit_standing():
    run = _run()
    run.note_answer(FIT_A)
    assert run.standing_fit == A
    run.note_answer(NONE)
    assert run.standing_fit is None
    run.note_answer({"value": 3})  # a structured answer names no fit: the standing fit is unchanged
    assert run.standing_fit is None


# --- the line ---------------------------------------------------------------------------------------------------


def test_the_line_is_one_line_in_the_one_primary_form_with_no_count():
    two = {"fits": [FIT_A["fits"][0], {"id": B, "call": f"memory.get({B!r})(rows)", "why": "w " * 600}]}
    line = L.reask_line(two)
    assert "\n" not in line and len(line) <= L.REASK_CHARS
    assert line.startswith(L.REASK_HEAD) and f"{L.BEST_FIT_LABEL}: {A}" in line and L.FALLBACK_LABEL in line
    assert L.reask_line(NONE) == f"{L.REASK_HEAD} nothing in the library fits this request."


def _ask_stub(monkeypatch, answer: dict, asked: list):
    async def ask(ctx, question, *, schema=None, failed=None, situation=None):
        asked.append((question, situation))
        return SimpleNamespace(status="ok", text=json.dumps(answer))

    monkeypatch.setattr(H, "_ask_librarian", ask)


def test_the_reask_asks_about_the_request_with_the_new_message_and_records_it(monkeypatch):
    asked: list = []
    _ask_stub(monkeypatch, FIT_A, asked)
    run = _run(reask_state=SimpleNamespace(request_text="Sort the rows by id.", open_librarian=object()))
    run.checker_seen.append({"label": "fail"})
    line = asyncio.run(L.request_reask_line(run, "Your answer was wrong."))
    assert asked == [("Sort the rows by id.", "Your answer was wrong.")]
    assert line.startswith(L.REASK_HEAD) and A in line
    assert run.reasks == [{"obs": 2, "events": ["outcome"], "fit": A}]


def test_with_the_gate_shut_nothing_is_asked(monkeypatch):
    asked: list = []
    _ask_stub(monkeypatch, FIT_A, asked)
    run = _run(reask_state=SimpleNamespace(request_text="Sort the rows by id.", open_librarian=object()))
    run.note_answer(FIT_A)
    run.checker_seen.append({"label": "pass"})
    assert asyncio.run(L.request_reask_line(run, "Correct.")) == "" and asked == [] and run.reasks == []


def test_without_the_actors_state_nothing_is_asked(monkeypatch):
    asked: list = []
    _ask_stub(monkeypatch, FIT_A, asked)
    run = _run()
    run.checker_seen.append({"label": "fail"})
    assert asyncio.run(L.request_reask_line(run, "Wrong.")) == "" and asked == []


# --- the ask path ------------------------------------------------------------------------------------------------


class _Session:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls: list = []

    async def ask(self, question, **kw):
        self.calls.append((question, kw))
        return json.dumps(FIT_A)


@pytest.mark.parametrize("evidence", [None, {}])
def test_the_reask_sends_the_new_message_as_the_situation(monkeypatch, evidence):
    session = _Session(evidence)
    run = SimpleNamespace(note_answer=lambda a: run.__dict__.setdefault("noted", a))
    monkeypatch.setattr(R, "current", lambda: run)
    monkeypatch.setattr(H, "_memory_run_failed", lambda: [A])
    monkeypatch.setattr(H, "_memory_run_failed_at", lambda: ({A: 1}, 1))
    ctx = SimpleNamespace(state=SimpleNamespace(librarian=session, request_text="the request"), agent="actor")
    asyncio.run(H._ask_librarian(ctx, "the request", situation="the new message"))
    (question, kw), = session.calls
    if evidence is None:
        assert question == "the request\n\nThe latest message of this request:\nthe new message"
    else:
        assert question == "the request" and kw["last_message"] == "the new message"
    # the message the ask is on counts: an item that failed at message 1 may be reconsidered on message 2
    assert kw["failed"] == [A] and kw["failed_at"] == {A: 1} and kw["messages_now"] == 2
    assert run.noted == FIT_A


def test_the_actors_own_ask_counts_no_message_it_has_not_received(monkeypatch):
    session = _Session({})
    monkeypatch.setattr(R, "current", lambda: None)
    monkeypatch.setattr(H, "_memory_run_failed", lambda: [A])
    monkeypatch.setattr(H, "_memory_run_failed_at", lambda: ({A: 1}, 1))
    monkeypatch.setattr(H, "_memory_run_last_message", lambda: "the latest")
    ctx = SimpleNamespace(state=SimpleNamespace(librarian=session, request_text="the request"), agent="actor")
    asyncio.run(H.ask_library(ctx, {"question": "what fits?"}))
    (question, kw), = session.calls
    assert kw["messages_now"] == 1 and kw["last_message"] == "the latest"


def test_the_librarian_reads_a_message_without_the_line_the_harness_put_above_it(mv2):  # noqa: F811
    from unify import transcripts

    run = _begin(mv2, "hi")
    try:
        path = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"type": "message", "message": {"role": "user", "content": "the request"}},
            {"type": "message", "message": {"role": "assistant", "content": "an answer"}},
            {"type": "message", "message": {"role": "user", "content": f"{L.REASK_HEAD} x\n\nthe new message"}},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert run.last_message() == "the new message"
    finally:
        _abort(mv2, run)


def test_the_guide_no_longer_says_to_ask_again_after_a_new_message():
    assert "after a new message arrives" not in library_prompts.ASK_LINE_IN_CODE
    assert "Ask again with a question of your own." in library_prompts.ASK_LINE_IN_CODE


# --- width: a final message asks nothing; at most one re-ask pending ---------------------------------------------


def test_on_a_message_the_world_declares_final_nothing_is_asked_and_its_events_are_taken(monkeypatch):
    asked: list = []
    _ask_stub(monkeypatch, FIT_A, asked)
    run = _run(reask_state=SimpleNamespace(request_text="r", open_librarian=object()))
    run.checker_seen.append({"label": "fail"})
    assert asyncio.run(L.request_reask_line(run, "The episode is over.", final=True)) == ""
    assert asked == [] and run.reasks == [] and run.reask_events("r") == []


def test_after_a_reask_the_next_waits_for_a_tool_call_and_keeps_the_events_meanwhile(monkeypatch):
    asked: list = []
    _ask_stub(monkeypatch, NONE, asked)
    run = _run(reask_state=SimpleNamespace(request_text="Sort the rows by id.", open_librarian=object()))
    run.checker_seen.append({"label": "fail"})
    assert asyncio.run(L.request_reask_line(run, "Wrong.")) and run.reask_pending is True
    run.checker_seen.append({"label": "fail"})
    assert asyncio.run(L.request_reask_line(run, '{"new": "shape"}')) == "" and len(asked) == 1  # coalesced
    run.reask_pending = False  # a cell of the actor ran (RequestRun.note_calls clears it)
    run.checker_seen.append({"label": "fail"})
    assert asyncio.run(L.request_reask_line(run, "Wrong again."))
    assert len(asked) == 2 and run.reasks[-1]["events"] == ["message", "outcome"]


def test_a_cell_clears_the_pending_reask(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.reask_pending = True
        run.note_calls({"calls": {"counts": {}}})
        assert run.reask_pending is False
    finally:
        _abort(mv2, run)


def test_the_cli_passes_the_final_field_and_only_a_true_one(monkeypatch):
    from tests.memory.integration import test_request as T
    from unify.memory.integration import hooks

    seen: list = []

    async def reask(run, message, final=False):
        seen.append((message, final))
        return ""

    async def submit(self, line):
        pass

    monkeypatch.setattr(hooks, "reask_line", reask)
    monkeypatch.setattr(T._Handle, "submit", submit, raising=False)
    lines = (b'{"message": "a"}\n{"message": "b", "final": true}\n{"message": "c", "final": "yes"}\n'
             b'{"quit": true}\n')
    code, *_ = T._drive(monkeypatch, ["act", "--persist", "--jsonl", "--no-clarify", "--quiet", "Say hi"], lines,
                        T._Run([]))  # fmt: skip
    assert code == 0 and seen == [("a", False), ("b", True), ("c", False)]
