"""Symbolic: ``UNIFY_REVIEW_OUTCOME``: the storage review states whether the session's answer was confirmed.

``UNIFY_ORIGIN_PROVENANCE`` tells a later session whether the session a
function came from had its answer accepted, but only from an outcome the
environment posts (``UNIFY_OUTCOME``), and most environments post none. In
the offline replay of 28 Continual-ARC repeat-visit openings (research
artifact ``overhaul-lanes/origin-link-v1``), adding that the origin
session's answer was accepted raised calling the listed function before any
demonstration from 15 to 28 of 84 openings, over naming the shared task id
alone. The conversation usually shows the verdict already, and the storage
review reads it: with the switch the review (and its gate) state it as one
JSON key, judged from the conversation alone, and the harness keeps it under
the request. Nothing in the conversation is parsed by the harness. Requests
are captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import review_gate
from unify.actor import review_outcome as ro
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

REQUEST = "Add 2 and 3, then tell me the sum. Ticket id: tkt-40a91c."
AGAIN = "Add 7 and 8, then tell me the sum. Ticket id: tkt-40a91c."
OTHERS = [
    "Rename the file report-q3x7.txt to summary.txt.",
    "Translate 'good morning' into French. Ref: ref-77b2d1.",
    "List three prime numbers above 50. Ref: ref-09c3e4.",
]
SESSION_REPLY = "The sum is 5."
CLOSING_REPLY = "Understood."
VERDICTS = {
    "confirmation": ("Correct, thank you.", "confirmed"),
    "rejection": ("That is wrong, the sum is not what I asked for.", "rejected"),
    "neither": ("Please also keep a note of it.", "unknown"),
}


@pytest.fixture
def switches(monkeypatch):
    def set_(*, review_outcome=True, provenance=True, origin=True):
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", provenance)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_OUTCOME", review_outcome)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_RECURRENCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_OUTCOME", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


# ── the judgement ───────────────────────────────────────────────────────


def test_the_switch_is_off_by_default_and_parses_as_a_bool():
    field = ProductionSettings.model_fields["UNIFY_REVIEW_OUTCOME"]
    assert field.default is False
    assert ProductionSettings(UNIFY_REVIEW_OUTCOME="1").UNIFY_REVIEW_OUTCOME is True


@pytest.mark.parametrize(
    "text, judgement",
    [
        ('Stored add_two.\n{"answer_outcome": "confirmed"}', "confirmed"),
        ('{"answer_outcome": "REJECTED"}', "rejected"),
        ('Nothing stored.\n{"answer_outcome": "unknown"}', "unknown"),
        (
            '{"answer_outcome": "confirmed"} then {"answer_outcome": "rejected"}',
            "rejected",
        ),
        ('{"answer_outcome": "maybe"}', None),
        ("The answer was CORRECT.", None),
        ("", None),
        (None, None),
    ],
)
def test_only_the_stated_key_is_read(text, judgement):
    assert ro.parse(text) == judgement


def test_the_review_judgement_wins_unless_it_is_unknown():
    assert ro.settle("rejected", "confirmed") == "rejected"
    assert ro.settle("unknown", "confirmed") == "confirmed"
    assert ro.settle(None, "unknown") == "unknown"
    assert ro.settle(None, None) is None


def test_the_texts_ask_for_a_judgement_and_name_no_benchmark():
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    for text in (ro.REVIEW_SECTION, ro.GATE_SECTION):
        assert not BENCHMARK_WORDS.search(text)
        words = set(re.findall(r"[a-z]+", text.lower()))
        for word in ("demo", "example", "examples", "verify", "must", "correct"):
            assert word not in words
        assert "conversation alone" in text


@pytest.mark.parametrize(
    "raw, judgement",
    [
        (
            '{"review": false, "reason": "x", "answer_outcome": "confirmed"}',
            "confirmed",
        ),
        ('{"review": true, "reason": "x"}', None),
    ],
)
def test_the_gate_reply_carries_the_judgement(raw, judgement):
    decision = review_gate.parse_decision(raw)
    assert decision.answer_outcome == judgement
    assert decision.review in (True, False)


# ── kept under the request, read by provenance ──────────────────────────


def _library():
    fm = FunctionManager(include_primitives=False)
    _in_task(
        REQUEST,
        lambda: fm.add_functions(
            implementations='def add_two(a, b):\n    """Add two numbers."""\n    return a + b\n',
        ),
    )
    for i, request in enumerate(OTHERS):
        _in_task(
            request,
            lambda i=i: fm.add_functions(
                implementations=f"def other_{i}(x):\n    return x\n",
            ),
        )
    return fm


def _origin_line(fm):
    rows = _in_task(AGAIN, lambda: fm._gated_shortlist_rows(0.175, 5))
    (row,) = [r for r in rows if r["name"] == "add_two"]
    return row.get("origin")


@pytest.mark.parametrize(
    "judgement, says",
    [
        ("confirmed", "that session's answer was confirmed, as judged by its review"),
        ("rejected", "that session's answer was rejected, as judged by its review"),
    ],
)
def test_a_judgement_is_kept_and_named_in_the_provenance_line(
    switches,
    judgement,
    says,
):
    switches()
    fm = _library()
    assert _in_task(REQUEST, lambda: ro.record(judgement)) == judgement
    assert _origin_line(fm) == (
        f"stored while handling a request that also named `tkt-40a91c`; {says}"
    )


def test_unknown_keeps_nothing(switches):
    switches()
    fm = _library()
    assert _in_task(REQUEST, lambda: ro.record("unknown")) == "unknown"
    assert _origin_line(fm) == (
        "stored while handling a request that also named `tkt-40a91c`"
    )


def test_a_checkers_outcome_wins_over_the_review(switches):
    switches()
    fm = _library()
    _in_task(REQUEST, lambda: ro.record("rejected"))
    _in_task(REQUEST, lambda: task_origin.record_outcome(True))
    assert _origin_line(fm).endswith("the checker accepted that session's answer")


def test_off_nothing_is_kept(switches):
    switches(review_outcome=False)
    fm = _library()
    assert _in_task(REQUEST, lambda: ro.record("confirmed")) is None
    assert _origin_line(fm) == (
        "stored while handling a request that also named `tkt-40a91c`"
    )
    assert caa._origin_link_notes([], outcome=None, answer=None, lessons=False) == (
        None,
        "",
        "",
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_actor_refuses_it_without_request_records(switches):
    switches(origin=False, provenance=False)
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match="UNIFY_TASK_ORIGIN"):
            await actor.act(AGAIN, persist=False)
    finally:
        await actor.close()


# ── scripted sessions ending in a confirmation, a rejection, neither ────


async def _next(handle, kinds) -> dict:
    while True:
        note = await asyncio.wait_for(handle.next_notification(), 30)
        if isinstance(note, dict) and note.get("type") in kinds:
            return note


async def _session(reply_from_requester, review_reply, *, gate_reply=None):
    """A persistent session: the answer, the requester's reply, its end and its review."""
    from unify.actor.code_act_actor import (
        SESSION_ENDED,
        CodeActActor,
        _StorageCheckHandle,
    )
    from unify.common.async_tool_loop import start_async_tool_loop

    actor = CodeActActor()
    token = task_origin.enter(REQUEST)
    try:
        replies = [
            lambda: h.completion(content=SESSION_REPLY),
            lambda: h.completion(content=CLOSING_REPLY),
        ]
        if gate_reply is not None:
            replies.append(lambda: h.completion(content=gate_reply))
        replies.append(lambda: h.completion(content=review_reply))
        with h.scripted(replies) as provider:
            inner = start_async_tool_loop(
                h.new_client("You are a scripted actor."),
                REQUEST,
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                persist=True,
            )
            handle = _StorageCheckHandle(inner=inner, actor=actor)
            await _next(handle, ("response",))
            await handle.interject(reply_from_requester)
            await _next(handle, ("response",))
            await handle.stop(SESSION_ENDED)
            note = await _next(
                handle,
                ("storage_review_complete", "storage_review_skipped"),
            )
            await asyncio.wait_for(handle._lifecycle_task, 30)
    finally:
        task_origin.leave(token)
        await actor.close()
    return note, provider.requests


def _kept():
    found = task_origin.origin_outcome(task_origin.bounded_text(REQUEST))
    return found


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("ending", sorted(VERDICTS))
async def test_the_review_judges_the_conversation_and_its_judgement_is_kept(
    switches,
    monkeypatch,
    ending,
):
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", False)
    said, judgement = VERDICTS[ending]
    note, requests = await _session(
        said,
        f'Nothing worth storing.\n{{"answer_outcome": "{judgement}"}}',
    )
    assert note["type"] == "storage_review_complete"
    review = requests[-1]
    text = json.dumps(review["messages"])
    # The review reads the requester's reply and is asked for the judgement.
    assert said in text
    assert ro.REVIEW_SECTION.strip().split("\n")[0] in review["messages"][0]["content"]
    expected = {"confirmed": (True, "review"), "rejected": (False, "review")}
    assert _kept() == expected.get(judgement)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_review_that_states_nothing_keeps_nothing(switches, monkeypatch):
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", False)
    await _session(VERDICTS["confirmation"][0], "Nothing worth storing.")
    assert _kept() is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_gate_that_skips_the_review_still_keeps_its_judgement(
    switches,
    monkeypatch,
):
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", True)
    monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))
    note, requests = await _session(
        VERDICTS["confirmation"][0],
        "unused",
        gate_reply='{"review": false, "reason": "one-off", "answer_outcome": "confirmed"}',
    )
    assert note["type"] == "storage_review_skipped"
    (gate,) = [
        r
        for r in requests
        if r["messages"][0]["content"] == review_gate.GATE_SYSTEM_PROMPT
    ]
    assert ro.GATE_SECTION in gate["messages"][1]["content"]
    assert _kept() == (True, "review")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_the_review_is_not_asked(switches, monkeypatch):
    switches(review_outcome=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", False)
    _, requests = await _session(
        VERDICTS["confirmation"][0],
        'Nothing.\n{"answer_outcome": "confirmed"}',
    )
    assert "Answer Outcome" not in json.dumps(requests[-1]["messages"])
    assert _kept() is None
