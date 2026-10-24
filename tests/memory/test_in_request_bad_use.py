"""A bad use counts inside the request (rev 2.3).

Inside one request, a bad use of an item is recorded when a tracked call of it raised or refused its input, or when an
answer the world then rejected equals, value for value, what a tracked call of it returned earlier in the request (the
canonical JSON ``fixtures.canonical`` makes, compared by digest; no word of either is read). A check run
(``tools.check_examples``) is never a use: no call inside it raises an audit event. A rejected answer equal to nothing
is kept as not attributed and gates nothing. The librarian then refuses the item as a fit for the rest of the request,
unless its answer names a later message of the request in ``reconsider``. The verdicts so attributed fill the use's
``answered`` and ``verdict`` (the fields the post-episode attribution fills) where verdicts count, and a card shows its
uses by outcome.
"""

from __future__ import annotations

import hashlib
import json
import pytest

from tests.memory.integration.test_request import _abort, _begin, mv2  # noqa: F401
from tests.memory.test_library_items_api import EVENTS, PAINT, ROWS, _json, _library
from unify.common import box_check_source
from unify.memory import fixtures
from unify.memory import librarian as L
from unify.memory import library_helper
from unify.memory import serving
from unify.memory.integration import request as request_mod


def _digest(value) -> str:
    return hashlib.sha256(fixtures.canonical(value).encode("utf-8", "surrogatepass")).hexdigest()[:16]


# --- the box: a returned call's digest, and check runs ----------------------------------------------------------


@pytest.mark.parametrize("value", [0, 2, "yes", [[7, 7]], {"b": [1, "ü"], "a": None}, (1, 2), 1.5])
def test_the_box_digest_is_the_digest_of_the_harness_canonical_json(value):
    assert library_helper.result_digest(value) == _digest(value)


def test_a_tracked_call_that_returned_raises_its_result_digest_once(tmp_path):
    root = _library(tmp_path)
    seen = _json(
        root,
        EVENTS
        + f"start = len(seen)\nmemory.get({ROWS!r})([[1], [2]])\n"
        + "from memory.grid.cells import rows\nrows([[1], [2], [3]])\n"
        + "print(json.dumps([e for e in seen[start:] if e[0] == 'memory.call_result']))\n",
    )
    # rev 2.3: how it ended, its result's digest and its one input's
    assert seen == [
        ["memory.call_result", [ROWS, "ok", _digest(2), _digest([[1], [2]])]],
        ["memory.call_result", [ROWS, "ok", _digest(3), _digest([[1], [2], [3]])]],
    ]


CHECK_PRELUDE = f"""
ns = {{"_difference": (lambda expected, actual: None if expected == actual else "differs"), "_contract": None}}
exec({box_check_source.SOURCE!r}, ns)
check_examples = ns["check_examples"]
"""


def test_a_check_run_raises_no_event_and_is_no_failure_but_a_real_call_is(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        EVENTS
        + CHECK_PRELUDE
        + f"paint = memory.get({PAINT!r})\n"
        "start = len(seen)\n"
        "report = check_examples(paint, [([[1, 2]], [[2, 1]]), ('not a grid', [])])\n"
        "in_check = [e for e in seen[start:] if e[0] != 'memory.helper_call' or e[1][0] != 'memory:get']\n"
        "failed_after_check = memory.failed_here()\n"
        "start = len(seen)\n"
        "err(paint, 'not a grid')\n"
        "real = [e[0] for e in seen[start:]]\n"
        "print(json.dumps([report.passed, report.total, in_check, failed_after_check, real, memory.failed_here()]))\n",
    )
    passed, total, in_check, failed_after_check, real, failed_after_real = got
    assert (passed, total) == (1, 2)  # the check ran both examples, the second raised inside it
    assert in_check == [] and failed_after_check == []
    assert "memory.call" in real and "memory.call_failed" in real
    assert failed_after_real == [PAINT]


def test_a_check_marker_is_restored_after_the_run_even_when_the_candidate_raises():
    with library_helper._check_run():
        with library_helper._check_run():
            assert library_helper._checking()
        assert library_helper._checking()
    assert not library_helper._checking()
    try:
        with library_helper._check_run():
            raise ValueError("x")
    except ValueError:
        pass
    assert not library_helper._checking()


def test_a_card_says_when_a_call_of_it_failed_in_this_request(monkeypatch):
    evidence = {PAINT: {"later_uses_text": "1 call in 1 episode: 1 returned"}}
    monkeypatch.setattr(library_helper, "failed_here", lambda: [PAINT])
    assert "In this request: a call of it failed" in library_helper._evidence_lines(evidence, PAINT)
    monkeypatch.setattr(library_helper, "failed_here", lambda: [])
    assert not any(line.startswith("In this request") for line in library_helper._evidence_lines(evidence, PAINT))


# --- the request: attribution by value ------------------------------------------------------------------------


def _msg(role: str, content: str) -> dict:
    return {"type": "message", "message": {"role": role, "content": content}}


def _say(run, *replies: str) -> None:
    """The request's transcript: its first message, then each reply."""
    from unify import transcripts

    path = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_msg("user", "the request")] + [_msg("assistant", r) for r in replies]
    path.write_text("".join(json.dumps(x) + "\n" for x in rows))


def _cell(**results) -> dict:
    """One drained cell: each item's call returned the given value."""
    return {
        "calls": {
            "counts": {item: {"ok": 1, "refused": 0, "error": 0} for item in results},
            "results": [{"item": item, "outcome": "ok", "digest": _digest(v), "input": ""} for item, v in results.items()],
        }
    }


A, B = "capture:00112233445566aa", "memory.tables.count:count_rows"


def test_a_rejected_answer_equal_to_what_a_call_returned_is_a_bad_use_of_that_item(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: [[7, 7], [7, 7]], B: 4}))
        _say(run, 'Submitting.\n{"action": "submit", "grid": [[7, 7], [7, 7]]}')
        run.take_checker({"label": "fail"})
        assert run.answer_attributions == [{"label": "fail", "obs": 1, "items": [A], "short_items": []}]
        assert run.bad_uses() == {A: 1}
        assert run.bad_here[A]["how"] == "value"
    finally:
        _abort(mv2, run)


def test_an_accepted_answer_is_attributed_and_opens_no_bad_use(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: "done"}))
        _say(run, "done")
        run.take_checker({"label": "pass"})
        assert run.answer_attributions[0]["items"] == [A] and run.bad_uses() == {}
    finally:
        _abort(mv2, run)


def test_a_rejected_answer_equal_to_nothing_is_kept_not_attributed_and_gates_nothing(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: [[7, 7]]}))
        _say(run, '{"action": "submit", "grid": [[1, 1]]}')
        run.take_checker({"label": "fail"})
        assert run.answer_attributions == [{"label": "fail", "obs": 1, "items": [], "short_items": []}]
        assert run.bad_uses() == {}
    finally:
        _abort(mv2, run)


def test_a_verdict_with_no_tracked_call_before_it_records_nothing(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        _say(run, "4")
        run.take_checker({"label": "fail"})
        assert run.answer_attributions == [] and run.bad_uses() == {}
    finally:
        _abort(mv2, run)


def test_a_match_on_a_short_value_is_shown_apart_and_gates_nothing(mv2):  # noqa: F811
    # 0, true, "yes" or [] may equal a call's result by coincidence: listed, but no bad use, no hold, no trust row
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{B: 0}))
        _say(run, '{"action": "count", "n": 0}')
        run.take_checker({"label": "fail"})
        assert run.answer_attributions[0]["items"] == [] and run.answer_attributions[0]["short_items"] == [B]
        assert run.bad_uses() == {} and run.held() == {}
    finally:
        _abort(mv2, run)


def test_keyword_values_of_the_answer_are_compared_each_whole(mv2):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: {"x": 1, "y": [2, 3]}}))
        _say(run, '{"action": "place", "where": {"y": [2, 3], "x": 1}, "note": "first"}')
        run.take_checker({"label": "fail"})
        assert run.answer_attributions[0]["items"] == [A]
    finally:
        _abort(mv2, run)


@pytest.mark.parametrize(
    "value, reply",
    [
        ([[7, 7]], "[[7, 7]]"),  # an answer handed back as JSON text (tools.answer of a list)
        (42, "42"),
        ("done", "done"),
        ({"a": 1}, 'Here it is:\n{"a": 1}'),
        ([1, 2], "[\n  1,\n  2\n]"),  # the whole reply is the JSON value
    ],
)
def test_an_answer_handed_back_as_json_text_is_compared_as_its_value(mv2, value, reply):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: value}))
        _say(run, reply)
        run.take_checker({"label": "fail"})
        assert run.answer_attributions[0]["items"] == [A]
    finally:
        _abort(mv2, run)


def test_a_member_of_an_answer_object_that_names_no_action_is_not_compared_alone(mv2):  # noqa: F811
    # the design's values: the whole action, its arguments and keyword values; an object with no action word is one
    # argument, the object itself
    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: [[1]]}))
        _say(run, '{"grid": [[1]]}')
        run.take_checker({"label": "fail"})
        assert run.answer_attributions[0]["items"] == []
    finally:
        _abort(mv2, run)


def test_a_call_that_raised_in_a_cell_is_a_bad_use_found_at_that_message(mv2, monkeypatch):  # noqa: F811
    run = _begin(mv2, "hi")
    try:
        monkeypatch.setattr(run, "_observations_now", lambda: 2)
        run.note_calls({"calls": {"counts": {B: {"ok": 0, "refused": 0, "error": 1}}}})
        assert run.bad_uses() == {B: 2} and run.bad_here[B]["how"] == "failed"
    finally:
        _abort(mv2, run)


def test_the_ask_tool_sends_every_bad_use_with_when_it_was_found(mv2, monkeypatch):  # noqa: F811
    from unify.actor import tool_handlers as th

    run = _begin(mv2, "hi")
    try:
        run.note_calls(_cell(**{A: [[1]]}))
        _say(run, '{"action": "submit", "grid": [[1]]}')
        run.take_checker({"label": "fail"})
        monkeypatch.setattr(request_mod, "current", lambda: run)
        monkeypatch.setattr(run, "_observations_now", lambda: 3)
        assert th._memory_run_failed() == [A]
        assert th._memory_run_failed_at() == ({A: 1}, 3)
    finally:
        _abort(mv2, run)


def test_the_use_record_carries_the_attributions(mv2, monkeypatch):  # noqa: F811
    from tests.memory.integration.test_request import _finish, _transcript
    from unify.memory import episodes

    written = []
    real = episodes.EpisodeWriter.write

    def spy(self, ep):
        written.append(ep.memory_use)
        return real(self, ep)

    monkeypatch.setattr(episodes.EpisodeWriter, "write", spy)
    run = _begin(mv2, "hi")
    run.note_calls(_cell(**{A: [[7, 7, 7], [7, 7, 7]]}))
    _say(run, '{"action": "submit", "grid": [[7, 7, 7], [7, 7, 7]]}')
    run.take_checker({"label": "fail"})
    _transcript(run)
    _finish(mv2, run)
    assert [w.get("answer_attributions") for w in written] == [
        [{"label": "fail", "obs": 1, "items": [A], "short_items": []}]
    ]


# --- the librarian: refused for the rest of the request, unless reconsidered on a later message -----------------


IDS = [A, B]
FIT_A = {"id": A, "call": f"memory.get({A!r})()", "why": "w"}


def test_an_item_with_a_bad_use_is_refused_as_a_fit():
    got, why = L.check_answer({"fits": [FIT_A]}, IDS, failed_here={A}, failed_at={A: 1}, messages_now=1)
    assert got is None and "tried and failed in this request" in why


def test_reconsider_on_a_later_message_offers_it_again():
    answer = {"fits": [FIT_A], "reconsider": [{"entry": A, "after_event": "message:2"}]}
    got, why = L.check_answer(answer, IDS, failed_here={A}, failed_at={A: 1}, messages_now=2)
    assert why == "" and got == answer


@pytest.mark.parametrize(
    "rows, at, now, reason",
    [
        ([{"entry": A, "after_event": "message:1"}], 1, 2, "after the one"),  # the message it failed at
        ([{"entry": A, "after_event": "message:3"}], 1, 2, "after the one"),  # a message not yet arrived
        ([{"entry": A, "after_event": "message:2"}], None, 2, "after the one"),  # when it failed is not known
        ([{"entry": B, "after_event": "message:2"}], 1, 2, "did not fail in this request"),
        ([{"entry": A, "after_event": "the second one"}], 1, 2, "message:<n>"),
        ([{"entry": A}], 1, 2, "is a list of"),
        ([], 1, 2, "is a list of"),
        ({"entry": A, "after_event": "message:2"}, 1, 2, "is a list of"),
    ],
)
def test_a_reconsider_that_names_no_later_message_is_refused(rows, at, now, reason):
    failed_at = {A: at} if at is not None else {}
    got, why = L.check_answer({"fits": [FIT_A], "reconsider": rows}, IDS, failed_here={A}, failed_at=failed_at,
                              messages_now=now)  # fmt: skip
    assert got is None and reason in why, why


def test_reconsider_names_only_a_fit_of_the_same_answer():
    answer = {"fits": [], "nearest": [], "reconsider": [{"entry": A, "after_event": "message:2"}]}
    got, why = L.check_answer(answer, IDS, failed_here={A}, failed_at={A: 1}, messages_now=2)
    assert got is None and "offers again as a fit" in why


def test_the_failed_line_says_how_to_reconsider_only_when_a_later_message_came():
    assert "reconsider" not in L.failed_line([A], {A: 2}, 2)
    assert "reconsider" not in L.failed_line([A])
    line = L.failed_line([A, B], {A: 1}, 3)
    assert f"{A} (failed at message 1)" in line and f"{B} (failed" not in line
    assert '"after_event": "message:<n>"' in line and "now at message 3" in line


def test_the_question_names_when_each_failed_and_its_key_changes_with_a_new_message():
    one = L.sent_to_librarian("q", {"commit": "c"}, failed=[A], failed_at={A: 1}, messages_now=1)
    two = L.sent_to_librarian("q", {"commit": "c"}, failed=[A], failed_at={A: 1}, messages_now=2)
    assert "reconsider" not in one["message"] and "reconsider" in two["message"]
    assert L.answer_key(one) != L.answer_key(two)


# --- the record: the same fields the post-episode attribution fills, and the card --------------------------------


def test_the_evidence_store_counts_the_verdicts_per_item():
    from unify.memory.evidence import _answer_verdicts

    use = {
        "answer_attributions": [
            {"label": "fail", "items": [A]},
            {"label": "pass", "items": [A, B]},
            {"label": "fail", "items": []},
            {"label": "fail", "items": [], "short_items": [B]},  # a short match counts for nothing
            {"label": "maybe", "items": [B]},
            "not a row",
        ]
    }
    assert _answer_verdicts(use) == {A: (1, 1), B: (1, 0)}


def _facts(tmp_path, regime: str, accepted: int, rejected: int):
    from unify.memory.evidence import EvidenceStore
    from unify.memory.lifecycle import facts_from_evidence

    tmp_path.mkdir(parents=True, exist_ok=True)
    ev = EvidenceStore(tmp_path / "ev.sqlite")
    ev.db.execute(
        "INSERT INTO episodes(episode_id, commit_sha, started_at, regime, memory_main, request) VALUES(?,?,?,?,?,?)",
        ("e1", "c", "t", regime, "m", "r"),
    )
    ev.db.execute(
        "INSERT INTO item_use(item, episode_id, called, runtime_basis, runtime_ok, answered_accepted, "
        "answered_rejected) VALUES(?,?,?,?,?,?,?)",
        (A, "e1", 1, 1, 1, accepted, rejected),
    )
    return facts_from_evidence(ev)[0].items[A]


def test_a_rejection_within_the_request_fills_answered_and_verdict_where_verdicts_count(tmp_path):
    u = _facts(tmp_path / "a", "dense", 1, 1)
    assert (u.answered, u.verdict) == (True, "rejected")
    u = _facts(tmp_path / "b", "dense", 1, 0)
    assert (u.answered, u.verdict) == (True, "accepted")
    u = _facts(tmp_path / "c", "dense", 0, 0)
    assert (u.answered, u.verdict) == (None, None)
    u = _facts(tmp_path / "d", "sparse", 0, 1)  # no verdict counts there: the episode-level attribution stays
    assert (u.answered, u.verdict) == (None, None)


def test_a_card_counts_its_uses_by_outcome():
    from types import SimpleNamespace

    from unify.memory.lifecycle import ItemUse

    def f(u):
        return SimpleNamespace(items={A: u})

    facts = [
        f(ItemUse(True, 1, False, False, False, calls=(1, 0, 0), answered=True, verdict="accepted")),
        f(ItemUse(True, 1, False, False, False, calls=(2, 0, 0), answered=True, verdict="rejected")),
        f(ItemUse(True, 1, False, False, False, calls=(1, 0, 0))),
        f(ItemUse(False, 1, False, True, False, calls=(0, 0, 1))),
    ]
    uses = serving.later_uses(A, facts)
    assert (uses["accepted"], uses["rejected"], uses["unknown"]) == (1, 1, 1)
    assert serving.later_uses_text(uses) == (
        "5 calls in 4 episodes: 4 returned, 1 raised; uses: 1 accepted, 1 rejected, 1 unknown"
    )
    assert serving.later_uses_text({"episodes": 1, "raised": 1}) == "1 call in 1 episode: 1 raised"


def test_a_failure_on_another_value_is_not_in_the_failed_list(mv2, monkeypatch):  # noqa: F811
    # the raw runtime counts include every tracked call; the list the librarian is given holds only real-input ones
    from unify.actor import tool_handlers as th
    from unify.common import declared_input

    run = _begin(mv2, "the request")
    try:
        declared_input.declare({"rows": [1]})
        other = _digest({"rows": [9]})
        run.note_calls({"calls": {"counts": {A: {"ok": 0, "refused": 0, "error": 1}},
                                  "results": [{"item": A, "outcome": "error", "digest": "", "input": other}]}})  # fmt: skip
        monkeypatch.setattr(request_mod, "current", lambda: run)
        assert run.runtime_calls["counts"][A]["error"] == 1 and th._memory_run_failed() == []
    finally:
        declared_input.clear()
        _abort(mv2, run)
