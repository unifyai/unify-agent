"""Symbolic: ``UNIFY_TRY_FIRST`` uses free information before paid actions.

On ARC return visits Unify bought a demonstration first in 101 of 101
visits, even when search had found a function stored for the same task;
the rows that run the stored program first (HPL/PL) solved 28-42 return
visits with no feedback, Unify none. With the switch the actor's prompt
says to run a stored function that fits on the inputs it already has and
act on its result when it works before an action that costs something
(never to check it against examples it was given), and a function stored
while handling a request records that request, so a search from a request
close to it shows ``similar_request: <score>``.

A whole-request hash never matched a return visit: a recorded opening
request carries the visit's own data (a fresh test grid, another requester).
In the decision-point replays of 2 Oct, marking the stored function of the
same task made the actor run it first in 29 of 105 replays against 17 of 105
(McNemar p=0.012). The match is therefore a similarity of the requests'
letter and digit runs, every token kept and weighted by its rarity among
the library's requests (4 Oct: the first version left numbers out, a rule
taken from the benchmarks' grids and amounts). Calibrated offline on a
tune/held-out split of recorded opening requests of six benchmarks and 60
hand-written assistant request pairs (research artifact similar-request-v1),
it marks at a score of 0.24 or more. Requests are captured at unillm's
transport (``tests/cache_discipline_helpers.py``), so nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import prompt_builders as pb
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

SOURCE = 'def double(x: int) -> int:\n    """Double a number."""\n    return 2 * x\n'
TASK = "Double the number 21 and reply with the result."


@pytest.fixture
def try_first(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", on)

    return set_


def _prompt() -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
    )


# ── prompt ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("framing", ["", "unified"])
def test_on_the_library_section_says_free_before_paid(try_first, monkeypatch, framing):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    try_first(False)
    off = pb._library_section()
    try_first(True)
    on = pb._library_section()
    assert pb._TRY_FIRST_NOTE not in off
    assert on.count(pb._TRY_FIRST_NOTE) == 1
    # Only the paragraph is added, just before the writing subsection.
    assert on.replace(pb._TRY_FIRST_NOTE + "\n\n", "", 1) == off
    assert on.index(pb._TRY_FIRST_NOTE) < on.index("#### Writing to the libraries")
    assert pb._TRY_FIRST_NOTE in _prompt()


def test_off_the_prompt_has_no_try_first_note(try_first):
    try_first(False)
    prompt = _prompt()
    assert "Free before paid" not in prompt
    assert "similar_request" not in prompt


def test_the_note_names_the_mark_and_no_task():
    note = pb._TRY_FIRST_NOTE
    assert f"`{task_origin.MARK}`" in note
    assert "same_task" not in note
    assert "task" not in note.lower()


# ── task keys ────────────────────────────────────────────────────────────


def test_a_task_key_ignores_whitespace_only():
    assert task_origin.task_key(" a  b\n") == task_origin.task_key("a b")
    assert task_origin.task_key("a b") != task_origin.task_key("a c")
    assert len(task_origin.task_key("a")) == 16
    assert task_origin.task_key("  ") is None
    assert task_origin.task_key({"b": 1, "a": 2}) == task_origin.task_key(
        {"a": 2, "b": 1},
    )


def test_a_sub_agent_keeps_the_key_of_its_task(try_first):
    try_first(True)
    outer = task_origin.enter("outer task")
    try:
        assert task_origin.enter("sub-agent task") is None
        assert task_origin.current() == task_origin.task_key("outer task")
    finally:
        task_origin.leave(outer)
    assert task_origin.current() is None
    try_first(False)
    assert task_origin.enter("outer task") is None


# ── similarity ──────────────────────────────────────────────────────────

PREAMBLE = (
    "You are working through a stream of table puzzles. Each instance gives a "
    "puzzle id and an input table; puzzles recur with fresh tables. Reply with "
    "the output table as rows of integers on the last line of your reply."
)


def _visit(puzzle: str, table: list[list[int]]) -> str:
    rows = "\n".join(" ".join(str(v) for v in row) for row in table)
    size = f"{len(table)}x{len(table[0])}"
    return f"{PREAMBLE}\n\nNew instance. Puzzle id: {puzzle}\nInput ({size}):\n{rows}\n"


def _score(a: str, b: str, *others: str) -> float:
    a, b = task_origin.bounded_text(a), task_origin.bounded_text(b)
    weights = task_origin.token_weights([a, b, *map(task_origin.bounded_text, others)])
    return task_origin.similarity(a, b, weights)


def test_letter_and_digit_runs_are_tokens_and_none_is_left_out():
    assert task_origin.tokens("Input (13x13): 8 0 21 2021 3x4x5 Q3 grid") == frozenset(
        {"input", "13", "x", "8", "0", "21", "2021", "3", "4", "5", "q", "grid"},
    )
    assert task_origin.tokens("Café №7, 東京") == frozenset({"café", "7", "東京"})


def test_with_one_stored_request_only_the_same_text_matches():
    # Nothing tells shared wording from distinctive wording yet.
    first = _visit("p-3d61a", [[1, 0, 2], [0, 1, 0]])
    again = _visit("p-3d61a", [[5, 5], [0, 5], [5, 0], [1, 1]])
    assert _score(again, first) == 0.0
    assert _score(first, first) == 1.0


def test_a_return_visit_with_fresh_data_matches_and_another_puzzle_does_not():
    first = _visit("p-3d61a", [[1, 0, 2], [0, 1, 0]])
    again = _visit("p-3d61a", [[5, 5], [0, 5], [5, 0], [1, 1]])
    other = _visit("p-9b07c", [[1, 0, 2], [0, 1, 0]])
    library = [
        _visit("p-55e10", [[3, 3], [3, 0]]),
        _visit("p-71c2f", [[2, 0], [0, 2], [2, 2]]),
        _visit("p-a04d9", [[7, 1, 7]]),
    ]
    assert task_origin.task_key(first) != task_origin.task_key(again)
    # Numbers are compared like any token: the fresh table weighs against
    # the match, the shared id for it.
    assert _score(again, first, *library) >= task_origin.SIMILAR_REQUEST_THRESHOLD
    # The same table under another id: the preamble weighs nothing.
    assert _score(other, first, *library) < task_origin.SIMILAR_REQUEST_THRESHOLD


def test_a_reworded_variant_matches_once_the_library_knows_other_requests():
    def ask(name: str, request: str) -> str:
        return f"{PREAMBLE}\nRequester: {name}.\nRequest: {request}"

    five = ask(
        "Ana Lee",
        "Give a 5-star rating to every song I have liked in my playlists.",
    )
    one = ask(
        "Bo Park",
        "Give a 1-star rating to every song I have not liked in my library.",
    )
    others = [
        ask("Cy Moss", "Remove all songs released before 2021 from my library."),
        ask("Di Ruiz", "How long is my longest playlist, in minutes?"),
        ask("Ed Wu", "Send $20 to each of my friends with a note."),
    ]
    assert _score(one, five, *others) >= task_origin.SIMILAR_REQUEST_THRESHOLD
    for other in others:
        assert _score(other, five, *others) < task_origin.SIMILAR_REQUEST_THRESHOLD
    # With nothing else known, a reworded request is not a match.
    assert _score(one, five) < task_origin.SIMILAR_REQUEST_THRESHOLD


# A few earlier requests of an assistant's library.
ASSISTANT_LIBRARY = [
    "Reply to the email from Jordan about the conference sponsorship and say "
    "we can do the silver tier.",
    "What's on my calendar for tomorrow? Move anything before 10am to the "
    "afternoon.",
    "Book a 30-minute meeting with Priya Shah next Tuesday afternoon to review "
    "the hiring plan.",
    "Pay the electricity bill from the joint account.",
    "Write release notes for version 2.8 from the merged pull requests since " "2.7.",
]


@pytest.mark.parametrize(
    "stored, current",
    [
        # The same deliverable with another period, recipient or payee.
        (
            "Send the Q3 revenue report to finance.",
            "Send the Q4 revenue report to finance.",
        ),
        (
            "Send the Q3 revenue report to finance.",
            "Send the Q3 revenue report to the legal team.",
        ),
        (
            "Pay the electricity bill from the joint account.",
            "Pay the water bill from the joint account.",
        ),
        (
            "Text each of my tenants at 14 Elm St a reminder that October rent "
            "of $1,850 is due on the 1st.",
            "Text each of my tenants at 22 Oak Ave a reminder that November "
            "rent of $2,100 is due on the 1st.",
        ),
    ],
)
def test_a_recurring_deliverable_with_new_parameters_matches(stored, current):
    library = [r for r in ASSISTANT_LIBRARY if r != stored]
    assert _score(current, stored, *library) >= task_origin.SIMILAR_REQUEST_THRESHOLD


def test_another_procedure_on_the_same_object_scores_lower_and_unrelated_ones_do_not_match():
    stored = "Send the Q3 revenue report to finance."
    recurring = _score(
        "Send the Q4 revenue report to finance.",
        stored,
        *ASSISTANT_LIBRARY,
    )
    other_procedure = _score(
        "Fix the formula in the Q3 revenue report.",
        stored,
        *ASSISTANT_LIBRARY,
    )
    assert other_procedure < recurring
    for unrelated in (
        "Plan a week of vegetarian dinners for two and write the shopping list.",
        "Check me in for my flight tomorrow and pick an aisle seat.",
    ):
        assert (
            _score(unrelated, stored, *ASSISTANT_LIBRARY)
            < task_origin.SIMILAR_REQUEST_THRESHOLD
        )


def test_a_recorded_request_is_bounded():
    long = "head " + "word " * 3000 + "tail"
    text = task_origin.bounded_text(long)
    assert len(text) <= 4001
    assert text.startswith("head word") and text.endswith("word tail")
    assert task_origin.bounded_text(" a \n b ") == "a b"


# ── storage and search ───────────────────────────────────────────────────


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _search(fm: FunctionManager) -> dict:
    (row,) = [r for r in fm.search_functions(query="double") if r["name"] == "double"]
    return row


def _stored(fm: FunctionManager, name: str = "double") -> dict:
    """The metadata as stored (library results never show the origin fields)."""
    (row,) = [r for r in fm._rows(fm._compositional_scope()) if r["name"] == name]
    return row["metadata"]


def _origin(request: str) -> dict:
    return {
        "origin_tasks": [task_origin.task_key(request)],
        "origin_requests": [task_origin.bounded_text(request)],
    }


@_handle_project
def test_on_a_search_from_the_same_request_marks_the_function(try_first):
    try_first(True)
    fm = FunctionManager()
    assert _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE)) == {
        "double": "added",
    }
    assert _stored(fm) == _origin(TASK)

    same = _in_task(" " + TASK + "\n", lambda: _search(fm))
    assert same["similar_request"] == 1.0
    assert "same_task" not in same
    # Where a function came from is never shown.
    assert "origin_" not in json.dumps(same, default=str)

    other = _in_task("Triple the number 5.", lambda: _search(fm))
    assert "similar_request" not in other
    assert "origin_" not in json.dumps(other, default=str)
    assert "similar_request" not in _search(fm)  # no request keyed
    for shown in (fm.list_functions(), fm.filter_functions()):
        assert "origin_" not in json.dumps(shown, default=str)


def _source(name: str, factor: int) -> str:
    return (
        f"def {name}(x: int) -> int:\n"
        f'    """Multiply a number by {factor}."""\n'
        f"    return {factor} * x\n"
    )


@_handle_project
def test_on_a_return_visit_with_fresh_data_shows_its_score(try_first):
    try_first(True)
    fm = FunctionManager()
    first = _visit("p-3d61a", [[1, 0, 2], [0, 1, 0]])
    _in_task(first, lambda: fm.add_functions(implementations=SOURCE))
    again = _visit("p-3d61a", [[5, 5], [0, 5], [5, 0]])
    # The only request known: nothing tells its wording apart yet.
    assert "similar_request" not in _in_task(again, lambda: _search(fm))
    # Functions stored while handling other puzzles weigh the shared wording.
    for name, factor, puzzle, table in (
        ("triple", 3, "p-55e10", [[3, 3], [3, 0]]),
        ("quadruple", 4, "p-71c2f", [[2, 0], [0, 2], [2, 2]]),
        ("quintuple", 5, "p-a04d9", [[7, 1, 7]]),
    ):
        _in_task(
            _visit(puzzle, table),
            lambda: fm.add_functions(implementations=_source(name, factor)),
        )
    row = _in_task(again, lambda: _search(fm))
    score = row["similar_request"]
    assert task_origin.SIMILAR_REQUEST_THRESHOLD <= score < 1
    assert score == round(score, 2)
    other = _visit("p-9b07c", [[1, 0, 2], [0, 1, 0]])
    assert "similar_request" not in _in_task(other, lambda: _search(fm))


@_handle_project
def test_on_an_overwrite_from_another_task_keeps_both_origins(try_first):
    try_first(True)
    fm = FunctionManager()
    _in_task("first task", lambda: fm.add_functions(implementations=SOURCE))
    _in_task(
        "second task",
        lambda: fm.add_functions(implementations=SOURCE, overwrite=True),
    )
    assert _stored(fm)["origin_tasks"] == [
        task_origin.task_key("first task"),
        task_origin.task_key("second task"),
    ]
    assert _stored(fm)["origin_requests"] == ["first task", "second task"]
    assert _in_task("first task", lambda: _search(fm))["similar_request"] == 1.0


@_handle_project
def test_on_a_function_keeps_its_latest_three_requests(try_first):
    try_first(True)
    fm = FunctionManager()
    for n in range(5):
        _in_task(
            f"task number {n} of {'abcde'[n]}",
            lambda: fm.add_functions(implementations=SOURCE, overwrite=True),
        )
    assert _stored(fm)["origin_requests"] == [
        "task number 2 of c",
        "task number 3 of d",
        "task number 4 of e",
    ]


@_handle_project
def test_off_nothing_is_recorded_or_marked(try_first):
    try_first(False)
    fm = FunctionManager()
    _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE))
    assert _stored(fm) == {}
    assert "similar_request" not in _in_task(TASK, lambda: _search(fm))


# ── through the actor ────────────────────────────────────────────────────


def _act_replies(provider: h.Provider, store: str):
    calls = [("FunctionManager_search_functions", {"query": "double"})]
    add = lambda: h.completion(  # noqa: E731
        calls=[("FunctionManager_add_functions", {"implementations": SOURCE})],
    )

    def later():
        # The storage review's first step stores the function when asked to.
        messages = provider.requests[-1]["messages"]
        review_start = any(
            "Review the trajectory" in str(m.get("content")) for m in messages
        ) and not any(m.get("role") == "assistant" for m in messages)
        return add() if store == "review" and review_start else h.completion("42")

    first = [lambda: h.completion(calls=[*calls, ("GuidanceManager_search", {})])]
    if store == "actor":
        first.append(add)
    # the answer however many turns the results take, and the storage review
    return [*first, *([later] * 12)]


async def _act(request: str, *, store: str = "") -> list[dict]:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted([]) as provider:
            provider.replies.extend(_act_replies(provider, store))
            handle = await actor.act(request, persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._lifecycle_task, 60)
    finally:
        await actor.close()
    return provider.requests


def _search_results_seen(requests: list[dict]) -> list[str]:
    return [
        str(m.get("content"))
        for r in requests
        for m in r["messages"]
        if m.get("role") == "tool" and '"double"' in str(m.get("content"))
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_on_the_same_request_again_the_actor_sees_similar_request(try_first):
    try_first(True)
    await _act(TASK, store="actor")
    assert _stored(FunctionManager()) == _origin(TASK)

    seen = _search_results_seen(await _act(TASK))
    assert seen and all('"similar_request": 1.0' in s for s in seen), seen[:1]

    seen = _search_results_seen(await _act("Double 7, please."))
    assert seen and not any("similar_request" in s for s in seen)
    assert task_origin.current() is None


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_on_a_function_the_storage_review_stores_records_the_task(try_first):
    try_first(True)
    requests = await _act(TASK, store="review")
    session = h.session_requests(requests)
    review = [r for r in requests if r not in session]
    stored_by = [
        r
        for r in review
        if any(
            m.get("role") == "tool" and "added" in str(m.get("content"))
            for m in r["messages"]
        )
    ]
    assert stored_by, "the review did not store the function"
    assert _stored(FunctionManager()) == _origin(TASK)


def test_the_setting_defaults_off():
    assert ProductionSettings().UNIFY_TRY_FIRST is False
    assert ProductionSettings(UNIFY_TRY_FIRST="1").UNIFY_TRY_FIRST is True
