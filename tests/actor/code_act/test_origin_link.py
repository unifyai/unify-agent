"""Symbolic: origin-linked functions (``UNIFY_ORIGIN_PROVENANCE``, ``UNIFY_CAPTURE_ACCEPTED``, ``UNIFY_REVIEW_RECURRENCE``).

On the 5 Oct ARC reuse-chain diagnosis (12 ordinary runs, 192 repeat
visits) a stored function produced a zero-cost repeat solve once. The gated
shortlist listed the right function on 9 of 9 eligible visits, but showed
only ``similar_request 0.29``; of the 8 right calls of a shown function, 7
came after the agent paid for a demonstration, saying it needed one "to
identify the transformation". Nothing it saw said the function was stored
for this very task. Storing failed more often: 80 repeat visits had an
earlier solved visit and no function, because the review kept a note or
its gate called the puzzle "a one-off grid", and the helpers that were
stored left the deciding value as an argument the next caller guessed wrong.

- provenance: a marked function's line says which whole identifiers its
  origin request shares with this one, and whether the checker accepted
  that session's answer when an outcome was posted;
- capture: the review is shown the code cell the session's answer repeats,
  and a function it stores that can be called with the cell's literal
  values is run on them (as a case replay runs it) and, when it returns the
  answer, gets that call as its recorded case;
- recurrence: the review and its gate are told how many earlier logged
  requests resemble this one.

Requests are captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.actor import review_gate
from unify.function_manager import origin_capture as oc
from unify.function_manager import store_cases
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

PREAMBLE = (
    "You are working through a stream of table puzzles. Each instance gives a "
    "puzzle id and an input table; puzzles recur with fresh tables. Reply with "
    "the output table as rows of integers on the last line of your reply."
)


def _visit(puzzle: str, table: list[list[int]]) -> str:
    rows = "\n".join(" ".join(str(v) for v in row) for row in table)
    return f"{PREAMBLE}\n\nNew instance. Puzzle id: {puzzle}\nInput:\n{rows}\n"


FIRST = _visit("p-3d61a", [[1, 0, 2], [0, 1, 0]])
AGAIN = _visit("p-3d61a", [[5, 5], [0, 5], [5, 0]])
OTHERS = [
    ("triple", 3, _visit("p-55e10", [[3, 3], [3, 0]])),
    ("quadruple", 4, _visit("p-71c2f", [[2, 0], [0, 2], [2, 2]])),
    ("quintuple", 5, _visit("p-a04d9", [[7, 1, 7]])),
]


def _source(name: str, factor: int, doc: str = "") -> str:
    doc = doc or f"Multiply a number by {factor}."
    return f'def {name}(x: int) -> int:\n    """{doc}"""\n    return {factor} * x\n'


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


@pytest.fixture
def origin(monkeypatch):
    def set_(*, provenance=True, recurrence=False, task_origin_on=True, corpus=""):
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", task_origin_on)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", corpus)
        monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", provenance)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_RECURRENCE", recurrence)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _library():
    fm = FunctionManager(include_primitives=False)
    _in_task(FIRST, lambda: fm.add_functions(implementations=_source("double", 2)))
    for name, factor, request in OTHERS:
        _in_task(
            request,
            lambda: fm.add_functions(implementations=_source(name, factor)),
        )
    return fm


# ── provenance ──────────────────────────────────────────────────────────


def test_the_switches_are_off_by_default_and_parse_as_bools():
    fields = ProductionSettings.model_fields
    for name in (
        "UNIFY_ORIGIN_PROVENANCE",
        "UNIFY_CAPTURE_ACCEPTED",
        "UNIFY_REVIEW_RECURRENCE",
    ):
        assert fields[name].default is False
        assert getattr(ProductionSettings(**{name: "1"}), name) is True


def test_a_gated_row_names_the_identifier_its_origin_request_shares(origin):
    origin()
    fm = _library()
    rows = _in_task(AGAIN, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert [r["name"] for r in rows] == ["double"]
    assert rows[0]["origin"] == (
        "stored while handling a request that also named `p-3d61a`"
    )
    line = ls._gated_function_line(rows[0])
    assert line.endswith(
        "· used 0×] (stored while handling a request that also named `p-3d61a`)",
    )


def test_the_same_request_says_so(origin):
    origin()
    fm = _library()
    row, *_ = _in_task(FIRST, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert row["origin"] == "stored while handling this same request"


def test_a_checked_outcome_is_named_with_the_origin(origin):
    origin()
    fm = _library()
    assert _in_task(FIRST, lambda: task_origin.record_outcome(True))
    (row,) = _in_task(AGAIN, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert row["origin"] == (
        "stored while handling a request that also named `p-3d61a`; "
        "the checker accepted that session's answer"
    )
    # The latest outcome of a request replaces the earlier one.
    assert _in_task(FIRST, lambda: task_origin.record_outcome(False))
    (row,) = _in_task(AGAIN, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert row["origin"].endswith("the checker did not accept that session's answer")
    # An unknown outcome keeps nothing.
    assert not _in_task(FIRST, lambda: task_origin.record_outcome(None))


def test_off_no_origin_field_and_no_outcome_is_kept(origin):
    origin(provenance=False)
    fm = _library()
    assert not _in_task(FIRST, lambda: task_origin.record_outcome(True))
    (row,) = _in_task(AGAIN, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert "origin" not in row
    assert "(" not in ls._gated_function_line(row).split("]", 1)[1]
    assert not task_origin.request_log_path().exists()


def test_a_marked_search_row_says_why_and_shows_no_origin_text(origin):
    origin()
    fm = _library()
    library = fm._rows(fm._compositional_scope())

    def marked():
        marker = task_origin.Marker(library)
        return fm._compact_function_search_rows(library, marker)

    rows = {r["name"]: r for r in _in_task(AGAIN, marked)}
    assert rows["double"]["origin"].endswith("named `p-3d61a`")
    assert all("origin" not in rows[name] for name in ("triple", "quintuple"))
    assert all(
        "origin_requests" not in (r.get("metadata") or {}) for r in rows.values()
    )
    assert PREAMBLE not in json.dumps(list(rows.values()))


def test_an_identifier_every_known_request_shares_is_not_named():
    known = [
        "run r-77aa1 task-00aa11 x",
        "run r-77aa1 task-00bb22 y",
        "run r-77aa1 task-00aa11 z",
    ]
    assert task_origin.shared_identifiers(known[2], known[0], known) == ["task-00aa11"]
    assert task_origin.shared_identifiers(known[1], known[0], known) == []


def test_the_rarest_shared_identifiers_come_first_at_most_two():
    current = "acct-9f1e2 order-41aa7 item-0c3d4 q"
    origin_text = "item-0c3d4 order-41aa7 acct-9f1e2 r"
    known = [current, origin_text, "x item-0c3d4", "y item-0c3d4", "z order-41aa7"]
    assert task_origin.shared_identifiers(current, origin_text, known) == [
        "acct-9f1e2",
        "order-41aa7",
    ]


def test_a_plain_word_is_never_an_identifier():
    assert task_origin.identifiers("Reply with rows of integers, 13 by 13") == {}


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize("name", ["UNIFY_ORIGIN_PROVENANCE", "UNIFY_REVIEW_RECURRENCE"])
async def test_an_actor_refuses_them_without_request_records(origin, monkeypatch, name):
    origin(provenance=False, task_origin_on=False)
    monkeypatch.setattr(SETTINGS, name, True)
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match="UNIFY_TASK_ORIGIN"):
            await actor.act(AGAIN, persist=False)
    finally:
        await actor.close()


# ── recurrence ──────────────────────────────────────────────────────────


def test_recurrence_counts_earlier_logged_requests_like_this_one(origin):
    origin(provenance=False, recurrence=True)
    _in_task(FIRST, lambda: None)
    for _, _, request in OTHERS:
        _in_task(request, lambda: None)
    found = _in_task(AGAIN, lambda: task_origin.recurrence(0.175))
    assert (found.similar, found.earlier, found.threshold) == (1, 4, 0.175)
    assert found.closest >= 0.175
    first = _in_task("Plan a dinner for two.", lambda: task_origin.recurrence())
    assert first.similar == 0 and first.earlier == 5
    assert first.threshold == task_origin.SIMILAR_REQUEST_THRESHOLD


def test_recurrence_off_logs_nothing(origin):
    origin(provenance=False, recurrence=False)
    assert _in_task(FIRST, lambda: task_origin.recurrence()) is None
    assert not task_origin.request_log_path().exists()


def test_the_recurrence_note_states_the_count(origin):
    origin(provenance=False, recurrence=True)
    assert _in_task(FIRST, caa._review_recurrence_note) == (
        "## Recurrence\n\nThis is the first request in this assistant's request "
        "log: no earlier request to compare it with."
    )
    for _, _, request in OTHERS:
        _in_task(request, lambda: None)
    note = _in_task(AGAIN, caa._review_recurrence_note)
    assert re.fullmatch(
        r"## Recurrence\n\n1 of the 4 earlier requests in this assistant's "
        r"request log resemble this one \(similar_request at least 0\.24, an "
        r"overlap of the two requests' words weighted by rarity; the closest "
        r"scores 0\.\d\d\)\.",
        note,
    ), note


# ── the answer cell ─────────────────────────────────────────────────────

GRID = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3]]
MIRRORED = [row[::-1] for row in GRID]
CELL = f"g = {GRID!r}\nlimit = 3\nout = [row[::-1] for row in g]\nprint(out)\n"
ANSWER = json.dumps({"action": "submit", "grid": MIRRORED})


def _call(call_id, code):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "arguments": json.dumps({"code": code, "language": "python"}),
                },
            },
        ],
    }


def _result(call_id, output):
    return {"role": "tool", "tool_call_id": call_id, "content": output}


def _trajectory(answer=ANSWER):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": FIRST},
        _call("c1", "print('looking')"),
        _result("c1", "--- stdout ---\nlooking"),
        _call("c2", CELL),
        _result("c2", f"--- stdout ---\n{MIRRORED}"),
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "Noted."},
        {"role": "assistant", "content": '{"action": "finish"}'},
    ]


def test_the_answer_is_linked_to_the_cell_whose_output_it_repeats():
    cell = oc.find_answer_cell(_trajectory())
    assert cell is not None
    assert cell.code == CELL and cell.answer == ANSWER
    assert dict(cell.bindings) == {"g": GRID, "limit": 3}


def test_a_typed_answer_or_a_bare_action_links_nothing():
    typed = json.dumps({"action": "submit", "grid": [[9, 9, 9, 9]] * 3})
    assert oc.find_answer_cell(_trajectory(typed)) is None
    trajectory = _trajectory()[:-3]  # no reply at all: only cells
    assert oc.find_answer_cell(trajectory) is None


def test_the_reply_the_outcome_arrived_on_is_the_answer():
    trajectory = _trajectory()
    later = json.dumps({"action": "submit", "grid": [[0, 0, 0, 0]] * 3})
    trajectory[-1] = {"role": "assistant", "content": later}
    assert oc.find_answer_cell(trajectory) is None
    assert oc.find_answer_cell(trajectory, answer=ANSWER).code == CELL


def test_a_cell_that_ran_after_the_answer_does_not_count():
    trajectory = _trajectory()
    reply = trajectory.pop(6)
    trajectory.insert(2, reply)
    assert oc.find_answer_cell(trajectory, answer=ANSWER) is None


def test_repeats_tolerates_only_the_replys_own_wording():
    answer = oc.tokens(ANSWER)
    assert oc.repeats(answer, oc.tokens(str(MIRRORED)))
    assert not oc.repeats(answer, oc.tokens(str(MIRRORED[:2])))
    assert not oc.repeats(oc.tokens("submit 1 2"), oc.tokens("submit 1 2"))


def test_argument_sets_match_by_name_or_the_one_required_parameter():
    bindings = {"g": GRID, "limit": 3}
    by_name = "def f(g, limit=1, other=None):\n    return g\n"
    assert oc.argument_sets(by_name, "f", bindings) == [
        {"args": [], "kwargs": {"g": GRID, "limit": 3}},
    ]
    one = "def f(grid, fill=0):\n    return grid\n"
    assert oc.argument_sets(one, "f", bindings) == [
        {"args": [], "kwargs": {"grid": GRID}},
        {"args": [], "kwargs": {"grid": 3}},
    ]
    two = "def f(grid, color):\n    return grid\n"
    assert oc.argument_sets(two, "f", bindings) == []


def test_the_review_note_offers_the_cell_and_asks_nothing_of_examples():
    cell = oc.find_answer_cell(_trajectory())
    note = oc.review_note(cell, None)
    assert note.startswith("## Code That Produced The Answer\n\n")
    assert CELL.strip() in note and "`g`, `limit`" in note
    assert "No checked outcome was posted" in note
    assert "checker accepted" in oc.review_note(cell, {"solved": True})
    words = set(re.findall(r"[a-z]+", note.lower()))
    for word in ("demo", "demonstration", "example", "examples", "verify", "must"):
        assert word not in words
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    assert not BENCHMARK_WORDS.search(note.replace(CELL, ""))


def test_no_cell_is_offered_after_a_failed_outcome_or_in_lessons_mode(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", True)
    cell, review, gate = caa._origin_link_notes(
        _trajectory(),
        outcome=None,
        answer=None,
        lessons=False,
    )
    assert cell is not None and review.startswith("## Code That Produced")
    assert gate.startswith("\n\n## Answer from code\n\n")
    for kwargs in (
        {"outcome": {"solved": False}, "lessons": False},
        {"outcome": None, "lessons": True},
    ):
        assert caa._origin_link_notes(_trajectory(), answer=None, **kwargs) == (
            None,
            "",
            "",
        )
    monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", False)
    assert caa._origin_link_notes(
        _trajectory(),
        outcome=None,
        answer=None,
        lessons=False,
    ) == (None, "", "")


# ── recording the answering call ────────────────────────────────────────

MIRROR = "def mirror_rows(g):\n    return [row[::-1] for row in g]\n"
MIRROR_ONE_ARG = "def mirror_table(table):\n    return [row[::-1] for row in table]\n"
WRONG = "def keep_rows(g):\n    return [list(row) for row in g]\n"
USES_ENV = (
    "def mirror_with_log(g):\n    primitives.music.list_tracks()\n"
    "    return [row[::-1] for row in g]\n"
)


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")


def _cases(fm, name):
    return store_cases.cases(int(fm.list_function_name_to_ids()[name]))


def test_a_function_that_returns_the_answer_gets_that_call_as_its_case(capture):
    fm = FunctionManager(include_primitives=False)
    cell = oc.find_answer_cell(_trajectory())
    with oc.reviewing(cell):
        out = fm.add_functions(implementations=[MIRROR, MIRROR_ONE_ARG, WRONG])
    assert out["mirror_rows"] == (
        "added; called as mirror_rows(g=<list>) on the answer cell's values, it "
        "returned the session's answer; that call is recorded as its case"
    )
    assert out["mirror_table"].startswith("added; called as mirror_table(table=<list>)")
    assert out["keep_rows"] == (
        "added; run on the answer cell's values, it did not return the session's "
        "answer; no case recorded"
    )
    (case,) = _cases(fm, "mirror_rows")
    assert case.call == {"args": [], "kwargs": {"g": GRID}}
    assert case.result["shown"] == repr(MIRRORED).replace("'", "")
    assert _cases(fm, "keep_rows") == []
    # Search results now show how it was called.
    rows = {r["name"]: r for r in fm.filter_functions()}
    assert rows["mirror_rows"]["cases"].startswith(
        f"#{case.case_id} mirror_rows(g=[[1, 2, 3, 4]",
    )


def test_a_function_that_calls_the_environment_is_not_run_on_the_values(capture):
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        out = fm.add_functions(implementations=[USES_ENV])
    assert out["mirror_with_log"] == (
        "added; not run to the end on the answer cell's values (it uses "
        "primitives.music, whose calls are not recorded); no case recorded"
    )
    assert _cases(fm, "mirror_with_log") == []


def test_outside_a_review_or_off_nothing_is_run(capture, monkeypatch):
    fm = FunctionManager(include_primitives=False)
    assert fm.add_functions(implementations=[MIRROR]) == {"mirror_rows": "added"}
    assert _cases(fm, "mirror_rows") == []
    monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        out = fm.add_functions(implementations=[MIRROR_ONE_ARG])
    assert out == {"mirror_table": "added"}
    assert _cases(fm, "mirror_table") == []


def test_without_function_cases_nothing_is_run(capture, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", False)
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        assert fm.add_functions(implementations=[MIRROR]) == {"mirror_rows": "added"}


# ── through the actor's review ──────────────────────────────────────────


async def _session(monkeypatch, *, review_calls, gate=False, gate_reply=""):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", gate)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))
    actor = caa.CodeActActor()
    reviewed = {"calls": 0}
    try:
        with h.scripted([]) as provider:

            def _reply():
                request = provider.requests[-1]
                system = request["messages"][0]["content"]
                if system == review_gate.GATE_SYSTEM_PROMPT:
                    return h.completion(content=gate_reply)
                if system.startswith("You are a skill librarian."):
                    reviewed["calls"] += 1
                    if reviewed["calls"] == 1 and review_calls:
                        return h.completion(calls=review_calls)
                    return h.completion(content="stored")
                actor_turns = sum(
                    1
                    for r in provider.requests
                    if not r["messages"][0]["content"].startswith("You are a skill")
                    and r["messages"][0]["content"] != review_gate.GATE_SYSTEM_PROMPT
                )
                if actor_turns == 1:
                    return h.completion(calls=[("execute_code", {"code": CELL})])
                return h.completion(content=ANSWER)

            provider.replies = [_reply] * 30
            handle = await actor.act("Mirror each row of the table.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    return provider.requests, actor


def _reviews(requests):
    return [
        r
        for r in requests
        if r["messages"][0]["content"].startswith("You are a skill librarian.")
    ]


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_review_is_shown_the_cell_and_its_stored_function_gets_the_call(
    capture,
    monkeypatch,
):
    requests, actor = await _session(
        monkeypatch,
        review_calls=[("FunctionManager_add_functions", {"implementations": [MIRROR]})],
    )
    reviews = _reviews(requests)
    assert reviews
    system = reviews[0]["messages"][0]["content"]
    assert "## Code That Produced The Answer" in system
    assert "out = [row[::-1] for row in g]" in system
    # After the trajectory, so the static prefix is unchanged.
    assert system.index("## Completed Trajectory") < system.index(
        "## Code That Produced The Answer",
    )
    tool_results = [
        m["content"]
        for m in reviews[-1]["messages"]
        if m["role"] == "tool" and "mirror_rows" in str(m["content"])
    ]
    assert any("recorded as its case" in str(t) for t in tool_results)
    fm = FunctionManager(include_primitives=False)
    (case,) = _cases(fm, "mirror_rows")
    assert case.call == {"args": [], "kwargs": {"g": GRID}}


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_gate_is_told_the_answer_came_from_code(capture, monkeypatch):
    requests, _ = await _session(
        monkeypatch,
        review_calls=[],
        gate=True,
        gate_reply='{"review": false, "reason": "x"}',
    )
    (gate_request,) = [
        r
        for r in requests
        if r["messages"][0]["content"] == review_gate.GATE_SYSTEM_PROMPT
    ]
    user = gate_request["messages"][1]["content"]
    assert "## Answer from code" in user
    assert user.index("## Answer from code") < user.index("## Final reply")


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_off_the_review_prompt_is_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CAPTURE_ACCEPTED", False)
    requests, _ = await _session(monkeypatch, review_calls=[])
    system = _reviews(requests)[0]["messages"][0]["content"]
    assert "Code That Produced" not in system and "## Recurrence" not in system
