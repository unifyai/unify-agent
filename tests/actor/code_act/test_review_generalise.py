"""Symbolic: ``UNIFY_REVIEW_GENERALISE``: the review sees the functions stored for similar requests.

On the 5 Oct ARC LOW runs the reviews of repeat visits stored siblings for
the same rule (two "stamp"/"recolor" functions in one run, two in another,
two mirror-tiling functions in a third), and reviews that found the earlier
function left a helper whose deciding value stayed a parameter. With the
switch on, the review is shown the stored functions whose recorded origin
request resembles this session's (``similar_request``, the shortlist gate's
rule), with their source, and told it can extend one of them to cover this
instance rather than store a sibling. It informs; it forces no write.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.function_manager import task_origin
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
]
UNRELATED = "Plan a week of vegetarian dinners for two and write the shopping list."
HEADER = "## Functions Stored For Similar Requests"


def _source(name: str, factor: int) -> str:
    return (
        f'def {name}(x: int) -> int:\n    """Multiply a number by {factor}."""\n'
        f"    return {factor} * x\n"
    )


def _note_row(**overrides):
    row = {
        "function_id": 7,
        "name": "double",
        "argspec": "(x: int) -> int",
        "implementation": _source("double", 2),
        "usage_calls": 2,
        "similar_request": 0.62,
    }
    row.update(overrides)
    return row


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


@pytest.fixture
def generalise(monkeypatch):
    def set_(*, on=True, origin=True, gate=""):
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GENERALISE", on)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", bool(gate))
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def test_the_note_lists_signature_score_calls_and_source():
    note = caa.render_generalise_note([_note_row()])
    assert note.startswith(HEADER + "\n\n")
    assert (
        "### `double(x: int) -> int` (function_id 7, similar_request 0.62, used 2×)"
        in note
    )
    assert "```python\n" + _source("double", 2).rstrip() + "\n```" in note
    flat = " ".join(note.split())
    assert "rather than storing a sibling" in flat
    assert "If it did a different kind of task, leave them as they are." in flat


def test_the_note_names_no_benchmark_and_no_example_check():
    words = set(
        re.findall(r"[a-z]+", caa.render_generalise_note([_note_row()]).lower()),
    )
    for word in ("arc", "appworld", "grid", "puzzle", "demo", "example", "verify"):
        assert word not in words, word


def test_no_rows_no_note():
    assert caa.render_generalise_note([]) == ""


def test_a_long_source_keeps_its_head_and_tail():
    body = "".join(f"    step_{i} = {i}\n" for i in range(600))
    source = "def long(x):\n" + body + "    return x\n"
    note = caa.render_generalise_note([_note_row(name="long", implementation=source)])
    assert "def long(x):" in note
    assert "    return x\n```" in note
    assert "characters omitted; read the function for all of it" in note
    assert len(note) < len(source)


class _FakeManager:
    def __init__(self, rows=None, error=None):
        self.rows, self.error, self.asked = rows or [], error, []

    def _similar_request_functions(self, threshold, k):
        self.asked.append((threshold, k))
        if self.error:
            raise self.error
        return self.rows


def test_off_asks_nothing(generalise):
    generalise(on=False)
    fm = _FakeManager([_note_row()])
    assert caa._review_generalise_note(fm) == ""
    assert fm.asked == []


def test_on_asks_for_three_at_the_search_threshold(generalise):
    generalise()
    fm = _FakeManager([_note_row()])
    assert caa._review_generalise_note(fm).startswith(HEADER)
    assert fm.asked == [(task_origin.SIMILAR_REQUEST_THRESHOLD, 3)]


def test_the_gate_threshold_is_used_when_set(generalise):
    generalise(gate="similar_request:0.175")
    fm = _FakeManager([_note_row()])
    caa._review_generalise_note(fm)
    assert fm.asked == [(0.175, 3)]


def test_a_failing_lookup_gives_no_note(generalise):
    generalise()
    assert caa._review_generalise_note(_FakeManager(error=RuntimeError("x"))) == ""


def test_the_manager_returns_functions_stored_for_a_similar_request(generalise):
    from unify.function_manager.function_manager import FunctionManager

    generalise()
    fm = FunctionManager()
    _in_task(FIRST, lambda: fm.add_functions(implementations=_source("double", 2)))
    for name, factor, request in OTHERS:
        _in_task(
            request,
            lambda: fm.add_functions(implementations=_source(name, factor)),
        )
    fm.add_functions(implementations=_source("no_origin", 9))
    rows = _in_task(
        AGAIN,
        lambda: fm._similar_request_functions(task_origin.SIMILAR_REQUEST_THRESHOLD, 3),
    )
    assert [r["name"] for r in rows] == ["double"]
    assert rows[0]["implementation"].startswith("def double(x: int) -> int:")
    assert 0.24 <= rows[0]["similar_request"] < 1
    assert "metadata" not in rows[0]
    unrelated = _in_task(UNRELATED, lambda: fm._similar_request_functions(0.24, 3))
    assert unrelated == []


async def _review_requests(task: str, seed) -> list[dict]:
    actor = caa.CodeActActor()
    seed(actor.function_manager)
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    return [
        r
        for r in provider.requests
        if "## Instructions" in str(r["messages"][0]["content"])
    ]


def _seed(fm):
    _in_task(FIRST, lambda: fm.add_functions(implementations=_source("double", 2)))
    for name, factor, request in OTHERS:
        _in_task(
            request,
            lambda: fm.add_functions(implementations=_source(name, factor)),
        )


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_sent_review_shows_the_function_stored_for_this_kind_of_task(
    generalise,
):
    generalise()
    reviews = await _review_requests(AGAIN, _seed)
    assert reviews
    system = str(reviews[0]["messages"][0]["content"])
    assert HEADER in system
    section = system[system.index(HEADER) :].split("## Completed Trajectory", 1)[0]
    assert "def double(x: int) -> int:" in section
    assert "def triple" not in section


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("on, task", [(True, UNRELATED), (False, AGAIN)])
async def test_no_similar_function_or_switch_off_sends_the_review_as_before(
    generalise,
    on,
    task,
):
    generalise(on=on)
    reviews = await _review_requests(task, _seed)
    assert reviews
    assert all(HEADER not in str(r["messages"]) for r in reviews)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_an_actor_refuses_generalise_without_request_records(generalise):
    generalise(origin=False)
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match="UNIFY_TASK_ORIGIN"):
            await actor.act(AGAIN, persist=False)
    finally:
        await actor.close()


def test_the_generalise_setting_defaults_off():
    assert ProductionSettings().UNIFY_REVIEW_GENERALISE is False
    assert ProductionSettings(UNIFY_REVIEW_GENERALISE="1").UNIFY_REVIEW_GENERALISE
