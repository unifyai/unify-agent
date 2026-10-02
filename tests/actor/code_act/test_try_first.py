"""Symbolic: ``UNIFY_TRY_FIRST`` uses free information before paid actions.

On ARC return visits Unify bought a demonstration first in 101 of 101
visits, even when search had found a function stored for the same task;
the rows that run the stored program first (HPL/PL) solved 28-42 return
visits with no feedback, Unify none. With the switch the actor's prompt
says to run a matching stored function on the inputs it already has and
check it against its evidence before an action that costs something, and a
function stored during a task records a hash of the task's request, so a
search from a task with the same request marks it ``same_task: true``.
Requests are captured at unillm's transport
(``tests/cache_discipline_helpers.py``), so nothing leaves the process.
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
    assert "same_task" not in prompt


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


@_handle_project
def test_on_a_search_from_the_same_task_marks_the_function(try_first):
    try_first(True)
    fm = FunctionManager()
    assert _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE)) == {
        "double": "added",
    }
    stored = fm.list_functions()["double"]["metadata"]
    assert stored == {"origin_tasks": [task_origin.task_key(TASK)]}

    same = _in_task(" " + TASK + "\n", lambda: _search(fm))
    assert same["same_task"] is True
    # The hash itself is never shown.
    assert "origin_tasks" not in json.dumps(same, default=str)

    other = _in_task("Triple the number 5.", lambda: _search(fm))
    assert "same_task" not in other
    assert "origin_tasks" not in json.dumps(other, default=str)
    assert "same_task" not in _search(fm)  # no task keyed


@_handle_project
def test_on_an_overwrite_from_another_task_keeps_both_origins(try_first):
    try_first(True)
    fm = FunctionManager()
    _in_task("first task", lambda: fm.add_functions(implementations=SOURCE))
    _in_task(
        "second task",
        lambda: fm.add_functions(implementations=SOURCE, overwrite=True),
    )
    assert fm.list_functions()["double"]["metadata"]["origin_tasks"] == [
        task_origin.task_key("first task"),
        task_origin.task_key("second task"),
    ]
    assert _in_task("first task", lambda: _search(fm))["same_task"] is True


@_handle_project
def test_off_nothing_is_recorded_or_marked(try_first):
    try_first(False)
    fm = FunctionManager()
    _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE))
    assert fm.list_functions()["double"]["metadata"] == {}
    assert "same_task" not in _in_task(TASK, lambda: _search(fm))


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
async def test_on_a_return_visit_to_the_same_task_sees_same_task(try_first):
    try_first(True)
    await _act(TASK, store="actor")
    stored = FunctionManager().list_functions()["double"]["metadata"]
    assert stored == {"origin_tasks": [task_origin.task_key(TASK)]}

    seen = _search_results_seen(await _act(TASK))
    assert seen and all('"same_task": true' in s for s in seen), seen[:1]

    seen = _search_results_seen(await _act("Double 7, please."))
    assert seen and not any("same_task" in s for s in seen)
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
    stored = FunctionManager().list_functions()["double"]["metadata"]
    assert stored == {"origin_tasks": [task_origin.task_key(TASK)]}


def test_the_setting_defaults_off():
    assert ProductionSettings().UNIFY_TRY_FIRST is False
    assert ProductionSettings(UNIFY_TRY_FIRST="1").UNIFY_TRY_FIRST is True
