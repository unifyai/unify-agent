"""Symbolic: ``UNIFY_SIMILAR_REQUEST_CORPUS=stream`` weighs ``similar_request`` over the stream's requests.

The retrieval-matching audit (research artifact retrieval-matching-audit-v1,
5 Oct) scored request-to-request ``similar_request`` on 2,277 recorded task
starts. Weights over the library's origin requests alone are
``ln(2/2) = 0`` for every word two known requests share, so a library that
knows one or two requests marks nothing: on those 26 AppWorld repeat visits
the shipped score found last time's entry once, stream-wide weights 23
times. With the switch each top-level request is logged (the latest 200
distinct, in the home, across restarts) and the weights count them too.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS


@pytest.fixture
def switches(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    def set_(*, corpus="", origin=True):
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", corpus)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)

    return set_


# ── stream-wide weights ──────────────────────────────────────────────────

SOURCE = 'def double(x: int) -> int:\n    """Double a number."""\n    return 2 * x\n'
FIRST = (
    "Ana Lee asks: add every song from the 1990s that I liked to a new "
    "playlist called Nineties Gold."
)
AGAIN = (
    "Bo Park asks: add every song from the 1980s that I liked to a new "
    "playlist called Eighties Hits."
)
UNRELATED = "Cy Moss asks: how much did I pay for groceries last month?"


def _in_task(request, fn=lambda: None):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _origin_row(fm: FunctionManager) -> dict:
    (row,) = [r for r in fm._rows(fm._compositional_scope()) if r["name"] == "double"]
    return row


def _score_in(fm: FunctionManager, request: str) -> float:
    library = fm._rows(fm._compositional_scope())
    return _in_task(request, lambda: task_origin.Marker(library).score(_origin_row(fm)))


@_handle_project
@pytest.mark.parametrize("corpus", ["", "stream"])
def test_one_stored_function_and_two_earlier_requests(switches, corpus):
    switches(corpus=corpus)
    fm = FunctionManager()
    _in_task(FIRST, lambda: fm.add_functions(implementations=SOURCE))
    _in_task(UNRELATED)
    score = _score_in(fm, AGAIN)
    if corpus:
        # The shared words that the unrelated request lacks now weigh.
        assert score >= 0.175
    else:
        # Two known requests: every word they share weighs ln(2/2) = 0.
        assert score == 0.0


@_handle_project
def test_the_log_keeps_the_latest_distinct_requests_across_restarts(
    switches,
    monkeypatch,
):
    switches(corpus="stream")
    monkeypatch.setattr(task_origin, "REQUEST_LOG_SIZE", 3)
    for n in range(5):
        _in_task(f"request number {n}")
    assert task_origin.logged_requests() == [
        "request number 2",
        "request number 3",
        "request number 4",
    ]
    # A request seen again becomes the latest; nothing is duplicated.
    _in_task("request  number 2\n")
    assert task_origin.logged_requests() == [
        "request number 3",
        "request number 4",
        "request number 2",
    ]
    # The log is a file in the home: a new process reads the same requests.
    assert task_origin.request_log_path().exists()
    task_origin._tokens.cache_clear()
    assert len(task_origin.logged_requests()) == 3


@_handle_project
def test_a_sub_agents_request_is_not_logged(switches):
    switches(corpus="stream")
    outer = task_origin.enter("the top-level task")
    try:
        assert task_origin.enter("a sub-agent's delegated task") is None
    finally:
        task_origin.leave(outer)
    assert task_origin.logged_requests() == ["the top-level task"]


@_handle_project
@pytest.mark.parametrize("corpus, origin", [("", True), ("stream", False)])
def test_nothing_is_logged_unless_both_switches_are_on(switches, corpus, origin):
    switches(corpus=corpus, origin=origin)
    _in_task("a request")
    assert not task_origin.request_log_path().exists()
    assert task_origin.logged_requests() == []


def test_the_setting_defaults_off_and_validates():
    assert ProductionSettings().UNIFY_SIMILAR_REQUEST_CORPUS == ""
    assert (
        ProductionSettings(
            UNIFY_SIMILAR_REQUEST_CORPUS=" Stream ",
        ).UNIFY_SIMILAR_REQUEST_CORPUS
        == "stream"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_SIMILAR_REQUEST_CORPUS="library")
