"""Symbolic: ``UNIFY_TASK_ORIGIN`` records request origins without the try-first paragraph.

``UNIFY_TRY_FIRST`` did two things at once: it recorded the request a stored
function came from (so a later search, or the library shortlist, can show
``similar_request``) and it added the "free before paid" paragraph to the
prompt. In the 4 Oct AppWorld LOW cells that paragraph preceded the 401
errors of stored functions run before the session had logged in, while the
request-gated shortlist the retrieval audit recommends needs only the
records. ``UNIFY_TASK_ORIGIN`` turns on the records and marks alone; the
prompt is the switch-off prompt. ``UNIFY_TRY_FIRST`` is unchanged.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers import _handle_project
from unify.actor import prompt_builders as pb
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

SOURCE = 'def double(x: int) -> int:\n    """Double a number."""\n    return 2 * x\n'
TASK = "Double the number 21 and reply with the result."


@pytest.fixture
def switches(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    def set_(*, origin: bool, try_first: bool = False) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", try_first)

    return set_


def _prompt() -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
    )


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _stored(fm: FunctionManager) -> dict:
    (row,) = [r for r in fm._rows(fm._compositional_scope()) if r["name"] == "double"]
    return row["metadata"]


def _search(fm: FunctionManager) -> dict:
    (row,) = [r for r in fm.search_functions(query="double") if r["name"] == "double"]
    return row


@pytest.mark.parametrize("framing", ["", "unified"])
def test_the_prompt_is_the_switch_off_prompt(switches, monkeypatch, framing):
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FRAMING", framing)
    switches(origin=False)
    off_section, off_prompt = pb._library_section(), _prompt()
    switches(origin=True)
    assert pb._library_section() == off_section
    assert _prompt() == off_prompt
    assert "Free before paid" not in off_prompt
    # UNIFY_TRY_FIRST still adds its paragraph, with or without the records.
    switches(origin=True, try_first=True)
    assert pb._TRY_FIRST_NOTE in _prompt()


def test_a_request_is_keyed_and_a_sub_agent_keeps_its_tasks_key(switches):
    switches(origin=True)
    outer = task_origin.enter("outer task")
    try:
        assert outer is not None
        assert task_origin.enter("sub-agent task") is None
        assert task_origin.current() == task_origin.task_key("outer task")
    finally:
        task_origin.leave(outer)
    assert task_origin.current() is None
    switches(origin=False)
    assert task_origin.enter("outer task") is None


@_handle_project
def test_a_stored_function_records_its_request_and_a_search_marks_it(switches):
    switches(origin=True)
    fm = FunctionManager()
    _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE))
    assert _stored(fm) == {
        "origin_tasks": [task_origin.task_key(TASK)],
        "origin_requests": [task_origin.bounded_text(TASK)],
    }
    same = _in_task(TASK, lambda: _search(fm))
    assert same["similar_request"] == 1.0
    assert "origin_" not in json.dumps(same, default=str)
    assert "similar_request" not in _in_task("Triple 5.", lambda: _search(fm))


@_handle_project
def test_off_nothing_is_recorded(switches):
    switches(origin=False)
    fm = FunctionManager()
    _in_task(TASK, lambda: fm.add_functions(implementations=SOURCE))
    assert _stored(fm) == {}


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False)])
def test_the_setting_defaults_off_and_parses_booleans(value, expected):
    assert ProductionSettings().UNIFY_TASK_ORIGIN is False
    assert ProductionSettings(UNIFY_TASK_ORIGIN=value).UNIFY_TASK_ORIGIN is expected
