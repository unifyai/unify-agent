"""Symbolic: ``UNIFY_FUNCTION_EMPTY_NOTICE``, one line when a stored function returns empty where it never did.

A reused function with a stale constant (a category, a date, a file name)
still runs; it just returns nothing, and an answer built on it reads as
plausible. With the switch on, a call that returns empty (``[]``, ``{}``,
``""``, ``None`` or 0), where every earlier call of the same source whose
request was accepted returned something non-empty, is followed by one plain
line in the cell's output, saying how many calls it rests on. It needs at
least one accepted earlier call from a complete trace (a recurring job is
often met once before: a first visit, then a return); an accepted empty
call, or one of unknown shape, keeps it silent. Functions run in-process against a fake registered environment; no
model or network is called.
"""

from __future__ import annotations

import decimal

import pytest

from tests.function_manager.core import test_function_cases as tfc
from tests.function_manager.core.test_function_cases import (  # noqa: F401 (fixtures)
    cases_on,
    music_env,
)
from tests.helpers import _handle_project
from unify.actor.execution import capture
from unify.function_manager import run_summary, task_origin
from unify.settings import ProductionSettings, SETTINGS

IDS_SINCE = (
    "def track_ids_since(year: int) -> list:\n"
    "    return [t['id'] for t in primitives.music.list_tracks() if t['year'] >= year]\n"
)
COUNT_SINCE = (
    "def count_tracks_since(year: int) -> int:\n"
    "    return len([t for t in primitives.music.list_tracks() if t['year'] >= year])\n"
)


@pytest.fixture
def notice_on(monkeypatch, cases_on):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_SUMMARY", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_EMPTY_NOTICE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    # Outcomes are kept under requests while request records are on.
    monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", True)


def _in_cell(request: str, fn, *args, solved=None):
    """Call ``fn`` as a cell would, under ``request``; return (value, the cell's stdout)."""
    origin = task_origin.enter(request)
    parts = capture._stdout_parts.set([])
    try:
        value = fn(*args)
        if solved is not None:
            task_origin.record_outcome(solved)
        return value, "".join(p.text for p in capture._stdout_parts.get())
    finally:
        capture._stdout_parts.reset(parts)
        task_origin.leave(origin)


def _function(name: str, source: str):
    fm = tfc._FM()
    fm.add_functions(implementations=[source])
    return tfc._load(fm)[name]


def test_the_switch_is_off_by_default():
    assert (
        ProductionSettings.model_fields["UNIFY_FUNCTION_EMPTY_NOTICE"].default is False
    )


def test_result_shapes():
    assert run_summary.result_shape([]) == ("list", True)
    assert run_summary.result_shape((1,)) == ("list", False)
    assert run_summary.result_shape({}) == ("dict", True)
    assert run_summary.result_shape("") == ("text", True)
    assert run_summary.result_shape(None) == ("none", True)
    assert run_summary.result_shape(0) == ("number", True)
    assert run_summary.result_shape(decimal.Decimal("0.00")) == ("number", True)
    assert run_summary.result_shape(2.5) == ("number", False)
    assert run_summary.result_shape(False) == ("bool", None)
    assert run_summary.result_shape(object()) == ("other", None)


@_handle_project
def test_an_empty_result_after_accepted_non_empty_ones_is_remarked_on(
    notice_on,
    music_env,
):
    fn = _function("track_ids_since", IDS_SINCE)
    for year in (2000, 1980):
        _, out = _in_cell(f"Which tracks are from {year} on?", fn, year, solved=True)
        assert out == ""
    value, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert value == []
    assert out == (
        "[track_ids_since returned an empty list here. Each of its 2 earlier calls "
        "whose request was accepted returned a non-empty list.]\n"
    )


@_handle_project
def test_a_zero_is_remarked_on_as_a_number(notice_on, music_env):
    fn = _function("count_tracks_since", COUNT_SINCE)
    for year in (2000, 1980):
        _in_cell(f"How many tracks from {year} on?", fn, year, solved=True)
    value, out = _in_cell("How many tracks from 2030 on?", fn, 2030)
    assert value == 0
    assert "returned 0 here" in out and "a non-zero number" in out


@_handle_project
def test_one_accepted_call_is_enough_and_the_line_says_so(notice_on, music_env):
    fn = _function("track_ids_since", IDS_SINCE)
    _in_cell("Which tracks are from 2000 on?", fn, 2000, solved=True)
    _, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert out == (
        "[track_ids_since returned an empty list here. Its one earlier call "
        "whose request was accepted returned a non-empty list.]\n"
    )


@_handle_project
def test_a_first_call_says_nothing(notice_on, music_env):
    fn = _function("track_ids_since", IDS_SINCE)
    _, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert out == ""


@_handle_project
def test_calls_with_unknown_outcomes_do_not_count(notice_on, music_env):
    fn = _function("track_ids_since", IDS_SINCE)
    for year in (2000, 1980):
        _in_cell(f"Which tracks are from {year} on?", fn, year)
    _, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert out == ""


@_handle_project
def test_an_accepted_empty_call_keeps_it_silent(notice_on, music_env):
    fn = _function("track_ids_since", IDS_SINCE)
    for year in (2000, 1980, 2040):
        _in_cell(f"Which tracks are from {year} on?", fn, year, solved=True)
    _, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert out == ""


@_handle_project
def test_a_non_empty_result_is_never_remarked_on(notice_on, music_env):
    fn = _function("track_ids_since", IDS_SINCE)
    for year in (2000, 1980):
        _in_cell(f"Which tracks are from {year} on?", fn, year, solved=True)
    _, out = _in_cell("Which tracks are from 2020 on?", fn, 2020)
    assert out == ""


@_handle_project
def test_off_shows_nothing_while_the_summary_records(
    notice_on,
    music_env,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_EMPTY_NOTICE", False)
    fn = _function("track_ids_since", IDS_SINCE)
    for year in (2000, 1980):
        _in_cell(f"Which tracks are from {year} on?", fn, year, solved=True)
    _, out = _in_cell("Which tracks are from 2030 on?", fn, 2030)
    assert out == ""
    assert run_summary.summary(_function_id("track_ids_since"))["runs"] == 2


def _function_id(name: str) -> int:
    return int(tfc._FM().list_function_name_to_ids()[name])
