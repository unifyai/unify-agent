"""Symbolic: ``UNIFY_FUNCTION_SUMMARY``, a running summary of every recorded call of a stored function.

Cases keep only the latest three passing and three failing calls, so a fact
over many runs ("accepted runs read 14-40 items", "this argument varied")
would be computed from three. With the switch on, each recorded call leaves
a small row -- endpoint calls and answer sizes, argument digests, whether the
trace is complete, its request -- and the summary is computed when read,
over calls whose session is known to be accepted. Missing evidence reads
"unknown". Functions run in-process against a fake registered environment;
no model or network is called.
"""

from __future__ import annotations


import pytest

from tests.function_manager.core import test_function_cases as tfc
from tests.function_manager.core.test_function_cases import (  # noqa: F401 (fixtures)
    cases_on,
    global_env,
    music_env,
)
from tests.helpers import _handle_project
from unify.function_manager import run_summary, task_origin
from unify.settings import ProductionSettings, SETTINGS

LIST_SINCE = (
    "def count_tracks_since(year: int) -> int:\n"
    "    return len([t for t in primitives.music.list_tracks() if t['year'] >= year])\n"
)


@pytest.fixture
def summary_on(monkeypatch, cases_on):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_SUMMARY", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    # Outcomes are kept under requests while request records are on.
    monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", True)


def _run_under(request: str, fn, *args, solved=None):
    token = task_origin.enter(request)
    try:
        value = fn(*args)
        if solved is not None:
            task_origin.record_outcome(solved)
        return value
    finally:
        task_origin.leave(token)


def _fid(fm, name: str) -> int:
    return int(fm.list_function_name_to_ids()[name])


def test_the_switch_is_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_FUNCTION_SUMMARY"].default is False


@_handle_project
def test_off_records_nothing(cases_on, music_env, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_SUMMARY", False)
    fm = tfc._FM()
    fm.add_functions(implementations=[LIST_SINCE])
    tfc._load(fm)["count_tracks_since"](2000)
    assert run_summary.summary(_fid(fm, "count_tracks_since")) is None


@_handle_project
def test_every_call_leaves_a_row_and_accepted_runs_are_summarised(
    summary_on,
    music_env,
):
    fm = tfc._FM()
    fm.add_functions(implementations=[LIST_SINCE])
    fn = tfc._load(fm)["count_tracks_since"]
    _run_under("Count my tracks since 2000.", fn, 2000, solved=True)
    _run_under("Count my tracks since 1980.", fn, 1980, solved=True)
    _run_under("Count my tracks since 2010.", fn, 2010)  # outcome unknown
    fid = _fid(fm, "count_tracks_since")

    every = run_summary.summary(fid, accepted_only=False)
    assert every["runs"] == 3 and every["unknown_outcome"] == 1

    accepted = run_summary.summary(fid)
    assert accepted["runs"] == 2 and accepted["complete"] is True
    (endpoint,) = accepted["endpoints"]
    assert endpoint == "music.list_tracks"
    assert accepted["endpoints"][endpoint]["calls_per_run"] == [1, 1]
    low, high = accepted["endpoints"][endpoint]["items"]
    assert 0 <= low <= high
    assert accepted["arguments"]["#0"] == {"distinct": 2, "runs": 2}


@_handle_project
def test_without_an_accepted_run_the_facts_are_unknown(summary_on, music_env):
    fm = tfc._FM()
    fm.add_functions(implementations=[LIST_SINCE])
    _run_under("Count my tracks since 2000.", tfc._load(fm)["count_tracks_since"], 2000)
    accepted = run_summary.summary(_fid(fm, "count_tracks_since"))
    assert accepted["runs"] == 0 and accepted["endpoints"] == "unknown"


@_handle_project
def test_an_incomplete_trace_makes_the_endpoint_facts_unknown(summary_on, global_env):
    fm = tfc._FM()
    fm.add_functions(implementations=[tfc.USES_GLOBAL, tfc.CALLS_IT])
    namespace = {"apis": global_env}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    _run_under("Shout the greeting.", namespace["shout_greeting"], solved=True)
    accepted = run_summary.summary(_fid(fm, "shout_greeting"))
    assert accepted["runs"] == 1
    assert accepted["complete"] is False and accepted["endpoints"] == "unknown"


@_handle_project
def test_rows_are_pruned_to_the_latest(summary_on, music_env, monkeypatch):
    monkeypatch.setattr(run_summary, "ROWS_KEPT", 2)
    fm = tfc._FM()
    fm.add_functions(implementations=[LIST_SINCE])
    fn = tfc._load(fm)["count_tracks_since"]
    for year in (1990, 2000, 2010):
        _run_under(f"Count my tracks since {year}.", fn, year, solved=True)
    assert run_summary.summary(_fid(fm, "count_tracks_since"))["runs"] == 2


def test_item_count_reads_lists_and_list_fields():
    assert run_summary.item_count([1, 2, 3]) == 3
    assert run_summary.item_count({"items": [1, 2], "page": 1}) == 2
    assert run_summary.item_count({"ok": True}) is None
    assert run_summary.item_count("text") is None


@_handle_project
def test_an_answer_or_arguments_too_large_to_keep_read_unknown(
    summary_on,
    music_env,
    monkeypatch,
):
    from unify.function_manager import store_cases

    monkeypatch.setattr(store_cases, "MAX_VALUE_CHARS", 8)
    fm = tfc._FM()
    fm.add_functions(implementations=[LIST_SINCE])
    _run_under(
        "Count my tracks since 2000.",
        tfc._load(fm)["count_tracks_since"],
        2000,
        solved=True,
    )
    accepted = run_summary.summary(_fid(fm, "count_tracks_since"))
    assert accepted["runs"] == 1
    assert accepted["endpoints"]["music.list_tracks"]["items"] == "unknown"
    assert accepted["arguments"] == "unknown"
