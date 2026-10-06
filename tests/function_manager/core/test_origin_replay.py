"""Symbolic: ``UNIFY_ORIGIN_REPLAY_STATUS``, show whether a stored function returns its own request's answer.

With ``UNIFY_CAPTURE_ACCEPTED`` each function the storage review writes is
run on the values of the code cell behind the session's answer. This keeps
the result per function and source -- returns it, returns something else,
or could not be run (and why) -- and shows it as one line on the evidence
record and in search results. A function changed since shows nothing. The
functions run in the confined case replay; no model or network is called.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.test_origin_link import (  # noqa: F401 (fixture)
    MIRROR,
    USES_ENV,
    WRONG,
    _trajectory,
    capture,
)
from unify.function_manager import entry_record, origin_capture as oc, origin_replay
from unify.function_manager import task_origin
from unify.function_manager.function_manager import FunctionManager
from unify.settings import ProductionSettings, SETTINGS

TWO_ARGS = "def blend_rows(first, second):\n    return [a + b for a, b in zip(first, second)]\n"


@pytest.fixture
def status_on(monkeypatch, capture):
    monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_REPLAY_STATUS", True)


def _stored(fm, name):
    return next(r for r in fm._library_rows() if r["name"] == name)


def _line(fm, name):
    row = _stored(fm, name)
    return origin_replay.line(row["function_id"], row["implementation"])


def test_the_switch_is_off_by_default():
    assert (
        ProductionSettings.model_fields["UNIFY_ORIGIN_REPLAY_STATUS"].default is False
    )


def test_each_outcome_is_kept_and_said(status_on):
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        fm.add_functions(implementations=[MIRROR, WRONG, USES_ENV, TWO_ARGS])
    assert _line(fm, "mirror_rows") == (
        "run on the values of the request it was stored for, it returns that "
        "request's answer"
    )
    assert _line(fm, "keep_rows") == (
        "run on the values of the request it was stored for, it does not return "
        "that request's answer"
    )
    assert _line(fm, "mirror_with_log").startswith(
        "not run on the values of the request it was stored for (it uses primitives.music",
    )
    assert _line(fm, "blend_rows") == (
        "not run on the values of the request it was stored for (its parameters do "
        "not take the values of the code behind the answer)"
    )


def test_a_function_changed_since_shows_nothing(status_on):
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        fm.add_functions(implementations=[MIRROR])
    changed = MIRROR.replace("row[::-1]", "list(reversed(row))")
    fm.add_functions(implementations=[changed], overwrite=True)
    assert _line(fm, "mirror_rows") is None


def test_the_line_reaches_search_notes_and_the_evidence_record(status_on):
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        fm.add_functions(implementations=[MIRROR])
    row = _stored(fm, "mirror_rows")
    marker = task_origin.Marker([row])
    assert task_origin.listing_notes_enabled()
    notes = task_origin.listing_notes(marker, "function", row)
    assert notes[task_origin.ORIGIN_REPLAY].endswith("it returns that request's answer")
    record = entry_record.record_text(marker, entry_record.FUNCTION, row)
    assert "it returns that request's answer" in record


def test_off_keeps_and_says_nothing(capture, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_REPLAY_STATUS", False)
    fm = FunctionManager(include_primitives=False)
    with oc.reviewing(oc.find_answer_cell(_trajectory())):
        fm.add_functions(implementations=[MIRROR])
    assert _line(fm, "mirror_rows") is None
    row = _stored(fm, "mirror_rows")
    assert task_origin.ORIGIN_REPLAY not in task_origin.listing_notes(
        task_origin.Marker([row]),
        "function",
        row,
    )
