"""Symbolic: ``UNIFY_PLACEHOLDER_NOTE`` names a credential argument that received a stand-in.

In an AppWorld HIGH cell the model held a fresh token in the session variable
``access_token`` and called a stored function through the ``execute_function``
JSON tool with ``{"access_token": "{{access_token}}"}``. That tool is gone
with the core surface; the detector it used is tested here. No model is called.
"""

from __future__ import annotations


import pytest

from unify.actor import placeholder_note

REAL_TOKEN = "eyJhbGciOiJIUzI1NiJ9realtoken42"


# --------------------------------------------------------------------------- #
#  The detector                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("access_token", "{{access_token}}"),
        ("access_token", "{{session.spotify_token}}"),
        ("access_token", "$spotify_token"),
        ("access_token", "<hidden-token>"),
        ("access_token", "access_token"),
        ("access_token", ""),
        ("password", " "),
        ("password", "unknown"),
        ("access_token", "x"),
        ("access_token", "token"),
        ("apiKey", "API_KEY"),
    ],
)
def test_stand_ins_seen_in_the_logs_are_recognised(parameter, value):
    assert placeholder_note.stand_in(parameter, value)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("access_token", REAL_TOKEN),
        ("password", "hunter2pw"),
        ("password", None),
        ("sort_key", ""),
        ("keyword", "x"),
        ("author", ""),
        ("title", "{{name}}"),
    ],
)
def test_real_values_and_ordinary_parameters_are_not_stand_ins(parameter, value):
    assert not placeholder_note.stand_in(parameter, value)


def test_a_long_stand_in_is_shortened_when_echoed():
    note = placeholder_note.note({"token": "{{" + "a" * 200 + "}}"})
    echoed = note.split("`")[3]
    assert len(echoed) == 60 and echoed.endswith("…")
