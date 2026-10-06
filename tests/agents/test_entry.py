"""Symbolic: the record entry and how @-mentions are read from text."""

import pytest

from unify.agents.entry import Entry, mention_tokens, valid_name


def test_round_trip_keeps_every_field_and_ignores_unknown_keys():
    e = Entry(
        seq=3,
        ts="2026-10-06T11:00:00.000Z",
        author="h1",
        kind="reply",
        text="done",
        mentions=("root",),
    )
    d = e.to_dict()
    assert d == {
        "seq": 3,
        "ts": "2026-10-06T11:00:00.000Z",
        "author": "h1",
        "kind": "reply",
        "text": "done",
        "mentions": ["root"],
    }
    assert Entry.from_dict({**d, "tags": ["later"], "reply_to": 1}) == e


@pytest.mark.parametrize(
    "text, expected",
    [
        ("@h1 please, and @root.", ["h1", "root"]),
        ("@H1, then @h1 again", ["h1"]),
        ("mail ops@root.dev or @@h1", []),
        ("@csv-stats: rows?", ["csv-stats"]),
        ("@all stop", ["all"]),
        ("trailing @h2-", ["h2"]),
        ("(@plots)", ["plots"]),
    ],
)
def test_mentions_in_prose(text, expected):
    assert mention_tokens(text) == expected


@pytest.mark.parametrize(
    "name, ok",
    [
        ("h1", True),
        ("csv-stats", True),
        ("a", True),
        ("-x", False),
        ("x-", False),
        ("Root", False),
        ("a" * 33, False),
        ("", False),
    ],
)
def test_names(name, ok):
    assert valid_name(name) is ok
