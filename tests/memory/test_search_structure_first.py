"""Structure in the search (rev 2.3).

The librarian's ``tools.search`` puts first the recorded procedures whose recorded input has the same structure profile
as the current request's: the type tree, the length class of the lists at each level and each object's key count, never
a key's text. It is a sort order only: every item is still listed, nothing is keyed on it, and with no equal profile the
result is byte for byte what it was.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import librarian as L
from unify.memory import library_search as S
from unify.memory.captures import shape


def _p(value):
    return S.shape_profile(shape(value))


def test_the_profile_keeps_no_key_text():
    assert _p({"train": [[1, 2]], "test": [[3]]}) == _p({"alpha": [[9, 9]], "beta": [[0]]})
    text = _p({"a_secret_key_name": [1, 2, 3]})
    assert "a_secret_key_name" not in text


@pytest.mark.parametrize(
    "a, b",
    [
        ({"x": 1, "y": 2}, {"x": 1}),  # key count
        ([1, 2], [1] * 12),  # list length class
        ([[1]], [["s"]]),  # leaf type
        ({"x": [1]}, {"x": {"y": 1}}),  # type tree
    ],
)
def test_a_different_structure_gives_a_different_profile(a, b):
    assert _p(a) != _p(b)


def test_the_request_profile_reads_a_whole_json_request_as_its_value():
    value = {"rows": [[1, 2], [3, 4]], "k": 3}
    assert S.request_profile(json.dumps(value)) == _p(value)
    assert S.request_profile(json.dumps(7)) == S.shape_profile({"kind": "scalar", "type": "int"})


def test_a_request_in_words_has_no_profile_and_reorders_nothing():
    # only a declared input or a whole-JSON request has a structure to compare (the search order needs one of them)
    assert S.request_profile("a request\nin two lines") is None
    assert S.request_profile("") is None and S.request_profile(None) is None
    assert S.shape_profile({"kind": "text", "length": "10-99"}) is None


ROWS = [
    {"id": "memory.a.b:f", "kind": "function", "path": "memory/a/b.py", "card": "f", "text": "t"},
    {"id": "capture:" + "1" * 16, "kind": "captured procedure", "path": f".memory/captures/{'1' * 16}/procedure.py",
     "card": "c1", "text": "t"},
    {"id": "notes/a/n.md", "kind": "note", "path": "notes/a/n.md", "card": "n", "text": "t"},
    {"id": "capture:" + "2" * 16, "kind": "captured procedure", "path": f".memory/captures/{'2' * 16}/procedure.py",
     "card": "c2", "text": "t"},
]
DATA = {"version": 1, "backend": "none", "model": None, "why": S.NOT_CONFIGURED, "items": ROWS}
REQUEST = {"rows": [[1, 2], [3, 4]]}


def _copy(tmp_path: Path) -> Path:
    copy = tmp_path / "copy"
    for cid, value in (("1" * 16, ["text"]), ("2" * 16, {"cells": [[0, 0], [0, 0]]})):
        d = copy / ".memory" / "captures" / cid
        d.mkdir(parents=True)
        (d / "signature.json").write_text(json.dumps({"input": shape(value)}))
    (copy / ".memory" / "search.json").write_text(json.dumps(DATA))
    return copy


def test_the_equal_structure_comes_first_and_every_item_is_still_listed(tmp_path):
    copy = _copy(tmp_path)
    profiles = S.capture_profiles(copy, DATA)
    assert set(profiles) == {"capture:" + "1" * 16, "capture:" + "2" * 16}
    text = S.search_text(DATA, "q", profiles=profiles, wanted=S.request_profile(json.dumps(REQUEST)))
    lines = text.splitlines()
    assert lines[0].startswith("search: 4 item(s), the first 1 recorded the same input structure as the current request")
    assert [ln.split()[2] for ln in lines[1:]] == [
        "capture:" + "2" * 16,
        "memory.a.b:f",
        "capture:" + "1" * 16,
        "notes/a/n.md",
    ]


def test_with_no_equal_structure_the_result_is_unchanged(tmp_path):
    copy = _copy(tmp_path)
    profiles = S.capture_profiles(copy, DATA)
    before = S.search_text(DATA, "q")
    assert S.search_text(DATA, "q", profiles=profiles, wanted=S.request_profile("plain text request")) == before
    assert S.search_text(DATA, "q", profiles=profiles, wanted=None) == before


def test_the_librarian_search_tool_orders_by_the_request_it_was_asked_about(tmp_path, monkeypatch):
    copy = _copy(tmp_path)
    monkeypatch.setattr(S, "backend", lambda on_call=None: S.NO_BACKEND)
    state = L.LibrarianCalls(ids=[r["id"] for r in ROWS], copy=copy)
    plain = L.search(SimpleNamespace(state=state), {"query": "q"})
    state.request_profile = S.request_profile(json.dumps(REQUEST))
    ordered = L.search(SimpleNamespace(state=state), {"query": "q"})
    assert "the same input structure" not in plain.text
    assert ordered.text.splitlines()[1].split()[2] == "capture:" + "2" * 16
