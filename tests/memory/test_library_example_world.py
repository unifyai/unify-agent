"""The shared library-mode world is what it says: recorded fixtures from two episodes, an item without input or
action covers, and canned example calls that match the test file's lines.
"""

import ast
import json

from unify.memory import fixtures as fx
from unify.memory.blobs import BlobStore
from unify.memory.example_calls import CallKey
from tests.memory import library_example_world as w


def test_the_fixtures_re_derive_from_two_episodes(tmp_path):
    blobs = BlobStore(tmp_path / "b")
    files, fixtures = w.files_and_fixtures(blobs.get)
    assert set(fixtures) == {*w.FIXTURE_PATHS, w.INPUTS_PATH}
    assert {fixtures[p]["episode"] for p in w.FIXTURE_PATHS} == {"e1", "e2"}
    for path in w.FIXTURE_PATHS:
        assert fixtures[path]["kind"] == "copy" and fixtures[path]["slice"] is None
    assert fixtures[w.INPUTS_PATH]["kind"] == "assembled"
    assert [r["episode"] for r in fixtures[w.INPUTS_PATH]["lines"]] == ["e1", "e2"]
    for path, entry in fixtures.items():
        assert fx.verify(path, files[path].encode(), entry, w.EPISODES.get, blobs.get) is None
    for rel in (w.MODULE, w.TEST_FILE):
        ast.parse(files[rel])


def test_two_fixtures_may_come_from_one_episode(tmp_path):
    blobs = BlobStore(tmp_path / "b")
    files, fixtures = w.files_and_fixtures(blobs.get, sources=(w.E1, w.E1))
    assert {fixtures[p]["episode"] for p in w.FIXTURE_PATHS} == {"e1"}
    assert {r["episode"] for r in fixtures[w.INPUTS_PATH]["lines"]} == {"e1"}
    assert files[w.TEST_FILE].count('== "u-1"') == 2


def test_the_item_has_no_input_and_no_action_cover():
    man = w.manifest({})
    (it,) = man["items"]
    assert "input" not in it and it["covers"] == []
    assert it["tests"] == [w.TEST_FILE] and set(it["source_episodes"]) == {"e1", "e2"}
    assert "Input:" not in w.MODULE_SOURCE and f"Example: {w.EXAMPLE_TEST}" in w.MODULE_SOURCE
    assert "def user_id(response: dict)" in w.MODULE_SOURCE


def test_recorded_calls_point_at_the_test_file_lines(tmp_path):
    blobs = BlobStore(tmp_path / "b")
    files, _ = w.files_and_fixtures(blobs.get)
    lines = files[w.TEST_FILE].splitlines()
    calls = w.recorded_calls(blobs.get)
    assert [c.outcome for c in calls] == ["returned"] * 4 + ["refused"]
    for c in calls:
        assert "user_id(" in lines[c.line - 1] and c.item == w.ITEM_ID
    for c, path in zip(calls, w.FIXTURE_PATHS):
        assert c.args[0].json == json.loads(files[path])
        assert c.result_json == c.args[0].json["user_id"]
    rows = [json.loads(x)["input"] for x in files[w.INPUTS_PATH].splitlines()]
    assert [c.args[0].json for c in calls[2:4]] == rows
    assert [c.key for c in calls[:4]] == [
        CallKey(w.EXAMPLE_TEST, (), 0),
        CallKey(w.EXAMPLE_TEST, (), 1),
        CallKey(w.INPUTS_TEST, (("row", 0),), 0),
        CallKey(w.INPUTS_TEST, (("row", 1),), 0),
    ]
    assert len({c.key for c in calls}) == len(calls)
