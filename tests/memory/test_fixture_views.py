"""The values a test can load from a fixture, each by the digest an argument holding it has, with its provenance."""

import hashlib
import json

from unify.memory import fixtures as fx
from unify.memory.fixtures import View

COPY = {"kind": "copy", "episode": "e1", "source": "0", "slice": None, "trust": 1, "sha256": "x"}


def d(value):
    return fx.value_digest(value)


def names(views, value):
    return sorted(v.name for v in views.get(d(value), []))


def test_a_digest_is_the_sha256_of_the_canonical_json():
    assert d({"b": 1, "a": [1, 2]}) == hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    assert d("é") == hashlib.sha256('"é"'.encode()).hexdigest()


def test_json_views_are_the_bytes_the_text_the_value_and_every_part():
    data = b'{"rows": [{"id": 7, "tags": ["a"]}], "n": 1}'
    views = fx.value_views("memory/acct/tests/data/r.json", data, COPY)
    value = json.loads(data)
    assert names(views, {"__bytes__": hashlib.sha256(data).hexdigest()}) == ["whole"]
    assert names(views, data.decode()) == ["text"]
    assert names(views, value) == ["value"]
    assert names(views, value["rows"]) == ["value:rows"]
    assert names(views, {"id": 7, "tags": ["a"]}) == ["value:rows[0]"]
    assert names(views, ["a"]) == ["value:rows[0].tags"]
    assert names(views, "a") == ["value:rows[0].tags[0]"]
    assert names(views, 7) == ["value:rows[0].id"]
    one = views[d(value)][0]
    assert one == View("value", "e1", "0", 1, None)


def test_an_assembled_file_gives_each_line_and_its_input_with_the_line_s_own_episode(tmp_path):
    lines = [
        {"input": {"user_id": "u-1"}, "source": {"episode": "e1", "source": "0", "slice": None}},
        {"input": {"user_id": "u-2"}, "source": {"episode": "e2", "source": "args:3", "slice": None}},
    ]
    data = "".join(json.dumps(x, sort_keys=True) + "\n" for x in lines).encode()
    entry = fx.assembled_entry(data)
    views = fx.value_views("memory/acct/tests/data/users.me.inputs.jsonl", data, entry)
    first = views[d({"user_id": "u-1"})]
    assert {(v.name, v.episode, v.line) for v in first} == {
        ("line:1:input", "e1", 1),
        ("value:[0].input", "e1", 1),
    }
    second = views[d({"user_id": "u-2"})]
    assert {(v.episode, v.source, v.trust, v.line) for v in second} == {("e2", "args:3", 2, 2)}
    assert {v.name for v in views[d(lines[1])]} == {"line:2", "value:[1]"}
    # the whole list of inputs, as a test loads it, comes from both episodes
    whole = views[d(lines)]
    assert {(v.name, v.episode) for v in whole} == {("value", "e1"), ("value", "e2")}
    assert {v.episode for v in views[d("u-1")]} == {"e1"}


def test_a_copied_json_lines_file_gives_its_lines_with_the_copy_s_provenance():
    data = b'{"a": 1}\n\n{"a": 2}\n'
    views = fx.value_views("memory/acct/tests/data/x.jsonl", data, COPY)
    assert [(v.name, v.line) for v in views[d({"a": 2})] if v.name.startswith("line")] == [("line:3", 3)]
    assert names(views, [{"a": 1}, {"a": 2}]) == ["value"]


def test_a_csv_with_a_header_gives_rows_dicts_each_row_and_each_column():
    data = b"name,amount\nAda,120\nBo,95\n"
    views = fx.value_views("memory/office/tests/data/pay.csv", data, COPY)
    rows = [["name", "amount"], ["Ada", "120"], ["Bo", "95"]]
    assert names(views, rows) == ["rows"]
    assert names(views, rows[1:]) == ["rows:body"]
    dicts = [{"name": "Ada", "amount": "120"}, {"name": "Bo", "amount": "95"}]
    assert names(views, dicts) == ["rows:dicts"]
    assert names(views, ["Ada", "120"]) == ["row:2"]
    assert names(views, {"name": "Bo", "amount": "95"}) == ["row:3"]
    assert names(views, ["120", "95"]) == ["column:amount"]
    assert names(views, ["Ada", "Bo"]) == ["column:name"]


def test_a_csv_without_a_header_names_columns_by_position():
    data = b"1,2\n3,4\n"
    views = fx.value_views("memory/office/tests/data/grid.csv", data, COPY)
    assert names(views, [["1", "2"], ["3", "4"]]) == ["rows"]
    assert names(views, ["1", "3"]) == ["column:#1"]
    assert names(views, ["3", "4"]) == ["row:2"]
    assert d([{"1": "3", "2": "4"}]) not in views  # no header, no dicts


def test_yaml_gives_only_the_whole_bytes_and_the_text():
    data = b"a: 1\nb: [2, 3]\n"
    views = fx.value_views("memory/office/tests/data/c.yaml", data, COPY)
    assert sorted(v.name for vs in views.values() for v in vs) == ["text", "whole"]


def test_bytes_that_are_not_text_give_only_the_whole_bytes():
    data = b"\xff\xfe\x00\x01binary"
    views = fx.value_views("memory/office/tests/data/blob.bin", data, COPY)
    assert list(views) == [d({"__bytes__": hashlib.sha256(data).hexdigest()})]
    assert views[list(views)[0]] == [View("whole", "e1", "0", 1, None)]


def test_a_file_without_provenance_has_no_views():
    assert fx.value_views("memory/office/tests/data/r.json", b"[1]", None) == {}


def test_parts_past_the_bound_are_not_views_and_a_note_says_so():
    data = json.dumps(list(range(fx.MAX_VIEW_NODES + 10))).encode()
    notes: list[str] = []
    views = fx.value_views("memory/office/tests/data/big.json", data, COPY, notes)
    parts = [v for vs in views.values() for v in vs if v.name.startswith("value")]
    assert len(parts) == fx.MAX_VIEW_NODES
    assert notes == [
        f"G2 examples: fixture memory/office/tests/data/big.json has more than {fx.MAX_VIEW_NODES} parts; only "
        f"the first {fx.MAX_VIEW_NODES} are matched against arguments",
    ]
