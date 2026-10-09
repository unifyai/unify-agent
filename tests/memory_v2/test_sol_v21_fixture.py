import json
import os

from unify.memory_v2 import fixtures as fx
from unify.memory_v2.sol_pass import (
    _fixture_call,
    _tests_changed,
    _with_fixtures,
    sol_tools,
)
from tests.memory_v2.test_fixtures import BLOB, EP, _blob

EPS = {"e1": EP}


def test_fixture_call_copies_recorded_bytes_and_records_provenance(tmp_path):
    made = {}
    out = _fixture_call(
        {"episode": "e1", "action": 2, "dest": "memory/office/tests/data/pay.csv"},
        EPS,
        tmp_path,
        _blob,
        made,
    )
    assert out.startswith("ok: wrote memory/office/tests/data/pay.csv")
    assert (tmp_path / "memory/office/tests/data/pay.csv").read_bytes() == BLOB
    assert made["memory/office/tests/data/pay.csv"]["kind"] == "copy"
    rel = fx.inputs_file("memory.acct.users:user_id")
    for action in (0, "1"):
        assert _fixture_call(
            {"episode": "e1", "action": action, "dest": rel, "append": True},
            EPS,
            tmp_path,
            _blob,
            made,
        ).startswith("ok: appended a line to")
    data = (tmp_path / rel).read_bytes()
    assert len(data.splitlines()) == 2
    assert fx.verify(rel, data, made[rel], EPS.get, _blob) is None


def test_fixture_call_refuses_links_escapes_and_unknown_episodes(tmp_path):
    made = {}
    outside = tmp_path / "outside"
    outside.mkdir()
    box = tmp_path / "box"
    (box / "memory").mkdir(parents=True)
    os.symlink(outside, box / "memory" / "office")
    out = _fixture_call(
        {"episode": "e1", "action": 0, "dest": "memory/office/tests/data/x.json"},
        EPS,
        box,
        _blob,
        made,
    )
    assert out == "refused: a directory on the way to dest is a link"
    assert list(outside.iterdir()) == [] and made == {}
    assert (
        _fixture_call(
            {"episode": "e9", "action": 0, "dest": "memory/a/tests/data/x.json"},
            EPS,
            box,
            _blob,
            made,
        )
        == "refused: 'e9' is not in this batch"
    )
    assert _fixture_call(
        {"episode": "e1", "action": 0, "dest": "memory/a/tests/data/../../x"},
        EPS,
        box,
        _blob,
        made,
    ).startswith("refused: ")
    (box / "memory/a/tests/data").mkdir(parents=True)
    (box / "memory/a/tests/data/own.jsonl").write_text("{}\n")
    assert (
        _fixture_call(
            {
                "episode": "e1",
                "action": 0,
                "dest": "memory/a/tests/data/own.jsonl",
                "append": True,
            },
            EPS,
            box,
            _blob,
            made,
        )
        == "refused: dest exists and did not come through fixture()"
    )


def test_harness_provenance_replaces_the_manifests():
    made = {"memory/a/tests/data/x.json": {"kind": "copy"}}
    forged = {"items": [], "fixtures": {"memory/a/tests/data/y.json": {"kind": "copy"}}}
    assert _with_fixtures(forged, made) == {"items": [], "fixtures": made}
    assert _with_fixtures(None, made) is None


def test_tests_changed_trailers():
    m = {
        "tests_changed": {
            "memory/a/tests/test_x.py::test_old": "merged into test_new",
            "bad": 3,
            "x": " ",
        },
    }
    assert _tests_changed(m) == [
        "memory/a/tests/test_x.py::test_old: merged into test_new",
    ]
    assert _tests_changed(None) == []


def test_fixture_tool_listed_only_with_v21():
    names = lambda ts: {t["function"]["name"] for t in ts}
    assert "fixture" not in names(sol_tools())
    tool = next(t for t in sol_tools(v21=True) if t["function"]["name"] == "fixture")
    assert tool["function"]["parameters"]["required"] == ["episode", "action", "dest"]
    assert json.dumps(tool)  # serialisable as sent to the model
