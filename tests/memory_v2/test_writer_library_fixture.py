"""The fixture tool: recorded bytes into tests/data/, with provenance derived from the files themselves."""

import json
import os

from unify.memory_v2 import fixtures as fx
from unify.memory_v2 import writer_tools as wt
from unify.memory_v2.writer_pass import _tests_changed, _with_fixtures, writer_tools
from tests.memory_v2.test_fixtures import BLOB, EP, _blob

EPS = {"e1": EP}


def _fixture(box, args):
    return wt.fixture(args, EPS.get, box, _blob, ["e1"])


def test_fixture_copies_recorded_bytes_and_the_ledger_comes_from_the_files(tmp_path):
    dest = "memory/office/tests/data/pay.csv"
    out = _fixture(tmp_path, {"episode": "e1", "action": 2, "dest": dest})
    assert out.status == "ok" and out.text.startswith(f"ok: wrote {dest}")
    assert (tmp_path / dest).read_bytes() == BLOB
    ledger = fx.derive_ledger(tmp_path)
    assert ledger[dest]["kind"] == "copy" and ledger[dest]["source"] == "2"
    rel = fx.inputs_file("memory.acct.users:user_id")
    for source in (0, "1"):
        got = _fixture(tmp_path, {"episode": "e1", "source": source, "dest": rel, "append": True})
        assert got.text.startswith("ok: appended a line to"), got.text
    data = (tmp_path / rel).read_bytes()
    assert len(data.splitlines()) == 2
    ledger = fx.derive_ledger(tmp_path)
    assert len(ledger[rel]["lines"]) == 2
    assert fx.verify(rel, data, ledger[rel], EPS.get, _blob) is None
    assert fx.verify(dest, BLOB, ledger[dest], EPS.get, _blob) is None


def test_delete_rewrite_and_replace_keep_the_ledger_equal_to_the_files(tmp_path):
    copy = "memory/office/tests/data/pay.csv"
    rel = "memory/office/tests/data/pay.export.inputs.jsonl"
    _fixture(tmp_path, {"episode": "e1", "source": 2, "dest": copy})
    for source in (0, "1", "request:0"):
        _fixture(tmp_path, {"episode": "e1", "source": source, "dest": rel, "append": True})
    assert len(fx.derive_ledger(tmp_path)[rel]["lines"]) == 3
    # rewriting the copied file drops its entry: the gate then asks for fixture() again
    (tmp_path / copy).write_bytes(b"edited\n")
    assert copy not in fx.derive_ledger(tmp_path)
    # deleting a line re-derives the entry from the lines left (no stale count)
    lines = (tmp_path / rel).read_text().splitlines()
    (tmp_path / rel).write_text("\n".join(lines[:2]) + "\n")
    entry = fx.derive_ledger(tmp_path)[rel]
    assert len(entry["lines"]) == 2
    assert fx.verify(rel, (tmp_path / rel).read_bytes(), entry, EPS.get, _blob) is None
    # deleting the file drops it entirely
    os.unlink(tmp_path / rel)
    assert rel not in fx.derive_ledger(tmp_path)
    # a hand-written non-input file refuses append, and replace=true starts it over
    (tmp_path / rel).write_text("{}\n")
    got = _fixture(tmp_path, {"episode": "e1", "source": 0, "dest": rel, "append": True})
    assert got.status == "refused" and got.form == "not_input_lines"
    got = _fixture(
        tmp_path,
        {"episode": "e1", "source": "observation:0", "dest": rel, "append": True, "replace": True},
    )
    assert got.text.startswith(f"ok: started {rel}")
    entry = fx.derive_ledger(tmp_path)[rel]
    assert [r["source"] for r in entry["lines"]] == ["request:1"]  # observation:0 is request:1
    # copying the same dest again overwrites it and lists it again
    _fixture(tmp_path, {"episode": "e1", "source": 2, "dest": copy})
    assert fx.derive_ledger(tmp_path)[copy]["sha256"] == fx.hashlib.sha256(BLOB).hexdigest()


def test_open_ended_and_clamped_slices(tmp_path):
    dest = "memory/office/tests/data/note.txt"
    got = _fixture(tmp_path, {"episode": "e1", "source": 1, "dest": dest, "slice": [6, None]})
    assert got.status == "ok" and (tmp_path / dest).read_bytes() == b"text reply"
    got = _fixture(tmp_path, {"episode": "e1", "source": 1, "dest": dest, "slice": [0, 999]})
    assert "truncated: slice end 999 is past the 16 recorded bytes; copied [0, 16]" in got.text
    got = _fixture(tmp_path, {"episode": "e1", "source": 1, "dest": dest, "slice": [16, None]})
    assert got.status == "refused" and "16 recorded bytes" in got.text


def test_fixture_refuses_links_escapes_and_unknown_episodes(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    box = tmp_path / "box"
    (box / "memory").mkdir(parents=True)
    os.symlink(outside, box / "memory" / "office")
    out = _fixture(box, {"episode": "e1", "action": 0, "dest": "memory/office/tests/data/x.json"})
    assert out.text == "refused: a directory on the way to dest is a link"
    assert list(outside.iterdir()) == [] and fx.derive_ledger(box) == {}
    out = _fixture(box, {"episode": "e9", "action": 0, "dest": "memory/a/tests/data/x.json"})
    assert out.text == "refused: 'e9' is not a recorded episode; this batch: e1"
    assert out.form == "unknown_episode"
    out = _fixture(box, {"episode": "e1", "action": 0, "dest": "memory/a/tests/data/../../x"})
    assert out.text.startswith("refused: ")
    assert _fixture(box, {"episode": "e1", "dest": "memory/a/tests/data/y.json"}).form == "no_source"


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


def test_fixture_tool_listed_only_with_library_mode():
    names = lambda ts: {t["function"]["name"] for t in ts}  # noqa: E731
    assert "fixture" not in names(writer_tools())
    tool = next(
        t for t in writer_tools(library_mode=True) if t["function"]["name"] == "fixture"
    )
    params = tool["function"]["parameters"]
    assert params["required"] == ["episode", "dest"]
    assert {"source", "action", "slice", "append", "replace"} <= set(params["properties"])
    assert "clamped" in tool["function"]["description"]
    assert json.dumps(tool)  # serialisable as sent to the model
