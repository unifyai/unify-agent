"""replay writes its test and its case file into the writer's box, which the writer's cells write: neither file is
written through a link, at the file or on the way to it, so a cell cannot turn a host write into one outside the box.
"""

import json
import os

import pytest

from unify.memory import replay_verify as rv
from unify.memory import workspace_trees
from unify.memory import writer_pass
from tests.memory.test_writer_pass import _writer

ITEM = "memory.pkg.mod:f"
TEST_REL = "memory/pkg/tests/test_f_replay.py"
CASE_REL = "memory/pkg/tests/data/f.replay.json"


@pytest.fixture
def replay_world(tmp_path, monkeypatch):
    box = tmp_path / "box"
    (box / ".pass").mkdir(parents=True)
    manifest = {"items": [{"item": ITEM, "replaces": [{"episode": "e1", "cell": 0}]}]}
    (box / ".pass" / "manifest.json").write_text(json.dumps(manifest))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.py").write_text("untouched\n")
    monkeypatch.setattr(workspace_trees, "trees_for", lambda *a, **k: {})
    monkeypatch.setattr(
        rv,
        "verify",
        lambda *a, **k: [
            {
                "verdict": "verified",
                "case": {"input": 1},
                "episode": "e1",
                "cell": 0,
                "lines": [1, 2],
            },
        ],
    )
    monkeypatch.setattr(
        rv,
        "test_source",
        lambda item, cases: (TEST_REL, "def test_f():\n    pass\n"),
    )
    monkeypatch.setattr(rv, "case_file", lambda item, cases: (CASE_REL, b"[]\n"))
    _, _, writer = _writer(tmp_path / "w", None, library_mode=True)
    return writer, box, outside, tmp_path / "scratch"


def test_a_link_at_the_test_file_is_refused_and_nothing_outside_changes(replay_world):
    writer, box, outside, tmp = replay_world
    tmp.mkdir()
    (box / TEST_REL).parent.mkdir(parents=True)
    os.symlink(outside / "victim.py", box / TEST_REL)
    reply = writer._replay_call({"item": ITEM}, box, tmp)
    assert "a link is on the way" in reply and TEST_REL in reply
    assert (outside / "victim.py").read_text() == "untouched\n"


def test_a_link_on_the_way_to_the_test_file_is_refused(replay_world):
    writer, box, outside, tmp = replay_world
    tmp.mkdir()
    (box / "memory" / "pkg").mkdir(parents=True)
    os.symlink(outside, box / "memory" / "pkg" / "tests")
    reply = writer._replay_call({"item": ITEM}, box, tmp)
    assert "a link is on the way" in reply
    assert sorted(p.name for p in outside.iterdir()) == ["victim.py"]


def test_without_links_both_files_are_written(replay_world):
    writer, box, outside, tmp = replay_world
    tmp.mkdir()
    reply = writer._replay_call({"item": ITEM}, box, tmp)
    assert "a link is on the way" not in reply
    assert (box / TEST_REL).read_text() == "def test_f():\n    pass\n"
    assert (box / CASE_REL).read_bytes() == b"[]\n"


def test_the_box_writer_never_follows_a_link_at_the_file(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "victim").write_text("untouched")
    os.symlink(tmp_path / "victim", tmp_path / "box" / "f.py")
    with pytest.raises(OSError):
        writer_pass._write_in_box(tmp_path / "box", "f.py", b"x")
    assert (tmp_path / "victim").read_text() == "untouched"
