"""Staging reaches the writer read-only as untrusted input."""

import asyncio
import json
import os

from unify.memory import staging
from unify.memory.trigger import PassRequest
from tests.memory.test_writer_pass import Turns, _call, _writer
from tests.memory.test_writer_library_tools import _episode


def _staged(root, eid, status="fork.json"):
    d = root / eid
    (d / "candidates").mkdir(parents=True)
    (d / "candidates" / "f.py").write_text("def f(x):\n    return x\n")
    if status:
        (d / status).write_text(json.dumps({"status": "ok"}))
    return d


def test_staging_export_refuses_paths_links_and_sizes(tmp_path):
    d = _staged(tmp_path, "e1")
    os.symlink("/etc/passwd", d / "candidates" / "link.py")
    (d / "evil.sh").write_text("rm -rf /")
    (d / "notes.md").write_bytes(b"x" * (staging.QUOTA_BYTES + 1))
    v = staging.read(tmp_path, "e1")
    assert (
        v.source == "fork" and v.status == "ok" and set(v.files) == {"candidates/f.py"}
    )
    assert {r["path"] for r in v.refused} == {
        "candidates/link.py",
        "evil.sh",
        "notes.md",
    }
    entry = staging.export(v, tmp_path / "out" / "e1")
    assert (
        entry["files"] == ["candidates/f.py"]
        and (tmp_path / "out" / "e1" / "candidates" / "f.py").is_file()
    )
    assert not (tmp_path / "out" / "e1" / "evil.sh").exists()


def test_a_staging_dir_without_its_status_file_is_still_being_written(tmp_path):
    _staged(tmp_path, "e2", status=None)
    assert staging.read(tmp_path, "e2") is None and staging.export(
        None,
        tmp_path / "x",
    ) == {"status": "none"}


def test_a_staging_dir_that_is_a_link_is_ignored(tmp_path):
    real = _staged(tmp_path / "elsewhere", "e3")
    (tmp_path / "root").mkdir()
    os.symlink(real, tmp_path / "root" / "e3")
    assert staging.read(tmp_path / "root", "e3") is None


def test_the_pass_lists_each_episodes_staging_in_the_batch_map(tmp_path):
    root = tmp_path / "staging-root"
    _staged(root, "e2")
    model = Turns(
        [
            _call("m", "read", {"path": "/inputs/batch_map.json"}),
            _call("f", "read", {"path": "/inputs/staging/e2/candidates/f.py"}),
        ],
    )
    _, _, writer = _writer(
        tmp_path,
        model,
        max_calls=6,
        staging_root=str(root),
    )
    writer.load = {"e2": _episode("e2")}.__getitem__
    asyncio.run(writer.run(PassRequest("incremental", "svc", ["e2"], False), "p1"))
    # the row lists the staged files but never names the reader that staged them
    assert '"files": [' in model.outputs["m"] and '"source": "fork"' not in model.outputs["m"]
    assert "def f(x)" in model.outputs["f"]
