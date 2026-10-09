import hashlib
from pathlib import Path

import pytest

from unify.memory_v2 import views as v


def test_view_marks_truncation_head_first():
    data = b"a" * 20000
    out = v.view(data, 0, 8000)
    assert out.startswith("a" * 10)
    assert out.endswith("[… shown bytes 0–8000 of 20000; next: offset=8000]")
    assert v.view(data, 16000, 8000).endswith("a")  # last page: no marker


def test_view_binary():
    data = b"\x00\x01\xff" * 10
    assert (
        v.view(data) == f"[binary: 30 bytes, sha256 {hashlib.sha256(data).hexdigest()}]"
    )


def test_resolve_refuses_escape(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    (tmp_path / "secret").write_text("x")
    (root / "link").symlink_to(tmp_path / "secret")
    roots = {"/inputs": root}
    with pytest.raises(PermissionError, match="refused: outside the readable roots"):
        v.resolve("/inputs/../secret", roots)
    with pytest.raises(PermissionError):
        v.resolve("/inputs/link", roots)
    with pytest.raises(PermissionError):
        v.resolve("/etc/passwd", roots)


def test_read_and_grep(tmp_path: Path):
    root = tmp_path / "inputs"
    (root / "episodes").mkdir(parents=True)
    (root / "episodes" / "e1.json").write_text(
        "alpha\nbeta needle\ngamma\n" + "needle\n" * 60,
    )
    roots = {"/inputs": root}
    assert v.read("/inputs/episodes/e1.json", roots).startswith("alpha\nbeta needle")
    out = v.grep("needle", "/inputs", roots, max_hits=50)
    first = out.splitlines()[0]
    assert first == "/inputs/episodes/e1.json:2: beta needle"
    assert out.endswith("[… hits 0–50 of 61; next: offset=50]")


def test_grep_does_not_follow_symlinked_directories(tmp_path: Path):
    root = tmp_path / "memory"
    (root / "d").mkdir(parents=True)
    (root / "d" / "f.txt").write_text("needle\n")
    (root / "loop").symlink_to(
        root,
    )  # a directory cycle the writer could make in its own box
    (root / "out").symlink_to(tmp_path)  # and one pointing outside the root
    out = v.grep("needle", "/memory", {"/memory": root})
    assert out == "/memory/d/f.txt:1: needle"
