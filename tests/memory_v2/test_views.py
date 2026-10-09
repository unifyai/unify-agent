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


def test_view_length_never_exceeds_view_bytes():
    out = v.view(b"a" * 20000, 0, 10_000_000)
    assert out.endswith("[… shown bytes 0–8000 of 20000; next: offset=8000]")
    assert len(out.split("\n[…")[0]) == v.VIEW_BYTES


def test_view_edges_snap_to_character_boundaries():
    data = ("é" * 5000).encode()  # 2 bytes per character
    text, a, b = v.view_range(data, 0, 8001)
    assert (a, b) == (0, 8000) and "�" not in text
    assert text.endswith("[… shown bytes 0–8000 of 10000; next: offset=8000]")
    text, a, b = v.view_range(data, 8001)
    assert (a, b) == (8002, 10000) and "�" not in text and text == "é" * 999


def test_read_pages_a_large_file_like_view(tmp_path: Path):
    root = tmp_path / "memory"
    root.mkdir()
    data = ("line\n" * 5000).encode()
    (root / "big.txt").write_bytes(data)
    roots = {"/memory": root}
    for off in (0, 8000, 24000):
        assert v.read("/memory/big.txt", roots, off) == v.view(data, off)
    (root / "blob.bin").write_bytes(b"\x00" * 20000)
    assert v.read("/memory/blob.bin", roots) == v.view(b"\x00" * 20000)


def test_grep_marks_a_cut_line_with_where_to_read_it(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    long = "x" * 600 + "needle" + "y" * 394
    (root / "f.txt").write_text("short\n" + long + "\n")
    out = v.grep("needle", "/inputs", {"/inputs": root})
    assert out == (
        "/inputs/f.txt:2: "
        + long[:300]
        + "… [line 2: 1000 chars, shown 300; read(/inputs/f.txt, offset=6)]"
    )


def test_grep_output_is_bounded_and_marked(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "f.txt").write_text(("needle " + "z" * 280 + "\n") * 50)
    out = v.grep("needle", "/inputs", {"/inputs": root})
    body, marker = out.rsplit("\n[… hits ", 1)
    shown = body.count("\n") + 1
    assert len(body.encode()) <= v.VIEW_BYTES
    assert shown < 50 and marker == f"0–{shown} of 50; next: offset={shown}]"
    nxt = v.grep("needle", "/inputs", {"/inputs": root}, offset=shown)
    assert nxt.splitlines()[0].startswith(f"/inputs/f.txt:{shown + 1}: needle")


def test_coverage_requires_full_reads_or_dismissal():
    cov = v.Coverage(
        required={"e1": ["request", "cell:0"], "e2": ["request"]},
        sizes={("e1", "request"): 10, ("e1", "cell:0"): 20000, ("e2", "request"): 5},
    )
    assert cov.missing() == ["e1", "e2"]
    cov.credit("e1", "request", 0, 10)
    cov.credit("e1", "cell:0", 0, 8000)
    cov.credit("e1", "cell:0", 16000, 20000)
    assert "e1" in cov.missing()  # 8000–16000 never read
    cov.credit("e1", "cell:0", 8000, 16000)
    assert cov.missing() == ["e2"]
    assert cov.dismiss("e2", "") == "refused: give a one-line reason"
    assert cov.dismiss("e2", "duplicate of e1's request") == "ok"
    s = cov.summary()
    assert s == {
        "episodes": 2,
        "covered": 1,
        "dismissed": {"e2": "duplicate of e1's request"},
        "missing": [],
        "parts_required": 3,
        "parts_read": 2,
        "bytes_required": 20015,
        "bytes_read": 20010,
    }


def test_coverage_fails_closed_on_a_missing_or_bad_size():
    with pytest.raises(ValueError, match="size"):
        v.Coverage(required={"e1": ["request", "cell:0"]}, sizes={("e1", "request"): 3})
    with pytest.raises(ValueError, match="size"):
        v.Coverage(required={"e1": ["request"]}, sizes={("e1", "request"): -1})


def test_coverage_counts_an_empty_part_as_read_and_bytes_are_clipped():
    cov = v.Coverage(
        required={"e1": ["request", "observation:0"]},
        sizes={("e1", "request"): 3, ("e1", "observation:0"): 0},
    )
    cov.credit(
        "e1",
        "request",
        0,
        8000,
    )  # a range past the end counts only the part's bytes
    s = cov.summary()
    assert s["missing"] == [] and s["covered"] == 1
    assert (s["parts_required"], s["parts_read"]) == (2, 2)
    assert (s["bytes_required"], s["bytes_read"]) == (3, 3)


def test_grep_refuses_an_invalid_pattern(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    assert v.grep("(", "/inputs", {"/inputs": root}).startswith(
        "refused: invalid pattern:",
    )


def test_grep_bounded_runs_in_a_child_and_times_out(tmp_path: Path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "f.txt").write_text("needle\n" + "a" * 30 + "!\n")
    roots = {"/inputs": root}
    assert (
        v.grep_bounded("needle", "/inputs", roots, timeout_s=10)
        == "/inputs/f.txt:1: needle"
    )
    assert v.grep_bounded("(a+)+$", "/inputs", roots, timeout_s=1) == (
        "[grep timed out after 1 s: narrow the pattern or the path]"
    )
