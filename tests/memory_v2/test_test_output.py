"""Whole test output under v2.1 (spec P5): spooled, kept as a blob, shown head-first with a marker; jail unchanged."""

import io
import os
import platform
import shutil

import pytest

from unify.memory_v2 import sandbox_run as sr
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.gate import _shown
from unify.memory_v2.gate_v21 import OUTPUT_VIEW_BYTES, keep_output
from unify.memory_v2.sandbox_run import PytestOutcome, SandboxResult
from tests.memory_v2.test_gate import _candidate
from tests.memory_v2.test_gate_v21 import (
    TEST,
    _files_and_fixtures,
    _man,
    world21,
)  # noqa: F401 (fixture)

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None
    or platform.machine() != "x86_64"
    or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap, x86_64 and prlimit required (as run_confined checks)",
)
BIG = b"x" * (3 * sr.CAPTURE_MAX_BYTES) + b"END"


def _read_fd(fd: int) -> bytes:
    out, off = b"", 0
    while chunk := os.pread(fd, 1 << 20, off):
        out, off = out + chunk, off + len(chunk)
    return out


class _FakePopen:
    seen: list = []

    def __init__(self, args, **kw):
        fds = kw["pass_fds"]
        # the fd numbers may differ between calls; what they hold must not
        norm = [
            ("<fd>" if i and args[i - 1] in ("--args", "--seccomp") else a)
            for i, a in enumerate(args)
        ]
        files = [_read_fd(fd) for fd in fds]
        _FakePopen.seen.append(
            (
                norm,
                {
                    k: v
                    for k, v in kw.items()
                    if k not in ("pass_fds", "stdout", "stderr")
                },
                files,
            ),
        )
        self.stdout = io.BufferedReader(io.BytesIO(BIG))
        self.stderr = io.BufferedReader(io.BytesIO(b"warn\n"))
        self.returncode = 0
        self.pid = 0

    def wait(self, timeout=None):
        return 0


@needs_box
def test_spool_keeps_the_whole_output_and_leaves_the_jail_byte_identical(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(sr.subprocess, "Popen", _FakePopen)
    _FakePopen.seen = []
    kw = dict(
        ro={tmp_path: "/memory"},
        rw={},
        cwd="/memory",
        timeout_s=5,
        env={"PYTHONPATH": "/memory"},
    )
    plain = sr.run_confined(["python", "-c", "pass"], **kw)
    spool = (tmp_path / "out", tmp_path / "err")
    spooled = sr.run_confined(["python", "-c", "pass"], spool=spool, **kw)
    (a_args, a_kw, a_files), (b_args, b_kw, b_files) = _FakePopen.seen
    assert (
        a_args == b_args and a_kw == b_kw and a_files == b_files
    )  # bwrap args, env file, seccomp: identical
    assert plain.stdout == spooled.stdout and plain.stdout.startswith(
        "[",
    )  # the 1 MiB in-memory tail, marked
    assert (
        spool[0].read_bytes() == BIG and spool[1].read_bytes() == b"warn\n"
    )  # every byte kept


def test_run_pytest_passes_the_spool_and_nothing_else(tmp_path, monkeypatch):
    seen = []

    def fake(argv, **kw):
        seen.append((argv, kw))
        return SandboxResult(5, "", "", False)

    monkeypatch.setattr(sr, "run_confined", fake)
    kw = dict(
        python=sr.PYTHON,
        ro={tmp_path: "/memory"},
        rw={},
        cwd="/memory",
        timeout_s=9,
        env={"PYTHONPATH": "/memory"},
    )
    sr.run_pytest("memory/a/tests", **kw)
    sr.run_pytest("memory/a/tests", spool_dir=tmp_path / "spool", **kw)
    (a_argv, a), (b_argv, b) = seen
    assert (
        a_argv == b_argv
        and a["ro"] == b["ro"]
        and a["env"] == b["env"]
        and a["cwd"] == b["cwd"]
    )
    assert sorted(a["rw"].values()) == sorted(b["rw"].values()) == ["/junit"]
    assert "spool" not in a and b["spool"] == (
        tmp_path / "spool" / "stdout",
        tmp_path / "spool" / "stderr",
    )


def test_keep_output_stores_the_blob_and_marks_the_head(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "stdout").write_bytes(b"a" * 10_000)
    (spool / "stderr").write_bytes(b"E")
    blobs = BlobStore(tmp_path / "b")
    o = PytestOutcome(output="the old tail")
    keep_output(o, spool, blobs)
    assert blobs.get(o.output_blob) == b"a" * 10_000 + b"E"
    assert o.output.startswith("a" * OUTPUT_VIEW_BYTES)
    assert o.output.endswith(
        f"[… shown bytes 0–{OUTPUT_VIEW_BYTES} of 10001; full output: blob {o.output_blob}]",
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    o2 = PytestOutcome(output="")
    keep_output(o2, empty, blobs)
    assert o2.output == "" and o2.output_blob is None


def test_put_file_streams_and_matches_put(tmp_path):
    blobs = BlobStore(tmp_path / "b")
    f = tmp_path / "f"
    f.write_bytes(b"z" * 3_000_000)
    assert blobs.put_file(f) == blobs.put(b"z" * 3_000_000)


def test_shown_is_v2_without_a_blob():
    assert _shown(PytestOutcome(output="x" * 500)) == "x" * 300
    o = PytestOutcome(output="y" * 500, output_blob="b" * 64)
    assert _shown(o) == "y" * 300 + f" [full output: blob {'b' * 64}]"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")
def test_failing_test_output_is_kept_whole_as_a_blob(world21):
    mem, ev, blobs, gate = world21
    files, fixtures = _files_and_fixtures(
        blobs,
        test=TEST.replace('== "u-1"', '== "u-1" and print("z" * 9000)'),
    )
    res = gate().check(mem.head(), _candidate(mem, files), _man(fixtures))
    assert not res.passed
    [line] = [r for r in res.reasons if "is not green on the candidate" in r]
    sha = line.rsplit("full output: blob ", 1)[1].rstrip("]")
    assert b"z" * 9000 in blobs.get(sha)


def test_spool_stops_at_its_limit_with_a_marker_and_the_run_is_still_kept(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    t = sr._Tail(
        io.BufferedReader(io.BytesIO(b"x" * 100)),
        spool=spool / "stdout",
        spool_max=60,
    )
    assert t.text(5) == "x" * 100  # the in-memory tail is unchanged
    marker = b"\n[spool limit reached: 40 further bytes not kept]\n"
    assert (spool / "stdout").read_bytes() == b"x" * 60 + marker
    assert sr.SPOOL_MAX_BYTES == 64 * 1024**2
    blobs = BlobStore(tmp_path / "b")
    o = PytestOutcome(output="x" * 100)
    keep_output(o, spool, blobs)
    assert blobs.get(o.output_blob) == b"x" * 60 + marker
