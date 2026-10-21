"""The host handlers open box paths from a root descriptor one component at a time, never through a link, and use
only regular files and directories (:mod:`unify.memory.safe_paths`). Box code can change /memory and /work while a
handler runs: a thread here swaps a directory for a link to a decoy directory outside the roots, a file for a FIFO
and a file for a link to a credential-like decoy file, in a tight loop, while the readers, grep, overview, fixture,
the host write and stage run many times. No reply may hold decoy bytes, nothing outside the roots may change and no
call may hang. Also: a FIFO leaf, hard links, the manifest written by ``pool(op="decide")``, grep's child failure
text and the helper's episode reader."""

import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from unify.memory import helpers, memory_files, safe_paths, tool_spec, views, writer_pass
from unify.memory import writer_tools as wt
from tests.memory.test_helpers import EPS as HELPER_EPS
from tests.memory.test_helpers import _ctx
from tests.memory.test_writer_library_tools import EPS, _episode
from tests.memory.test_writer_pass import _writer

DECOY = "DECOY-planted-key-for-the-resolver-race-test"
#: the whole race's budget; a call that hangs fails the test instead of the suite
RACE_S = 8.0


def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(box, decoy directory, decoy file): the box holds memory/pkg/ (swapped for a link to the decoy directory) and
    memory/other/{mod.py,data.txt} (swapped for a link to the decoy file and for a FIFO)."""
    box = tmp_path / "box"
    pkg = box / "memory" / "pkg"
    (pkg / "tests" / "data").mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1  # needle\n")
    (pkg / "data.txt").write_text("needle data\n")
    other = box / "memory" / "other"
    other.mkdir()
    (other / "mod.py").write_text("y = 2  # needle\n")
    (other / "data.txt").write_text("needle other\n")
    (box / ".pass").mkdir()
    (box / ".pass" / "manifest.json").write_text("{}\n")
    decoy = tmp_path / "decoy"
    (decoy / "tests" / "data").mkdir(parents=True)
    for name in ("mod.py", "data.txt", "INDEX.md"):
        (decoy / name).write_text(f"{DECOY} needle\n")
    secret = tmp_path / "credentials.txt"
    secret.write_text(f"token={DECOY} needle\n")
    return box, decoy, secret


def _snapshot(*paths: Path) -> dict:
    out = {}
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                out[str(f)] = f.read_bytes() if f.is_file() else None
        else:
            out[str(p)] = p.read_bytes()
    return out


class _Swapper(threading.Thread):
    """Swaps, in a tight loop: memory/pkg for a link to *decoy*; memory/other/data.txt for a FIFO;
    memory/other/mod.py for a link to *secret*. Each swap is undone at once; a handler's write in between is
    cleared away so the next swap starts from the same tree."""

    def __init__(self, box: Path, decoy: Path, secret: Path) -> None:
        super().__init__(daemon=True)
        self.box, self.decoy, self.secret = box, decoy, secret
        self.done = threading.Event()
        self.swaps = 0

    def _dir(self) -> None:
        mem = self.box / "memory"
        os.rename(mem / "pkg", mem / ".pkg-saved")
        try:
            os.symlink(self.decoy, mem / "pkg")
            os.unlink(mem / "pkg")
        except OSError:
            pass
        finally:
            for _ in range(1000):
                try:
                    if (mem / "pkg").is_symlink():
                        os.unlink(mem / "pkg")
                    elif (mem / "pkg").exists():  # a handler made it while it was away
                        shutil.rmtree(mem / "pkg")
                    os.rename(mem / ".pkg-saved", mem / "pkg")
                    break
                except OSError:
                    continue

    def _leaf(self, name: str, make) -> None:
        other = self.box / "memory" / "other"
        saved = other / f".{name}-saved"
        os.rename(other / name, saved)
        try:
            make(other / name)
        except OSError:
            pass
        finally:
            os.replace(saved, other / name)

    def run(self) -> None:
        while not self.done.is_set():
            try:
                self._dir()
                self._leaf("data.txt", os.mkfifo)
                self._leaf("mod.py", lambda p: os.symlink(self.secret, p))
                self.swaps += 1
            except OSError:
                continue


def _writer_for(tmp_path: Path, box: Path):
    _, _, writer = _writer(tmp_path / "w", None)
    writer.load = EPS.__getitem__
    writer._shown_full, writer._read_logs = {}, {}
    writer._batch, writer._shared, writer._bmap_rows = ["e1"], {}, []
    writer._episodes, writer._recorded = {}, {"e1", "e2"}
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "batch_map.json").write_text(json.dumps({"episodes": []}))
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    return writer, {"/inputs": inputs, "/memory": box, "/outputs": outputs}


def _helper_state(tmp_path: Path, box: Path) -> tuple:
    ctx = _ctx(tmp_path / "h")
    task = tmp_path / "h" / "task"
    task.mkdir()
    st = {
        "call": "call-0",
        "hin": ctx.inputs,
        "task": task,
        "lib": box,
        "work": box / "memory",
        "cells": tmp_path / "h" / "cells",
        "work_over": None,
        "staging": helpers._Staging(tmp_path / "h" / "staging" / "call-0"),
    }
    return ctx, st, helpers._resolver(ctx)


def _written(write) -> str:
    """A host write's reply text: what it did, or its refusal."""
    try:
        return write()
    except memory_files.Refused as r:
        return r.text


def _calls(writer, roots, box, ctx, st, resolve):
    """One round of every handler on the swapped paths: (label, reply text)."""

    def lib(name, args):
        return asyncio.run(writer._library_tool(name, args, roots)).text

    def helper(name, args):
        return str(asyncio.run(helpers._tool(ctx, name, args, st, resolve))[0])

    fixture_args = {
        "episode": "e1",
        "source": "request:0",
        "dest": "memory/pkg/tests/data/req.json",
    }
    return [
        ("read pkg file", lambda: lib("read", {"path": "/memory/memory/pkg/mod.py"})),
        ("read pkg dir", lambda: lib("read", {"path": "/memory/memory/pkg"})),
        ("read fifo", lambda: lib("read", {"path": "/memory/memory/other/data.txt"})),
        ("read link", lambda: lib("read", {"path": "/memory/memory/other/mod.py"})),
        (
            "grep",
            lambda: views.grep("needle", "/memory", roots),
        ),
        ("overview", lambda: lib("overview", {"episode": "e1"})),
        (
            "fixture",
            lambda: wt.fixture(
                fixture_args,
                writer._episode,
                box,
                writer.gate._blob,
                ["e1"],
            ).text,
        ),
        (
            "write",
            lambda: _written(lambda: memory_files.write(box, "memory/other/mod.py", "y = 3\n")),
        ),
        (
            "write in pkg",
            lambda: _written(lambda: memory_files.write(box, "memory/pkg/new.py", "z = 1\n")),
        ),
        ("helper read", lambda: helper("read", {"path": "/work/pkg/mod.py"})),
        ("helper read link", lambda: helper("read", {"path": "/work/other/mod.py"})),
        (
            "helper overview",
            lambda: helper("overview", {"episode": "e1"}),
        ),
        (
            "helper stage",
            lambda: helper("stage", {"path": "notes/a.md", "text": "seen"}),
        ),
    ]


def test_swapped_links_and_fifos_never_reach_a_reply_or_hang_a_call(tmp_path):
    box, decoy, secret = _tree(tmp_path)
    writer, roots = _writer_for(tmp_path, box)
    ctx, st, resolve = _helper_state(tmp_path, box)
    outside = _snapshot(decoy, secret)
    calls = _calls(writer, roots, box, ctx, st, resolve)
    replies: list[tuple[str, str]] = []
    errors: list[str] = []
    bounded = {"n": 0}

    def drive() -> None:
        end = time.monotonic() + RACE_S
        i = 0
        while time.monotonic() < end:
            label, fn = calls[i % len(calls)]
            i += 1
            try:
                replies.append((label, fn()))
            except Exception as exc:  # noqa: BLE001 - a raised error is recorded, then checked for decoy bytes
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
            if i % 40 == 0:  # the tools' bounded grep (a child per call): fewer, but still racing
                replies.append(
                    (
                        "bounded grep",
                        asyncio.run(
                            writer._library_tool(
                                "grep",
                                {"pattern": "needle", "path": "/memory"},
                                roots,
                            ),
                        ).text,
                    ),
                )
                replies.append(
                    (
                        "helper grep",
                        str(
                            asyncio.run(
                                helpers._tool(
                                    ctx,
                                    "grep",
                                    {"pattern": "needle", "path": "/work"},
                                    st,
                                    resolve,
                                ),
                            )[0],
                        ),
                    ),
                )
                bounded["n"] += 1

    swapper = _Swapper(box, decoy, secret)
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # interleave the two threads as finely as the interpreter allows
    try:
        swapper.start()
        driver = threading.Thread(target=drive, daemon=True)
        driver.start()
        driver.join(RACE_S + 30)
        hung = driver.is_alive()
    finally:
        swapper.done.set()
        swapper.join(10)
        sys.setswitchinterval(old)
    assert not hung, "a call did not return"
    assert swapper.swaps > 50 and len(replies) > 200 and bounded["n"] > 2
    leaked = [(label, text[:200]) for label, text in replies if DECOY in text]
    assert leaked == []
    assert [e for e in errors if DECOY in e] == []
    assert _snapshot(decoy, secret) == outside


def test_a_fifo_leaf_is_refused_by_every_reader_and_never_blocks(tmp_path):
    box, _, _ = _tree(tmp_path)
    writer, roots = _writer_for(tmp_path, box)
    ctx, st, resolve = _helper_state(tmp_path, box)
    fifo = box / "memory" / "pipe"
    os.mkfifo(fifo)
    got = asyncio.run(writer._library_tool("read", {"path": "/memory/memory/pipe"}, roots))
    assert got.form == "unreadable" and "not a regular file" in got.text
    out = asyncio.run(helpers._tool(ctx, "read", {"path": "/work/pipe"}, st, resolve))[0]
    assert tool_spec.form_of(out) == "unreadable"
    got = asyncio.run(
        writer._library_tool("grep", {"pattern": "x", "path": "/memory/memory"}, roots),
    )
    assert got.status == "ok"
    assert views.grep("x", "/memory/memory/pipe", roots) == "(no hits)"
    fixture_fifo = box / "memory" / "pkg" / "tests" / "data" / "f.jsonl"
    os.mkfifo(fixture_fifo)
    got = wt.fixture(
        {
            "episode": "e1",
            "source": "request:0",
            "dest": "memory/pkg/tests/data/f.jsonl",
            "append": True,
        },
        writer._episode,
        box,
        writer.gate._blob,
        ["e1"],
    )
    assert got.status == "refused"
    os.unlink(box / ".pass" / "manifest.json")
    os.mkfifo(box / ".pass" / "manifest.json")
    found, manifest, problem = writer_pass._read_manifest(box)
    assert found and manifest is None and "not a regular file" in problem
    assert wt._read_manifest(box)[0] is None
    with pytest.raises(OSError):
        wt.write_manifest(box, {"items": []})


def test_decide_never_follows_a_link_at_the_manifest(tmp_path):
    box, _, secret = _tree(tmp_path)
    secret.write_text("{}")
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    (box / ".pass" / "manifest.json").unlink()
    os.symlink(secret, box / ".pass" / "manifest.json")
    args = {
        "op": "decide",
        "id": "new",
        "decision": "defer",
        "reason": "later",
        "episodes": ["e1"],
    }
    got = wt.pool(args, pool_dir, box)
    assert got.form == "manifest_unreadable"
    assert secret.read_text() == "{}"
    # a link swapped in after the read: the write refuses it rather than write through it
    with pytest.raises(safe_paths.LinkFound):
        wt.write_manifest(box, {"items": []})
    assert secret.read_text() == "{}"
    # a link at .pass itself
    os.unlink(box / ".pass" / "manifest.json")
    os.rmdir(box / ".pass")
    os.symlink(tmp_path, box / ".pass")
    assert wt.pool(args, pool_dir, box).form == "manifest_unreadable"
    with pytest.raises(OSError):
        wt.write_manifest(box, {"items": []})
    assert not (tmp_path / "manifest.json").exists()


def test_a_grep_child_failure_is_a_fixed_text_with_no_host_path(tmp_path, monkeypatch):
    box, _, _ = _tree(tmp_path)
    writer, roots = _writer_for(tmp_path, box)
    ctx, st, resolve = _helper_state(tmp_path, box)
    host = str(tmp_path)

    def crash(argv, **kw):
        return subprocess.CompletedProcess(
            argv,
            1,
            b"",
            f'Traceback\n  File "{host}/views.py", line 1\nMemoryError: {host}\n'.encode(),
        )

    monkeypatch.setattr(views.subprocess, "run", crash)
    with pytest.raises(views.SearchFailed) as raised:
        views.grep_bounded("needle", "/memory", roots)
    assert host not in str(raised.value)
    got = asyncio.run(
        writer._library_tool("grep", {"pattern": "needle", "path": "/memory"}, roots),
    )
    assert got.form == "search_failed" and got.status == "refused"
    assert host not in got.text and got.text == "refused: " + wt.SEARCH_FAILED
    out = asyncio.run(helpers._tool(ctx, "grep", {"pattern": "needle", "path": "/work"}, st, resolve))[0]
    assert tool_spec.form_of(out) == "search_failed" and host not in out
    assert "search_failed" in tool_spec.refusal_codes()["grep"]


def test_the_box_paths_are_parsed_before_anything_is_opened(tmp_path):
    box, _, secret = _tree(tmp_path)
    roots = {"/memory": box}
    for bad in (
        "/memory/../credentials.txt",
        "/memory/memory/../../credentials.txt",
        "/etc/passwd",
        "/memoryx/a",
        "/memory/a\x00b",
        "/memory/" + "a/" * 3000,
    ):
        with pytest.raises(safe_paths.Outside):
            views.read(bad, roots)
    os.symlink(secret, box / "memory" / "cred")
    with pytest.raises(safe_paths.LinkFound):
        views.read("/memory/memory/cred", roots)
    os.symlink(box / "memory" / "pkg", box / "memory" / "inside")  # a link inside the roots is not followed either
    with pytest.raises(safe_paths.LinkFound):
        views.read("/memory/memory/inside/mod.py", roots)
    assert views.read_entry("/memory//memory/./pkg", roots)[1] == "list"


# --- the helper's episode reader is the writer's ----------------------------------------------------------------


def _read_episode(tmp_path, args, st=None):
    ctx = _ctx(tmp_path)
    if st is None:
        st = {}
    if "hin" not in st:
        for name in ("task", "work"):
            (tmp_path / name).mkdir()
        st.update(
            hin=ctx.inputs,
            task=tmp_path / "task",
            lib=ctx.library,
            work=tmp_path / "work",
            call="call-0",
        )
    return asyncio.run(helpers._tool(ctx, "read_episode", args, st, helpers._resolver(ctx)))


def test_the_helpers_read_episode_shows_at_most_eight_parts_per_call(tmp_path, monkeypatch):
    many = _episode("e1", request=tuple(f"message {i}" for i in range(12)))
    monkeypatch.setitem(HELPER_EPS, "e1", many)
    parts = ["request", *(f"observation:{i}" for i in range(10))]
    out, paths = _read_episode(tmp_path, {"episode": "e1", "parts": parts})
    assert out.count("== ") == wt.READ_PARTS_PER_CALL
    assert f"3 more part(s) not read in this call ({wt.READ_PARTS_PER_CALL} per call)" in out
    assert 'next: parts=["observation:7", "observation:8", "observation:9"]' in out
    assert len(paths) == wt.READ_PARTS_PER_CALL


def test_the_helpers_read_episode_points_to_an_identical_part_and_lists_fixture_sources(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setitem(HELPER_EPS, "e1", EPS["e1"])  # request ("req", "same", "same")
    st: dict = {}
    first, _ = _read_episode(tmp_path / "a", {"episode": "e1", "part": "observation:0"}, st)
    assert '"same"' in first
    second, _ = _read_episode(tmp_path / "b", {"episode": "e1", "part": "observation:1"}, st)
    assert "identical to e1 observation:0" in second
    toc, _ = _read_episode(tmp_path / "c", {"episode": "e1"}, st)
    assert "fixture source: request:1 (" in toc and "per call" in toc
    bad, _ = _read_episode(tmp_path / "d", {"episode": "e1", "parts": 3}, st)
    assert tool_spec.form_of(bad) == "bad_parts"


# --- the quota measures restore access without following a link -------------------------------------------------


def _decoy_at_mode_zero(tmp_path: Path) -> Path:
    decoy = tmp_path / "decoy-dir"
    (decoy / "inner").mkdir(parents=True)
    os.chmod(decoy / "inner", 0)
    os.chmod(decoy, 0)
    return decoy


def _modes(decoy: Path) -> tuple[int, int]:
    outer = os.lstat(decoy).st_mode
    os.chmod(decoy, 0o700)
    try:
        return outer, os.lstat(decoy / "inner").st_mode
    finally:
        os.chmod(decoy, 0)


@pytest.mark.parametrize("measure", ["writer", "helper"])
def test_a_directory_swapped_for_a_link_while_measured_never_changes_a_mode_outside(
    tmp_path,
    monkeypatch,
    measure,
):
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f.py").write_text("x = 1\n")
    os.chmod(root / "a" / "b", 0)  # the box removed its owner's access: the measure restores it
    os.chmod(root / "a", 0)
    decoy = _decoy_at_mode_zero(tmp_path)
    before = _modes(decoy)
    real = os.scandir
    swapped = []

    def swap_after_listing(target):
        # the root is listed, then box code swaps a/ for a link to the decoy before a/ is opened
        it = real(target)
        if not swapped:
            swapped.append(True)
            os.rename(root / "a", root / ".a-away")  # within its directory: a has mode 0
            os.symlink(decoy, root / "a")
        return it

    monkeypatch.setattr(safe_paths.os, "scandir", swap_after_listing)
    run = writer_pass._measure if measure == "writer" else helpers.work_over
    run(root)
    monkeypatch.setattr(safe_paths.os, "scandir", real)
    assert swapped and _modes(decoy) == before
    # without the swap the same tree is unlocked and measured in full
    os.unlink(root / "a")
    os.rename(root / ".a-away", root / "a")
    assert run(root) is None
    assert stat.S_IMODE(os.lstat(root / "a" / "b").st_mode) & 0o700 == 0o700


def test_a_tight_swap_loop_never_changes_a_mode_outside_the_measured_tree(tmp_path):
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f.py").write_text("x = 1\n")
    decoy = _decoy_at_mode_zero(tmp_path)
    before = _modes(decoy)
    done = threading.Event()

    def swap():
        while not done.is_set():
            try:
                os.rename(root / "a", tmp_path / "a-away")
                try:
                    os.symlink(decoy, root / "a")
                    os.unlink(root / "a")
                finally:
                    os.rename(tmp_path / "a-away", root / "a")
            except OSError:
                continue

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    t = threading.Thread(target=swap, daemon=True)
    t.start()
    try:
        end = time.monotonic() + 3.0
        n = 0
        while time.monotonic() < end:
            writer_pass._measure(root)
            helpers.work_over(root)
            n += 1
    finally:
        done.set()
        t.join(10)
        sys.setswitchinterval(old)
    assert n > 20 and _modes(decoy) == before


def test_the_measures_fail_closed_without_proc(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    real = os.path.isdir
    monkeypatch.setattr(
        safe_paths.os.path,
        "isdir",
        lambda p: False if str(p) == "/proc/self/fd" else real(p),
    )
    assert writer_pass._measure(tmp_path) == writer_pass.NO_PROC_BREACH
    assert "cannot be measured safely" in helpers.work_over(tmp_path)
