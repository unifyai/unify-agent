"""The worker audit hook (spec §C2, §C5 option A): raw records only.

Audit hooks are permanent, so every hook test runs in a fresh subprocess
interpreter (``-I``) that loads ``audit.py`` by file path, which also checks
that the module needs only the standard library.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import unify.memory_v2.integration.adapters.audit as audit_mod
from unify.memory_v2.integration.adapters.audit import (
    CLIP_MARK,
    MAX_ARGV,
    MAX_BYTES,
    MAX_RECORDS,
    MAX_STR,
)

AUDIT_PY = Path(audit_mod.__file__).resolve()
FAKE_KEY = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret
ENV_SECRET = "env-only-value-9f8e7d6c"  # pragma: allowlist secret

PRELUDE = """
import importlib.util, json, os, shutil, subprocess, sys, threading
def load():
    spec = importlib.util.spec_from_file_location("cell_audit", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
mod = load()
assert not any(m == "unify" or m.startswith("unify.") for m in sys.modules)
ws, outside = sys.argv[2], sys.argv[3]
audit = mod.install([ws])
def emit(value):
    print(json.dumps(value))
"""


def run_child(body: str, ws: Path, outside: Path, cwd: Path | None = None) -> dict:
    script = PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-I", "-c", script, str(AUDIT_PY), str(ws), str(outside)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(cwd or ws),
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def make_tree(tmp_path: Path) -> tuple[Path, Path]:
    ws = tmp_path / "ws"
    (ws / "data" / "sub").mkdir(parents=True)
    (ws / "data" / "a.csv").write_text("x,y\n1,2\n")
    (ws / "data" / "sub" / "b.csv").write_text("z\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not ours\n")
    return ws, outside


def strip(records: list, ws: Path) -> list:
    """(event, path relative to ws, mode) for the path records."""
    out = []
    for r in records:
        if "path" not in r:
            continue
        rel = os.path.relpath(r["path"], ws) if r["path"] else None
        out.append((r["event"], rel, r.get("mode")))
    return out


CELL = """
audit.begin()
with open(os.path.join(ws, "data", "a.csv")) as f:
    f.read()
with open(os.path.join(ws, "out.txt"), "w") as f:
    f.write("CONTENT-MUST-NOT-APPEAR")
os.listdir(ws)
with os.scandir(os.path.join(ws, "data")):
    pass
with open(os.path.join(outside, "secret.txt")) as f:
    f.read()
os.listdir(outside)
import json as _json, decimal  # stdlib opens are outside the tree
subprocess.run(["echo", "x"], stdout=subprocess.DEVNULL,
               env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "MY_TOKEN": "%s"})
os.system("true %s")
audit.end()
open(os.path.join(ws, "data", "a.csv")).close()  # after the cell: not recorded
emit(audit.drain())
""" % (
    ENV_SECRET,
    FAKE_KEY,
)


def test_cell_records_opens_lists_and_spawns_only_inside_the_tree(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(CELL, ws, outside)
    raw = json.dumps(drained)
    assert "CONTENT-MUST-NOT-APPEAR" not in raw
    assert ENV_SECRET not in raw and "MY_TOKEN" not in raw
    assert FAKE_KEY not in raw
    assert str(outside) not in raw
    assert drained["dropped"] == 0 and drained["failed"] == 0
    assert drained["bytes"] > 0
    recs = drained["records"]
    assert strip(recs, ws) == [
        ("open", "data/a.csv", "r"),
        ("open", "out.txt", "w"),
        ("os.listdir", ".", None),
        ("os.scandir", "data", None),
    ]
    spawns = [(r["event"], r["exe"], r["argv"]) for r in recs if "argv" in r]
    assert spawns == [
        ("subprocess.Popen", "echo", ["echo", "x"]),
        ("os.system", None, ["true <redacted:key-shaped>"]),
    ]
    assert len(recs) == 6
    assert all(r["cell_thread"] is True and isinstance(r["tid"], int) for r in recs)


def test_dir_fd_events_and_rmtree_land_at_their_real_paths(tmp_path):
    ws, outside = make_tree(tmp_path)
    (ws / "d2").mkdir()
    (ws / "d2" / "z.txt").write_text("z")
    (outside / "o.txt").write_text("o")
    # cwd is the tree, so a name resolved against the cwd would look in-tree.
    drained = run_child(
        """
        audit.begin()
        fd = os.open(os.path.join(outside), os.O_RDONLY | os.O_DIRECTORY)
        os.remove("o.txt", dir_fd=fd)  # outside: dropped
        os.close(fd)
        fd = os.open(os.path.join(ws, "d2"), os.O_RDONLY | os.O_DIRECTORY)
        os.remove("z.txt", dir_fd=fd)
        os.mkdir("m", dir_fd=fd)
        os.rename("m", "n", src_dir_fd=fd, dst_dir_fd=fd)
        os.close(fd)
        shutil.rmtree(os.path.join(ws, "data"))
        emit(audit.drain())
        """,
        ws,
        outside,
    )
    got = strip(drained["records"], ws)
    assert ("os.remove", "o.txt", None) not in got
    assert got[:3] == [
        ("os.remove", "d2/z.txt", None),
        ("os.mkdir", "d2/m", None),
        ("os.rename", "d2/m", None),
    ]
    assert drained["records"][2]["dst"] == str(ws / "d2" / "n")
    rest = got[3:]
    assert rest[0] == ("shutil.rmtree", "data", None)
    assert ("os.remove", "data/a.csv", None) in rest
    assert ("os.remove", "data/sub/b.csv", None) in rest
    assert ("os.rmdir", "data/sub", None) in rest
    assert ("os.rmdir", "data", None) in rest
    # Nothing landed at the tree root by mistake, and no directory handles.
    assert not [g for g in got if g[1] in ("a.csv", "b.csv", "sub", "z.txt")]
    assert not [g for g in got if g[0] == "open"]


def test_os_open_modes_and_relative_open_is_marked(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        os.close(os.open(os.path.join(ws, "data", "a.csv"), os.O_RDONLY))
        os.close(os.open(os.path.join(ws, "new.bin"), os.O_WRONLY | os.O_CREAT))
        os.close(os.open("rel.txt", os.O_RDWR | os.O_CREAT))
        emit(audit.drain())
        """,
        ws,
        outside,
    )
    recs = drained["records"]
    assert strip(recs, ws) == [
        ("open", "data/a.csv", "r"),
        ("open", "new.bin", "w"),
        ("open", "rel.txt", "w"),
    ]
    assert [r.get("cwd_assumed", False) for r in recs] == [False, False, True]


def test_scandir_without_argument_is_the_cwd(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        with os.scandir():
            pass
        os.listdir()
        emit(audit.drain())
        """,
        ws,
        outside,
        cwd=ws / "data",
    )
    assert strip(drained["records"], ws) == [
        ("os.scandir", "data", None),
        ("os.listdir", "data", None),
    ]


def test_rename_across_the_root_keeps_only_the_inside_side(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        os.rename(os.path.join(ws, "data", "a.csv"), os.path.join(outside, "a.csv"))
        os.rename(os.path.join(outside, "secret.txt"), os.path.join(ws, "s.txt"))
        emit(audit.drain())
        """,
        ws,
        outside,
    )
    recs = drained["records"]
    assert [(r["path"], r["dst"]) for r in recs] == [
        (str(ws / "data" / "a.csv"), None),
        (None, str(ws / "s.txt")),
    ]
    assert str(outside) not in json.dumps(drained)


def test_keys_are_masked_before_clipping_and_argv_is_capped(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        straddle = "a" * (%d - 20) + "%s"
        sys.audit("subprocess.Popen", "tool", ["tool", straddle] + ["x"] * 100, None, None)
        emit(audit.drain())
        """ % (MAX_STR, FAKE_KEY),
        ws,
        outside,
    )
    (rec,) = drained["records"]
    raw = json.dumps(rec)
    assert "sk-or" not in raw and "cdcd" not in raw
    assert rec["clipped"] is True and rec["argv_truncated"] is True
    assert len(rec["argv"]) == MAX_ARGV
    assert rec["argv"][1].endswith(CLIP_MARK)
    assert len(rec["argv"][1]) <= MAX_STR + len(CLIP_MARK)


def test_spawn_seen_through_a_lower_layer_is_recorded_once(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        sys.audit("subprocess.Popen", "true", ["true"], None, None)
        sys.audit("os.posix_spawn", "/bin/true", ["true"], {"K": "%s"})
        pid = os.posix_spawn("/bin/true", ["true", "y"], {"K": "%s"})
        os.waitpid(pid, 0)
        emit(audit.drain())
        """ % (ENV_SECRET, ENV_SECRET),
        ws,
        outside,
    )
    recs = drained["records"]
    assert [(r["event"], r["argv"]) for r in recs] == [
        ("subprocess.Popen", ["true"]),
        ("os.posix_spawn", ["true", "y"]),
    ]
    assert ENV_SECRET not in json.dumps(drained)


def test_hook_failure_never_reaches_the_cell(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        audit.roots = None  # breaks the hook's own path check
        open(os.path.join(ws, "data", "a.csv")).read()
        os.listdir(ws)
        failed_roots = audit.drain()
        class Hostile:
            def __bool__(self):
                raise RuntimeError("truth test")
        audit.roots = audit.prefixes([ws])
        audit.begin()
        audit.on = Hostile()  # never truth-tested: treated as off
        os.listdir(ws)
        hostile = audit.drain()
        audit.begin()
        audit._local = None  # tampered internals
        os.listdir(ws)
        tampered = audit.drain()
        emit({"roots": failed_roots, "hostile": hostile, "tampered": tampered})
        """,
        ws,
        outside,
    )
    assert drained["roots"]["records"] == [] and drained["roots"]["failed"] == 2
    assert drained["hostile"]["records"] == [] and drained["hostile"]["failed"] == 0
    assert drained["tampered"]["records"] == [] and drained["tampered"]["failed"] == 1


def test_near_the_recursion_limit_the_hook_returns_at_once(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        sys.setrecursionlimit(300)
        audit.begin()
        errors = []
        def deep(n):
            if n == 0:
                try:
                    os.listdir(ws)
                except RecursionError as exc:
                    errors.append(repr(exc))
                return
            deep(n - 1)
        deep(300 - 30)   # within the margin: skipped, no error
        deep(100)        # far from the limit: recorded
        out = audit.drain()
        out["errors"] = errors
        emit(out)
        """,
        ws,
        outside,
    )
    assert drained["errors"] == []
    assert len(drained["records"]) == 1 and drained["failed"] == 0


def test_count_and_byte_budgets_are_enforced(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        for _ in range(%d):
            os.listdir(ws)
        by_count = audit.drain()
        audit.begin()
        big = ["b" * 4000] * 64
        for _ in range(20):
            sys.audit("subprocess.Popen", "tool", ["tool%%d" %% _] + big, None, None)
        by_bytes = audit.drain()
        emit({"count": [len(by_count["records"]), by_count["dropped"]],
              "bytes": [len(by_bytes["records"]), by_bytes["dropped"], by_bytes["bytes"],
                        len(json.dumps(by_bytes))]})
        """ % (MAX_RECORDS + 25),
        ws,
        outside,
    )
    assert drained["count"] == [MAX_RECORDS, 25]
    kept, dropped, used, serialised = drained["bytes"]
    assert kept + dropped == 20 and dropped > 0 and kept > 0
    assert used <= MAX_BYTES and serialised < 2 * MAX_BYTES


def test_install_is_once_per_process_across_double_loads(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        again = load().install([ws])  # a second module object, same hook
        assert again is audit and mod.install([ws]) is audit
        audit.begin(); os.listdir(ws); first = audit.drain()
        os.listdir(ws)  # between cells: off
        audit.begin(); os.mkdir(os.path.join(ws, "new")); second = audit.drain()
        emit({"first": first, "second": second})
        """,
        ws,
        outside,
    )
    assert len(drained["first"]["records"]) == 1  # one hook, not two
    assert [r["event"] for r in drained["second"]["records"]] == ["os.mkdir"]


def test_other_threads_are_marked(tmp_path):
    ws, outside = make_tree(tmp_path)
    drained = run_child(
        """
        audit.begin()
        t = threading.Thread(target=lambda: os.listdir(os.path.join(ws, "data")))
        t.start(); t.join()
        os.listdir(ws)
        emit(audit.drain())
        """,
        ws,
        outside,
    )
    recs = drained["records"]
    assert [(os.path.relpath(r["path"], ws), r["cell_thread"]) for r in recs] == [
        ("data", False),
        (".", True),
    ]
    assert recs[0]["tid"] != recs[1]["tid"]
