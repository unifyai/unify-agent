"""The worker audit hook under memory v2 (integration Task 28, spec §3a).

The harness sends ``audit`` in the worker's init message only while a memory-v2 request captures the
work tree; the child then installs the approved audit adapter (loaded by file path, no ``unify`` import)
and returns each cell's drained records in its ``done`` message; ``hooks.worker_cell_done`` stamps them on
the harness clock. Off (or no request, or no capture) means no key, no hook, nothing kept.

The child-half tests drive ``worker_child.Worker`` in a fresh ``python -I -S`` process (an audit hook is
process-wide and cannot be removed); the last tests run cells in the real sandboxed worker.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution import worker_child
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration import worktree_capture as capture_mod
from unify.memory_v2.integration.adapters import audit as audit_mod
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.worktree_capture import WorktreeCapture
from unify.memory_v2.redact import Redactor
from unify.settings import SETTINGS

ROW_CHANNEL = "worktree:workspace"


def _capture(tmp_path, monkeypatch, *, switch="on"):
    """A begun capture over ``tmp_path/ws``, the switch at *switch* and a run current."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", switch)
    home = tmp_path / "home"
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.csv").write_text("x,y\n1,2\n")
    paths = Paths.under(home)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    cap = WorktreeCapture(paths, ws, lambda: Redactor({}), hidden=lambda rel: False)
    cap.begin()
    return cap, ws


def _init_message(monkeypatch) -> dict:
    from unify.actor.execution.worker import PythonWorker

    return PythonWorker()._init_message()


# -- harness side: the init message and the stamping ----------------------------------------------------
def test_off_means_no_audit_and_no_init_key(tmp_path, monkeypatch):
    cap, ws = _capture(tmp_path, monkeypatch, switch="")
    assert hooks.worker_audit() is None
    assert "audit" not in _init_message(monkeypatch)
    hooks.worker_cell_done(
        {"records": [{"event": "open", "path": str(ws / "a.csv"), "mode": "r"}]},
    )
    assert cap._events == []


def test_on_without_a_capture_or_a_run_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=Paths.under(tmp_path)),
    )
    assert hooks.worker_audit() is None  # a run, but no capture
    assert "audit" not in _init_message(monkeypatch)
    cap, _ = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    assert hooks.worker_audit() is None  # a capture, but no run


def test_on_with_a_capture_sends_the_workspace_root_and_the_hook_file(
    tmp_path,
    monkeypatch,
):
    cap, ws = _capture(tmp_path, monkeypatch)
    want = {"roots": [str(ws.resolve())], "path": str(Path(audit_mod.__file__))}
    assert hooks.worker_audit() == want
    assert _init_message(monkeypatch)["audit"] == want
    cap.finish([])
    assert hooks.worker_audit() is None  # the request ended


def test_worker_cell_done_stamps_records_on_the_harness_clock(tmp_path, monkeypatch):
    cap, ws = _capture(tmp_path, monkeypatch)
    monkeypatch.setattr(hooks.time, "time", lambda: 1234.5)
    read = {
        "event": "open",
        "path": str(ws / "a.csv"),
        "mode": "r",
        "tid": 1,
        "cell_thread": True,
    }
    spawn = {
        "event": "subprocess.Popen",
        "exe": "true",
        "argv": ["true"],
        "tid": 1,
        "cell_thread": True,
    }
    hooks.worker_cell_done(
        {"records": [read, spawn], "dropped": 2, "failed": 1, "bytes": 9},
    )
    hooks.worker_cell_done(None)  # a cell with no audit (the hook not installed)
    hooks.worker_cell_done("garbage")  # never raises
    assert cap._events == [
        (1234.5, [read]),
    ]  # process starts are not kept (no shell rows here)
    assert (cap.dropped, cap.failed) == (2, 1)


# -- child side: Worker.init / run_cell in a fresh interpreter -------------------------------------------
_CHILD_PROBE = r"""
import asyncio, builtins, json, os, runpy, sys, tempfile
child = runpy.run_path(sys.argv[1], run_name="worker_child_probe")
spec = json.loads(sys.argv[2])
class Sink:
    def __init__(self):
        self.msgs = []
    def write(self, data):
        self.msgs += [json.loads(line) for line in data.decode().splitlines()]
    def flush(self):
        pass
sink = Sink()
out = tempfile.TemporaryFile()
w = child["Worker"](None, sink, out.fileno())
init = {"op": "init", "sys_path": [p for p in sys.path if p], "builtins": dir(builtins), "globals": {}}
init.update(spec["init"])
missing = w.init(init)
loop = asyncio.new_event_loop()
w.loop = loop
loop.run_until_complete(w.run_cell({"op": "exec", "id": 1, "source": spec["source"]}))
done = [m for m in sink.msgs if m.get("op") == "done"][0]
print(json.dumps({
    "missing": missing,
    "state": getattr(w, "audit_state", None),
    "installed": "_unify_memory_v2_cell_audit" in sys.modules,
    "done": done,
}))
"""

_CELL = (
    "async def __exec_wrapper():\n"
    "    import os, subprocess, json\n"
    "    open({root!r} + '/a.csv').read()\n"
    "    os.listdir({root!r})\n"
    "    subprocess.run(['true'])\n"
    "    return 1\n"
)


def _run_child(init: dict, root: Path) -> dict:
    spec = {"init": init, "source": _CELL.format(root=str(root))}
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _CHILD_PROBE,
            worker_child.__file__,
            json.dumps(spec),
        ],
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_child_without_the_audit_key_installs_nothing(tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    got = _run_child({}, tmp_path)
    assert got["installed"] is False and got["state"] is None
    assert "audit" not in got["done"] and got["done"]["error"] is None


def test_child_with_the_audit_key_returns_the_cells_records(tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    root = str(tmp_path.resolve())
    got = _run_child({"audit": {"roots": [root], "path": audit_mod.__file__}}, tmp_path)
    assert got["installed"] is True and got["state"] == "on"
    assert got["done"]["error"] is None
    audit = got["done"]["audit"]
    assert set(audit) == {"records", "dropped", "failed", "bytes"}
    seen = [
        (r["event"], r.get("path"), r.get("mode"), r.get("argv"))
        for r in audit["records"]
    ]
    # the import's opens are outside the roots; the start is recorded once
    assert seen == [
        ("open", root + "/a.csv", "r", None),
        ("os.listdir", root, None, None),
        ("subprocess.Popen", None, None, ["true"]),
    ]
    assert all(r["cell_thread"] is True for r in audit["records"])


def test_child_reports_a_hook_that_could_not_load(tmp_path):
    (tmp_path / "a.csv").write_text("x\n")
    got = _run_child(
        {"audit": {"roots": [str(tmp_path)], "path": str(tmp_path / "nope.py")}},
        tmp_path,
    )
    assert got["installed"] is False and got["state"].startswith("FileNotFoundError")
    assert (
        "audit" not in got["done"] and got["done"]["error"] is None
    )  # the cell is unaffected


# -- the real sandboxed worker (bubblewrap; remote run) -------------------------------------------------
async def _cell(ex, code: str):
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0),
        timeout=60,
    )
    assert res["error"] is None, res["error"]
    return res["result"]


_PROBE_HOOK = "import sys\n'_unify_memory_v2_cell_audit' in sys.modules"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_real_worker_off_installs_no_hook(world, monkeypatch):  # noqa: F811
    from unify.actor.execution.session import SessionExecutor

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    ex = SessionExecutor(environments={})
    try:
        assert await _cell(ex, _PROBE_HOOK) is False
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_real_worker_audits_a_cell_into_worktree_rows(
    world,
    monkeypatch,
):  # noqa: F811
    from unify.actor.execution.session import SessionExecutor

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(world["state"])
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    ws = world["workspace"]
    (ws / "pay.csv").write_text("id\tname\n1\tann\n")
    cap = WorktreeCapture(
        paths,
        ws,
        lambda: Redactor.from_environ({}),
    )  # the sandbox policy hides
    cap.begin()
    assert capture_mod.active() is cap
    ex = SessionExecutor(environments={})
    try:
        assert await _cell(ex, _PROBE_HOOK) is True
        await _cell(
            ex,
            "import csv, json, os, subprocess\n"
            "rows = list(csv.reader(open('pay.csv'), delimiter='\\t'))\n"
            "json.dump({'n': len(rows)}, open('out.json', 'w'))\n"
            "subprocess.run(['true'])\n"
            "try:\n"
            "    open('.env', 'w').write('X=1')\n"  # secret-named: hidden from cells, never recorded
            "except OSError:\n"
            "    pass\n"
            "os.listdir('.')\n"
            "len(rows)",
        )
    finally:
        await ex.close()
    result = cap.finish([SimpleNamespace(start=0.0, end=float("inf"))])
    got = sorted((a.method, a.args[0], a.cell) for a in result.actions)
    assert ("read", "pay.csv", 0) in got and ("write", "out.json", 0) in got
    assert ("list", ".", 0) in got
    assert not any(".env" in a.args[0] for a in result.actions)
    assert all(
        a.kind == "worktree" and a.channel == ROW_CHANNEL for a in result.actions
    )
    read = [a for a in result.actions if a.method == "read" and a.args[0] == "pay.csv"][
        0
    ]
    assert read.response["shape"]["delimiter"] == "\t"
    assert result.before and result.after and "b/out.json" in result.diff
