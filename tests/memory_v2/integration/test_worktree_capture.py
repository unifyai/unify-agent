"""Work-tree capture for a memory-v2 request (integration Task 30).

``WorktreeCapture`` snapshots the workspace into ``<UNIFY_HOME>/worktree.git`` at begin and finish and turns
the worker's audit records (here produced by the real audit adapter, run in a fresh interpreter over a
scripted cell: the producer-to-consumer round trip) into the approved work-tree adapter's rows, with the
request's redactor. No pytest-only machinery beyond ``tmp_path`` and ``monkeypatch``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.actor.code_act.sandbox_world import world  # noqa: F401
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import env_channel
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration import worktree_capture as capture_mod
from unify.memory_v2.integration.adapters import audit as audit_mod
from unify.memory_v2.integration.adapters.worktree import (
    CHANNEL,
    SNAPSHOT_CAP,
    _git_raw,
    tree_files,
)
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.worktree_capture import (
    WorktreeCapture,
    WorktreeResult,
    cell_at,
)
from unify.memory_v2.redact import Redactor
from unify.settings import SETTINGS

SECRET = "hunter2-secret-value"  # pragma: allowlist secret

# Runs *cell* (Python source, ``ROOT`` bound) under the audit adapter loaded by file path, as the worker
# child does, and prints the drained records.
_AUDITED_CELL = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("_audit", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ROOT = sys.argv[2]
hook = mod.install([ROOT])
code = compile(sys.argv[3], "<cell>", "exec")
hook.begin()
try:
    exec(code, {"ROOT": ROOT})
finally:
    hook.end()
print(json.dumps(hook.drain()))
"""


def _audited(root: Path, cell: str) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _AUDITED_CELL,
            audit_mod.__file__,
            str(root),
            cell,
        ],
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _world(tmp_path, monkeypatch, *, factory=None, hidden=None):
    """A request home and workspace, the switch on and a run current; returns (paths, ws, capture)."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    paths = Paths.under(tmp_path / "home")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pay.csv").write_text(
        "id\tname\tamount\n1\tann\t3.5\n2\tbob\t4\n",
    )
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    cap = WorktreeCapture(
        paths,
        ws,
        factory or (lambda: Redactor({})),
        hidden=hidden or (lambda rel: False),
    )
    return paths, ws, cap


def _done(monkeypatch, stamp: float, drained: dict) -> None:
    monkeypatch.setattr(hooks.time, "time", lambda: stamp)
    hooks.worker_cell_done(drained)


def _cells(*windows):
    return [SimpleNamespace(start=a, end=b) for a, b in windows]


def test_a_scripted_cell_reading_a_csv_and_writing_json_gives_two_rows_with_shapes(
    tmp_path,
    monkeypatch,
):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    cap.begin()
    assert capture_mod.active() is cap
    drained = _audited(
        ws.resolve(),
        "import csv, json\n"
        "rows = list(csv.reader(open(ROOT + '/pay.csv'), delimiter='\\t'))\n"
        "json.dump({'total': len(rows), 'names': [r[1] for r in rows[1:]]}, open(ROOT + '/out.json', 'w'))\n",
    )
    assert [r["event"] for r in drained["records"]] == ["open", "open"]
    _done(monkeypatch, 10.0, drained)
    result = cap.finish(_cells((1.0, 2.0), (5.0, 15.0)))
    assert isinstance(result, WorktreeResult)
    assert capture_mod.active() is None
    assert len(result.actions) == 2
    read, write = sorted(result.actions, key=lambda a: a.method)
    assert all(
        a.kind == "worktree" and a.channel == "worktree:workspace" and a.cell == 1
        for a in result.actions
    )
    assert env_channel("worktree", read.channel) == "worktree_workspace"
    assert (read.method, read.args, read.status, read.effect) == (
        "read",
        ["pay.csv"],
        "ok",
        "read",
    )
    shape = read.response["shape"]
    assert (
        shape["format"] == "tsv" and shape["delimiter"] == "\t"
    )  # tab-delimited, whatever the suffix
    assert shape["columns"] == ["id", "name", "amount"] and shape["types"] == [
        "int",
        "str",
        "float",
    ]
    assert "ann" not in json.dumps(shape)
    assert (write.method, write.args, write.status, write.effect) == (
        "write",
        ["out.json"],
        "ok",
        "write",
    )
    assert write.response["blob_before"] is None and write.response["blob_after"]
    assert write.response["shape"]["format"] == "json"
    assert set(write.response["shape"]["keys"]) == {"total", "names"}
    blobs = BlobStore(paths.blobs)
    assert blobs.get(write.response["blob_after"]) == (ws / "out.json").read_bytes()
    assert result.before and result.after and result.before != result.after
    assert "+++ b/out.json" in result.diff and "pay.csv" not in result.diff
    repo = Repo(paths.worktree_git)
    assert set(tree_files(repo, result.after)) == {"pay.csv", "out.json"}
    assert not (ws / ".git").exists()


def test_channel_constant_is_kind_qualified():
    assert CHANNEL == "worktree:workspace"
    assert env_channel("worktree", CHANNEL) == "worktree_workspace"


def test_records_are_credited_to_the_cell_whose_window_holds_the_stamp(
    tmp_path,
    monkeypatch,
):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    cap.begin()
    read = {"event": "open", "path": str(ws.resolve() / "pay.csv"), "mode": "r"}
    listing = {"event": "os.listdir", "path": str(ws.resolve())}
    _done(monkeypatch, 3.0, {"records": [read]})
    _done(monkeypatch, 99.0, {"records": [listing]})  # in no cell's window
    result = cap.finish(_cells((1.0, 2.0), (2.5, 4.0)))
    assert sorted((a.method, a.cell) for a in result.actions) == [
        ("list", -1),
        ("read", 1),
    ]
    assert (
        cell_at(1.0, _cells((1.0, 2.0))) == 0 and cell_at(0.5, _cells((1.0, 2.0))) == -1
    )
    assert cell_at(1.0, [SimpleNamespace(start=None, end=None)]) == -1


def test_snapshot_skips_symlinks_and_big_files(tmp_path, monkeypatch):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    outside = tmp_path / "host-secret.txt"
    outside.write_text("HOSTSECRET-LINKED\n")
    (ws / "link.txt").symlink_to(outside)
    (ws / "linkdir").symlink_to(tmp_path)
    with open(ws / "big.bin", "wb") as fh:  # sparse: over the 16 MiB cap, never read
        fh.truncate(SNAPSHOT_CAP + 1)
    cap.begin()
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / "link.txt"), "mode": "r"},
                {"event": "os.listdir", "path": str(ws.resolve() / "linkdir")},
            ],
        },
    )
    result = cap.finish(_cells((0.0, 2.0)))
    repo = Repo(paths.worktree_git)
    for sha in (result.before, result.after):
        assert set(tree_files(repo, sha)) == {"pay.csv"}
    rows = {(a.method, a.args[0]): a for a in result.actions}
    assert rows[("read", "link.txt")].status == "unrecorded"  # refused, not followed
    assert rows[("list", "linkdir")].status == "unrecorded"
    assert not any(
        b"HOSTSECRET-LINKED" in p.read_bytes()
        for p in paths.blobs.rglob("*")
        if p.is_file()
    )
    assert "HOSTSECRET-LINKED" not in json.dumps([a.response for a in result.actions])


def test_a_gitattributes_filter_runs_nothing(tmp_path, monkeypatch):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    marker = tmp_path / "filter-ran"
    # the worst case: the snapshot repo itself configures the filter the work tree asks for
    repo = capture_mod.snapshot_repo(paths.worktree_git)
    repo.run("config", "filter.x.clean", f"touch {marker}; cat")
    repo.run("config", "filter.x.smudge", f"touch {marker}; cat")
    (ws / ".gitattributes").write_text("* filter=x diff=x\n")
    repo.run("config", "diff.x.textconv", f"touch {marker}; cat")
    cap.begin()
    (ws / "pay.csv").write_text("id\tname\tamount\n9\tzed\t1\n")
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / "pay.csv"), "mode": "w"},
            ],
        },
    )
    result = cap.finish(_cells((0.0, 2.0)))
    assert not marker.exists()
    assert [(a.method, a.args[0]) for a in result.actions] == [("write", "pay.csv")]
    assert "+9\tzed\t1" in result.diff


def test_paths_hidden_from_cells_never_reach_snapshots_rows_or_the_diff(
    tmp_path,
    monkeypatch,
):
    paths, ws, cap = _world(tmp_path, monkeypatch, hidden=lambda rel: rel == ".env")
    (ws / ".env").write_text("TOKEN=HIDDEN-ENV-1\n")
    cap.begin()
    (ws / ".env").write_text("TOKEN=HIDDEN-ENV-2\n")
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / ".env"), "mode": "r"},
            ],
        },
    )
    result = cap.finish(_cells((0.0, 2.0)))
    assert result.actions == [] and result.diff == ""
    repo = Repo(paths.worktree_git)
    for sha in (result.before, result.after):
        assert ".env" not in tree_files(repo, sha)


def test_the_requests_redactor_covers_rows_blobs_and_the_diff(tmp_path, monkeypatch):
    built = []

    def factory():
        built.append(1)
        return Redactor({"API_TOKEN": SECRET})

    paths, ws, cap = _world(tmp_path, monkeypatch, factory=factory)
    (ws / "conf.txt").write_text(f"token={SECRET}\n")
    cap.begin()
    assert built == []  # built at finish, when the request's secrets are known
    (ws / "conf.txt").write_text(f"token={SECRET}\nmore=1\n")
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / "conf.txt"), "mode": "r"},
                {"event": "open", "path": str(ws.resolve() / "conf.txt"), "mode": "a"},
            ],
        },
    )
    result = cap.finish(_cells((0.0, 2.0)))
    assert built == [1]
    assert SECRET not in result.diff and "<secret:API_TOKEN>" in result.diff
    assert SECRET not in json.dumps([a.response for a in result.actions])
    exported = b"".join(p.read_bytes() for p in paths.blobs.rglob("*") if p.is_file())
    assert SECRET.encode() not in exported and b"<secret:API_TOKEN>" in exported
    write = [a for a in result.actions if a.method == "write"][0]
    assert (
        write.response["redacted_after"] is True and write.response["oid_after"] is None
    )


def test_finish_never_raises(tmp_path, monkeypatch):
    def broken():
        raise RuntimeError(f"no redactor {SECRET}")

    paths, ws, cap = _world(tmp_path, monkeypatch, factory=broken)
    cap.begin()
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / "pay.csv"), "mode": "r"},
            ],
        },
    )
    result = cap.finish(_cells((0.0, 2.0)))
    assert (
        result.actions == []
        and result.after is None
        and result.before
        and result.diff == ""
    )
    assert capture_mod.active() is None
    # a capture that could not begin (its snapshot repo inside the work tree) records nothing
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    inside = WorktreeCapture(
        Paths.under(ws),
        ws,
        lambda: Redactor({}),
        hidden=lambda rel: False,
    )
    inside.begin()
    assert capture_mod.active() is None and hooks.worker_audit() is None
    assert not (ws / "worktree.git").exists() and not (ws / "episodes-blobs").exists()
    assert inside.finish([]) == WorktreeResult([], None, None, "")
    # finish twice, or without begin
    assert cap.finish([]) == WorktreeResult([], None, None, "")


def test_the_snapshot_repo_is_bare_harness_side_and_keeps_history(
    tmp_path,
    monkeypatch,
):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    cap.begin()
    first = cap.finish([])
    second = WorktreeCapture(paths, ws, lambda: Redactor({}), hidden=lambda rel: False)
    second.begin()
    (ws / "new.txt").write_text("n\n")
    result = second.finish([])
    assert (paths.worktree_git / "HEAD").exists() and not (
        paths.worktree_git / "index"
    ).exists()
    assert paths.worktree_git in Paths.under(paths.home).harness_only()
    repo = Repo(paths.worktree_git)
    parents = (
        _git_raw(repo, ["rev-list", "--parents", "-n", "1", result.before])
        .decode()
        .split()
    )
    assert parents == [result.before, first.after]
    assert [(a.method, a.args[0], a.cell) for a in result.actions] == [
        ("write", "new.txt", -1),
    ]


def test_off_the_capture_is_never_consulted(tmp_path, monkeypatch):
    paths, ws, cap = _world(tmp_path, monkeypatch)
    cap.begin()
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    assert hooks.worker_audit() is None
    _done(
        monkeypatch,
        1.0,
        {
            "records": [
                {"event": "open", "path": str(ws.resolve() / "pay.csv"), "mode": "r"},
            ],
        },
    )
    assert cap._events == []


def test_default_hidden_paths_follow_the_sandbox_policy(
    world,
    monkeypatch,
):  # noqa: F811
    """Remote run (the real sandbox policy over the sandbox test world): what cells cannot see is never
    captured."""
    from unify import sandbox

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(capture_mod, "_ACTIVE", None)
    ws = world["workspace"]
    (ws / ".env").write_text("TOKEN=POLICY-HIDDEN\n")
    (ws / "id_rsa").write_text("POLICY-HIDDEN-KEY\n")
    assert sandbox.build_policy(fresh=True).workspace == ws.resolve()
    paths = Paths.under(world["state"])
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    cap = WorktreeCapture(paths, ws, lambda: Redactor({}))
    cap.begin()
    result = cap.finish([])
    files = tree_files(Repo(paths.worktree_git), result.before)
    assert "data.txt" in files and ".env" not in files and "id_rsa" not in files
