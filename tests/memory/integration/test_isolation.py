"""Memory v2's harness-side state is invisible to cells, and checker text never lands (integration Task 26).

Visibility: with a request's run open, the sandbox would let a cell read none of memory, episodes, blobs,
evidence, the state directory, ``worktree.git`` or the events file; none of them lies under a root a cell
reads (the workspace, ``transcripts/`` and the run record are readable by design); the worker's bubblewrap
command binds none of them; and a cell in the real worker sees the export and nothing else.

The sentinel: an outcome whose summary and check reasons carry ``SENTINEL-7f3a`` is taken, the request is
recorded and a consolidation pass runs (E forced to 1, a fake writer that records what it was sent and what
was staged for it). The sentinel is in no file, git object (notes included) or database row under
``UNIFY_HOME``, in nothing the writer was sent or staged, and in no event. That part runs on the real
trajectory, consolidation, cost and work-tree modules, so it skips until they are merged.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.integration import fake_tracks
from tests.memory.integration.test_checkout import _seed
from unify import sandbox, transcripts
from unify.memory.integration import hooks
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.settings import SETTINGS

SENTINEL = "SENTINEL-7f3a"


def _within(p: Path, root: Path) -> bool:
    p, root = Path(os.path.realpath(p)), Path(os.path.realpath(root))
    return p == root or root in p.parents


def _stores(paths: Paths) -> list[Path]:
    """Every harness-side path; the events file (in the state directory) is named on its own as well."""
    return [*paths.harness_only(), paths.events]


def _open_run(world, monkeypatch):  # noqa: F811
    """A run begun under the sandbox world (other tracks faked), with every store present on disk."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    fakes = fake_tracks.install(monkeypatch)
    paths = Paths.under(world["state"])
    _seed(world["state"])
    paths.worktree_git.mkdir()
    (paths.worktree_git / "HEAD").write_text("ref: refs/heads/main\n")
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    paths.events.write_text('{"type": "consolidation"}\n')
    paths.captures.mkdir(parents=True, exist_ok=True)  # made on the first capture, like the work tree and events
    ctx = contextvars.copy_context()
    run = ctx.run(request_mod.RequestRun.begin, "hi")
    for p in _stores(paths):
        assert p.exists(), p
    return SimpleNamespace(run=run, paths=paths, ctx=ctx, fakes=fakes)


def test_no_store_lies_under_a_root_a_cell_reads(world, monkeypatch):  # noqa: F811
    o = _open_run(world, monkeypatch)
    try:
        policy = sandbox.build_policy(fresh=True)
        readable_roots = [
            policy.workspace,
            *policy.readonly_state,
            *policy.root_visible,
            transcripts.transcripts_dir(),  # cells read transcripts by design
        ]
        for store in _stores(o.paths):
            assert policy.readable_violation(store) is not None, store
            for root in readable_roots:
                assert not _within(store, root), (store, root)
        # the export is the one path put back for the worker, read-only (rev 2: nothing is bound read-write)
        assert hooks.worker_mounts() == [] and hooks.worker_readonly_mounts() == [o.paths.checkout]
        assert not any(_within(o.paths.checkout, s) for s in _stores(o.paths))
    finally:
        o.ctx.run(o.run.abort)


@needs_bwrap
def test_the_worker_command_binds_no_store(world, monkeypatch):  # noqa: F811
    o = _open_run(world, monkeypatch)
    try:
        policy = sandbox.build_policy(fresh=True)
        # passes the derived roots' rule and the workspace's secret scan (raises SandboxRefusal otherwise)
        argv = sandbox.wrap_argv(
            ["true"],
            policy,
            cwd=str(policy.workspace),
            writable=hooks.worker_mounts(),
            late_readonly=hooks.worker_readonly_mounts(),
        )
        end = argv.index("--") if "--" in argv else len(argv)
        binds = [
            (argv[i], argv[i + 1], argv[i + 2])
            for i in range(end - 2)
            if argv[i]
            in ("--bind", "--ro-bind", "--dev-bind", "--bind-try", "--ro-bind-try")
        ]
        for opt, src, dst in binds:
            for store in _stores(o.paths):
                assert not _within(Path(src), store), (opt, src, store)
                assert not _within(Path(dst), store), (opt, dst, store)
        checkout = str(Path(os.path.realpath(o.paths.checkout)))
        # rev 2: the copy is bound read-only, as the worker binds it (late, after the masks), never read-write
        mine = [b for b in binds if b[1] == checkout]
        assert mine and all(opt == "--ro-bind" for opt, _, _ in mine), mine
    finally:
        o.ctx.run(o.run.abort)


async def _cell(ex, code: str) -> Any:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0),
        timeout=60,
    )
    assert res["error"] is None, res["error"]
    return res["result"]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_cell_sees_the_export_and_no_store(world, monkeypatch):  # noqa: F811
    from tests.memory.cells import Cells

    o = _open_run(world, monkeypatch)
    stores = [str(p) for p in _stores(o.paths)]
    # The worker's sandbox shows the transcripts and the export at fixed paths.
    transcripts_dir = sandbox.TRANSCRIPTS_MOUNT
    ex = Cells()
    try:
        out = await _cell(
            ex,
            "import os\n"
            f"t = {transcripts_dir!r}\n"
            f"[os.path.isdir({sandbox.LIBRARY_MOUNT!r}), "
            f"[os.path.exists(p) for p in {stores!r}], "
            "sorted(os.listdir(t)) if os.path.isdir(t) else []]",
        )
        seen_export, seen_stores, listed = list(out)
        assert seen_export is True
        assert list(seen_stores) == [False] * len(stores)
        names = {Path(s).name for s in stores}
        assert not names & set(listed)
    finally:
        await ex.close()
        o.ctx.run(o.run.abort)


# ── the sentinel ─────────────────────────────────────────────────────────────


def test_checker_text_never_lands_anywhere(monkeypatch, tmp_path):
    """The checker's free text (an outcome's summary and reasons) never lands in the home, the request's events, the
    signals, or anything a writer pass reads. Rev 2: the request starts the writer service and never runs a pass
    itself, so the pass runs here as the service runs it, in-process (``writer_service.step``)."""
    from unify.memory import writer_service as ws
    from unify.memory.integration import consolidate

    home = tmp_path / "unify"
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "pay.csv").write_text("name,amount\nada,5\n")
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    # the request's end starts the writer service; here the service's one step runs below, in this process
    started: list = []
    monkeypatch.setattr(
        ws,
        "ensure_service",
        lambda home, **kw: started.append(home) or SimpleNamespace(status="started", pid=None, detail=None),
    )
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    _seed(home)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(
        tempfile,
        "tempdir",
        str(scratch),
    )  # the writer's staging lands here, to be read

    sent: list[str] = []
    staged: list[bytes] = []

    def fake_turn_factory(model, effort, **_kw):
        async def turn(messages, tools):
            sent.append(json.dumps(messages, default=str))
            for path in sorted(scratch.glob("memory-writer-*/inputs/**/*")):
                if path.is_file() and not path.is_symlink():
                    staged.append(path.read_bytes())
            return {"role": "assistant", "content": "nothing to add"}, "0.01"

        return turn

    monkeypatch.setattr(consolidate, "unillm_turn", fake_turn_factory)

    ctx = contextvars.copy_context()
    run = ctx.run(request_mod.RequestRun.begin, "Pay Ada back from pay.csv.")
    answer = run.take_outcome(
        {
            "solved": False,
            "summary": f"the checker says {SENTINEL}",
            "checks": [{"name": "c", "passed": False, "reason": SENTINEL}],
        },
    )
    assert answer["accepted"] is True and SENTINEL not in json.dumps(answer)
    ts = "2026-10-08T12:00:00+00:00"
    tpath = transcripts.transcripts_dir() / f"{run.episode_id}.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "seq": 0,
            "ts": ts,
            "session": run.episode_id,
            "type": "message",
            "message": {"role": "user", "content": "Pay Ada back from pay.csv."},
        },
        {
            "seq": 1,
            "ts": ts,
            "session": run.episode_id,
            "type": "message",
            "message": {"role": "assistant", "content": "Paid."},
        },
    ]
    tpath.write_text("".join(json.dumps(r) + "\n" for r in rows))
    emitted: list[dict] = []
    progress: list[str] = []
    with asyncio.Runner() as runner:
        runner.run(
            run.finish(None, progress=progress.append, emit=emitted.append),
            context=ctx,
        )
    assert started, f"the request started no writer service (progress: {progress})"
    # the episode-end fork, settled as a fork leaves it (here no fork lane exists, so a stand-in writes what a fork
    # writes), gives the ledger the episode's entry for the service's pass
    from unify.memory.integration.paths import Paths

    fork_dir = Paths.under(home).state_dir / "memory-staging" / run.episode_id
    fork_dir.mkdir(parents=True, exist_ok=True)
    (fork_dir / "summary.md").write_text("## Asked\nPay Ada back.\n## Approach\n- read pay.csv\n")
    (fork_dir / "fork.json").write_text(
        json.dumps({"episode": run.episode_id, "status": "ok", "files": ["summary.md"]}),
    )
    # rev 2.1: a capture alone waits for its group; the service steps once the idle time has passed, so it is due
    from datetime import datetime, timedelta, timezone

    from unify.memory import capture_groups

    later = datetime.now(timezone.utc) + timedelta(seconds=capture_groups.IDLE_S + 1)
    stepped = ws.step(home, settings=SETTINGS, now=later)

    assert (
        sent
    ), f"no pass ran (progress: {progress}; step: {stepped})"  # otherwise the check below proves nothing
    assert staged, "nothing was staged for the writer"
    assert any(e.get("phase") == "service" for e in emitted), emitted
    assert SENTINEL.encode() not in fake_tracks.dump_home(home)
    assert not any(SENTINEL in m for m in sent)
    assert not any(SENTINEL.encode() in b for b in staged)
    assert SENTINEL not in json.dumps(emitted)
    assert SENTINEL not in " ".join(progress)
    # rev 2 infers the regime from the signals the actor saw: with none, the regime is "none" and the hidden grader's
    # verdict decides nothing, so it is not posted at all (its text never could be)
    signals = os.popen(
        f"git --git-dir {home / 'episodes.git'} log --format=%N --notes=signals main",
    ).read()
    assert SENTINEL not in signals and "fail" not in signals
    from unify.memory.evidence import EvidenceStore

    assert EvidenceStore(Paths.under(home).evidence).regime_record(run.episode_id) == ("none", "nothing")
