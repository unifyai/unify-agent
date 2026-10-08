"""Memory v2's harness-side state is invisible to cells, and checker text never lands (integration Task 26).

Visibility: with a request's run open, the sandbox would let a cell read none of memory, episodes, blobs,
evidence, the state directory, ``worktree.git`` or the events file; none of them lies under a root a cell
reads (the workspace, ``transcripts/`` and the run record are readable by design); the worker's bubblewrap
command binds none of them; and a cell in the real worker sees the export and nothing else.

The sentinel: an outcome whose summary and check reasons carry ``SENTINEL-7f3a`` is taken, the request is
recorded and a consolidation pass runs (E forced to 1, a fake Sol that records what it was sent and what
was staged for it). The sentinel is in no file, git object (notes included) or database row under
``UNIFY_HOME``, in nothing Sol was sent or staged, and in no event. That part runs on the real
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
from tests.memory_v2.integration import fake_tracks
from tests.memory_v2.integration.test_checkout import _seed
from unify import sandbox, transcripts
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.paths import Paths
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
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    fakes = fake_tracks.install(monkeypatch)
    paths = Paths.under(world["state"])
    _seed(world["state"])
    paths.worktree_git.mkdir()
    (paths.worktree_git / "HEAD").write_text("ref: refs/heads/main\n")
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    paths.events.write_text('{"type": "consolidation"}\n')
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
        # the export is the one path put back, read-write, for the worker
        assert hooks.worker_mounts() == [o.paths.checkout]
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
        assert ("--bind", checkout, checkout) in binds
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
    from unify.actor.execution.session import SessionExecutor

    o = _open_run(world, monkeypatch)
    stores = [str(p) for p in _stores(o.paths)]
    transcripts_dir = str(transcripts.transcripts_dir())
    ex = SessionExecutor(environments={})
    try:
        out = await _cell(
            ex,
            "import os\n"
            f"t = {transcripts_dir!r}\n"
            f"[os.path.isdir({str(o.paths.checkout)!r}), "
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


def _real(name: str):
    """The real module of another track, or skip until it is merged."""
    return pytest.importorskip(f"unify.memory_v2.integration.{name}")


def test_checker_text_never_lands_anywhere(monkeypatch, tmp_path):
    consolidate = _real("consolidate")
    _real("trajectory")
    _real("cost")
    _real("worktree_capture")
    for attr in ("run_due_passes", "post_checker", "open_stores", "unillm_turn"):
        if not hasattr(consolidate, attr):
            pytest.skip(f"consolidate.{attr} is not merged yet")

    home = tmp_path / "unify"
    (home / "workspace").mkdir(parents=True)
    (home / "workspace" / "pay.csv").write_text("name,amount\nada,5\n")
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.delenv("UNIFY_STORE_PATH", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_MEMORY_V2_E",
        1,
    )  # the first episode makes a pass due
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD", "")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    monkeypatch.setattr(request_mod, "_CURRENT", None)
    _seed(home)
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(
        tempfile,
        "tempdir",
        str(scratch),
    )  # Sol's staging lands here, to be read

    sent: list[str] = []
    staged: list[bytes] = []

    def fake_turn_factory(model, effort, **_kw):
        async def turn(messages, tools):
            sent.append(json.dumps(messages, default=str))
            for path in sorted(scratch.glob("memv2-sol-*/inputs/**/*")):
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

    assert (
        sent
    ), f"no pass ran (progress: {progress})"  # otherwise the check below proves nothing
    assert staged, "nothing was staged for Sol"
    assert [e["phase"] for e in emitted][:2] == ["start", "end"]
    assert SENTINEL.encode() not in fake_tracks.dump_home(home)
    assert not any(SENTINEL in m for m in sent)
    assert not any(SENTINEL.encode() in b for b in staged)
    assert SENTINEL not in json.dumps(emitted)
    assert SENTINEL not in " ".join(progress)
    # the verdict itself did land, as pass/fail only
    signals = os.popen(
        f"git --git-dir {home / 'episodes.git'} log --format=%N --notes=signals main",
    ).read()
    assert "fail" in signals and SENTINEL not in signals
