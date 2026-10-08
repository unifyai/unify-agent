"""The sandboxed worker imports from the request's memory export and may write it; nothing else is shown (Task 20).

Cells run in the real sandboxed worker (bubblewrap); no model is called.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory_v2.integration.test_checkout import _seed
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import export_checkout
from unify.memory_v2.integration.paths import Paths
from unify.settings import SETTINGS


async def _cell(ex: SessionExecutor, code: str) -> Any:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0),
        timeout=60,
    )
    assert res["error"] is None, res["error"]
    return res["result"]


def _home(world) -> tuple[Paths, list]:  # noqa: F811
    paths = Paths.under(world["state"])
    mem, sha = _seed(world["state"])  # the bare memory repo at <UNIFY_HOME>/memory
    assert mem.git_dir == paths.memory
    export_checkout(mem.git_dir, sha, paths.checkout)
    paths.episodes.mkdir()
    (paths.episodes / "HEAD").write_text("ref: refs/heads/main\n")
    paths.blobs.mkdir()
    paths.evidence.write_text("evidence")
    paths.state_dir.mkdir()
    (paths.state_dir / "state.json").write_text("{}")
    paths.worktree_git.mkdir()
    return paths, paths.harness_only()


def _probe(paths: Paths, hidden: list) -> str:
    return (
        "import os\n"
        f"[os.path.isdir({str(paths.checkout)!r}), "
        f"[os.path.exists(p) for p in {[str(p) for p in hidden]!r}]]"
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_worker_imports_and_writes_the_export_and_sees_nothing_else(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths, hidden = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    policy = sandbox.build_policy(fresh=True)
    assert all(policy.readable_violation(p) is not None for p in hidden)
    ex = SessionExecutor(environments={})
    try:
        out = await _cell(
            ex,
            "from env.spotify import hello\n"
            f"open({str(paths.checkout)!r} + '/env/spotify/scratch.py', 'w').write('x = 1')\n"
            "hello(None, 'ada')",
        )
        assert out == "hi ada"
        assert list(await _cell(ex, _probe(paths, hidden))) == [
            True,
            [False] * len(hidden),
        ]
    finally:
        await ex.close()
    assert (paths.checkout / "env/spotify/scratch.py").read_text() == "x = 1"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_with_the_switch_off_the_export_is_neither_mounted_nor_imported(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "")
    paths, hidden = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    ex = SessionExecutor(environments={})
    try:
        assert list(await _cell(ex, _probe(paths, hidden))) == [
            False,
            [False] * len(hidden),
        ]
        res = await asyncio.wait_for(
            ex.execute(
                code="from env.spotify import hello",
                state_mode="stateful",
                session_id=0,
            ),
            timeout=60,
        )
        assert res["error"] is not None
    finally:
        await ex.close()
