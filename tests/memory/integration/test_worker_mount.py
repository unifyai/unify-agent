"""The sandboxed worker imports from the request's memory export and may write it; nothing else is shown (Task 20).

Cells run in the real sandboxed worker (bubblewrap); no model is called.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.integration.test_checkout import _seed
from unify import sandbox
from tests.memory.cells import Cells
from unify.memory.integration import request as request_mod
from unify.memory.integration.checkout import export_checkout
from unify.memory.integration.paths import Paths
from unify.settings import SETTINGS


async def _cell(ex: Cells, code: str) -> Any:
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


# Where the worker's sandbox binds the export: a fixed path, never its host path.
LIBRARY = "/memory-library"


def _probe(paths: Paths, hidden: list) -> str:
    return (
        "import os\n"
        f"[os.path.isdir({LIBRARY!r}), "
        f"[os.path.exists(p) for p in {[str(p) for p in (*hidden, paths.checkout)]!r}]]"
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_worker_imports_and_writes_the_export_and_sees_nothing_else(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths, hidden = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    policy = sandbox.build_policy(fresh=True)
    assert all(policy.readable_violation(p) is not None for p in hidden)
    ex = Cells()
    try:
        out = await _cell(
            ex,
            "from env.spotify import hello\n"
            f"open({LIBRARY!r} + '/env/spotify/scratch.py', 'w').write('x = 1')\n"
            "hello(None, 'ada')",
        )
        assert out == "hi ada"
        assert list(await _cell(ex, _probe(paths, hidden))) == [
            True,
            [False] * (len(hidden) + 1),
        ]
    finally:
        await ex.close()
    assert (paths.checkout / "env/spotify/scratch.py").read_text() == "x = 1"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_worker_writes_no_bytecode_into_the_export(
    world,  # noqa: F811
    monkeypatch,
):
    """Under memory v2 the child sets ``sys.dont_write_bytecode`` (it runs under ``-I``, so the
    environment variable would be ignored); importing the library leaves no ``__pycache__`` in the export.
    """
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths, _ = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    ex = Cells()
    try:
        out = await _cell(
            ex,
            "import sys\nfrom env.spotify import hello\n[sys.dont_write_bytecode, hello(None, 'ada')]",
        )
    finally:
        await ex.close()
    assert list(out) == [True, "hi ada"]
    assert not list(paths.checkout.rglob("__pycache__"))
    assert not list(paths.checkout.rglob("*.pyc"))


def test_the_init_message_asks_for_no_bytecode_only_under_memory(
    world,
    monkeypatch,
):  # noqa: F811
    from unify.actor.execution.worker import PythonWorker

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    assert "no_bytecode" not in PythonWorker()._init_message()
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths, _ = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    assert PythonWorker()._init_message()["no_bytecode"] is True


REFUSER = (
    "\n\nclass MemoryInputError(ValueError):\n    pass\n\n\n"
    "def need_name(apis, name):\n"
    '    """Refuse an empty name.\n\n    Effect: read\n    """\n'
    "    if not name:\n"
    '        raise MemoryInputError("name must be a non-empty string")\n'
    "    return name\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_with_the_switch_off_the_export_is_neither_mounted_nor_imported(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    paths, hidden = _home(world)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths),
    )
    ex = Cells()
    try:
        assert list(await _cell(ex, _probe(paths, hidden))) == [
            False,
            [False] * (len(hidden) + 1),
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


def test_no_mount_or_import_path_until_the_export_exists(tmp_path, monkeypatch):
    """A bind source that does not exist would stop the worker: nothing is mounted before the export is made."""
    from unify.memory.integration import hooks

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(tmp_path / "state")
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    assert not paths.checkout.exists()
    assert hooks.worker_mounts() == [] and hooks.worker_paths() == []
    paths.checkout.mkdir(parents=True)
    assert hooks.worker_mounts() == [paths.checkout] and hooks.worker_paths() == [
        str(paths.checkout),
    ]
    # the library test kit inside the export (its tests use memlab): after the export on the import path,
    # mounted with it
    (paths.checkout / ".memlab").mkdir()
    assert hooks.worker_mounts() == [paths.checkout] and hooks.worker_paths() == [
        str(paths.checkout),
        str(paths.checkout / ".memlab"),
    ]


def test_a_failing_audit_lookup_never_stops_the_worker(tmp_path, monkeypatch):
    from unify.memory.integration import hooks, worktree_capture

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=Paths.under(tmp_path / "state")),
    )

    def boom():
        raise RuntimeError("capture state broken")

    monkeypatch.setattr(worktree_capture, "active", boom)
    assert hooks.worker_audit() is None
