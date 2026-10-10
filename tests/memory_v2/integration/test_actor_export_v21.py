"""The actor's copy of a v2.1 library (spec §6, §20 item 3): read-only, no tests or fixtures, generated files."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory_v2.integration.test_worker_mount import _cell
from tests.memory_v2.test_layout import LIB, _tree
from tests.memory_v2.test_library_helper import _commit, _py
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration import request as request_mod
from unify.memory_v2.integration.checkout import (
    export_actor_v21,
    export_checkout,
    remove_checkout,
)
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.library_export import generated_v21
from unify.settings import SETTINGS

STRAYS = {
    "README.md": "stray\n",
    "memory/text/notes.txt": "stray\n",
    "memory/text/tests/helper_data.py": "X = 1\n",
    "memory/text/__pycache__/dates.cpython-312.pyc": "x",
    "INDEX.md": "written by a model\n",
    "memory/__init__.py": "raise SystemExit('a model wrote this')\n",
}
KEPT = [
    "memory/text/__init__.py",
    "memory/text/dates.py",
    "memory/text/parse.py",
    "memory/text/report.py",
    "memory/web/fetch.py",
    "notes/text/dates.md",
]


def _library_repo(git_dir: Path, extra: dict | None = None) -> tuple[Repo, str]:
    mem = Repo.init_bare(git_dir)
    return mem, _commit(mem, {**LIB, **(extra or {})}, "seed the library")


def _files(root: Path) -> list[str]:
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    )


def test_export_leaves_out_tests_fixtures_and_strays(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory", STRAYS)
    dest = tmp_path / "co"
    generated = export_actor_v21(mem.git_dir, sha, dest)
    assert _files(dest) == sorted(KEPT + list(generated))
    assert not any("/tests/" in f for f in _files(dest))
    # a committed generated path is never exported as written: the harness's file replaces it
    assert (dest / "INDEX.md").read_bytes() == generated["INDEX.md"]
    assert (dest / "memory/__init__.py").read_bytes() == generated["memory/__init__.py"]


def test_generated_files_replace_committed_ones(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory", STRAYS)
    dest = tmp_path / "co"
    export_actor_v21(mem.git_dir, sha, dest)
    out = _py(
        dest,
        "import memory.text.dates as d, memory\nprint(d.parse_date('2026-10-09'), memory.index('web').splitlines()[0])",
    )
    assert out == "2026-10-09 ## memory.web\n"


def test_the_export_is_read_only_and_still_removable(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory")
    dest = tmp_path / "co"
    first = export_actor_v21(mem.git_dir, sha, dest)
    for p in [dest, *dest.rglob("*")]:
        assert stat.S_IMODE(p.lstat().st_mode) & 0o222 == 0, p
    assert (
        export_actor_v21(mem.git_dir, sha, dest) == first
    )  # replaces a read-only export
    export_checkout(mem.git_dir, sha, dest)  # so does v2's export
    remove_checkout(dest)
    assert not dest.exists()


def test_the_export_is_deterministic(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory")
    a = export_actor_v21(mem.git_dir, sha, tmp_path / "a")
    b = export_actor_v21(mem.git_dir, sha, tmp_path / "b")
    assert a == b and _files(tmp_path / "a") == _files(tmp_path / "b")
    # INDEX.md and links.json are pure functions of the library files: the tests beside them change nothing
    fresh = generated_v21(_tree(tmp_path / "plain"))
    assert {k: a[k] for k in ("INDEX.md", "links.json")} == {
        k: fresh[k] for k in ("INDEX.md", "links.json")
    }


def test_hooks_bind_the_v21_export_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(tmp_path)
    paths.checkout.mkdir(parents=True)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, v21=True),
    )
    assert hooks.worker_mounts() == [] and hooks.worker_readonly_mounts() == [
        paths.checkout,
    ]
    assert hooks.worker_paths() == [
        str(paths.checkout),
    ]  # the copy's root: the parent of memory/
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(index="", paths=paths))
    assert (
        hooks.worker_mounts() == [paths.checkout]
        and hooks.worker_readonly_mounts() == []
    )


@needs_bwrap
def test_wrap_argv_binds_late_mounts_read_only_after_the_masks(world):  # noqa: F811
    policy = sandbox.build_policy(fresh=True)
    target = Path(world["state"]) / "memory-checkout"
    target.mkdir()
    real = str(Path(os.path.realpath(target)))
    argv = sandbox.wrap_argv(["true"], policy, late_readonly=[target])
    ro = [
        k
        for k, a in enumerate(argv)
        if a == "--ro-bind" and argv[k + 1] == real and argv[k + 2] == real
    ]
    masks = [
        k
        for k, a in enumerate(argv)
        if a == "--tmpfs" and real.startswith(argv[k + 1].rstrip("/") + "/")
    ]
    assert ro and masks and ro[-1] > max(masks)
    assert sandbox.wrap_argv(["true"], policy) == sandbox.wrap_argv(
        ["true"],
        policy,
        late_readonly=(),
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_cell_imports_the_library_and_cannot_write_it(
    world,
    monkeypatch,
):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2", "on")
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_v21(mem.git_dir, sha, paths.checkout)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, v21=True),
    )
    target = str(paths.checkout / "memory/text/dates.py")
    ex = SessionExecutor(environments={})
    try:
        out = await _cell(
            ex,
            "import errno, os, memory, memory.text.dates as d\n"
            "try:\n"
            f"    os.chmod({target!r}, 0o644)\n"
            "    tried = 'changed'\n"
            "except OSError as exc:\n"
            "    tried = errno.errorcode.get(exc.errno, str(exc.errno))\n"
            f"root = {str(paths.checkout)!r}\n"
            "seen = sorted(os.path.relpath(os.path.join(r, f), root) for r, _, fs in os.walk(root) for f in fs)\n"
            "[d.parse_date('2026-10-09'), tried, memory.index('web').splitlines()[0], seen]",
        )
    finally:
        await ex.close()
    # EROFS comes only from the read-only bind: the owner may chmod a file of mode 0444 on a writable mount
    assert list(out)[:3] == ["2026-10-09", "EROFS", "## memory.web"]
    # what the actor itself sees (RUNTIME, T4): the library and notes, never its tests or their data
    seen = list(out)[3]
    assert set(KEPT) <= set(seen) and not any("tests" in f.split("/") for f in seen)


@needs_bwrap
def test_wrap_argv_binds_late_only_the_memory_checkout_of_the_state_directory(
    world,  # noqa: F811
    tmp_path,
):
    """RUNTIME, P3 Task 4 (S-1): no caller mounts any other host path into a cell through *late_readonly*,
    not even another directory of the state directory (the memory repo, episodes, blobs, evidence).
    """
    policy = sandbox.build_policy(fresh=True)
    state = Path(world["state"])
    inside = state / "memory-checkout"
    inside.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for name in ("memory", "episodes.git", "blobs"):
        (state / name).mkdir(exist_ok=True)
    (state / "nested" / "memory-checkout").mkdir(parents=True)
    (state / "memory-alias").symlink_to(inside)
    (state / "memory-file").write_text("x")
    for bad in (
        outside,
        state / "memory",
        state / "episodes.git",
        state / "blobs",
        state / "nested" / "memory-checkout",
        state / "memory-alias",
        state / "memory-missing",
        state / "memory-file",
        state,
        Path("/etc"),
    ):
        with pytest.raises(sandbox.SandboxRefusal, match="late-readonly-state-only"):
            sandbox.wrap_argv(["true"], policy, late_readonly=[bad])
    assert sandbox.LATE_READONLY_DIRS == ("memory-checkout",)
    argv = sandbox.wrap_argv(["true"], policy, late_readonly=[inside])
    real = os.path.realpath(inside)
    assert any(
        argv[k : k + 3] == ["--ro-bind", real, real] for k in range(len(argv) - 2)
    )
    # off, a v2 worker's command line is unchanged element for element
    assert sandbox.wrap_argv(["true"], policy, writable=[inside]) == sandbox.wrap_argv(
        ["true"],
        policy,
        writable=[inside],
        late_readonly=[],
    )


def test_make_read_only_never_follows_a_link(tmp_path):
    from unify.memory_v2.integration.checkout import make_read_only

    target = tmp_path / "host.txt"
    target.write_text("x")
    target.chmod(0o644)
    root = tmp_path / "co"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "link.py").symlink_to(target)
    make_read_only(root)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert stat.S_IMODE(root.lstat().st_mode) == 0o555


def test_the_removal_retry_restores_write_bits_only_inside_the_export(tmp_path):
    from unify.memory_v2.integration.checkout import _writable_retry

    root = tmp_path / "co"
    root.mkdir()
    outside = tmp_path / "host"
    outside.mkdir()
    outside.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            _writable_retry(
                os.rmdir,
                str(outside / "x"),
                PermissionError(13, "denied"),
                root=root,
            )
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
        (root / "alias").symlink_to(outside)
        with pytest.raises(PermissionError):
            _writable_retry(
                os.unlink,
                str(root / "alias" / "x"),
                PermissionError(13, "denied"),
                root=root,
            )
        assert stat.S_IMODE(outside.stat().st_mode) == 0o555
    finally:
        outside.chmod(0o755)
