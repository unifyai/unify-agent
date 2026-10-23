"""The actor's copy of a library-mode library: read-only, no tests or fixtures, generated files."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.integration.test_worker_mount import _cell
from tests.memory.test_layout import LIB, _tree
from tests.memory.test_library_helper import _commit, _py
from unify import sandbox
from tests.memory.cells import Cells
from unify.memory.gitio import Repo
from unify.memory.integration import hooks
from unify.memory.integration import request as request_mod
from unify.memory.integration.checkout import (
    export_actor_library_mode,
    remove_checkout,
)
from unify.memory.integration.paths import Paths
from unify.memory.library_export import generated_library_mode
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
    generated = export_actor_library_mode(mem.git_dir, sha, dest)
    assert _files(dest) == sorted(KEPT + list(generated))
    assert not any("/tests/" in f for f in _files(dest))
    # a committed generated path is never exported as written: the harness's file replaces it
    assert (dest / "INDEX.md").read_bytes() == generated["INDEX.md"]
    assert (dest / "memory/__init__.py").read_bytes() == generated["memory/__init__.py"]


def test_generated_files_replace_committed_ones(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory", STRAYS)
    dest = tmp_path / "co"
    export_actor_library_mode(mem.git_dir, sha, dest)
    out = _py(
        dest,
        "import memory.text.dates as d, memory\nprint(d.parse_date('2026-10-09'), [i.id for i in memory.items('web')])",
    )
    assert out == "2026-10-09 ['memory.web.fetch:get']\n"


def test_the_export_is_read_only_and_still_removable(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory")
    dest = tmp_path / "co"
    first = export_actor_library_mode(mem.git_dir, sha, dest)
    for p in [dest, *dest.rglob("*")]:
        assert stat.S_IMODE(p.lstat().st_mode) & 0o222 == 0, p
    assert (
        export_actor_library_mode(mem.git_dir, sha, dest) == first
    )  # replaces a read-only export
    remove_checkout(dest)
    assert not dest.exists()


def test_the_export_is_deterministic(tmp_path):
    mem, sha = _library_repo(tmp_path / "memory")
    a = export_actor_library_mode(mem.git_dir, sha, tmp_path / "a")
    b = export_actor_library_mode(mem.git_dir, sha, tmp_path / "b")
    assert a == b and _files(tmp_path / "a") == _files(tmp_path / "b")
    # INDEX.md and links.json are pure functions of the library files: the tests beside them change nothing
    fresh = generated_library_mode(_tree(tmp_path / "plain"))
    assert {k: a[k] for k in ("INDEX.md", "links.json")} == {
        k: fresh[k] for k in ("INDEX.md", "links.json")
    }


def test_hooks_bind_the_library_export_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(tmp_path)
    paths.checkout.mkdir(parents=True)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, library_mode=True),
    )
    assert hooks.worker_mounts() == [] and hooks.worker_readonly_mounts() == [
        paths.checkout,
    ]
    assert hooks.worker_paths() == [
        str(paths.checkout),
    ]  # the copy's root: the parent of memory/
    # (rev 2 has one layout: the v2 read-write export half went with it)


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
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, library_mode=True),
    )
    # The cell sees the copy where the worker's sandbox binds it.
    target = sandbox.LIBRARY_MOUNT + "/memory/text/dates.py"
    ex = Cells()
    try:
        out = await _cell(
            ex,
            "import errno, os, memory, memory.text.dates as d\n"
            "try:\n"
            f"    os.chmod({target!r}, 0o644)\n"
            "    tried = 'changed'\n"
            "except OSError as exc:\n"
            "    tried = errno.errorcode.get(exc.errno, str(exc.errno))\n"
            f"root = {sandbox.LIBRARY_MOUNT!r}\n"
            "seen = sorted(os.path.relpath(os.path.join(r, f), root) for r, _, fs in os.walk(root) for f in fs)\n"
            "[d.parse_date('2026-10-09'), tried, [i.id for i in memory.items('web')], seen]",
        )
    finally:
        await ex.close()
    # EROFS comes only from the read-only bind: the owner may chmod a file of mode 0444 on a writable mount
    assert list(out)[:3] == ["2026-10-09", "EROFS", ["memory.web.fetch:get"]]
    # what the actor itself sees: the library and notes, never its tests or their data
    seen = list(out)[3]
    assert set(KEPT) <= set(seen) and not any("tests" in f.split("/") for f in seen)


@needs_bwrap
def test_wrap_argv_binds_late_only_the_memory_checkout_of_the_state_directory(
    world,  # noqa: F811
    tmp_path,
):
    """No caller mounts any other host path into a cell through *late_readonly*,
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
    from unify.memory.integration.checkout import make_read_only

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
    from unify.memory.integration.checkout import _writable_retry

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
