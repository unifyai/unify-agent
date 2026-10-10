"""Symbolic: the interpreter roots' secret scan is kept on disk, once per venv.

The scan of the interpreter roots (about 60k entries) was kept only in the
harness process, so every new process (a test session, a benchmark episode's
``unify act``) paid it again on its first ``build_policy``. It is now kept in
``$XDG_CACHE_HOME/unify/sandbox-scans`` too, keyed on the root and a
fingerprint of its directories down to depth 2 (and of its site-packages'),
and anything wrong with a cache entry rescans. Pure Python: nothing here runs
bubblewrap, and every tree is one the test makes.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest

from unify import sandbox

SITE = Path("lib") / "python3.12" / "site-packages"


@pytest.fixture
def scan(tmp_path, monkeypatch):
    """A fake interpreter root, a cache directory, and a scan of the root as a
    fresh harness process would run it (empty in-memory cache), counting the
    directories the walk reads."""
    root = tmp_path / "venv"
    (root / SITE / "pkg" / "sub").mkdir(parents=True)
    (root / "bin").mkdir()
    (root / ".env").write_text("A=1\n")
    (root / SITE / "pkg" / "service-key.json").write_text("{}\n")
    (root / SITE / "pkg" / "cacert.pem").write_text("public\n")
    cache_dir = tmp_path / "xdg" / "unify" / "sandbox-scans"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    # The trees are made just now; the settle window would keep them out.
    monkeypatch.setattr(sandbox, "_ROOT_SCAN_SETTLE_NS", -(10**18))
    reads: list[Path] = []
    real = sandbox._secret_entries

    def counting(directory, *args, **kwargs):
        reads.append(Path(directory))
        return real(directory, *args, **kwargs)

    monkeypatch.setattr(sandbox, "_secret_entries", counting)

    def run(*, cache=cache_dir, skip=()):
        monkeypatch.setattr(sandbox, "_SECRET_SCAN_CACHE", {})
        reads.clear()
        files, dirs = sandbox._find_secret_files(
            [root],
            list(skip),
            cached=[root],
            cache_dir=cache,
        )
        return {p for p, _ in files}, {p for p, _ in dirs}, len(reads)

    return root, cache_dir, run


def _entry(cache_dir: Path) -> Path:
    (path,) = list(cache_dir.glob("*.json"))
    return path


def test_a_second_process_reads_the_scan_from_disk_without_walking(scan):
    root, cache_dir, run = scan
    files, _, walked = run()
    assert files == {root / ".env", root / SITE / "pkg" / "service-key.json"}
    assert walked > 0
    again, _, walked_again = run()
    assert again == files
    assert walked_again == 0


def test_the_scan_is_the_default_cache_dir_under_xdg(scan, tmp_path):
    assert sandbox._scan_cache_dir() == Path(
        os.path.realpath(tmp_path / "xdg" / "unify" / "sandbox-scans"),
    )


@pytest.mark.parametrize(
    "made",
    [
        "new.pem",  # depth 1 under the root
        "bin/.envrc",  # depth 2 under the root
        str(SITE / "id.pem"),  # depth 1 under site-packages
        str(SITE / "pkg" / "other-key.json"),  # depth 2
        str(SITE / "pkg" / "sub" / ".env.local"),  # depth 3: sub's mtime
    ],
)
def test_a_file_made_within_the_fingerprint_is_found(scan, made):
    root, _, run = scan
    run()
    (root / made).write_text("secret\n")
    files, _, walked = run()
    assert root / made in files
    assert walked > 0


def test_a_new_package_directory_is_found(scan):
    root, _, run = scan
    run()
    new = root / SITE / "newpkg" / "a" / "b"
    new.mkdir(parents=True)
    (new / "x.pem").write_text("secret\n")
    files, _, _ = run()
    assert new / "x.pem" in files


def test_a_removed_secret_rescans(scan):
    root, _, run = scan
    run()
    (root / ".env").unlink()
    files, _, walked = run()
    assert root / ".env" not in files and walked > 0


@pytest.mark.parametrize(
    "corrupt",
    ["garbage", "truncated", "other-root", "other-key", "extra-field", "bad-rule"],
)
def test_a_corrupt_or_foreign_cache_entry_rescans_and_is_rewritten(scan, corrupt):
    root, cache_dir, run = scan
    expected, _, _ = run()
    path = _entry(cache_dir)
    data = json.loads(path.read_text())
    if corrupt == "garbage":
        text = "\x00not json"
    elif corrupt == "truncated":
        text = path.read_text()[:20]
    else:
        if corrupt == "other-root":
            data["root"] = "/elsewhere"
        elif corrupt == "other-key":
            data["key"] = "0" * 64
        elif corrupt == "extra-field":
            data["trusted"] = True
        else:
            data["files"] = [[str(root / ".env"), "no-such-rule"]]
        text = json.dumps(data)
    path.write_text(text)
    os.chmod(path, 0o600)
    files, _, walked = run()
    assert files == expected
    assert walked > 0
    rewritten = json.loads(path.read_text())
    assert rewritten["root"] == str(root) and "trusted" not in rewritten


def test_a_cache_entry_naming_a_gone_path_or_loosely_moded_rescans(scan):
    """An entry naming a path that no longer exists, or a file another user
    could have written (mode other than 0600), is not trusted."""
    root, cache_dir, run = scan
    run()
    path = _entry(cache_dir)
    data = json.loads(path.read_text())
    data["files"].append([str(root / "gone.pem"), "mask-credentials"])
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)
    files, _, walked = run()
    assert walked > 0 and root / ".env" in files
    os.chmod(path, 0o644)
    files, _, walked = run()
    assert walked > 0 and root / ".env" in files


def test_an_old_cache_entry_rescans(scan, monkeypatch):
    root, cache_dir, run = scan
    run()
    path = _entry(cache_dir)
    data = json.loads(path.read_text())
    data["created"] = time.time() - sandbox._ROOT_SCAN_MAX_AGE_S - 1
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)
    _, _, walked = run()
    assert walked > 0


def test_a_rule_version_bump_rescans(scan, monkeypatch):
    _, _, run = scan
    run()
    monkeypatch.setattr(
        sandbox,
        "_ROOT_SCAN_RULES_VERSION",
        sandbox._ROOT_SCAN_RULES_VERSION + 1,
    )
    _, _, walked = run()
    assert walked > 0


def test_a_skip_that_meets_the_root_is_part_of_the_key(scan):
    root, _, run = scan
    run()
    files, _, walked = run(skip=[root / SITE])
    assert walked > 0
    assert files == {root / ".env"}


def test_an_unwritable_cache_dir_still_masks(scan, tmp_path):
    root, _, run = scan
    blocker = tmp_path / "a-file"
    blocker.write_text("")
    unwritable = blocker / "sandbox-scans"  # under a file: never creatable
    files, _, walked = run(cache=unwritable)
    assert files == {root / ".env", root / SITE / "pkg" / "service-key.json"}
    assert walked > 0
    files, _, walked = run(cache=unwritable)
    assert root / ".env" in files and walked > 0


def test_the_cache_file_is_mode_0600_in_a_0700_directory(scan):
    _, cache_dir, run = scan
    run()
    assert stat.S_IMODE(os.stat(_entry(cache_dir)).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(cache_dir).st_mode) == 0o700
    assert not list(cache_dir.glob(".scan-*"))


def test_an_unsettled_tree_is_not_cached(scan, monkeypatch):
    _, cache_dir, run = scan
    monkeypatch.setattr(sandbox, "_ROOT_SCAN_SETTLE_NS", 10**18)
    run()
    assert not cache_dir.exists() or not list(cache_dir.glob("*.json"))


def test_a_cache_dir_inside_what_cells_see_is_not_used(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(workspace / "cache"))
    assert sandbox._scan_cache_dir([workspace]) is None
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
    assert sandbox._scan_cache_dir() == Path(
        os.path.realpath(Path.home() / ".cache" / "unify" / "sandbox-scans"),
    )


def test_a_home_cache_is_refused_as_a_root_and_as_a_workspace(tmp_path, monkeypatch):
    """No cell can write the default cache directory: a home's ``.cache`` is
    never a mounted root or the workspace, and a root or workspace inside it
    that holds the cache directory turns the disk cache off."""
    # The account's own home (a home under /tmp is not counted as one); the
    # refusals only compare paths, nothing there is read or made.
    account = Path(os.path.realpath(sandbox._pwd_home()))
    assert sandbox._root_refusal(account / ".cache") is not None
    assert sandbox._root_refusal(account) is not None
    assert sandbox._workspace_refusal(account / ".cache") is not None
    home = tmp_path / "home"
    (home / ".cache" / "unify" / "sandbox-scans").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert sandbox._scan_cache_dir([home / ".cache" / "unify"]) is None
    assert sandbox._scan_cache_dir([home / "workspace"]) == Path(
        os.path.realpath(home / ".cache" / "unify" / "sandbox-scans"),
    )
