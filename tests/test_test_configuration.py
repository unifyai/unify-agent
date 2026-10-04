"""Symbolic: tests run sandboxed.

The test sandbox (tests/_test_sandbox.py) re-executes every pytest process
inside bubblewrap; these tests check, from inside it, what code a model
writes can see, by running such code the way the actor runs a Python cell.
"""

from __future__ import annotations

import os
import pwd
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from unify.actor.execution.session import SessionExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent
# The account's real home: HOME is the test home inside a session.
REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir

sandboxed = pytest.mark.skipif(
    os.environ.get("UNIFY_TEST_SANDBOXED") != "1",
    reason="this session runs without the test sandbox",
)

# What a model asked to "search my notes files" wrote on 2 Oct, reduced to
# the reads that matter: list the home directory and the Windows drives, walk
# them, and ask a subprocess to do the same.
MODEL_CODE = textwrap.dedent(
    f"""
    import os, subprocess
    seen = {{}}
    for path in (os.path.expanduser("~"), {REAL_HOME!r}, "/mnt/c", "/mnt/d"):
        try:
            seen[path] = sorted(os.listdir(path))
        except OSError as error:
            seen[path] = type(error).__name__
    walked = []
    for root in ("/mnt", {REAL_HOME!r}):
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in (".venv", ".git", "logs")]
            walked += [os.path.join(d, f) for f in files if f.endswith((".md", ".docx"))]
    shell = subprocess.run(
        ["ls", "/mnt/c", {REAL_HOME + "/.ssh"!r}], capture_output=True, text=True
    )
    (seen, walked, shell.returncode, shell.stderr)
    """,
)


def test_a_session_runs_sandboxed_where_bubblewrap_exists():
    if os.environ.get("UNIFY_TEST_SANDBOX", "auto") == "off":
        pytest.skip("UNIFY_TEST_SANDBOX=off")
    if not sys.platform.startswith("linux") or shutil.which("bwrap") is None:
        pytest.skip("no bubblewrap")
    assert os.environ.get("UNIFY_TEST_SANDBOXED") == "1"


@sandboxed
@pytest.mark.asyncio
async def test_model_code_cannot_list_the_home_directory_or_the_windows_drives():
    ex = SessionExecutor()
    try:
        res = await ex.execute(code=MODEL_CODE, state_mode="stateless", session_id=None)
    finally:
        await ex.close()
    assert res["error"] is None, res["error"]
    seen, walked, ls_status, ls_error = res["result"]

    assert seen["/mnt/c"] == "FileNotFoundError"
    assert seen["/mnt/d"] == "FileNotFoundError"
    # The real home holds only the directories leading to what the tests
    # need: the checkouts, the interpreter and the shared LLM cache.
    home = seen[REAL_HOME]
    assert isinstance(home, list), home
    assert not {".ssh", ".config", ".aws", ".gnupg", ".bashrc", ".claude"} & set(home)
    assert set(home) <= {".local", "unify-agent", *(p.name for p in _home_parents())}
    # The walk finds only the checkouts' own files, never a document of the
    # account's; the main checkout's files are there by name, and empty.
    main = _load_sandbox()._cache_dir(REPO_ROOT)
    visible = tuple(str(p) for p in (*_home_parents(), main))
    for path in walked:
        assert path.startswith(visible), path
        if path.startswith(str(main) + "/") and not path.startswith(str(REPO_ROOT)):
            assert os.path.getsize(path) == 0, path
    assert not any(p.startswith("/mnt/c") or p.startswith("/mnt/d") for p in walked)
    # A subprocess the code starts sees the same.
    assert ls_status != 0
    assert "No such file or directory" in ls_error


def _home_parents() -> list[Path]:
    """The visible roots under the real home that contain checkouts."""
    roots = []
    for path in (REPO_ROOT, *(Path(p) for p in sys.path if p)):
        try:
            rel = path.resolve().relative_to(REAL_HOME)
        except ValueError:
            continue
        roots.append(Path(REAL_HOME) / rel.parts[0])
    return roots


@sandboxed
@pytest.mark.asyncio
async def test_model_code_cannot_write_the_checkout_or_read_the_main_checkouts_files():
    code = textwrap.dedent(
        f"""
        import os
        out = {{}}
        try:
            open({str(REPO_ROOT / "unify" / "sandbox_probe.py")!r}, "w").write("x")
            out["write"] = "written"
        except OSError as error:
            out["write"] = type(error).__name__
        main = os.path.join({REAL_HOME!r}, "unify-agent")
        out["main"] = {{
            name: (os.path.isdir(os.path.join(main, name))
                   and os.listdir(os.path.join(main, name)))
            for name in ("unify", ".git", ".venv")
            if os.path.exists(os.path.join(main, name))
        }}
        out
        """,
    )
    ex = SessionExecutor()
    try:
        res = await ex.execute(code=code, state_mode="stateless", session_id=None)
    finally:
        await ex.close()
    assert res["error"] is None, res["error"]
    out = res["result"]
    assert out["write"] == "OSError"
    assert not (REPO_ROOT / "unify" / "sandbox_probe.py").exists()
    # The main checkout's directories, if this is a worktree, are empty.
    assert all(listing in ([], False) for listing in out["main"].values()), out


def test_the_sandbox_mounts_neither_the_home_directory_nor_the_drives():
    sandbox = _load_sandbox()
    args = sandbox.bwrap_args(REPO_ROOT, REPO_ROOT)
    mounted = {
        args[i + 2]
        for i, a in enumerate(args)
        if a in ("--bind", "--ro-bind") and i + 2 < len(args)
    }
    for forbidden in (REAL_HOME, "/home", "/mnt", "/mnt/c", "/mnt/d", "/", "/run"):
        assert forbidden not in mounted
    assert "--die-with-parent" in args and "--unshare-pid" in args
    writable = {args[i + 2] for i, a in enumerate(args) if a == "--bind"}
    assert str(REPO_ROOT) not in writable or sandbox._cache_dir(REPO_ROOT) == REPO_ROOT
    assert str(REPO_ROOT / "logs") in writable


def _load_sandbox():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_test_sandbox_under_test",
        REPO_ROOT / "tests" / "_test_sandbox.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
