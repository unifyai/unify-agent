"""Symbolic: tests run sandboxed, a test's time limit ends it, a test can
insist on fresh model calls, and the clocks a prompt can show are frozen.

The test sandbox (tests/_test_sandbox.py) re-executes every pytest process
inside bubblewrap; these tests check, from inside it, what code a model
writes can see, by running such code the way the actor runs a Python cell.
The time-limit tests (tests/_test_timeouts.py) run a pytest of their own on a
throwaway test file and check how it ends.
"""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# Imported here, before any test's fixtures run, so these modules bind the
# real prompt clock by name the way a test session's first import does.
import unify.conversation_manager.conversation_manager  # noqa: F401
import unify.conversation_manager.domains.brain_action_tools as brain_action_tools
from unify.actor.execution.session import SessionExecutor
from unify.conversation_manager.domains.renderer import Renderer

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


# ── time limits ─────────────────────────────────────────────────────────────


@pytest.mark.timeout(123)
def test_the_marker_sets_the_limit(request):
    from tests._test_timeouts import limit_for

    assert limit_for(request.node) == 123


def test_an_unmarked_test_gets_the_default_limit(request, monkeypatch):
    from tests._test_timeouts import limit_for

    monkeypatch.delenv("UNIFY_TEST_TIMEOUT", raising=False)
    assert limit_for(request.node) == 900
    monkeypatch.setenv("UNIFY_TEST_TIMEOUT", "0")
    assert limit_for(request.node) is None


def _run_pytest(tmp_path: Path, body: str, *, grace: float) -> tuple[int, str, float]:
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    timeout\n")
    (tmp_path / "test_limited.py").write_text(textwrap.dedent(body))
    env = {
        **os.environ,
        "UNIFY_TEST_TIMEOUT_GRACE": str(grace),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("UNIFY_TEST_TIMEOUTS", None)
    started = time.monotonic()
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests._test_timeouts",
            "-p",
            "no:cacheprovider",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--rootdir",
            str(tmp_path),
            str(tmp_path / "test_limited.py"),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return done.returncode, done.stdout + done.stderr, time.monotonic() - started


def test_a_test_past_its_limit_fails_and_the_next_one_runs(tmp_path):
    status, output, _ = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.timeout(1)
        def test_spins():
            while True:
                pass

        def test_after():
            pass
        """,
        grace=30,
    )
    assert status == 1, output
    assert (
        "Timeout: the call of test_limited.py::test_spins took more than 1s" in output
    )
    assert "1 failed, 1 passed" in output


def test_a_test_that_swallows_the_timeout_is_killed(tmp_path):
    # Code that catches every exception (as model-written code may) keeps
    # running past the soft limit; the watchdog ends the process anyway.
    status, output, elapsed = _run_pytest(
        tmp_path,
        """
        import time
        import pytest

        @pytest.mark.timeout(1)
        def test_never_stops():
            while True:
                try:
                    time.sleep(0.05)
                except BaseException:
                    pass
        """,
        grace=2,
    )
    assert status == 1, output
    assert "Timeout (0:00:03)!" in output
    assert "test_never_stops" in output
    assert elapsed < 60


# ── fresh model calls ───────────────────────────────────────────────────────


@pytest.mark.fresh_llm_calls
def test_a_fresh_calls_test_builds_clients_without_the_cache():
    import unillm

    assert unillm.SETTINGS.UNILLM_CACHE is False
    assert os.environ["UNILLM_CACHE"] == "false"
    from unify.common.llm_client import new_llm_client

    assert new_llm_client().cache is False


@pytest.mark.parametrize(
    "name",
    [
        "test_pause_resume_inflight_handle",
        "test_two_concurrent_handles_pause_one_other_completes",
    ],
)
def test_the_steering_pause_tests_always_call_the_model(name):
    from tests.conversation_manager.actions.integration import test_steerability

    marks = {m.name for m in getattr(test_steerability, name).pytestmark}
    assert "fresh_llm_calls" in marks


# ── frozen clocks ───────────────────────────────────────────────────────────

FROZEN_PROMPT_TIME = "Friday, June 13, 2025 at 12:00 PM UTC"


def test_every_by_name_copy_of_the_prompt_clock_is_frozen():
    """A module that imports the prompt clock by name holds its own copy;
    any copy left on the real clock puts the wall-clock time in a prompt
    and misses the LLM cache on every run."""
    leaks = []
    for name, module in list(sys.modules.items()):
        if not (name == "unify" or name.startswith("unify.")):
            continue
        for attr, value in list(getattr(module, "__dict__", {}).items()):
            if (
                getattr(value, "__module__", None) == "unify.common.prompt_helpers"
                and getattr(value, "__qualname__", None) == "now"
            ):
                leaks.append(f"{name}.{attr}")
    assert leaks == []
    assert brain_action_tools.prompt_now() == FROZEN_PROMPT_TIME


def test_an_action_history_event_shows_the_frozen_time():
    history = Renderer._render_action_history(
        [
            {
                "action_name": "act_started",
                "query": "find the report",
                "timestamp": brain_action_tools.prompt_now(),
            },
        ],
        short_name="act",
        handle_id=0,
        max_history=5,
    )
    assert f"<event type='act_started' timestamp='{FROZEN_PROMPT_TIME}'>" in history
