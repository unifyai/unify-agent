import sys
import os
import re
import pytest
from pathlib import Path
from typing import Optional
from datetime import datetime


def _enter_test_sandbox() -> None:
    """Re-run this pytest process inside the test sandbox (tests/_test_sandbox.py).

    First, before anything is imported or written: model-written code runs in
    this process, and the sandbox decides what it can read.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "tests" / "_test_sandbox.py"
    spec = importlib.util.spec_from_file_location("_unify_test_sandbox", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.enter(path.parent.parent)


_enter_test_sandbox()

try:  # pragma: no cover - defensive logging hygiene
    import logging

    logging.getLogger("faker").setLevel(logging.WARNING)
    logging.getLogger("faker.factory").setLevel(logging.WARNING)
    logging.getLogger("faker.providers").setLevel(logging.WARNING)
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Early Environment Setup (MUST be before any unify/unify imports)
# ─────────────────────────────────────────────────────────────────────────────
# Pytest establishes a per-test context root; ``SETTINGS.TEST`` is what makes
# the runtime honour it instead of the session identity.
os.environ.setdefault("TEST", "true")

# Set UNILLM_CACHE_DIR to use the MAIN repo's cache, not the worktree's.
# This ensures all worktrees share the same LLM cache (.cache.ndjson) for
# consistent cache hits. This must happen before unillm is imported because
# the cache directory is captured at class definition time.
if "UNILLM_CACHE_DIR" not in os.environ:
    repo_root = Path(__file__).resolve().parent
    git_path = repo_root / ".git"
    # Check if we're in a worktree (.git is a file, not a directory)
    if git_path.is_file():
        try:
            # .git file contains: "gitdir: /path/to/main/.git/worktrees/name"
            gitdir_line = git_path.read_text().strip()
            if gitdir_line.startswith("gitdir:"):
                gitdir = gitdir_line[7:].strip()
                # Go up from .git/worktrees/name to main repo root
                main_repo = Path(gitdir).parent.parent.parent
                if main_repo.exists():
                    repo_root = main_repo
        except Exception:
            pass  # Fall back to current repo root
    os.environ["UNILLM_CACHE_DIR"] = str(repo_root)

# Every pytest process owns its own store, so sessions running side by side
# (one tmux session per test under parallel_run.sh) never share tables and
# nothing touches the real ``~/.unify``. The process home only covers what
# runs outside a test: each test gets a home of its own from ``unify_home``
# in tests/conftest.py. The embeddings cache is the one file shared across
# processes: embeddings are keyed by (model, text) and cost real compute, so
# reusing them between sessions is pure gain. Set before anything imports
# ``unify.db``, which opens the store lazily on first use.
import tempfile

_TEST_TMP = Path(tempfile.gettempdir())
_TEST_STORE_DIR = _TEST_TMP / "unity_test_stores"
_TEST_STORE_DIR.mkdir(parents=True, exist_ok=True)
_TEST_HOME_DIR = _TEST_TMP / "unity_test_home"
_TEST_HOME_DIR.mkdir(parents=True, exist_ok=True)
_OWNS_STORE = "UNIFY_STORE_PATH" not in os.environ
_OWNS_UNIFY_HOME = "UNIFY_HOME" not in os.environ
os.environ.setdefault(
    "UNIFY_STORE_PATH",
    str(_TEST_STORE_DIR / f"unify_test_{os.getpid()}.sqlite"),
)
os.environ.setdefault(
    "UNIFY_HOME",
    str(_TEST_STORE_DIR / f"unify_home_{os.getpid()}"),
)
os.makedirs(os.environ["UNIFY_HOME"], exist_ok=True)
os.environ.setdefault("UNIFY_EMBED_CACHE", str(_TEST_HOME_DIR / "embeddings.sqlite"))

from unify.settings import SETTINGS

_TEE_FILE_HANDLE: Optional[object] = None
_TEE_ORIG_STREAM: Optional[object] = None
_TEE_STREAM_ATTR: Optional[str] = None
_TEE_LOG_PATH: Optional[Path] = None


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Log Naming Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _path_to_name(path: str) -> str:
    """Convert a test path to a filename-safe string.

    Examples:
        tests/function_manager/storage/test_venvs.py → function_manager-storage-test_venvs
        test_foo.py → test_foo
        /abs/path/to/workspace/tests/foo.py → foo
    """
    name = path.rstrip("/\\")

    # Handle absolute paths: strip workspace root first
    path_obj = Path(name)
    if path_obj.is_absolute():
        # Try to make it relative to the workspace root
        try:
            workspace_root = Path(__file__).parent  # conftest.py is at workspace root
            name = str(path_obj.relative_to(workspace_root))
        except ValueError:
            # Path is not relative to workspace; use as-is
            pass

    # Strip common prefixes
    for prefix in ("tests/", "tests\\", "./", ".\\"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    # Strip .py suffix
    if name.endswith(".py"):
        name = name[:-3]
    # Replace path separators with dashes
    return name.replace("/", "-").replace("\\", "-")


def _sanitize_filename(name: str, max_length: int = 200) -> str:
    """Sanitize a string to be safe for use in filenames on all platforms.

    Removes/replaces characters that are invalid on Windows/NTFS:
    - Double quote "
    - Colon :
    - Less than <
    - Greater than >
    - Vertical bar |
    - Asterisk *
    - Question mark ?
    - Carriage return \r
    - Line feed \n
    - Backslash \\ (path separator on Windows)

    Also truncates long names while preserving uniqueness via hash suffix.

    Args:
        name: The string to sanitize
        max_length: Maximum length for the resulting filename (default 200)

    Returns:
        Sanitized filename safe for all platforms
    """
    import re
    import hashlib

    # Replace invalid characters with underscore or dash
    # Also replace / which appears in test params like text/plain
    name = re.sub(r'["\:<>|*?\r\n\\/]', "_", name)
    # Collapse multiple underscores/dashes
    name = re.sub(r"[_-]{2,}", "-", name)
    # Remove leading/trailing underscores/dashes
    name = name.strip("_-")

    # If name exceeds max_length, truncate and add hash for uniqueness
    if len(name) > max_length:
        # Create a short hash of the full name
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        # Truncate to leave room for hash suffix (hash + underscore)
        truncate_at = max_length - 9  # Leave room for "_" + 8-char hash
        name = f"{name[:truncate_at]}_{name_hash}"

    return name


def _derive_multi_target_name(paths: list) -> str:
    """Derive a meaningful name when multiple test targets are provided.

    Strategy:
    1. If all targets are from the same .py file, use file name + first test + count
    2. If all targets share a common directory, use that + first file/test + count
    3. Otherwise, use first target + count
    """
    # Extract base file (before ::) for each path
    bases = []
    nodes = []
    for p in paths:
        if "::" in p:
            base, node = p.split("::", 1)
            bases.append(base)
            nodes.append(node)
        else:
            bases.append(p)
            nodes.append(None)

    # Check if all from the same file
    unique_bases = set(bases)
    if len(unique_bases) == 1:
        # All targets from the same file
        base_name = _path_to_name(bases[0])
        if nodes[0]:
            # Sanitize node: replace :: with -, remove brackets, sanitize for filename
            first_node = nodes[0].replace("::", "-").replace("[", "-").replace("]", "")
            first_node = _sanitize_filename(first_node)
            # Truncate if too long
            if len(first_node) > 40:
                first_node = first_node[:37] + "..."
            extra = len(paths) - 1
            combined = (
                f"{base_name}--{first_node}+{extra}more"
                if extra > 0
                else f"{base_name}--{first_node}"
            )
            return _sanitize_filename(combined, max_length=200)
        extra = len(paths) - 1
        combined = f"{base_name}+{extra}more" if extra > 0 else base_name
        return _sanitize_filename(combined, max_length=200)

    # Check for common directory prefix
    def get_dir(p: str) -> str:
        # Get directory part of a path
        if "/" in p:
            return p.rsplit("/", 1)[0]
        if "\\" in p:
            return p.rsplit("\\", 1)[0]
        return ""

    dirs = [get_dir(b) for b in bases]
    unique_dirs = set(dirs)
    if len(unique_dirs) == 1 and dirs[0]:
        # All from same directory
        dir_name = _path_to_name(dirs[0])
        first_file = _path_to_name(bases[0])
        # Remove directory prefix from first_file for brevity
        if first_file.startswith(dir_name + "-"):
            first_file = first_file[len(dir_name) + 1 :]
        extra = len(paths) - 1
        combined = (
            f"{dir_name}--{first_file}+{extra}more"
            if extra > 0
            else f"{dir_name}--{first_file}"
        )
        return _sanitize_filename(combined, max_length=200)

    # Fallback: use first target + count
    first_name = _path_to_name(paths[0].split("::")[0])
    if "::" in paths[0]:
        node = paths[0].split("::", 1)[1]
        node = node.replace("::", "-").replace("[", "-").replace("]", "")
        node = _sanitize_filename(node)
        if len(node) > 30:
            node = node[:27] + "..."
        first_name = f"{first_name}--{node}"
    extra = len(paths) - 1
    combined = f"{first_name}+{extra}more" if extra > 0 else first_name
    return _sanitize_filename(combined, max_length=200)


def _derive_log_name_from_args(args: list) -> str:
    """Derive a semantic log filename from pytest command-line args.

    Examples:
        ['tests/function_manager/storage/test_venvs.py']
            → 'function_manager-storage-test_venvs'
        ['tests/function_manager/storage/test_venvs.py::test_foo']
            → 'function_manager-storage-test_venvs--test_foo'
        ['tests/function_manager/']
            → 'function_manager'
        ['tests/']
            → 'tests'
        [] (no args)
            → 'all'
        Multiple from same file:
            → 'test_session_behavior--TestA-test_x+1more'
        Multiple from same directory:
            → 'function_manager--storage-test_venvs+2more'
    """
    if not args:
        return "all"

    # Filter to actual test paths (ignore flags like -v, --tb=short, etc.)
    paths = []
    for a in args:
        if isinstance(a, str) and not a.startswith("-"):
            # Looks like a path or node id
            if a.endswith(".py") or "::" in a or "/" in a or "\\" in a:
                paths.append(a)
            elif os.path.exists(a):
                paths.append(a)

    if not paths:
        return "all"

    # For single target, use detailed naming
    if len(paths) == 1:
        target = paths[0]
        # Handle pytest node ids (path::test_name or path::Class::test_name)
        if "::" in target:
            base, node = target.split("::", 1)
            base_name = _path_to_name(base)
            # Sanitize node: replace :: with -, remove brackets, sanitize for filename
            node_name = node.replace("::", "-").replace("[", "-").replace("]", "")
            node_name = _sanitize_filename(node_name)
            # Combine and ensure total length is within limits (accounting for .txt extension)
            combined = f"{base_name}--{node_name}"
            # Truncate combined name if needed (max 200 chars to leave room for path/extension)
            return _sanitize_filename(combined, max_length=200)
        return _path_to_name(target)

    # Multiple targets: find common structure for a meaningful name
    return _derive_multi_target_name(paths)


def _derive_socket_name() -> str:
    """Derive a unique socket name from the terminal's TTY device.

    Mirrors the logic in tests/_shell_common.sh::_derive_socket_name() to ensure
    consistent naming whether tests are run via parallel_run.sh or directly via pytest.

    Returns:
        - 'unity_dev_ttysXXX' if running in a TTY (e.g., terminal session)
        - 'unity_pidXXX' if not running in a TTY (e.g., background process)
    """
    try:
        # Try to get the TTY device path (e.g., '/dev/ttys042')
        tty_path = os.ttyname(sys.stdout.fileno())
        # Sanitize: /dev/ttys042 -> unity_dev_ttys042
        tty_id = tty_path.replace("/", "_")
        return f"unity{tty_id}"
    except (OSError, AttributeError):
        # Not a TTY (e.g., piped output, background process)
        return f"unity_pid{os.getpid()}"


def _get_log_subdir() -> str:
    """Determine the log subdirectory for pytest logs.

    Returns a datetime-prefixed directory name for natural time-based ordering:
        - UNIFY_LOG_SUBDIR if set (e.g., '2025-12-05T14-30-45_unity_dev_ttys042')
        - Falls back to UNIFY_TEST_SOCKET for legacy compatibility
        - Derives terminal ID for direct pytest invocations (same as parallel_run.sh would)
    """
    # Prefer the datetime-prefixed log subdir if available
    log_subdir = os.environ.get("UNIFY_LOG_SUBDIR", "").strip()
    if log_subdir:
        return log_subdir
    # Fallback to socket name for backward compatibility
    socket = os.environ.get("UNIFY_TEST_SOCKET", "").strip()
    if socket:
        return socket
    # Derive terminal ID (same logic as _shell_common.sh) for direct pytest invocations
    socket_name = _derive_socket_name()
    return f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{socket_name}"


def _get_log_root(config_rootpath: Path) -> Path:
    """Determine the root directory for pytest logs.

    Prefers UNIFY_LOG_ROOT env var if set, allowing explicit worktree targeting.
    Otherwise derives from this file's location, which correctly resolves to
    the worktree when running from one.

    This fixes the issue where Cursor Background Agents (which use git worktrees)
    would have logs written to the main repo instead of their worktree.
    """
    # Allow explicit override for flexibility
    log_root = os.environ.get("UNIFY_LOG_ROOT", "").strip()
    if log_root:
        return Path(log_root)

    # Derive repo root from this file's location (works correctly in worktrees)
    # __file__ is conftest.py at repo root
    try:
        return Path(__file__).resolve().parent
    except Exception:
        # Fallback to pytest's rootpath if __file__ resolution fails
        return config_rootpath


# ─────────────────────────────────────────────────────────────────────────────
# Worktree Log Symlink Management
# ─────────────────────────────────────────────────────────────────────────────
# When running tests from a git worktree, create symlinks in the main repo's
# log directories pointing to this worktree's logs. This lets you browse all
# worktree logs from a single location (the main repo).


def _is_git_worktree(repo_root: Path) -> bool:
    """Check if repo_root is a git worktree (not the main repo).

    In a worktree, .git is a file containing 'gitdir: /path/to/main/.git/worktrees/name'.
    In the main repo, .git is a directory.
    """
    git_path = repo_root / ".git"
    return git_path.is_file()


def _get_main_repo_path(repo_root: Path) -> Optional[Path]:
    """Get the main (non-worktree) repo path from a worktree.

    Reads the .git file to find the main repo's .git directory,
    then returns its parent.
    """
    git_file = repo_root / ".git"
    if not git_file.is_file():
        return None

    try:
        content = git_file.read_text().strip()
        # Format: "gitdir: /path/to/main/.git/worktrees/name"
        if content.startswith("gitdir:"):
            gitdir = content[7:].strip()
            # gitdir is like /main/repo/.git/worktrees/worktree-name
            # We want /main/repo (parent of .git, which is 3 levels up from gitdir)
            gitdir_path = Path(gitdir)
            # Go up: worktrees/name -> worktrees -> .git -> repo
            main_git = gitdir_path.parent.parent
            if main_git.name == ".git":
                return main_git.parent
    except Exception:
        pass
    return None


def _get_worktree_name(repo_root: Path) -> str:
    """Get a descriptive name for the worktree (used in symlink names)."""
    return repo_root.name


def _ensure_worktree_log_symlinks(repo_root: Path) -> None:
    """Create symlinks in main repo's log directories pointing to worktree's logs.

    Creates symlinks like:
        /main/repo/logs/pytest/worktree-oty -> /path/to/worktree/oty/logs/pytest
        /main/repo/logs/unillm/worktree-oty -> /path/to/worktree/oty/logs/unillm

    This lets you browse all worktree logs from the main repo.
    """
    if not _is_git_worktree(repo_root):
        return

    main_repo = _get_main_repo_path(repo_root)
    if main_repo is None:
        return

    worktree_name = _get_worktree_name(repo_root)

    for log_subdir in ("pytest", "unillm"):
        main_log_dir = main_repo / "logs" / log_subdir
        worktree_log_dir = repo_root / "logs" / log_subdir
        symlink_path = main_log_dir / f"worktree-{worktree_name}"

        try:
            # Ensure both directories exist
            main_log_dir.mkdir(parents=True, exist_ok=True)
            worktree_log_dir.mkdir(parents=True, exist_ok=True)

            # Create or update symlink
            if symlink_path.is_symlink():
                # Check if it points to the right place
                if symlink_path.resolve() != worktree_log_dir.resolve():
                    symlink_path.unlink()
                    symlink_path.symlink_to(worktree_log_dir)
            elif not symlink_path.exists():
                symlink_path.symlink_to(worktree_log_dir)
            # If something else exists at that path, leave it alone
        except Exception:
            # Symlink creation is best-effort; don't fail tests if it errors
            pass


class _TeeStream:
    # Regex to strip ANSI escape sequences (colors, cursor movement, etc.)
    _ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, s):
        # Strip ANSI codes from file output for clean logs
        clean = self._ANSI_ESCAPE_RE.sub("", s)
        self._mirror.write(clean)
        self._mirror.flush()
        return self._primary.write(s)

    def flush(self):
        self._mirror.flush()
        return self._primary.flush()

    def isatty(self):
        fn = getattr(self._primary, "isatty", None)
        return bool(fn and fn())

    def fileno(self):
        fn = getattr(self._primary, "fileno", None)
        return fn() if fn else -1

    def writable(self):
        fn = getattr(self._primary, "writable", None)
        return bool(fn and fn())

    @property
    def closed(self):
        return getattr(self._primary, "closed", False)


def pytest_sessionstart(session):
    # Configure file-based logging directories.
    # Unity LOGGER output goes to pytest stdout (captured in logs/pytest/),
    # so we don't configure a separate logs/unity/ directory during tests.
    root_path = _get_log_root(Path(session.config.rootpath))
    subdir = _get_log_subdir()

    # Unillm LLM I/O file logging (raw request/response traces)
    unillm_log_dir = root_path / "logs" / "unillm" / subdir
    unillm_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from unillm import configure_log_dir as configure_unillm_log_dir

        configure_unillm_log_dir(str(unillm_log_dir))
    except ImportError:
        os.environ["UNILLM_LOG_DIR"] = str(unillm_log_dir)

    if not SETTINGS.PYTEST_LOG_TO_FILE:
        return
    config = session.config
    if getattr(config.option, "collectonly", False):
        # Skip logging for test discovery runs to avoid huge collection trees
        return
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is None:
        return

    # Use worktree-aware log root instead of pytest's rootpath
    root_path = _get_log_root(Path(config.rootpath))

    # If running from a worktree, create symlinks in main repo for easy log browsing
    _ensure_worktree_log_symlinks(root_path)

    # Determine subdirectory based on terminal context
    # Directory names are datetime-prefixed for natural time-based ordering
    subdir = _get_log_subdir()
    logs_dir = root_path / "logs" / "pytest" / subdir
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Derive semantic name from command-line args
    semantic_name = _derive_log_name_from_args(list(config.args))

    # Open a unique file using exclusive creation to avoid races across concurrent runs
    def _open_unique(path_no_suffix: Path):
        suffix_index = 0
        while True:
            name = f"{path_no_suffix.stem}{'' if suffix_index == 0 else f' ({suffix_index})'}{path_no_suffix.suffix}"
            candidate = path_no_suffix.with_name(name)
            try:
                fh = open(candidate, mode="x", encoding="utf-8")
                return candidate, fh
            except FileExistsError:
                suffix_index += 1

    # Build the log filename (subdir is always datetime-prefixed now)
    base_path = logs_dir / f"{semantic_name}.txt"
    log_path, fh = _open_unique(base_path)

    global _TEE_FILE_HANDLE, _TEE_ORIG_STREAM, _TEE_STREAM_ATTR
    _TEE_FILE_HANDLE = fh
    global _TEE_LOG_PATH
    _TEE_LOG_PATH = log_path

    # Mirror the IDE runner's pre-launch banner into the file (file-only, no terminal dup).
    _TEE_FILE_HANDLE.write(f"Running pytest with args: {sys.argv[1:]}\n")
    _TEE_FILE_HANDLE.flush()

    # Identify the TerminalWriter's underlying stream attribute.
    if hasattr(tr._tw, "_file"):
        _TEE_STREAM_ATTR = "_file"
    elif hasattr(tr._tw, "file"):
        _TEE_STREAM_ATTR = "file"
    else:
        _TEE_STREAM_ATTR = None

    if _TEE_STREAM_ATTR is None:
        return

    _TEE_ORIG_STREAM = getattr(tr._tw, _TEE_STREAM_ATTR)
    setattr(tr._tw, _TEE_STREAM_ATTR, _TeeStream(_TEE_ORIG_STREAM, _TEE_FILE_HANDLE))

    # Test body prints are captured by pytest; we mirror them via pytest_runtest_logreport.


def _remove_process_store() -> None:
    """Delete the store this process created, if it created one.

    A store handed in through ``UNIFY_STORE_PATH`` belongs to whoever set it
    and is left alone.
    """
    import shutil

    from unify import db

    db.reset_store()
    if _OWNS_STORE:
        Path(os.environ["UNIFY_STORE_PATH"]).unlink(missing_ok=True)
    if _OWNS_UNIFY_HOME:
        shutil.rmtree(os.environ["UNIFY_HOME"], ignore_errors=True)


def pytest_unconfigure(config):
    """Print the log file path after pytest's own terminal summary has been emitted."""
    _remove_process_store()
    if not SETTINGS.PYTEST_LOG_TO_FILE:
        return
    global _TEE_FILE_HANDLE, _TEE_ORIG_STREAM, _TEE_STREAM_ATTR, _TEE_LOG_PATH
    tr = config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None and _TEE_FILE_HANDLE is not None:
        # Use worktree-aware log root for consistent path display
        root_path = _get_log_root(Path(config.rootpath))
        log_file = (
            _TEE_LOG_PATH or (root_path / "logs" / "pytest" / "unknown.txt")
        ).resolve()
        subdir = _get_log_subdir()

        # Print a clear banner showing where logs are
        tr.write_line("")
        tr.write_line("=" * 72)
        tr.write_line(f"📄 Test log: {log_file}")
        tr.write_line(
            f"📁 This run's logs: {root_path / 'logs' / 'pytest' / subdir}/",
        )
        tr.write_line(f"📂 LLM I/O logs:     {root_path / 'logs' / 'unillm' / subdir}/")
        tr.write_line(f"📂 All log directories:  {root_path / 'logs'}/*/")
        tr.write_line("=" * 72)
    # Append a file-only trailer to match the IDE runner's banner.
    if _TEE_FILE_HANDLE is not None:
        _TEE_FILE_HANDLE.write("Finished running tests!\n")
        _TEE_FILE_HANDLE.flush()
    if _TEE_FILE_HANDLE is not None:
        _TEE_FILE_HANDLE.flush()
        _TEE_FILE_HANDLE.close()
        _TEE_FILE_HANDLE = None
    if _TEE_ORIG_STREAM is not None and _TEE_STREAM_ATTR is not None:
        tr = config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            setattr(tr._tw, _TEE_STREAM_ATTR, _TEE_ORIG_STREAM)
        _TEE_ORIG_STREAM = None
        _TEE_STREAM_ATTR = None

    _TEE_LOG_PATH = None
    # No sys.stdout/sys.stderr monkeypatch remains; nothing to restore here.


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    """Ensure captured stdout/stderr from tests is mirrored into the file.

    With --capture=tee-sys, prints are shown live in terminal but may not reach our
    TerminalReporter tee due to capture internals. Here we append the captured
    streams from the test's call phase directly to the file to guarantee inclusion.
    """
    if not SETTINGS.PYTEST_LOG_TO_FILE:
        return
    # Only mirror the main call phase where test body runs
    if getattr(report, "when", None) != "call":
        return
    global _TEE_FILE_HANDLE
    if _TEE_FILE_HANDLE is None:
        return
    out = []
    capout = getattr(report, "capstdout", "")
    caperr = getattr(report, "capstderr", "")
    if capout:
        out.append(capout)
    if caperr:
        out.append(caperr)
    if out:
        _TEE_FILE_HANDLE.write("".join(out))
        _TEE_FILE_HANDLE.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Log capture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def unify_logs(caplog):
    """Capture ``unify.*`` log records for assertions.

    Use this instead of ``capsys``/``capfd`` for anything the unify LOGGER
    emits. Two reasons it has to work this way:

    * The logger owns its own terminal formatting and does not propagate to
      root, so pytest's ``caplog`` handler reaches it only when attached
      directly.
    * The terminal handler binds ``sys.stdout`` when ``unify.logger`` is
      first imported. Capture fixtures replace stdout per test, so whether a
      record lands in ``capsys``/``capfd`` depends on which capture happened
      to be active at import — an assertion that passes alone and fails in a
      full run. Records carry no such dependency.

    Yields the ``caplog`` fixture, so ``.text``, ``.records`` and
    ``.messages`` work as usual.
    """
    from unify.logger import LOGGER

    LOGGER.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        LOGGER.removeHandler(caplog.handler)
