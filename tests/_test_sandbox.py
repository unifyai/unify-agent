"""Run each pytest process inside bubblewrap, so model-written code cannot read the machine.

Tests drive real models whose code runs in this process (the actor ``exec``s a
Python cell in its own event loop). On 2 Oct such code, asked to "search my
notes files", walked ``/home``, ``/mnt/c`` and ``/mnt/d`` for over twenty
minutes. Nothing a test does needs to see beyond the checkout, so the root
conftest calls :func:`enter` before it imports anything else, and the pytest
process re-executes itself inside a bubblewrap sandbox in which:

* the root is an empty tmpfs; ``/usr``, ``/etc`` and the other system
  directories are mounted read-only; ``/home``, ``/root``, ``/mnt`` (the
  Windows drives under WSL), ``/media``, ``/srv``, ``/opt``, ``/run`` and
  ``/sys`` do not exist, except for the paths below (and WSL's
  ``/mnt/wsl/resolv.conf``, which ``/etc/resolv.conf`` links to);
* the checkout is read-only, except ``logs/`` and ``.pytest_cache/``
  (bytecode goes to ``logs/.test-sandbox/pycache``, uv's cache to
  ``logs/.test-sandbox/uv``); the interpreter installation and any editable
  install the venv points outside the checkout (``sys.path``) are read-only,
  and so are ``uv`` and ``rg``; a worktree's ``.git`` file leads to an empty
  repository, since its metadata lives in the hidden main checkout;
* the checkout's ``.env`` (or the file it links to) is read-only, since the
  harness reads its provider keys from it, as it does outside the sandbox;
* the shared LLM cache (``UNILLM_CACHE_DIR``: the main checkout of a
  worktree) stays writable, since its index is rewritten through a temporary
  file beside it; every other entry of that directory is hidden, directories
  behind an empty tmpfs and files behind ``/dev/null``. When the checkout is
  the main checkout, it is writable and each of its entries read-only;
* ``/tmp`` is private, except the two directories every test session shares
  with the others: the test home with its embeddings cache
  (``/tmp/unity_test_home``) and tiktoken's encodings (``/tmp/data-gym-cache``);
* process, IPC and hostname namespaces are private, the session has no
  controlling terminal, and the sandbox dies with its parent: when the pytest
  process ends or is killed, every process a test started ends with it.

Credential files under the home directory (``~/.ssh``, ``~/.config`` with its
gcloud keys, ``~/.aws``) are therefore out of reach, which also means unillm
cannot fall back to Google Secret Manager for a provider key missing from the
environment: that fallback needs the service-account key in ``~/.config``,
and it ran as a network call at import in every pytest process (on 3 Oct a
401 from it failed a session at conftest import). Inside the sandbox a key
comes from the environment or ``.env`` only.

The network is shared: tests call model providers. The sandbox therefore
bounds what model code can *read and write*; it does not hide the provider
keys this process holds in its environment and ``.env``, nor stop a request
to the network. Code a model writes runs in this same process, so anything
the process can reach, the code can reach; only the Python worker
(``UNIFY_WORKSPACE=sandboxed`` with ``UNIFY_WORKSPACE_PYTHON=worker``) runs
cells in a child process without credentials or network, and turning it on
changes the actor's tools and prompt, so it is not a test default.

``UNIFY_TEST_SANDBOX`` selects the behaviour: ``auto`` (the default) confines
the run where bubblewrap exists (Linux) and runs as before elsewhere, with a
warning; ``required`` refuses to run unconfined; ``off`` never confines.
Where bubblewrap exists but cannot start a sandbox, ``auto`` refuses too
rather than run unconfined. Nested pytest processes see
``UNIFY_TEST_SANDBOXED=1`` and run in the sandbox they were started in.
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

MODE_VAR = "UNIFY_TEST_SANDBOX"
INSIDE_VAR = "UNIFY_TEST_SANDBOXED"

# Read-only system directories (or the symlinks a merged /usr leaves there).
SYSTEM_DIRS = ("/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/etc")

# Directories under /tmp shared by every test session on the machine.
SHARED_TMP_DIRS = ("/tmp/unity_test_home", "/tmp/data-gym-cache")

# Executables the tests start that may live outside the system directories.
TOOLS = ("uv", "uvx", "rg")

# Names in the LLM cache directory that stay visible: the cache, its index and
# the temporary files the index is rewritten through.
CACHE_PREFIX = ".cache.ndjson"

# Writable directories of the checkout.
WRITABLE = ("logs", ".pytest_cache")

# Variables naming sockets or agents outside the sandbox; meaningless inside.
DROPPED_ENV = (
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "GPG_AGENT_INFO",
    "DBUS_SESSION_BUS_ADDRESS",
    "XDG_RUNTIME_DIR",
    "WAYLAND_DISPLAY",
    "DISPLAY",
    "TMUX",
    "TMUX_PANE",
)


class SandboxUnavailable(RuntimeError):
    pass


def mode() -> str:
    value = os.environ.get(MODE_VAR, "auto").strip().lower() or "auto"
    if value not in ("auto", "required", "off"):
        raise SandboxUnavailable(
            f"{MODE_VAR} must be auto, required or off, not {value!r}",
        )
    return value


def inside() -> bool:
    return os.environ.get(INSIDE_VAR) == "1"


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _real(path: str | os.PathLike) -> Path:
    return Path(os.path.realpath(path))


def _is_system(path: Path) -> bool:
    return any(_within(path, _real(d)) for d in SYSTEM_DIRS if os.path.exists(d))


def main_checkout(repo_root: Path) -> Path:
    """The main checkout of a worktree (where the shared LLM cache lives)."""
    git = repo_root / ".git"
    if git.is_file():
        line = git.read_text().strip()
        if line.startswith("gitdir:"):
            main = Path(line[len("gitdir:") :].strip()).parent.parent.parent
            if main.exists():
                return main
    return repo_root


def _cache_dir(repo_root: Path) -> Path:
    raw = os.environ.get("UNILLM_CACHE_DIR", "").strip()
    return _real(raw) if raw else _real(main_checkout(repo_root))


def _python_roots(repo: Path) -> list[Path]:
    """Read-only roots the interpreter needs beyond the system and the checkout.

    Each is listed as written and resolved: uv links a minor version
    (``cpython-3.12-...``) to the installed patch release, and the venv's
    interpreter is a link to the former.
    """
    roots: list[Path] = []
    for raw in (sys.base_prefix, sys.prefix, *sys.path):
        if not raw:
            continue
        for path in (Path(os.path.abspath(raw)), _real(raw)):
            if not path.exists() or _is_system(path) or _within(_real(path), repo):
                continue
            if any(_within(path, r) for r in roots):
                continue
            roots = [r for r in roots if not _within(r, path)] + [path]
    return roots


def _tools(repo: Path) -> list[Path]:
    found = []
    for name in TOOLS:
        where = shutil.which(name)
        if where is None:
            continue
        for path in {Path(where).absolute(), _real(where)}:
            if not _is_system(path) and not _within(path, repo):
                found.append(path)
    return found


def _ro(path: Path) -> list[str]:
    """*path* read-only at its own name (a link's target is mounted there)."""
    return ["--ro-bind", str(_real(path)), str(path)]


def _mask_dir_except(directory: Path, keep_prefix: str) -> list[str]:
    """Hide every entry of *directory* except names starting with *keep_prefix*."""
    args: list[str] = []
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        if entry.name.startswith(keep_prefix) or entry.is_symlink():
            # A symlink stays a symlink; its target is visible only if it is
            # mounted for another reason.
            continue
        if entry.is_dir(follow_symlinks=False):
            args += ["--tmpfs", entry.path]
        else:
            args += ["--ro-bind", "/dev/null", entry.path]
    return args


def _neutral_gitfile(repo: Path) -> list[str]:
    """Point a worktree's ``.git`` file at an empty repository.

    A worktree's ``.git`` names its metadata in the main checkout, which the
    sandbox hides; git then refuses every command started in the checkout,
    even ``git ls-remote`` of another repository. Inside, the checkout is a
    repository without history instead.
    """
    gitfile = repo / ".git"
    git = shutil.which("git")
    if not gitfile.is_file() or git is None:
        return []
    empty = repo / "logs" / ".test-sandbox" / "git"
    if not (empty / ".git").is_dir():
        empty.mkdir(parents=True, exist_ok=True)
        subprocess.run([git, "init", "-q", str(empty)], check=False, timeout=60)
    pointer = repo / "logs" / ".test-sandbox" / "gitfile"
    pointer.write_text(f"gitdir: {empty / '.git'}\n")
    return ["--ro-bind", str(pointer), str(gitfile)]


def bwrap_args(repo_root: Path, cwd: Path) -> list[str]:
    """The bubblewrap command line (without the command) for this checkout."""
    repo = _real(repo_root)
    args = [
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup-try",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for name in SYSTEM_DIRS:
        if os.path.islink(name):
            args += ["--symlink", os.readlink(name), name]
        elif os.path.isdir(name):
            args += _ro(Path(name))
    # WSL links /etc/resolv.conf into /mnt/wsl; without its target no host resolves.
    resolver = _real("/etc/resolv.conf")
    if resolver.is_file() and not _is_system(resolver):
        args += _ro(resolver)
    args += ["--tmpfs", "/tmp", "--tmpfs", "/var/tmp"]
    for shared in SHARED_TMP_DIRS:
        if os.path.isdir(shared) and not os.path.islink(shared):
            args += ["--bind", shared, shared]

    for root in _python_roots(repo):
        args += _ro(root)
    for tool in _tools(repo):
        args += _ro(tool)

    # The checkout: read-only, then its writable parts on top. The LLM cache
    # index is rewritten through a temporary file beside it, so the cache's
    # directory itself must be writable.
    cache = _cache_dir(repo)
    if cache == repo:
        # The main checkout holds the cache: writable, every entry read-only.
        args += ["--bind", str(repo), str(repo)]
        for entry in sorted(os.scandir(repo), key=lambda e: e.name):
            if entry.name.startswith(CACHE_PREFIX) or entry.is_symlink():
                continue
            if entry.name not in WRITABLE:
                args += _ro(Path(entry.path))
    else:
        args += _ro(repo)
        if _within(cache, repo):
            cache.mkdir(parents=True, exist_ok=True)
            args += ["--bind", str(cache), str(cache)]
        elif cache.is_dir():
            # A worktree shares the main checkout's cache; nothing else of
            # the main checkout is visible.
            args += ["--bind", str(cache), str(cache)]
            args += _mask_dir_except(cache, CACHE_PREFIX)
    for writable in WRITABLE:
        path = repo / writable
        path.mkdir(exist_ok=True)
        args += ["--bind", str(path), str(path)]
    args += _neutral_gitfile(repo)
    env_file = repo / ".env"
    if env_file.exists():
        target = _real(env_file)
        if not _within(target, repo):
            args += _ro(target)

    workdir = cwd if _within(_real(cwd), repo) else repo
    args += ["--chdir", str(workdir)]
    return args


def _env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in DROPPED_ENV:
        env.pop(name, None)
    cache = Path(repo_root) / "logs" / ".test-sandbox"
    (cache / "pycache").mkdir(parents=True, exist_ok=True)
    (cache / "uv").mkdir(parents=True, exist_ok=True)
    # The checkout and the venv are read-only, so bytecode is written here.
    env.setdefault("PYTHONPYCACHEPREFIX", str(cache / "pycache"))
    env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
    # Fixed here: inside, the checkout's .git no longer leads to the main
    # checkout that holds the shared cache (_neutral_gitfile).
    env.setdefault("UNILLM_CACHE_DIR", str(_cache_dir(_real(repo_root))))
    env[INSIDE_VAR] = "1"
    return env


def _probe(bwrap: str) -> Optional[str]:
    """Why bubblewrap cannot start a sandbox here, or None when it can."""
    try:
        done = subprocess.run(
            [
                bwrap,
                "--ro-bind",
                "/",
                "/",
                "--unshare-pid",
                "--die-with-parent",
                "true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)
    if done.returncode != 0:
        return done.stderr.strip() or f"exit status {done.returncode}"
    return None


def _stop_pytest_capture() -> None:
    """Give the standard streams back before replacing the process.

    The root conftest is imported while pytest captures output; with fd
    capture, descriptors 1 and 2 point at temporary files until it stops.
    """
    for obj in gc.get_objects():
        cls = type(obj)
        if cls.__name__ == "CaptureManager" and cls.__module__ == "_pytest.capture":
            try:
                obj.stop_global_capturing()
            except Exception:
                pass
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            stream.flush()
        except Exception:
            pass


def command(
    repo_root: Path,
    argv: Iterable[str],
    cwd: Optional[Path] = None,
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise SandboxUnavailable("bubblewrap (bwrap) is not installed")
    return [bwrap, *bwrap_args(Path(repo_root), Path(cwd or os.getcwd())), "--", *argv]


def enter(repo_root: Path) -> None:
    """Re-execute this pytest process inside the sandbox, unless it already is."""
    if inside() or os.environ.get("PYTEST_XDIST_WORKER"):
        return
    wanted = mode()
    if wanted == "off":
        return
    bwrap = shutil.which("bwrap")
    if bwrap is None or not sys.platform.startswith("linux"):
        if wanted == "required":
            raise SandboxUnavailable(
                f"{MODE_VAR}=required, but bubblewrap is not available here",
            )
        sys.stderr.write(
            "WARNING: tests run unconfined: bubblewrap is not available, so "
            "model-written code can read this machine (tests/_test_sandbox.py)\n",
        )
        return
    problem = _probe(bwrap)
    if problem is not None:
        raise SandboxUnavailable(
            f"bubblewrap cannot start the test sandbox ({problem}); set "
            f"{MODE_VAR}=off to run unconfined",
        )
    cwd = Path(os.getcwd())
    if not _within(_real(cwd), _real(repo_root)):
        raise SandboxUnavailable(
            f"run pytest from inside the checkout ({repo_root}), not {cwd}, "
            f"or set {MODE_VAR}=off",
        )
    argv = [sys.executable, *sys.orig_argv[1:]]
    full = command(repo_root, argv, cwd)
    env = _env(repo_root)
    _stop_pytest_capture()
    os.execve(full[0], full, env)
