"""Observe-only audit hook for the writer box's Python: one JSON line per file open under /inputs or /memory.

Installed as the box's ``sitecustomize``: the harness writes this file, read-only, as ``/inputs/sitecustomize.py``.
The box runs ``python /cell/code.py`` without ``-I`` or ``-S`` and with ``PYTHONPATH=/inputs:/memory``, so ``site``
imports it before any user code, and ``/inputs`` comes first on the path (a ``sitecustomize`` the writer puts in
/memory is shadowed). Importing it installs the hook; nothing else is needed.

Transport: ``MEMORY_BOX_OPEN_LOG=fd:<n>`` names the write end of a pipe the harness passes through ``run_confined``'s
``pass_fds`` (the box's environment is cleared; the harness adds this one variable). The harness reads the other end,
stamps each line with the pass and the call (values from inside the box are never trusted) and appends the lines to
``<pass out>/box-opens.jsonl``. Without the variable, or when ``<n>`` is not an open pipe, nothing is installed.

Lines written: {"event": "start"} once, when the hook is installed (so the host can tell a working hook from a
missing one, even in a cell that opens nothing), then {"event": "open", "path": "/inputs/...", "mode": "r|w|a|x"}

Guarantees:
  * only paths under /inputs or /memory are written, after normalisation (``..`` and ``//`` resolved; a relative
    path is joined to the working directory, read once at install and again at the first relative open after an
    ``os.chdir`` audit event). Nothing else is logged;
  * it never blocks: the descriptor is set non-blocking at install, and each line is one ``os.write`` of at most
    ``PIPE_BUF`` bytes (atomic on a pipe; a longer line is skipped);
  * it never raises into user code. An event it cannot convert is skipped. On any write failure (EAGAIN: the pipe is
    full; EPIPE: no reader; a short write; the descriptor closed or replaced by another file, checked by device and
    inode before each write) it marks itself dead and stops logging for the rest of the process. A dead hook is a
    missing measurement, never a crash;
  * per open it does one string comparison on the event name and a prefix test; only an in-area open costs a
    ``json.dumps``, an ``fstat`` and an ``os.write``. Every other audit event returns at the first comparison;
  * it is observe-only: it opens nothing, adds no path, mount or permission, and changes nothing user code sees.

Shadowing: this file takes the name ``sitecustomize``. The interpreter install's own ``sitecustomize`` (if it has one)
is then run, found ONLY on ``sys.path`` entries inside ``sys.prefix`` or ``sys.base_prefix`` (the interpreter's
standard library and site-packages, read-only in the box). A ``sitecustomize`` in any other folder (/memory, /cell, a
working-directory entry, any writable folder) never runs.

Limits: opens through a descriptor relative to a ``dir_fd`` are resolved against the working directory; opens by child
processes are not seen (the pipe is not passed on: a child's hook finds no pipe and installs nothing); code in the box
can still close or flood the pipe, which kills the hook (a measurement, not a security control).
"""

import os
import sys

_AREAS = ("/inputs", "/memory")
_ENV_LOG = "MEMORY_BOX_OPEN_LOG"
_PIPE_BUF = (
    4096  # POSIX guarantees at least 512; Linux pipes are atomic up to 4096 bytes
)
#: the installed hook's state ("dead", "cwd"), for inspection only
STATE = {}


def in_area(path, cwd):
    """The normalised absolute path when *path* is under /inputs or /memory, else None. Pure; never raises on str."""
    if not path.startswith("/"):
        if cwd is None:
            return None
        path = cwd + "/" + path
    if "/." in path or "//" in path:
        path = os.path.normpath(path)
        if path.startswith("//"):
            path = "/" + path.lstrip("/")
    for a in _AREAS:
        if path.startswith(a) and (len(path) == len(a) or path[len(a)] == "/"):
            return path
    return None


def _mode(mode, flags):
    if isinstance(mode, str):
        for ch in ("w", "a", "x"):
            if ch in mode:
                return ch
        return "w" if "+" in mode else "r"
    if isinstance(flags, int):
        if flags & os.O_APPEND:
            return "a"
        if flags & os.O_EXCL and flags & os.O_CREAT:
            return "x"
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            return "w"
    return "r"


def make_hook(write, cwd=None, state=None):
    """An audit hook writing one JSON line per in-area open through *write(bytes)*, which raises on failure.
    *cwd*: the starting working directory, or None to read it now. *state* (a dict) is exposed for tests.
    """
    import json

    state = {} if state is None else state
    state.update(
        {"cwd": cwd if cwd is not None else _getcwd(), "dead": False, "stale": False},
    )
    dumps, fsdecode, fspath = json.dumps, os.fsdecode, os.fspath

    def hook(event, args):
        if event != "open":
            if (
                event == "os.chdir"
            ):  # raised before the change (which may fail): re-read lazily at the next relative open
                state["stale"] = True
            return
        if state["dead"]:
            return
        try:
            p = args[0]
            if isinstance(p, int):
                return
            if not isinstance(p, str):
                p = fsdecode(fspath(p))
            if state["stale"] and not p.startswith("/"):
                state["cwd"], state["stale"] = _getcwd(), False
            p = in_area(p, state["cwd"])
            if p is None:
                return
            line = (
                '{"event": "open", "path": '
                + dumps(p)
                + ', "mode": "'
                + _mode(args[1], args[2])
                + '"}\n'
            ).encode("utf-8")
            if len(line) > _PIPE_BUF:  # would not be atomic on a pipe: skip this one
                return
        except Exception:  # an odd path object: skip this event only
            return
        try:
            write(line)
        except (
            Exception
        ):  # pipe full, no reader, short write, descriptor gone or replaced: stop for good
            state["dead"] = True

    return hook


def _getcwd():
    try:
        return os.getcwd()
    except Exception:
        return None


def pipe_writer(fd):
    """A writer for the pipe *fd*: non-blocking, one atomic ``os.write`` per line, and a device/inode check so a
    descriptor number reused for another file is never written to. Raises (caller marks the hook dead) on any
    failure. Returns None when *fd* is not an open pipe."""
    import stat

    st = os.fstat(fd)
    if not stat.S_ISFIFO(st.st_mode):
        return None
    os.set_blocking(fd, False)
    ident = (st.st_dev, st.st_ino)
    fstat, write = os.fstat, os.write

    def _write(data):
        s = fstat(fd)
        if (s.st_dev, s.st_ino) != ident:
            raise OSError("log descriptor replaced")
        if write(fd, data) != len(data):
            raise OSError("short write")

    return _write


def install(environ=None, state=None):
    """Install the hook from the environment. Returns True when installed. Never raises."""
    state = STATE if state is None else state
    try:
        env = os.environ if environ is None else environ
        target = env.get(_ENV_LOG) or ""
        if not target.startswith("fd:") or not target[3:].isdigit():
            return False
        write = pipe_writer(int(target[3:]))
        if write is None:
            return False
        write(
            b'{"event": "start"}\n',
        )  # the hook is live: the host counts a cell without it as unmeasured
        sys.addaudithook(make_hook(write, state=state))
        return True
    except Exception:
        return False


def install_dirs(paths, prefixes, own_dir=None):
    """The entries of *paths* inside one of *prefixes* (real paths compared), minus *own_dir* and the areas. A prefix
    of "/" (or none) allows nothing."""
    roots = []
    for p in prefixes:
        if p:
            r = os.path.realpath(p)
            if r != "/":
                roots.append(r.rstrip("/") + "/")
    out = []
    for p in paths:
        if not p or not os.path.isabs(p):
            continue
        r = os.path.realpath(p)
        if own_dir is not None and r == os.path.realpath(own_dir):
            continue
        if in_area(r, None) is not None:
            continue
        if any((r + "/").startswith(root) for root in roots):
            out.append(p)
    return out


def _chain_install_sitecustomize():
    """Run the interpreter install's own ``sitecustomize``, if any, searching only the install's own folders (see the
    module docstring). An error in it propagates to ``site``, which reports it as it would without this hook.
    """
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = install_dirs(sys.path, (sys.prefix, sys.base_prefix), own_dir=here)
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


if __name__ == "sitecustomize":
    install()
    _chain_install_sitecustomize()
