"""A persistent Python workspace in a confined child process.

The model's code runs in a separate interpreter that keeps its variables between cells,
so the model does not have to remember to save state. After every cell the worker reports
what changed (new or changed variables, the value of a trailing expression, files read and
written, and anything passed to deliver()).

CONFINEMENT. Every workspace (live and replay) runs under bubblewrap; there is no unconfined
mode. The child sees:
  * its own user, pid, network, ipc, uts and cgroup namespaces (--unshare-all): no network
    at all, only a loopback device; it cannot see or signal host processes;
  * /usr read-only (the system interpreter /usr/bin/python3 and its stdlib), with the usual
    /bin, /lib, /lib64 symlinks; a fresh /proc and a minimal /dev;
  * the task directory, read-write, at /work (its cwd); nothing else from the host: no /home,
    no repository, no /etc, no benchmark data;
  * a private tmpfs at /tmp holding HOME=/tmp/home; it vanishes with the process;
  * the root itself remounted read-only, so writes anywhere but /work and /tmp fail;
  * a cleared environment plus PATH, HOME, LANG and PYTHONDONTWRITEBYTECODE only.
Limits: address space, file size and open files (setrlimit, inherited through bwrap), and a
wall-clock timeout per cell, after which bwrap's process group is killed; --die-with-parent
and the pid namespace take the worker with it, and termination is verified by looking for the
worker's unique marker in the host process table.
NOT confined here (needs a cgroup, see design.md): total CPU time, total memory including the
tmpfs, and the number of processes.
"""
from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
import uuid
from pathlib import Path

from . import limits as L

BWRAP = "/usr/bin/bwrap"
INTERPRETER = "/usr/bin/python3"  # inside the read-only /usr bind
INSIDE = "/work"

WORKER = r'''
import ast, builtins, contextlib, hashlib, io, json, os, sys, traceback, types
G = {"__name__": "__workspace__"}
CWD = os.getcwd()
_reads, _writes, _truncs, _delivered = set(), set(), set(), []
_real_open = builtins.open
def _rel(p):
    full = os.path.abspath(os.fspath(p))
    return os.path.relpath(full, CWD) if full.startswith(CWD + os.sep) else full
def _open(file, mode="r", *a, **k):
    if not isinstance(file, int):  # file descriptors are not paths
        (_writes if any(c in mode for c in "wax+") else _reads).add(_rel(file))
        if "w" in mode:
            _truncs.add(_rel(file))
    return _real_open(file, mode, *a, **k)
builtins.open = io.open = _open  # pathlib goes through io.open
def _file_entry(full):
    data = _real_open(full, "rb").read()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "head": data[:4000].decode("utf-8", "replace")}
def _files_of(p):
    """{relative path: entry} for a file or directory inside the workspace, else None."""
    if not isinstance(p, (str, os.PathLike)) or (isinstance(p, str) and (not p or "\n" in p or len(p) > 255)):
        return None
    full = os.path.abspath(os.fspath(p))
    if not (full == CWD or full.startswith(CWD + os.sep)) or not os.path.exists(full):
        return None
    if os.path.isfile(full):
        return {_rel(full): _file_entry(full)}
    out = {}
    for root, dirs, names in os.walk(full):
        dirs.sort()
        for n in sorted(names):
            f = os.path.join(root, n)
            if os.path.isfile(f) and not os.path.islink(f):
                out[_rel(f)] = _file_entry(f)
    return out
def deliver(value=None, note=""):
    """Hand the final answer to the harness. A path (or list of paths) to files or folders in the
    working folder delivers those files; anything else is delivered as a value."""
    items = list(value) if isinstance(value, (list, tuple)) and value else [value]
    found = [_files_of(v) for v in items] if all(isinstance(v, (str, os.PathLike)) for v in items) else [None]
    if all(f is not None for f in found):
        files = {k: v for f in found for k, v in f.items()}
        _delivered.append(({"__files__": {k: v["sha256"] for k, v in sorted(files.items())}}, note, files))
    else:
        _delivered.append((value, note, None))
G["deliver"] = deliver
def _short(v, n=160):
    try:
        r = repr(v)
    except Exception:
        r = "<unprintable %s>" % type(v).__name__
    return r if len(r) <= n else r[:n] + "... (%d chars)" % len(r)
def _jsonable(v):
    try:
        json.dumps(v, allow_nan=True)
        return v
    except Exception:
        return {"__repr__": _short(v, 2000)}
def _shown():
    hidden = G.get("__hidden__", ())
    return {k: v for k, v in G.items() if not k.startswith("_") and k not in ("deliver", "__builtins__")
            and k not in hidden and not isinstance(v, types.ModuleType)}
def _state():
    return {k: _short(v, 80) for k, v in _shown().items()}
def _describe(v):
    if callable(v):
        return "function"
    t = type(v).__name__
    try:
        return "%s[%d]" % (t, len(v)) if not isinstance(v, str) else t
    except Exception:
        return t
for line in sys.stdin:
    msg = json.loads(line)
    op = msg.get("op")
    if op == "set":
        G.update(msg["values"]); reply = {"ok": True}
    elif op == "preamble":
        exec(compile(msg["code"], "<preamble>", "exec"), G); reply = {"ok": True}
    elif op == "inventory":
        reply = {"ok": True, "inventory": {k: [_describe(v), _short(v, 120)] for k, v in _shown().items()}}
    else:
        before = _state()
        _reads.clear(); _writes.clear(); _truncs.clear(); _delivered.clear()
        out, err, last = io.StringIO(), None, None
        try:
            tree = ast.parse(msg["code"])
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    exec(compile(ast.Module(tree.body[:-1], []), "<cell>", "exec"), G)
                    v = eval(compile(ast.Expression(tree.body[-1].value), "<cell>", "eval"), G)
                    if v is not None and not _delivered:
                        last = _short(v, 600)
                else:
                    exec(compile(tree, "<cell>", "exec"), G)
        except BaseException:
            err = traceback.format_exc(limit=-2)
        after = _state()
        changed = {k: [_describe(G[k]), _short(G[k])] for k, r in after.items() if before.get(k) != r}
        reply = {"ok": err is None, "stdout": out.getvalue(), "error": err, "last": last,
                 "changed": changed, "reads": sorted(_reads), "writes": sorted(_writes),
                 "truncates": sorted(_truncs)}
        if _delivered:
            value, note, files = _delivered[-1]
            reply["delivered"] = {"value": _jsonable(value), "note": note, "files": files}
    sys.__stdout__.write(json.dumps(reply) + "\n"); sys.__stdout__.flush()
'''


class ConfinementError(RuntimeError):
    pass


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def check_task_dir(task_dir: str, hidden: list[str] = ()) -> Path:
    """The task directory is the only host path the workspace sees. Refuse one that would expose
    the user's home, the root, or any path the caller says must stay hidden (checkers, generators,
    expected answers, the repository)."""
    d = Path(task_dir).resolve()
    if not d.is_dir():
        raise ConfinementError(f"task directory does not exist: {d}")
    for must_hide in [Path("/"), Path.home().resolve(), *(Path(h).resolve() for h in hidden)]:
        if _inside(must_hide, d):
            raise ConfinementError(f"task directory {d} would expose {must_hide}")
        if must_hide != Path("/") and must_hide != Path.home().resolve() and _inside(d, must_hide):
            raise ConfinementError(f"task directory {d} lies inside hidden path {must_hide}")
    return d


def bwrap_argv(task_dir: Path, marker: str, world_dir: str | None = None) -> list[str]:
    world = ["--bind", world_dir, "/run/world", "--setenv", "WORLD_SOCKET", "/run/world/gw.sock"] if world_dir else []
    return [BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev",
            "--tmpfs", "/tmp", "--dir", "/tmp/home",
            "--bind", str(task_dir), INSIDE, "--chdir", INSIDE,
            *world,
            "--remount-ro", "/",
            "--hostname", "workspace",
            "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp/home",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            INTERPRETER, "-I", "-c", WORKER, marker]


def _pid_ns(pid) -> str | None:
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return None


def ns_members(namespaces: set[str]) -> list[int]:
    """Host pids of every process (of ours) living in one of these pid namespaces."""
    return [int(e.name) for e in os.scandir("/proc") if e.name.isdigit() and _pid_ns(e.name) in namespaces]


LAUNCHED: list[tuple[str, frozenset]] = []  # (marker, pid namespaces) of every workspace started here


def survivors(since: int = 0) -> list[int]:
    """Host pids still alive from any workspace launched by this process (after index `since`)."""
    return sorted({p for m, ns in LAUNCHED[since:] for p in marker_alive(m) + ns_members(set(ns))})


def marker_alive(marker: str) -> list[int]:
    """Host pids with this exact workspace marker as one of their arguments."""
    pids = []
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                with open(f"/proc/{entry.name}/cmdline", "rb") as f:
                    if marker.encode() in f.read().split(b"\0"):
                        pids.append(int(entry.name))
            except OSError:
                continue
    return pids


class Sandbox:
    """One persistent, confined workspace process bound to one task directory."""

    def __init__(self, workdir: str, timeout: float = 10.0, max_output: int = 4000,
                 hidden: list[str] = (), limits: L.Limits | None = None, world=None):
        if not os.access(BWRAP, os.X_OK) or not os.access(INTERPRETER, os.X_OK):
            raise ConfinementError(f"{BWRAP} and {INTERPRETER} are required; there is no unconfined fallback")
        self.task_dir = check_task_dir(workdir, hidden)
        self.workdir, self.timeout, self.max_output = workdir, timeout, max_output
        self.limits = limits or L.default()
        self.world = world  # a cleanslate.world.Gateway: its socket is the workspace's only way out
        try:
            self.mode = L.resolve_mode(self.limits)
        except L.LimitError as exc:
            raise ConfinementError(str(exc)) from None
        self.proc: subprocess.Popen | None = None
        self.marker = ""
        self.namespaces: set[str] = set()
        self.unit, self.cgroup, self.limits_report = None, None, {}
        self._start()

    def _start(self):
        token = uuid.uuid4().hex
        self.marker = f"cleanslate-ws-{token}"
        argv, env = bwrap_argv(self.task_dir, self.marker, self.world.dir if self.world else None), {}
        if self.mode == "scope":
            self.unit = f"cleanslate-{token[:20]}"
            argv, env = L.scope_prefix(self.limits, self.unit) + argv, L.systemd_env()
        self.proc = subprocess.Popen(
            argv, env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, preexec_fn=L.preexec(self.limits, self.mode))
        if not self._call({"op": "set", "values": {}}, _starting=True).get("ok"):  # wait until the worker answers
            self.close()
            raise ConfinementError("the confined workspace did not start")
        if self.world is not None:
            from .world import CLIENT
            if not self._call({"op": "preamble", "code": CLIENT}, _starting=True).get("ok"):
                self.close()
                raise ConfinementError("the API client did not load in the workspace")
        own = _pid_ns("self")
        self.namespaces = {n for n in map(_pid_ns, marker_alive(self.marker)) if n and n != own}
        LAUNCHED.append((self.marker, frozenset(self.namespaces)))
        if not self.namespaces:
            self.close()
            raise ConfinementError("the workspace is not in its own pid namespace")
        self.limits_report = {"mode": self.mode, "rlimits": {
            "as_mb": self.limits.address_space_mb, "fsize_mb": self.limits.file_size_mb, "nofile": self.limits.open_files,
            "cpu_s": self.limits.cpu_seconds, "nproc": self.limits.tasks_max if self.mode == "prlimit-user" else None},
            "workdir_max_mb": self.limits.workdir_max_mb}
        if self.mode == "scope":
            try:
                found = L.verify_scope(self.proc.pid, self.limits, self.unit)
            except L.LimitError as exc:
                self.close()
                raise ConfinementError(str(exc)) from None
            self.cgroup = found["cgroup"]
            self.limits_report["cgroup"] = found

    def _call(self, msg: dict, _starting: bool = False) -> dict:
        assert self.proc and self.proc.stdin and self.proc.stdout
        why = None
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            why = "the workspace process died"
        if why is None:
            ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout)
            line = self.proc.stdout.readline() if ready else None
            if line:
                return json.loads(line)
            why = "the workspace process exited (memory or file-size limit?)" if ready else f"timeout after {self.timeout:g}s"
        return {"ok": False, "error": why} if _starting else self._reset(why)

    def _reset(self, why: str) -> dict:
        self.close()
        self._start()
        return {"ok": False, "error": f"{why}; the workspace was restarted and its variables were lost",
                "stdout": "", "changed": {}, "reads": [], "writes": [], "reset": True}

    def run(self, code: str) -> dict:
        res = self._call({"op": "run", "code": code})
        size = L.dir_size_mb(str(self.task_dir))
        if size > self.limits.workdir_max_mb:
            out = self._reset(f"the task folder holds {size:.0f} MB, over its {self.limits.workdir_max_mb} MB limit")
            out["limit"] = "workdir"
            return out
        out = res.get("stdout", "")
        if len(out) > self.max_output:  # keep head and tail, say how much was cut
            half = self.max_output // 2
            res["stdout"] = f"{out[:half]}\n... [{len(out) - self.max_output} chars cut] ...\n{out[-half:]}"
        return res

    def set(self, **values) -> dict:
        return self._call({"op": "set", "values": values})

    def inventory(self) -> dict:
        return self._call({"op": "inventory"}).get("inventory", {})

    def close(self):
        """Kill bwrap's process group and verify that no process carrying the marker survives."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait(timeout=5)
        for f in (self.proc.stdin, self.proc.stdout):
            try:
                f and f.close()
            except OSError:
                pass
        deadline = time.monotonic() + 5
        while (left := marker_alive(self.marker) + ns_members(self.namespaces)) and time.monotonic() < deadline:
            time.sleep(0.05)
        if left:
            raise ConfinementError(f"workspace processes survived termination: {left}")
        if self.unit and not L.stop_scope(self.unit, self.cgroup):
            raise ConfinementError(f"scope {self.unit} still exists after termination")
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
