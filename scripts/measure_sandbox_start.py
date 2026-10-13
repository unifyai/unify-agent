"""Measure how long the sandbox takes to start, cold and warm, and how often
the sandboxed Python worker's start hangs.

Run from a worktree root on a host with bubblewrap::

    python -I scripts/measure_sandbox_start.py --starts 50

It prints one JSON object on stdout:

* ``policy``: the first ``build_policy(fresh=True)`` in a fresh process with
  an empty scan cache (``cold``) and with the cache the cold run left
  (``warm``); each also times the interpreter roots' fingerprint and says
  how many interpreter roots the scan walked (``interpreter_roots_walked``,
  0 when the cache served them all).
* ``first_worker``: the first worker start (a ``SessionExecutor`` runs ``1``)
  in a fresh process, cold and warm, split into the policy build, the
  bubblewrap spawn to the worker's ready line, and the first cell.
* ``loop``: ``--starts`` worker starts in one process, a fresh executor each:
  min, p50, p90, p99 and max of each part, and the failures. A start past
  ``--hang-s`` gets a dump of the worker's processes (``py-spy dump`` where
  py-spy is installed, else each process's ``/proc`` stack, ``wchan`` and
  state) in ``loop.hangs``, and the loop continues.

The worker parts need bubblewrap; without it they are ``null`` with a reason.
Every run gets its own ``UNIFY_HOME`` and scan cache under ``--base``
(default ``~/unify-measure-sandbox-start``, outside ``/tmp``, whose sandbox
view is private), removed afterwards unless ``--keep``. Nothing here reaches
a model or reads a credential.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# ── the child process: one measurement in a fresh interpreter ──────────────


def _import_unify():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from unify import sandbox
    from unify.actor.execution import worker as worker_mod

    return sandbox, worker_mod


class _Timers:
    """Wraps the policy build, the roots' walk and the worker's start so each
    start's time can be split into parts."""

    def __init__(self, sandbox, worker_mod):
        self.parts: dict[str, float] = {}
        self.walked = 0
        self.fingerprint_s = 0.0
        self.worker = None
        self._fingerprinted = None
        bp = sandbox.build_policy
        # Absent on builds before the scan cache (a baseline run): those
        # parts are then not split out.
        walk = getattr(sandbox, "_walk_secret_files", None)
        fp = getattr(sandbox, "_root_fingerprint", None)
        start = worker_mod.PythonWorker._start
        timers = self

        def build_policy(*a, **k):
            t = time.perf_counter()
            try:
                return bp(*a, **k)
            finally:
                timers._add("policy_s", time.perf_counter() - t)

        def walk_secret_files(root, *a, **k):
            # Only the interpreter roots are fingerprinted and cached, and
            # their walk follows their fingerprint at once (a cache miss).
            if root == timers._fingerprinted:
                timers.walked += 1
            return walk(root, *a, **k)

        def root_fingerprint(root, *a, **k):
            timers._fingerprinted = root
            t = time.perf_counter()
            try:
                return fp(root, *a, **k)
            finally:
                timers.fingerprint_s += time.perf_counter() - t

        async def worker_start(self_, *a, **k):
            timers.worker = self_
            t = time.perf_counter()
            try:
                return await start(self_, *a, **k)
            finally:
                timers._add("start_s", time.perf_counter() - t)

        sandbox.build_policy = build_policy
        if walk is not None:
            sandbox._walk_secret_files = walk_secret_files
        if fp is not None:
            sandbox._root_fingerprint = root_fingerprint
        worker_mod.PythonWorker._start = worker_start

    def _add(self, name: str, value: float) -> None:
        self.parts[name] = self.parts.get(name, 0.0) + value

    def reset(self) -> None:
        self.parts = {}
        self.walked = 0
        self.fingerprint_s = 0.0
        self.worker = None
        self._fingerprinted = None


def _descendants(pid: int) -> list[int]:
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                stat = fh.read()
        except OSError:
            continue
        ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        children.setdefault(ppid, []).append(int(entry))
    out, stack = [], [pid]
    while stack:
        p = stack.pop()
        out.append(p)
        stack += children.get(p, [])
    return out


def _read(path: str) -> str:
    try:
        with open(path, errors="replace") as fh:
            return fh.read().strip()
    except OSError as exc:
        return f"<{type(exc).__name__}: {exc.strerror}>"


def _dump(pid: int | None) -> dict:
    """The worker's processes as they are now, and the harness's own stack."""
    out: dict = {"harness_pid": os.getpid(), "processes": []}
    pyspy = shutil.which("py-spy")
    pids = _descendants(pid) if pid else []
    for p in pids:
        info = {
            "pid": p,
            "cmdline": _read(f"/proc/{p}/cmdline").replace("\0", " ")[:300],
            "state": next(
                (
                    line
                    for line in _read(f"/proc/{p}/status").splitlines()
                    if line.startswith("State:")
                ),
                None,
            ),
            "wchan": _read(f"/proc/{p}/wchan"),
            "stack": _read(f"/proc/{p}/stack"),
        }
        if pyspy and "python" in info["cmdline"]:
            info["py_spy"] = _run([pyspy, "dump", "--pid", str(p)])
        out["processes"].append(info)
    if pyspy:
        out["harness_py_spy"] = _run([pyspy, "dump", "--pid", str(os.getpid())])
    return out


def _run(argv: list[str]) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return (done.stdout + done.stderr)[-8000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<{type(exc).__name__}>"


async def _one_start(timers, worker_mod, hang_s: float) -> dict:
    from unify.actor.execution.session import SessionExecutor

    timers.reset()
    ex = SessionExecutor()
    t0 = time.perf_counter()
    task = asyncio.ensure_future(
        ex.execute(code="1", state_mode="stateful", session_id=0),
    )
    rec: dict = {}
    try:
        done, _ = await asyncio.wait({task}, timeout=hang_s)
        if not done:
            w = timers.worker
            proc = getattr(w, "_proc", None)
            rec["hang"] = _dump(proc.pid if proc is not None else None)
            # The worker's own START_TIMEOUT_S ends a hung start; this bounds
            # a hang anywhere else.
            done, _ = await asyncio.wait(
                {task},
                timeout=worker_mod.START_TIMEOUT_S + 30,
            )
            if not done:
                task.cancel()
                rec["error"] = "no result: cancelled"
        if done:
            try:
                res = task.result()
                if res.get("error") is not None or res.get("result") != 1:
                    rec["error"] = repr(res.get("error"))[:2000]
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
    finally:
        try:
            await asyncio.wait_for(ex.close(), timeout=15)
        except Exception as exc:
            rec["close_error"] = f"{type(exc).__name__}: {exc}"
    total = time.perf_counter() - t0
    policy = timers.parts.get("policy_s", 0.0)
    start = timers.parts.get("start_s", 0.0)
    rec.update(
        total_s=total,
        # The policy build inside the worker's start, the spawn to the ready
        # line after it, and the cell (with everything else) after that.
        policy_s=policy,
        spawn_to_ready_s=max(start - policy, 0.0),
        first_cell_s=max(total - start, 0.0),
        interpreter_roots_walked=timers.walked,
    )
    return rec


def _child(mode: str, args) -> dict:
    sandbox, worker_mod = _import_unify()
    timers = _Timers(sandbox, worker_mod)
    if mode == "policy":
        t = time.perf_counter()
        sandbox.build_policy(fresh=True)
        first = time.perf_counter() - t
        walked, fingerprint = timers.walked, timers.fingerprint_s
        timers.reset()
        t = time.perf_counter()
        sandbox.build_policy(fresh=True)
        again = time.perf_counter() - t
        return {
            "first_build_policy_s": first,
            "fingerprint_s": fingerprint,
            "interpreter_roots_walked": walked,
            "second_build_policy_in_process_s": again,
        }
    if mode == "worker":
        return asyncio.run(_one_start(timers, worker_mod, args.hang_s))
    if mode == "loop":

        async def loop():
            return [
                await _one_start(timers, worker_mod, args.hang_s)
                for _ in range(args.starts)
            ]

        return {"starts": asyncio.run(loop())}
    raise SystemExit(f"unknown child mode {mode}")


# ── the parent: fresh processes, cold and warm caches ───────────────────────


def _spawn(mode: str, args, home: Path, xdg: Path) -> dict:
    env = dict(os.environ)
    env["UNIFY_HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(xdg)
    env.pop("UNIFY_LOCAL_ROOT", None)
    argv = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        mode,
        "--starts",
        str(args.starts),
        "--hang-s",
        str(args.hang_s),
    ]
    budget = 600 + args.starts * (args.hang_s + 120) if mode == "loop" else 600
    try:
        done = subprocess.run(
            argv,
            env=env,
            cwd=str(home.parent),
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"the {mode} child ran past {budget}s and was killed"}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        return {
            "error": f"the {mode} child exited {done.returncode}",
            "stderr_tail": done.stderr[-4000:],
        }
    return json.loads(lines[-1])


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)

    def q(p: float) -> float:
        return s[min(len(s) - 1, max(0, round(p * len(s) + 0.5) - 1))]

    return {
        "n": len(s),
        "min": s[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": s[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--starts", type=int, default=50)
    ap.add_argument("--hang-s", type=float, default=30.0)
    ap.add_argument("--base", default=str(Path.home() / "unify-measure-sandbox-start"))
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--child", choices=["policy", "worker", "loop"])
    args = ap.parse_args()
    if args.child:
        print(json.dumps(_child(args.child, args)))
        return

    base = Path(args.base).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=base))
    bwrap = shutil.which("bwrap") is not None and sys.platform.startswith("linux")
    out: dict = {
        "host": os.uname().nodename,
        "python": sys.executable,
        "repo": str(REPO),
        "bwrap": shutil.which("bwrap"),
        "starts": args.starts,
        "hang_s": args.hang_s,
    }
    try:
        home = run_dir / "unify-home"
        home.mkdir()
        xdg = run_dir / "xdg-policy"
        out["policy"] = {
            "cold": _spawn("policy", args, home, xdg),
            "warm": _spawn("policy", args, home, xdg),
        }
        if not bwrap:
            reason = "bubblewrap is not installed; the worker cannot start"
            out["first_worker"] = out["loop"] = None
            out["skipped"] = reason
        else:
            xdg = run_dir / "xdg-worker"
            out["first_worker"] = {
                "cold": _spawn("worker", args, home, xdg),
                "warm": _spawn("worker", args, home, xdg),
            }
            loop = _spawn("loop", args, home, xdg)
            starts = loop.get("starts")
            if starts is None:
                out["loop"] = loop
            else:
                ok = [s for s in starts if "error" not in s]
                out["loop"] = {
                    **{
                        part: _quantiles([s[part] for s in ok])
                        for part in (
                            "total_s",
                            "policy_s",
                            "spawn_to_ready_s",
                            "first_cell_s",
                        )
                    },
                    "failures": [
                        {"start": i, "error": s["error"], "total_s": s["total_s"]}
                        for i, s in enumerate(starts)
                        if "error" in s
                    ],
                    "hangs": [
                        {"start": i, "total_s": s["total_s"], "dump": s["hang"]}
                        for i, s in enumerate(starts)
                        if "hang" in s
                    ],
                    "starts_that_walked_the_interpreter_roots": sum(
                        1 for s in starts if s["interpreter_roots_walked"]
                    ),
                }
    finally:
        if not args.keep:
            shutil.rmtree(run_dir, ignore_errors=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
