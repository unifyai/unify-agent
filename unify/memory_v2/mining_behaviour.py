"""The replay box: runs an episode's recorded Python cells again, in a fresh interpreter inside a bubblewrap box
(no network, read-only root, a private ``/tmp``, process limits, a per-cell alarm), and reports the plain values the
run left. Two jobs: ``replay`` snapshots the inputs of chosen lines, and ``verify`` (:mod:`.replay_verify`) checks
that a library function reproduces the recorded code it replaces.

Only plain data (numbers including ``Decimal``, strings, dates, lists, tuples, dicts, sets) leaves the box, as
tagged JSON; the host never unpickles or executes anything it receives. Code that reads its working directory
(office tasks) runs in a disposable copy of the episode's recorded workspace, bound writable at ``/w``.

Standard library only: the box runs this file by path (``python3 -I mining_behaviour.py --child``), with the job as
JSON on stdin and the result as JSON on stdout.
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import contextlib
import datetime as _dt
import decimal
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

try:
    from . import mining as _mining
except ImportError:  # run as a script beside mining.py (inside the box)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mining as _mining  # type: ignore[no-redef]

CELL_TIMEOUT_S = 5.0  # one replayed cell (operational: a hung loop must end)
RUN_TIMEOUT_S = 2.0  # one prelude statement
MAX_ELEMENTS = 200_000  # a value larger than this is reported, not copied (memory safety in the box)
BOX_LIMITS = ["--as=4294967296", "--nproc=64", "--nofile=256", "--cpu=900"]


class NotPlain(ValueError):
    pass


# --- plain data as tagged JSON ------------------------------------------------------------------------------


def enc(value, _count=None):
    """*value* as tagged JSON; :class:`NotPlain` for anything but plain data (or more than MAX_ELEMENTS)."""
    count = _count if _count is not None else [0]
    count[0] += 1
    if count[0] > MAX_ELEMENTS:
        raise NotPlain("large input")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"t": "f", "v": repr(value)}
    if isinstance(value, list):
        return [enc(v, count) for v in value]
    if isinstance(value, tuple):
        return {"t": "t", "v": [enc(v, count) for v in value]}
    if isinstance(value, (set, frozenset)):
        items = [enc(v, count) for v in value]
        return {
            "t": "s",
            "v": sorted(items, key=lambda x: json.dumps(x, sort_keys=True)),
        }
    if isinstance(value, dict):
        return {
            "t": "d",
            "v": [[enc(k, count), enc(v, count)] for k, v in value.items()],
        }
    if isinstance(value, decimal.Decimal):
        return {"t": "dec", "v": str(value)}
    if isinstance(value, _dt.datetime):  # before date: a datetime is a date
        return {"t": "dt", "v": value.isoformat()}
    if isinstance(value, _dt.date):
        return {"t": "date", "v": value.isoformat()}
    raise NotPlain(f"non-plain input: {type(value).__name__}")


def dec(data, hashable: bool = False):
    """The value :func:`enc` encoded; *hashable* (a set element or a dict key): sets come back frozen."""
    if isinstance(data, list):
        return [dec(v) for v in data]
    if isinstance(data, dict):
        t, v = data.get("t"), data.get("v")
        if t == "t":
            return tuple(dec(x, hashable) for x in v)
        if t == "s":
            items = [dec(x, True) for x in v]
            return frozenset(items) if hashable else set(items)
        if t == "d":
            return {dec(k, True): dec(x) for k, x in v}
        if t == "f":
            return float(v)
        if t == "dec":
            return decimal.Decimal(v)
        if t == "dt":
            return _dt.datetime.fromisoformat(v)
        if t == "date":
            return _dt.date.fromisoformat(v)
    return data


# --- inside the box -------------------------------------------------------------------------------------------


class _Alarm(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Alarm()


def _run_code(code, ns: dict, timeout: float) -> str | None:
    """Run compiled *code* in *ns* with an alarm; the exception's type name, or None."""
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        r = eval(code, ns)  # noqa: S307 - runs only inside the box
        if asyncio.iscoroutine(r):
            asyncio.run(r)
        return None
    except _Alarm:
        return "timeout"
    except BaseException as exc:  # noqa: BLE001 - the actor's code may raise anything
        return type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def child_replay(job: dict) -> dict:
    """Replay one episode's cells; snapshot each target unit's free names at its first execution."""
    signal.signal(signal.SIGALRM, _on_alarm)
    if job.get("cwd"):
        os.chdir(job["cwd"])
    ns: dict = {"__name__": "__main__", "__builtins__": builtins}
    snaps: dict[str, dict] = {}
    causes: dict[str, str] = {}
    cell_errors: dict[int, str] = {}
    fidelity = [0, 0]
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    for cell in job["cells"]:
        idx, fname = cell["index"], f"<cell-{cell['index']}>"
        want: dict[int, list[dict]] = {}
        for t in job["targets"].get(str(idx), []):
            want.setdefault(t["line"], []).append(t)

        def local(frame, event, arg, _want=want):
            if event == "line" and frame.f_lineno in _want:
                for t in _want[frame.f_lineno]:
                    if t["uid"] in snaps or t["uid"] in causes:
                        continue
                    vals = {}
                    try:
                        for n in t["names"]:
                            if n in frame.f_locals:
                                vals[n] = enc(frame.f_locals[n])
                            elif n in frame.f_globals:
                                vals[n] = enc(frame.f_globals[n])
                        snaps[t["uid"]] = vals
                    except NotPlain as exc:
                        causes[t["uid"]] = str(exc)
                    except RecursionError:
                        causes[t["uid"]] = "non-plain input: too deep"
            return local

        def tracer(frame, event, arg, _f=fname, _local=local):
            return _local if frame.f_code.co_filename == _f else None

        try:
            code = compile(cell["code"], fname, "exec", flags=flags)
        except (SyntaxError, ValueError):
            cell_errors[idx] = "SyntaxError"
            continue
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            sys.settrace(tracer)
            try:
                err = _run_code(code, ns, CELL_TIMEOUT_S)
            finally:
                sys.settrace(None)
        if err:
            cell_errors[idx] = err
        fidelity[1] += 1
        fidelity[0] += out.getvalue().strip() == (cell.get("output") or "").strip()
    for cell in job["cells"]:
        for t in job["targets"].get(str(cell["index"]), []):
            if t["uid"] not in snaps and t["uid"] not in causes:
                err = cell_errors.get(cell["index"])
                causes[t["uid"]] = (
                    f"not reached: {err}" if err else "not reached: line not executed"
                )
    return {
        "snaps": snaps,
        "causes": causes,
        "cell_errors": cell_errors,
        "fidelity": fidelity,
    }


def _run_cells(
    cells: list[dict],
    snap: tuple[int, int] | None,
    names: list[str],
    prelude: list[str],
) -> dict:
    """Replay *cells* in one namespace; at line *snap* = (cell index, line) record the names then bound and the
    values of *names*. The namespace's plain values after the last cell, tagged (replay verification).
    """
    ns: dict = {"__name__": "__main__", "__builtins__": builtins}
    for code in prelude:
        err = _run_code(compile(code, "<prelude>", "exec"), ns, RUN_TIMEOUT_S)
        if err:
            return {"error": f"prelude {err}"}
    seen: dict = {}
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    for cell in cells:
        fname = f"<cell-{cell['index']}>"

        def local(frame, event, arg, _idx=cell["index"]):
            if (
                snap
                and not seen
                and event == "line"
                and (_idx, frame.f_lineno) == tuple(snap)
            ):
                scope = {**frame.f_globals, **frame.f_locals}
                seen["bound"] = sorted(k for k in scope if not k.startswith("__"))
                vals = {}
                for n in names:
                    if n in scope:
                        try:
                            vals[n] = enc(scope[n])
                        except (NotPlain, RecursionError) as exc:
                            seen["error"] = str(exc)
                seen["inputs"] = vals
            return local

        def tracer(frame, event, arg, _f=fname, _local=local):
            return _local if frame.f_code.co_filename == _f else None

        try:
            code = compile(cell["code"], fname, "exec", flags=flags)
        except (SyntaxError, ValueError):
            return {"error": "SyntaxError"}
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            sys.settrace(tracer)
            try:
                err = _run_code(code, ns, CELL_TIMEOUT_S)
            finally:
                sys.settrace(None)
        if err:
            return {"error": err}
    after = {}
    for k, v in ns.items():
        if k.startswith("__"):
            continue
        try:
            after[k] = enc(v)
        except (NotPlain, RecursionError):
            continue
    return {"after": after, **seen}


def child_verify(job: dict) -> dict:
    """Replay verification: the original cells twice (determinism), then the rewritten last cell calling the
    function."""
    signal.signal(signal.SIGALRM, _on_alarm)
    if job.get("cwd"):
        os.chdir(job["cwd"])
    if job.get("path"):
        sys.path.insert(0, job["path"])
    snap = (job["cell"], job["line"])
    first = _run_cells(job["cells"], snap, job["names"], [])
    second = _run_cells(job["cells"], snap, job["names"], [])
    rewritten = [*job["cells"][:-1], {**job["cells"][-1], "code": job["rewritten"]}]
    third = _run_cells(rewritten, snap, job["names"], job.get("imports", []))
    return {"first": first, "second": second, "third": third}


def child_main() -> int:
    job = json.loads(sys.stdin.read())
    modes = {"replay": child_replay, "verify": child_verify}
    out = modes[job["mode"]](job)
    sys.stdout.write(json.dumps(out))
    return 0


# --- the host -------------------------------------------------------------------------------------------------


def box_argv(script_dir: Path, binds: list[tuple[str, str, bool]] = ()) -> list[str]:
    """bubblewrap around the system python: no network, read-only root, private /tmp, bounded processes; *binds*:
    ``(host path, box path, writable)``."""
    bwrap, prlimit = shutil.which("bwrap"), shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise RuntimeError(
            "bubblewrap and prlimit are required: the actor's code never runs outside a box",
        )
    args = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for link, target in (
        ("/lib", "usr/lib"),
        ("/lib64", "usr/lib64"),
        ("/bin", "usr/bin"),
        ("/sbin", "usr/sbin"),
    ):
        if os.path.islink(link):
            args += ["--symlink", target, link]
    args += [
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind", str(script_dir), "/s",
        *[x for src, dst, rw in binds for x in ("--bind" if rw else "--ro-bind", src, dst)],
        "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        prlimit, *BOX_LIMITS, "/usr/bin/python3", "-I", "/s/mining_behaviour.py", "--child",
    ]  # fmt: skip
    return args


def in_box(job: dict, timeout: float, binds: list[tuple[str, str, bool]] = ()) -> dict:
    argv = box_argv(Path(__file__).resolve().parent, binds)
    try:
        p = subprocess.run(
            argv,
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_mining.helper_env(),  # bwrap itself; the box then clears and sets its own
        )
    except subprocess.TimeoutExpired:
        return {"box_error": "box timeout"}
    if p.returncode:
        return {"box_error": f"box exit {p.returncode}: {p.stderr[-300:]}"}
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"box_error": "unreadable box output"}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--child"]:
        return child_main()
    sys.stderr.write("usage: python3 -I mining_behaviour.py --child  (a job as JSON on stdin)\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
