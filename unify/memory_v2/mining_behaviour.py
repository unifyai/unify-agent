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
        "obs": OBS_VERSION,
    }


#: The replay verifier's observation contract. Each run records its live-out values (every plain value its namespace
#: ends with, scalars and empty containers included), its stdout per cell and its working tree's file deltas as
#: ``[path, created | modified | deleted, sha256]``. Alias groups, deleted names and values that cannot be encoded
#: are recorded as unknown and never compared. Results of another version are never mixed with these.
OBS_VERSION = 2


def _tree_state(root: str | None) -> dict[str, str]:
    """Every regular file under *root* (a run's working tree), by relative path, with its sha256."""
    import hashlib

    out: dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return out
    for d, _dirs, files in os.walk(root):
        for f in files:
            fp = os.path.join(d, f)
            if os.path.islink(fp) or not os.path.isfile(fp):
                continue
            try:
                with open(fp, "rb") as fh:
                    out[os.path.relpath(fp, root)] = hashlib.sha256(
                        fh.read(),
                    ).hexdigest()
            except OSError:
                out[os.path.relpath(fp, root)] = "unreadable"
    return out


def _file_deltas(before: dict[str, str], after: dict[str, str]) -> list[list]:
    return sorted(
        [
            p,
            (
                "deleted"
                if p not in after
                else "created" if p not in before else "modified"
            ),
            after.get(p),
        ]
        for p in set(before) | set(after)
        if before.get(p) != after.get(p)
    )


def _stdout_digest(text: str) -> list:
    import hashlib

    return [
        len(text),
        hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(),
    ]


def _not_a_value(v) -> bool:
    """Code objects a namespace holds besides data (functions, classes, modules): neither values nor unknowns."""
    import types

    return callable(v) or isinstance(v, types.ModuleType)


def _reset_dir(path: str) -> None:
    for entry in os.listdir(path):
        p = os.path.join(path, entry)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                os.unlink(p)


def _sandbox_run():
    """The confined runner's module (its seccomp program and file-size bound), imported where it is needed so the
    box's child does not load it."""
    try:
        from . import sandbox_run
    except ImportError:  # run as a script beside sandbox_run.py
        import sandbox_run  # type: ignore[no-redef]
    return sandbox_run


def _run_cells(
    cells: list[dict],
    snap: tuple[int, int] | None,
    names: list[str],
    prelude: list[str],
    cwd: str | None = None,
) -> dict:
    """Replay *cells* in one namespace with *cwd* (the run's own working tree) as the working directory; at line
    *snap* = (cell index, line) record the names then bound and the values of *names*. The run's observation
    (:data:`OBS_VERSION`): ``after``, every plain value the namespace ends with, tagged; ``stdout``, each cell's
    ``[length, sha256]``; ``files``, the working tree's deltas; ``unknown``, what is recorded but not compared.
    """
    if cwd:
        os.chdir(cwd)
    files0 = _tree_state(cwd)
    ns: dict = {"__name__": "__main__", "__builtins__": builtins}
    for code in prelude:
        err = _run_code(compile(code, "<prelude>", "exec"), ns, RUN_TIMEOUT_S)
        if err:
            return {"error": f"prelude {err}"}
    seen: dict = {}
    stdout: list = []
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            sys.settrace(tracer)
            try:
                err = _run_code(code, ns, CELL_TIMEOUT_S)
            finally:
                sys.settrace(None)
        stdout.append(_stdout_digest(out.getvalue()))
        if err:
            return {"error": err}
    after, unobservable, ids = {}, [], {}
    for k, v in ns.items():
        if k.startswith("__") or _not_a_value(v):
            continue
        try:
            after[k] = enc(v)
        except (NotPlain, RecursionError):
            unobservable.append(k)
        if isinstance(v, (list, dict, set)):
            ids.setdefault(id(v), []).append(k)
    unknown = {
        "aliases": sorted(sorted(g) for g in ids.values() if len(g) > 1),
        "deleted": sorted(n for n in seen.get("bound", []) if n not in ns),
        "unobservable": sorted(unobservable),
    }
    return {
        "after": after,
        "stdout": stdout,
        "files": _file_deltas(files0, _tree_state(cwd)),
        "unknown": unknown,
        "obs": OBS_VERSION,
        **seen,
    }


def child_verify(job: dict) -> dict:
    """The replay verifier's three runs: the original cells twice (determinism), then the rewritten last cell calling
    the function. Each run starts in its own pristine copy of the working tree (``job["cwds"]``); in the box the
    private ``/tmp`` is emptied before each run, so no run sees another's writes."""
    signal.signal(signal.SIGALRM, _on_alarm)
    if job.get("path"):
        sys.path.insert(0, job["path"])
    cwds = list(job.get("cwds") or [job.get("cwd")] * 3)
    snap = (job["cell"], job["line"])
    rewritten = [*job["cells"][:-1], {**job["cells"][-1], "code": job["rewritten"]}]
    runs = (
        ("first", job["cells"], []),
        ("second", job["cells"], []),
        ("third", rewritten, job.get("imports", [])),
    )
    out = {}
    for (label, cells, prelude), cwd in zip(runs, cwds):
        if job.get("in_box"):
            os.chdir("/tmp")
            _reset_dir("/tmp")
        out[label] = _run_cells(cells, snap, job["names"], prelude, cwd)
    return out


def child_main() -> int:
    job = json.loads(sys.stdin.read())
    job["in_box"] = (
        True  # set only here, the box's entry point: in-process callers never reset a host directory
    )
    modes = {"replay": child_replay, "verify": child_verify}
    out = modes[job["mode"]](job)
    sys.stdout.write(json.dumps(out))
    return 0


# --- the host -------------------------------------------------------------------------------------------------


def box_argv(
    script_dir: Path,
    binds: list[tuple[str, str, bool]] = (),
    *,
    seccomp_fd: int,
) -> list[str]:
    """bubblewrap around the system python: no network, read-only root, private /tmp, bounded processes and file
    size, and the confined runner's seccomp filter read from *seccomp_fd* (sockets only AF_UNIX, AF_INET and
    AF_INET6, since a network namespace does not isolate AF_VSOCK; no new user namespaces; no io_uring); *binds*:
    ``(host path, box path, writable)``."""
    bwrap, prlimit = shutil.which("bwrap"), shutil.which("prlimit")
    if not bwrap or not prlimit:
        raise RuntimeError(
            "bubblewrap and prlimit are required: the actor's code never runs outside a box",
        )
    fsize = _sandbox_run().RLIMIT_FSIZE_BYTES
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
        "--seccomp", str(seccomp_fd), "--",
        prlimit, *BOX_LIMITS, f"--fsize={fsize}:{fsize}", "--",
        "/usr/bin/python3", "-I", "/s/mining_behaviour.py", "--child",
    ]  # fmt: skip
    return args


def in_box(job: dict, timeout: float, binds: list[tuple[str, str, bool]] = ()) -> dict:
    import platform
    import tempfile

    if platform.machine() != "x86_64":
        raise RuntimeError("the box's seccomp filter is built for x86_64 only")
    # An unlinked temporary file holds the filter; bubblewrap reads it from the inherited descriptor.
    with tempfile.TemporaryFile() as bpf:
        bpf.write(_sandbox_run().seccomp_program())
        bpf.flush()
        bpf.seek(0)
        argv = box_argv(Path(__file__).resolve().parent, binds, seccomp_fd=bpf.fileno())
        try:
            p = subprocess.run(
                argv,
                input=json.dumps(job),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_mining.helper_env(),  # bwrap itself; the box then clears and sets its own
                pass_fds=(bpf.fileno(),),
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
    sys.stderr.write(
        "usage: python3 -I mining_behaviour.py --child  (a job as JSON on stdin)\n",
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
