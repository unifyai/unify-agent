"""Memory v2.1 S0 tier (b) (design r4 §4): behavioural equivalence of the actor's code across episodes.

Tier (a) (:mod:`.mining`) clusters code that is the same after renaming names and lifting literals. Code that does
the same thing but is written differently (a flood fill over direction tuples or over a ``dr, dc`` loop) falls into
separate tier-(a) clusters. Tier (b) runs the code instead of reading it (observational equivalence, as in
DreamCoder and TroVE):

1. **Replay.** Each episode's successful cells are re-run in order, in a fresh interpreter inside a bubblewrap box
   (no network, read-only root, a private ``/tmp``, process limits, a per-cell alarm). At the first execution of
   each substantive unit (a tier-(a) statement or cell of at least :data:`MIN_NODES` AST nodes) the values of its
   free names are snapshotted. Only plain data (numbers including ``Decimal``, strings, dates, lists, tuples,
   dicts, sets) leaves the box, as tagged JSON; the host never unpickles or executes anything it receives.
2. **Cross-run.** Every unit is run, again in a box, on other units' snapshots: its free names are bound to the other
   unit's values by type (a grid to a grid, an int to an int, trying orders when a type repeats), and its literal
   parameters (numbers other than -1, 0 and 1, and strings) are bound to its own values, to the other unit's in
   order, or (one parameter) to each of the other's of the same type. A run's **effect** is every non-empty
   container it created or changed. Two units match when one reproduces the other's effect on the other's inputs,
   exactly or up to the order of list elements, in **both directions** (each on the other's inputs: two input
   sets). A one-way match is counted, never joined: on one input set, unrelated grid edits can agree by chance.
3. **Cluster.** Matches are joined (union-find); a behavioural cluster spans at least two episodes with different
   requests. Each is compared with tier (a): which tier-(a) clusters it merges, and which cross-episode clusters
   tier (a) missed.

Code that reads its working directory (office tasks) is replayed in the episode's recorded workspace: with
``--worktree <worktree.git>`` each episode's ``worktree_before`` is exported once; the replay gets a fresh,
disposable copy of it as its working directory (``/w``, writable, so later cells see what earlier ones wrote), and
cross-runs see every exported tree read-only, each run starting in the tree of the episode whose inputs it uses.
The recorded export itself is never writable.

Units that cannot take part are reported by cause, never forced: not standalone (``return``, ``break`` or ``await``
outside their context), not reached in the replay (with the exception that stopped the cell), non-plain inputs,
timeouts, no data effect. Standard library only; runnable as a script beside ``mining.py``:

    python3 -I mining_behaviour.py --git <episodes.git> --out <dir> [--jobs N] [--worktree <worktree.git>]
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import builtins
import contextlib
import datetime as _dt
import decimal
import io
import itertools
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from . import mining as _mining
except ImportError:  # run as a script beside mining.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mining as _mining  # type: ignore[no-redef]

MIN_NODES = 40  # the substantive units of the step-0 report
CELL_TIMEOUT_S = 5.0  # one replayed cell (operational: a hung loop must end)
RUN_TIMEOUT_S = 2.0  # one cross-run
MAX_ELEMENTS = 200_000  # a snapshot larger than this is reported, not copied (memory safety in the box)
MAX_BINDINGS = 24  # orders tried when a type repeats among a unit's inputs
PAIRS_PER_CHILD = 400
_STRUCTURAL = (-1, 0, 1)
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


def canonical(value, unordered: bool = False) -> str:
    def norm(x):
        if isinstance(x, list):
            items = [norm(v) for v in x]
            return (
                sorted(items, key=lambda y: json.dumps(y, sort_keys=True))
                if unordered
                else items
            )
        if isinstance(x, dict):
            if x.get("t") in ("t",):
                return {"t": "t", "v": norm(x["v"])}
            if x.get("t") == "d":
                pairs = [[norm(k), norm(v)] for k, v in x["v"]]
                return {
                    "t": "d",
                    "v": sorted(pairs, key=lambda y: json.dumps(y, sort_keys=True)),
                }
            return x
        return x

    return json.dumps(norm(enc(value)), sort_keys=True)


def kind(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float, decimal.Decimal)):
        return "num"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "grid" if value and all(isinstance(r, list) for r in value) else "list"
    return type(value).__name__


# --- units: source, free names, lifted parameters ---------------------------------------------------------


def free_names(tree: ast.AST, keep: set[str]) -> list[str]:
    """Names *tree* reads before it binds them, in source order (comprehension, lambda and argument names
    excluded); names in *keep* (builtins, imports) are not inputs."""
    inner: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.comprehension):
            inner |= {x.id for x in ast.walk(n.target) if isinstance(x, ast.Name)}
        elif isinstance(n, ast.arg):
            inner.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            inner.add(n.name)
    seen: dict[str, str] = {}
    for n in sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Name)),
        key=lambda n: (n.lineno, n.col_offset),
    ):
        if n.id in keep or n.id in inner or n.id in seen:
            continue
        seen[n.id] = "load" if isinstance(n.ctx, ast.Load) else "store"
    return [k for k, v in seen.items() if v == "load"]


class _Lift(ast.NodeTransformer):
    """Literal parameters (numbers other than -1, 0, 1; strings) become names ``__q0, __q1, …``."""

    def __init__(self) -> None:
        self.values: list = []

    def visit_JoinedStr(self, node):
        return node  # f-string text stays

    def visit_Constant(self, node):
        v = node.value
        lift = (isinstance(v, str) and v) or (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and v not in _STRUCTURAL
        )
        if not lift:
            return node
        self.values.append(v)
        return ast.copy_location(
            ast.Name(id=f"__q{len(self.values) - 1}", ctx=ast.Load()),
            node,
        )


def lift(src: str) -> tuple[str, list]:
    tree = ast.parse(src)
    t = _Lift()
    tree = ast.fix_missing_locations(t.visit(tree))
    return ast.unparse(tree), t.values


def unit_source(code: str, lines: tuple[int, int]) -> str:
    return textwrap.dedent("\n".join(code.splitlines()[lines[0] - 1 : lines[1]]))


def standalone(src: str) -> str | None:
    """None when *src* compiles as a module on its own; else why not."""
    try:
        compile(src, "<unit>", "exec")
    except SyntaxError as exc:
        return f"not standalone: {exc.msg}"
    return None


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


def _effects(before: dict, after: dict, unordered: bool) -> list[str] | None:
    out = []
    for name, v in after.items():
        if (
            name.startswith("__")
            or not isinstance(v, (list, tuple, dict, set, frozenset))
            or not v
        ):
            continue
        try:
            if name in before and before[name] == canonical(v):
                continue  # an input the run left unchanged
            out.append(canonical(v, unordered))
        except (NotPlain, RecursionError):
            continue
    return sorted(out) if out else None


def _bindings(names: list[str], values: dict) -> list[dict]:
    """Ways to bind *names* (their types from their own snapshot, in ``values['__own']``) to the other unit's
    values by type; [] when some type has too few values."""
    own, other = values["__own"], values["__other"]
    by_kind: dict[str, list[str]] = {}
    for n, v in other.items():
        by_kind.setdefault(kind(v), []).append(n)
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(kind(own[n]), []).append(n)
    choices = []
    for k, ns in groups.items():
        pool = by_kind.get(k, [])
        if len(pool) < len(ns):
            return []
        choices.append(
            [
                list(zip(ns, p))
                for p in itertools.islice(
                    itertools.permutations(pool, len(ns)),
                    MAX_BINDINGS,
                )
            ],
        )
    out = []
    for combo in itertools.islice(itertools.product(*choices), MAX_BINDINGS):
        out.append({a: b for pairs in combo for a, b in pairs})
    return out


def _param_sets(own: list, other: list) -> list[list]:
    sets = [list(own)]
    if (
        other
        and len(other) == len(own)
        and [type(x) for x in other] == [type(x) for x in own]
        and other != own
    ):
        sets.append(list(other))
    if len(own) == 1:
        sets += [[v] for v in other if type(v) is type(own[0]) and v != own[0]][:8]
    return sets


def _execute(
    unit: dict,
    inputs: dict,
    params: list,
    timeout: float,
    cwd: str | None = None,
):
    if cwd:
        os.chdir(cwd)
    ns: dict = {"__name__": "__main__", "__builtins__": builtins}
    for imp in unit["imports"]:
        _run_code(compile(imp, "<import>", "exec"), ns, timeout)
    ns.update({k: dec(v) for k, v in inputs.items()})
    ns.update({f"__q{i}": v for i, v in enumerate(params)})
    before = {}
    for k, v in ns.items():
        if not k.startswith("__") and isinstance(
            v,
            (list, tuple, dict, set, frozenset),
        ):
            try:
                before[k] = canonical(v)
            except (NotPlain, RecursionError):
                pass
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        err = _run_code(compile(unit["lifted"], "<unit>", "exec"), ns, timeout)
    return err, ns, before


def child_cross(job: dict) -> dict:
    """Reference effects of each unit on its own inputs, then each pair (u on w's inputs)."""
    signal.signal(signal.SIGALRM, _on_alarm)
    units, snaps = job["units"], job["snaps"]
    cwd = job.get(
        "cwd_of",
        {},
    )  # unit id -> the working directory of its episode's tree (office)
    ref: dict[str, dict] = {}
    for uid in {x for p in job["pairs"] for x in p}:
        u = units[uid]
        err, ns, before = _execute(
            u,
            snaps[uid],
            u["params"],
            RUN_TIMEOUT_S,
            cwd.get(uid),
        )
        ref[uid] = {
            "err": err,
            "exact": _effects(before, ns, False) if not err else None,
            "unordered": _effects(before, ns, True) if not err else None,
        }
    results = []
    for a, b in job["pairs"]:
        ua, ub = units[a], units[b]
        target = ref[b]
        if target["err"] or not target["exact"]:
            results.append([a, b, "reference " + (target["err"] or "no-effect")])
            continue
        names = [n for n in ua["names"] if n in snaps[a]]
        binds = _bindings(
            names,
            {
                "__own": {n: dec(snaps[a][n]) for n in names},
                "__other": {n: dec(v) for n, v in snaps[b].items()},
            },
        )
        if not binds:
            results.append([a, b, "no-binding"])
            continue
        verdict = "differ"
        for bind, params in itertools.product(
            binds,
            _param_sets(ua["params"], ub["params"]),
        ):
            inputs = {n: snaps[b][bind[n]] for n in names}
            err, ns, before = _execute(ua, inputs, params, RUN_TIMEOUT_S, cwd.get(b))
            if err:
                verdict = (
                    "timeout" if err == "timeout" and verdict == "differ" else verdict
                )
                continue
            if _effects(before, ns, False) == target["exact"]:
                verdict = "exact"
                break
            if _effects(before, ns, True) == target["unordered"]:
                verdict = "unordered"
        results.append([a, b, verdict])
    return {
        "results": results,
        "ref": {
            k: (v["err"] or ("ok" if v["exact"] else "no-effect"))
            for k, v in ref.items()
        },
    }


def _run_cells(
    cells: list[dict],
    snap: tuple[int, int] | None,
    names: list[str],
    prelude: list[str],
) -> dict:
    """Replay *cells* in one namespace; at line *snap* = (cell index, line) record the names then bound and the
    values of *names*. The namespace's plain values after the last cell, tagged (S2's verification).
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
    """S2 (r5 §4): the original cells twice (determinism), then the rewritten last cell calling the function."""
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
    modes = {"replay": child_replay, "cross": child_cross, "verify": child_verify}
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
        )
    except subprocess.TimeoutExpired:
        return {"box_error": "box timeout"}
    if p.returncode:
        return {"box_error": f"box exit {p.returncode}: {p.stderr[-300:]}"}
    try:
        return json.loads(p.stdout)
    except ValueError:
        return {"box_error": "unreadable box output"}


def collect_units(episodes) -> tuple[dict, dict, dict]:
    """(units by id, replay jobs by episode, causes for units that cannot run) for every substantive unit."""
    units, jobs, causes = {}, {}, {}
    for eid, req, cells in episodes:
        trees = [t for t in (_mining._parse(c.code) for c in cells) if t is not None]
        keep = _mining.keep_names(trees)
        imports = sorted(
            {
                ast.unparse(n)
                for t in trees
                for n in t.body
                if isinstance(n, (ast.Import, ast.ImportFrom))
            },
        )
        targets: dict[str, list] = {}
        ok = [c for c in cells if _mining.successful(c)]
        for c in ok:
            for u in _mining.units(c, keep):
                if u.size < MIN_NODES or u.kind == "window":
                    continue
                uid = f"{eid}:{c.index}:{u.lines[0]}-{u.lines[1]}"
                src = unit_source(c.code, u.lines)
                why = standalone(src)
                if why:
                    causes[uid] = why
                    continue
                lifted, params = lift(src)
                tree = ast.parse(src)
                names = free_names(tree, keep)
                first = u.lines[0] + (tree.body[0].lineno - 1 if tree.body else 0)
                units[uid] = {
                    "uid": uid, "episode": eid, "request": req, "cell": c.index, "lines": list(u.lines),
                    "akey": u.key, "size": u.size, "src": src, "lifted": lifted, "params": params,
                    "names": names, "imports": imports,
                }  # fmt: skip
                targets.setdefault(str(c.index), []).append(
                    {"uid": uid, "line": first, "names": names},
                )
        if targets:
            jobs[eid] = {
                "mode": "replay",
                "cells": [
                    {"index": c.index, "code": c.code, "output": c.output} for c in ok
                ],
                "targets": targets,
            }
    return units, jobs, causes


def export_trees(
    worktree: str,
    episodes_git: str,
    rev: str,
    eids: list[str],
    dest: Path,
) -> dict[str, Path]:
    """Each episode's recorded ``worktree_before`` exported once under *dest* (episode id -> tree)."""
    names = (
        _mining._git(episodes_git, "ls-tree", "-r", "--name-only", rev)
        .decode()
        .split("\n")
    )
    dirs = {
        n.rsplit("/", 2)[-2]: n.rsplit("/", 1)[0]
        for n in names
        if n.endswith("/meta.json")
    }
    out = {}
    for eid in eids:
        if eid not in dirs:
            continue
        meta = json.loads(
            _mining._git(episodes_git, "show", f"{rev}:{dirs[eid]}/meta.json"),
        )
        sha = meta.get("worktree_before")
        if not sha:
            continue
        tree = dest / eid
        if not tree.exists():
            tree.mkdir(parents=True)
            data = subprocess.run(
                ["git", "--git-dir", worktree, "archive", sha],
                capture_output=True,
                check=True,
            ).stdout
            subprocess.run(["tar", "x", "-C", str(tree)], input=data, check=True)
        out[eid] = tree
    return out


def _signature(names: list[str], snap: dict) -> tuple:
    return tuple(sorted(kind(dec(snap[n])) for n in names if n in snap))


def candidate_pairs(units: dict, snaps: dict) -> list[tuple[str, str]]:
    """Ordered pairs (u, w) from episodes with different requests whose input types fit (u's types within w's)."""
    sig = {uid: _signature(units[uid]["names"], snaps[uid]) for uid in snaps}
    pairs = []
    ids = sorted(snaps)
    for a in ids:
        need = sig[a]
        if not need:
            continue
        for b in ids:
            if units[a]["request"] == units[b]["request"]:
                continue
            have = list(sig[b])
            if all(k in have and not have.remove(k) for k in need):
                pairs.append((a, b))
    return pairs


class _UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _contained(small: list[dict], big: list[dict]) -> bool:
    spans = {}
    for u in big:
        spans.setdefault((u["episode"], u["cell"]), []).append(u["lines"])
    return all(
        any(
            a <= u["lines"][0] and u["lines"][1] <= b
            for a, b in spans.get((u["episode"], u["cell"]), [])
        )
        for u in small
    )


def analyse(units: dict, snaps: dict, results: list, a_clusters: set[str]) -> dict:
    """Behavioural clusters from the pair verdicts, compared with tier (a)."""
    verdict = {(a, b): v for a, b, v in results}
    uf = _UF()
    edges = []
    one_way = 0
    match = ("exact", "unordered")
    for a, b in sorted({tuple(sorted(k)) for k in verdict}):
        fwd, back = verdict.get((a, b)), verdict.get((b, a))
        if fwd not in match and back not in match:
            continue
        if fwd in match and back in match:  # both directions: two input sets agree
            uf.union(a, b)
            edges.append((a, b, fwd, back))
        else:
            one_way += 1
    groups: dict[str, list[str]] = {}
    for a, b, _, _ in edges:
        for x in (a, b):
            groups.setdefault(uf.find(x), [])
            if x not in groups[uf.find(x)]:
                groups[uf.find(x)].append(x)
    clusters = []
    for members in groups.values():
        us = [units[m] for m in members]
        if len({u["request"] for u in us}) < 2:
            continue
        akeys = {u["akey"] for u in us}
        by_a: dict[str, set] = {}
        for u in us:
            by_a.setdefault(u["akey"], set()).add(u["request"])
        clusters.append(
            {
                "units": sorted(members),
                "episodes": len({u["request"] for u in us}),
                "akeys": sorted(akeys),
                "a_clusters_merged": sorted(k for k in akeys if k in a_clusters),
                "new_vs_a": not any(len(r) >= 2 for r in by_a.values()),
                "max_size": max(u["size"] for u in us),
            },
        )
    # drop a behavioural cluster whose units all lie inside another's (the inner statement of a matched loop)
    clusters.sort(key=lambda c: -c["max_size"])
    kept = []
    for c in clusters:
        inner = [units[m] for m in c["units"]]
        if not any(
            _contained(inner, [units[m] for m in k["units"]])
            and k["episodes"] >= c["episodes"]
            for k in kept
        ):
            kept.append(c)
    kept.sort(key=lambda c: (-c["episodes"], -c["max_size"]))
    return {"clusters": kept, "edges": len(edges), "one_way": one_way}


def run_behaviour(
    episodes,
    out: Path,
    *,
    jobs: int,
    trees: dict[str, Path] | None = None,
    batch: set[str] | None = None,
    cached: dict[str, dict] | None = None,
) -> tuple[dict, dict[str, dict]]:
    """Tier (b) over *episodes* (``(episode_id, request_key, cells)``): (the behaviour document, each newly
    replayed episode's replay result). *cached* replay results (by episode) are reused, so an episode is
    replayed once in its lifetime; with *batch*, only pairs with a unit from a batch episode are cross-run.
    *trees* (episode -> its exported recorded workspace) replays office code in place.
    """
    trees = trees or {}
    cached = dict(cached or {})
    out.mkdir(parents=True, exist_ok=True)
    units, jobs_by_ep, causes = collect_units(episodes)
    reqs_by_key: dict[str, set] = {}
    for u in units.values():
        reqs_by_key.setdefault(u["akey"], set()).add(u["request"])
    a_clusters = {
        k for k, r in reqs_by_key.items() if len(r) >= 2
    }  # tier (a)'s recurring keys
    for eid in trees:
        if eid in jobs_by_ep:
            jobs_by_ep[eid]["cwd"] = "/w"

    def replay(eid):
        if eid not in trees:
            return in_box(jobs_by_ep[eid], 600)
        run = (
            out / "run" / eid
        )  # a fresh, disposable copy: the export is never writable
        shutil.rmtree(run, ignore_errors=True)
        shutil.copytree(trees[eid], run, symlinks=True)
        try:
            return in_box(jobs_by_ep[eid], 600, [(str(run), "/w", True)])
        finally:
            shutil.rmtree(run, ignore_errors=True)

    todo = [e for e in jobs_by_ep if e not in cached]
    fresh: dict[str, dict] = {}
    with ThreadPoolExecutor(jobs) as pool:
        for eid, r in zip(todo, pool.map(replay, todo)):
            if "box_error" in r:
                r = {
                    "snaps": {},
                    "causes": {
                        t["uid"]: r["box_error"]
                        for t in itertools.chain.from_iterable(
                            jobs_by_ep[eid]["targets"].values(),
                        )
                    },
                    "fidelity": [0, 0],
                }
            fresh[eid] = cached[eid] = r
    snaps: dict[str, dict] = {}
    fidelity = [0, 0]
    for eid in jobs_by_ep:
        r = cached.get(eid) or {}
        snaps.update({u: v for u, v in (r.get("snaps") or {}).items() if u in units})
        causes.update({u: v for u, v in (r.get("causes") or {}).items() if u in units})
        fidelity[0] += (r.get("fidelity") or [0, 0])[0]
        fidelity[1] += (r.get("fidelity") or [0, 0])[1]
    pairs = candidate_pairs(units, snaps)
    if batch is not None:
        pairs = [
            (x, y)
            for x, y in pairs
            if units[x]["episode"] in batch or units[y]["episode"] in batch
        ]
    chunks = [
        pairs[i : i + PAIRS_PER_CHILD] for i in range(0, len(pairs), PAIRS_PER_CHILD)
    ]
    tree_root = out / "trees"
    if trees:
        tree_root.mkdir(parents=True, exist_ok=True)
        for eid, t in trees.items():
            link = tree_root / eid
            if not link.exists() and Path(t).resolve() != link.resolve():
                shutil.copytree(t, link, symlinks=True)

    def cross(chunk):
        ids = {x for p in chunk for x in p}
        job = {
            "mode": "cross",
            "units": {i: units[i] for i in ids},
            "snaps": {i: snaps[i] for i in ids},
            "pairs": chunk,
        }
        if not trees:
            return in_box(job, 1800)
        job["cwd_of"] = {
            i: f"/trees/{units[i]['episode']}"
            for i in ids
            if units[i]["episode"] in trees
        }
        return in_box(job, 1800, [(str(tree_root), "/trees", False)])

    results, ref = [], {}
    with ThreadPoolExecutor(jobs) as pool:
        for chunk, r in zip(chunks, pool.map(cross, chunks)):
            if "box_error" in r:
                results += [[x, y, r["box_error"]] for x, y in chunk]
                continue
            results += r["results"]
            ref.update(r["ref"])
    report = analyse(units, snaps, results, a_clusters)
    for uid, why in ref.items():
        if why not in ("ok",) and uid not in causes:
            causes[uid] = "reference run: " + why
    counts: dict[str, int] = {}
    for why in causes.values():
        key = (
            why.split(":")[0] if why.startswith(("box exit", "not standalone")) else why
        )
        counts[key] = counts.get(key, 0) + 1
    verdicts: dict[str, int] = {}
    for _, _, v in results:
        verdicts[v] = verdicts.get(v, 0) + 1
    summary = {
        "episodes": len(episodes),
        "units": len(units)
        + sum(1 for w in causes.values() if w.startswith("not standalone")),
        "replayed_with_inputs": len(snaps),
        "cell_replay_stdout_matches_record": fidelity,
        "pairs": len(pairs),
        "verdicts": verdicts,
        "matched_pairs_both_ways": report["edges"],
        "matched_one_way_only": report["one_way"],
        "behavioural_clusters": len(report["clusters"]),
        "a_clusters_merged": len(
            {
                k
                for c in report["clusters"]
                if len(c["akeys"]) > 1
                for k in c["a_clusters_merged"]
            },
        ),
        "clusters_merging_a": sum(1 for c in report["clusters"] if len(c["akeys"]) > 1),
        "new_vs_a": sum(1 for c in report["clusters"] if c["new_vs_a"]),
        "not_runnable_by_cause": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }
    doc = {
        "summary": summary,
        "clusters": report["clusters"],
        "units": units,
        "causes": causes,
        "results": results,
    }
    return doc, fresh


def main(argv: list[str] | None = None) -> int:
    if argv is None and "--child" in sys.argv:
        return child_main()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--git", required=True)
    ap.add_argument("--rev", default="main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument(
        "--worktree",
        help="a copy of the run's worktree repo: replay in each episode's recorded tree",
    )
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = _mining.read_git(a.git, a.rev)
    trees: dict[str, Path] = {}
    if a.worktree:
        trees = export_trees(
            a.worktree,
            a.git,
            a.rev,
            [e for e, _, _ in episodes],
            out / "trees",
        )
    doc, _ = run_behaviour(episodes, out, jobs=a.jobs, trees=trees)
    (out / "behaviour.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps(doc["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
