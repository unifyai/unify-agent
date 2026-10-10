"""Memory v2.1 r5 S2 (§4): verify a function by replaying the recorded instances it replaces (ASI, CRAFT).

A function item may say which pieces of the actor's recorded code it replaces: ``replaces: [{"episode", "cell",
"lines": [a, b], "call": "<statement>"}]`` in its manifest entry, where ``call`` is the statement that does the same
by calling the function (``comps = components(g, 8)``). For each instance, in a bubblewrap box with no network and
the writer's tree read-only on the import path:

1. the episode's successful cells up to and including that cell are replayed twice; if the two runs end with
   different values, the instance is ``not replayable: nondeterministic``;
2. they are replayed once more with lines ``a..b`` of the last cell replaced by ``call`` (the function imported);
3. the instance is **verified** when every value bound before line ``a`` and every name ``call`` assigns ends the
   same as in the original run. Otherwise it ``differs``.

The equivalence needs no reward signal: the recording is the oracle. A verified instance becomes a case of the
item's harness-written test (:func:`test_source`): the call's inputs as recorded at line ``a``, and its expected
results. Nothing staged or written by the model runs on the host.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
import shutil
from pathlib import Path
from typing import Callable

from . import mining_behaviour as _mb
from .mining import successful

_ITEM = re.compile(
    r"memory\.(?P<pkg>[a-z_][a-z0-9_]*)\.(?P<mod>[a-z_][a-z0-9_]*):(?P<fn>[A-Za-z_][A-Za-z0-9_]*)\Z",
)


def call_names(call: str) -> tuple[list[str], list[str]]:
    """(the names *call* reads, the names it assigns), builtins left out; ValueError when it is not one statement."""
    tree = ast.parse(call)
    if len(tree.body) != 1:
        raise ValueError("call must be one statement")
    stmt = tree.body[0]
    targets: list[str] = []
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        for t in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
            targets += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
    reads = []
    for n in ast.walk(stmt):
        if (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id not in reads
            and not hasattr(builtins, n.id)
        ):
            reads.append(n.id)
    return reads, targets


def rewrite(code: str, lines: tuple[int, int], call: str) -> str:
    rows = code.splitlines()
    a, b = lines
    if not (1 <= a <= b <= len(rows)):
        raise ValueError("lines outside the cell")
    indent = rows[a - 1][: len(rows[a - 1]) - len(rows[a - 1].lstrip())]
    return "\n".join([*rows[: a - 1], indent + call.strip(), *rows[b:]]) + "\n"


def _job(ep, inst: dict, fn_import: str, fn: str) -> dict:
    cell_idx, (a, b) = int(inst["cell"]), tuple(int(x) for x in inst["lines"])
    cells = [c for c in ep.cells if successful(c) and c.index <= cell_idx]
    if not cells or cells[-1].index != cell_idx:
        raise ValueError("the cell did not run without an error")
    reads, targets = call_names(str(inst["call"]))
    return {
        "mode": "verify",
        "cells": [{"index": c.index, "code": c.code} for c in cells],
        "cell": cell_idx,
        "line": a,
        "names": [n for n in reads if n != fn],
        "targets": targets,
        "rewritten": rewrite(cells[-1].code, (a, b), str(inst["call"])),
        "imports": [fn_import],
        "path": "/lib",
    }


def verdict(job: dict, got: dict) -> dict:
    """The instance's verdict from the three runs, and its test case when verified."""
    first, second, third = (
        got.get("first", {}),
        got.get("second", {}),
        got.get("third", {}),
    )
    for run, label in ((first, "original"), (third, "rewritten")):
        if "error" in run:
            return {"verdict": f"not replayable: {label} {run['error']}"}
    if first.get("after") != second.get("after"):
        return {"verdict": "not replayable: nondeterministic"}
    if "bound" not in first:
        return {"verdict": "not replayable: line not executed"}
    before = [n for n in first["bound"] if n in first["after"]]
    must = sorted(set(before) | set(job["targets"]))
    a1, a3 = first["after"], third.get("after", {})
    if any(n not in a1 or a3.get(n) != a1[n] for n in must):
        return {"verdict": "differs"}
    expect = {n: a1[n] for n in job["targets"]}
    expect |= {
        n: a1[n]
        for n in job["names"]
        if n in a1 and first.get("inputs", {}).get(n) != a1[n]
    }
    return {
        "verdict": "verified",
        "case": {
            "inputs": third.get("inputs", {}),
            "call": job["rewritten_call"],
            "expect": expect,
        },
    }


def verify(
    load: Callable,
    item: str,
    replaces: list,
    tree: Path,
    *,
    trees: dict[str, Path] | None = None,
    scratch: Path,
) -> list[dict]:
    """Each instance's verdict (``verified`` with its case, ``differs`` or ``not replayable: <cause>``)."""
    m = _ITEM.match(item)
    if not m:
        return [{"verdict": "refused: not a function item id"}]
    fn_import = f"from memory.{m['pkg']}.{m['mod']} import {m['fn']}"
    out = []
    for inst in replaces if isinstance(replaces, list) else []:
        at = (
            {k: inst.get(k) for k in ("episode", "cell", "lines")}
            if isinstance(inst, dict)
            else {}
        )
        try:
            ep = load(str(inst["episode"]))
            job = _job(ep, inst, fn_import, m["fn"])
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a bad entry is a verdict, never a crash
            out.append(
                {**at, "verdict": f"refused: {type(exc).__name__}: {str(exc)[:120]}"},
            )
            continue
        job["rewritten_call"] = str(inst["call"]).strip()
        binds = [(str(tree), "/lib", False)]
        run = None
        if trees and str(inst["episode"]) in trees:
            run = scratch / f"w-{inst['episode']}"
            shutil.rmtree(run, ignore_errors=True)
            shutil.copytree(trees[str(inst["episode"])], run, symlinks=True)
            binds.append((str(run), "/w", True))
            job["cwd"] = "/w"
        try:
            got = _mb.in_box(job, 900, binds)
        finally:
            if run is not None:
                shutil.rmtree(run, ignore_errors=True)
        if "box_error" in got:
            out.append({**at, "verdict": f"not replayable: {got['box_error'][:120]}"})
            continue
        out.append({**at, **verdict(job, got)})
    return out


_TEST = '''"""Harness-written (memory v2.1 S2): {item} reproduces the recorded instances it replaces."""

import json

from memory.{pkg}.{mod} import {fn}

CASES = json.loads({cases!r})


def _dec(d, frozen=False):
    if isinstance(d, list):
        return [_dec(v) for v in d]
    if isinstance(d, dict):
        t, v = d.get("t"), d.get("v")
        if t == "t":
            return tuple(_dec(x, frozen) for x in v)
        if t == "s":
            s = [_dec(x, True) for x in v]
            return frozenset(s) if frozen else set(s)
        if t == "d":
            return {{_dec(k, True): _dec(x) for k, x in v}}
        if t == "f":
            return float(v)
        if t == "dec":
            import decimal
            return decimal.Decimal(v)
        if t in ("date", "dt"):
            import datetime
            return (datetime.date if t == "date" else datetime.datetime).fromisoformat(v)
    return d


def test_{fn}_reproduces_its_recorded_instances():
    for case in CASES:
        ns = {{"{fn}": {fn}}}
        ns.update({{k: _dec(v) for k, v in case["inputs"].items()}})
        exec(case["call"], ns)
        for name, want in case["expect"].items():
            assert ns[name] == _dec(want), (case["at"], name)
'''


def test_source(item: str, cases: list[dict]) -> tuple[str, str]:
    """(the test's path in the tree, its source) for *item*'s verified *cases* (each with ``at``)."""
    m = _ITEM.match(item)
    if not m:
        raise ValueError(item)
    rel = f"memory/{m['pkg']}/tests/test_{m['fn'].lower()}_replay.py"
    return rel, _TEST.format(
        item=item,
        pkg=m["pkg"],
        mod=m["mod"],
        fn=m["fn"],
        cases=json.dumps(cases, sort_keys=True),
    )
