"""Memory v2.1 S0 (design r3): deterministic mining of the actor's repeated working code, before any model call.

For every recorded episode, the actor's Python cells that ran without error are parsed and normalised:

* names become ``v0, v1, …`` in order of first use within the unit, whether the unit binds them or an earlier cell
  did, except builtins and names imported anywhere in the episode (modules, the memory library, imported
  functions), which are kept, as are attribute and keyword names, since they carry the meaning;
* literals become numbered parameters ``p0, p1, …``: a constant, or a list, tuple, set or dict display made only of
  constants (one parameter, its value kept per instance); ``None``, ``True`` and ``False`` stay;
* docstrings are dropped (comments never reach the AST).

Units are hashed over the normalised AST: every statement subtree of at least :data:`MIN_NODES` nodes, every whole
cell of that size, and every run of :data:`WINDOW` consecutive such cells (SGDR's 2-5 step windows). A **cluster** is
a hash seen in at least :data:`MIN_EPISODES` independent episodes: episodes whose recorded request differs (the
request's bytes, hashed; never a task id). A cluster whose every instance lies inside an instance of a larger cluster
is dropped as implied by it.

Nothing here chooses what to store or filters anything: it finds shared shape, with every instance's pointer
(episode, cell, lines), its parameter values and its recorded output kept beside it. The raw episodes stay as they
are. Standard library only, so it also runs as a script on a worker without importing the package:

    python3 -I mining.py --git <episodes.git> [--rev main] --out <dir>

which writes ``clusters.json`` and ``episodes.json`` (per episode: request key, cells, successful cells, cells in a
cluster) under *dir*.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MIN_NODES = 12  # a unit smaller than this ("pass", "print(x)", a bare call) is not code worth a function
MIN_EPISODES = 2  # independent episodes a cluster needs
WINDOW = (2, 5)  # consecutive-cell runs considered, inclusive
_TRACEBACK = "Traceback (most recent call last)"
_BUILTINS = frozenset(dir(builtins))


@dataclass
class Cell:
    index: int
    code: str
    output: str = ""
    error: str | None = None
    language: str = "python"


@dataclass
class Instance:
    episode: str
    cell: int  # the first cell of a window
    lines: tuple[int, int]  # 1-based, inclusive; a window spans its cells: (1, cells)
    params: list
    output: str  # the recorded output of the instance's (last) cell


@dataclass
class Cluster:
    key: str
    kind: str  # "stmt" | "cell" | "window"
    size: int  # AST nodes of the normalised unit (a window: the sum over its cells)
    template: str  # the normalised source
    instances: list[Instance] = field(default_factory=list)
    requests: set[str] = field(default_factory=set)

    @property
    def episodes(self) -> int:
        return len(self.requests)


def successful(cell: Cell) -> bool:
    """A Python cell that ran without a recorded error and printed no traceback (what the actor's code did, not
    whether the request was solved)."""
    return (
        (cell.language or "python") == "python"
        and cell.error is None
        and _TRACEBACK not in (cell.output or "")
    )


def keep_names(trees: Iterable[ast.AST]) -> set[str]:
    """Names kept as written: builtins and every name imported in *trees* (an episode's cells)."""
    out = set(_BUILTINS)
    for tree in trees:
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    out.add((a.asname or a.name).split(".")[0])
    return out


def _literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return (
            node.value is not None
            and not isinstance(node.value, bool)
            and node.value is not Ellipsis
        )
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return bool(node.elts) and all(_literal(e) or _const(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return bool(node.keys) and all(
            k is not None and (_literal(k) or _const(k)) and (_literal(v) or _const(v))
            for k, v in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value,
            (int, float),
        )
    return False


def _const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant)  # None, True, False inside a display


class _Normaliser(ast.NodeTransformer):
    def __init__(self, keep: set[str]) -> None:
        self.keep, self.names, self.params = keep, {}, []

    def _param(self, node: ast.AST) -> ast.Name:
        try:
            value = ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            value = ast.unparse(node)
        self.params.append(
            (
                value
                if isinstance(value, (str, int, float, bool)) or value is None
                else _jsonable(value)
            ),
        )
        return ast.copy_location(
            ast.Name(id=f"p{len(self.params) - 1}", ctx=ast.Load()),
            node,
        )

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None  # a docstring or a bare string: no behaviour
        return self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        parts = [v for v in node.values if not isinstance(v, ast.Constant)]
        node.values = [
            self.visit(v) for v in parts
        ]  # the f-string's text is a literal; its fields stay
        return node

    def generic_visit(self, node: ast.AST):
        if (
            isinstance(node, ast.expr)
            and not isinstance(node, ast.Name)
            and _literal(node)
        ):
            return self._param(node)
        return super().generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        return self._param(node) if _literal(node) else node

    def _rename(self, name: str) -> str:
        if name in self.keep:
            return name
        if name not in self.names:
            self.names[name] = f"v{len(self.names)}"
        return self.names[name]

    def visit_Name(self, node: ast.Name):
        node.id = self._rename(node.id)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._rename(node.arg)
        node.annotation = None
        return node

    def _def(self, node):
        node.name = self._rename(node.name)
        node.returns = None
        node.decorator_list = []
        return self.generic_visit(node)

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _def

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            node.name = self._rename(node.name)
        return self.generic_visit(node)


def _jsonable(value):
    if isinstance(value, (list, tuple, set)):
        return [
            _jsonable(v)
            for v in (sorted(value, key=repr) if isinstance(value, set) else value)
        ]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return (
        value
        if isinstance(value, (str, int, float, bool)) or value is None
        else repr(value)
    )


def normalise(node: ast.AST, keep: set[str]) -> tuple[str, str, int, list]:
    """(hash, normalised source, size in AST nodes, parameter values) of *node*; names in *keep* stay as written."""
    norm = _Normaliser(keep)
    tree = norm.visit(copy.deepcopy(node))
    if tree is None:
        return "", "", 0, []
    if isinstance(tree, ast.stmt):
        tree = ast.Module(body=[tree], type_ignores=[])
    ast.fix_missing_locations(tree)
    dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    size = sum(1 for _ in ast.walk(tree))
    try:
        src = ast.unparse(tree)
    except Exception:  # noqa: BLE001 - a template that cannot be printed keeps its dump
        src = dump
    return hashlib.sha256(dump.encode()).hexdigest()[:16], src, size, norm.params


def _parse(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


@dataclass
class _Unit:
    key: str
    kind: str
    size: int
    template: str
    cell: int
    lines: tuple[int, int]
    params: list


def units(cell: Cell, keep: set[str] | None = None) -> list[_Unit]:
    """Every statement subtree and the whole cell of *cell*, normalised; [] for code that does not parse. *keep*:
    the episode's :func:`keep_names` (default: this cell's)."""
    tree = _parse(cell.code)
    if tree is None:
        return []
    keep = keep_names([tree]) if keep is None else keep
    out: list[_Unit] = []
    whole = normalise(tree, keep)
    if whole[2] >= MIN_NODES:
        last = max((getattr(n, "end_lineno", 1) or 1 for n in tree.body), default=1)
        out.append(
            _Unit(
                whole[0],
                "cell",
                whole[2],
                whole[1],
                cell.index,
                (1, last),
                whole[3],
            ),
        )
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt):
            key, src, size, params = normalise(n, keep)
            if size >= MIN_NODES and key != whole[0]:
                out.append(
                    _Unit(
                        key,
                        "stmt",
                        size,
                        src,
                        cell.index,
                        (n.lineno, n.end_lineno or n.lineno),
                        params,
                    ),
                )
    return out


def mine(
    episodes: Iterable[tuple[str, str, list[Cell]]],
) -> tuple[list[Cluster], dict[str, dict]]:
    """Clusters over *episodes* (``(episode_id, request_key, cells)``), largest first, and per-episode counts."""
    found: dict[str, Cluster] = {}
    per: dict[str, dict] = {}

    def add(u: _Unit, eid: str, req: str, output: str, kind: str | None = None) -> None:
        c = found.setdefault(u.key, Cluster(u.key, kind or u.kind, u.size, u.template))
        c.instances.append(Instance(eid, u.cell, u.lines, u.params, output))
        c.requests.add(req)

    for eid, req, cells in episodes:
        ok = [c for c in cells if successful(c)]
        per[eid] = {"request": req, "cells": len(cells), "successful_cells": len(ok)}
        keep = keep_names(t for t in (_parse(c.code) for c in cells) if t is not None)
        whole: list[_Unit | None] = []
        for c in ok:
            us = units(c, keep)
            for u in us:
                add(u, eid, req, c.output or "")
            whole.append(next((u for u in us if u.kind == "cell"), None))
        lo, hi = WINDOW
        for k in range(lo, hi + 1):
            for i in range(0, len(ok) - k + 1):
                run = whole[i : i + k]
                if any(u is None for u in run):
                    continue
                key = hashlib.sha256("|".join(u.key for u in run).encode()).hexdigest()[
                    :16
                ]
                tmpl = "\n# --- next cell ---\n".join(u.template for u in run)
                u = _Unit(
                    key,
                    "window",
                    sum(x.size for x in run),
                    tmpl,
                    run[0].cell,
                    (1, k),
                    [x.params for x in run],
                )
                add(u, eid, req, ok[i + k - 1].output or "")
    clusters = [c for c in found.values() if c.episodes >= MIN_EPISODES]
    clusters = _maximal(clusters)
    clusters.sort(key=lambda c: (-c.episodes, -c.size, c.key))
    in_cluster: dict[str, set[int]] = {}
    for c in clusters:
        for i in c.instances:
            span = range(i.cell, i.cell + (i.lines[1] if c.kind == "window" else 1))
            in_cluster.setdefault(i.episode, set()).update(span)
    for eid, row in per.items():
        row["cells_in_clusters"] = len(in_cluster.get(eid, ()))
    return clusters, per


def _covers(big: Cluster, small: Cluster) -> bool:
    """Every instance of *small* lies inside an instance of *big* (same episode, cell and enclosing lines)."""
    spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for i in big.instances:
        if big.kind == "window":
            for cell in range(i.cell, i.cell + i.lines[1]):
                spans.setdefault((i.episode, cell), []).append((1, 10**9))
        else:
            spans.setdefault((i.episode, i.cell), []).append(i.lines)
    for i in small.instances:
        cells = (
            range(i.cell, i.cell + i.lines[1]) if small.kind == "window" else [i.cell]
        )
        lines = (1, 10**9) if small.kind == "window" else i.lines
        for cell in cells:
            if not any(
                a <= lines[0] and lines[1] <= b
                for a, b in spans.get((i.episode, cell), [])
            ):
                return False
    return True


def _maximal(clusters: list[Cluster]) -> list[Cluster]:
    """Drop a cluster implied by a larger one: every instance inside it, in no more independent episodes."""
    order = sorted(clusters, key=lambda c: -c.size)
    kept: list[Cluster] = []
    for c in order:
        if not any(
            k.size > c.size and k.episodes >= c.episodes and _covers(k, c) for k in kept
        ):
            kept.append(c)
    return kept


# --- reading recorded episodes from a git copy (the script) ------------------------------------------------


def _git(repo: str, *args: str) -> bytes:
    return subprocess.run(
        ["git", "--git-dir", repo, *args],
        capture_output=True,
        check=True,
    ).stdout


def read_git(repo: str, rev: str = "main") -> list[tuple[str, str, list[Cell]]]:
    """``(episode_id, request_key, cells)`` for every episode folder at *rev*, in path order."""
    names = _git(repo, "ls-tree", "-r", "--name-only", rev).decode().split("\n")
    dirs = sorted({n.rsplit("/", 1)[0] for n in names if n.endswith("/cells.jsonl")})
    out = []
    for d in dirs:
        req = hashlib.sha256(_git(repo, "show", f"{rev}:{d}/request.json")).hexdigest()[
            :16
        ]
        cells = []
        for ln in (
            _git(repo, "show", f"{rev}:{d}/cells.jsonl")
            .decode(errors="replace")
            .splitlines()
        ):
            if not ln.strip():
                continue
            row = json.loads(ln)
            cells.append(
                Cell(
                    int(row.get("index", len(cells))),
                    str(row.get("code") or ""),
                    str(row.get("output") or ""),
                    row.get("error"),
                    str(row.get("language") or "python"),
                ),
            )
        out.append((d.rsplit("/", 1)[-1], req, cells))
    return out


def as_json(clusters: list[Cluster]) -> list[dict]:
    return [
        {
            "key": c.key,
            "kind": c.kind,
            "size": c.size,
            "episodes": c.episodes,
            "instances": len(c.instances),
            "template": c.template,
            "at": [
                {
                    "episode": i.episode,
                    "cell": i.cell,
                    "lines": list(i.lines),
                    "params": i.params,
                    "output_sha256": hashlib.sha256(i.output.encode()).hexdigest(),
                    "output_head": i.output[:400],
                }
                for i in c.instances
            ],
        }
        for c in clusters
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--git", required=True, help="a bare copy of an episodes repo")
    ap.add_argument("--rev", default="main")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    eps = read_git(a.git, a.rev)
    clusters, per = mine(eps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clusters.json").write_text(
        json.dumps(as_json(clusters), indent=1, default=str) + "\n",
    )
    (out / "episodes.json").write_text(json.dumps(per, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"episodes": len(eps), "clusters": len(clusters)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
