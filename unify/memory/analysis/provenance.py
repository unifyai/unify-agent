from __future__ import annotations

import ast
from typing import Any

from ..episodes import Cell
from .cells import CallSite, Hole


def _names(code: str) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    defs, uses = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            (defs if isinstance(n.ctx, ast.Store) else uses).add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            defs.update((a.asname or a.name).split(".")[0] for a in n.names)
    return defs, uses


def def_use_edges(cells: list[Cell]) -> set[tuple[int, int]]:
    last_def: dict[str, int] = {}
    edges: set[tuple[int, int]] = set()
    for c in cells:
        defs, uses = _names(c.code)
        for u in uses:
            if u in last_def and last_def[u] != c.index:
                edges.add((last_def[u], c.index))
        for d in defs:
            last_def[d] = c.index
    return edges


def _scalars(value: Any, out: set) -> None:
    if isinstance(value, dict):
        for v in value.values():
            _scalars(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _scalars(v, out)
    elif isinstance(value, (str, int)) and not isinstance(value, bool):
        if (isinstance(value, str) and len(value) >= 4) or (
            isinstance(value, int) and abs(value) >= 1000
        ):
            out.add(value)


def _parsed_scalars(output: str) -> set:
    found: set = set()
    for chunk in [output] + output.splitlines():
        try:
            _scalars(ast.literal_eval(chunk.strip()), found)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
    return found


def value_edges(cells: list[Cell], sites: list[CallSite]) -> set[tuple[int, int, str]]:
    """Exact value identity between a literal call argument and a scalar in an earlier cell's parsed output."""
    produced = {c.index: _parsed_scalars(c.output) for c in cells}
    edges = set()
    for s in sites:
        for v in list(s.kwargs.values()) + s.args:
            if (
                isinstance(v, Hole)
                or isinstance(v, bool)
                or not isinstance(v, (str, int))
            ):
                continue
            for idx in range(s.cell - 1, -1, -1):
                if v in produced.get(idx, ()):
                    edges.add((idx, s.order, "suspected"))
                    break
    return edges
