"""Normalised code units of the actor's recorded Python cells: the shared basis for finding the same code in
different episodes (the ``similar`` tool's index) and for replay verification.

A cell is parsed and normalised into units: the whole cell and every statement subtree of at least
:data:`MIN_NODES` AST nodes. Normalising a unit:

* names become ``v0, v1, …`` in order of first use within the unit, except builtins and names the unit itself
  imports, which are kept, as are attribute and keyword names, since they carry the meaning. Which names are kept
  depends only on the unit's own text, never on what other cells of its episode import, so the same code hashes the
  same in every episode;
* literals become numbered parameters ``p0, p1, …``: a constant, or a list, tuple, set or dict display made only of
  constants (one parameter, its value kept per instance); ``None``, ``True`` and ``False`` stay;
* docstrings are dropped (comments never reach the AST).

A unit's key is a hash of its normalised AST. Standard library only: the replay box imports this module by path.
"""

from __future__ import annotations

import ast
import builtins
import copy
import hashlib
from dataclasses import dataclass
from typing import Callable

MIN_NODES = 12  # a unit smaller than this ("pass", "print(x)", a bare call) is not code worth a function
_TRACEBACK = "Traceback (most recent call last)"
_BUILTINS = frozenset(dir(builtins))


@dataclass
class Cell:
    index: int
    code: str
    output: str = ""
    error: str | None = None
    language: str = "python"


def successful(cell: Cell) -> bool:
    """A Python cell that ran without a recorded error and printed no traceback (what the actor's code did, not
    whether the request was solved)."""
    return (
        (cell.language or "python") == "python"
        and cell.error is None
        and _TRACEBACK not in (cell.output or "")
    )


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


def own_keep(node: ast.AST) -> set[str]:
    """The names *node* keeps as written: the names it imports itself, and the builtins it does not rebind (an
    assignment, argument, definition or handler name is a variable). Text only: the same code gets the same set in
    any episode."""
    imported: set[str] = set()
    bound: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                imported.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            bound.add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return (set(_BUILTINS) - bound) | imported


def units(cell: Cell, stop: Callable[[], bool] | None = None) -> list[_Unit]:
    """The whole cell and every statement subtree of *cell* of at least :data:`MIN_NODES` nodes, normalised, each
    with the names it keeps from its own text (:func:`own_keep`); [] for code that does not parse. *stop*, checked
    between units, ends the walk early (the units found so far are returned)."""
    tree = _parse(cell.code)
    if tree is None:
        return []
    out: list[_Unit] = []
    whole = normalise(tree, own_keep(tree))
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
        if stop is not None and stop():
            break
        if isinstance(n, ast.stmt):
            key, src, size, params = normalise(n, own_keep(n))
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


