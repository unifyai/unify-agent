"""Syntax helpers for recorded procedures: an entry function, a literal value, a ``return`` that hands back a value.

Nothing here judges a procedure. The one syntax-only judgement serving reads, record or procedure, is
:func:`.capture_shape.record_kind` (a capture that never reads its input, or returns only literals, is served
as a record of what was done last time); no ranking, trust, grouping or gate rests on it. The helpers read syntax only
and run nothing: :func:`_entry` (the entry function: ``run``, else ``main``, else the only
function, else the first public one), :func:`is_literal`, :func:`returns_a_value` (used by the writer's own test
labels) and :data:`ACTION_SURFACES` (the harness's action namespaces, as the shape address reads calls).
"""

from __future__ import annotations

import ast
from typing import Any

#: The harness's own action surfaces: a call through one of them is an action, never part of a stored answer.
ACTION_SURFACES = frozenset({"tools", "primitives"})


def _body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """*node*'s body without its docstring."""
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def is_literal(node: ast.AST | None) -> bool:
    """Whether *node* is a literal: a constant, a container of literals, a negated literal or an f-string without
    fields."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(is_literal(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is None or is_literal(k) for k in node.keys) and all(is_literal(v) for v in node.values)
    if isinstance(node, ast.JoinedStr):
        return all(isinstance(x, ast.Constant) for x in node.values)
    if isinstance(node, ast.UnaryOp):
        return is_literal(node.operand)
    return False


def _entry(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    fns = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if not fns:
        return None
    if "run" in fns:
        return fns["run"]
    if "main" in fns:
        return fns["main"]
    if len(fns) == 1:
        return next(iter(fns.values()))
    public = [f for name, f in fns.items() if not name.startswith("_")]
    return public[0] if public else next(iter(fns.values()))


def _gives_value(node: ast.Return) -> bool:
    """Whether a ``return`` hands back a value: not bare, and not the constant ``None``."""
    return node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None)


def returns_a_value(module_source: Any, name: str) -> bool:
    """Whether the function *name* defined at the top of *module_source* hands back a value on some ``return``
    (:func:`_gives_value`); False when every return is bare or ``None``, or there is none (its outcome is what it
    did). An unreadable module or an absent function counts as handing one back, so a rule about results still
    applies to it."""
    try:
        tree = ast.parse(module_source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return True
    fn = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name),
        None,
    )
    if fn is None:
        return True
    return any(_gives_value(n) for stmt in _body(fn) for n in ast.walk(stmt) if isinstance(n, ast.Return))
