"""Deterministic AST mutation operators for the gate's mutation check (memory v2.1 stage 5).

A *site* is one place in one function's body where one operator applies. :func:`sites` lists them in a fixed
order (each statement of the function's body walked breadth first by :func:`ast.walk`), :func:`choose` picks a
bounded subset by a seeded hash, spread over the operator kinds, and :func:`apply` returns the whole module
re-printed by :func:`ast.unparse` with that one change. :func:`reprint` is the same module re-printed with no
change: the control a mutant is compared against, so formatting alone never counts as a kill.

The operators:

* ``cmp``: one comparison operator flipped (``==``/``!=``, ``<``/``>=``, ``>``/``<=``, ``in``/``not in``,
  ``is``/``is not``); each operator of a chained comparison is its own site;
* ``negate``: the condition of an ``if``, ``while``, conditional expression or ``assert`` negated;
* ``drop_raise``: a ``raise`` statement replaced by ``pass`` (a dropped guard);
* ``const``: an integer or float constant moved by one (off by one; booleans are not numbers here);
* ``boolop``: ``and`` and ``or`` swapped;
* ``return_none``: ``return <value>`` replaced by a bare ``return`` (``None``).

Only the body of the named top-level function is mutated (its decorators, defaults and annotations are not).
A gate reason names an operator kind and a line of the committed module, never a value (ruling R10).
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass

OPERATORS = ("cmp", "negate", "drop_raise", "const", "boolop", "return_none")

_FLIP: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}


@dataclass(frozen=True)
class Site:
    """One mutation: the operator kind, its line in the module, the node's position in the function's
    walk, and which operator of a chained comparison (0 otherwise)."""

    op: str
    line: int
    node: int
    part: int = 0


def _function(
    module: ast.Module,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The module's top-level function *name* (the last definition, as Python binds it)."""
    found = None
    for node in module.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            found = node
    return found


def _walk(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    out: list[ast.AST] = []
    for stmt in fn.body:
        out.extend(ast.walk(stmt))
    return out


def _is_number(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _parse(source: str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None


def sites(source: str, function: str) -> list[Site]:
    """Every site of every operator in *function*'s body, in walk order; [] if it is absent or unparsable."""
    module = _parse(source)
    fn = _function(module, function) if module is not None else None
    if fn is None:
        return []
    out: list[Site] = []
    for i, node in enumerate(_walk(fn)):
        line = int(getattr(node, "lineno", fn.lineno))
        if isinstance(node, ast.Compare):
            out += [
                Site("cmp", line, i, j)
                for j, op in enumerate(node.ops)
                if type(op) in _FLIP
            ]
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            out.append(Site("negate", line, i))
        if isinstance(node, ast.Raise):
            out.append(Site("drop_raise", line, i))
        if _is_number(node):
            out.append(Site("const", line, i))
        if isinstance(node, ast.BoolOp):
            out.append(Site("boolop", line, i))
        if (
            isinstance(node, ast.Return)
            and node.value is not None
            and not (isinstance(node.value, ast.Constant) and node.value.value is None)
        ):
            out.append(Site("return_none", line, i))
    return out


def _rank(seed: bytes, item: str, *parts: object) -> bytes:
    return hashlib.sha256(
        seed + "\0".join([item, *(str(p) for p in parts)]).encode(),
    ).digest()


def choose(all_sites: list[Site], seed: bytes, item: str, limit: int) -> list[Site]:
    """At most *limit* sites: per operator kind in a seeded order, the kinds taken in turn (round robin).

    The order within a kind and the order of the kinds come from SHA-256 of *seed* and *item*, so the choice
    is reproducible from the candidate commit and spreads over the kinds before taking a second of any.
    """
    groups = {
        op: sorted(
            (s for s in all_sites if s.op == op),
            key=lambda s: _rank(seed, item, s.op, s.node, s.part),
        )
        for op in OPERATORS
    }
    kinds = sorted(
        (op for op in OPERATORS if groups[op]),
        key=lambda op: _rank(seed, item, op),
    )
    out: list[Site] = []
    while len(out) < limit and any(groups[op] for op in kinds):
        for op in kinds:
            if groups[op] and len(out) < limit:
                out.append(groups[op].pop(0))
    return out


def _unparse(module: ast.Module) -> str | None:
    ast.fix_missing_locations(module)
    try:
        return ast.unparse(module) + "\n"
    except (ValueError, TypeError, AttributeError, RecursionError):
        return None


def reprint(source: str) -> str | None:
    """*source* re-printed by :func:`ast.unparse` with no change (None if it does not parse)."""
    module = _parse(source)
    return None if module is None else _unparse(module)


def _replace_statement(fn: ast.AST, nodes: list[ast.AST], target: ast.AST) -> bool:
    for parent in [fn, *nodes]:
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for k, v in enumerate(value):
                    if v is target:
                        value[k] = ast.Pass()
                        return True
    return False


def apply(source: str, function: str, site: Site) -> str | None:
    """The module *source* with *site* applied, re-printed; None if the site no longer fits the source."""
    module = _parse(source)
    fn = _function(module, function) if module is not None else None
    if fn is None:
        return None
    nodes = _walk(fn)
    if not 0 <= site.node < len(nodes):
        return None
    node = nodes[site.node]
    if site.op == "cmp":
        if not isinstance(node, ast.Compare) or not 0 <= site.part < len(node.ops):
            return None
        flipped = _FLIP.get(type(node.ops[site.part]))
        if flipped is None:
            return None
        node.ops[site.part] = flipped()
    elif site.op == "negate":
        if not isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            return None
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif site.op == "drop_raise":
        if not isinstance(node, ast.Raise) or not _replace_statement(fn, nodes, node):
            return None
    elif site.op == "const":
        if not _is_number(node):
            return None
        node.value = node.value + 1
    elif site.op == "boolop":
        if not isinstance(node, ast.BoolOp):
            return None
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif site.op == "return_none":
        if not isinstance(node, ast.Return) or node.value is None:
            return None
        node.value = None
    else:
        return None
    return _unparse(module)
