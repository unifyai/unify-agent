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

**Structural negatives** (:func:`negatives`). A surviving mutant is left out of the kill share as likely
equivalent when the function's outputs do not change. Judged only on recorded inputs the environment
accepted, which are almost always well-shaped, every mutant that drops or weakens a *shape guard* would look
equivalent, and a suite that never tests a refusal would pass. So the equivalence probe also feeds inputs
derived from the recorded covers that are broken on purpose: a field dropped, a field's type changed, an extra
field, an empty container (or string) where a non-empty one was recorded, and the whole value's type changed or
emptied. Each kind of break is made once per item (the first cover that has the field), at most
:data:`MAX_NEGATIVES` chosen by a seeded hash. They are used only to tell mutants apart, never to judge the
function itself.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

OPERATORS = ("cmp", "negate", "drop_raise", "const", "boolop", "return_none")
MAX_NEGATIVES = 16  # structural negatives per item in the equivalence probe
EXTRA_FIELD = "unrecorded_field"

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


# --- structural negatives for the equivalence probe ------------------------------------------------------


def _retyped(value: Any) -> Any:
    """*value* as another JSON type (deterministic): what a careless reader would mistake for it."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return len(value)
    if value is None:
        return ""
    if isinstance(value, list):
        return {}
    if isinstance(value, dict):
        return []
    return str(value)


_NA = object()  # "this break does not apply"


def _emptied(value: Any) -> Any:
    """*value* emptied when it is a non-empty container or string, else a sentinel meaning "not applicable"."""
    if isinstance(value, (list, dict, str)) and value:
        return type(value)()
    return _NA


def _breaks(value: Any) -> list[tuple[str, Any]]:
    """Every structural break of one recorded value: ``(label, broken value)``; labels name the break only."""
    out: list[tuple[str, Any]] = [("retype", _retyped(value))]
    empty = _emptied(value)
    if empty is not _NA:
        out.append(("empty", empty))
    record, path = value, ""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        record, path = value[0], "[0]"  # a list of records: break its first record

    def put(broken: dict) -> Any:
        return [broken, *value[1:]] if path else broken

    if isinstance(record, dict):
        for key in sorted(record, key=str):
            dropped = {k: v for k, v in record.items() if k != key}
            out.append((f"drop{path}:{key}", put(dropped)))
            out.append(
                (f"retype{path}:{key}", put({**record, key: _retyped(record[key])})),
            )
            empty = _emptied(record[key])
            if empty is not _NA:
                out.append((f"empty{path}:{key}", put({**record, key: empty})))
        extra = EXTRA_FIELD
        while extra in record:
            extra += "_"
        out.append((f"extra{path}", put({**record, extra: 0})))
    return out


def negatives(
    values: list[Any],
    seed: bytes,
    item: str,
    limit: int = MAX_NEGATIVES,
) -> list[tuple[int, str, Any]]:
    """Structurally broken variants of the recorded *values* (an item's covers): ``(value index, label,
    broken value)``, each break once (from the first value it applies to), at most *limit* by a seeded rank.
    """
    found: dict[str, tuple[int, Any]] = {}
    for i, value in enumerate(values):
        for label, broken in _breaks(value):
            if label not in found and broken != value:
                found[label] = (i, broken)
    ranked = sorted(found, key=lambda label: _rank(seed, item, "negative", label))
    return [(found[label][0], label, found[label][1]) for label in ranked[:limit]]
