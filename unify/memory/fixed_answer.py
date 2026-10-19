"""Whether a captured procedure is a stored answer rather than a procedure: a structural rule on its code alone.

A captured procedure's ``run`` gives a *fixed answer* when it never reads its input (no parameter of ``run`` is
named anywhere in its body), or when every ``return`` in its body returns a literal (a constant, or a list, tuple,
set or dict of literals, a negated literal, or an f-string without fields). Such a procedure gives the same value
whatever request it is called on: it is the answer one attempt handed back, kept as code. The rule reads the code's
syntax only: no value, no request, no word and no identifier of any task is looked at, and nothing is run.

``run`` is the module's function of that name; else ``main``; else its only function; else its first public one
(the order the capture contract and earlier procedure layouts name their entry). A body's docstring is not read as
code. A procedure that does not parse, or has no function, is not a fixed answer (it is not classified at all).
"""

from __future__ import annotations

import ast
from typing import Any

#: The most characters of one returned literal a fixed answer's description shows.
LITERAL_CHARS = 400
#: The most returned literals it shows.
LITERALS_SHOWN = 4
NEVER_READS = "its run() never reads its input"
ONLY_LITERALS = "every return of its run() is a fixed value"


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


def _params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = node.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg is not None:
        names.add(a.vararg.arg)
    if a.kwarg is not None:
        names.add(a.kwarg.arg)
    return names


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


def _shown(node: ast.AST) -> str:
    try:
        text = ast.unparse(node)
    except (ValueError, RecursionError):
        return "(not shown)"
    return text if len(text) <= LITERAL_CHARS else text[: LITERAL_CHARS - 1] + "…"


def fixed_answer(source: Any) -> dict | None:
    """``{"why", "returns"}`` when *source*'s ``run`` gives a fixed answer (module docstring), else None. ``why``
    says which part of the rule held (:data:`NEVER_READS` first when both do); ``returns`` holds the literals its
    returns give, as source text (at most :data:`LITERALS_SHOWN`, each at most :data:`LITERAL_CHARS` characters),
    empty when a return is not a literal."""
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    fn = _entry(tree)
    if fn is None:
        return None
    body = _body(fn)
    names = {n.id for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    reads = bool(_params(fn) & names)
    returns = [n for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Return) and n.value is not None]
    literal = bool(returns) and all(is_literal(r.value) for r in returns)
    if reads and not literal:
        return None
    shown = list(dict.fromkeys(_shown(r.value) for r in returns)) if literal else []
    return {"why": NEVER_READS if not reads else ONLY_LITERALS, "returns": shown[:LITERALS_SHOWN]}
