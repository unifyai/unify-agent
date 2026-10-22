"""Whether a captured procedure is a stored answer rather than a procedure: a structural rule on its code alone.

A procedure's entry function gives a *fixed answer* when it hands back a value and that value cannot depend on what
it is called on: it never reads its input (no parameter is named anywhere in its body) and some ``return`` gives a
value, or every ``return`` gives a literal value (a constant other than ``None``, a list, tuple, set or dict of
literals, a negated literal, or an f-string without fields). Such a procedure gives the same value whatever request
it is called on: it is the answer one attempt handed back, kept as code.

What is never a fixed answer (rev 2.1), in any world:

* a procedure that hands back no value: ``return None``, a bare ``return`` or no ``return`` at all means "no value",
  the outcome being what it did (the capture contract's ``run`` returns None when the outcome of the work is what it
  did), never a fixed value;
* a procedure that acts: its entry takes the state and the environment (two or more positional parameters, the
  capture contract's ``state_env`` form), or it calls through the harness's action surfaces (``tools.<name>``,
  ``primitives.<namespace>``). In tool, action and conversation worlds the work is those calls.

The rule reads the code's syntax only: no value, no request, no word and no identifier of any task is looked at, and
nothing is run. Whether the entry reads its input (:func:`reads_input`) is a fact of its own, used where a mechanism
asks only that.

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


def _parsed_entry(source: Any) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    return _entry(tree)


def _reads(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {n.id for stmt in _body(fn) for n in ast.walk(stmt) if isinstance(n, ast.Name)}
    return bool(_params(fn) & names)


def reads_input(source: Any) -> bool | None:
    """Whether *source*'s entry function names one of its parameters in its body; None when it is not classified
    (it does not parse, or has no function)."""
    fn = _parsed_entry(source)
    return None if fn is None else _reads(fn)


def _root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _acts(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether *fn* acts: it takes the state and the environment (two or more positional parameters), or it calls
    through one of :data:`ACTION_SURFACES`."""
    if len(fn.args.posonlyargs) + len(fn.args.args) >= 2:
        return True
    return any(
        isinstance(n, ast.Call) and _root(n.func) in ACTION_SURFACES for stmt in _body(fn) for n in ast.walk(stmt)
    )


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


def fixed_answer(source: Any) -> dict | None:
    """``{"why", "returns"}`` when *source*'s entry gives a fixed answer (module docstring), else None. ``why``
    says which part of the rule held (:data:`NEVER_READS` first when both do); ``returns`` holds the literals its
    returns give, as source text (at most :data:`LITERALS_SHOWN`, each at most :data:`LITERAL_CHARS` characters),
    empty when a return is not a literal."""
    fn = _parsed_entry(source)
    if fn is None or _acts(fn):
        return None
    body = _body(fn)
    returns = [n for stmt in body for n in ast.walk(stmt) if isinstance(n, ast.Return)]
    if not any(_gives_value(r) for r in returns):
        return None  # no value handed back: the outcome is what it did
    reads = _reads(fn)
    literal = all(_gives_value(r) and is_literal(r.value) for r in returns)
    if reads and not literal:
        return None
    shown = list(dict.fromkeys(_shown(r.value) for r in returns)) if literal else []
    return {"why": NEVER_READS if not reads else ONLY_LITERALS, "returns": shown[:LITERALS_SHOWN]}
