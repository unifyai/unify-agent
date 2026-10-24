"""The closure of a function an answer was made with, built from the request's recorded cells by AST nodes only.

When a request's accepted answer was made by ``tools.answer_from(fn)``, the harness stores ``fn`` as a recorded
procedure (rev 2.3). The procedure is built here from the code cells the request recorded, never by running anything:

* ``fn``'s own ``def``, the last one the cells bound under its name;
* the ``def`` and ``class`` nodes that bind every session name ``fn`` refers to, transitively, each the last binding
  of its name;
* the ``import`` nodes those names need;
* each constant ``fn`` reaches, only when the last binding of its name is plain JSON data (``None``, booleans, numbers,
  text, and lists and dicts of these): read by :func:`ast.literal_eval` of the recorded assignment, never evaluated.

Whole cells are never copied and no cell is re-run. The closure fails, and nothing is stored, when a name ``fn``
reaches is one of the harness's own (:data:`HARNESS_NAMES`), is bound by anything but a ``def``, a ``class``, an
``import`` or a plain-data assignment, or is not bound in the recorded cells at all; or when any kept node (``fn``'s
own body included) imports a harness or key-routed module (:data:`HARNESS_MODULES`). These are hard security checks;
beyond them the guarantee is the confined child a stored procedure runs in (no keys, no tool channel). The stored
procedure reads the declared input as data: its ``read_input`` is the identity and its ``run(value)`` calls
``fn(value)``, or ``fn(*value)`` for a list or tuple when ``fn`` needs two or more positional arguments, as
``tools.answer_from`` called it. The code is untrusted text from the agent's box: the capture step's safety checks still apply.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterable
from typing import Any

#: Names the harness binds in an agent's cells, or that reach it: a closure that reaches one fails.
HARNESS_NAMES = frozenset({"tools", "memory", "query_llm", "list_llms"})
#: Modules a closure may never import (by their first dotted part): the harness's own, its box modules, and the
#: key-routed model client.
HARNESS_MODULES = frozenset({"unify", "unillm", "tools", "memory"})
_BUILTINS = frozenset(dir(builtins))


class ClosureError(ValueError):
    """Why the closure cannot be built (nothing is stored)."""


def plain(value: Any) -> bool:
    """Whether *value* is plain JSON data (exact types only, never a subclass)."""
    if value is None or type(value) in (bool, int, float, str):
        return True
    if type(value) in (list, tuple):
        return all(plain(x) for x in value)
    if type(value) is dict:
        return all(type(k) is str and plain(x) for k, x in value.items())
    return False


def _const(name: str, node: ast.AST | None) -> ast.stmt | None:
    """``name = <value>`` for a plain-data assigned expression (:func:`ast.literal_eval`), else None."""
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not plain(value):
        return None
    return ast.parse(f"{name} = {value!r}").body[0]


def _bindings(cells: Iterable[str]) -> dict[str, tuple[str, ast.AST, int]]:
    """Each top-level name the cells bind, with how its last binding bound it (``def``, ``class``, ``import``,
    ``const`` for a plain-data assignment, ``other`` for anything else) and its position in the recorded order; a cell
    that does not parse binds nothing."""
    out: dict[str, tuple[str, ast.AST, int]] = {}
    pos = 0
    for code in cells:
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        for node in tree.body:
            pos += 1
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[node.name] = ("def", node, pos)
            elif isinstance(node, ast.ClassDef):
                out[node.name] = ("class", node, pos)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    one = (ast.Import(names=[alias]) if isinstance(node, ast.Import)
                           else ast.ImportFrom(module=node.module, names=[alias], level=node.level))  # fmt: skip
                    out[(alias.asname or alias.name).split(".")[0]] = ("import", one, pos)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and all(
                isinstance(t, ast.Name) for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
            ):
                for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    stmt = _const(t.id, node.value)
                    out[t.id] = ("const", stmt, pos) if stmt is not None else ("other", node, pos)
            else:
                # any other binding of a name (a tuple assignment, a loop, a with, an augmented assignment, ...)
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                        out[n.id] = ("other", node, pos)
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        out[n.name] = ("other", node, pos)
                    elif isinstance(n, (ast.Import, ast.ImportFrom)):
                        for a in n.names:
                            out[(a.asname or a.name).split(".")[0]] = ("other", node, pos)
    return out


def free_names(node: ast.AST) -> set[str]:
    """The names *node* loads that it does not bind itself (its parameters, assignments, loop and comprehension
    targets, nested definitions and imports, handler names), builtins left out."""
    bound: set[str] = set()
    loaded: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            (loaded if isinstance(n.ctx, ast.Load) else bound).add(n.id)
        elif isinstance(n, ast.arg):
            bound.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
            bound.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    return loaded - bound - _BUILTINS


def harness_import(node: ast.AST) -> str | None:
    """The harness module an ``import`` anywhere in *node* names (:data:`HARNESS_MODULES`), or None."""
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] in HARNESS_MODULES:
                    return a.name
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").split(".")[0] in HARNESS_MODULES:
            return n.module or ""
    return None


def build(cells: Iterable[str], fn: str) -> str:
    """The stored procedure's source for the function *fn* of the recorded *cells*: the closure's imports, constants
    and definitions in their recorded order, then ``read_input`` (the identity) and ``run(value)``. Raises
    :class:`ClosureError` with the reason when it cannot be built."""
    if not isinstance(fn, str) or not fn.isidentifier() or fn in HARNESS_NAMES or fn in ("run", "read_input"):
        raise ClosureError(f"{fn!r} is not a function name of the session that can be stored")
    bindings = _bindings(list(cells))
    if bindings.get(fn, ("",))[0] != "def":
        raise ClosureError(f"{fn} is not bound by a def in the recorded cells")
    kept: dict[str, tuple[str, ast.AST, int]] = {}
    todo = [fn]
    while todo:
        name = todo.pop()
        if name in kept:
            continue
        if name in HARNESS_NAMES:
            raise ClosureError(f"it reaches {name}, a name of the harness")
        if name in ("run", "read_input"):
            raise ClosureError(f"it reaches {name}, a name the stored procedure defines itself")
        if name not in bindings:
            raise ClosureError(f"it reaches {name}, which the recorded cells do not bind")
        how, node, pos = bindings[name]
        if how == "other":
            raise ClosureError(
                f"it reaches {name}, which is bound to something other than a def, a class, an import or plain data"
            )
        bad = harness_import(node)
        if bad is not None:
            raise ClosureError(f"it imports {bad}, a module of the harness")
        kept[name] = (how, node, pos)
        if how in ("def", "class"):
            todo.extend(sorted(free_names(node)))
    nodes: list[ast.AST] = []
    rank = {"import": 0, "const": 1}
    for _name, (_how, node, _pos) in sorted(kept.items(), key=lambda kv: (rank.get(kv[1][0], 2), kv[1][2], kv[0])):
        if all(node is not n for n in nodes):
            nodes.append(node)
    body = ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])))
    if spreads(bindings[fn][1]):
        # as tools.answer_from called it: a list or tuple input passed as its arguments (two or more required)
        call = f"    if isinstance(value, (list, tuple)):\n        return {fn}(*value)\n    return {fn}(value)\n"
    else:
        call = f"    return {fn}(value)\n"
    return (
        f'"""The function an accepted answer was made with, {fn}, stored with what it needs."""\n\n'
        f"{body}\n\n\n"
        "def read_input(value):\n"
        '    """The request\'s declared input, as data."""\n'
        "    return value\n\n\n"
        "def run(value):\n"
        f"{call}"
    )


def spreads(node: ast.AST) -> bool:
    """Whether the ``def`` *node* needs two or more positional arguments (required positional parameters, read from
    its AST): then a list or tuple input is passed as its arguments, the rule ``tools.answer_from`` and
    ``tools.check_examples`` call a function by."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    positional = [*node.args.posonlyargs, *node.args.args]
    return len(positional) - len(node.args.defaults) >= 2
