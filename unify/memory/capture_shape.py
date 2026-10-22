"""A captured procedure's shape address, and whether a pause's cells computed anything: from the code's syntax alone.

**The shape address** (:func:`shape_of`) names what a procedure computes, whatever its prose and its local names. Two
captured procedures with the same address fold into one capture (:func:`.captures.capture_trace_step`): the second is
an occurrence of the first, its raw bytes kept beside it as evidence, nothing deleted. The address is the sha256 of:

1. the module's syntax tree (``ast.dump``, field names on, positions off) with every docstring removed (the module's
   and each function's and class's; a comment is not in the tree). The whole module counts, so two procedures whose
   ``run`` is the same but whose helpers differ never fold;
2. inside each function, every local and parameter name renamed in the order it is first bound (``_0``, ``_1``, ...):
   a name the module or the builtins give, an attribute and a keyword stay as written;
3. every literal kept as written, so two procedures that differ in one constant never fold;
4. the entry form and the number of ``run``'s parameters (:func:`.capture_trace.entry_point`), and the sorted
   ``<env>.<channel>.<method>`` chains ``run``'s tool namespace parameter names.

Nothing is run, imported or evaluated, and no world name, word, task id or value is read: the address is a function of
the procedure's syntax only. A procedure that does not parse has no address (it is not captured anyway).

**Work** (:func:`computes`): a pause's cells did work when one of them binds a name, defines a function or class,
loops, comprehends, branches, calls with an argument that is not a literal, or calls into the environment or through
an action surface at all: any call through :data:`.fixed_answer.ACTION_SURFACES` (``tools``, ``primitives``), the
same surfaces that make a stored procedure an action, and any call whose name is neither the harness's own
read-only names (:data:`HARNESS_NAMES`: ``memory``, ``display``, ``query_llm``) nor a Python builtin, such as
``env.board.step("left")`` or ``tools.wait()`` with literal arguments (rev 2.1: in action and tool worlds a run of
literal action calls is the work). A cell in another language, or one that does not parse, counts as work. Only cells
with no call into the environment or an action surface and no computation (literal calls of builtins or of the
harness's read-only names, such as ``print("x")`` or ``memory.items()``) start no segment fork
(:meth:`.integration.in_episode.InEpisode.segment`).
"""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any, Iterable

from .fixed_answer import ACTION_SURFACES, is_literal

SHAPE_VERSION = 1

_DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _DOC_OWNERS):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]


def _args(a: ast.arguments) -> list[ast.arg]:
    out = [*a.posonlyargs, *a.args]
    if a.vararg is not None:
        out.append(a.vararg)
    out += a.kwonlyargs
    if a.kwarg is not None:
        out.append(a.kwarg)
    return out


class _Binder(ast.NodeVisitor):
    """The names a function binds (its parameters, and every name stored, defined, imported or caught inside it, nested
    scopes included), in the order first bound; names declared ``global`` or ``nonlocal`` are not its own."""

    def __init__(self) -> None:
        self.order: list[str] = []
        self.free: set[str] = set()

    def bind(self, name: str | None) -> None:
        if name and name not in self.order:
            self.order.append(name)

    def visit_arguments(self, node: ast.arguments) -> None:
        for a in _args(node):
            self.bind(a.arg)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bind(node.id)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bind(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bind(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.bind(node.name)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        self.bind((node.asname or node.name).split(".")[0])

    def visit_Global(self, node: ast.Global) -> None:
        self.free.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.free.update(node.names)


class _Renamer(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self.names.get(node.arg, node.arg)
        node.annotation = self.visit(node.annotation) if node.annotation is not None else None
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.name:
            node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.names.get(node.name, node.name)  # a nested function's own name is a local
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
    visit_ClassDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_alias(self, node: ast.alias) -> ast.AST:
        if node.asname:
            node.asname = self.names.get(node.asname, node.asname)
        elif "." not in node.name and node.name in self.names:
            node.asname = self.names[node.name]
        return node


class _Nested(ast.NodeTransformer):
    """Renames the names each top-level function binds (nested functions' own names included), function by function."""

    def __init__(self, top: ast.AST) -> None:
        self.top = top

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        b = _Binder()
        b.visit(node.args)
        for stmt in node.body:
            b.visit(stmt)
        names = {n: f"_{i}" for i, n in enumerate(x for x in b.order if x not in b.free)}
        r = _Renamer(names)
        node.args = r.visit(node.args)
        node.body = [r.visit(s) for s in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return self._function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return self._function(node)


def _chains(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The ``<env>.<channel>.<method>`` chains rooted at *fn*'s second positional parameter (its tool namespace), as
    ``env.<channel>.<method>``."""
    params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
    if len(params) < 2:
        return []
    env = params[1]
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            inner = node.value
            if isinstance(inner.value, ast.Name) and inner.value.id == env:
                out.add(f"env.{inner.attr}.{node.attr}")
    return sorted(out)


def _env_chains(tree: ast.Module) -> list[str]:
    """The tool calls ``run`` names (:func:`_chains`)."""
    run = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "run"),
               None)  # fmt: skip
    return _chains(run) if run is not None else []


def entry_env_chains(source: Any) -> list[str]:
    """The tool calls a procedure's entry function names (``run``; else ``main``; else its only function; else its
    first public one, as :mod:`.fixed_answer` reads the entry), from its syntax; [] when it does not parse."""
    from .fixed_answer import _entry

    if not isinstance(source, str):
        return []
    try:
        fn = _entry(ast.parse(source))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    return _chains(fn) if fn is not None else []


def shape_of(source: Any) -> dict | None:
    """``{"v", "shape", "form", "params", "env_calls"}`` for a procedure's source (module docstring), or None when it
    does not parse. Nothing is run."""
    from .capture_trace import entry_point

    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
        entry = entry_point(source) or {"form": None, "params": []}
        env_calls = _env_chains(tree)
        _strip_docstrings(tree)
        tree = _Nested(tree).visit(tree)
        dumped = ast.dump(tree, annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    doc = {
        "v": SHAPE_VERSION,
        "tree": dumped,
        "form": entry.get("form"),
        "params": len(entry.get("params") or []),
        "env_calls": env_calls,
    }
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return {"v": SHAPE_VERSION, "shape": digest, "form": doc["form"], "params": doc["params"], "env_calls": env_calls}


def function_shape(source: Any, name: str) -> str | None:
    """The shape of the one module-level function *name* of *source* (:func:`shape_of`'s rules on that function alone,
    its own name not counted): two functions with the same shape compute the same, whatever their names, prose and
    local names. None when *source* does not parse or has no such function. Nothing is run."""
    import copy

    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    if fn is None:
        return None
    fn = copy.deepcopy(fn)
    fn.name = "_"
    module = ast.Module(body=[fn], type_ignores=[])
    _strip_docstrings(module)
    module = _Nested(module).visit(module)
    dumped = ast.dump(module, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8", "surrogatepass")).hexdigest()[:16]


# --- whether a pause's cells computed anything ---------------------------------------------------------------------------

_COMPUTES = (
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.NamedExpr, ast.Delete,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
    ast.For, ast.AsyncFor, ast.While,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
    ast.If, ast.IfExp, ast.Try, ast.With, ast.AsyncWith,
    ast.Import, ast.ImportFrom,
)  # fmt: skip
if hasattr(ast, "Match"):
    _COMPUTES = (*_COMPUTES, ast.Match)
if hasattr(ast, "TryStar"):
    _COMPUTES = (*_COMPUTES, ast.TryStar)


#: The harness's own read-only names a cell calls: a literal call of one of them (or of a Python builtin) is no work.
#: The action surfaces (:data:`.fixed_answer.ACTION_SURFACES`) are never among them: a call through one acts.
HARNESS_NAMES = frozenset({"memory", "display", "query_llm"})


def _call_root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def cell_computes(code: Any, language: str = "python") -> bool:
    """Whether one cell did work (module docstring): computed something or called into the environment; a cell that
    is not Python, or does not parse, does."""
    import builtins

    if (language or "python") != "python" or not isinstance(code, str):
        return True
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, _COMPUTES):
            return True
        if isinstance(node, ast.Call):
            root = _call_root(node.func)
            if root in ACTION_SURFACES:
                return True  # an action, whatever its arguments (as fixed_answer counts it)
            if root is not None and root not in HARNESS_NAMES and not hasattr(builtins, root):
                return True  # a call into the environment is work, whatever its arguments
            if any(not is_literal(a) for a in node.args):
                return True
            if any(not is_literal(k.value) for k in node.keywords):
                return True
    return False


def computes(cells: Iterable[Any]) -> bool:
    """Whether any of *cells* (each with ``code`` and ``language``) did work."""
    return any(cell_computes(getattr(c, "code", None), getattr(c, "language", "python")) for c in cells)
