"""Aliases and calls in a v2.1 library (spec v2.1 §10.4, §9.1). Pure AST: nothing is imported or run.

An **alias** keeps an old name working after CURATE merges, generalises or moves an item (§10.4). Three forms count,
each a top-level binding of the old name in its module, where the last binding of a name wins:

* a binding, ``old = new``, where ``new`` is a function of the module or a name imported from a library module;
* an import, ``from memory.<package>.<module> import new as old`` (relative imports too);
* a forwarding wrapper, ``def old(<parameters>): return new(<arguments>)``, with at most a docstring before the
  return, where each argument is one of ``old``'s parameters, each passed at most once (by position or keyword,
  starred or not), or a constant. A wrapper is still a function item; P5's record marks it an alias.

A name imported through a package's re-export (``from memory.p import f``) resolves to the package, not to the
function; an alias imports from the function's own module. :func:`forwards` lists every forwarding binding of a
module; the gate (:mod:`.gate_v21`) checks a declared alias against it. :func:`calls_function` is the v2.1 form of
:func:`.snapshot.calls_item`.
"""

from __future__ import annotations

import ast

from . import layout

_FN = (ast.FunctionDef, ast.AsyncFunctionDef)


def _parse(source: bytes | str) -> ast.Module | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _package(module: str) -> str:
    """The package a module's relative imports resolve against: ``memory.p`` for ``memory.p.m`` and ``memory.p``."""
    return ".".join(module.split(".")[:2])


def _resolve(package: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = package.split(".")
    if node.level > len(parts):
        return None
    base = parts[: len(parts) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def _target(
    node: ast.AST | None,
    ids: dict[str, str | None],
    mods: dict[str, str],
) -> str | None:
    """The function item a name or ``module.attribute`` expression names, else None."""
    if isinstance(node, ast.Name):
        return ids.get(node.id)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in mods
    ):
        return f"{mods[node.value.id]}:{node.attr}"
    return None


def _wrapped(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ids: dict,
    mods: dict,
) -> str | None:
    """What a forwarding wrapper forwards to, or None when *fn* is not one."""
    body = list(fn.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if fn.decorator_list or len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    call = body[0].value
    if not isinstance(call, ast.Call):
        return None
    target = _target(call.func, ids, mods)
    if target is None:
        return None
    a = fn.args
    params = {p.arg for p in [*a.posonlyargs, *a.args, *a.kwonlyargs]}
    params |= {p.arg for p in (a.vararg, a.kwarg) if p is not None}
    used: list[str] = []
    for arg in [*call.args, *(k.value for k in call.keywords)]:
        value = arg.value if isinstance(arg, ast.Starred) else arg
        if isinstance(value, ast.Name) and value.id in params:
            used.append(value.id)
        elif not isinstance(value, ast.Constant):
            return None
    return target if len(used) == len(set(used)) else None


def forwards(
    source: bytes | str,
    module: str,
    modules: set[str],
) -> dict[str, str] | None:
    """Every public top-level name of *module* that forwards to a library function: name -> item id.

    *modules* are the library's modules (:func:`.layout.library_modules`). A plain ``def`` forwards nowhere and is
    not listed. None when *source* does not parse.
    """
    tree = _parse(source)
    if tree is None:
        return None
    package = _package(module)
    ids: dict[str, str | None] = (
        {}
    )  # what calling the name runs: an item id, or None (unknown)
    mods: dict[str, str] = {}  # local name -> the library module it is bound to
    alias: dict[str, str | None] = {}

    def bind(name: str, target: str | None, forwarded: str | None) -> None:
        ids[name] = target
        alias[name] = forwarded
        mods.pop(name, None)

    for node in tree.body:
        if isinstance(node, _FN):
            wrapped = _wrapped(node, ids, mods)
            bind(node.name, wrapped or f"{module}:{node.name}", wrapped)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve(package, node)
            for a in node.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                sub = f"{base}.{a.name}" if base else None
                if sub in modules:
                    bind(local, None, None)
                    mods[local] = sub
                elif base in modules:
                    bind(local, f"{base}:{a.name}", f"{base}:{a.name}")
                else:
                    bind(local, None, None)
        elif isinstance(node, ast.Import):
            for a in node.names:
                local = a.asname or a.name.split(".")[0]
                bind(local, None, None)
                if a.asname and a.name in modules:
                    mods[local] = a.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _target(node.value, ids, mods)
            for t in targets:
                if isinstance(t, ast.Name):
                    bind(t.id, value, value)
    return {
        n: t
        for n, t in sorted(alias.items())
        if t is not None and not n.startswith("_")
    }


def calls_function(source: bytes | str, item: str, package: str = "") -> bool:
    """Whether *source* (a test file) calls function *item*, ``memory.<p>.<m>:<name>``.

    A call counts when its callee is a name bound by ``from <module> import <name> [as x]`` (relative imports
    resolve against *package*), or ``<name>`` as an attribute of the module bound by ``import <module> as m``,
    ``from memory.<p> import <m> [as m]`` or the dotted ``memory.<p>.<m>.<name>``. An import alone does not count.
    """
    tree = _parse(source)
    if layout.FUNCTION_ID.match(item) is None or tree is None:
        return False
    module, name = item.split(":", 1)
    names: set[str] = set()  # local names bound to the function
    holders: set[str] = set()  # dotted expressions bound to its module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _resolve(package, node)
            for a in node.names:
                if base == module and a.name in (name, "*"):
                    names.add(name if a.name == "*" else (a.asname or a.name))
                elif base is not None and f"{base}.{a.name}" == module:
                    holders.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module:
                    holders.add(a.asname or module)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in names:
            return True
        if (
            isinstance(f, ast.Attribute)
            and f.attr == name
            and _dotted(f.value) in holders
        ):
            return True
    return False


def uses_name(source: bytes | str, package: str, item: str, modules: set[str]) -> bool:
    """Whether test code *source* imports *item* by name or calls it (a test of an old name, spec §9.1)."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    found = layout.imports(data, package, modules)
    return (found is not None and item in found[0]) or calls_function(
        source,
        item,
        package,
    )
