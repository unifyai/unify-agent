"""The storage-time check (``UNIFY_STORE_CHECK=resolve``): what is stored must resolve and load.

Nothing else on the storage path runs a function, so a function that names
something its sandbox will not have (an invented namespace such as
``primitives.spotify``, a helper defined only in the trajectory, a module the
environment does not supply, a dependency that cannot be installed) is stored
anyway and then fails every time it is loaded or called. This check refuses
it before it is stored, naming what failed and why, so that the caller (the
storage review) can fix the function and add it again:

1. every global name the function's code reads must be a builtin of the
   sandbox, one of its globals (``primitives``, ``query_llm``, the globals a
   registered environment binds), a stored function (or one added in the same
   call), or the function itself;
2. every ``primitives.<namespace>.<method>`` reference must name a namespace in
   scope and one of its methods;
3. every module it imports must be importable here, unless a declared
   dependency supplies it (an import guarded by ``except ImportError`` is
   left to the function);
4. it must load the way a search loads it: its declared dependencies installed,
   its stored callees injected, its ``def`` executed in a scratch namespace
   (``FunctionManager._store_check``). The body is never run.

This module holds the static part (1-3).
"""

from __future__ import annotations

import ast
import difflib
import importlib.util
import symtable
from typing import Any, Iterable, Mapping, Optional, Sequence

from .dependency_analysis import _dynamic_import_target

_IMPORT_GUARDS = frozenset(
    {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"},
)


def _close(name: str, candidates: Iterable[str]) -> Optional[str]:
    found = difflib.get_close_matches(name, sorted(set(candidates)), n=1, cutoff=0.75)
    return found[0] if found else None


def _function_tables(source: str, name: str) -> list[symtable.SymbolTable]:
    table = symtable.symtable(source, "<stored function>", "exec")
    return [child for child in table.get_children() if child.get_name() == name]


def global_reads(source: str, name: str) -> list[str]:
    """Names the function, or anything nested in it, reads from module scope."""
    found: set[str] = set()

    def walk(table: symtable.SymbolTable) -> None:
        for symbol in table.get_symbols():
            if not symbol.is_referenced() or not symbol.is_global():
                continue
            if symbol.is_declared_global() and symbol.is_assigned():
                continue
            found.add(symbol.get_name())
        for child in table.get_children():
            walk(child)

    for table in _function_tables(source, name):
        walk(table)
    return sorted(found)


def binds_locally(source: str, function: str, name: str) -> bool:
    """Whether ``name`` is a parameter or local variable of the top-level function."""
    for table in _function_tables(source, function):
        try:
            symbol = table.lookup(name)
        except KeyError:
            continue
        if symbol.is_parameter() or symbol.is_local():
            return True
    return False


def primitive_chains(node: ast.AST) -> list[tuple[str, ...]]:
    """Every maximal attribute chain rooted at the name ``primitives``."""
    inner = {
        id(n.value)
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute)
    }
    chains: list[tuple[str, ...]] = []
    for n in ast.walk(node):
        if not isinstance(n, ast.Attribute) or id(n) in inner:
            continue
        parts: list[str] = []
        current: ast.AST = n
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id == "primitives":
            chain = ("primitives", *reversed(parts))
            if chain not in chains:
                chains.append(chain)
    return chains


def _guarded(handlers: Sequence[ast.ExceptHandler]) -> bool:
    for handler in handlers:
        kinds = handler.type
        if kinds is None:
            return True
        names = kinds.elts if isinstance(kinds, ast.Tuple) else [kinds]
        for kind in names:
            label = (
                kind.attr
                if isinstance(kind, ast.Attribute)
                else getattr(kind, "id", "")
            )
            if label in _IMPORT_GUARDS:
                return True
    return False


def imported_modules(node: ast.AST) -> list[str]:
    """Root names of the modules the function imports, outside ``except ImportError`` guards."""
    guarded: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Try) and _guarded(n.handlers):
            for statement in n.body:
                guarded.update(id(inner) for inner in ast.walk(statement))
    roots: list[str] = []
    for n in ast.walk(node):
        if id(n) in guarded:
            continue
        names: list[str] = []
        if isinstance(n, ast.Import):
            names = [alias.name for alias in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            names = [n.module]
        elif isinstance(n, ast.Call):
            target = _dynamic_import_target(n)
            if target:
                names = [target]
        for name in names:
            root = name.split(".")[0]
            if root and root not in roots:
                roots.append(root)
    return roots


def _methods_text(namespace: str, methods: Sequence[str]) -> str:
    if len(methods) <= 12:
        return f"its methods are {', '.join(f'`{m}`' for m in methods)}"
    from unify.actor import core_surface

    search = (
        "functions.search"
        if core_surface.enabled()
        else "FunctionManager_search_functions"
    )
    return (
        f"it has {len(methods)} methods; find the right one with "
        f"{search} or help(primitives.{namespace})"
    )


def unresolved(
    *,
    source: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    sandbox_globals: Mapping[str, Any],
    stored_functions: Iterable[str],
    namespaces: Mapping[str, Sequence[str]],
    environment_modules: Iterable[str],
    pip_supplied: Iterable[str] = (),
) -> list[str]:
    """What in ``source`` would not resolve where the function runs, one message each.

    ``sandbox_globals`` is a fresh sandbox's globals (its ``__builtins__``
    included); ``namespaces`` maps each ``primitives`` namespace in scope to its
    method names; ``pip_supplied`` are third-party modules a declared
    dependency installs before the function runs.
    """
    problems: list[str] = []
    builtins_obj = sandbox_globals.get("__builtins__")
    builtin_names = set(builtins_obj) if isinstance(builtins_obj, Mapping) else set()
    known = set(sandbox_globals) | builtin_names | set(stored_functions) | {name}

    for missing in (g for g in global_reads(source, name) if g not in known):
        hint = ""
        if missing in namespaces:
            hint = f" (did you mean `primitives.{missing}`?)"
        else:
            close = _close(missing, known - builtin_names)
            if close:
                hint = f" (did you mean `{close}`?)"
        problems.append(
            f"`{missing}` is not defined where the function runs (it is not a "
            f"builtin, a sandbox global, a stored function or a `primitives` "
            f"namespace){hint}",
        )

    if not binds_locally(source, name, "primitives"):
        registered = sorted(namespaces)
        for chain in primitive_chains(node):
            if len(chain) < 2 or chain[1].startswith("__"):
                continue
            alias = chain[1]
            if alias not in namespaces:
                close = _close(alias, registered)
                listed = ", ".join(f"`primitives.{a}`" for a in registered)
                problems.append(
                    f"`primitives.{alias}` does not exist: the namespaces under "
                    f"`primitives` are {listed}"
                    + (f" (did you mean `primitives.{close}`?)" if close else ""),
                )
                continue
            if len(chain) < 3 or chain[2].startswith("__"):
                continue
            methods = sorted(namespaces[alias])
            method = chain[2]
            if method not in methods:
                close = _close(method, methods)
                problems.append(
                    f"`primitives.{alias}` has no method `{method}` ("
                    + (f"did you mean `{close}`? " if close else "")
                    + f"{_methods_text(alias, methods)})",
                )
                continue
            if len(chain) > 3 and not chain[3].startswith("__"):
                problems.append(
                    f"`{'.'.join(chain)}` does not exist: `primitives.{alias}.{method}` "
                    f"is a method; call it",
                )

    supplied = set(pip_supplied)
    env_modules = set(environment_modules)
    for module in imported_modules(node):
        if module in supplied:
            continue
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if found:
            continue
        if module in sandbox_globals:
            problems.append(
                f"`import {module}` fails: `{module}` is a sandbox global, not a "
                f"module; use it without importing it",
            )
        elif module in env_modules:
            problems.append(
                f"`import {module}` fails: the environment lists `{module}` as a "
                f"module it supplies, but it cannot be imported here",
            )
        else:
            problems.append(
                f"`import {module}` fails: no module named `{module}` is installed; "
                f"declare the pip package that provides it in `dependencies`",
            )
    return problems


__all__ = [
    "global_reads",
    "binds_locally",
    "imported_modules",
    "primitive_chains",
    "unresolved",
]
