"""Per-item admission (stage 7): a refused candidate reduced to the items the gate did not refuse.

When every failure of a pass belongs to manifest items (:attr:`.gate._Run.item_fail`, never a pass-wide
failure), :meth:`.gate.Gate.merge` refuses those items alone. This module works out what else goes with
them and writes the reduced tree; the gate commits it as a child of the parent and runs the whole gate on
it again (one round, no search over subsets).

What goes with a refused item (:func:`with_dependents`, a closure over the manifest's items): an item that shares a
test file with it (the file cannot be split), and an item whose module or tests import it (:func:`uses_library_mode`):
by name, through a package that re-exports it (resolved to the module that defines it), or with its whole module; a
function of the same module that it reaches by name, directly or through private helpers (``ast``, never text).

What the reduction does (:func:`build`): each refused item's file, and every test file it lists, is restored to the
parent's version, or removed when the parent had none. No reduction is made (the pass stays refused whole) when
nothing would be admitted.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

from . import layout
from .manifest import Manifest, ManifestItem

DEPENDENCY = "dependency"


def _defs(module: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        n.name: n
        for n in module.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called(tree: Path, it: ManifestItem) -> set[str]:
    """The module's other public functions *it* reaches by name, directly or through private helpers."""
    try:
        defs = _defs(ast.parse((tree / it.path).read_bytes()))
    except (OSError, SyntaxError, ValueError):
        return set()
    name = it.item.split(":", 1)[1]
    seen, todo, out = {name}, [name], set()
    while todo:
        node = defs.get(todo.pop())
        if node is None:
            continue
        for sub in ast.walk(node):
            ref = sub.id if isinstance(sub, ast.Name) else None
            if ref is None or ref in seen or ref not in defs:
                continue
            seen.add(ref)
            if ref.startswith("_"):
                todo.append(ref)
            else:
                out.add(ref)
    prefix = it.item.split(":", 1)[0]
    return {f"{prefix}:{n}" for n in out}


_FUNCTION_KINDS = ("function",)
_MAX_REEXPORT_DEPTH = 8


def _library_context(
    tree: Path,
) -> tuple[set[str], dict[str, dict[str, str]], dict[str, set[str]]]:
    """(library modules, package -> its init's bindings, module -> its function ids) of the candidate *tree*.

    An init's bindings map each name it binds by import to what defines it: ``"<module>:<name>"`` for a name,
    or ``"<module>"`` for a submodule. A name the init defines itself is ``"<package>:<name>"``.
    """
    modules = layout.library_modules(tree)
    by_module: dict[str, set[str]] = {}
    for item in layout.function_bodies(tree):
        by_module.setdefault(item.split(":", 1)[0], set()).add(item)
    inits: dict[str, dict[str, str]] = {}
    for pkg in sorted(m for m in modules if m.count(".") == 1):
        try:
            source = ast.parse((tree / layout.module_path(pkg)).read_bytes())
        except (OSError, SyntaxError, ValueError):
            continue
        binds: dict[str, str] = {}
        for node in source.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                binds[node.name] = f"{pkg}:{node.name}"
            elif isinstance(node, ast.ImportFrom):
                base = layout._resolve(pkg, node)
                if base is None or base not in modules:
                    continue
                for a in node.names:
                    if a.name == "*":
                        continue
                    sub = f"{base}.{a.name}"
                    binds[a.asname or a.name] = (
                        sub if sub in modules else f"{base}:{a.name}"
                    )
        inits[pkg] = binds
    return modules, inits, by_module


def _through(inits: dict[str, dict[str, str]], target: str) -> str:
    """*target* (``"<module>:<name>"`` or a module) followed through package re-exports to its definition."""
    for _ in range(_MAX_REEXPORT_DEPTH):
        mod, _, name = target.partition(":")
        if not name or mod.count(".") != 1 or name not in inits.get(mod, {}):
            return target
        nxt = inits[mod][name]
        if nxt == target:
            return target
        target = nxt
    return target


def _imports_library_mode(
    source: bytes,
    package: str,
    modules: set[str],
    inits: dict[str, dict[str, str]],
) -> set[str] | None:
    """What *source* takes from the library (Amendment A: per name, not per package init). Returns item ids
    (``"<module>:<name>"``) and whole modules (``"<module>"``):

    - ``from memory.a.b import f`` gives ``memory.a.b:f``; ``import memory.a.b`` and ``from memory.a import b``
      give the module ``memory.a.b``;
    - ``from memory.a import f``, where the package init re-exports ``f``, gives the module that defines ``f``;
    - a package imported whole (``import memory.a``, a star import from it) gives every name its init binds;
    - a package init that only runs on the way gives nothing: running it calls no item.

    Relative imports resolve against *package*. None when *source* does not parse.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    out: set[str] = set()

    def package_whole(pkg: str) -> None:
        out.add(pkg)
        for target in inits.get(pkg, {}).values():
            out.add(_through(inits, target))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in modules:
                    package_whole(a.name) if a.name.count(".") == 1 else out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            base = layout._resolve(package, node)
            if base is None or base.split(".")[0] != layout.PACKAGE_ROOT:
                continue
            for a in node.names:
                sub = f"{base}.{a.name}"
                if sub in modules:
                    package_whole(sub) if sub.count(".") == 1 else out.add(sub)
                elif base not in modules:
                    continue
                elif base.count(".") == 1:
                    if a.name == "*":
                        package_whole(base)
                    else:
                        out.add(_through(inits, f"{base}:{a.name}"))
                else:
                    out.add(base if a.name == "*" else f"{base}:{a.name}")
    return out


def uses_library_mode(tree: Path, it: ManifestItem, ctx=None) -> set[str]:
    """In library mode: item ids *it* depends on in the candidate *tree*:

    - the functions of its own module it calls, directly or through private helpers (:func:`_called`);
    - what its module and its tests import from the library (:func:`_imports_library_mode`): a name, resolved through
      a package's re-exports to the module that defines it, or every function of a module
      imported whole.
    """
    modules, inits, by_module = ctx or _library_context(tree)
    out = _called(tree, it) if it.kind in _FUNCTION_KINDS else set()
    for rel in ([it.path] if it.kind in _FUNCTION_KINDS else []) + list(it.tests):
        try:
            data = (tree / rel).read_bytes()
        except OSError:
            continue
        for ref in (
            _imports_library_mode(data, layout.package_of_path(rel), modules, inits)
            or set()
        ):
            if ":" in ref:
                out.add(ref)
            else:
                out |= by_module.get(ref, set())
    out.discard(it.item)
    return out


def with_dependents(
    man: Manifest,
    failed: dict[str, list[str]],
    tree: Path
) -> tuple[dict[str, list[str]], list[str]]:
    """Every refused item with its codes (the failed ones' checks, ``dependency`` for the rest) and why.

    With *library_mode* an item goes with a refused item that its module or its tests import
    (:func:`uses_library_mode`), and, through the loop, with what goes with that one."""
    refused = {i: list(c) for i, c in failed.items()}
    reasons: list[str] = []
    by_id = {it.item: it for it in man.items}
    ctx = _library_context(tree)
    needs = {
        it.item: (uses_library_mode(tree, it, ctx))
        for it in man.items
    }
    grew = True
    while grew:  # at most one pass per item
        grew = False
        for it in man.items:
            if it.item in refused:
                continue
            for r in sorted(refused):
                shared = set(it.tests) & set(by_id[r].tests) if r in by_id else set()
                if r in needs[it.item] or shared:
                    refused[it.item] = [DEPENDENCY]
                    how = "shares a test file with" if shared else "uses"
                    reasons.append(f"{it.item} is refused with {r}, which it {how}")
                    grew = True
                    break
    return refused, reasons


def _restore_file(dest: Path, parent: Path, rel: str, had: bool) -> None:
    target = dest / rel
    if had:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / rel, target)
    elif target.is_file():
        target.unlink()


def build(
    man: Manifest,
    raw: dict,
    refused: dict[str, list[str]],
    parent_files: set[str],
    parent_tree: Path,
    cand_tree: Path,
    dest: Path,
) -> dict | None:
    """Write the reduced tree at *dest* and return the reduced manifest, or None when there is none.

    *parent_files* are the parent's paths: a refused item's file and every test file it lists are restored to the
    parent's version, or removed when the parent had none.
    """
    kept = [it for it in man.items if it.item not in refused]
    gone = [it for it in man.items if it.item in refused]
    if not kept:
        return None
    shutil.copytree(cand_tree, dest)
    for it in gone:
        _restore_file(dest, parent_tree, it.path, it.path in parent_files)
        for t in it.tests:
            _restore_file(dest, parent_tree, t, t in parent_files)
    out = dict(raw)
    keep_ids = {it.item for it in kept}
    out["items"] = [r for r in raw.get("items", []) if r.get("item") in keep_ids]
    return out
