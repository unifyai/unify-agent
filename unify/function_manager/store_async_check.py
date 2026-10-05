"""Store-time warning (``UNIFY_STORE_ASYNC_CHECK``): stored code that awaits a synchronous environment method.

A method an environment registers (``primitives.<namespace>.<method>``)
keeps its callable's kind: a synchronous one returns its value directly, so
``await primitives.<namespace>.<method>(...)`` raises "object list can't be
used in 'await' expression" when the stored function runs from code or as
another function's dependency. Nothing on the storage path runs the body,
and ``execute_function``'s instrumented snippet tolerates the ``await``, so
such a function can be stored and even pass its first call. In the AppWorld
HIGH cell of the overhaul build with every lean switch on, all 11 stored
functions that called the environment did this.

With the switch on, ``add_functions`` and ``patch_function`` read every
stored function after a write and return a warning, beside the stored
status, that names each such line in the function just written and every
other stored function with the same pattern. It informs and never refuses:
the function is stored as it was given. The check is static: an ``await``
whose call is ``primitives.<namespace>.<method>(...)``, or ``<alias>.<method>(...)``
where the function bound ``<alias> = primitives.<namespace>``, and whose
method is registered and not asynchronous (``environment.is_async_method``).
A function that binds ``primitives`` itself is left alone. Off, nothing is
read and results are as shipped.
"""

from __future__ import annotations

import ast
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

_EXCERPT_CHARS = 100


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(SETTINGS.UNIFY_STORE_ASYNC_CHECK)


def synchronous_methods() -> Dict[str, frozenset[str]]:
    """Each registered environment namespace's synchronous method names."""
    from .primitives.environment import environment_namespaces, is_async_method

    return {
        alias: frozenset(m.name for m in namespace.methods if not is_async_method(m))
        for alias, namespace in environment_namespaces().items()
    }


def _namespace_of(node: ast.expr, aliases: Mapping[str, str]) -> Optional[str]:
    """The namespace ``node`` names: ``primitives.<namespace>`` or a bound alias of one."""
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "primitives"
    ):
        return node.attr
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    return None


def awaited_sync_calls(
    source: str,
    sync_methods: Mapping[str, Iterable[str]],
) -> List[Tuple[int, str]]:
    """``(line, excerpt)`` of every ``await`` of a synchronous environment method in ``source``.

    Lines count from 1 in ``source`` as stored. A source that does not parse,
    or whose function binds ``primitives`` itself, has none.
    """
    if not sync_methods:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.arg == "primitives":
            return []
        if (
            isinstance(node, ast.Name)
            and node.id == "primitives"
            and isinstance(node.ctx, ast.Store)
        ):
            return []
    aliases: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                namespace = _namespace_of(node.value, {})
                if namespace is not None:
                    aliases[target.id] = namespace
    lines = source.splitlines()
    found: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not isinstance(func, ast.Attribute):
            continue
        namespace = _namespace_of(func.value, aliases)
        if namespace is None or func.attr not in set(sync_methods.get(namespace, ())):
            continue
        text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        if len(text) > _EXCERPT_CHARS:
            text = text[: _EXCERPT_CHARS - 3] + "..."
        found.append((node.lineno, text))
    return sorted(set(found))


def library_hits(
    sources: Mapping[str, str],
    sync_methods: Mapping[str, Iterable[str]],
) -> Dict[str, List[Tuple[int, str]]]:
    """Every stored function (name -> source) that awaits a synchronous environment method."""
    hits: Dict[str, List[Tuple[int, str]]] = {}
    for name, source in sources.items():
        found = awaited_sync_calls(source, sync_methods)
        if found:
            hits[name] = found
    return hits


def _lines_text(found: List[Tuple[int, str]]) -> str:
    return "; ".join(f"line {line}: `{text}`" for line, text in found)


def _others_text(others: Mapping[str, List[Tuple[int, str]]]) -> str:
    return "; ".join(
        f"'{name}' (line{'s' if len(found) > 1 else ''} "
        f"{', '.join(str(line) for line, _ in found)})"
        for name, found in sorted(others.items())
    )


def warnings_for(
    written: Iterable[str],
    hits: Mapping[str, List[Tuple[int, str]]],
) -> Dict[str, str]:
    """The warning for each function just written, by name.

    A written function with the pattern gets its own lines and the other
    stored functions with it. When none of the written functions has it but
    other stored functions do, the first written function carries the list,
    once per write.
    """
    warnings: Dict[str, str] = {}
    written = list(written)
    lead = (
        "awaits a synchronous environment method, which returns its value "
        "directly, so `await` raises TypeError when the function runs"
    )
    for name in written:
        own = hits.get(name)
        if not own:
            continue
        others = {n: f for n, f in hits.items() if n != name}
        text = f"'{name}' {lead}: {_lines_text(own)}. Call it without `await`."
        if others:
            text += (
                " Other stored functions with the same pattern: "
                f"{_others_text(others)}."
            )
        warnings[name] = text
    if not warnings and written and hits:
        warnings[written[0]] = (
            "Stored functions that await a synchronous environment method "
            "(it returns its value directly, so `await` raises TypeError when "
            f"they run): {_others_text(hits)}."
        )
    return warnings


__all__ = [
    "awaited_sync_calls",
    "enabled",
    "library_hits",
    "synchronous_methods",
    "warnings_for",
]
