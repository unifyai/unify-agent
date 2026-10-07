"""``UNIFY_STORE_FROM_SESSION``: store a function the session defined by its name, so its source is never retyped.

The storage review stores a function by writing its source into the
``implementations`` argument of ``add_functions`` (or ``functions.add``):
code the session already ran, retyped from the conversation into a JSON
string. On the 6-7 Oct office runs (research artifact
``overhaul-lanes/runtime-20261006/escaped-newline-v1``) 16 of 41 review
writes whose source held an escape came out with one escaping level too
many (``"\\n"`` for ``"\n"``, ``r"\\b"`` for ``r"\b"``); none of 74 task
cells did. The stored function then wrote a literal backslash-n, or matched
nothing, on every reuse -- 17 of 31 office failures, 0 of 213 passes.

With this switch on, while the storage review runs (:func:`reviewing`), an
implementation that is only a name (``"find_weekly_overtime_names"``) is
the source of that function as the latest of the session's Python code cells
that defines it at top level ran it: the ``def`` with its decorators, byte for
byte, with the module-level imports of the session's cells that it uses
moved to the top of its body (a stored function may hold no module-level
statement). The result goes through every check a written source goes
through. A name no cell defines is reported as an error for that entry. The
review is told it may do this; writing the source out stays possible, for a
changed version. Off, or outside a review: nothing is resolved and the
review is told nothing.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import keyword
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

_TRAJECTORY: contextvars.ContextVar[Optional[Sequence[Mapping[str, Any]]]] = (
    contextvars.ContextVar("unify_store_from_session", default=None)
)

_PARSE_FLAGS = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT

REVIEW_NOTE = (
    "## Storing A Function This Session Ran\n\n"
    "To store a function the session defined in one of its code cells, you "
    "may pass its name alone as the implementation, for example "
    '`implementations=["total_payments"]`: the source exactly as that cell '
    "ran it is stored, with the imports it uses moved inside it, so nothing "
    "has to be retyped. To store a changed version, write its full source as "
    "usual.\n\n"
)


@contextlib.contextmanager
def reviewing(trajectory: Optional[Sequence[Mapping[str, Any]]]) -> Iterator[None]:
    """While the block runs (and in the tasks it starts), names resolve against *trajectory*'s cells."""
    token = _TRAJECTORY.set(trajectory)
    try:
        yield
    finally:
        _TRAJECTORY.reset(token)


def review_note() -> str:
    """The review's paragraph on storing by name."""
    return REVIEW_NOTE


def cells() -> List[str]:
    """The Python code of the reviewed session's cells that ran, in order ([] outside a review)."""
    trajectory = _TRAJECTORY.get()
    if not trajectory:
        return []
    from .origin_capture import _code_cells

    return [
        code
        for _, code, language, _ in _code_cells(trajectory)
        if language.lower() == "python"
    ]


def is_name(implementation: Any) -> bool:
    """Whether *implementation* is a bare function name rather than source."""
    if not isinstance(implementation, str):
        return False
    text = implementation.strip()
    return text.isidentifier() and not keyword.iskeyword(text)


def _parse(code: str) -> Optional[ast.Module]:
    try:
        return compile(code, "<cell>", "exec", flags=_PARSE_FLAGS)
    except (SyntaxError, ValueError):
        return None


def _import_text(node: ast.stmt, alias: ast.alias) -> str:
    name = alias.name + (f" as {alias.asname}" if alias.asname else "")
    if isinstance(node, ast.ImportFrom):
        return f"from {'.' * node.level}{node.module or ''} import {name}"
    return f"import {name}"


def _bound(alias: ast.alias, node: ast.stmt) -> str:
    if alias.asname:
        return alias.asname
    return alias.name if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]


def _imports(trees: Sequence[ast.Module]) -> Dict[str, str]:
    """Each name the cells' top-level imports bind, to its import (the latest wins)."""
    out: Dict[str, str] = {}
    for tree in trees:
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                out[_bound(alias, node)] = _import_text(node, alias)
    return out


def _local_names(fn: ast.AST) -> set:
    """Parameters and names the function binds itself: an import must not shadow them."""
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(_bound(a, node) for a in node.names if a.name != "*")
    return names


def _with_imports(source: str, fn: ast.AST, imports: Mapping[str, str]) -> str:
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    wanted = sorted(
        {imports[name] for name in used - _local_names(fn) if name in imports},
    )
    if not wanted:
        return source
    tree = _parse(source)
    node = tree.body[0] if tree and tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return source
    body = node.body
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
        and len(body) > 1
    ):
        anchor = body[1]  # after the docstring
    else:
        anchor = first
    if anchor.lineno == node.lineno:
        return source  # a one-line body: left as it ran
    lines = source.splitlines(keepends=True)
    indent = " " * anchor.col_offset
    at = anchor.lineno - 1
    return "".join(lines[:at] + [f"{indent}{text}\n" for text in wanted] + lines[at:])


def resolve(name: str) -> Tuple[Optional[str], str]:
    """``(source, "")`` for the session's latest top-level definition of *name*, or ``(None, why)``."""
    trees: List[Tuple[str, ast.Module]] = []
    for code in cells():
        tree = _parse(code)
        if tree is not None:
            trees.append((code, tree))
    for index in range(len(trees) - 1, -1, -1):
        code, tree = trees[index]
        found = None
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                found = node
        if found is None:
            continue
        start = min([d.lineno for d in found.decorator_list] + [found.lineno])
        lines = code.splitlines(keepends=True)
        source = "".join(lines[start - 1 : found.end_lineno])
        if not source.endswith("\n"):
            source += "\n"
        imports = _imports([t for _, t in trees[: index + 1]])
        return _with_imports(source, found, imports), ""
    return None, (
        f"no Python code cell of this session defines a top-level function "
        f"`{name}`; write its source out instead"
    )


def expand(implementations: Sequence[Any]) -> Tuple[List[Any], Dict[str, str]]:
    """*implementations* with each bare name replaced by its session source, and an error per name not found.

    Outside a review: returned unchanged, with no errors.
    """
    if _TRAJECTORY.get() is None:
        return list(implementations), {}
    out: List[Any] = []
    errors: Dict[str, str] = {}
    for implementation in implementations:
        if not is_name(implementation):
            out.append(implementation)
            continue
        name = implementation.strip()
        source, why = resolve(name)
        if source is None:
            errors[name] = f"error: {why}"
        else:
            out.append(source)
    return out, errors


__all__ = [
    "REVIEW_NOTE",
    "cells",
    "expand",
    "is_name",
    "resolve",
    "review_note",
    "reviewing",
]
