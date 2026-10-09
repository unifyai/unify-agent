"""The literal lint of memory v2.1 (spec §9.1): a function may not hard-code a value that varies across its
recorded inputs.

A value *varies* (:func:`varying_values`) when:
- it is a JSON leaf whose key path holds at least two different values across the item's recorded inputs; or
- it is a token of a text that does not occur in every recorded input.

A code literal is refused (:func:`literal_problems`) when it is a string of at least :data:`LINT_MIN` characters,
or a number with at least 2 significant digits, and it equals a varying value. It must be a parameter.

What the lint looks at:
- the function's body, plus the module-level assignments whose names the function reads;
- not its docstring, not the messages of its ``raise`` statements, and not its parameter defaults (a default is a
  parameter).

Reasons name a line and a literal's kind and length, never its value (R10).
"""

from __future__ import annotations

import ast
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from .episodes import Action
from .held_out import _blob_sha, _form

LINT_MIN = 4
_TEXT_TOKEN = re.compile(r"[A-Za-z0-9_.:/@+-]+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?\Z")


def _parsed(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def input_value(a: Action, form: str | None, blob: Callable[[str], bytes]) -> Any:
    """A recorded input as the function receives it (an ``env`` input as its call's keywords and response)."""
    kind = getattr(a, "kind", "tool")
    if kind == "worktree":
        sha = _blob_sha(a)
        try:
            data = blob(sha) if sha else b""
        except (OSError, KeyError):
            data = b""
        return _parsed(data.decode("utf-8", "replace"))
    if kind == "tool" and _form(a, form) == "env":
        return {"kwargs": dict(a.kwargs or {}), "response": _parsed(a.response)}
    return _parsed(a.response)


def _key(v: Any) -> str | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        try:
            return "num:" + format(Decimal(str(v)).normalize(), "f")
        except (InvalidOperation, ValueError):
            return None
    if isinstance(v, str):
        return "str:" + v
    return None


def _num_key(tok: str) -> str | None:
    if not _NUM.match(tok):
        return None
    return _key(float(tok) if "." in tok else int(tok))


def _leaves(v: Any, path: str, out: list, depth: int = 0) -> None:
    if depth > 16:
        return
    if isinstance(v, dict):
        for k, x in v.items():
            _leaves(x, f"{path}.{k}", out, depth + 1)
    elif isinstance(v, list):
        for x in v:
            _leaves(x, path + "[]", out, depth + 1)
    else:
        out.append((path, v))


def varying_values(values: list[Any]) -> set[str]:
    by_path: dict[str, set[str]] = {}
    texts: list[set[str]] = []
    for v in values:
        leaves: list[tuple[str, Any]] = []
        _leaves(v, "", leaves)
        tokens: set[str] = set()
        for path, leaf in leaves:
            k = _key(leaf)
            if k is not None:
                by_path.setdefault(path, set()).add(k)
            if isinstance(leaf, str):
                for t in _TEXT_TOKEN.findall(leaf):
                    tokens.add("str:" + t)
                    nk = _num_key(t)
                    if nk is not None:
                        tokens.add(nk)
        texts.append(tokens)
    out: set[str] = set()
    for keys in by_path.values():
        if len(keys) >= 2:
            out |= keys
    if len(texts) >= 2:
        out |= set.union(*texts) - set.intersection(*texts)
    return out


def sig_digits(v: int | float) -> int:
    try:
        d = Decimal(str(abs(v))).normalize()
    except (InvalidOperation, ValueError):
        return 0
    if not d.is_finite():
        return 0
    return 1 if d == 0 else len(d.as_tuple().digits)


def literal_problems(
    source: bytes,
    function: str,
    varying: set[str],
    *,
    lint_min: int = LINT_MIN,
) -> list[tuple[int, str]]:
    try:
        mod = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    fn = None
    for n in mod.body:
        if (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function
        ):
            fn = n
    if fn is None:
        return []
    skip: set[int] = set()
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        skip.add(id(body[0].value))
    for d in [*fn.args.defaults, *(x for x in fn.args.kw_defaults if x is not None)]:
        skip |= {id(x) for x in ast.walk(d)}
    for n in ast.walk(fn):
        if isinstance(n, ast.Raise):
            skip |= {id(x) for x in ast.walk(n)}
    used = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    consts = [
        n for n in ast.walk(fn) if isinstance(n, ast.Constant) and id(n) not in skip
    ]
    for s in mod.body:
        if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            if any(isinstance(t, ast.Name) and t.id in used for t in targets):
                consts += [n for n in ast.walk(s.value) if isinstance(n, ast.Constant)]
    out: set[tuple[int, str]] = set()
    for c in consts:
        v = c.value
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, str):
            if len(v) >= lint_min and "str:" + v in varying:
                out.add((c.lineno, f"string of {len(v)} characters"))
        elif isinstance(v, (int, float)):
            if sig_digits(v) >= 2 and _key(v) in varying:
                out.add((c.lineno, "number"))
    return sorted(out)
