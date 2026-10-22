"""Input shapes and the shape matcher (rev 2's library mode).

:func:`file_shape` and :func:`value_shape` describe a value by its shape (named key paths with each leaf's type,
list length classes, a file's format), never its words; :func:`match`, :func:`compare` and :func:`same_shape`
compare two shapes. The library export ships this module, from its imports on, as ``.memory/matcher.py``
(:func:`.library_export.matcher_source`), so the actor's copy matches shapes with the same functions the harness
recorded them with.

Matching is by data shape only. A shape's **signature** is its named key paths (a JSON value's or file's keys,
``a.b``, ``items[].id``; a table's header columns, ``[].column``) with the type of each leaf: int, float, bool, str,
null, a CSV column's type, ``list`` or ``object``. Only a value whose signature has at least one named key can
match; ``[]``, ``{}``, scalars, lists of scalars and headerless tables match nothing, because their shape says
almost nothing about what they are. A recorded shape matches when no shared key path has another type (null and
empty columns fit any type) and the shared named keys are at least half of both signatures' named keys. ``exact``
means the same signature, list length classes (0, 1, 2–9, 10–99, 100+) and file format; ``structure`` means a match
with some keys or types differing; a match on a single named key is ``weak`` unless a typed structure sits under that
key in both. A value of more than :data:`MAX_NODES` nodes is shaped from its first part (its descriptor says
``truncated``).
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any

SHAPES_FILE = ".memory/shapes.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAX_PATH_CHARS = 4096
_STRUCTURELESS = frozenset({"text", "binary", "xlsx-unparsed"})
MAX_RESULTS = 5
MIN_SHARE = 0.5  # shared named keys over the union of both signatures' named keys
_ANY = frozenset({"null", "empty"})  # leaf types that fit any other
_MAX_DEPTH = 8
_MAX_ELEMENTS = 50  # list elements walked for length classes
# Nodes each shape walk of a value visits (its length classes, its key tree); past it the descriptor is
# marked ``"truncated": true`` and shows the part walked, so a huge nested value is shaped in bounded time.
MAX_NODES = 20_000
_MAX_SIGNATURE_PATHS = 20_000


def _load_shapes() -> Any:
    """The shape functions: the harness's own module, or the copy shipped beside this file."""
    if __package__:  # inside the harness package
        from .analysis import shapes

        return shapes
    import importlib.util

    path = os.path.join(_HERE, SHAPES_FILE)
    spec = importlib.util.spec_from_file_location("_memory_shapes", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"memory: the shape functions are missing ({path})")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_shapes = _load_shapes()


# --- input-shape descriptors (the harness records these when it admits a function) -------------------------


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def file_shape(name: str, data: bytes) -> dict | None:
    """The input shape of a file called *name* holding *data*, or None when it has no structure."""
    s = _shapes.shape(name or "", bytes(data))
    if s.get("format") in _STRUCTURELESS or isinstance(s.get("tree"), str):
        return None
    return {
        "kind": "file",
        "ext": _shapes.extension(name or ""),
        "shape": {k: v for k, v in s.items() if k != "stats"},
    }


def _parsed(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        try:
            parsed = json.loads(value)
        except (ValueError, UnicodeDecodeError, RecursionError):
            return None
        return parsed if isinstance(parsed, (dict, list)) else None
    return None


def _length_class(n: int) -> str:
    return (
        "0"
        if n == 0
        else "1" if n == 1 else "2-9" if n < 10 else "10-99" if n < 100 else "100+"
    )


class _Budget:
    """Nodes a walk may still visit."""

    __slots__ = ("left",)

    def __init__(self, n: int) -> None:
        self.left = n

    def take(self) -> bool:
        self.left -= 1
        return self.left >= 0


def _lengths(
    value: Any,
    path: str,
    out: dict[str, str],
    depth: int = 0,
    budget: _Budget | None = None,
) -> None:
    """The length class of each list in *value*, by its key path (the first list seen at a path wins)."""
    if depth > _MAX_DEPTH:
        return
    if isinstance(value, dict):
        for k in sorted(value, key=str):
            if budget is not None and not budget.take():
                return
            _lengths(
                value[k],
                f"{path}.{k}" if path else str(k),
                out,
                depth + 1,
                budget,
            )
    elif isinstance(value, (list, tuple)):
        out.setdefault(path, _length_class(len(value)))
        for v in itertools.islice(value, _MAX_ELEMENTS):
            if budget is not None and not budget.take():
                return
            _lengths(v, path + "[]", out, depth + 1, budget)


_CUT = object()  # a container the budget ran out in before any of its children was kept


def _pruned(value: Any, budget: _Budget, depth: int = 0) -> Any:
    """*value* as :func:`.analysis.shapes.tree` reads it, within *budget* nodes: a dict keeps the values of
    its first ``MAX_KEYS`` keys (the others map to None, so its key count stays), a list its first
    ``SAMPLE_ROWS`` elements, and a container at ``MAX_DEPTH`` is emptied (the tree reads none of it). Under
    the budget the tree of the copy is the tree of the value; past it the rest is left out, and a container
    cut before any child is dropped (:data:`_CUT`) rather than shown empty."""
    if isinstance(value, dict):
        if depth >= _shapes.MAX_DEPTH:
            return {}
        keys = sorted(value, key=str)
        out: dict = {}
        for k in keys[: _shapes.MAX_KEYS]:
            child = _pruned(value[k], budget, depth + 1) if budget.take() else _CUT
            if child is _CUT:
                return out if out else _CUT
            out[k] = child
        out.update(dict.fromkeys(keys[_shapes.MAX_KEYS :]))
        return out
    if isinstance(value, (list, tuple)):
        if depth >= _shapes.MAX_DEPTH:
            return []
        items: list = []
        for v in itertools.islice(value, _shapes.SAMPLE_ROWS):
            child = _pruned(v, budget, depth + 1) if budget.take() else _CUT
            if child is _CUT:
                return items if items else _CUT
            items.append(child)
        return items
    return value


def value_shape(value: Any) -> dict | None:
    """The input shape of a parsed value (a dict or list, or a string holding one), or None."""
    parsed = _parsed(value)
    if parsed is None:
        return None
    try:
        lengths: dict[str, str] = {}
        walked, kept = _Budget(MAX_NODES), _Budget(MAX_NODES)
        _lengths(parsed, "", lengths, budget=walked)
        pruned = _pruned(parsed, kept)
        if pruned is _CUT:
            pruned = type(parsed)()
        tree = json.loads(_canon(_shapes.tree(pruned)))
    except (TypeError, ValueError, RecursionError):
        return None
    out = {"kind": "value", "tree": tree, "lengths": lengths}
    if walked.left < 0 or kept.left < 0:
        out["truncated"] = True  # only the first MAX_NODES nodes were shaped
    return out


def _walk(tree: Any, path: str, out: dict[str, str]) -> None:
    """Flatten a key tree (:func:`.analysis.shapes.tree`) into key paths and leaf types (at most
    ``_MAX_SIGNATURE_PATHS`` of them)."""
    if len(out) >= _MAX_SIGNATURE_PATHS:
        return
    if isinstance(tree, dict):
        for k in sorted(tree):
            p = f"{path}.{k}" if path else str(k)
            sub = tree[k]
            if isinstance(sub, dict):
                out[p] = "object"
                _walk(sub, p, out)
            elif isinstance(sub, list):
                out[p] = "list"
                _walk(sub, p, out)
            else:
                out[p] = str(sub)
    elif isinstance(tree, list):  # the distinct shapes of a list's elements
        p = path + "[]"
        scalars = sorted({str(t) for t in tree if not isinstance(t, (dict, list))})
        if scalars:
            out[p] = "|".join(scalars)
        for t in tree:
            if isinstance(t, dict):
                _walk(t, p, out)
            elif isinstance(t, list):
                out.setdefault(p, "list")
                _walk(t, p, out)


def signature(desc: dict) -> dict[str, str]:
    """A descriptor's key paths and leaf types (see the module docstring); ``{}`` when it has none."""
    out: dict[str, str] = {}
    if desc.get("kind") == "value":
        _walk(desc.get("tree"), "", out)
        return out
    s = desc.get("shape") or {}
    fmt = s.get("format")
    if fmt == "csv" and s.get("header") and s.get("columns"):
        types = list(s.get("types") or [])
        for i, col in enumerate(s["columns"]):
            out[f"[].{col}"] = types[i] if i < len(types) else "empty"
    elif fmt in ("json", "yaml"):
        _walk(s.get("tree"), "", out)
    elif fmt == "jsonl":
        _walk(list(s.get("tree") or []), "", out)
    elif fmt == "xlsx":
        for sheet in s.get("sheets") or []:
            for col in sheet.get("header") or []:
                out[f"{sheet.get('name')}[].{col}"] = "cell"
    return out


def _named(path: str) -> bool:
    return bool(path) and not path.endswith("[]")


def _fits(a: str, b: str) -> bool:
    return a == b or a in _ANY or b in _ANY


def _format(desc: dict) -> Any:
    if desc.get("kind") == "value":
        return ["value", desc.get("lengths")]
    s = desc.get("shape") or {}
    return ["file", s.get("format"), s.get("delimiter"), s.get("encoding")]


def _strong(matched: list[str], a: dict[str, str], b: dict[str, str]) -> bool:
    """Whether a match says something: two or more shared named keys, or one with a typed structure nested
    under it in both signatures (a list, or a list of lists, of the same scalar type)."""
    if len(matched) >= 2:
        return True
    if not matched:
        return False
    key = matched[0]
    return any(
        p.startswith(key + "[")
        and a[p] == b[p]
        and a[p] not in _ANY | {"list", "object"}
        for p in a.keys() & b.keys()
    )


def match(mine: dict, recorded: dict) -> tuple[int, float, list[str]] | None:
    """:func:`compare` with the single-key floor: (2 exact, 1 structure or 0 weak, share, matched keys)."""
    found = compare(mine, recorded)
    if found is None:
        return None
    level, share, matched = found
    if not _strong(matched, signature(mine), signature(recorded)):
        level = 0
    return level, share, matched


def compare(mine: dict, recorded: dict) -> tuple[int, float, list[str]] | None:
    """How descriptor *mine* matches *recorded*: (2 exact or 1 structure, share of named keys matched,
    the matched named keys), or None. Structural only: key paths and leaf types, never words.
    """
    a, b = signature(mine), signature(recorded)
    named_a = {p for p in a if _named(p)}
    named_b = {p for p in b if _named(p)}
    if not named_a or not named_b:
        return None  # too little information to tell what the value is
    if any(not _fits(a[p], b[p]) for p in a.keys() & b.keys()):
        return None
    matched = sorted(named_a & named_b)
    share = len(matched) / len(named_a | named_b)
    if not matched or share < MIN_SHARE:
        return None
    exact = a == b and _format(mine) == _format(recorded)
    return (2 if exact else 1), share, matched


def shape_key(desc: dict) -> tuple:
    """A descriptor's exact shape, with no information floor: its kind, its whole :func:`signature` (every
    key path, named or not, with its leaf type, so ``[]`` -> ``int`` for a list of ints and ``[][]`` for a
    list of lists) and, for a file, its format, delimiter, header flag, columns and column types. Lengths, counts and
    encodings are left out: a fixture may be a shorter copy of a recorded input.
    """
    sig = tuple(sorted(signature(desc).items()))
    if desc.get("kind") == "value":
        return ("value", sig)
    s = desc.get("shape") or {}
    return (
        "file",
        s.get("format"),
        s.get("delimiter"),
        bool(s.get("header")),
        tuple(s.get("columns") or ()),
        tuple(s.get("types") or ()),
        tuple(
            (sheet.get("name"), tuple(sheet.get("header") or ()))
            for sheet in s.get("sheets") or []
        ),
        sig,
    )


def same_shape(mine: dict, recorded: dict) -> bool:
    """Whether two descriptors have exactly the same shape (:func:`shape_key`). Unlike :func:`compare`,
    which serves :func:`find` and needs a named key to say anything, this has no floor: headerless tables,
    raw lists of lists and scalar lists match an identical shape. The gate uses it for an example's fixture.
    """
    return shape_key(mine) == shape_key(recorded)
