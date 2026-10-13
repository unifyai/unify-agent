"""The ``memory`` helper: find the functions of this memory library by the shape of data you hold.

In a cell::

    import memory
    memory.find(value)      # the functions whose recorded inputs have the shape of value

*value* is data you hold: a path to a file, the file's bytes or text, or a parsed value (a dict or a list,
or a string holding JSON).

The library's catalogue (its channels and functions, and one function's documentation) is ``env.items()`` and
``env.get(fn)`` (:mod:`.env_helper`, the export's ``env/__init__.py``). ``memory.catalog`` and ``memory.describe``
still work for one build: each prints one line naming its new call, then returns that call's text.

Matching is by data shape only, never by words. A shape's **signature** is its named key paths (a JSON
value's or file's keys, ``a.b``, ``items[].id``; a table's header columns, ``[].column``) with the type
of each leaf: int, float, bool, str, null, a CSV column's type, ``list`` or ``object``. Only a value whose
signature has at least one named key can match (a dict with a key, a list of records, a table with a
header); ``[]``, ``{}``, scalars, lists of scalars and headerless tables find nothing, because their shape
says almost nothing about what they are. A recorded shape matches when no shared key path has another type
(null and empty columns fit any type) and the shared named keys are at least half of both signatures' named
keys. ``exact`` means the same signature, list length classes (0, 1, 2–9, 10–99, 100+) and file format;
``structure`` means a match with some keys or types differing. A match on a single named key says little
(``{"status": "error"}`` fits every status envelope), so it is labelled ``weak`` and ranked last, unless a
typed structure sits under that key in both (``{"grid": [[0, 1]]}``: ``grid[][]`` holds ints). Results are
ranked by level (exact, structure, weak), then by the share of named keys matched, then by how many recorded
shapes match, then by name, at most :data:`MAX_RESULTS`, each with the keys or columns that matched. A
value of more than :data:`MAX_NODES` nodes is shaped from its first part (its descriptor says
``truncated``), so a huge nested value is matched in bounded time. A DataFrame, array or other object
that is not a dict, list or JSON text has no shape here: pass its records or its file instead.

The helper reads only the catalogue the harness wrote beside it (``.memory/catalog.json``, rendered from
the library's commit, with the harness's suspect flags); it makes no network call and no call to the
harness, and it imports only the standard library. Its file is generated with the library: edits are
discarded.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, NamedTuple

CATALOG_FILE = ".memory/catalog.json"
README_FILE = "README.md"
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
# What one catalog() print may cost, in estimated tokens (ceil(characters / 4)); past it the view pages.
CATALOG_MAX_TOKENS = 1500
_CATALOG_MAX_CHARS = CATALOG_MAX_TOKENS * 4
_LINE_MAX_CHARS = (
    300  # one listed line in a paged view (a long summary is cut with "…")
)


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
    under it in both signatures (a list or grid of the same scalar type)."""
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


_LEVELS = {2: "exact", 1: "structure", 0: "weak"}


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
    grid) and, for a file, its format, delimiter, header flag, columns and column types. Lengths, counts and
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
    raw grids and scalar lists match an identical shape. The gate uses it for an example's fixture.
    """
    return shape_key(mine) == shape_key(recorded)


# --- the catalogue -----------------------------------------------------------------------------------------


class _Text(str):
    """Text that shows as itself when a cell's last expression evaluates to it."""

    def __repr__(self) -> str:
        return str(self)


def _catalog() -> dict:
    path = os.path.join(_HERE, CATALOG_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"memory: the catalogue {path} cannot be read ({type(exc).__name__}); use "
            "help(env.<channel>) and the files under the library root instead",
        ) from exc


class Found(NamedTuple):
    """One function :func:`find` lists: its import name, how to call it and how its inputs matched."""

    name: str  # env.<channel>.<function>
    signature: str
    input: (
        str  # the form its first parameter takes (path, text, bytes, observation, env)
    )
    match: str  # exact | structure | weak (a single named key)
    summary: str
    reason: str  # the keys or columns that matched


def _value_candidates(value: Any, exts: list[str]) -> list[dict]:
    """The descriptors *value* can be seen as: a file (by its own name, or by each recorded extension)
    and a parsed value."""
    out: list[dict] = []
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if (
        isinstance(value, str)
        and value
        and len(value) <= _MAX_PATH_CHARS
        and "\n" not in value
        and "\0" not in value
        and os.path.isfile(value)
    ):
        with open(value, "rb") as fh:
            data = fh.read(_shapes.PARSE_LIMIT + 1)
        desc = file_shape(value, data)
        return [desc] if desc is not None else []
    if isinstance(value, bytearray):
        value = bytes(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    if isinstance(data, bytes):
        for ext in exts:
            desc = file_shape("value" + ext, data[: _shapes.PARSE_LIMIT + 1])
            if desc is not None:
                out.append(desc)
    desc = value_shape(value)
    if desc is not None:
        out.append(desc)
    return out


def find(value: Any) -> list[Found]:
    """The functions whose recorded inputs have the shape of *value*, best match first.

    Ranked by match level (``exact``, then ``structure``, then ``weak``: one named key), then by the share of named keys matched, then
    by how many recorded input shapes match, then by name; at most :data:`MAX_RESULTS`. An empty list
    means no stored function was built on data of this shape, or the value has too little structure.
    """
    entries = _catalog().get("functions", [])
    exts = sorted(
        {
            d.get("ext", "")
            for e in entries
            for d in e.get("input_shapes", [])
            if d.get("kind") == "file"
        },
    )
    mine = _value_candidates(value, exts)
    ranked: list[tuple[int, float, int, str, Found]] = []
    for e in entries:
        best: tuple[int, float, list[str]] | None = None
        count = 0
        for d in e.get("input_shapes", []):
            hits = [m for m in (match(v, d) for v in mine) if m is not None]
            if not hits:
                continue
            top = max(hits, key=lambda m: (m[0], m[1]))
            if best is None or top[:2] > best[:2]:
                best, count = top, 1
            elif top[:2] == best[:2]:
                count += 1
        if best is None:
            continue
        name = f"{e['module']}.{e['name']}"
        level, share, matched = best
        shown = [p[3:] if p.startswith("[].") else p for p in matched]
        reason = (
            "matched "
            + ", ".join(shown[:8])
            + (f" and {len(shown) - 8} more" if len(shown) > 8 else "")
        )
        found = Found(
            name,
            e.get("signature", ""),
            e.get("input", ""),
            _LEVELS[level],
            e.get("summary", ""),
            reason,
        )
        ranked.append((-level, -share, -count, name, found))
    return [f for *_, f in sorted(ranked, key=lambda r: r[:4])][:MAX_RESULTS]


# --- the environment catalogue's old names: env.items and env.get ------------------------------------------

#: The environment catalogue's module beside this file: the export's ``env/__init__.py``.
ENV_HELPER_FILE = os.path.join("env", "__init__.py")
_ENV: list[Any] = []


def _env() -> Any:
    """The module ``env.items`` and ``env.get`` live in: the harness's own, or the export's ``env/__init__.py``
    beside this file, loaded once by its path (a cell's own ``import env`` is left as it is)."""
    if _ENV:
        return _ENV[0]
    if __package__:  # inside the harness package
        from . import env_helper as module
    else:
        import importlib.util
        import sys

        name = "_memory_env_catalogue"
        spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, ENV_HELPER_FILE))
        if spec is None or spec.loader is None:
            raise ImportError("memory: the environment catalogue (env.items, env.get) is missing")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    _ENV.append(module)
    return module


def _arg(value: Any) -> str:
    """*value* as the renamed call shows it: a function by its name, anything else by its repr (cut)."""
    if callable(value) and not isinstance(value, str):
        return str(getattr(value, "__name__", type(value).__name__))
    text = repr(value)
    return text if len(text) <= 60 else text[:59] + "…"


def _renamed(old: str, new: str, args: list[str]) -> None:
    """The one line an old name prints: the call as the new name takes it."""
    shown = ", ".join(args)
    print(f"{old}({shown}) is now {new}({shown}); import env to call it. The old name will be removed.")


def catalog(channel: Any = None, page: int = 1) -> str:
    """Renamed :func:`env.items` (``import env``): prints one line naming it, then returns its listing as text.

    Every channel (function and note counts, summary, suspect flag) and every function, or with *channel* one
    channel's functions, in pages (*page*) past the bound; ``help(env.items)`` says the rest.
    """
    args = ([] if channel is None else [_arg(channel)]) + ([] if page == 1 else [f"page={_arg(page)}"])
    _renamed("memory.catalog", "env.items", args)
    items = _env().items
    return _Text(str(items(page=page) if channel is None else items(channel, page=page)))


def describe(fn_or_name: Any) -> str:
    """Renamed :func:`env.get` (``import env``): prints one line naming it, then returns the function's
    documentation as text (signature, summary, input form, effect, sections and recorded input shapes).
    ``help(env.get)`` says the rest."""
    _renamed("memory.describe", "env.get", [_arg(fn_or_name)])
    return _Text(str(_env().get(fn_or_name)))
