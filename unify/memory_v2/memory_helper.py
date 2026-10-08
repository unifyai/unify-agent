"""The ``memory`` helper: find, inspect and list the functions of this memory library.

In a cell::

    import memory
    memory.find(value)      # the functions whose recorded inputs have the shape of value
    memory.describe(fn)     # one function's documentation: signature, input form, sections, example
    memory.catalog()        # the library's README: every channel and function

*value* is data you hold: a path to a file, the file's bytes or text, or a parsed value (a dict or a list,
or a string holding JSON). *fn* is an imported function, or its name (``"env.<channel>.<function>"`` or the
bare function name when only one channel has it).

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
the library's commit) and the README; it makes no network call and no call to the harness, and it imports
only the standard library. Its file is generated with the library: edits are discarded.
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


def _entry(fn_or_name: Any) -> dict:
    entries = _catalog().get("functions", [])
    if callable(fn_or_name) and not isinstance(fn_or_name, str):
        module = getattr(fn_or_name, "__module__", "") or ""
        wanted = f"{module}.{getattr(fn_or_name, '__name__', '')}"
    else:
        wanted = str(fn_or_name).strip()
        if (
            wanted.startswith("env/") and ":" in wanted
        ):  # an item id, env/<channel>:<function>
            wanted = wanted.replace("/", ".", 1).replace(":", ".", 1)
        if wanted.count(".") == 1:
            wanted = "env." + wanted
    exact = [e for e in entries if f"{e['module']}.{e['name']}" == wanted]
    if exact:
        return exact[0]
    by_name = [e for e in entries if e["name"] == wanted]
    if len(by_name) == 1:
        return by_name[0]
    if by_name:
        names = ", ".join(f"{e['module']}.{e['name']}" for e in by_name)
        raise LookupError(
            f"memory: {wanted!r} names several functions ({names}); pass one of them",
        )
    raise LookupError(
        f"memory: no function {wanted!r} in the catalogue; memory.catalog() lists every function",
    )


def _shape_text(desc: dict) -> str:
    if desc.get("kind") == "value":
        tree = desc.get("tree")
        if isinstance(tree, dict):
            return "a value with keys " + ", ".join(sorted(tree))
        return "a list value"
    s = desc.get("shape") or {}
    fmt = s.get("format")
    ext = desc.get("ext") or ""
    if fmt == "csv":
        cols = s.get("columns")
        what = (
            f"columns {', '.join(cols)}"
            if cols
            else f"{len(s.get('types') or [])} columns, no header"
        )
        return f"a {ext or 'delimited'} table with {what}"
    if fmt in ("json", "yaml", "jsonl"):
        tree = s.get("tree")
        if isinstance(tree, dict):
            return f"a {fmt} file with keys " + ", ".join(sorted(tree))
        return f"a {fmt} file of lists or records"
    if fmt == "xlsx":
        return "a workbook with sheets " + ", ".join(
            str(x.get("name")) for x in s.get("sheets") or []
        )
    return f"a {fmt} file"


def _indent(text: str) -> str:
    return "\n".join(("    " + ln) if ln.strip() else "" for ln in text.splitlines())


def describe(fn_or_name: Any) -> str:
    """One function's documentation: signature, summary, input form, effect, sections and recorded inputs."""
    e = _entry(fn_or_name)
    cat = _catalog()
    doc = e.get("doc", {})
    out = [f"{e['module']}.{e.get('signature', e['name'])}", e.get("summary", "")]
    if doc.get("description"):
        out += ["", doc["description"]]
    form = e.get("input", "")
    meaning = cat.get("input_forms", {}).get(form, "")
    out.append("")
    if form:
        out.append(f"Input: {form}" + (f" ({meaning})" if meaning else ""))
    if e.get("effect"):
        out.append(f"Effect: {e['effect']}")
    for title, key in (
        ("Args", "args"),
        ("Returns", "returns"),
        ("Raises", "raises"),
        ("Example", "example"),
        ("Use when", "use_when"),
        ("Don't use when", "dont_use_when"),
        ("Notes", "notes"),
    ):
        if doc.get(key):
            out += [f"{title}:", _indent(doc[key])]
    shapes = e.get("input_shapes")
    if shapes:
        out.append("Built and checked on inputs shaped as:")
        out += [f"    - {_shape_text(d)}" for d in shapes]
    if doc.get("example"):
        out.append(
            f"Example paths are relative to the library root ({_HERE}); check the example before "
            "relying on the function.",
        )
    return _Text("\n".join(out).rstrip() + "\n")


def catalog() -> str:
    """The library's README: every channel and function, with signatures, summaries and input forms."""
    with open(os.path.join(_HERE, README_FILE), encoding="utf-8") as fh:
        return _Text(fh.read())
