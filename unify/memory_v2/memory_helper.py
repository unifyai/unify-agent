"""The ``memory`` helper: find, inspect and list the functions of this memory library.

In a cell::

    import memory
    memory.find(value)      # the functions whose recorded inputs have the shape of value
    memory.describe(fn)     # one function's documentation: signature, input form, sections, example
    memory.catalog()        # the library's README: every channel and function

*value* is data you hold: a path to a file, the file's bytes or text, or a parsed value (a dict or a list,
or a string holding JSON). *fn* is an imported function, or its name (``"env.<channel>.<function>"`` or the
bare function name when only one channel has it).

Matching is by data shape only, never by words: a file's format, delimiter, header and columns, or the key
tree of a JSON value, against the shapes of the inputs each function was built and checked on (the harness
records them when the function is admitted). ``exact`` means the same shape, column types included;
``structure`` means the same columns or keys with other value types (a column that is empty here, say;
a list of scalars or an empty list counts as any such list).
A plain text or a binary file has no shape to match and finds nothing.

The helper reads only the catalogue the harness wrote beside it (``.memory/catalog.json``, rendered from
the library's commit) and the README; it makes no network call and no call to the harness, and it imports
only the standard library. Its file is generated with the library: edits are discarded.
"""

from __future__ import annotations

import json
import os
from typing import Any, NamedTuple

CATALOG_FILE = ".memory/catalog.json"
README_FILE = "README.md"
SHAPES_FILE = ".memory/shapes.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAX_PATH_CHARS = 4096
_STRUCTURELESS = frozenset({"text", "binary", "xlsx-unparsed"})


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


def value_shape(value: Any) -> dict | None:
    """The input shape of a parsed value (a dict or list, or a string holding one), or None."""
    parsed = _parsed(value)
    if parsed is None:
        return None
    try:
        return {"kind": "value", "tree": json.loads(_canon(_shapes.tree(parsed)))}
    except (TypeError, ValueError, RecursionError):
        return None


def _skeleton(tree: Any) -> Any:
    """A key tree without leaf types: keys kept, scalars alike, a list of records keeps its records' keys."""
    if isinstance(tree, dict):
        return {k: _skeleton(v) for k, v in tree.items()}
    if isinstance(tree, list):
        records = sorted({_canon(_skeleton(t)) for t in tree if isinstance(t, dict)})
        return ["list", records] if records else "list"
    return "scalar"


def keys(desc: dict) -> tuple[str | None, str | None]:
    """(exact key, structure key) of a descriptor: equal keys mean the same shape at that level."""
    if desc.get("kind") == "value":
        tree = desc.get("tree")
        return _canon(["tree", tree]), _canon(["tree", _skeleton(tree)])
    s = desc.get("shape") or {}
    fmt = s.get("format")
    exact = _canon(["file", s])
    if fmt == "csv":
        cols = s.get("columns") if s.get("header") else len(s.get("types") or [])
        return exact, _canon(["table", s.get("delimiter"), bool(s.get("header")), cols])
    if fmt in ("json", "yaml"):
        return exact, _canon(["tree", _skeleton(s.get("tree"))])
    if fmt == "jsonl":
        lines = sorted({_canon(_skeleton(t)) for t in s.get("tree") or []})
        return exact, _canon(["lines", lines])
    if fmt == "xlsx":
        return exact, _canon(["sheets", s.get("sheets")])
    return exact, None


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
    match: str  # exact | structure
    summary: str


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

    Ranked by match level (``exact`` before ``structure``), then by how many recorded input shapes
    match, then by name. An empty list means no stored function was built on data of this shape.
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
    mine = [keys(d) for d in _value_candidates(value, exts)]
    exact = {k for k, _ in mine if k is not None}
    structure = {k for _, k in mine if k is not None}
    ranked: list[tuple[int, int, str, Found]] = []
    for e in entries:
        hits = {2: 0, 1: 0}
        for d in e.get("input_shapes", []):
            k_exact, k_structure = keys(d)
            if k_exact in exact:
                hits[2] += 1
            elif k_structure is not None and k_structure in structure:
                hits[1] += 1
        level = 2 if hits[2] else 1 if hits[1] else 0
        if level:
            name = f"{e['module']}.{e['name']}"
            found = Found(
                name,
                e.get("signature", ""),
                e.get("input", ""),
                "exact" if level == 2 else "structure",
                e.get("summary", ""),
            )
            ranked.append((-level, -hits[level], name, found))
    return [f for *_, f in sorted(ranked)]


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
