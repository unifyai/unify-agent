"""The memory library's helpers: ``memory.index``, ``memory.show`` and ``memory.find`` (spec v2.1 §6).

The harness generates this file as ``memory/__init__.py`` in every copy of the library; edits are discarded.
It imports only the standard library and reads only files beside the package: ``INDEX.md``, ``links.json``
and the harness's data under ``.memory/``. In a cell::

    import memory
    print(memory.index())                          # the whole index
    print(memory.index("<package>"))               # one package's section ("notes/<topic>" for notes)
    print(memory.show("memory.<package>.<module>:<name>"))   # source, links, status, records, history
    memory.find(value)                             # functions built on recorded inputs shaped like value

``find`` matches structured data only: a dict, a list of records, JSON text, or a structured file such as a
CSV with a header. It compares key paths and types with the shapes of the inputs each function was admitted
on, never words, so it cannot match plain text.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from typing import Any, NamedTuple

INDEX_FILE = "INDEX.md"
LINKS_FILE = "links.json"
ITEMS_FILE = ".memory/items.json"
FIND_FILE = ".memory/find.json"
MATCHER_FILE = ".memory/matcher.py"
SHAPES_FILE = ".memory/shapes.py"
ITEM_LINE = "- `"


def split_index(text: str) -> tuple[str, list[tuple[str, str]]]:
    """An index as its head (before the first ``## `` heading) and its sections (heading, body)."""
    head: list[str] = []
    sections: list[list[str]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            sections.append([line[3:].rstrip("\n"), ""])
        elif sections:
            sections[-1][1] += line
        else:
            head.append(line)
    return "".join(head), [(h, b) for h, b in sections]


def section_key(heading: str) -> str:
    """``memory.<package>`` or ``notes/<topic>``: a heading without its summary."""
    return heading.split(" — ", 1)[0].strip()


def wanted_key(name: object) -> str:
    """The section a name means: ``"office"`` and ``"memory/office"`` both mean ``memory.office``."""
    n = str(name).strip().strip("/")
    if n.startswith(("memory.", "notes/")):
        return n
    if n.startswith("memory/"):
        return "memory." + n[len("memory/") :].replace("/", ".")
    return "memory." + n


def find_section(text: str, name: object) -> str:
    """The full section of the index *text* that *name* means; LookupError naming the sections otherwise."""
    key = wanted_key(name)
    _, sections = split_index(text)
    for heading, body in sections:
        if section_key(heading) == key:
            return f"## {heading}\n{body}"
    keys = ", ".join(section_key(h) for h, _ in sections) or "(none)"
    raise LookupError(f"memory: no index section {key!r}; the sections are: {keys}")


def item_count(body: str) -> int:
    """The item lines of a section body."""
    return sum(1 for line in body.splitlines() if line.startswith(ITEM_LINE))


__all__ = ["Found", "find", "index", "show"]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(
    _PKG_DIR,
)  # the copy's root: INDEX.md, links.json, memory/, notes/, .memory/
_LEVELS = {2: "exact", 1: "structure", 0: "weak"}
_MATCHER: Any = None


class _Text(str):
    """Text that shows as itself when a cell's last expression evaluates to it."""

    def __repr__(self) -> str:
        return str(self)


class Found(NamedTuple):
    """One function :func:`find` lists: its id, how to call it and how its inputs matched."""

    name: str  # memory.<package>.<module>:<name>
    signature: str
    input: str
    match: str  # exact | structure | weak (a single named key)
    summary: str
    reason: str  # the keys or columns that matched


def _read(rel: str) -> str:
    path = os.path.join(_ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"memory: {path} cannot be read ({type(exc).__name__})",
        ) from exc


def _json(rel: str) -> dict:
    return json.loads(_read(rel))


def index(package: str | None = None) -> str:
    """The whole index (``INDEX.md``), or the full section of one package (``"<package>"``,
    ``"memory.<package>"``) or notes topic (``"notes/<topic>"``)."""
    text = _read(INDEX_FILE)
    return _Text(text if package is None else find_section(text, package))


def _item_id(item: Any) -> str:
    if callable(item) and not isinstance(item, str):
        return f"{getattr(item, '__module__', '')}:{getattr(item, '__name__', '')}"
    s = str(item).strip()
    if s.startswith("notes/") or ":" in s:
        return s
    if s.startswith("memory.") and s.count(".") >= 3:
        module, _, name = s.rpartition(".")
        return f"{module}:{name}"
    return s


def _source(path: str, name: str) -> str:
    text = _read(path)
    for node in ast.parse(text).body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "\n".join(text.splitlines()[start - 1 : node.end_lineno])
    raise LookupError(f"memory: {name} is not defined in {path}")


def _summary(record: Any) -> str:
    if not record:
        return "no record yet"
    return ", ".join(f"{k} {record[k]}" for k in sorted(record))


def show(item: Any) -> str:
    """One item in full: a function's source with its docstring, or a note's text. Then its links both ways,
    its status (and, for an item left out of the index, why), its verification and use records, and the
    subjects of the library commits that changed it, newest first."""
    wanted = _item_id(item)
    data = _json(ITEMS_FILE)
    row = data.get("items", {}).get(wanted)
    if row is None:
        raise LookupError(
            f"memory: no item {wanted!r}; memory.index() lists the items (a function is "
            "memory.<package>.<module>:<name>, a note notes/<topic>/<slug>.md)",
        )
    links = _json(LINKS_FILE).get("items", {}).get(wanted, {})
    status = row.get("status", "experimental")
    if row["kind"] == "function":
        out = [f"{row['signature']}  [{status}]"]
    else:
        out = [
            f"{wanted}  [{status}]" + (f"  {row['title']}" if row.get("title") else ""),
        ]
    if status in ("suspect", "deprecated"):
        out.append(
            f"Not in the index: {row.get('status_reason') or 'its status is ' + status}.",
        )
    out.append("")
    if row["kind"] == "function":
        out.append(_source(row["path"], wanted.split(":", 1)[1]))
    else:
        out.append(_read(wanted).rstrip("\n"))
    out += [
        "",
        "Links to: " + (", ".join(links.get("links_to", [])) or "none"),
        "Linked from: " + (", ".join(links.get("linked_from", [])) or "none"),
        "Verification: " + _summary(row.get("verification")),
        "Use: " + _summary(row.get("use")),
    ]
    history = row.get("history") or []
    if history:
        out.append("History (newest first):")
        out += [f"  {h}" for h in history]
        if row.get("history_more"):
            out.append(f"  … and {row['history_more']} earlier change(s)")
    else:
        out.append("History: none recorded")
    if not data.get("history_complete", True):
        out.append(
            f"(history covers the newest {data.get('history_scanned')} library commits only)",
        )
    return _Text("\n".join(out) + "\n")


def _matcher() -> Any:
    """The shape matcher shipped beside the package (``.memory/matcher.py``, v2's own matching code). Its
    ``__file__`` is set at the root, so it loads its shape functions from ``.memory/shapes.py``.
    """
    global _MATCHER
    if _MATCHER is None:
        path = os.path.join(_ROOT, MATCHER_FILE)
        spec = importlib.util.spec_from_loader("_memory_matcher", loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__file__ = os.path.join(_ROOT, "matcher.py")
        with open(path, "rb") as fh:
            code = compile(fh.read(), path, "exec")
        exec(code, module.__dict__)  # noqa: S102 - the harness's own generated file
        _MATCHER = module
    return _MATCHER


def _candidates(m: Any, value: Any, exts: list[str]) -> list[dict]:
    """The descriptors *value* can be seen as (v2's ``memory_helper._value_candidates``, unchanged)."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if (
        isinstance(value, str)
        and value
        and len(value) <= m._MAX_PATH_CHARS
        and "\n" not in value
        and "\0" not in value
        and os.path.isfile(value)
    ):
        with open(value, "rb") as fh:
            data = fh.read(m._shapes.PARSE_LIMIT + 1)
        desc = m.file_shape(value, data)
        return [desc] if desc is not None else []
    out: list[dict] = []
    if isinstance(value, bytearray):
        value = bytes(value)
    data = value.encode("utf-8") if isinstance(value, str) else value
    if isinstance(data, bytes):
        for ext in exts:
            desc = m.file_shape("value" + ext, data[: m._shapes.PARSE_LIMIT + 1])
            if desc is not None:
                out.append(desc)
    desc = m.value_shape(value)
    if desc is not None:
        out.append(desc)
    return out


def find(value: Any) -> list[Found]:
    """The functions whose recorded inputs have the shape of *value*, best match first: v2's input-shape
    matcher and ranking, unchanged (level, then share of named keys matched, then how many recorded shapes
    match, then name; at most five). An empty list means that no listed function was built on data of this
    shape, or that the value has too little structure (plain text has none)."""
    m = _matcher()
    entries = _json(FIND_FILE).get("functions", [])
    exts = sorted(
        {
            d.get("ext", "")
            for e in entries
            for d in e.get("input_shapes", [])
            if d.get("kind") == "file"
        },
    )
    mine = _candidates(m, value, exts)
    ranked: list[tuple[int, float, int, str, Found]] = []
    for e in entries:
        best = None
        count = 0
        for d in e.get("input_shapes", []):
            hits = [x for x in (m.match(v, d) for v in mine) if x is not None]
            if not hits:
                continue
            top = max(hits, key=lambda x: (x[0], x[1]))
            if best is None or top[:2] > best[:2]:
                best, count = top, 1
            elif top[:2] == best[:2]:
                count += 1
        if best is None:
            continue
        level, share, matched = best
        shown = [p[3:] if p.startswith("[].") else p for p in matched]
        reason = (
            "matched "
            + ", ".join(shown[:8])
            + (f" and {len(shown) - 8} more" if len(shown) > 8 else "")
        )
        name = e["id"]
        found = Found(
            name,
            e.get("signature", ""),
            e.get("input", ""),
            _LEVELS[level],
            e.get("summary", ""),
            reason,
        )
        ranked.append((-level, -share, -count, name, found))
    return [f for *_, f in sorted(ranked, key=lambda r: r[:4])][: m.MAX_RESULTS]
