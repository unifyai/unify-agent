"""Deterministic file shapes over recorded bytes (spec §C3): format plus structure, never values.

:func:`shape` reads at most :data:`PARSE_LIMIT` bytes and returns a JSON-able dict:

* ``{"format": "csv", "delimiter": ",", "header": True, "columns": [...], "types": [...], "stats": {...}}``
  for delimited tables. Column types come from the data rows: ``int``, ``float``, ``date`` (``YYYY-MM-DD``),
  ``datetime`` (ISO 8601), ``bool``, ``str``, ``empty`` or ``mixed:<a>+<b>``;
* ``{"format": "json" | "yaml", "tree": ...}``: the key tree (a dict of key -> subtree, a list of the
  distinct element shapes, or a scalar type name), bounded in depth and width;
* ``{"format": "xlsx", "sheets": [{"name": ..., "header": [...]}]}`` when openpyxl imports, else
  ``{"format": "xlsx-unparsed"}``;
* ``{"format": "text", "encoding": ...}`` and ``{"format": "binary"}`` otherwise.

Detection is structural: the file name's extension and the bytes, never the words in them. Counts that
vary from file to file (lines, rows) sit under ``"stats"``; :func:`signature` drops them, so it names the
shape a fingerprint compares. Column names, JSON keys and sheet names are structure; cell values,
JSON leaves and text lines are values and never appear. Imports only the standard library (openpyxl is
optional), so the module runs inside the consolidation sandbox as ``memlab.analysis.shapes``.
"""

from __future__ import annotations

import csv
import io
import json
import posixpath
import re
import zipfile
from typing import Any

PARSE_LIMIT = 1024**2
SAMPLE_ROWS = 1000
MAX_DEPTH = 6
MAX_KEYS = 64
MAX_COLUMNS = 256
DELIMITERS = (",", "\t", ";", "|")
_TABLE_EXT = {".csv": ",", ".tsv": "\t", ".tab": "\t", ".psv": "|"}
_JSON_EXT = {".json", ".jsonl", ".ndjson", ".geojson"}
_YAML_EXT = {".yaml", ".yml"}
_XLSX_EXT = {".xlsx", ".xlsm"}

_INT = re.compile(r"^[+-]?\d+\Z")
_FLOAT = re.compile(r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?\Z")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}\Z")
_DATETIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\Z",
)
_BOOL = {"true", "false", "True", "False", "TRUE", "FALSE"}


def extension(name: str) -> str:
    """The lower-cased extension of *name* (``""`` if none)."""
    return posixpath.splitext(posixpath.basename(name or ""))[1].lower()


def decode(data: bytes, truncated: bool = False) -> tuple[str | None, str]:
    """(encoding, text): BOMs first, then UTF-8, then latin-1; ``None`` encoding for binary (a NUL byte).

    With *truncated* (the bytes were cut at the parse limit), a multi-byte sequence cut at the very end
    still counts as UTF-8.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", data.decode("utf-16", "replace")
    if b"\0" in data:
        return None, ""
    try:
        return "utf-8", data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if truncated and exc.reason == "unexpected end of data":
            return "utf-8", data[: exc.start].decode("utf-8", "replace")
        return "latin-1", data.decode("latin-1")


def value_type(cell: str) -> str:
    """The type a text cell holds: int, float, date, datetime, bool, empty or str."""
    s = cell.strip()
    if not s:
        return "empty"
    if _INT.match(s):
        return "int"
    if _FLOAT.match(s):
        return "float"
    if _DATE.match(s):
        return "date"
    if _DATETIME.match(s):
        return "datetime"
    if s in _BOOL:
        return "bool"
    return "str"


def column_type(types: set[str]) -> str:
    """One column's type from the types of its cells (``empty`` cells are ignored)."""
    t = set(types) - {"empty"}
    if not t:
        return "empty"
    if t == {"int", "float"}:
        return "float"
    if t == {"date", "datetime"}:
        return "datetime"
    if len(t) == 1:
        return next(iter(t))
    return "mixed:" + "+".join(sorted(t))


# --- tables --------------------------------------------------------------------------------------------------


def _rows(text: str, delimiter: str, limit: int) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    out: list[list[str]] = []
    try:
        for row in reader:
            if row == []:
                continue
            out.append(row)
            if len(out) >= limit:
                break
    except csv.Error:
        pass
    return out


def _lines_complete(text: str) -> str:
    """*text* without a last line the parse limit may have cut."""
    return text if text.endswith("\n") else text.rsplit("\n", 1)[0]


def sniff_delimiter(text: str, prefer: str | None = None) -> str | None:
    """The delimiter that splits every sampled row into the same number (≥ 2) of fields.

    Among consistent candidates the widest wins; ties keep *prefer*, then the order of
    :data:`DELIMITERS`. ``None`` when no candidate is consistent over at least two rows.
    """
    best: tuple[int, int] | None = None
    chosen = None
    order = ([prefer] if prefer else []) + [d for d in DELIMITERS if d != prefer]
    for rank, d in enumerate(order):
        rows = _rows(text, d, 50)
        widths = {len(r) for r in rows}
        if len(rows) < 2 or len(widths) != 1:
            continue
        width = widths.pop()
        if width < 2:
            continue
        key = (width, -rank)
        if best is None or key > best:
            best, chosen = key, d
    return chosen


def table_shape(text: str, delimiter: str) -> dict:
    rows = _rows(text, delimiter, SAMPLE_ROWS + 1)
    if not rows:
        return {
            "format": "csv",
            "delimiter": delimiter,
            "header": False,
            "columns": None,
            "types": [],
            "stats": {"rows": 0, "width": 0},
        }
    first = rows[0]
    header = all(value_type(c) == "str" for c in first) and len(
        set(c.strip() for c in first),
    ) == len(first)
    body = rows[1:] if header else rows
    width = max(len(r) for r in rows)
    width = min(width, MAX_COLUMNS)
    types = [
        column_type({value_type(r[i]) if i < len(r) else "empty" for r in body})
        for i in range(width)
    ]
    widths = sorted({len(r) for r in rows})
    out = {
        "format": "csv",
        "delimiter": delimiter,
        "header": header,
        "columns": [c.strip() for c in first[:MAX_COLUMNS]] if header else None,
        "types": types,
        "ragged": len(widths) > 1,
        "stats": {"rows": len(body), "width": width},
    }
    return out


# --- trees ---------------------------------------------------------------------------------------------------


def tree(value: Any, depth: int = 0) -> Any:
    """The key tree of parsed JSON/YAML: keys kept, leaves replaced by type names, lists by element shapes."""
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            return "dict"
        keys = sorted(value, key=str)
        out = {str(k): tree(value[k], depth + 1) for k in keys[:MAX_KEYS]}
        if len(keys) > MAX_KEYS:
            out["…"] = f"+{len(keys) - MAX_KEYS} keys"
        return out
    if isinstance(value, (list, tuple)):
        if depth >= MAX_DEPTH:
            return ["list"]
        seen: list[Any] = []
        for v in value[:SAMPLE_ROWS]:
            t = tree(v, depth + 1)
            if t not in seen:
                seen.append(t)
            if len(seen) >= 4:
                break
        return sorted(seen, key=lambda x: json.dumps(x, sort_keys=True))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _yaml_scalar(text: str) -> str:
    s = text.strip()
    if s.startswith(("'", '"')):
        return "str"
    if s in ("", "~", "null", "Null", "NULL"):
        return "null"
    if s in _BOOL or s in ("yes", "no", "Yes", "No"):
        return "bool"
    if s.startswith("["):
        return ["str"]
    if s.startswith("{"):
        return "dict"
    t = value_type(s)
    return t if t in ("int", "float", "date", "datetime") else "str"


_YAML_LINE = re.compile(
    r"^(?P<key>'[^']*'|\"[^\"]*\"|[^\s#:'\"\-][^:#]*?|-[^\s:#][^:#]*?):(?:\s+(?P<val>.*))?$",
)


def yaml_tree(text: str) -> Any:
    """A key tree from YAML text: PyYAML's ``safe_load`` when it imports, else :func:`yaml_tree_basic`."""
    try:
        import yaml  # optional
    except Exception:
        return yaml_tree_basic(text)
    try:
        return tree(yaml.safe_load(text))
    except Exception:
        return yaml_tree_basic(text)


def yaml_tree_basic(text: str) -> Any:
    """A key tree by indentation alone (deterministic, standard library only).

    ``key:`` opens a mapping, ``key: value`` is a leaf typed by :func:`_yaml_scalar`; a block of ``- ``
    items becomes ``["list"]``. Comments, document markers and blank lines are skipped.
    """
    root: dict = {}
    # (indent, mapping, holder mapping, key in holder)
    stack: list[tuple[int, dict, dict | None, str | None]] = [(-1, root, None, None)]
    for raw in text.splitlines():
        body = raw.strip()
        if not body or body.startswith("#") or body in ("---", "..."):
            continue
        ind = len(raw) - len(raw.lstrip(" "))
        if body == "-" or body.startswith("- "):
            # a list item: the innermost open key (at this indent or shallower) holds a list
            while len(stack) > 1 and stack[-1][0] > ind:
                stack.pop()
            _, mapping, holder, key = stack[-1]
            if holder is not None and key is not None and not mapping:
                holder[key] = ["list"]
            continue
        m = _YAML_LINE.match(body.split(" #", 1)[0].rstrip())
        if m is None:
            continue
        while len(stack) > 1 and stack[-1][0] >= ind:
            stack.pop()
        parent = stack[-1][1]
        key = m.group("key").strip().strip("'\"")
        val = m.group("val")
        if val is None or not val.strip():
            child: dict = {}
            parent[key] = child
            stack.append((ind, child, parent, key))
        else:
            parent[key] = _yaml_scalar(val)
    return _yaml_finish(root)


def _yaml_finish(node: Any, depth: int = 0) -> Any:
    if isinstance(node, dict):
        if depth >= MAX_DEPTH:
            return "dict"
        if not node:
            return "null"
        keys = sorted(node)
        out = {k: _yaml_finish(node[k], depth + 1) for k in keys[:MAX_KEYS]}
        if len(keys) > MAX_KEYS:
            out["…"] = f"+{len(keys) - MAX_KEYS} keys"
        return out
    return node


# --- spreadsheets --------------------------------------------------------------------------------------------


def xlsx_shape(data: bytes) -> dict:
    try:
        import openpyxl  # optional
    except Exception:
        return {"format": "xlsx-unparsed"}
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception:
        return {"format": "xlsx-unparsed"}
    sheets = []
    try:
        for ws in wb.worksheets[:MAX_KEYS]:
            header: list[str] | None = None
            for row in ws.iter_rows(min_row=1, max_row=1, values_only=True):
                cells = list(row)[:MAX_COLUMNS]
                if cells and all(isinstance(c, str) and c.strip() for c in cells):
                    header = [c.strip() for c in cells]
                break
            sheets.append({"name": ws.title, "header": header})
    finally:
        try:
            wb.close()
        except Exception:
            pass
    return {"format": "xlsx", "sheets": sheets}


# --- the entry points ----------------------------------------------------------------------------------------


def _json_shape(text: str, ext: str) -> dict | None:
    s = text.strip()
    if not s:
        return None
    try:
        return {"format": "json", "tree": tree(json.loads(s))}
    except (ValueError, RecursionError):
        pass
    if ext in (".jsonl", ".ndjson") or (s.startswith("{") and "\n{" in s):
        shapes: list[Any] = []
        n = 0
        for ln in _lines_complete(text).splitlines():
            if not ln.strip():
                continue
            try:
                t = tree(json.loads(ln))
            except (ValueError, RecursionError):
                return None
            n += 1
            if t not in shapes and len(shapes) < 4:
                shapes.append(t)
        if n:
            return {
                "format": "jsonl",
                "tree": sorted(shapes, key=lambda x: json.dumps(x, sort_keys=True)),
                "stats": {"rows": n},
            }
    return None


def shape(name: str, data: bytes) -> dict:
    """The shape of a file called *name* holding *data* (see the module docstring)."""
    ext = extension(name)
    lines = data.count(b"\n") + (0 if not data or data.endswith(b"\n") else 1)
    head = data[:PARSE_LIMIT]
    if ext in _XLSX_EXT or (head.startswith(b"PK\x03\x04") and _is_xlsx(data)):
        return xlsx_shape(data)
    encoding, text = decode(head, truncated=len(data) > PARSE_LIMIT)
    if encoding is None:
        return {"format": "binary", "stats": {"bytes": len(data)}}
    if len(data) > PARSE_LIMIT:
        text = _lines_complete(text)
    stats = {"lines": lines, "bytes": len(data)}
    if ext in _JSON_EXT or (ext not in _TABLE_EXT and text.lstrip()[:1] in ("{", "[")):
        out = _json_shape(text, ext)
        if out is not None:
            out.setdefault("stats", {}).update(stats)
            out["encoding"] = encoding
            return out
    if ext in _YAML_EXT:
        return {
            "format": "yaml",
            "encoding": encoding,
            "tree": yaml_tree(text),
            "stats": stats,
        }
    if ext in _TABLE_EXT or ext in ("", ".txt", ".dat"):
        d = sniff_delimiter(text, _TABLE_EXT.get(ext))
        if d is None and ext in _TABLE_EXT:
            d = _TABLE_EXT[ext] if len(_rows(text, _TABLE_EXT[ext], 2)) else None
        if d is not None:
            out = table_shape(text, d)
            out["encoding"] = encoding
            out["stats"].update(stats)
            return out
    return {"format": "text", "encoding": encoding, "stats": stats}


def _is_xlsx(data: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return "xl/workbook.xml" in z.namelist()
    except (zipfile.BadZipFile, ValueError, OSError):
        return False


def signature(s: dict) -> str:
    """A canonical string for a shape without its ``stats`` (row and line counts vary per file)."""
    return json.dumps(
        {k: v for k, v in s.items() if k != "stats"},
        sort_keys=True,
        ensure_ascii=False,
    )


def conforms(s: dict, expected: dict) -> list[str]:
    """Where shape *s* differs from *expected* on the keys *expected* names (``stats`` ignored).

    An empty list means *s* agrees on every structural field *expected* states, so a reader can say
    ``conforms(shape(p, b), {"format": "csv", "delimiter": ",", "columns": [...]})`` as its schema check.
    """
    problems = []
    for k, v in expected.items():
        if k == "stats":
            continue
        if s.get(k) != v:
            problems.append(f"{k}: expected {v!r}, found {s.get(k)!r}"[:300])
    return problems
