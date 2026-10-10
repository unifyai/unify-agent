"""File shapes (spec §C3) for the work-tree adapter, computed in a bounded child process.

The harness never parses a work-tree file itself: :func:`worktree.file_shape` runs this file as a
script (``python -I -S -B worktree_shapes.py <format> <truncated> [<library dir> ...]``) under
``prlimit``, with an empty environment, the file's (redacted) bytes on stdin and no path, and reads the
shape back as JSON on stdout. The parsers here are therefore free to be slow or greedy on hostile input (PyYAML's base-60 ints
are quadratic; a zip can lie about its sizes): the child's wall-clock timeout and rlimits bound them. They
still defend themselves where that is cheap and exact (the CSV delimiter vote is linear, YAML merge keys
are never expanded, every ``.xlsx`` entry is inflated in counted chunks and only stored or deflated
entries are accepted).

The child is not trusted: the harness checks every answer against the shape schema here
(:func:`parse_answer`, :func:`valid_shape`) before it can enter a record, and then redacts every string
in it. Nothing is cut here before that redaction, since a name cut first could keep the head of a secret
that only the whole name shows (a secret written with escapes is invisible to the byte scan before the
child). Key and column names are sent whole up to ``MAX_NAME_SENT`` characters, and the harness cuts them
to ``MAX_NAME`` only after redacting them (:func:`cut_names`); a longer name is sent as ``<name i: n
chars>``, its position and length only. Cell values of an ``.xlsx`` (sheet titles and first-row cells) are
sent whole up to ``MAX_NAME`` characters, a longer one as ``{"chars": n}``.

The module imports nothing outside the standard library at import time, and nothing from ``unify``, so the
child starts without the package and the harness can import it for the constants, the schema and the
plain-text shape.
"""

from __future__ import annotations

import collections
import csv
import io
import itertools
import json
import re
import struct
import sys
import zipfile
import zlib
from typing import Any, Iterable

PARSE_CAP = 1024 * 1024  # bytes parsed for a shape
TYPE_ROWS = 200  # rows inspected for CSV column types
SNIFF_BYTES = 8192  # prefix of a CSV used to choose its delimiter
SNIFF_ROWS = 20  # rows of that prefix compared
MAX_COLUMNS = 256  # columns kept in a tabular shape
MAX_NAME = 128  # characters recorded of a column or key name (cut by the harness after redaction)
MAX_NAME_SENT = 512  # characters of a name the child sends whole; a longer one is sent as its length
TREE_DEPTH = 3  # JSON / YAML key-tree depth
YAML_DEPTH = 1000  # YAML nesting allowed before libyaml's composer, which recurses on the C stack, runs
MAX_KEYS = 64  # keys kept per object level
SHAPE_NODES = 512  # keys and columns named per shape (an alias counts per use)
SHAPE_CHARS = 40 * 1024  # JSON-escaped characters of those names per shape
SHAPE_BYTES = 96 * 1024  # serialised size of a recorded shape; refused over it
ANSWER_BYTES = 1024 * 1024  # serialised size of the child's answer; refused over it
XLSX_ENTRIES = 4096  # zip entries of an .xlsx opened at all
XLSX_INFLATED = 16 * 1024 * 1024  # bytes an .xlsx may inflate to, declared or actual
INFLATE_CHUNK = 64 * 1024  # bytes inflated per step, each counted before the next
STDIN_CAP = XLSX_INFLATED  # bytes the child reads; the harness never sends more
LIMIT_EXIT = 3  # the child's exit status after a MemoryError
ANSWER_DEPTH = 16  # JSON nesting of a child's answer (an honest one: 9 at most)
ANSWER_COUNT = 1 << 53  # the largest count an answer may hold

FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xlsx": "xlsx",
}
# The only packages outside the standard library the child may import: the parsers, openpyxl's XML
# writer dependency, and defusedxml (which openpyxl uses, when present, to refuse entity tricks). Anything
# else openpyxl would pick up when present is refused: numpy (whose OpenBLAS reserves hundreds of MiB of
# address space for its thread pool), PIL, pandas, and lxml (another XML parser with its own surface).
LIBRARIES = frozenset({"yaml", "_yaml", "openpyxl", "et_xmlfile", "defusedxml"})
_SAFE_METHODS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
_DELIMITERS = (",", "\t", ";", "|")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_INT = re.compile(r"^[+-]?\d+$")


class _Budget:
    """What one shape may still describe: ``SHAPE_NODES`` names and ``SHAPE_CHARS`` of them, counted as
    recorded (at most ``MAX_NAME`` characters each)."""

    def __init__(self) -> None:
        self.nodes = SHAPE_NODES
        self.chars = SHAPE_CHARS
        self.spent = False

    def take(self, name: str) -> bool:
        cost = len(json.dumps(name[:MAX_NAME]))
        if self.nodes <= 0 or cost > self.chars:
            self.spent = True
            return False
        self.nodes -= 1
        self.chars -= cost
        return True


def shape_of(fmt: str, data: bytes, truncated: bool) -> dict:
    """The shape of *data* (at most ``PARSE_CAP`` bytes; an ``.xlsx`` whole) read as *fmt*.

    Any parse failure (including ``RecursionError`` and ``csv.Error``) gives ``{"format", "error":
    "unparsed"}``; a ``MemoryError`` propagates (the child then exits with ``LIMIT_EXIT``). A shape names
    at most ``SHAPE_NODES`` keys or columns in ``SHAPE_CHARS`` characters, the rest counted and the shape
    marked ``shape_truncated``; a shape still over ``ANSWER_BYTES`` serialised is refused as ``{"format",
    "error": "shape_too_large", "bytes"}``.
    """
    try:
        budget = _Budget()
        shape = _shape(fmt, data, truncated, budget)
        if budget.spent:
            shape["shape_truncated"] = True
        size = len(json.dumps(shape))
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001 - a hostile file gives a marker, never a crash
        return {"format": fmt, "error": "unparsed"}
    if size > ANSWER_BYTES:
        return {
            "format": shape.get("format", fmt),
            "error": "shape_too_large",
            "bytes": size,
        }
    return shape


def _shape(fmt: str, data: bytes, truncated: bool, budget: _Budget) -> dict:
    if fmt in ("csv", "tsv"):
        return _csv_shape(data, "\t" if fmt == "tsv" else None, truncated, budget)
    if fmt == "json":
        return _json_shape(data, truncated, budget)
    if fmt == "jsonl":
        return _jsonl_shape(data, budget)
    if fmt == "yaml":
        return _yaml_shape(data, truncated, budget)
    if fmt == "xlsx":
        return _xlsx_shape(data, budget)
    text, encoding = _decode(data)
    if text is None:
        return {"format": "binary"}
    if looks_like_json(text, truncated):
        shaped = _json_shape(data, truncated, budget)
        if "error" not in shaped:
            return shaped
    return _text_shape(text, encoding, truncated)


def looks_like_json(text: str, truncated: bool) -> bool:
    """Whether a text file is worth parsing as JSON (only these go to the child; plain text does not)."""
    return text.lstrip()[:1] in ("{", "[") and not truncated


def _decode(data: bytes) -> tuple[str | None, str]:
    if b"\0" in data[:8192]:
        return None, "binary"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError as exc:
        # a multi-byte character cut by the parse cap is still UTF-8
        if exc.start >= len(data) - 3 and exc.reason == "unexpected end of data":
            return data[: exc.start].decode("utf-8"), "utf-8"
    return data.decode("latin-1"), "latin-1"


def _text_shape(text: str, encoding: str, truncated: bool) -> dict:
    shape: dict[str, Any] = {
        "format": "text",
        "encoding": encoding,
        "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
    }
    if truncated:
        shape["truncated"] = True
    return shape


def plain_text_shape(data: bytes, truncated: bool) -> dict | None:
    """The shape of a text or binary file without parsing it, or None when it should be parsed as JSON.

    Linear (a decode and a newline count), so the harness computes it itself without a child.
    """
    text, encoding = _decode(data)
    if text is None:
        return {"format": "binary"}
    if looks_like_json(text, truncated):
        return None
    return _text_shape(text, encoding, truncated)


def _cell_type(value: str) -> str:
    v = value.strip()
    if not v:
        return "empty"
    if _INT.match(v):
        return "int"
    try:
        float(v)
        return "float"
    except ValueError:
        pass
    if _DATE.match(v):
        return "date"
    return "str"


def _merge_types(types: list[str]) -> str:
    seen = {t for t in types if t != "empty"}
    if not seen:
        return "empty"
    if seen == {"int"}:
        return "int"
    if seen <= {"int", "float"}:
        return "float"
    if seen == {"date"}:
        return "date"
    return "str"


# Keys that hold data rather than name it: a YAML !!omap / !!pairs key may be a list or a mapping (an
# alias makes it any size), and a !!binary key is bytes. Such a key is named by its type only.
_KEY_TYPES = (
    (dict, "object"),
    ((list, tuple), "array"),
    ((set, frozenset), "set"),
    ((bytes, bytearray), "bytes"),
)


def _name(s: Any, i: int) -> tuple[str, bool]:
    """Name *s* (the *i*-th key or column) whole, or ``<name i: n chars>`` beyond ``MAX_NAME_SENT``.

    A key that is a collection or bytes is ``<key i: type>`` (a ``TYPE_NAMES`` type), never ``str()``-ed,
    so its contents are never a name and an alias-expanded key costs one key of the budget. Never cut
    here: the harness redacts the whole name before it cuts it (:func:`cut_names`).
    """
    for types, kind in _KEY_TYPES:
        if isinstance(s, types):
            return f"<key {i}: {kind}>", False
    s = str(s)
    return (
        (s, False) if len(s) <= MAX_NAME_SENT else (f"<name {i}: {len(s)} chars>", True)
    )


def _delimiter(text: str) -> str:
    """The candidate delimiter that splits the most of the first rows into the same number (>1) of
    fields; ties keep the earlier candidate, and ``,`` when none splits a row.

    Linear in a fixed prefix: ``csv.Sniffer``'s quote regex rescans to the end of the sample from every
    match, which is quadratic on hostile input."""
    sample = text[:SNIFF_BYTES]
    if len(text) > SNIFF_BYTES and "\n" in sample:
        sample = sample[: sample.rfind("\n") + 1]  # drop the row the prefix cut
    best, best_score = ",", (0, 0)
    for delim in _DELIMITERS:
        try:
            reader = csv.reader(io.StringIO(sample), delimiter=delim)
            widths = [len(r) for r in _take(reader, SNIFF_ROWS) if r]
        except csv.Error:
            continue
        if not widths:
            continue
        width, rows = collections.Counter(widths).most_common(1)[0]
        if width > 1 and (rows, width) > best_score:
            best, best_score = delim, (rows, width)
    return best


def _column_type(rows: list[list[str]], i: int) -> str:
    return _merge_types([_cell_type(r[i]) if i < len(r) else "empty" for r in rows])


def _csv_shape(
    data: bytes,
    delimiter: str | None,
    truncated: bool,
    budget: _Budget | None = None,
) -> dict:
    budget = budget if budget is not None else _Budget()
    text, encoding = _decode(data)
    if text is None:
        return {"format": "csv", "error": "binary"}
    if truncated and "\n" in text:
        text = text[: text.rfind("\n") + 1]  # drop a row cut by the parse cap
    # process-wide (the child's); restored below
    old_limit = csv.field_size_limit(PARSE_CAP + 1)
    try:
        delim = delimiter or _delimiter(text)
        rows = list(
            _take(csv.reader(io.StringIO(text), delimiter=delim), TYPE_ROWS + 1),
        )
    finally:
        csv.field_size_limit(old_limit)
    width = max((len(r) for r in rows), default=0)
    kept = min(width, MAX_COLUMNS)  # only kept columns are typed: a wide row is cheap
    first = rows[0] if rows else []
    body_types = [_column_type(rows[1:], i) for i in range(kept)]
    # A header needs distinct non-empty non-numeric cells, and evidence that it differs from the body
    # (some body column is not plain strings); otherwise the first row is data and the columns are
    # positional, so no data row ever becomes "column names".
    has_header = (
        bool(first)
        and len(set(first)) == len(first)
        and all(_cell_type(c) == "str" for c in first)
        and any(t not in ("str", "empty") for t in body_types)
    )
    names_truncated = False
    if has_header:
        total = len(first)
        columns = []
        for i, cell in enumerate(first[:MAX_COLUMNS]):
            name, cut = _name(cell, i)
            if not budget.take(name):
                break
            columns.append(name)
            names_truncated = names_truncated or cut
        types = body_types[: len(columns)]
    else:
        total = width
        columns = [f"c{i}" for i in range(kept)]
        types = [_column_type(rows[:TYPE_ROWS], i) for i in range(kept)]
    shape: dict[str, Any] = {
        "format": "tsv" if delim == "\t" else "csv",
        "delimiter": delim,
        "header": has_header,
        "columns": columns,
        "types": types,
        "encoding": encoding,
    }
    if total > len(columns):
        shape["more_columns"] = total - len(columns)
    if names_truncated:
        shape["names_truncated"] = True
    if truncated:
        shape["truncated"] = True
    return shape


def _take(it: Iterable, n: int) -> Iterable:
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


def _pairs(obj: Any) -> Any:
    """The ``(key, value)`` pairs of a mapping, or None.

    YAML's ``!!omap`` and ``!!pairs`` are mappings too: the safe loaders build them as a list of
    2-tuples, and nothing else they (or ``json``) build holds a tuple, so the first item tells.
    """
    if isinstance(obj, dict):
        return obj.items()
    if isinstance(obj, list) and obj and type(obj[0]) is tuple:
        return obj
    return None


def _tree(obj: Any, depth: int, budget: _Budget) -> Any:
    pairs = _pairs(obj)
    if pairs is not None:
        if depth <= 0:
            return {"type": "object"}
        total = len(pairs)
        out: dict[str, Any] = {"type": "object", "keys": {}}
        for i, (k, v) in enumerate(itertools.islice(pairs, MAX_KEYS)):
            name, cut = _name(k, i)
            if name in out["keys"]:
                continue  # a key repeated in a !!pairs: the first is kept
            if not budget.take(name):
                out["more_keys"] = total - i
                return out
            if cut:
                out["names_truncated"] = True
            out["keys"][name] = _tree(v, depth - 1, budget)
        if total > MAX_KEYS:
            out["more_keys"] = total - MAX_KEYS
        return out
    if isinstance(obj, list):
        out = {"type": "array"}
        if obj and depth > 0:
            out["items"] = _tree(obj[0], depth - 1, budget)
        return out
    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "bool"
    if isinstance(obj, int):
        return "int"
    if isinstance(obj, float):
        return "float"
    if isinstance(obj, str):
        return "date" if _DATE.match(obj) else "str"
    return type(obj).__name__


def _json_shape(data: bytes, truncated: bool, budget: _Budget) -> dict:
    if truncated:
        return {"format": "json", "truncated": True}
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return {"format": "json", "error": "unparsed"}
    tree = _tree(obj, TREE_DEPTH, budget)
    return {"format": "json", **(tree if isinstance(tree, dict) else {"type": tree})}


def _jsonl_shape(data: bytes, budget: _Budget) -> dict:
    text, _ = _decode(data)
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    shape: dict[str, Any] = {"format": "jsonl", "records": len(lines)}
    if lines:
        try:
            shape["record"] = _tree(json.loads(lines[0]), TREE_DEPTH, budget)
        except (ValueError, RecursionError):
            shape["error"] = "unparsed"
    return shape


def _yaml_loader(yaml: Any) -> type:
    """``CSafeLoader`` (``SafeLoader`` without libyaml) without merge keys: ``<<`` (and an explicit
    ``!!merge``) stays a plain key.

    A merge copies the merged mappings' key pairs, and nested aliases make that exponential (nine
    levels of nine merges are 9**9 pairs from 600 bytes). ``flatten_mapping`` belongs to the Python
    ``SafeConstructor``, which both loaders use, and the tags come from the same Python resolver, so the
    override holds for the C loader too. The C loader parses about 5 times faster; the constructor
    (base-60 ints included) is Python in both."""

    class Loader(getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader):
        def flatten_mapping(self, node: Any) -> None:
            for key_node, _ in node.value:
                if key_node.tag in (
                    "tag:yaml.org,2002:merge",
                    "tag:yaml.org,2002:value",
                ):
                    key_node.tag = "tag:yaml.org,2002:str"

    return Loader


def _yaml_shape(data: bytes, truncated: bool, budget: _Budget) -> dict:
    try:
        import yaml
    except ImportError:
        return {"format": "yaml"}
    if truncated:
        return {"format": "yaml", "truncated": True}
    loader = _yaml_loader(yaml)
    try:
        text = data.decode("utf-8")
        if not issubclass(loader, yaml.SafeLoader) and not _yaml_nesting_within(
            yaml,
            text,
            loader,
        ):
            return {"format": "yaml", "error": "unparsed"}
        obj = yaml.load(text, Loader=loader)
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001 - any parse failure is just an unparsed shape
        return {"format": "yaml", "error": "unparsed"}
    # Aliases load as shared objects, so the walk of the shared graph is bounded by the budget. Scalar
    # construction is not linear (base-60 ints): the child's timeout bounds it.
    tree = _tree(obj, TREE_DEPTH, budget)
    return {"format": "yaml", **(tree if isinstance(tree, dict) else {"type": tree})}


def _yaml_nesting_within(yaml: Any, text: str, loader: type) -> bool:
    """Whether *text* nests at most ``YAML_DEPTH`` collections deep, from libyaml's events alone.

    libyaml's parser keeps its state on the heap, but the C loader's composer recurses on the C stack
    once per level, so ``- - - …`` (two bytes a level) crashed it at a few tens of thousands of levels.
    The pure-Python loader raises ``RecursionError`` instead, so it needs no such pass.
    """
    depth = 0
    for event in yaml.parse(text, Loader=loader):
        if isinstance(event, (yaml.SequenceStartEvent, yaml.MappingStartEvent)):
            depth += 1
            if depth > YAML_DEPTH:
                return False
        elif isinstance(event, (yaml.SequenceEndEvent, yaml.MappingEndEvent)):
            depth -= 1
    return True


class _Refused(Exception):
    """An ``.xlsx`` that is not opened; the message is the shape's ``error``."""


def _inflate(data: bytes, info: zipfile.ZipInfo, cap: int) -> bytes:
    """The bytes of entry *info*, inflated from *data* in counted ``INFLATE_CHUNK`` steps.

    Nothing the zip declares is trusted as a bound: the entry is refused as soon as it produces more than
    its declared size (``size_mismatch``) or than *cap* (``inflated_too_large``), and it must end at
    exactly its declared size and CRC. ``zipfile``'s own reader is not used, because its whole-entry
    ``read()`` runs one ``decompress`` over all the compressed bytes before truncating, and passes no
    output bound at all for bzip2 and LZMA.
    """
    off = info.header_offset
    head = data[off : off + 30]
    if len(head) != 30 or head[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile("bad local header")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    start = off + 30 + name_len + extra_len
    raw = data[start : start + info.compress_size]
    if len(raw) != info.compress_size:
        raise zipfile.BadZipFile("truncated entry")
    limit = min(info.file_size, cap)

    def check(total: int) -> None:
        if total > info.file_size:
            raise _Refused("size_mismatch")
        if total > cap:
            raise _Refused("inflated_too_large")

    if info.compress_type == zipfile.ZIP_STORED:
        check(len(raw))
        content = raw
    else:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        chunks: list[bytes] = []
        total, pending = 0, raw
        while not inflater.eof:
            piece = inflater.decompress(pending, min(INFLATE_CHUNK, limit + 1))
            total += len(piece)
            check(total)
            chunks.append(piece)
            if not piece and len(inflater.unconsumed_tail) == len(pending):
                break  # no progress: the stream ended early
            pending = inflater.unconsumed_tail
        if not inflater.eof:
            raise zipfile.BadZipFile("truncated deflate stream")
        content = b"".join(chunks)
    if len(content) != info.file_size:
        raise _Refused("size_mismatch")
    if zlib.crc32(content) != info.CRC:
        raise zipfile.BadZipFile("bad CRC")
    return content


def _xlsx_stored(data: bytes) -> bytes:
    """*data* rebuilt as a zip of stored entries, each inflated here within ``XLSX_INFLATED`` in all.

    Refuses more than ``XLSX_ENTRIES`` entries, an entry compressed other than stored or deflated (bzip2
    and LZMA are inflated unbounded by ``zipfile``), an encrypted entry, more than ``XLSX_INFLATED``
    declared bytes (a cheap early refusal), and any entry whose actual bytes differ from its declared
    size. openpyxl then reads only stored entries, whose sizes are the bytes written here.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
    if len(infos) > XLSX_ENTRIES:
        raise _Refused("too_many_entries")
    for info in infos:
        if info.compress_type not in _SAFE_METHODS:
            raise _Refused("unsafe_compression")
        if info.flag_bits & 0x1:
            raise _Refused("encrypted")
    if sum(i.file_size for i in infos) > XLSX_INFLATED:
        raise _Refused("inflated_too_large")
    left = XLSX_INFLATED
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as rebuilt:
        for info in infos:
            content = _inflate(data, info, left)
            left -= len(content)
            rebuilt.writestr(zipfile.ZipInfo(info.filename), content)
    return out.getvalue()


def _whole(value: Any) -> str | dict:
    """A cell value of an ``.xlsx`` (a sheet title or a first-row cell) whole, or ``{"chars": n}`` when it
    is longer than ``MAX_NAME``: never cut, because the harness redacts it only after this returns.
    """
    s = str(value)
    return s if len(s) <= MAX_NAME else {"chars": len(s)}


def _xlsx_shape(data: bytes, budget: _Budget) -> dict:
    try:
        stored = _xlsx_stored(data)
    except _Refused as exc:
        return {"format": "xlsx", "error": str(exc)}
    except (
        zipfile.BadZipFile,
        OSError,
        ValueError,
        EOFError,
        struct.error,
        zlib.error,
    ):
        return {"format": "xlsx", "error": "unparsed"}
    try:
        import openpyxl
    except ImportError:
        return {"format": "xlsx"}
    try:
        wb = openpyxl.load_workbook(io.BytesIO(stored), read_only=True, data_only=True)
    except MemoryError:
        raise
    except Exception:  # noqa: BLE001
        return {"format": "xlsx", "error": "unparsed"}
    sheets = []
    try:
        for ws in wb.worksheets[:MAX_KEYS]:
            title = _whole(ws.title)
            if not budget.take(title if isinstance(title, str) else ""):
                break
            first = next(ws.iter_rows(max_row=1, values_only=True), ())
            headers: list[Any] = []
            for v in first[:MAX_COLUMNS]:
                name = None if v is None else _whole(v)
                if name is not None and not budget.take(
                    name if isinstance(name, str) else "",
                ):
                    break
                headers.append(name)
            sheets.append({"name": title, "headers": headers})
    finally:
        wb.close()
    shape: dict[str, Any] = {"format": "xlsx", "sheets": sheets}
    if len(wb.worksheets) > len(sheets):
        shape["more_sheets"] = len(wb.worksheets) - len(sheets)
    return shape


# -- the shape schema: checked by the harness on every answer ------------------------------------------
OUTPUT_FORMATS = {  # the format asked for -> the formats its shape may report
    "csv": frozenset({"csv", "tsv"}),
    "tsv": frozenset({"tsv", "csv"}),
    "json": frozenset({"json"}),
    "jsonl": frozenset({"jsonl"}),
    "yaml": frozenset({"yaml"}),
    "xlsx": frozenset({"xlsx"}),
    "text": frozenset({"text", "json", "binary"}),
}
ERRORS = frozenset(
    {
        "unparsed",
        "binary",
        "budget",
        "shape_too_large",
        "too_large",
        "inflated_too_large",
        "too_many_entries",
        "unsafe_compression",
        "encrypted",
        "size_mismatch",
    },
)
UNPARSED = frozenset({"timeout", "limit", "error", "invalid"})  # the harness's markers
TYPE_NAMES = frozenset(  # what _tree names a value (YAML adds dates, timestamps, binary and sets)
    {
        "object",
        "array",
        "null",
        "bool",
        "int",
        "float",
        "str",
        "date",
        "datetime",
        "bytes",
        "set",
    },
)
CELL_TYPES = frozenset({"empty", "int", "float", "date", "str"})
ENCODINGS = frozenset({"utf-8", "latin-1"})
_NODE_KEYS = frozenset({"type", "keys", "more_keys", "names_truncated", "items"})
_ESCAPE = re.compile(r"\\.", re.S)
_STRING = re.compile(r'"[^"]*"')
_BRACKET = re.compile(r"[\[\]{}]")


def _refuse_constant(name: str) -> None:
    raise ValueError(f"{name} is not JSON")


def _nesting_within(text: str, limit: int) -> bool:
    """Whether the JSON *text* nests at most *limit* deep: linear, and never recursive.

    Escapes are dropped first, then whole strings, so only brackets outside strings are counted (an
    unterminated string leaves its brackets counted, which only rejects what JSON rejects anyway).
    """
    depth = 0
    for b in _BRACKET.findall(_STRING.sub("", _ESCAPE.sub("", text))):
        if b in "[{":
            depth += 1
            if depth > limit:
                return False
        else:
            depth -= 1
    return True


def parse_answer(out: bytes) -> Any:
    """The child's stdout as JSON, or None: strict UTF-8, at most ``ANSWER_DEPTH`` deep (checked before
    parsing, so the parser cannot recurse out of bounds), and no ``NaN`` or ``Infinity``.
    """
    try:
        text = out.decode("utf-8")
        if not _nesting_within(text, ANSWER_DEPTH):
            return None
        return json.loads(text, parse_constant=_refuse_constant)
    except (ValueError, RecursionError):  # UnicodeDecodeError is a ValueError
        return None


def _count(v: Any) -> bool:
    return type(v) is int and 0 <= v <= ANSWER_COUNT


def _flag(v: Any) -> bool:
    return type(v) is bool


def _name_ok(v: Any, most: int = MAX_NAME_SENT) -> bool:
    if type(v) is not str or len(v) > most:
        return False
    try:
        v.encode("utf-8")  # no lone surrogates
    except UnicodeEncodeError:
        return False
    return True


def _one_of(options: frozenset) -> Any:
    return lambda v: type(v) is str and v in options


def _list_of(item: Any, most: int) -> Any:
    return lambda v: type(v) is list and len(v) <= most and all(item(x) for x in v)


def _cell_ok(v: Any) -> bool:
    if type(v) is dict:
        return v.keys() == {"chars"} and _count(v["chars"])
    return _name_ok(v, MAX_NAME)


def _sheet_ok(v: Any) -> bool:
    return (
        type(v) is dict
        and v.keys() == {"name", "headers"}
        and _cell_ok(v["name"])
        and _list_of(lambda h: h is None or _cell_ok(h), MAX_COLUMNS)(v["headers"])
    )


def _keys_ok(v: Any, depth: int) -> bool:
    return (
        type(v) is dict
        and len(v) <= MAX_KEYS
        and all(_name_ok(k) and _node_ok(x, depth) for k, x in v.items())
    )


def _node_ok(v: Any, depth: int) -> bool:
    """A value of ``_tree(obj, depth)``: a type name, or an object or array node."""
    if type(v) is str:
        return v in TYPE_NAMES
    if type(v) is not dict or not v.keys() <= _NODE_KEYS:
        return False
    if v.get("type") not in ("object", "array"):
        return False
    if ("keys" in v or "items" in v) and depth <= 0:
        return False
    return (
        ("keys" not in v or _keys_ok(v["keys"], depth - 1))
        and ("items" not in v or _node_ok(v["items"], depth - 1))
        and _count(v.get("more_keys", 0))
        and _flag(v.get("names_truncated", False))
    )


_COMMON = {
    "format": lambda v: True,  # checked against the format asked for
    "error": _one_of(ERRORS),
    "unparsed": _one_of(UNPARSED),
    "truncated": _flag,
    "shape_truncated": _flag,
    "bytes": _count,
}
_TREE = {
    "type": _one_of(TYPE_NAMES),
    "keys": lambda v: _keys_ok(v, TREE_DEPTH - 1),
    "items": lambda v: _node_ok(v, TREE_DEPTH - 1),
    "more_keys": _count,
    "names_truncated": _flag,
}
_FIELDS = {
    "csv": {
        "delimiter": _one_of(frozenset(_DELIMITERS)),
        "header": _flag,
        "columns": _list_of(_name_ok, MAX_COLUMNS),
        "types": _list_of(_one_of(CELL_TYPES), MAX_COLUMNS),
        "encoding": _one_of(ENCODINGS),
        "more_columns": _count,
        "names_truncated": _flag,
    },
    "json": _TREE,
    "yaml": _TREE,
    "jsonl": {"records": _count, "record": lambda v: _node_ok(v, TREE_DEPTH)},
    "xlsx": {"sheets": _list_of(_sheet_ok, MAX_KEYS), "more_sheets": _count},
    "text": {"encoding": _one_of(ENCODINGS), "lines": _count},
    "binary": {},
}
_FIELDS["tsv"] = _FIELDS["csv"]


def valid_shape(fmt: str, shape: Any) -> bool:
    """Whether *shape* is a shape a file asked to be read as *fmt* can have: only the keys of its
    format, each of its type, names of at most ``MAX_NAME_SENT`` characters and ``.xlsx`` values of at
    most ``MAX_NAME`` (type names, errors and the like from fixed sets), counts that are non-negative ints, and lists and objects within the caps the
    child applies. Iterative in the answer's size and recursive only along the fixed ``TREE_DEPTH``.
    """
    if type(shape) is not dict:
        return False
    reported = shape.get("format")
    if type(reported) is not str or reported not in OUTPUT_FORMATS.get(fmt, ()):
        return False
    fields = {**_COMMON, **_FIELDS[reported]}
    return all(k in fields and fields[k](v) for k, v in shape.items())


def cut_names(shape: dict) -> dict:
    """*shape* (valid and already redacted) with every key and column name cut to ``MAX_NAME``.

    The harness calls this only after redacting the whole names, so a cut never keeps the head of a
    secret the whole name showed. A cut name is marked ``names_truncated`` on its node; two keys of a node
    that cut to the same name keep the first, the other counted in ``more_keys``.
    """
    shape = _cut_node(shape)
    if isinstance(shape.get("record"), dict):
        shape["record"] = _cut_node(shape["record"])
    columns = shape.get("columns")
    if isinstance(columns, list) and any(len(c) > MAX_NAME for c in columns):
        shape["columns"] = [c[:MAX_NAME] for c in columns]
        shape["names_truncated"] = True
    return shape


def _cut_node(node: Any) -> Any:
    """A tree node with its key names cut (recursive along the ``TREE_DEPTH`` the schema checked)."""
    if type(node) is not dict:
        return node
    node = dict(node)
    if isinstance(node.get("keys"), dict):
        keys: dict[str, Any] = {}
        for k, v in node["keys"].items():
            name = k[:MAX_NAME]
            if len(k) > MAX_NAME:
                node["names_truncated"] = True
            if name in keys:
                node["more_keys"] = node.get("more_keys", 0) + 1
                continue
            keys[name] = _cut_node(v)
        node["keys"] = keys
    if "items" in node:
        node["items"] = _cut_node(node["items"])
    return node


class _ParsersOnly:
    """A meta-path finder that refuses every top-level module outside the standard library and
    ``LIBRARIES``, before any other finder looks."""

    @staticmethod
    def find_spec(name: str, path: Any = None, target: Any = None) -> None:
        top = name.partition(".")[0]
        if top in sys.stdlib_module_names or top in sys.builtin_module_names:
            return None
        if top in LIBRARIES:
            return None
        raise ModuleNotFoundError(
            f"{top} is not imported by the shape child",
            name=name,
        )


def main(argv: list[str]) -> int:
    """The child: ``argv`` is ``[script, format, "1"|"0" (truncated), library dir, ...]``; the bytes come
    on stdin and the shape goes to stdout as JSON."""
    fmt, truncated = argv[1], argv[2] == "1"
    sys.meta_path.insert(0, _ParsersOnly)
    # after the standard library, so nothing there can shadow it
    sys.path.extend(argv[3:])
    try:
        data = sys.stdin.buffer.read(STDIN_CAP + 1)
        if len(data) > STDIN_CAP:
            shape = {"format": fmt, "error": "too_large"}
        else:
            shape = shape_of(fmt, data, truncated)
        out = json.dumps(shape).encode("ascii")
    except MemoryError:
        return LIMIT_EXIT
    sys.stdout.buffer.write(out)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
