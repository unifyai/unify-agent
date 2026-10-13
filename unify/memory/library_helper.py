"""The memory library: three calls and one item type.

    memory.ask(request) -> list[Item]               # with the librarian on: every item it judges applicable,
                                                    # each with .reason; [] when nothing applies
    memory.items(topic=None, kind=None) -> Items    # browse; print() shows a compact table, page by page
    memory.get(ref) -> Item                         # one item, by any name that names exactly one

An :class:`Item` is a frozen record of one function or note: ``id``, ``kind``, ``summary``, ``applies``,
``evidence``, ``status``, ``example``, ``text``, ``links``, ``history`` (and ``reason``, from ``ask``).
``print(item)`` shows its card; ``item.fits(inputs)`` is its own applicability check (a :class:`Fit`); a function
item is called directly, ``item(*args)``: the call is tracked, and what the function raises reaches the caller
unchanged; calling a note raises an error that says what to do instead. Every call checks its arguments' types
first, and every error of the package's own names the call as written, what was expected with an example call,
and the nearest valid targets or the right call. The whole typed contract is ``memory/__init__.pyi`` beside this
file; ``help(memory)`` shows the same calls and types. Library functions also import as ordinary modules:
``from memory.text.dates import parse_date``.

For one build the earlier calls stay as aliases that work and print one line naming the new call:
``index`` (now ``items``), ``show`` (now ``get``), ``find`` and ``match`` (now ``Item.fits``, and ``ask``) and
``run`` (now calling the item).

The harness generates this file as ``memory/__init__.py`` in every copy of the library; edits are discarded. It
imports only the standard library and its sibling ``memory/_contract.py``, and reads only files beside the package:
``INDEX.md``, ``links.json`` and the harness's data under ``.memory/``.

Imported as the package ``memory``, it also wraps each public function of every library module as that module is
imported (:class:`_CallFinder`), so that each call raises the audit event ``memory.call`` with the item id, a short
digest of the arguments (never their values) and the outcome: ``ok``, ``refused`` (a ``MemoryInputError`` left it)
or ``error`` (any other exception). The wrapper returns what the function returns and re-raises what it raises,
unchanged; whatever audit hooks the process has decide what is kept. The package's own calls raise
``memory.helper_call`` the same way: ``memory:get``, ``memory:items`` and ``memory:ask`` (lookups and questions),
the aliases under their own names (``memory:index``, ``memory:show``, ``memory:find``, ``memory:match``,
``memory:run``), a function item's call as ``memory:run`` (a tracked call, as ``run`` is) and ``Item.fits`` as
``memory:match``.

``match`` and ``Item.fits`` read what each stored procedure was saved against (the landed procedures in
``.memory/find.json``'s ``procedures``, the harness's captured attempts under ``.memory/captures/``) and hand it,
identifiers masked, to :data:`MATCHER`, a replaceable module-level hook; the default compares structure only and
runs nothing. Each raises ``memory.match_result`` with the ids it listed. A tracked call (``run``, or calling a
function item) also raises ``memory.run_result`` with its target and outcome. Neither event carries a value or a
result. A call that raised also raises ``memory.call_failed`` with the id, the exception's type, the first line of
its message and, for the package's own calls, the arguments' shape (types and sizes, never values).
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
from collections.abc import Callable
from typing import Annotated, Any, Literal, NamedTuple

# The contract runtime: typed arguments checked at the call and errors that teach. In the library package it is
# the sibling file memory/_contract.py (the harness's module byte for byte); in the harness, the module itself.
if __name__ == "memory":
    from memory._contract import (  # noqa: F401 - the package's own contract
        Ambiguous,
        ArgumentError,
        ContractError,
        Ge,
        Le,
        MinLen,
        NotFound,
        WrongCall,
        call_text,
        checked,
        nearest,
        render_pyi,
        shown,
    )
else:
    from unify.common.contract_runtime import (  # noqa: F401
        Ambiguous,
        ArgumentError,
        ContractError,
        Ge,
        Le,
        MinLen,
        NotFound,
        WrongCall,
        call_text,
        checked,
        nearest,
        render_pyi,
        shown,
    )

INDEX_FILE = "INDEX.md"
LINKS_FILE = "links.json"
ITEMS_FILE = ".memory/items.json"
FIND_FILE = ".memory/find.json"
MATCHER_FILE = ".memory/matcher.py"
SHAPES_FILE = ".memory/shapes.py"
ITEM_LINE = "- `"
CALL_EVENT = "memory.call"
#: The audit event of a call of this package's own helpers (``index``, ``show``, ``find``): the same arguments
#: as ``memory.call``, kept apart because the helpers are not library items.
HELPER_EVENT = "memory.helper_call"
CALL_OUTCOMES = ("ok", "refused", "error")
#: The audit event of a call that raised: the id, the exception's type, its message's first line and, for the
#: helpers, the arguments' shape (:func:`args_shape`).
FAILED_EVENT = "memory.call_failed"
#: The most characters of a failed call's message line and of its arguments' shape the event carries.
FAILURE_TEXT = 200
REFUSAL = "MemoryInputError"
DIGEST_CHARS = 16
# The status and evidence-level names (``item_states``'s; this file imports only the standard library, so it
# keeps its own copy, which a test holds equal): the status a row without one has, the statuses the index leaves
# out, and the old stored names each now means.
_NEW_ITEM_STATUS = "on-trial"
_HIDDEN_STATUSES = ("retired",)
_OLD_STATUSES = {
    "experimental": "on-trial",
    "stable": "adopted",
    "suspect": "retired",
    "deprecated": "retired",
}
_OLD_LEVELS = {
    "unverified": "unchecked",
    "experimental": "unchecked",
    "verified": "reproduces-its-case",
}


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


def wanted_keys(name: object) -> list[str]:
    """The sections a name can mean, in order: :func:`wanted_key`, then for a bare name (no ``memory.``,
    ``memory/`` or ``notes/`` prefix) the notes topic of that name, ``notes/<name>``."""
    n = str(name).strip().strip("/")
    keys = [wanted_key(name)]
    if n and not n.startswith(("memory.", "memory/", "notes/")):
        keys.append("notes/" + n)
    return keys


def find_section(text: str, name: object) -> str:
    """The full section of the index *text* that *name* means: a package, else for a bare name the notes
    topic of that name; LookupError naming the sections otherwise."""
    _, sections = split_index(text)
    wanted = wanted_keys(name)
    for key in wanted:
        for heading, body in sections:
            if section_key(heading) == key:
                return f"## {heading}\n{body}"
    keys = ", ".join(section_key(h) for h, _ in sections) or "(none)"
    tried = " or ".join(repr(k) for k in wanted)
    raise LookupError(
        f"memory: no index section {tried}; the sections are: {keys}; "
        'memory.show("<id>") shows one item',
    )


def item_count(body: str) -> int:
    """The item lines of a section body."""
    return sum(1 for line in body.splitlines() if line.startswith(ITEM_LINE))


#: The contract: what ``help(memory)`` and ``from memory import *`` show. The aliases (``index``, ``show``, ``find``,
#: ``match``, ``run``) and the matcher hook stay attributes of the package for one build, outside the contract.
__all__ = [
    "Ambiguous",
    "ArgumentError",
    "ContractError",
    "Fit",
    "Item",
    "Items",
    "NotFound",
    "WrongCall",
    "get",
    "items",
]

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
    """One function :func:`find` lists: its id, how to call it, and which of its parameters the value fits."""

    name: str  # memory.<package>.<module>:<name>
    signature: str
    param: str  # the parameter whose recorded arguments matched ("" for a shape recorded without one)
    match: str  # exact | structure | weak (a single named key, or a whole shape without named keys)
    summary: str
    reason: str  # the keys or columns that matched


class _FileUnreadable(RuntimeError):
    """A generated file of the copy that cannot be read or parsed; it names the file by its path in the copy (never a
    host path) and the error's type. The public calls turn it into an error that says what to do (:func:`_guard`)."""

    def __init__(self, rel: str, error: str) -> None:
        self.rel = rel
        self.error = error
        super().__init__(f"memory: {rel} cannot be read ({error})")


def _read(rel: str) -> str:
    path = os.path.join(_ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise _FileUnreadable(rel, type(exc).__name__) from None


def _json(rel: str) -> dict:
    text = _read(rel)
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise _FileUnreadable(rel, type(exc).__name__) from None
    if not isinstance(data, dict):
        raise _FileUnreadable(rel, "not an object")
    return data


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


def _summary(record: Any) -> str:
    if not record:
        return "no record yet"
    return ", ".join(f"{k} {record[k]}" for k in sorted(record))


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
        try:
            with open(path, "rb") as fh:
                code = compile(fh.read(), path, "exec")
            exec(code, module.__dict__)  # noqa: S102 - the harness's own generated file
        except Exception as exc:  # noqa: BLE001 - a missing or malformed matcher is the copy's, named as such
            raise _FileUnreadable(MATCHER_FILE, type(exc).__name__) from None
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


def _param(desc: dict) -> str:
    p = desc.get("param")
    return p if isinstance(p, str) else ""


def _match(m: Any, mine: dict, recorded: dict) -> Any:
    """v2's match; else, for two shapes without a named key (a grid, a list of numbers),
    their exact equality as the weakest match, so such an argument is still found, after every v2 match.
    """
    found = m.match(mine, recorded)
    if found is None and m.signature(mine) and m.same_shape(mine, recorded):
        return 0, 0.0, []
    return found


def _find(value: Any) -> list[Found]:
    """The functions whose tests call them on an argument with the shape of *value*, best match first, each
    with the parameter that argument was passed as. v2's input-shape matcher and ranking, unchanged (level,
    then share of named keys matched, then how many recorded shapes match, then name; at most five). A shape
    without named keys (a grid) matches only an identical one, ranked weak and below every named match. A
    function's best-matching shape names the parameter, the first in recorded order on a tie. An empty list
    means that no listed function's tests call it on data of this shape, or that the value has too little
    structure (plain text has none)."""
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
        param = ""
        for d in e.get("input_shapes", []):
            shape = {k: v for k, v in d.items() if k != "param"}
            hits = [x for x in (_match(m, v, shape) for v in mine) if x is not None]
            if not hits:
                continue
            top = max(hits, key=lambda x: (x[0], x[1]))
            if best is None or top[:2] > best[:2]:
                best, count, param = top, 1, _param(d)
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
            if shown
            else "matched the whole shape (it has no named keys)"
        )
        name = e["id"]
        found = Found(
            name,
            e.get("signature", ""),
            param,
            _LEVELS[level],
            e.get("summary", ""),
            reason,
        )
        ranked.append((-level, -share, -count, name, found))
    return [f for *_, f in sorted(ranked, key=lambda r: r[:4])][: m.MAX_RESULTS]


# --- recognition by content (``match``) and trying a stored procedure (``run``) ---------------------------------
#
# ``match`` informs and runs nothing: it reads what each stored procedure was saved against and asks the matcher
# (:data:`MATCHER`, a module-level hook) which of them fit what the caller passed. ``run`` is the agent's own call of
# one procedure, in its own cell; nothing here runs on its own. Everything read here was built by the harness with
# identifiers already removed (signatures, docstrings, summary sections, what broke); this file strips nothing.

#: Captured attempts the harness exported beside the package, one directory per capture id (written by the capture
#: store, never by this file): ``signature.json``, ``record.json``, ``procedure.py`` and ``saved.json`` (what the
#: attempt was saved against besides its signature: ``docstring``, ``asked``, ``approach`` and ``broke``).
CAPTURES_DIR = ".memory/captures"
CAPTURE_PREFIX = "capture:"
#: The audit events of a ``match`` result (each listed row's id, level, status and how it matched; the failed
#: attempts apart) and of a ``run`` (its target, outcome, repeatability, level, conditions and the ids that agreed).
#: Ids and short labels only: never a value, a result or a request's text.
MATCH_EVENT = "memory.match_result"
RUN_EVENT = "memory.run_result"
RUN_OUTCOMES = ("returned", "refused", "error")
REPEATABLE = ("yes", "no", "unchecked")
CONDITIONS = ("none", "held", "not-held", "error")
#: How long one run of a captured procedure may take, in seconds, before its process group is stopped.
RUN_TIMEOUT_S = 120.0
#: The most a captured procedure's run may print; past it the run is stopped, so a chatty run never fills the box's
#: memory (its report is the last line, well within it).
RUN_OUTPUT_MAX_BYTES = 4 * 1024 * 1024
_CAPTURE_ID = re.compile(r"[0-9a-f]{16}")
_LABEL = re.compile(r"[a-z][a-z0-9-]{0,39}")
_RESULT_CHARS = 2000
_DETAIL_CHARS = 500
_RUN_MARK = "\x1ememory-run-report\x1e"


class Candidate(NamedTuple):
    """One stored procedure as the matcher sees it: what it was saved against, identifiers already removed."""

    id: str  # an item id (memory.<package>.<module>:<name>) or capture:<16 hex digits>
    source: str  # library | capture
    signature: dict  # its applicability signature: input, output, relations, constraints (never request text)
    docstring: str
    asked: str  # the "Asked" section of its source episode's summary.md
    approach: str  # the "Approach" section of the same summary
    level: str  # its evidence level, as recorded ("unknown" when none is)
    status: str  # its status, as recorded
    failed: bool  # a captured attempt that failed: listed apart, never offered to run
    broke: tuple  # for a failed capture, what broke, one line each


class Query(NamedTuple):
    """What a ``match`` call was given: the request text, the caller's inputs (unchanged), and the shapes the
    structural reading gives them (none for plain text)."""

    request: Any
    inputs: Any
    shapes: tuple


class MatchRow(NamedTuple):
    """One procedure ``match`` lists."""

    id: str
    source: str
    level: str
    status: str
    matched: str  # how the matcher says it matched (the structural default: exact | structure)
    relations: tuple
    conditions: str
    goal: str
    broke: tuple


def _row_text(row: MatchRow) -> str:
    parts = [f"level {row.level}", f"status {row.status}", f"matched: {row.matched}"]
    if row.relations:
        parts.append("relations: " + ", ".join(row.relations))
    parts.append("conditions: " + row.conditions)
    lines = [f"- {row.id}  [" + "; ".join(parts) + "]"]
    if row.goal:
        lines.append("  goal: " + " ".join(row.goal.split())[:300])
    for line in row.broke:
        lines.append("  broke: " + line)
    return "\n".join(lines)


class Matches(list):
    """The runnable procedures ``match`` lists (a list of :class:`MatchRow`), with ``failed``, the captured attempts
    that were tried on input like this and failed, kept apart. It prints as a few lines."""

    def __init__(self, rows: list, failed: list, note: str = "") -> None:
        super().__init__(rows)
        self.failed = list(failed)
        self.note = note

    def __repr__(self) -> str:
        out: list[str] = []
        if self:
            out.append(
                f"{len(self)} stored procedure(s) match what you passed. The match compares structure only, so "
                "it can be wrong; memory.get(id) opens one and calling it tries it:",
            )
            out += [_row_text(r) for r in self]
        else:
            out.append("No stored procedure matches what you passed.")
        if self.failed:
            out.append("Tried on input like this and failed:")
            out += [_row_text(r) for r in self.failed]
        if self.note:
            out.append(self.note)
        return "\n".join(out) + "\n"

    __str__ = __repr__


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: str) -> Any:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _str(value: Any, default: str = "") -> str:
    return value if isinstance(value, str) and value else default


def _label(value: Any, default: str = "other") -> str:
    """A short lower-case label (a level, a status, how a row matched) as the audit event may carry it."""
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else default


def _library_candidates() -> list[Candidate]:
    """The landed procedures the export lists in ``find.json`` (``procedures``), each with what it was saved
    against."""
    try:
        rows = _json(FIND_FILE).get("procedures", [])
    except (RuntimeError, ValueError, AttributeError):
        return []
    out: list[Candidate] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        sig = row.get("signature")
        out.append(
            Candidate(
                id=row["id"],
                source="library",
                signature=sig if isinstance(sig, dict) else {},
                docstring=_str(row.get("docstring")),
                asked=_str(row.get("asked")),
                approach=_str(row.get("approach")),
                level=_str(row.get("level"), "unknown"),
                status=_str(row.get("status"), "unknown"),
                failed=False,
                broke=(),
            ),
        )
    return out


def _capture_dir(capture_id: str) -> str:
    return os.path.join(_ROOT, CAPTURES_DIR, capture_id)


def _capture_candidates() -> tuple[list[Candidate], int]:
    """The exported captures, in id order, and how many could not be read (no ``signature.json`` object)."""
    base = os.path.join(_ROOT, CAPTURES_DIR)
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return [], 0
    out: list[Candidate] = []
    unreadable = 0
    for name in names:
        if _CAPTURE_ID.fullmatch(name) is None:
            continue
        d = os.path.join(base, name)
        sig = _read_json(os.path.join(d, "signature.json"))
        if not isinstance(sig, dict):
            unreadable += 1
            continue
        rec = _read_json(os.path.join(d, "record.json"))
        rec = rec if isinstance(rec, dict) else {}
        saved = _read_json(os.path.join(d, "saved.json"))
        saved = saved if isinstance(saved, dict) else {}
        failed = rec.get("outcome") == "failed"
        broke = saved.get("broke") if isinstance(saved.get("broke"), list) else []
        out.append(
            Candidate(
                id=CAPTURE_PREFIX + name,
                source="capture",
                signature={k: v for k, v in sig.items() if k != "goal"},
                docstring=_str(saved.get("docstring")),
                asked=_str(saved.get("asked")) or _str(sig.get("goal")),
                approach=_str(saved.get("approach")),
                level=_str(rec.get("level"), "unchecked"),
                status=_str(rec.get("status"), "on-trial"),
                failed=failed,
                broke=tuple(str(b) for b in broke if isinstance(b, str)) if failed else (),
            ),
        )
    return out, unreadable


def _candidates_all() -> tuple[list[Candidate], int]:
    captures, unreadable = _capture_candidates()
    return _library_candidates() + captures, unreadable


def _query(m: Any, request: Any, inputs: Any, candidates: list[Candidate]) -> Query:
    """The query for *request* (read as JSON text only: sniffing prose as a table would make its first line's words
    column names) and *inputs* (read as ``find`` reads a value: a parsed value, JSON text, a structured file or its
    path)."""
    exts = sorted(
        {
            str(c.signature["input"].get("ext", ""))
            for c in candidates
            if isinstance(c.signature.get("input"), dict)
            and c.signature["input"].get("kind") == "file"
        }
        | {".json", ".jsonl", ".csv", ".tsv"},
    )
    shapes: list[dict] = []
    if isinstance(request, (str, bytes, bytearray)) and request:
        desc = m.value_shape(request)
        if desc is not None:
            shapes.append(desc)
    if isinstance(inputs, (str, bytes, bytearray)) and not _is_file(m, inputs):
        # text that is not a file's path is read as JSON only, like the request: never sniffed as a table
        desc = m.value_shape(inputs) if inputs else None
        if desc is not None:
            shapes.append(desc)
    elif inputs is not None:
        shapes += _candidates(m, inputs, exts)
    return Query(request, inputs, tuple(shapes))


def _is_file(m: Any, value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= m._MAX_PATH_CHARS
        and "\n" not in value
        and "\0" not in value
        and os.path.isfile(value)
    )


def structural_match(query: Query, candidates: list[Candidate]) -> list[tuple[str, str]]:
    """The default matcher: structure only, never words. A candidate matches when one of the query's shapes has the
    same shape as its signature's ``input`` (``exact``) or matches it as ``find`` matches an argument, with at least
    a structure-level match (``structure``). The docstring and the summary sections are not compared, and nothing
    is run: a stated ``applies`` is shown, not called. Candidates keep their given order (no ranking)."""
    if not query.shapes:
        return []
    m = _matcher()
    out: list[tuple[str, str]] = []
    for c in candidates:
        recorded = c.signature.get("input") if isinstance(c.signature, dict) else None
        if not isinstance(recorded, dict):
            continue
        best = ""
        for mine in query.shapes:
            try:
                if m.same_shape(mine, recorded):
                    best = "exact"
                    break
                found = m.match(mine, recorded)
            except Exception:  # noqa: BLE001 - a malformed signature matches nothing
                continue
            if found is not None and found[0] >= 1 and best != "exact":
                best = "exact" if found[0] == 2 else "structure"
        if best:
            out.append((c.id, best))
    return out


#: The matching strategy: called as ``MATCHER(query, candidates)`` with a :class:`Query` and the
#: :class:`Candidate` list (each with its signature, docstring and source episode's Asked and Approach sections), it
#: returns ``(candidate id, how it matched)`` pairs in the order to show them. Replace it (``memory.MATCHER = fn``, or
#: in the harness's copy of this file) to change how ``match`` matches; the call, its audit event and its printed
#: form stay the same.
MATCHER: Any = structural_match


def _conditions(signature: dict) -> str:
    cons = signature.get("constraints") if isinstance(signature, dict) else None
    cons = cons if isinstance(cons, dict) else {}
    parts = []
    for name in ("applies", "holds"):
        c = cons.get(name)
        if isinstance(c, dict) and (c.get("stated") or c.get("exists")):
            held = c.get("held")
            state = "held" if held is True else "did not hold" if held is False else "not shown"
            parts.append(f"{name} {state} on its recorded case")
    return "; ".join(parts) if parts else "none stated"


def _match_row(c: Candidate, matched: str) -> MatchRow:
    rel = c.signature.get("relations") if isinstance(c.signature, dict) else None
    return MatchRow(
        id=c.id,
        source=c.source,
        level=c.level,
        status=c.status,
        matched=matched,
        relations=tuple(str(r) for r in rel) if isinstance(rel, list) else (),
        conditions=_conditions(c.signature),
        goal=c.asked,
        broke=c.broke,
    )


def _emit_event(event: str, *args: Any) -> None:
    try:
        sys.audit(event, *args)
    except Exception:  # noqa: BLE001 - an audit hook never changes the call
        pass


def _matched(request: Any, inputs: Any) -> tuple[list[MatchRow], list[MatchRow], Query, int, str]:
    """The rows :data:`MATCHER` gives for *request* and *inputs*: runnable, failed, the query, unreadable captures,
    and why the matcher gave nothing when it raised (``""`` otherwise): a failing matcher lists nothing and never
    breaks the cell."""
    m = _matcher()
    candidates, unreadable = _candidates_all()
    query = _query(m, request, inputs, candidates)
    by_id = {c.id: c for c in candidates}
    rows: list[MatchRow] = []
    failed: list[MatchRow] = []
    seen: set[str] = set()
    try:
        pairs = list(MATCHER(query, list(candidates)) or ())
    except Exception as exc:  # noqa: BLE001 - the matcher's failure is reported, not raised
        return rows, failed, query, unreadable, f"the matcher failed ({type(exc).__name__}), so nothing is listed."
    for pair in pairs:
        try:
            cid, how = pair
        except (TypeError, ValueError):
            continue
        c = by_id.get(cid) if isinstance(cid, str) else None
        if c is None or cid in seen:
            continue
        seen.add(cid)
        (failed if c.failed else rows).append(_match_row(c, str(how)[:40]))
    return rows, failed, query, unreadable, ""


def _event_rows(rows: list[MatchRow]) -> tuple:
    return tuple((r.id, _label(r.level, "unknown"), _label(r.status, "unknown"), _label(r.matched)) for r in rows)


def _match_all(request: Any = None, inputs: Any = None) -> Matches:
    """The stored procedures whose recorded input fits what you pass: *request*, the request's text, and *inputs*,
    data you were given (a dict, a list, JSON text, or a structured file or its path). Each row gives the
    procedure's id, evidence level, status, how it matched, the relations and conditions it was saved with, and
    the goal it was built for; captured attempts that were tried on input like this and failed are listed apart,
    under ``.failed``, and are never offered to run. How it matches is :data:`MATCHER`'s choice (structure only by
    default, so plain text with no structure matches nothing); a match can be wrong. It runs nothing."""
    rows, failed, query, unreadable, broken = _matched(request, inputs)
    note = broken
    if not query.shapes and not broken:
        note = (
            "What you passed has no structure the matcher reads (plain text has none); pass data you were "
            "given as inputs=..., or browse with memory.items()."
        )
    if unreadable:
        note = (note + " " if note else "") + f"{unreadable} captured attempt(s) could not be read."
    _emit_event(MATCH_EVENT, _event_rows(rows), _event_rows(failed))
    return Matches(rows, failed, note)


class RunReport:
    """How one ``run`` went. ``outcome``: returned, refused (``MemoryInputError``) or error; ``result``: what it
    returned (None otherwise); ``repeatable``: whether a second run gave an equal result (None: not checked, with
    the reason in ``repeat_note``); ``conditions``: whether its stated conditions held; ``agrees_with``: the other
    procedures ``match`` lists for the same argument that gave an equal result."""

    def __init__(self, target: str, **kw: Any) -> None:
        self.target = target
        self.outcome: str = kw.get("outcome", "error")
        self.result: Any = kw.get("result")
        self.error: str = kw.get("error", "")
        self.detail: str = kw.get("detail", "")
        self.repeatable: bool | None = kw.get("repeatable")
        self.repeat_note: str = kw.get("repeat_note", "")
        self.conditions: str = kw.get("conditions", "none stated")
        self.conditions_code: str = kw.get("conditions_code", "none")
        self.level: str = kw.get("level", "unknown")
        self.status: str = kw.get("status", "unknown")
        self.record: str = kw.get("record", "")
        self.agrees_with: list[str] = list(kw.get("agrees_with") or [])
        self.agreement_note: str = kw.get("agreement_note", "")
        # the value a captured procedure read from the text it was given (JSON, not too large), for agreement
        self._read_value: Any = kw.get("read_value", _NO_VALUE)
        # the exception a library function raised in this process (None otherwise): an item call re-raises it, the
        # same object with its traceback, as a direct call would
        self._raised: BaseException | None = kw.get("raised")

    def __repr__(self) -> str:
        out = [f"run of {self.target}: {self.outcome}"]
        if self.outcome == "returned":
            out.append("result: " + _bounded_repr(self.result))
        else:
            out.append(f"error: {self.error}" + (f": {self.detail}" if self.detail else ""))
        if self.repeatable is None:
            out.append("repeatable: not checked" + (f" ({self.repeat_note})" if self.repeat_note else ""))
        else:
            out.append(
                "repeatable: " + ("yes, two runs gave equal results" if self.repeatable else "no, two runs differed"),
            )
        out.append("conditions: " + self.conditions)
        if self.agrees_with:
            out.append("agrees with: " + ", ".join(self.agrees_with))
        if self.agreement_note:
            out.append(self.agreement_note)
        out.append(f"level {self.level}, status {self.status}")
        out.append(self.record or "it has not been used since it was stored")
        out.append("Its result is unchecked: nothing here compares it with what your request needs.")
        return "\n".join(out) + "\n"

    __str__ = __repr__


def _bounded_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - a result that cannot be shown is named by its type
        return f"<{type(value).__name__}: repr failed ({type(exc).__name__})>"
    return text if len(text) <= _RESULT_CHARS else text[:_RESULT_CHARS] + " …"


def _same(a: Any, b: Any) -> bool:
    try:
        return json.dumps(a, sort_keys=True, default=repr) == json.dumps(b, sort_keys=True, default=repr)
    except Exception:  # noqa: BLE001 - values JSON cannot order are compared by their text
        try:
            return repr(a) == repr(b)
        except Exception:  # noqa: BLE001
            return False


def _record_line(use: Any) -> str:
    if not isinstance(use, dict) or not use.get("uses"):
        return ""
    return "its record: " + _summary(use)


def _item_row(target: str) -> dict | None:
    try:
        row = _json(ITEMS_FILE).get("items", {}).get(target)
    except (RuntimeError, ValueError, AttributeError):
        return None
    return row if isinstance(row, dict) and row.get("kind") == "function" and ":" in target else None


def _procedure_level(target: str) -> str:
    """A landed procedure's evidence level as ``find.json``'s ``procedures`` gives it; ``unknown`` otherwise."""
    for c in _library_candidates():
        if c.id == target:
            return c.level
    return "unknown"


def _item_function(target: str) -> Any:
    import importlib

    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _run_item(target: str, args: tuple, kwargs: dict) -> RunReport:
    """Call a library function by id in this process (an ordinary use: the wrapper raises ``memory.call``). A
    second call checks that it repeats, unless the harness derived its effect as ``write`` (a call known to change state
    is never repeated), on a copy of the arguments taken first."""
    import copy

    row = _item_row(target)
    if row is None:
        return RunReport(
            target,
            outcome="error",
            error="LookupError",
            detail="no stored procedure has this id; memory.items() lists the items and memory.get(ref) opens one",
        )
    base = {
        "level": _str(row.get("level")) or _procedure_level(target),
        "status": _str(row.get("status"), "unknown"),
        "record": _record_line(row.get("use")),
    }
    try:
        fn = _item_function(target)
    except Exception as exc:  # noqa: BLE001 - reported, never raised past the cell
        return RunReport(target, outcome="error", error=type(exc).__name__, detail=str(exc)[:_DETAIL_CHARS], **base)
    copied: Any = None
    note = ""
    if row.get("effect") == "write":
        note = "a procedure whose recorded effect is write is called once"
    else:
        try:
            copied = copy.deepcopy((args, kwargs))
        except Exception:  # noqa: BLE001 - arguments that cannot be copied are passed once
            note = "its arguments could not be copied for a second call"
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - reported, never raised past the cell
        return RunReport(target, outcome=call_outcome(exc), error=type(exc).__name__,
                         detail=str(exc)[:_DETAIL_CHARS], repeat_note="it did not return", raised=exc, **base)
    repeatable: bool | None = None
    if copied is not None:
        try:
            repeatable = _same(result, fn(*copied[0], **copied[1]))
        except Exception as exc:  # noqa: BLE001
            repeatable, note = False, f"the second call raised {type(exc).__name__}"
    return RunReport(target, outcome="returned", result=result, repeatable=repeatable, repeat_note=note, **base)


#: The child process that runs a captured procedure (stdlib only): it loads ``procedure.py``, reads the request text
#: through ``read_input`` when the procedure has one and it was given one text, checks ``applies``, runs ``run``
#: twice (once with ``"once": true``), checks ``holds``, and prints one JSON report after a marker line.
_CAPTURE_RUNNER = r'''
import json, sys
MARK = sys.argv[2]
out = {}
def same(a, b):
    try:
        return json.dumps(a, sort_keys=True, default=repr) == json.dumps(b, sort_keys=True, default=repr)
    except Exception:
        return repr(a) == repr(b)
def refused(exc):
    return any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__)
try:
    payload = json.loads(sys.stdin.read())
    args, kwargs, once = payload["args"], payload["kwargs"], payload.get("once", False)
    path = sys.argv[1]
    ns = {"__name__": "captured_procedure", "__file__": path}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    run = ns.get("run")
    if not callable(run):
        raise LookupError("the captured procedure defines no run(value)")
    read = ns.get("read_input")
    if callable(read) and len(args) == 1 and not kwargs and isinstance(args[0], str):
        value = read(args[0])
        call = lambda: run(value)
        try:
            text = json.dumps(value)
            if len(text) <= 262144:
                out["value"] = value
        except Exception:
            pass
    else:
        value = args[0] if len(args) == 1 and not kwargs else None
        call = lambda: run(*args, **kwargs)
    applies = ns.get("applies")
    if callable(applies) and value is not None and not once:
        try:
            out["applies"] = bool(applies(value))
        except Exception as exc:
            out["applies_error"] = type(exc).__name__
    first = call()
    out["outcome"] = "returned"
    if not once:
        try:
            out["repeatable"] = same(first, call())
        except Exception as exc:
            out["repeatable"] = False
            out["repeat_note"] = "the second run raised " + type(exc).__name__
        holds = ns.get("holds")
        if callable(holds) and value is not None:
            try:
                out["holds"] = bool(holds(value, first))
            except Exception as exc:
                out["holds_error"] = type(exc).__name__
    try:
        json.dumps(first)
        out["result"] = first
    except Exception:
        out["result"] = repr(first)[:20000]
        out["result_is_text"] = True
except Exception as exc:
    out["outcome"] = "refused" if refused(exc) else "error"
    out["error"] = type(exc).__name__
    out["detail"] = str(exc)[:500]
sys.stdout.write("\n" + MARK + json.dumps(out, default=repr))
'''


def _capture_conditions(rep: dict) -> tuple[str, str]:
    parts = []
    code = "none"
    for name in ("applies", "holds"):
        if name in rep:
            parts.append(f"{name} {'held' if rep[name] else 'did not hold'}")
            if not rep[name]:
                code = "not-held"
            elif code == "none":
                code = "held"
        elif f"{name}_error" in rep:
            parts.append(f"{name} raised {rep[name + '_error']}")
            code = "error" if code != "not-held" else code
    return ("; ".join(parts) if parts else "none stated"), code


def _confined(capture_id: str, args: tuple, kwargs: dict, once: bool = False) -> dict:
    """Run a captured procedure in a child process confined as the capture step runs it: an isolated interpreter
    (``-I``), an empty environment, its own process group, the capture's directory as its working directory and a
    bounded time (:data:`RUN_TIMEOUT_S`); the child's process group is stopped when the time runs out or the run
    ends. Inputs and the result cross as JSON. The child's report, or ``{"outcome": "error", ...}``."""
    import subprocess

    d = _capture_dir(capture_id)
    proc_path = os.path.join(d, "procedure.py")
    if _CAPTURE_ID.fullmatch(capture_id) is None or not os.path.isfile(proc_path):
        return {"outcome": "error", "error": "LookupError",
                "detail": "no captured procedure has this id; memory.items('captures') lists them"}
    try:
        payload = json.dumps({"args": list(args), "kwargs": kwargs, "once": once})
    except (TypeError, ValueError) as exc:
        return {"outcome": "error", "error": type(exc).__name__,
                "detail": "a captured procedure takes text or JSON values"}
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-c", _CAPTURE_RUNNER, proc_path, _RUN_MARK],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={},
            cwd=d,
            start_new_session=True,
        )
    except OSError as exc:
        return {"outcome": "error", "error": type(exc).__name__, "detail": "the run could not start"}
    try:
        raw, timed_out, over = _bounded_exchange(child, payload.encode("utf-8"), RUN_TIMEOUT_S)
    finally:
        _stop_child(child)
    if timed_out:
        return {"outcome": "error", "error": "TimeoutExpired",
                "detail": f"it ran past {RUN_TIMEOUT_S:g} s and was stopped"}
    if over:
        return {"outcome": "error", "error": "OutputLimit",
                "detail": f"it printed more than {RUN_OUTPUT_MAX_BYTES // 1024**2} MiB and was stopped"}
    stdout = (raw or b"").decode("utf-8", "replace")
    rep: Any = None
    if _RUN_MARK in stdout:
        try:
            rep = json.loads(stdout.rsplit(_RUN_MARK, 1)[1])
        except ValueError:
            rep = None
    if not isinstance(rep, dict) or rep.get("outcome") not in RUN_OUTCOMES:
        return {"outcome": "error", "error": "NoReport",
                "detail": f"the run ended (exit {child.returncode}) without a report"}
    return rep


def _bounded_exchange(child: Any, data: bytes, timeout_s: float) -> tuple[bytes, bool, bool]:
    """Send *data* to *child*'s stdin and read its stdout until it closes, the time runs out or more than
    :data:`RUN_OUTPUT_MAX_BYTES` arrive: ``(output, timed out, over the bound)``. The input is written from a thread
    that ends with the child's pipe, so a child that prints before reading never blocks the bounded read. The caller
    stops the child's process group afterwards, whatever this returns."""
    import select
    import time

    def feed() -> None:
        try:
            child.stdin.write(data)
            child.stdin.close()
        except OSError:
            pass

    feeder = threading.Thread(target=feed, daemon=True, name="memory-run-input")
    feeder.start()
    deadline = time.monotonic() + timeout_s
    fd = child.stdout.fileno()
    chunks: list = []
    size = 0
    timed_out = over = False
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([fd], [], [], left)
            if not ready:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            size += len(chunk)
            if size > RUN_OUTPUT_MAX_BYTES:
                over = True
                break
            chunks.append(chunk)
    finally:
        child._unify_feeder = feeder  # joined by _stop_child once the child is gone
    return b"".join(chunks), timed_out, over


def _stop_child(child: Any) -> None:
    """Stop *child*'s whole process group, whatever it left running, and wait for it; the input thread of
    :func:`_bounded_exchange` then ends with its pipe and is joined, so no thread outlives the call."""
    import signal

    try:
        os.killpg(child.pid, signal.SIGKILL)
    except OSError:
        pass
    if child.poll() is None:
        child.kill()
    child.wait()
    feeder = getattr(child, "_unify_feeder", None)
    if feeder is not None:
        feeder.join(5)


def _run_capture(capture_id: str, args: tuple, kwargs: dict) -> RunReport:
    """Run a captured procedure confined (:func:`_confined`) and report it. Its call raises ``memory.call`` with the
    id ``capture:<id>``, as a library function's call does."""
    target = CAPTURE_PREFIX + capture_id
    if _CAPTURE_ID.fullmatch(capture_id) is None:
        return RunReport(target, outcome="error", error="LookupError",
                         detail="a captured procedure's id is capture: and 16 hex digits; "
                         "memory.items('captures') lists them")
    rec = _read_json(os.path.join(_capture_dir(capture_id), "record.json"))
    rec = rec if isinstance(rec, dict) else {}
    base = {"level": _str(rec.get("level"), "unchecked"), "status": _str(rec.get("status"), "on-trial")}
    if rec.get("outcome") == "failed":
        return RunReport(target, outcome="error", error="LookupError",
                         detail="this captured attempt failed; it is listed to show what broke, not to run", **base)
    rep = _confined(capture_id, args, kwargs)
    if rep.get("error") != "LookupError" or rep["outcome"] != "error":
        _emit(target, args_digest(args, kwargs), "ok" if rep["outcome"] == "returned" else
              ("refused" if rep["outcome"] == "refused" else "error"))
    conditions, code = _capture_conditions(rep)
    return RunReport(
        target,
        outcome=rep["outcome"],
        result=rep.get("result"),
        error=_str(rep.get("error")),
        detail=_str(rep.get("detail")),
        repeatable=rep.get("repeatable") if isinstance(rep.get("repeatable"), bool) else None,
        repeat_note=_str(rep.get("repeat_note"), "" if rep["outcome"] == "returned" else "it did not return"),
        conditions=conditions,
        conditions_code=code,
        record=_record_line(rec.get("track") if isinstance(rec.get("track"), dict) else None),
        read_value=rep["value"] if "value" in rep else _NO_VALUE,
        **base,
    )


_NO_VALUE = object()


def _agreement(
    target: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    read_value: Any = _NO_VALUE,
) -> tuple[list[str], str]:
    """The other runnable procedures :data:`MATCHER` lists for the same single argument whose result equals
    *result*, and a line saying how many were tried. A library function is tried unless its effect is ``write``
    (called unwrapped, and the library calls it makes raise no event, so the check is not a use); a captured one
    runs confined. The other procedures are the ones that match the value the target read from its text, when it
    read one (*read_value*), else the argument itself. Agreement between stored procedures only: nothing is compared
    with anything the request states."""
    if len(args) != 1 or kwargs:
        return [], "agreement with other procedures is checked for a single argument only"
    arg = args[0]
    try:
        if read_value is not _NO_VALUE:
            rows = _matched(None, read_value)[0]
        else:
            rows = _matched(arg, None)[0] if isinstance(arg, str) else _matched(None, arg)[0]
    except Exception as exc:  # noqa: BLE001 - the report still stands without agreement
        return [], f"agreement was not checked ({type(exc).__name__})"
    others = [r for r in rows if r.id != target]
    agree: list[str] = []
    tried = 0
    for r in others:
        try:
            if r.id.startswith(CAPTURE_PREFIX):
                rep = _confined(r.id[len(CAPTURE_PREFIX) :], args, kwargs, once=True)
                if rep.get("outcome") != "returned":
                    tried += 1
                    continue
                other = rep.get("result")
            else:
                row = _item_row(r.id)
                if row is None or row.get("effect") == "write":
                    continue
                fn = inspect.unwrap(_item_function(r.id))
                _QUIET.depth = getattr(_QUIET, "depth", 0) + 1
                try:
                    other = fn(*args)  # nor any library call it makes in turn
                finally:
                    _QUIET.depth -= 1
        except Exception:  # noqa: BLE001 - a procedure that fails here does not agree
            tried += 1
            continue
        tried += 1
        if _same(result, other):
            agree.append(r.id)
    if not others:
        return agree, "no other stored procedure matches this argument"
    return agree, f"checked against {tried} of {len(others)} other matching procedure(s)"


def _tracked_run(target: str, args: tuple, kwargs: dict) -> RunReport:
    """The tracked call of one stored procedure (``memory.run``, and calling a function :class:`Item`): try it on the
    arguments, in this cell, and report how it went: whether it returned, its result, whether a second run agreed,
    whether its stated conditions held, which other matching procedures gave the same result, and its evidence
    level and status. *target* is an item id (``memory.<package>.<module>:<name>``) or a ``capture:<id>``. A
    library function is called here, as an import would call it; a captured procedure runs in a separate confined
    process and takes text or JSON values (given one text, it reads its input from that text). A failure is
    reported, never raised. Nothing runs unless the caller calls this."""
    if target.startswith(CAPTURE_PREFIX):
        report = _run_capture(target[len(CAPTURE_PREFIX) :], args, kwargs)
    else:
        report = _run_item(target, args, kwargs)
    if report.outcome == "returned":
        report.agrees_with, report.agreement_note = _agreement(
            report.target,
            args,
            kwargs,
            report.result,
            report._read_value,
        )
    repeat = "unchecked" if report.repeatable is None else ("yes" if report.repeatable else "no")
    _emit_event(
        RUN_EVENT,
        report.target,
        report.outcome,
        repeat,
        _label(report.level, "unknown"),
        report.conditions_code if report.conditions_code in CONDITIONS else "none",
        tuple(report.agrees_with),
    )
    return report


# --- the contract: memory.get, memory.items and the Item type ------------------------------------------------------
#
# One read of the copy's generated files serves one call (:class:`_Catalog`). Every way of naming an item resolves
# in one place (:meth:`_Catalog.resolve`), and a malformed generated file never escapes a call as a raw exception
# (:func:`_guard`).

Kind = Literal["function", "note"]
Evidence = Literal["unchecked", "reproduces-its-case", "confirmed-by-outcomes"]
Status = Literal["on-trial", "adopted", "retired"]
_EVIDENCE = ("unchecked", "reproduces-its-case", "confirmed-by-outcomes")
_STATUSES = ("on-trial", "adopted", "retired")
#: The rows one printed page of :class:`Items` shows.
PAGE_ROWS = 40
#: The topic the exported captured procedures are listed under.
CAPTURES_TOPIC = "captures"
#: Where the library is under every agent's cells besides the copy's own root: a path under it names an item.
LIBRARY_ROOT = "/memory"
_SUMMARY_CHARS = 100
_ID_COLUMN = 60
_GET_EXPECTED = "an item id, a function name, or a note slug"
_GET_EXAMPLE = "memory.get('notes/grid/stencil.md')"
_TOPIC_EXPECTED = "a package or a notes topic"
_MODULE_PATH = re.compile(r"memory/([a-z][a-z0-9_]*)/([a-z][a-z0-9_]*)\.py")
#: What reading a malformed generated file can raise inside a call (:func:`_guard`).
_UNREADABLE = (
    OSError,
    ValueError,
    LookupError,
    SyntaxError,
    RuntimeError,
    TypeError,
    AttributeError,
)


def _unreadable(exc: BaseException) -> str:
    """What could not be read, for the error of a call that met a malformed or missing generated file."""
    if isinstance(exc, _FileUnreadable):
        return f"its {exc.rel} cannot be read ({exc.error})"
    return f"one of its files cannot be read ({type(exc).__name__})"


def _guard(call: str, fn: Callable[..., Any], *args: Any) -> Any:
    """*fn*(*args*), with any raw error from the copy's files turned into one that says the copy cannot be read and
    what to do; the calls' own errors pass unchanged."""
    try:
        return fn(*args)
    except ContractError:
        raise
    except _UNREADABLE as exc:
        raise NotFound(
            call,
            f"the library copy is unreadable: {_unreadable(exc)}, so nothing could be looked up.",
            hint="Library modules may still import directly; otherwise do the step yourself.",
        ) from None


def _status_of(row: dict) -> str:
    """A row's status in today's names (a row without one is on trial)."""
    status = row.get("status")
    status = status if isinstance(status, str) and status else _NEW_ITEM_STATUS
    return _OLD_STATUSES.get(status, status)


def _level_of(row: dict) -> str | None:
    """A row's evidence level in today's names; None when none is recorded."""
    level = row.get("level")
    return _OLD_LEVELS.get(level, level) if isinstance(level, str) else None


def _one_line(text: Any) -> str:
    """The first non-blank line of *text*, its spaces collapsed ('' for anything else)."""
    if not isinstance(text, str):
        return ""
    for line in text.splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""


def _cut(text: str, size: int = _SUMMARY_CHARS) -> str:
    return text if len(text) <= size else text[: size - 1] + "…"


def _use_when(doc: str | None) -> str:
    """A docstring's ``Use when:`` section on one line ('' when it has none)."""
    lines = (doc or "").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("Use when:"):
            body = [line[len("Use when:") :].strip()]
            for nxt in lines[i + 1 :]:
                if not nxt[:1].isspace() or not nxt.strip():
                    break
                body.append(nxt.strip())
            return " ".join(b for b in body if b)
    return ""


def _history(row: dict) -> tuple[str, ...]:
    history = row.get("history")
    return tuple(str(h) for h in history) if isinstance(history, list) else ()


def _topic_of(iid: str) -> str:
    """The topic an item is listed under: ``memory.<package>``, ``notes/<topic>`` or ``captures``."""
    if iid.startswith(CAPTURE_PREFIX):
        return CAPTURES_TOPIC
    if iid.startswith("notes/"):
        return iid.rsplit("/", 1)[0]
    return ".".join(iid.split(":", 1)[0].split(".")[:2])


def _relative(text: str) -> str | None:
    """*text* as a path inside the copy, when it is a path under the copy's root or under ``/memory``."""
    for root in dict.fromkeys((_ROOT.rstrip("/") + "/", LIBRARY_ROOT + "/")):
        if text.startswith(root):
            return text[len(root) :]
    return None


def _capture_ids() -> list[str]:
    """The exported captured procedures' ids, in id order."""
    try:
        names = sorted(os.listdir(os.path.join(_ROOT, CAPTURES_DIR)))
    except OSError:
        return []
    return [CAPTURE_PREFIX + n for n in names if _CAPTURE_ID.fullmatch(n) is not None]


@dataclasses.dataclass(frozen=True)
class Fit:
    """Whether one item applies to the inputs passed to :meth:`Item.fits`: ``ok`` and ``why``.

    It is true or false as ``ok`` is, so ``if item.fits(inputs):`` reads as it should. The check compares structure
    only, so a fit can be wrong; nothing runs.

    Members:
        ok: whether the input the item was recorded on has the structure of what was passed.
        why: how it matched, or why it did not or could not be checked.
    """

    ok: bool
    why: str

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        return ("fits: " if self.ok else "does not fit: ") + self.why


@dataclasses.dataclass(frozen=True, repr=False)
class Item:
    """One item of the library, a function or a note: a frozen record with every member typed.

    ``print(item)`` shows its full card: a function's signature, a recorded example call, its source with its
    docstring, or a note's text; then its links both ways, its status, its evidence level with the findings that
    set it, its verification and use records and the library commits that changed it.

    Members:
        id: a function is ``memory.<package>.<module>:<name>`` (a captured procedure ``capture:`` and 16 hex
            digits); a note is ``notes/<topic>/<slug>.md``.
        kind: ``'function'`` or ``'note'``.
        summary: one line.
        applies: when it applies, written from the request alone ('' when none is recorded).
        evidence: ``'unchecked'``, ``'reproduces-its-case'`` or ``'confirmed-by-outcomes'`` (a note is unchecked).
        status: ``'on-trial'``, ``'adopted'`` or ``'retired'``.
        example: one recorded call, for a function ('' when none is shown).
        text: a note's text, or a function's source with its docstring.
        links: the ids of the items it links to and of those that link to it.
        history: the library commits that changed it, newest first.
        reason: why ``memory.ask`` judged it applicable ('' from ``memory.get`` and ``memory.items``).

    A function item is called directly, ``item(*args, **kwargs)``: the call is tracked, returns what the function
    returned and raises what it raised. ``item.fits(inputs)`` checks whether it applies to data you were given.
    Calling a note raises an error that says to read it instead.
    """

    id: str
    kind: Kind
    summary: str
    applies: str
    evidence: Evidence
    status: Status
    example: str
    text: str
    links: tuple[str, ...]
    history: tuple[str, ...]
    reason: str = ""

    def __repr__(self) -> str:
        return f"Item({self.id!r}" + (f", reason={shown(self.reason)}" if self.reason else "") + ")"

    def __str__(self) -> str:
        card = self.__dict__.get("_card") or _plain_card(self)
        if self.reason:
            card = card.rstrip("\n") + f"\nWhy it applies: {self.reason}\n"
        return card

    @checked(name="Item.fits")
    def fits(self, inputs: Any) -> Fit:
        """Whether this item applies to *inputs*: its own applicability check. It compares the structure of what you
        pass with the input the item was recorded on, and runs nothing.

        Args:
            inputs: data you were given: a dict, a list, JSON text, or a structured file or its path.

        Returns:
            A Fit: ``ok``, and ``why`` saying how it matched or why not (also when nothing is recorded to compare).

        Raises:
            WrongCall: the item is a note, which takes no inputs.
            NotFound: the library copy cannot be read.

        Example:
            memory.get('memory.text.dates:parse_date').fits({'date': '2026-10-09'})
        """
        return _FIT(self.id, inputs)

    @checked(name="Item.__call__")
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call a function item on *args* and *kwargs*, as calling the function directly would. The call is tracked:
        a library function runs in this cell, as an import would call it, and what it raises reaches you unchanged
        (its refusal is its own ``MemoryInputError``); a captured procedure runs in a separate confined process and
        takes text or JSON values.

        Args:
            args: the function's own positional arguments.
            kwargs: the function's own keyword arguments.

        Returns:
            What the function returned.

        Raises:
            Exception: whatever a library function raised, the same exception with its traceback.
            ArgumentError: a captured procedure refused the input or did not return; the error says why.
            NotFound: the function could not be loaded from this copy of the library.
            WrongCall: the item is a note, which is read, not called.

        Example:
            memory.get('memory.text.dates:parse_date')('2026-10-09')
        """
        report = _RUN_TARGET(self.id, *args, **kwargs)
        if report.outcome == "returned":
            return report.result
        if report._raised is not None:
            raise report._raised  # the function's own exception, as a direct call raises it
        call = call_text(f"memory.get({self.id!r})", args, kwargs)
        detail = (f"{report.error}: {report.detail}" if report.detail else report.error).rstrip(".")
        if self.id.startswith(CAPTURE_PREFIX):
            what = "refused this input" if report.outcome == "refused" else "did not return"
            raise ArgumentError(
                call,
                f"it {what} ({detail}).",
                expected="text or a JSON value like the input it was recorded on",
                example=f"memory.get({self.id!r})(text)",
                hint=f"Its card shows what it was built for: print(memory.get({self.id!r})).",
            )
        raise NotFound(
            call,
            f"it could not be called ({detail}).",
            hint=f"Its card shows its source: print(memory.get({self.id!r})).",
        )


def _plain_card(item: Item) -> str:
    """The card of an :class:`Item` made by hand (one ``get`` made carries its own)."""
    head = f"{item.id}  [{item.status}" + (f", evidence: {item.evidence}" if item.kind == "function" else "") + "]"
    out = [head, item.summary]
    if item.example:
        out.append(f"Example: {item.example}")
    out += ["", item.text.rstrip("\n"), "", "Links: " + (", ".join(item.links) or "none")]
    if item.history:
        out += ["History (newest first):"] + [f"  {h}" for h in item.history]
    return "\n".join(out) + "\n"


@dataclasses.dataclass(frozen=True, repr=False)
class Items:
    """A listing of the library's items (:func:`items`): a frozen sequence of :class:`Item` in id order, functions
    first, then notes, then captured procedures. ``len()``, indexing, slicing and ``for`` work as on a tuple.

    ``print(items)`` shows a compact table one page at a time (id, kind, evidence level, one-line summary), with
    the topics and the call that shows the next page.

    Members:
        rows: every item listed, on every page.
        topic: the topic asked for (None: the whole library).
        kind: the kind asked for (None: both).
        shown_page: the page ``print()`` shows, from 1.
    """

    rows: tuple[Item, ...]
    topic: str | None = None
    kind: Kind | None = None
    shown_page: int = 1

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Any:
        return iter(self.rows)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, slice):
            return Items(self.rows[key], self.topic, self.kind)
        if isinstance(key, int):
            return self.rows[key]
        raise ArgumentError(
            f"{self._call()}[{shown(key)}]",
            f"the index is {shown(key)} ({type(key).__name__}).",
            expected="an int position or a slice",
            example=f"{self._call()}[0]",
        )

    def _call(self) -> str:
        return call_text(
            "memory.items",
            [self.topic] if self.topic is not None else [],
            {"kind": self.kind} if self.kind is not None else {},
        )

    def __str__(self) -> str:
        call = self._call()
        n = len(self.rows)
        if not n:
            return f"{call}: no items" + (" yet" if self.topic is None and self.kind is None else "") + ".\n"
        pages = -(-n // PAGE_ROWS)
        page = max(1, min(self.shown_page, pages))
        rows = self.rows[(page - 1) * PAGE_ROWS : page * PAGE_ROWS]
        functions = sum(1 for r in self.rows if r.kind == "function")
        out = [
            f"{call}: {n} item(s), {functions} function(s) and {n - functions} note(s). Open one with "
            "memory.get(id); call a function item directly.",
        ]
        width = min(max(len(r.id) for r in rows), _ID_COLUMN)
        out.append(f"  {'id':<{width}}  {'kind':<8}  {'evidence':<21}  summary")
        for r in rows:
            evidence = r.evidence if r.kind == "function" else "-"
            out.append(f"  {r.id:<{width}}  {r.kind:<8}  {evidence:<21}  {_cut(r.summary)}".rstrip())
        if self.topic is None:
            counts: dict[str, int] = {}
            for r in self.rows:
                counts[_topic_of(r.id)] = counts.get(_topic_of(r.id), 0) + 1
            out.append(
                "Topics: " + ", ".join(f"{t} ({c})" for t, c in counts.items()) + "; memory.items(topic) lists one.",
            )
        if pages > 1:
            more = f" print({call}.page({page + 1})) shows the next." if page < pages else ""
            out.append(f"Page {page} of {pages}.{more}")
        return "\n".join(out) + "\n"

    __repr__ = __str__

    @checked(name="Items.page")
    def page(self, n: Annotated[int, Ge(1)]) -> Items:
        """This listing with page *n* shown when it is printed.

        Args:
            n: the page to show, from 1 (past the last page, the last page is shown).

        Returns:
            The same items, printing page *n*.

        Raises:
            ArgumentError: n is not a whole number of at least 1.

        Example:
            print(memory.items().page(2))
        """
        return dataclasses.replace(self, shown_page=n)


class _Catalog:
    """One read of the copy's generated files, for one call: the items (``.memory/items.json``), their links, what
    each landed procedure was saved against and the exported captured procedures. *call* is the call as written,
    for its errors. A missing or malformed ``items.json`` raises :class:`_FileUnreadable`."""

    def __init__(self, call: str) -> None:
        self.call = call
        data = _json(ITEMS_FILE)
        rows = data.get("items")
        if not isinstance(rows, dict):
            raise _FileUnreadable(ITEMS_FILE, "no items")
        self.data = data
        self.rows = {
            k: v
            for k, v in rows.items()
            if isinstance(k, str) and isinstance(v, dict) and v.get("kind") in ("function", "note")
        }
        self.functions = sorted(k for k, v in self.rows.items() if v["kind"] == "function")
        self.notes = sorted(k for k, v in self.rows.items() if v["kind"] == "note")
        self.captures = _capture_ids()
        self._links: dict | None = None
        self._asked: dict | None = None
        self._modules: dict[str, tuple[str, Any]] = {}

    # -- what there is

    def all_ids(self) -> list[str]:
        return self.functions + self.notes + self.captures

    def listed(self) -> list[str]:
        """What ``items()`` lists: every function and note but the retired ones, and every captured procedure but
        those that were tried and failed."""
        out = [i for i in self.functions + self.notes if _status_of(self.rows[i]) not in _HIDDEN_STATUSES]
        return out + [c for c in self.captures if not self._capture_failed(c)]

    def topics(self) -> list[str]:
        out = sorted({_topic_of(i) for i in self.functions}) + sorted({_topic_of(i) for i in self.notes})
        return out + ([CAPTURES_TOPIC] if self.captures else [])

    def links(self) -> dict:
        if self._links is None:
            try:
                data = _json(LINKS_FILE).get("items")
            except _FileUnreadable:
                data = None
            self._links = data if isinstance(data, dict) else {}
        return self._links

    def linked(self, iid: str, key: str) -> list[str]:
        row = self.links().get(iid)
        got = row.get(key) if isinstance(row, dict) else None
        return [str(x) for x in got] if isinstance(got, list) else []

    def asked(self) -> dict:
        """What each landed procedure was built for: its source episode's Asked section, by id."""
        if self._asked is None:
            self._asked = {c.id: c.asked for c in _library_candidates() if c.asked}
        return self._asked

    def _capture_failed(self, cid: str) -> bool:
        rec = _read_json(os.path.join(_capture_dir(cid[len(CAPTURE_PREFIX) :]), "record.json"))
        return isinstance(rec, dict) and rec.get("outcome") == "failed"

    # -- resolution: every way of naming an item, in one place

    def matches(self, text: str) -> list[str]:
        """The ids *text* names: a full id; a path under the copy's root or ``/memory`` (a note, or a module's
        functions); ``memory.<package>.<module>.<name>``; ``module:name`` or ``package.module:name``; a function's
        plain name; a note's slug or ``topic/slug``; a module (``package.module``)."""
        s = text.strip()
        if not s:
            return []
        if s in self.rows or s in self.captures:
            return [s]
        rel = _relative(s)
        if rel is not None:
            s = rel.strip("/")
            if s in self.rows:
                return [s]
        found: list[str] = []
        m = _MODULE_PATH.fullmatch(s)
        if m is not None:
            found += [f for f in self.functions if f.startswith(f"memory.{m[1]}.{m[2]}:")]
        elif ":" in s:
            module, _, name = s.rpartition(":")
            module = module.removeprefix("memory.")
            found += [
                f
                for f in self.functions
                if f.rsplit(":", 1)[1] == name
                and (f.split(":", 1)[0] == "memory." + module or f.split(":", 1)[0].endswith("." + module))
            ]
        elif "/" in s:
            slug = s.removeprefix("notes/").removesuffix(".md")
            if slug.count("/") == 1:
                found += [n for n in self.notes if n == f"notes/{slug}.md"]
        else:
            if s.startswith("memory.") and s.count(".") == 3:
                module, _, name = s.rpartition(".")
                found += [f for f in self.functions if f == f"{module}:{name}"]
            found += [f for f in self.functions if f.rsplit(":", 1)[1] == s]
            found += [n for n in self.notes if n.rsplit("/", 1)[1] == s + ".md"]
            module = s if s.startswith("memory.") else "memory." + s
            if module.count(".") == 2:
                found += [f for f in self.functions if f.startswith(module + ":")]
        return list(dict.fromkeys(found))

    def topic_ids(self, topic: str) -> list[str] | None:
        """The listed items of *topic*: a package (``text``, ``memory.text``, ``memory/text``), a module
        (``text.dates``, ``memory.text.dates``), a notes topic (``notes/text``, or ``text`` too) or ``captures``;
        None when *topic* names none."""
        s = topic.strip()
        rel = _relative(s)
        s = (rel if rel is not None else s).strip("/")
        if s.startswith("memory/"):
            s = "memory." + s[len("memory/") :].removesuffix(".py").replace("/", ".")
        listed = self.listed()
        packages = {_topic_of(f) for f in self.functions}
        modules = {f.split(":", 1)[0] for f in self.functions}
        notes = {_topic_of(n) for n in self.notes}

        def under(prefix: str) -> list[str]:
            return [i for i in listed if i.startswith(prefix)]

        if s == CAPTURES_TOPIC and self.captures:
            return under(CAPTURE_PREFIX)
        if s in packages:
            return under(s + ".")
        if s in modules:
            return under(s + ":")
        if s in notes:
            return under(s + "/")
        found: list[str] = []
        known = False
        if "memory." + s in packages:
            found, known = found + under(f"memory.{s}."), True
        elif "memory." + s in modules:
            found, known = found + under(f"memory.{s}:"), True
        if "notes/" + s in notes:
            found, known = found + under(f"notes/{s}/"), True
        return found if known else None

    def resolve(self, ref: Any) -> Item:
        """The one item *ref* names (:meth:`matches`; a library function by its own id; an :class:`Item` by its id,
        keeping its reason), or the error that says what was wrong and what to call."""
        reason = ""
        if isinstance(ref, Item):
            text, reason = ref.id, ref.reason
        elif isinstance(ref, str):
            text = ref
        elif callable(ref):
            iid = getattr(ref, "__memory_item__", None)
            if not isinstance(iid, str):
                iid = f"{getattr(ref, '__module__', '')}:{getattr(ref, '__name__', '')}"
            if iid in self.rows:
                return self.item(iid)
            raise NotFound(
                self.call,
                f"{shown(ref)} is not a function of the library.",
                expected=_GET_EXPECTED,
                example=f"memory.get({self._example_id()!r})",
                hint="Browse with memory.items().",
            )
        else:
            raise ArgumentError(
                self.call,
                f"ref is {shown(ref)} ({type(ref).__name__}).",
                expected="ref: str | Item | Callable",
                example=_GET_EXAMPLE,
            )
        found = self.matches(text)
        if len(found) == 1:
            return self.item(found[0], reason)
        if found:
            raise Ambiguous(
                self.call,
                f"{text!r} names {len(found)} items.",
                expected="one item's full id",
                example=f"memory.get({found[0]!r})",
                nearest=[f"memory.get({i!r})" for i in found],
            )
        if self.topic_ids(text) is not None:
            raise WrongCall(
                self.call,
                "that is a topic, not an item.",
                hint=f"Browse it with memory.items({text.strip()!r}).",
            )
        near = nearest(text, self.all_ids())
        browse = f"memory.items({_topic_of(near[0])!r})" if near else "memory.items()"
        raise NotFound(
            self.call,
            f"no item named {text!r}.",
            expected=_GET_EXPECTED,
            example=f"memory.get({(near[0] if near else self._example_id())!r})",
            nearest=near,
            hint=f"Browse with {browse}.",
        )

    def not_a_topic(self, topic: str) -> None:
        """Raise the error of a topic that names none: the item it names instead, or the nearest topics."""
        named = self.matches(topic)
        if len(named) == 1:
            raise WrongCall(
                self.call,
                "that is an item, not a topic.",
                hint=f"Open it with memory.get({named[0]!r}).",
            )
        if named:
            raise WrongCall(
                self.call,
                f"that names {len(named)} items, not a topic.",
                hint="Open one with " + " or ".join(f"memory.get({i!r})" for i in named) + ".",
            )
        topics = self.topics()
        near = nearest(topic, topics)
        raise NotFound(
            self.call,
            f"no topic named {topic!r}.",
            expected=_TOPIC_EXPECTED,
            example=f"memory.items({(near or topics or ['notes/grid'])[0]!r})",
            nearest=near,
            hint="Browse every topic with memory.items().",
        )

    def _example_id(self) -> str:
        listed = self.listed()
        return listed[0] if listed else "notes/grid/stencil.md"

    # -- building items

    def item(self, iid: str, reason: str = "") -> Item:
        """The :class:`Item` of a known id, with its card."""
        if iid.startswith(CAPTURE_PREFIX):
            built, card = self._capture(iid)
        elif self.rows[iid]["kind"] == "function":
            built, card = self._function(iid, self.rows[iid])
        else:
            built, card = self._note(iid, self.rows[iid])
        if reason:
            built = dataclasses.replace(built, reason=reason)
        object.__setattr__(built, "_card", card)
        return built

    def _definition(self, path: str, name: str) -> tuple[Any, str]:
        if path not in self._modules:
            text = _read(path)
            self._modules[path] = (text, ast.parse(text))
        text, tree = self._modules[path]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                return node, "\n".join(text.splitlines()[start - 1 : node.end_lineno])
        raise LookupError(f"memory: {name} is not defined in its module")

    def _function(self, iid: str, row: dict) -> tuple[Item, str]:
        name = iid.split(":", 1)[1]
        node, text, missing = None, "", ""
        try:
            node, text = self._definition(str(row.get("path") or ""), name)
        except _UNREADABLE as exc:
            missing = exc.error if isinstance(exc, _FileUnreadable) else type(exc).__name__
        doc = ast.get_docstring(node) if node is not None else None
        level = _level_of(row)
        status = _status_of(row)
        example = row.get("example")
        built = Item(
            id=iid,
            kind="function",
            summary=_one_line(row.get("summary")) or _one_line(doc),
            applies=_use_when(doc) or self.asked().get(iid, ""),
            evidence=level if level in _EVIDENCE else "unchecked",
            status=status if status in _STATUSES else _NEW_ITEM_STATUS,
            example=example["call"] if isinstance(example, dict) and isinstance(example.get("call"), str) else "",
            text=text,
            links=tuple(dict.fromkeys(self.linked(iid, "links_to") + self.linked(iid, "linked_from"))),
            history=_history(row),
        )
        return built, self._card(iid, row, text, missing)

    def _note(self, iid: str, row: dict) -> tuple[Item, str]:
        text, missing = "", ""
        try:
            text = _read(iid)
        except _FileUnreadable as exc:
            missing = exc.error
        status = _status_of(row)
        built = Item(
            id=iid,
            kind="note",
            summary=_one_line(row.get("description")) or _one_line(row.get("title")),
            applies="",
            evidence="unchecked",
            status=status if status in _STATUSES else _NEW_ITEM_STATUS,
            example="",
            text=text,
            links=tuple(dict.fromkeys(self.linked(iid, "links_to") + self.linked(iid, "linked_from"))),
            history=_history(row),
        )
        return built, self._card(iid, row, text.rstrip("\n"), missing)

    def _card(self, wanted: str, row: dict, body: str, missing: str) -> str:
        """One item in full, as ``memory.show`` printed it: a function's signature, with an example call quoted from
        its tests (``Example (from <test id>): <call>``) when one is recorded and still in its test file, then its
        source with its docstring; or a note's text. Then its links both ways, its status with, for a function, its
        effect (and why, for an item left out of the index or given a reason), a function's evidence level with the
        findings that set it, its verification and use records, and the subjects of the library commits that
        changed it, newest first."""
        status = _status_of(row)
        level = _level_of(row)
        function = row["kind"] == "function"
        if function:
            effect = row.get("effect")
            out = [
                f"{row.get('signature') or wanted}  [{status}"
                + (f", evidence: {level}" if level else "")
                + (f", effect: {effect}" if effect else "")
                + "]",
            ]
            example = row.get("example")
            if isinstance(example, dict) and example.get("call"):
                out.append(f"Example (from {example.get('test')}): {example['call']}")
        else:
            out = [f"{wanted}  [{status}]" + (f"  {row['title']}" if row.get("title") else "")]
        if status in _HIDDEN_STATUSES:
            out.append(f"Not in the index: {row.get('status_reason') or 'its status is ' + status}.")
        elif row.get("status_reason"):
            out.append(f"Status: {row['status_reason']}.")
        if level:
            reasons = [str(r) for r in (row.get("level_reasons") or [])]
            out.append(f"Evidence level: {level}" + (", set by:" if reasons else ""))
            out += [f"  {r}" for r in reasons]
        out.append("")
        out.append(f"(its {'source' if function else 'text'} cannot be read: {missing})" if missing else body)
        out += [
            "",
            "Links to: " + (", ".join(self.linked(wanted, "links_to")) or "none"),
            "Linked from: " + (", ".join(self.linked(wanted, "linked_from")) or "none"),
            "Verification: " + _summary(row.get("verification")),
            "Use: " + _summary(row.get("use")),
        ]
        history = _history(row)
        if history:
            out.append("History (newest first):")
            out += [f"  {h}" for h in history]
            if row.get("history_more"):
                out.append(f"  … and {row['history_more']} earlier change(s)")
        else:
            out.append("History: none recorded")
        if not self.data.get("history_complete", True):
            out.append(f"(history covers the newest {self.data.get('history_scanned')} library commits only)")
        return "\n".join(out) + "\n"

    def _capture(self, iid: str) -> tuple[Item, str]:
        d = _capture_dir(iid[len(CAPTURE_PREFIX) :])
        rec, sig, saved = (_read_json(os.path.join(d, f)) for f in ("record.json", "signature.json", "saved.json"))
        rec, sig, saved = (x if isinstance(x, dict) else {} for x in (rec, sig, saved))
        text = _read_text(os.path.join(d, "procedure.py"))
        asked = _str(saved.get("asked")) or _str(sig.get("goal"))
        level = _str(rec.get("level"), "unchecked")
        level = _OLD_LEVELS.get(level, level)
        status = _str(rec.get("status"), _NEW_ITEM_STATUS)
        status = _OLD_STATUSES.get(status, status)
        built = Item(
            id=iid,
            kind="function",
            summary=_one_line(saved.get("docstring")) or _one_line(asked),
            applies=asked,
            evidence=level if level in _EVIDENCE else "unchecked",
            status=status if status in _STATUSES else _NEW_ITEM_STATUS,
            example="",
            text=text or "",
            links=(),
            history=(),
        )
        out = [
            f"{iid}  [{status}, evidence: {level}]",
            "A captured procedure: calling it runs it in a separate confined process on text or JSON values.",
        ]
        if rec.get("outcome") == "failed":
            out.append("Tried and failed: it is listed to show what broke, never run.")
            broke = saved.get("broke")
            out += [f"  broke: {b}" for b in (broke if isinstance(broke, list) else []) if isinstance(b, str)]
        if asked:
            out.append(f"Asked: {' '.join(asked.split())}")
        out += ["", (text or "(its procedure cannot be read)").rstrip("\n")]
        return built, "\n".join(out) + "\n"


@checked(name="memory.get")
def get(ref: str | Item | Callable) -> Item:
    """One item of the library, by any name that names exactly one: its full id, a function's plain name,
    ``module:name``, a note's slug or ``topic/slug``, a path under ``/memory``, the function itself, or an Item.

    Args:
        ref: the item's id or name (a str), the library function itself, or an Item.

    Returns:
        The Item. ``print()`` shows its card; a function item is called directly; ``.fits(inputs)`` checks it.

    Raises:
        ArgumentError: ref is not a str, a function or an Item.
        NotFound: no item has that name; the error gives the nearest ids and the topic to browse.
        Ambiguous: the name names several items; the error lists each exact id with its call.
        WrongCall: the name is a topic, not an item; the error gives the call that lists it.

    Example:
        memory.get('notes/grid/stencil.md')
    """
    call = call_text("memory.get", (ref,))
    return _guard(call, _get, ref, call)


def _get(ref: Any, call: str) -> Item:
    return _Catalog(call).resolve(ref)


@checked(name="memory.items")
def items(topic: str | None = None, kind: Kind | None = None) -> Items:
    """Browse the library: every item (but the retired), or one topic's, of one kind or both.

    Args:
        topic: a package (``'grid'``, ``'memory.grid'``), a module (``'grid.stencil'``), a notes topic
            (``'notes/grid'``; a bare ``'grid'`` also takes the notes topic of that name) or ``'captures'``;
            None for the whole library.
        kind: ``'function'`` or ``'note'``; None for both.

    Returns:
        Items: a sequence of Item. ``print()`` shows a compact table, one page at a time, and the call for the
        next page.

    Raises:
        ArgumentError: topic is not a str, or kind is not 'function', 'note' or None.
        NotFound: no topic has that name; the error gives the nearest topics.
        WrongCall: the name is an item's, not a topic's; the error gives the call that opens it.

    Example:
        memory.items('notes/grid', kind='note')
    """
    call = call_text(
        "memory.items",
        [topic] if topic is not None else [],
        {"kind": kind} if kind is not None else {},
    )
    return _guard(call, _items, topic, kind, call)


def _items(topic: str | None, kind: str | None, call: str) -> Items:
    cat = _Catalog(call)
    if topic is None:
        ids = cat.listed()
    else:
        found = cat.topic_ids(topic)
        if found is None:
            cat.not_a_topic(topic)
        ids = found or []
    if kind is not None:
        ids = [i for i in ids if (i.startswith("notes/") == (kind == "note"))]
    return Items(tuple(cat.item(i) for i in ids), topic, kind)


def _fit(target: str, inputs: Any) -> Fit:
    """``Item.fits``: the applicability check of one item, as ``match`` checks every procedure. It raises
    ``memory.match_result`` with the row it listed (none when it does not fit), as ``match`` does."""
    call = call_text(f"memory.get({target!r}).fits", (inputs,))
    return _guard(call, _fit_check, target, inputs, call)


def _fit_check(target: str, inputs: Any, call: str) -> Fit:
    if target.startswith("notes/"):
        raise WrongCall(
            call,
            "a note has no inputs to check; it is text to read.",
            hint=f"Read it with print(memory.get({target!r})).",
        )
    candidates, _ = _candidates_all()
    mine = [c for c in candidates if c.id == target]
    if not mine:
        _emit_event(MATCH_EVENT, (), ())
        return Fit(
            False,
            "no input it was saved against is recorded, so nothing here can say whether it fits; "
            f"print(memory.get({target!r})) shows what it takes",
        )
    m = _matcher()
    query = _query(m, None, inputs, mine)
    if not query.shapes:
        _emit_event(MATCH_EVENT, (), ())
        return Fit(
            False,
            "what you passed has no structure to compare (plain text has none); pass the data you were given",
        )
    try:
        pairs = list(MATCHER(query, list(mine)) or ())
    except Exception as exc:  # noqa: BLE001 - the matcher's failure is reported, not raised
        _emit_event(MATCH_EVENT, (), ())
        return Fit(False, f"the matcher failed ({type(exc).__name__}), so nothing here says whether it fits")
    how = None
    for pair in pairs:
        try:
            cid, matched = pair
        except (TypeError, ValueError):
            continue
        if cid == target:
            how = str(matched)[:40]
            break
    if how is None:
        _emit_event(MATCH_EVENT, (), ())
        return Fit(False, "the input it was saved against has a different structure from what you passed")
    row = _match_row(mine[0], how)
    if mine[0].failed:
        _emit_event(MATCH_EVENT, (), _event_rows([row]))
        return Fit(False, "it was tried on input like this and failed; it is listed to show what broke, never run")
    _emit_event(MATCH_EVENT, _event_rows([row]), ())
    return Fit(
        True,
        f"the input it was saved against has the structure of what you passed ({how}); stated conditions: "
        f"{row.conditions}. A structural match can be wrong.",
    )


def _run_target(target: str, /, *args: Any, **kwargs: Any) -> RunReport:
    """A function item's call, ``item(...)``: ``run``'s tracked path (:func:`_tracked_run`). In the package it raises
    ``memory.helper_call`` as ``memory:run`` once per call, ``ok`` whatever the item did (its outcome is in
    ``memory.run_result``), as ``run`` does; a note's call is refused here."""
    if target.startswith("notes/"):
        raise WrongCall(
            call_text(f"memory.get({target!r})", args, kwargs),
            "a note cannot be called; it is text to read.",
            hint=f"Read it with print(memory.get({target!r})).",
        )
    return _tracked_run(target, args, kwargs)


#: An item's applicability check and its tracked call; in the package each raises ``memory.helper_call`` (as
#: ``memory:match`` and ``memory:run``), bound at the end of this file.
_FIT: Any = _fit
_RUN_TARGET: Any = _run_target


# --- the earlier calls, kept for one build --------------------------------------------------------------------------

#: The one line each earlier call prints, to the cell's error output beside what the cell prints, naming its
#: replacement.
ALIAS_LINES = {
    "index": "memory.index is now memory.items(topic); print(memory.items(...))",
    "show": "memory.show is now memory.get(ref); print(memory.get(...))",
    "find": "memory.find is now memory.get(ref).fits(inputs); browse with print(memory.items())",
    "match": "memory.match is now memory.get(ref).fits(inputs); browse with print(memory.items())",
    "run": "memory.run is now a call of the item: memory.get(ref)(...)",
}


def _alias(name: str) -> None:
    line = ALIAS_LINES[name]
    if name in ("find", "match") and callable(globals().get("ask")):
        line += ", or ask with memory.ask(request)"
    try:
        print(line, file=sys.stderr)
    except Exception:  # noqa: BLE001 - the note never changes the call
        pass


@checked(name="memory.index")
def index(package: str | None = None) -> str:
    """The index (``INDEX.md``), one package's or notes topic's section in full, or the card of an item named
    there. Kept for one build: ``memory.items(topic)`` replaces it and ``memory.get(ref)`` opens one item; each
    call prints one line saying so.

    Args:
        package: a package (``'text'``, ``'memory.text'``), a notes topic (``'notes/text'``, or ``'text'`` when no
            package has that name), or an item's id or name; None for the whole index.

    Returns:
        The text: the whole index, one section, or one item's card.

    Raises:
        ArgumentError: package is not a str.
        NotFound: no section or item has that name; the error gives the nearest topics.
        Ambiguous: the name names several items; the error lists each exact id with its call.

    Example:
        memory.index('text')
    """
    _alias("index")
    call = call_text("memory.index", [package] if package is not None else [])
    return _guard(call, _index_text, package, call)


def _index_text(package: str | None, call: str) -> str:
    """``index``'s text: the index, a section (:func:`find_section`), or the card of the item a name names; else the
    error naming the nearest of the index's sections."""
    text = _read(INDEX_FILE)
    if package is None:
        return _Text(text)
    try:
        return _Text(find_section(text, package))
    except LookupError:
        pass
    try:
        cat: _Catalog | None = _Catalog(call)
    except _FileUnreadable:
        cat = None  # a copy without its items: the index's own sections still name the topics
    if cat is not None and cat.matches(package):
        return _Text(str(cat.resolve(package)))
    topics = [section_key(h) for h, _ in split_index(text)[1]]
    near = nearest(package, topics)
    raise NotFound(
        call,
        f"no section or item named {package!r}.",
        expected=_TOPIC_EXPECTED,
        example=f"memory.items({(near or topics or ['notes/grid'])[0]!r})",
        nearest=near,
        hint="Browse every topic with memory.items(); memory.get(ref) opens one item.",
    )


@checked(name="memory.show")
def show(item: Any) -> str:
    """One item's full card, as ``print(memory.get(ref))`` shows it. Kept for one build: ``memory.get(ref)``
    replaces it; each call prints one line saying so.

    Args:
        item: anything ``memory.get`` takes: an item's id or name, the library function itself, or an Item.

    Returns:
        The card's text.

    Raises:
        ArgumentError: item is not a str, a function or an Item.
        NotFound: no item has that name; the error gives the nearest ids.
        Ambiguous: the name names several items; the error lists each exact id with its call.
        WrongCall: the name is a topic, not an item.

    Example:
        memory.show('notes/grid/stencil.md')
    """
    _alias("show")
    call = call_text("memory.show", (item,))
    return _guard(call, _show_text, item, call)


def _show_text(item: Any, call: str) -> str:
    return _Text(str(_Catalog(call).resolve(item)))


@checked(name="memory.find")
def find(value: Any) -> list[Found]:
    """The functions whose tests call them on an argument with the shape of *value*, best match first, each with the
    parameter that argument was passed as. Kept for one build: an item's own check, ``memory.get(ref).fits(inputs)``,
    replaces it; each call prints one line saying so.

    Args:
        value: structured data: a dict, a list of records, JSON text, or a structured file such as a CSV with a
            header, or its path. Plain text has no structure and finds nothing.

    Returns:
        At most five Found rows (name, signature, param, match, summary, reason); an empty list when nothing
        matches.

    Raises:
        NotFound: the library copy cannot be read.

    Example:
        memory.find({'date': '2026-10-09', 'zone': 'UTC'})
    """
    _alias("find")
    return _guard(call_text("memory.find", (value,)), _find, value)


@checked(name="memory.match")
def match(request: Any = None, inputs: Any = None) -> Matches:
    """The stored procedures whose recorded input fits what you pass: *request*, the request's text, and *inputs*,
    data you were given. Kept for one build: an item's own check, ``memory.get(ref).fits(inputs)``, replaces it;
    each call prints one line saying so. Nothing runs.

    Args:
        request: the request's text (read as JSON only).
        inputs: data you were given: a dict, a list, JSON text, or a structured file or its path.

    Returns:
        Matches: the rows that fit (id, level, status, how it matched, relations, conditions, goal), with the
        captured attempts that failed on input like this under ``.failed``.

    Raises:
        NotFound: the library copy cannot be read.

    Example:
        memory.match('Total the table by month.', inputs={'rows': [[1, 2, 3]]})
    """
    _alias("match")
    call = call_text(
        "memory.match",
        [request] if request is not None else [],
        {"inputs": inputs} if inputs is not None else {},
    )
    return _guard(call, _match_all, request, inputs)


@checked(name="memory.run")
def run(item: Any, /, *args: Any, **kwargs: Any) -> RunReport:
    """Try one stored procedure on the arguments you pass and report how it went: whether it returned, its result,
    whether a second run agreed, whether its stated conditions held, which other matching procedures gave the same
    result, and its evidence level and status. Kept for one build: calling the item, ``memory.get(ref)(...)``,
    replaces it; each call prints one line saying so. A failure is reported, never raised.

    Args:
        item: the procedure: anything ``memory.get`` takes, a ``capture:`` id, or a row ``match`` returned.
        args: the procedure's own positional arguments.
        kwargs: the procedure's own keyword arguments.

    Returns:
        RunReport: outcome (returned, refused or error), result, repeatable, conditions, agrees_with, level and
        status.

    Raises:
        ArgumentError: no procedure is given.

    Example:
        memory.run('memory.text.dates:parse_date', '2026-10-09')
    """
    _alias("run")
    return _tracked_run(_run_target_of(item), args, kwargs)


def _run_target_of(item: Any) -> str:
    """The id ``run`` tries: a row's or an Item's own, a capture id, the one item a name names, else the name as
    given (whose run then reports that no stored procedure has it)."""
    if isinstance(item, (MatchRow, Item)):
        return item.id
    if isinstance(item, str) and not item.strip().startswith(CAPTURE_PREFIX):
        try:
            found = _Catalog("memory.run").matches(item)
        except Exception:  # noqa: BLE001 - an unreadable copy is reported by the run itself
            found = []
        if len(found) == 1:
            return found[0]
    return _item_id(item)


# --- runtime call evidence ------------------------------------------------------------------------------
#
# Static call sites in cell code say what the actor wrote, not what ran. The wrapper below lets the process
# that runs the cells observe each call that actually happened and how it ended. It changes no behaviour:
# the wrapped function receives the same arguments, its return value and exceptions pass through unchanged,
# and a failing audit hook is ignored.

_DIGEST_DEPTH = 3
_DIGEST_ITEMS = 16
_DIGEST_TEXT = 64
_DIGEST_SET = 256
_ATOMS = (type(None), bool, int, float, complex)


def _shape_text(value: Any, depth: int = 0) -> str:
    """A bounded text of *value* built without running any of its own code (no ``__repr__``, ``__len__`` or
    ``__iter__`` of user types): plain values by type and content prefix, containers by type, length and
    their first items (a set's sorted, small sets only, since its order differs between processes),
    anything else by its type name only."""
    t = type(value)
    if t in _ATOMS:
        if t is int and value.bit_length() > 256:
            return f"int:{value.bit_length()}bits"
        return f"{t.__name__}:{value!r}"
    if t is str or t is bytes:
        return f"{t.__name__}[{len(value)}]:{value[:_DIGEST_TEXT]!r}"
    if t not in (list, tuple, set, frozenset, dict):
        return f"<{t.__module__}.{t.__qualname__}>"
    head = f"{t.__name__}[{len(value)}]"
    if depth >= _DIGEST_DEPTH:
        return head
    if t is dict:
        parts = [
            f"{_shape_text(k, depth + 1)}={_shape_text(v, depth + 1)}"
            for k, v in list(dict.items(value))[:_DIGEST_ITEMS]
        ]
    elif t is list or t is tuple:
        parts = [_shape_text(v, depth + 1) for v in value[:_DIGEST_ITEMS]]
    elif len(value) <= _DIGEST_SET:
        parts = sorted(_shape_text(v, depth + 1) for v in value)[:_DIGEST_ITEMS]
    else:
        return head
    return f"{head}({','.join(parts)})"


def args_digest(args: tuple, kwargs: dict) -> str:
    """A short sha256 of the call's arguments as :func:`_shape_text` sees them: equal arguments give equal
    digests, and no argument value appears in it."""
    try:
        text = (
            _shape_text(tuple(args)) + "|" + _shape_text(dict(sorted(kwargs.items())))
        )
    except Exception:  # noqa: BLE001 - a digest never fails a call
        text = "unreadable"
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[
        :DIGEST_CHARS
    ]


#: Exact builtin types whose size :func:`args_shape` shows (their own ``len``, never a user type's ``__len__``).
_SIZED = (str, bytes, bytearray, list, tuple, dict, set, frozenset)
_SHAPE_ARGS = 8


def _type_size(value: Any) -> str:
    t = type(value)
    if t in _SIZED:
        return f"{t.__name__}[{len(value)}]"
    return t.__name__


def args_shape(args: tuple, kwargs: dict) -> str:
    """The call's arguments as types and sizes only, never values: ``(dict[3], str[12]; limit=int)``."""
    try:
        parts = [_type_size(v) for v in tuple(args)[:_SHAPE_ARGS]]
        if len(args) > _SHAPE_ARGS:
            parts.append(f"+{len(args) - _SHAPE_ARGS}")
        named = [
            f"{k}={_type_size(v)}"
            for k, v in list(kwargs.items())[:_SHAPE_ARGS]
            if type(k) is str and k.isidentifier()
        ]
        text = "(" + ", ".join(parts) + ("; " + ", ".join(named) if named else "") + ")"
    except Exception:  # noqa: BLE001 - a shape never fails a call
        text = "(unreadable)"
    return text[:FAILURE_TEXT]


def _first_line(exc: BaseException) -> str:
    """The first non-blank line of *exc*'s message, at most :data:`FAILURE_TEXT` characters."""
    try:
        text = str(exc)
    except Exception:  # noqa: BLE001 - an unprintable message is empty
        return ""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:FAILURE_TEXT]
    return ""


def call_outcome(exc: BaseException) -> str:
    """``refused`` when *exc* is a ``MemoryInputError`` (any class of that name in its type's bases: each
    library module may define its own), else ``error``."""
    return (
        "refused" if any(c.__name__ == REFUSAL for c in type(exc).__mro__) else "error"
    )


#: While an agreement check runs another procedure (:func:`_agreement`), its library calls raise no event: the
#: check is not a use.
_QUIET = threading.local()


def _emit(item: str, digest: str, outcome: str, event: str = CALL_EVENT) -> None:
    if getattr(_QUIET, "depth", 0):
        return
    try:
        sys.audit(event, item, digest, outcome)
    except Exception:  # noqa: BLE001 - an audit hook never changes the call
        pass


def _emit_failed(item: str, exc: BaseException, args: tuple, kwargs: dict, event: str) -> None:
    try:
        shape = args_shape(args, kwargs) if event == HELPER_EVENT else ""
        sys.audit(FAILED_EVENT, item, type(exc).__name__, _first_line(exc), shape)
    except Exception:  # noqa: BLE001 - an audit hook never changes the call
        pass


def wrap_call(fn: Any, item: str, event: str = CALL_EVENT) -> Any:
    """*fn* wrapped to raise one *event* (``memory.call``) per call (module docstring). A call interrupted by
    something that is not an ``Exception`` (an interrupt, an exit, a cancellation) did not end on its own,
    so it raises no event."""
    if getattr(fn, "__memory_item__", None) is not None:
        return fn
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def call(*args: Any, **kwargs: Any) -> Any:
            digest = args_digest(args, kwargs)
            try:
                result = await fn(*args, **kwargs)
            except Exception as exc:
                _emit(item, digest, call_outcome(exc), event)
                _emit_failed(item, exc, args, kwargs, event)
                raise
            _emit(item, digest, "ok", event)
            return result

    else:

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            digest = args_digest(args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                _emit(item, digest, call_outcome(exc), event)
                _emit_failed(item, exc, args, kwargs, event)
                raise
            _emit(item, digest, "ok", event)
            return result

    call.__memory_item__ = item  # type: ignore[attr-defined]
    return call


def wrap_module(module: Any) -> None:
    """Wrap each public function *module* defines itself (not one it imports, which its own module wraps;
    not a class or other callable). A library test module is left as it is."""
    name = module.__name__
    if "tests" in name.split(".")[1:]:
        return
    for attr, value in list(vars(module).items()):
        if (
            not attr.startswith("_")
            and inspect.isfunction(value)
            and getattr(value, "__module__", None) == name
        ):
            setattr(module, attr, wrap_call(value, f"{name}:{attr}"))


class _CallLoader:
    """A library module's own loader that wraps the module's functions once it has run."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec: Any) -> Any:
        return self._inner.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._inner.exec_module(module)
        try:
            wrap_module(module)
        except Exception:  # noqa: BLE001 - an unwrapped module still works
            pass

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


class _CallFinder:
    """Finds the modules under this package the normal way and gives each a :class:`_CallLoader`."""

    memory_call_finder = True

    def __init__(self, package: str) -> None:
        self.prefix = package + "."

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        # a library's own tests are left to the other finders (a test runner's import hooks among them)
        if not fullname.startswith(self.prefix) or "tests" in fullname.split(".")[1:]:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        # source modules only: a namespace package or a compiled extension keeps its own loader
        if (
            spec is not None
            and isinstance(spec.origin, str)
            and spec.origin.endswith(".py")
            and hasattr(spec.loader, "exec_module")
        ):
            spec.loader = _CallLoader(spec.loader)
        return spec


def _install_call_finder(package: str) -> None:
    if not any(
        getattr(f, "memory_call_finder", False)
        and getattr(f, "prefix", None) == package + "."
        for f in sys.meta_path
    ):
        sys.meta_path.insert(0, _CallFinder(package))


# Only as the library package itself: the harness also imports this file as a module of its own.
# The package's own calls each raise ``memory.helper_call`` once (module docstring): the lookups, the aliases, and an
# item's tracked call and applicability check, under the ids of the calls they replace.
if __name__ == "memory":
    _install_call_finder(__name__)
    get = wrap_call(get, "memory:get", HELPER_EVENT)
    items = wrap_call(items, "memory:items", HELPER_EVENT)
    index = wrap_call(index, "memory:index", HELPER_EVENT)
    show = wrap_call(show, "memory:show", HELPER_EVENT)
    find = wrap_call(find, "memory:find", HELPER_EVENT)
    match = wrap_call(match, "memory:match", HELPER_EVENT)
    run = wrap_call(run, "memory:run", HELPER_EVENT)
    _RUN_TARGET = wrap_call(_run_target, "memory:run", HELPER_EVENT)
    _FIT = wrap_call(_fit, "memory:match", HELPER_EVENT)
