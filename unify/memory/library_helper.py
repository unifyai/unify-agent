"""The memory library: four calls and one item type.

    memory.ask(question=None) -> list[Item]         # the librarian's answer for the current request (or your
                                                    # question): the best fit and at most one fallback for when
                                                    # it fails, each with .reason; [] when nothing fits
    memory.items(topic=None, kind=None) -> Items    # browse; print() shows a compact table, page by page
    memory.get(ref) -> Item                         # one item, by any name that names exactly one
    memory.request() -> str                         # the current request, as a procedure's recipe takes it

An :class:`Item` is a frozen record of one function, note or recorded procedure: ``id``, ``kind``, ``summary``,
``applies``, ``evidence``, ``status``, ``example``, ``text``, ``links``, ``history`` (and ``reason``, from ``ask``).
``print(item)`` shows its card, a bounded summary with no code; ``item.text`` holds a function's source, a recorded
procedure's code or a note's Rule. ``item.fits(inputs)`` is its own applicability check (a :class:`Fit`); a function
item is called directly, ``item(*args)``: the call is tracked, and what the function raises reaches the caller
unchanged; calling a note raises an error that says what to do instead. A procedure with a recipe (one function that
applies it to a request where one call can do the job) shows on its card the one call that applies it to the current
request; nothing runs it until that call is made. A recorded procedure that acts through its episode's tools
(``run(state, env)``) is not run by any call: its card says how its work is done from here. Library functions also
import as ordinary modules: ``from memory.<package>.<module> import <name>``.

Every call checks its arguments' types first. The package's errors, ``memory.NotFound``, ``memory.Ambiguous``,
``memory.WrongCall`` and ``memory.ArgumentError`` (each a ``memory.ContractError``), name the call as written, what was
expected with an example call, and the nearest valid targets or the right call. The whole typed contract is
``memory/__init__.pyi`` beside this file.
"""

# How the package works (for the harness's readers; not part of what help(memory) shows):
#
# The harness generates this file as ``memory/__init__.py`` in every copy of the library; edits are discarded. It
# imports only the standard library and its sibling ``memory/_contract.py``, and reads only files beside the package:
# ``INDEX.md``, ``links.json`` and the harness's data under ``.memory/``.
#
# Imported as the package ``memory``, it also wraps each public function of every library module as that module is
# imported (:class:`_CallFinder`), so that each call raises the audit event ``memory.call`` with the item id, a short
# digest of the arguments (never their values) and the outcome: ``ok``, ``refused`` (a ``MemoryInputError`` left it)
# or ``error`` (any other exception). The wrapper returns what the function returns and re-raises what it raises,
# unchanged; whatever audit hooks the process has decide what is kept. The package's own calls raise
# ``memory.helper_call`` the same way: ``memory:get``, ``memory:items`` and ``memory:ask`` (lookups and questions), a
# function item's call as ``memory:run`` and ``Item.fits`` as ``memory:match``.
#
# ``Item.fits`` reads what the stored procedure was saved against (a landed procedure's row in ``.memory/find.json``'s
# ``procedures``, a recorded procedure's files under ``.memory/captures/``) and compares its structure with what was
# passed; it runs nothing (no reader and no ``applies`` runs to recognise or rank a procedure). It raises
# ``memory.match_result`` with the row it found. A tracked call (calling a function item) also raises
# ``memory.run_result`` with its target and outcome. Neither event carries a value or a result. A call that raised
# also raises ``memory.call_failed`` with the id, the exception's type, the first line of its message and, for the
# package's own calls, the arguments' shape (types and sizes, never values).
#
# With the harness's evidence beside the package (``.memory/item_evidence.json``), each card also states what the
# item's status rests on, its recorded failures, its later uses and its stated conditions; ``Item.fits`` gives a
# :class:`FitEvidence` (structurally similar or not, with the item's recorded failures: evidence, never a claim that
# it does what is needed); and a library item the writer retired or superseded (``.memory/withdrawn_items.json``) is
# withdrawn as a capture is: no listing shows it, and asking for it by its id gives what replaced it.

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
#: The audit event of a call of this package's own helpers (``get``, ``items``, ``ask``): the same arguments
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
    "verified": "unchecked",
    "reproduces-its-case": "unchecked",
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


def item_count(body: str) -> int:
    """The item lines of a section body."""
    return sum(1 for line in body.splitlines() if line.startswith(ITEM_LINE))


#: The contract: what ``help(memory)`` and ``from memory import *`` show.
#: The errors (``Ambiguous``, ``ArgumentError``, ``ContractError``, ``NotFound``, ``WrongCall``) stay names of the
#: package, ``memory.NotFound`` and the rest, but are left out here: the module's documentation names them in one line
#: rather than printing each one's inherited exception machinery, so ``help(memory)`` stays short.
__all__ = [
    "Fit",
    "Item",
    "Items",
    "get",
    "items",
    "request",
]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(
    _PKG_DIR,
)  # the copy's root: INDEX.md, links.json, memory/, notes/, .memory/


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


def _summary(record: Any) -> str:
    if not record:
        return "no record yet"
    return ", ".join(f"{k} {record[k]}" for k in sorted(record))


def _matcher() -> Any:
    """The shape comparison shipped beside the package (``.memory/matcher.py``), loaded once per process. Its
    ``__file__`` is set at the root, so it loads its shape functions from ``.memory/shapes.py``.
    """
    loaded = _matcher.__dict__.get("module")
    if loaded is None:
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
        _matcher.__dict__["module"] = loaded = module
    return loaded


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


# --- an item's applicability check (``Item.fits``) and its tracked call ----------------------------------------------
#
# ``Item.fits`` informs and runs no procedure: it reads what the stored procedure was saved against and compares its
# structure with what the caller passed (no reader or ``applies`` runs to recognise or rank anything). Calling an item
# is the agent's own call of one procedure, in its own cell; nothing here runs on its own. Everything read here was
# built by the harness as written (signatures, docstrings, summary sections, what broke); nothing strips or masks an
# identifier.

#: Captured attempts the harness exported beside the package, one directory per capture id (written by the capture
#: store, never by this file): ``signature.json``, ``record.json``, ``procedure.py`` and ``saved.json`` (what the
#: attempt was saved against besides its signature: ``docstring``, ``asked``, ``approach`` and ``broke``).
CAPTURES_DIR = ".memory/captures"
CAPTURE_PREFIX = "capture:"
#: A captured attempt's outcomes that make it negative knowledge: never offered to run (``failed``; ``contrary``, a
#: signal contradicted the episode of a capture that records what its episode did).
_NEGATIVE = ("failed", "contrary")
#: Beside the capture folders, when the harness withdraws closed captures: each capture no longer offered (the
#: library's writer superseded or retired it), with what is offered in its place (``by``) and what it was saved
#: against. Written by the harness's ``captures.export_files``, never by this file.
WITHDRAWN_FILE = ".memory/captures/withdrawn.json"
#: Beside the package with the harness's evidence serving on: each item's
#: evidence (its recorded failures, contradictions, what its status rests on, its stated conditions), and the library
#: items the writer retired or superseded with what replaces each. Without the first file every call is as before.
EVIDENCE_FILE = ".memory/item_evidence.json"
WITHDRAWN_ITEMS_FILE = ".memory/withdrawn_items.json"
#: The audit events of an applicability check (its row's id, level, status and how it matched; a failed attempt apart)
#: and of a tracked call (its target, outcome, level and conditions).
#: Ids and short labels only: never a value, a result or a request's text.
MATCH_EVENT = "memory.match_result"
RUN_EVENT = "memory.run_result"
RUN_OUTCOMES = ("returned", "refused", "error")
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
    """One stored procedure as the matcher sees it: what it was saved against, as written."""

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
    """What an applicability check was given: the request text, the caller's inputs (unchanged), and the shapes the
    structural reading gives them (none for plain text)."""

    request: Any
    inputs: Any
    shapes: tuple


class MatchRow(NamedTuple):
    """One procedure an applicability check found similar in structure."""

    id: str
    source: str
    level: str
    status: str
    matched: str  # how the matcher says it matched (the structural default: exact | structure)
    relations: tuple
    conditions: str
    goal: str
    broke: tuple


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


def _withdrawn() -> dict[str, dict]:
    """The withdrawn captures (:data:`WITHDRAWN_FILE`) by ``capture:<id>``, and the withdrawn library items
    (:data:`WITHDRAWN_ITEMS_FILE`) by their ids; {} when the harness withdrew none."""
    doc = _read_json(os.path.join(_ROOT, WITHDRAWN_FILE))
    rows = doc.get("captures") if isinstance(doc, dict) else None
    out = {
        CAPTURE_PREFIX + k: v
        for k, v in (rows if isinstance(rows, dict) else {}).items()
        if isinstance(k, str) and _CAPTURE_ID.fullmatch(k) is not None and isinstance(v, dict)
    }
    doc = _read_json(os.path.join(_ROOT, WITHDRAWN_ITEMS_FILE))
    rows = doc.get("items") if isinstance(doc, dict) else None
    for k, v in (rows if isinstance(rows, dict) else {}).items():
        if isinstance(k, str) and not k.startswith(CAPTURE_PREFIX) and isinstance(v, dict):
            out[k] = v
    return out


def _evidence() -> dict | None:
    """Each item's evidence by id (:data:`EVIDENCE_FILE`), or None when the harness serves none (the file is
    absent): then every card, check and listing is as before."""
    doc = _read_json(os.path.join(_ROOT, EVIDENCE_FILE))
    rows = doc.get("items") if isinstance(doc, dict) else None
    return rows if isinstance(rows, dict) else None


def _failures(evidence: dict | None, iid: str) -> list[dict]:
    row = (evidence or {}).get(iid)
    rows = row.get("failure_record") if isinstance(row, dict) else None
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def _failure_text(rows: list[dict]) -> str:
    """Recorded failures on one line: ``none``, or the count and each one's why, episode and the library version
    that was served."""
    if not rows:
        return "none"
    parts = [
        f"{_str(r.get('why'), 'failed')} in episode {_str(r.get('episode'), '?')[:8]}"
        + (f" (version {_str(r.get('pin'))[:7]})" if _str(r.get("pin")) else "")
        for r in rows
    ]
    return f"{len(rows)}: " + "; ".join(parts)


def _evidence_lines(evidence: dict | None, iid: str) -> list[str]:
    """A card's evidence lines (with the harness's evidence serving on): what its status rests on, its recorded
    failures, its later uses (the calls made of it since it was stored, by how they ended) and its stated
    conditions; [] without it."""
    if evidence is None:
        return []
    row = evidence.get(iid) if isinstance(evidence.get(iid), dict) else {}
    out = [
        "Status by evidence: "
        + _str(row.get("status_text"), "no outcome signal has arrived for any use of this item"),
        "Recorded failures: " + _failure_text(_failures(evidence, iid)),
        "Later uses: " + _str(row.get("later_uses_text"), "none yet"),
    ]
    if _str(row.get("conditions")):
        out.append("States that it applies when: " + _str(row.get("conditions")))
    return out


def _count_text(counts: Any, names: tuple[str, ...]) -> str:
    """``3 as stated, 1 not as stated`` from *counts* (name -> int), naming only the non-zero ones; ``none``."""
    rows = counts if isinstance(counts, dict) else {}
    parts = [
        f"{rows[n]} {n.replace('_', ' ')}"
        for n in names
        if isinstance(rows.get(n), int) and not isinstance(rows.get(n), bool) and rows[n] > 0
    ]
    return ", ".join(parts) or "none"


def _label_lines(label: Any, served_evidence: bool) -> list[str]:
    """A card's lines from the item's evidence label (the harness's record of what it is and what its tests and uses
    showed; never a filter): its form and declared effects, the episodes it covers,
    its inputs and constants, its stated tests and its uses since this version; its failure record and what its
    status rests on only when the harness's evidence lines do not already state them (*served_evidence*). [] without
    a label."""
    if not isinstance(label, dict):
        return []
    effects = label.get("effects") if isinstance(label.get("effects"), dict) else {}
    out = [
        f"Form: {_str(label.get('form'), 'function')}; effects: {_str(effects.get('declares'), 'unknown')}"
        + ("" if effects.get("declared") else " (not declared)"),
    ]
    if isinstance(label.get("situation"), str) and label["situation"]:
        out.append(f"For: {label['situation']}")
    covers = label.get("covers") if isinstance(label.get("covers"), dict) else {}
    episodes = [e for e in covers.get("episodes") or [] if isinstance(e, str)]
    if episodes:
        out.append("Built from episodes (its evidence): " + ", ".join(e[:16] for e in episodes))
    ivc = label.get("inputs_vs_constants") if isinstance(label.get("inputs_vs_constants"), dict) else {}
    inputs = [str(i) for i in ivc.get("inputs") or []]
    constants = [
        f"{c.get('value')} (source: {c.get('source')})" if isinstance(c, dict) else str(c)
        for c in ivc.get("constants") or []
    ]
    if inputs or constants:
        out.append(f"Inputs: {', '.join(inputs) or 'none stated'}; constants: {', '.join(constants) or 'none stated'}")
    uses = label.get("uses") if isinstance(label.get("uses"), dict) else {}
    out.append("Uses of this version: " + _count_text(uses, ("clean", "failed", "neutral")))
    if not served_evidence:
        rows = [r for r in label.get("failure_record") or [] if isinstance(r, dict)]
        out.append("Recorded failures: " + _failure_text(rows))
        out.append("Status rests on: " + _str(label.get("status_basis"), "no_outcome_signals").replace("_", " "))
    return out


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


#: Where a writer pass is handed its captures (its own box's inputs), so ``memory.get('capture:<id>')`` opens a
#: handed capture there as it opens an exported one in a request's copy.
HANDED_CAPTURES = "/inputs/captures"


def _capture_dir(capture_id: str) -> str:
    exported = os.path.join(_ROOT, CAPTURES_DIR, capture_id)
    if os.path.isdir(exported):
        return exported
    handed = os.path.join(HANDED_CAPTURES, capture_id)
    return handed if os.path.isdir(handed) else exported


def _capture_candidates() -> tuple[list[Candidate], int]:
    """The exported captures, in id order, and how many could not be read (no ``signature.json`` object)."""
    base = os.path.join(_ROOT, CAPTURES_DIR)
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return [], 0
    out: list[Candidate] = []
    unreadable = 0
    evidence = _evidence() or {}
    for name in names:
        if _CAPTURE_ID.fullmatch(name) is None:
            continue
        d = os.path.join(base, name)
        row = evidence.get(CAPTURE_PREFIX + name)
        row = row if isinstance(row, dict) else {}
        sig = _read_json(os.path.join(d, "signature.json"))
        if not isinstance(sig, dict):
            unreadable += 1
            continue
        rec = _read_json(os.path.join(d, "record.json"))
        rec = rec if isinstance(rec, dict) else {}
        saved = _read_json(os.path.join(d, "saved.json"))
        saved = saved if isinstance(saved, dict) else {}
        failed = rec.get("outcome") in _NEGATIVE
        broke = saved.get("broke") if isinstance(saved.get("broke"), list) else []
        out.append(
            Candidate(
                id=CAPTURE_PREFIX + name,
                source="capture",
                signature={k: v for k, v in sig.items() if k != "goal"},
                docstring=_str(saved.get("docstring")),
                asked=_str(saved.get("asked")) or _str(sig.get("goal")),
                approach=_str(saved.get("approach")),
                # what its uses gave it (the harness's evidence), never its record's
                level=_str(row.get("level"), "unchecked"),
                status=_str(row.get("status"), "on-trial"),
                failed=failed,
                broke=tuple(str(b) for b in broke if isinstance(b, str)) if failed else (),
            ),
        )
    return out, unreadable


def _in_place(candidates: list[Candidate]) -> list[Candidate]:
    """For each withdrawn capture (:data:`WITHDRAWN_FILE`) whose replacement is one of *candidates*, that
    replacement once more, saved against what the withdrawn capture was saved against: it is matched where the
    capture would have been, and listed, run and recorded as itself."""
    by_id = {c.id: c for c in candidates}
    out: list[Candidate] = []
    for _cid, row in sorted(_withdrawn().items()):
        own = by_id.get(row.get("by")) if isinstance(row.get("by"), str) else None
        sig = row.get("signature")
        if own is None or not isinstance(sig, dict):
            continue
        out.append(
            own._replace(
                signature={k: v for k, v in sig.items() if k != "goal"},
                docstring=_str(row.get("docstring")),
                asked=_str(row.get("asked")) or _str(sig.get("goal")),
                approach=_str(row.get("approach")),
            ),
        )
    return out


def _candidates_all() -> tuple[list[Candidate], int]:
    captures, unreadable = _capture_candidates()
    found = _library_candidates() + captures
    return found + _in_place(found), unreadable


def _query(m: Any, request: Any, inputs: Any, candidates: list[Candidate]) -> Query:
    """The query for *request* (read as JSON text only: sniffing prose as a table would make its first line's words
    column names) and *inputs* (a parsed value, JSON text, a structured file or its path)."""
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


def _structure_match(query: Query, candidate: Candidate) -> str:
    """How the structure of what was passed matches the input *candidate* was saved against, never by words:
    ``exact`` (one of the query's shapes has the same shape as its signature's ``input``, or matches it at the exact
    level), ``structure`` (a structure-level match), or '' (no match, no shapes or no recorded input). The docstring
    and the summary sections are not compared, and nothing is run."""
    recorded = candidate.signature.get("input") if isinstance(candidate.signature, dict) else None
    if not query.shapes or not isinstance(recorded, dict):
        return ""
    m = _matcher()
    best = ""
    for mine in query.shapes:
        try:
            if m.same_shape(mine, recorded):
                return "exact"
            found = m.match(mine, recorded)
        except Exception:  # noqa: BLE001 - a malformed signature matches nothing
            continue
        if found is not None and found[0] >= 1:
            best = "exact" if found[0] == 2 else (best or "structure")
    return best


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


# --- the current request, and the harness's catalogue ---------------------------------------------------------------
#
# the harness writes ``.memory/dispatch.json``: the structural catalogue, one row
# per stored procedure from its syntax and records. Nothing here runs a reader or an ``applies`` on it.
#
# The current request reaches this file from the host tool :data:`REQUEST_TOOL`, called through the channel client
# (``tools._call``): no name for it is bound in a cell, and its text crosses only as data.

DISPATCH_FILE = ".memory/dispatch.json"
#: How every reader is handed the request (the capture contract, ``request_value.REQUEST_CLASS`` on the harness side,
#: byte for byte): its text, a str, with ``.text`` and ``.data``, the JSON objects and arrays it contains, in order.
_REQUEST_CLASS = '''
class Request(str):
    """The request: its text (a str), with .text and .data (the JSON objects and arrays it contains, in order)."""

    def __new__(cls, text, data=None):
        obj = super().__new__(cls, text)
        obj._data = data
        return obj

    @property
    def text(self):
        return str.__str__(self)

    @property
    def data(self):
        return self._data
'''
_JSON_DECODER = json.JSONDecoder()
_JSON_OPEN = re.compile(r"[\[{]")


def _json_values(text: str) -> list:
    """Every top-level JSON object and array in *text*, parsed, in order: the bound request's ``data``
    (``worker_child.json_values`` on the harness side, the same scan line for line)."""
    values: list = []
    index = 0
    while True:
        found = _JSON_OPEN.search(text, index)
        if found is None:
            return values
        start = found.start()
        try:
            value, end = _JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError as exc:
            index = max(exc.pos, start + 1)
            continue
        except RecursionError:
            return values
        values.append(value)
        index = end


#: The host tool that gives the current request, as JSON text ``{"text": ..., "data": ...}``.
REQUEST_TOOL = "current_request"
def _tools() -> Any:
    """The box module ``tools`` (the agent's host tools), or None where there is none."""
    tools = sys.modules.get("tools")
    if tools is None:  # the box imports it before any cell; elsewhere it may be importable but not yet imported
        try:
            tools = importlib.import_module("tools")
        except ImportError:
            tools = None
    return tools


def _current_request() -> tuple[str | None, Any]:
    """The current request's text, and its data (the JSON objects and arrays the text contains, in order), from the
    host tool :data:`REQUEST_TOOL` through the channel client; ``(None, None)`` when it is not available here (no
    channel, a refusal, a reply that cannot be read). Never raises."""
    call = getattr(_tools(), "_call", None)
    if not callable(call):
        return None, None
    try:
        reply = json.loads(str(call(REQUEST_TOOL, {})))
    except Exception:  # noqa: BLE001 - a refusal or a closed channel: no request here
        return None, None
    text = reply.get("text") if isinstance(reply, dict) else None
    if not isinstance(text, str) or not text:
        return None, None
    return text, reply.get("data")


#: The call that applies a recipe to the current request (the harness's ``item_records.RECIPE_CALL``, the same text).
_RECIPE_CALL = "memory.get({item!r})(memory.request())"


def _recipe_call(recipe: str) -> str:
    """The one-call entry of the recipe *recipe*: the recipe applied to the current request."""
    return _RECIPE_CALL.format(item=recipe)


_REQUEST_NAMES: dict = {}
exec(_REQUEST_CLASS, _REQUEST_NAMES)  # noqa: S102 - the class's one definition, a constant of this file
#: The request as every reader and recipe takes it: its text (a str), with ``.text`` and ``.data``.
Request = _REQUEST_NAMES["Request"]


@checked(name="memory.request")
def request() -> str:
    """The request you are working on (in a teammate's cells, the request the actor works on, which the teammate's
    work serves), as a stored procedure's recipe and reader take it: its text (a str), with ``.text`` and ``.data``
    (the JSON objects and arrays the text contains, in order). A procedure's card shows the one call that applies its
    recipe to it.

    Returns:
        The current request: a str with ``.text`` and ``.data``.

    Raises:
        NotFound: the current request is not available in these cells.

    Example:
        memory.request().data
    """
    text, data = _current_request()
    if text is None:
        raise NotFound(
            "memory.request()",
            "the current request is not available in these cells.",
            hint="Pass the request's text to the item yourself.",
        )
    return Request(text, data if isinstance(data, list) else _json_values(text))


def _event_rows(rows: list[MatchRow]) -> tuple:
    return tuple((r.id, _label(r.level, "unknown"), _label(r.status, "unknown"), _label(r.matched)) for r in rows)


class RunReport:
    """How one tracked call went: the one call it made (a stored procedure is called once, never a second time and
    never beside another procedure). ``outcome``: returned, refused (``MemoryInputError``) or error; ``result``: what
    it returned (None otherwise); ``conditions``: whether its stated conditions held."""

    def __init__(self, target: str, **kw: Any) -> None:
        self.target = target
        self.outcome: str = kw.get("outcome", "error")
        self.result: Any = kw.get("result")
        self.error: str = kw.get("error", "")
        self.detail: str = kw.get("detail", "")
        self.conditions: str = kw.get("conditions", "none stated")
        self.conditions_code: str = kw.get("conditions_code", "none")
        self.level: str = kw.get("level", "unknown")
        self.status: str = kw.get("status", "unknown")
        self.record: str = kw.get("record", "")
        # what a recorded procedure rests on, printed in place of the level and status a curated item shows
        self.rests_on: str = kw.get("rests_on", "")
        # the exception a library function raised in this process (None otherwise): an item call re-raises it, the
        # same object with its traceback, as a direct call would
        self._raised: BaseException | None = kw.get("raised")

    def __repr__(self) -> str:
        out = [f"run of {self.target}: {self.outcome}"]
        if self.outcome == "returned":
            out.append("result: " + _bounded_repr(self.result))
        else:
            out.append(f"error: {self.error}" + (f": {self.detail}" if self.detail else ""))
        out.append("conditions: " + self.conditions)
        out.append(f"rests on: {self.rests_on}" if self.rests_on else f"level {self.level}, status {self.status}")
        out.append(self.record or "no later use recorded yet")
        out.append("Its result is unchecked: nothing here compares it with what your request needs.")
        return "\n".join(out) + "\n"

    __str__ = __repr__


def _bounded_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 - a result that cannot be shown is named by its type
        return f"<{type(value).__name__}: repr failed ({type(exc).__name__})>"
    return text if len(text) <= _RESULT_CHARS else text[:_RESULT_CHARS] + " …"


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


class AsyncItemError(TypeError):
    """A tracked call of an ``async def`` item from a cell whose event loop is already running: it cannot be awaited
    from inside a synchronous call there. Import the item's function and await it instead."""


def _settled(result: Any) -> Any:
    """*result*, or, when it is a coroutine (an ``async def`` item), what it gives once awaited: run to completion
    here when no event loop runs in this thread; else closed, and :class:`AsyncItemError` raised (never the
    coroutine's text as a result)."""
    import asyncio
    import inspect

    if not inspect.iscoroutine(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    result.close()
    raise AsyncItemError(
        "this item is an async def and this cell's event loop is running: import its function from its module and "
        "await it instead"
    )


def _item_function(target: str) -> Any:
    import importlib

    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)


def _run_item(target: str, args: tuple, kwargs: dict) -> RunReport:
    """Call a library function by id in this process, once (an ordinary use: the wrapper raises ``memory.call``).
    It is never called a second time, so an item that acts on the world acts once."""
    gone = _withdrawn().get(target) if _evidence() is not None else None
    if gone is not None:
        by = gone.get("by")
        return RunReport(
            target,
            outcome="error",
            error="LookupError",
            detail=f"this item is no longer offered: the library's writer {_str(gone.get('status'), 'closed')} it"
            + (f"; {by} is offered in its place: memory.get({by!r})" if isinstance(by, str) and by
               else "; memory.ask() names what applies"),
        )
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
    try:
        result = _settled(fn(*args, **kwargs))
    except Exception as exc:  # noqa: BLE001 - reported, never raised past the cell
        return RunReport(target, outcome=call_outcome(exc), error=type(exc).__name__,
                         detail=str(exc)[:_DETAIL_CHARS], raised=exc, **base)
    return RunReport(target, outcome="returned", result=result, **base)


#: The child process that runs a captured procedure (stdlib only): it loads ``procedure.py``, reads the request text
#: through ``read_input`` when the procedure has one and it was given one text (handed over with its data, as every
#: reader is: :data:`_REQUEST_CLASS`), checks ``applies``, runs ``run`` once, checks ``holds``, and prints one JSON
#: report after a marker line.
_CAPTURE_RUNNER = r'''
import asyncio, inspect, json, sys
MARK = sys.argv[2]
out = {}
''' + _REQUEST_CLASS + r'''
class State:
    """The episode's state as a recorded procedure reads it: .request, .messages, .last and .data."""
    def __init__(self, request, messages, last, data):
        self.request, self.messages, self.last, self.data = request, messages, last, data
def settle(x):
    # an async def (run, read_input, applies or holds) gives a coroutine: it is awaited here, in the child, and its
    # value is what the call gave (never the coroutine's text)
    return asyncio.run(x) if inspect.iscoroutine(x) else x
def refused(exc):
    return any(c.__name__ == "MemoryInputError" for c in type(exc).__mro__)
try:
    payload = json.loads(sys.stdin.read())
    args, kwargs = payload["args"], payload["kwargs"]
    path = sys.argv[1]
    ns = {"__name__": "captured_procedure", "__file__": path}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), ns)
    run = ns.get("run")
    if not callable(run):
        raise LookupError("the captured procedure defines no run(value)")
    read = ns.get("read_input")
    text_only = len(args) == 1 and not kwargs and isinstance(args[0], str)
    if callable(read) and text_only:
        value = settle(read(Request(args[0], payload.get("data"))))
        call = lambda: settle(run(value))
    elif payload.get("form") == "state" and text_only:
        # a procedure of the episode's state, handed a request's text: the state of a request that has just begun
        req = Request(args[0], payload.get("data"))
        value = State(req, [req], req, list(payload.get("data") or []))
        call = lambda: settle(run(value))
    else:
        value = args[0] if len(args) == 1 and not kwargs else None
        call = lambda: settle(run(*args, **kwargs))
    applies = ns.get("applies")
    if callable(applies) and value is not None:
        try:
            out["applies"] = bool(settle(applies(value)))
        except Exception as exc:
            out["applies_error"] = type(exc).__name__
    first = call()
    out["outcome"] = "returned"
    holds = ns.get("holds")
    if callable(holds) and value is not None:
        try:
            out["holds"] = bool(settle(holds(value, first)))
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


def _confined(capture_id: str, args: tuple, kwargs: dict) -> dict:
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
        data = _json_values(args[0]) if len(args) == 1 and not kwargs and isinstance(args[0], str) else None
        payload = json.dumps({"args": list(args), "kwargs": kwargs, "data": data, "form": _capture_form(capture_id)})
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
    gone = _withdrawn().get(target)
    if gone is not None:
        by = gone.get("by")
        return RunReport(target, outcome="error", error="LookupError",
                         detail=f"this captured procedure is no longer offered: the library's writer "
                         f"{_str(gone.get('status'), 'closed')} it"
                         + (f"; {by} is offered in its place: memory.get({by!r})" if isinstance(by, str) and by
                            else "; memory.ask() names what applies"))
    rec, row = _capture_facts(capture_id)
    base = {"level": _str(row.get("level"), "unchecked"), "status": _str(row.get("status"), "on-trial")}
    if rec.get("outcome") in _NEGATIVE:
        return RunReport(target, outcome="error", error="LookupError",
                         detail="this captured attempt failed; it is listed to show what broke, not to run", **base)
    if _capture_form(capture_id) == "state_env":
        # nothing is run and no use is recorded: no call can pass it the tools it acts through
        return RunReport(target, outcome="error", error="WrongCall", detail=BY_HAND.format(cid=target), **base)
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
        conditions=conditions,
        conditions_code=code,
        record="later uses before this one: " + _str(row.get("later_uses_text"), "none recorded"),
        rests_on=_support_line("recorded procedure", _capture_support(rec, row)),
        **base,
    )


def _tracked_run(target: str, args: tuple, kwargs: dict) -> RunReport:
    """The tracked call of one stored procedure (calling a function :class:`Item`): try it on the
    arguments, in this cell, once, and report how it went: whether it returned, its result, whether its stated
    conditions held, and its evidence level and status. Nothing calls it a second time or calls another
    procedure beside it. *target* is an item id (``memory.<package>.<module>:<name>``) or a ``capture:<id>``. A
    library function is called here, as an import would call it; a captured procedure runs in a separate confined
    process and takes text or JSON values (given one text, it reads its input from that text). A failure is
    reported, never raised. Nothing runs unless the caller calls this."""
    if target.startswith(CAPTURE_PREFIX):
        report = _run_capture(target[len(CAPTURE_PREFIX) :], args, kwargs)
    else:
        report = _run_item(target, args, kwargs)
    _emit_event(
        RUN_EVENT,
        report.target,
        report.outcome,
        _label(report.level, "unknown"),
        report.conditions_code if report.conditions_code in CONDITIONS else "none",
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
#: Where the actor's and a teammate's box mounts the library copy (``unify.sandbox.LIBRARY_MOUNT``): the name their
#: texts give the library, which names the request's copy in every memory mode; a path under it names an item too.
SANDBOX_LIBRARY = "/memory-library"
_SUMMARY_CHARS = 100
_ID_COLUMN = 60
_GET_EXPECTED = "an item id, a function name, or a note slug"
_GET_EXAMPLE = "memory.get('notes/<topic>/<name>.md')"
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


def _support_line(kind: str, facts: Any) -> str:
    """What an item rests on, in words (every card shows it): its kind
    (``curated item`` or ``recorded procedure``), the episodes it comes from and, when it has any,
    the uses whose outcome confirmed it. *facts* is the export's ``support`` (``{"episodes", "confirmed_uses"}``)."""
    if not isinstance(facts, dict):
        return f"{kind}, no record"  # no record of what it rests on: said so, never a count it does not have
    n = facts.get("episodes") if isinstance(facts.get("episodes"), int) else 0
    m = facts.get("confirmed_uses") if isinstance(facts.get("confirmed_uses"), int) else 0
    a = facts.get("accepted") if isinstance(facts.get("accepted"), int) else 0
    text = f"{kind} from {n} episode{'' if n == 1 else 's'}" + (f", accepted in {a}" if a else "")
    return text + (f", {m} confirmed use{'' if m == 1 else 's'}" if m else "")


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


def _note_summary(row: dict) -> str:
    """A note's one line: its description (or title), then its scope when its front matter gives one: ``(applies to:
    <scope>; stated by: <who>)``, as the library's index shows it (``layout.note_scope``)."""
    about = _one_line(row.get("description")) or _one_line(row.get("title"))
    scope = "; ".join(
        f"{label}: {_one_line(row.get(key))}"
        for key, label in (("applies_to", "applies to"), ("stated_by", "stated by"))
        if _one_line(row.get(key))
    )
    return f"{about} ({scope})" if scope else about


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


def _shown_root() -> str:
    """The library's root as this box's texts name it: :data:`SANDBOX_LIBRARY` in a box that mounts the copy there,
    else the copy's own root."""
    root = _ROOT.rstrip("/")
    if root == SANDBOX_LIBRARY:
        return SANDBOX_LIBRARY
    return root


def _relative(text: str) -> str | None:
    """*text* as a path inside the copy, when it is a path under the copy's root, under :data:`SANDBOX_LIBRARY` or under
    ``/memory`` (each names the copy)."""
    roots = (_ROOT.rstrip("/"), SANDBOX_LIBRARY, LIBRARY_ROOT)
    for root in dict.fromkeys(r + "/" for r in roots):
        if text.startswith(root):
            return text[len(root) :]
    return None


#: The most characters of a note an agent is served by default: its ``## Rule`` section (or, for a note written before
#: notes had one, its opening), never its evidence appendix; the whole file stays readable at its library path.
NOTE_RULE_CHARS = 2000


def _note_rule(text: str, iid: str) -> str:
    """What ``memory.get`` serves of note *iid*: its front matter and its ``## Rule`` section (a note without one: its
    text), at most :data:`NOTE_RULE_CHARS` characters, ending with where the rest is when there is more."""
    lines = text.splitlines()
    head: list[str] = []
    body = lines
    if lines and lines[0].strip() == "---":
        end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
        if end is not None:
            head, body = lines[: end + 1], lines[end + 1 :]
    start = next((i for i, line in enumerate(body) if line.strip().lower() == "## rule"), None)
    if start is not None:
        stop = next((i for i, line in enumerate(body[start + 1 :], start + 1) if line.startswith("## ")), len(body))
        body = body[start:stop]
    served = "\n".join(head + body).rstrip("\n")
    rest = len(served) < len(text.rstrip("\n"))
    if len(served) > NOTE_RULE_CHARS:
        served, rest = served[:NOTE_RULE_CHARS].rstrip(), True
    if rest:
        # the library's root as this box's texts name it, so the path opens with this agent's own tools.read
        served += f"\n… (the rest of the note, its evidence included: tools.read('{_shown_root()}/{iid}'))"
    return served + "\n"


def _capture_form(capture_id: str) -> str:
    """A recorded procedure's entry form, from its record: ``value`` (``read_input`` and ``run(value)``), ``state``
    (``run(state)``) or ``state_env`` (``run(state, env)``: it acts through the episode's tools); '' when unknown."""
    rec = _read_json(os.path.join(_capture_dir(capture_id), "record.json"))
    entry = rec.get("entry_point") if isinstance(rec, dict) else None
    form = entry.get("form") if isinstance(entry, dict) else None
    return form if isinstance(form, str) else ""


#: What a recorded procedure that acts through tools is, and how its work is done from here, as its card and a call's
#: refusal say it: no call runs it (nothing here supplies the ``env`` its code takes: :meth:`Item.__call__` refuses
#: before running anything), what ``env`` stands for in its code (the fork writes each call its episode made as
#: ``env.<channel>.<method>(...)``, :data:`.integration.fork.CAPTURE_BULLETS`, whatever reached that channel: an
#: application namespace, a global API, a module or client the episode imported), its code is shown, and the same
#: calls made in a cell through what reaches the same thing here do its work.
BY_HAND = (
    "it records an earlier request's work as run(state, env), and no call runs it here: nothing here supplies env. "
    "In its code, env is not an object you have: env.<name>.<method>(...) stands for a call that request made of "
    "<method> on <name>, through whatever reached <name> there (an environment namespace such as primitives.<name>, "
    "a global API object, or a module or client it imported), which tools.list() need not show. "
    "print(memory.get({cid!r}).text) shows its code; making the same calls in a cell, through whatever reaches <name> "
    "here, on the current request (memory.request() gives its text), does its work"
)


def _capture_facts(capture_id: str) -> tuple[dict, dict]:
    """A recorded procedure's record (what the capture step saw) and its evidence row (its uses, what it rests on and
    its status, from the harness's evidence store); {} for either when it is not here."""
    rec = _read_json(os.path.join(_capture_dir(capture_id), "record.json"))
    row = (_evidence() or {}).get(CAPTURE_PREFIX + capture_id)
    return (rec if isinstance(rec, dict) else {}), (row if isinstance(row, dict) else {})


def _capture_support(rec: dict, row: dict) -> Any:
    """What a recorded procedure rests on: its evidence row's (its episodes, accepted work and confirmed uses), else
    its record's (its episodes and accepted work; its uses are not known here); None when neither has it."""
    got = row.get("support")
    if isinstance(got, dict):
        return got
    got = rec.get("support")
    return got if isinstance(got, dict) else None


#: The most characters a function's or a recorded procedure's card takes (no prompt flooding). A card never prints
#: code: ``item.text`` holds a function's source and a recorded procedure's whole code, and the card names that call.
CARD_CHARS = 2000
#: The most characters of an item's docstring its card shows.
CARD_DOC_CHARS = 800


def _short_doc(doc: str | None) -> str:
    """*doc* as a card shows it: at most :data:`CARD_DOC_CHARS` characters, cut with "…"."""
    text = (doc or "").strip()
    return text if len(text) <= CARD_DOC_CHARS else text[: CARD_DOC_CHARS - 1].rstrip() + "…"


def _code_pointer(iid: str, what: str) -> str:
    """The card's line that names the call printing an item's *what* (its source, its code)."""
    return f"Its {what}: print(memory.get({iid!r}).text)"


def _module_outline(text: str | None) -> tuple[str, str]:
    """A recorded procedure's code as its card describes it, never the code itself: its module docstring and its
    functions' names with their parameters (``read_input(text), run(value)``), from a parse; ('', '') when it does not
    parse."""
    try:
        tree = ast.parse(text or "")
    except (SyntaxError, ValueError):
        return "", ""
    names = [
        f"{node.name}({', '.join(a.arg for a in node.args.posonlyargs + node.args.args)})"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return ast.get_docstring(tree) or "", ", ".join(names)


def _bounded_card(head: list[str], body: list[str], iid: str, what: str) -> str:
    """A card of *head* lines, a blank line and *body* lines, at most :data:`CARD_CHARS` characters: the body is cut
    first and the call that prints the item's *what* always ends it."""
    pointer = _code_pointer(iid, what)
    text = "\n".join(head + [""] + body + [pointer])
    if len(text) <= CARD_CHARS:
        return text + "\n"
    room = CARD_CHARS - len(pointer) - 3
    cut = "\n".join(head + [""] + body)[: max(0, room - 1)].rstrip() + "…"
    return cut + "\n" + pointer + "\n"


def _capture_lines(iid: str, rec: dict) -> list[str]:
    """What a recorded procedure's card says it is and how it runs, from its record only: accepted work only when its
    record shows an accepted episode (``support.accepted``); a no-argument call only for the forms that take the
    request (``value``, ``state``); for one that acts through tools, that no call runs it and how its work is done
    from here (:data:`BY_HAND`)."""
    if rec.get("outcome") in _NEGATIVE:
        return ["A recorded attempt; its episode failed."]
    support = rec.get("support") if isinstance(rec.get("support"), dict) else {}
    accepted = support.get("accepted") if isinstance(support.get("accepted"), int) else 0
    what = (
        "A recorded procedure: it re-runs work an earlier request accepted."
        if accepted
        else "A recorded procedure: it re-runs the work of an earlier request (no outcome of that request is on record)."
    )
    form = _capture_form(iid[len(CAPTURE_PREFIX) :])
    if form in ("value", "state"):
        how = "Calling it with no arguments runs it on the current request, in a separate confined process."
    elif form == "state_env":
        how = BY_HAND.format(cid=iid)
        how = how[0].upper() + how[1:] + "."
    else:
        how = "Calling it runs it in a separate confined process on text or JSON values."
    return [what, how]


def _capture_ids() -> list[str]:
    """The captured procedures' ids, in id order: those exported into the copy, and those handed to a writer pass
    (:data:`HANDED_CAPTURES`)."""
    names: set[str] = set()
    for base in (os.path.join(_ROOT, CAPTURES_DIR), HANDED_CAPTURES):
        try:
            names.update(n for n in os.listdir(base) if os.path.isfile(os.path.join(base, n, "procedure.py")))
        except OSError:
            continue
    return [CAPTURE_PREFIX + n for n in sorted(names) if _CAPTURE_ID.fullmatch(n) is not None]


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


@dataclasses.dataclass(frozen=True)
class FitEvidence(Fit):
    """A :class:`Fit` given as evidence (with the harness's evidence serving on): ``ok`` says only that the input the
    item was recorded on is structurally similar to what was passed, never that the item does what you need.

    Members (beside ``ok`` and ``why``):
        level: how the structures matched (``exact``, ``structure`` or ``weak``; '' when nothing matched).
        conditions: the item's stated conditions and whether each held on its recorded case.
        failures: how many failed uses the item's record holds.
    """

    level: str = ""
    conditions: str = ""
    failures: int = 0

    def __str__(self) -> str:
        head = "structurally similar: " if self.ok else "not structurally similar: "
        return head + self.why + f"; {self.failures} recorded failure{'s' if self.failures != 1 else ''}"


@dataclasses.dataclass(frozen=True, repr=False)
class Item:
    """One item of the library, a function or a note: a frozen record with every member typed.

    ``print(item)`` shows its card, bounded and with no code: a function's signature, a recorded example call and its
    docstring; a note's front matter and its ``## Rule`` section (the whole note stays readable at its path); a
    recorded procedure's docstring and the names of its functions; then its status and what it rests on, its recorded
    failures and later uses, its links both ways and the library commits that changed it. ``item.text`` holds the
    code (a function's source, a recorded procedure's whole code), and the card ends with that call.

    Members:
        id: a function is ``memory.<package>.<module>:<name>`` (a captured procedure ``capture:`` and 16 hex
            digits); a note is ``notes/<topic>/<slug>.md``.
        kind: ``'function'`` or ``'note'``.
        summary: one line.
        applies: when it applies, written from the request alone ('' when none is recorded).
        evidence: ``'unchecked'`` (nothing has shown more) or ``'confirmed-by-outcomes'`` (outcomes after use confirm
            it); a note is unchecked.
        status: ``'on-trial'``, ``'adopted'`` or ``'retired'``.
        example: one recorded call, for a function ('' when none is shown).
        text: a note's front matter and ``## Rule`` section, a function's source with its docstring, or a recorded
            procedure's whole code.
        links: the ids of the items it links to and of those that link to it.
        history: the library commits that changed it, newest first.
        reason: why ``memory.ask`` judged it applicable ('' from ``memory.get`` and ``memory.items``).
        support: what it rests on, in words: its kind (curated item or recorded procedure), the
            episodes it comes from and its confirmed uses, for example ``'curated item from 3 episodes, 1 confirmed
            use'``.

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
    support: str = ""

    def _on_request(self) -> "Item | None":
        """What a call with no arguments runs on the current request: this item when it is a captured procedure
        (one of the value or state form reads the request itself), its recipe when it is a procedure with one, else
        None."""
        if self.id.startswith(CAPTURE_PREFIX):
            return self
        try:
            row = _Catalog("memory.get(...)()").rows.get(self.id) or {}
        except Exception:  # noqa: BLE001 - no catalogue here: the call is made as written
            return None
        recipe = row.get("recipe")
        if isinstance(recipe, str) and recipe and recipe != self.id:
            return dataclasses.replace(self, id=recipe)
        return self if recipe == self.id else None

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
            memory.get('memory.<package>.<module>:<name>').fits({'date': '2026-10-09'})
        """
        return _FIT(self.id, inputs)

    @checked(name="Item.__call__")
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call a function item on *args* and *kwargs*, as calling the function directly would. The call is tracked:
        a library function runs in this cell, as an import would call it, and what it raises reaches you unchanged
        (its refusal is its own ``MemoryInputError``); a captured procedure of the value or state form runs in a
        separate confined process and takes text or JSON values. A captured procedure that acts through its episode's
        tools (``run(state, env)``) is not run by any call: the call raises WrongCall, which says how its work is done
        from here (its card says the same).

        Args:
            args: the function's own positional arguments.
            kwargs: the function's own keyword arguments.

        Returns:
            What the function returned.

        Raises:
            Exception: whatever a library function raised, the same exception with its traceback and one
                note on what to do next (for its MemoryInputError: do the step yourself or find another item).
            ArgumentError: a captured procedure refused the input or did not return; the error says why.
            NotFound: the function could not be loaded from this copy of the library.
            WrongCall: the item is a note, which is read, not called; or a captured procedure that acts through its
                episode's tools, which no call runs.

        Example:
            memory.get('memory.<package>.<module>:<name>')('2026-10-09')

        With no arguments, a procedure that takes the request (a captured procedure of the value or state form, or a
        procedure with a recipe) runs on the current request, as ``memory.request()`` gives it: one call re-runs the
        stored work on it.
        """
        if self.id.startswith(CAPTURE_PREFIX) and _capture_form(self.id[len(CAPTURE_PREFIX) :]) == "state_env":
            # a fact of the harness, not a failure of the procedure: nothing is run and no use is recorded. The error
            # names the item, not the call as written: that call cannot work, so the text never shows it as one
            raise WrongCall(
                f"a call of {self.id}",
                "nothing was run: " + BY_HAND.format(cid=self.id) + ".",
            )
        if not args and not kwargs:
            target = self._on_request()
            if target is not None:
                return target(request())
        report = _RUN_TARGET(self.id, *args, **kwargs)
        if report.outcome == "returned":
            return report.result
        if report._raised is not None:
            raised = report._raised
            # the function's own exception, as a direct call raises it, with one note on what to do next: the
            # function's message is its writer's, and may not say what it expected
            if type(raised).__name__ == "MemoryInputError":
                note = (
                    f"{self.id} was not built for this input; its card shows what it was built for: "
                    f"print(memory.get({self.id!r})). Do this step yourself, or look for another item with "
                    "memory.items()."
                )
            else:
                note = (
                    f"raised inside the stored function {self.id}; its card shows its records: "
                    f"print(memory.get({self.id!r})), and its source: print(memory.get({self.id!r}).text)."
                )
            if hasattr(raised, "add_note") and note not in getattr(raised, "__notes__", ()):
                raised.add_note(note)
            raise raised
        call = call_text(f"memory.get({self.id!r})", args, kwargs)
        detail = (f"{report.error}: {report.detail}" if report.detail else report.error).rstrip(".")
        if self.id.startswith(CAPTURE_PREFIX):
            what = "refused this input" if report.outcome == "refused" else "did not return"
            raise ArgumentError(
                call,
                f"it {what} ({detail}).",
                expected="no arguments (the current request), or a request's text",
                example=f"memory.get({self.id!r})()",
                hint=f"Its card shows what it was built for: print(memory.get({self.id!r})).",
            )
        raise NotFound(
            call,
            f"it could not be called ({detail}).",
            hint=f"Its source: print(memory.get({self.id!r}).text).",
        )


def _plain_card(item: Item) -> str:
    """The card of an :class:`Item` made by hand (one ``get`` made carries its own)."""
    head = f"{item.id}  [{item.status}" + (f", {item.support}" if item.support else "") + "]"
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

    ``print(items)`` shows a compact table one page at a time (id, kind, what each rests on, one-line summary), with
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
        rests = max([len("rests on")] + [len(r.support or "-") for r in rows])
        out.append(f"  {'id':<{width}}  {'kind':<8}  {'rests on':<{rests}}  summary")
        for r in rows:
            out.append(f"  {r.id:<{width}}  {r.kind:<8}  {r.support or '-':<{rests}}  {_cut(r.summary)}".rstrip())
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
        self.withdrawn = _withdrawn()
        self._links: dict | None = None
        self._asked: dict | None = None
        self._modules: dict[str, tuple[str, Any]] = {}

    # -- what there is

    def all_ids(self) -> list[str]:
        return self.functions + self.notes + self.captures

    def listed(self) -> list[str]:
        """What ``items()`` lists: every function and note but the retired ones, and every captured procedure but
        those that were tried and failed."""
        out = [
            i
            for i in self.functions + self.notes
            if _status_of(self.rows[i]) not in _HIDDEN_STATUSES and i not in self.withdrawn
        ]
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

    def _named(self, ids: list[str]) -> list[str]:
        """*ids* as a card names them: one that names no item of the library (a link to an item that is gone) says so,
        so every id a card prints as an item opens with ``memory.get``."""
        return [i if i in self.rows or i in self.captures else f"{i} (not in the library)" for i in ids]

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
        return isinstance(rec, dict) and rec.get("outcome") in _NEGATIVE

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
        key = text.strip()
        # a withdrawn capture has left the copy; a withdrawn library item is still in it, and is not offered either
        gone = self.withdrawn.get(key) if (key in self.rows or not self.matches(text)) else None
        if gone is not None:
            return self._in_place_of(key, gone, reason)
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

    def _in_place_of(self, cid: str, gone: dict, reason: str) -> Item:
        """The item offered in the place of withdrawn capture *cid* (its card says so), or the error that says it is
        no longer offered."""
        by = gone.get("by")
        status = _str(gone.get("status"), "closed")
        if isinstance(by, str) and (by in self.rows or by in self.captures):
            built = self.item(by, reason)
            card = (
                f"{cid} is no longer offered: the library's writer {status} it"
                + (f" ({_one_line(gone.get('reason'))})" if _one_line(gone.get("reason")) else "")
                + f". {by} is offered in its place.\n\n"
                + built.__dict__.get("_card", "")
            )
            object.__setattr__(built, "_card", card)
            return built
        raise NotFound(
            self.call,
            f"{cid} is no longer offered: the library's writer {status} it"
            + (f" ({_one_line(gone.get('reason'))})." if _one_line(gone.get("reason")) else "."),
            expected=_GET_EXPECTED,
            example=f"memory.get({self._example_id()!r})",
            hint="Ask which items apply with memory.ask(), or browse with memory.items().",
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
            example=f"memory.items({(near or topics or ['notes/<topic>'])[0]!r})",
            nearest=near,
            hint="Browse every topic with memory.items().",
        )

    def _example_id(self) -> str:
        listed = self.listed()
        return listed[0] if listed else "notes/<topic>/<name>.md"

    # -- building items

    def generalises(self, iid: str) -> list[str]:
        """The recorded procedures item *iid* generalises: those whose episodes are among the episodes it comes
        from (its row's ``episodes``)."""
        mine = set((self.rows.get(iid) or {}).get("episodes") or ())
        out = []
        for cid in self.captures:
            rec = _read_json(os.path.join(_capture_dir(cid[len(CAPTURE_PREFIX) :]), "record.json"))
            eps = rec.get("episodes") if isinstance(rec, dict) else None
            if mine and isinstance(eps, list) and mine & {e for e in eps if isinstance(e, str)}:
                out.append(cid)
        return out

    def generalised_by(self, cid: str, rec: dict) -> list[str]:
        """The library items that generalise recorded procedure *cid*: those that come from one of its episodes."""
        eps = {e for e in rec.get("episodes") or () if isinstance(e, str)}
        return [i for i, row in self.rows.items() if eps & set(row.get("episodes") or ())] if eps else []

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
            support=_support_line("curated item", row.get("support")),
        )
        return built, self._card(iid, row, _short_doc(doc), missing)

    def _note(self, iid: str, row: dict) -> tuple[Item, str]:
        text, missing = "", ""
        try:
            text = _note_rule(_read(iid), iid)
        except _FileUnreadable as exc:
            missing = exc.error
        status = _status_of(row)
        built = Item(
            id=iid,
            kind="note",
            summary=_note_summary(row),
            applies="",
            evidence="unchecked",
            status=status if status in _STATUSES else _NEW_ITEM_STATUS,
            example="",
            text=text,
            links=tuple(dict.fromkeys(self.linked(iid, "links_to") + self.linked(iid, "linked_from"))),
            history=_history(row),
            support=_support_line("curated item", row.get("support")),
        )
        return built, self._card(iid, row, text.rstrip("\n"), missing)

    def _card(self, wanted: str, row: dict, body: str, missing: str) -> str:
        """One item, as ``print(memory.get(ref))`` shows it: a function's signature, with an example call quoted from
        its tests (``Example (from <test id>): <call>``) when one is recorded and still in its test file, then its
        docstring (*body*), never its source, and at the end the call that prints the source; or a note's Rule (*body*).
        Then its links both ways, its status with what it rests on (its kind, the episodes it comes from and its
        confirmed uses) and, for a function, its effect (and why, for an item left out of the index or given a
        reason), its verification and use records, and the subjects of the library commits that changed it, newest
        first. A function's card is at most :data:`CARD_CHARS` characters (:func:`_bounded_card`)."""
        status = _status_of(row)
        function = row["kind"] == "function"
        if function:
            rests = _support_line("curated item", row.get("support"))
            out = [f"{row.get('signature') or wanted}  [{status}" + (f", {rests}" if rests else "") + "]"]
            example = row.get("example")
            if isinstance(example, dict) and example.get("call"):
                out.append(f"Example (from {example.get('test')}): {example['call']}")
            if isinstance(row.get("recipe"), str) and row["recipe"]:
                # a procedure's one-call entry: its recipe applied to the current request (run only when called)
                out.append(f"Applies to the current request: {_recipe_call(row['recipe'])}")
        else:
            rests = _support_line("curated item", row.get("support"))
            out = [f"{wanted}  [{status}" + (f", {rests}" if rests else "") + "]"
                   + (f"  {row['title']}" if row.get("title") else "")]
        if status in _HIDDEN_STATUSES:
            out.append(f"Not in the index: {row.get('status_reason') or 'its status is ' + status}.")
        elif row.get("status_reason"):
            out.append(f"Status: {row['status_reason']}.")
        out += _evidence_lines(_evidence(), wanted)
        out += _label_lines(row.get("label"), _evidence() is not None)
        head = list(out)
        out = [f"(its {'source' if function else 'text'} cannot be read: {missing})" if missing else body]
        out += [
            "",
            "Links to: " + (", ".join(self._named(self.linked(wanted, "links_to"))) or "none"),
            "Linked from: " + (", ".join(self._named(self.linked(wanted, "linked_from"))) or "none"),
            "Generalises: " + (", ".join(self.generalises(wanted)) or "no recorded procedure"),
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
        if function:
            return _bounded_card(head, out, wanted, "source")
        return "\n".join(head + [""] + out) + "\n"

    def _capture(self, iid: str) -> tuple[Item, str]:
        d = _capture_dir(iid[len(CAPTURE_PREFIX) :])
        rec, sig, saved = (_read_json(os.path.join(d, f)) for f in ("record.json", "signature.json", "saved.json"))
        rec, sig, saved = (x if isinstance(x, dict) else {} for x in (rec, sig, saved))
        text = _read_text(os.path.join(d, "procedure.py"))
        asked = _str(saved.get("asked")) or _str(sig.get("goal"))
        # its status and level are what its uses gave it (the evidence store's), never its record's
        _, row = _capture_facts(iid[len(CAPTURE_PREFIX) :])
        level = _str(row.get("level"), "unchecked")
        level = _OLD_LEVELS.get(level, level)
        status = _str(row.get("status"), _NEW_ITEM_STATUS)
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
            support=_support_line("recorded procedure", _capture_support(rec, row)),
        )
        out = [f"{iid}  [{built.support}]", *_capture_lines(iid, rec)]
        by = self.generalised_by(iid, rec)
        if by:
            out.append("Generalised by: " + ", ".join(by))
        out += _evidence_lines(_evidence(), iid)
        if rec.get("outcome") in _NEGATIVE:
            out.append("Tried and failed: it is listed to show what broke, never run.")
            broke = saved.get("broke")
            out += [f"  broke: {b}" for b in (broke if isinstance(broke, list) else []) if isinstance(b, str)]
        if asked:
            out.append(f"Asked: {_cut(' '.join(asked.split()), 300)}")
        doc, functions = _module_outline(text)
        body = [_short_doc(doc)] if doc else []
        if functions:
            body.append(f"Functions: {functions}")
        if text is None:
            body.append("(its procedure cannot be read)")
        return built, _bounded_card(out, body, iid, "code")


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
        memory.get('notes/<topic>/<name>.md')
    """
    call = call_text("memory.get", (ref,))
    return _guard(call, _get, ref, call)


def _get(ref: Any, call: str) -> Item:
    return _Catalog(call).resolve(ref)


@checked(name="memory.items")
def items(topic: str | None = None, kind: Kind | None = None) -> Items:
    """Browse the library: every item (but the retired), or one topic's, of one kind or both.

    Args:
        topic: a package (``'<package>'``, ``'memory.<package>'``), a module (``'<package>.<module>'``), a notes
            topic (``'notes/<topic>'``; a bare ``'<topic>'`` also takes the notes topic of that name) or
            ``'captures'``; None for the whole library.
        kind: ``'function'`` or ``'note'``; None for both.

    Returns:
        Items: a sequence of Item. ``print()`` shows a compact table, one page at a time, and the call for the
        next page.

    Raises:
        ArgumentError: topic is not a str, or kind is not 'function', 'note' or None.
        NotFound: no topic has that name; the error gives the nearest topics.
        WrongCall: the name is an item's, not a topic's; the error gives the call that opens it.

    Example:
        memory.items('notes/<topic>', kind='note')
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
        ids = [i for i in ids if i.startswith("notes/") == (kind == "note")]
    return Items(tuple(cat.item(i) for i in ids), topic, kind)


def _fit(target: str, inputs: Any) -> Fit:
    """``Item.fits``: the applicability check of one item (:func:`_structure_match`). It raises
    ``memory.match_result`` with the row it found (none when it does not fit)."""
    call = call_text(f"memory.get({target!r}).fits", (inputs,))
    return _guard(call, _fit_check, target, inputs, call)


def _fitted(target: str, ok: bool, why: str, level: str = "", conditions: str = "") -> Fit:
    """The :class:`Fit` of one check, or with the harness's evidence serving on a :class:`FitEvidence` that also
    carries how the structures matched, the stated conditions and the item's recorded failures."""
    evidence = _evidence()
    if evidence is None:
        return Fit(ok, why)
    return FitEvidence(ok, why, level=level, conditions=conditions, failures=len(_failures(evidence, target)))


def _fit_check(target: str, inputs: Any, call: str) -> Fit:
    gone = _withdrawn().get(target) if _evidence() is not None else None
    if gone is not None:
        by = gone.get("by")
        _emit_event(MATCH_EVENT, (), ())
        return _fitted(
            target,
            False,
            f"it is no longer offered: the library's writer {_str(gone.get('status'), 'closed')} it"
            + (f"; {by} is offered in its place: memory.get({by!r})" if isinstance(by, str) and by else ""),
        )
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
        return _fitted(
            target,
            False,
            "no input it was saved against is recorded, so nothing here can say whether it fits; "
            f"print(memory.get({target!r})) shows what it takes",
        )
    m = _matcher()
    query = _query(m, None, inputs, mine)
    if not query.shapes:
        _emit_event(MATCH_EVENT, (), ())
        return _fitted(
            target,
            False,
            "what you passed has no structure to compare (plain text has none); pass the data you were given",
        )
    # the item as it was saved, then as it is offered in a withdrawn capture's place, saved against what that was
    matched = [(c, _structure_match(query, c)) for c in mine]
    hit = next(((c, h) for c, h in matched if h), None)
    how = hit[1] if hit is not None else None
    if hit is not None:
        mine = [hit[0]]
    if how is None:
        _emit_event(MATCH_EVENT, (), ())
        return _fitted(target, False, "the input it was saved against has a different structure from what you passed")
    row = _match_row(mine[0], how)
    if mine[0].failed:
        _emit_event(MATCH_EVENT, (), _event_rows([row]))
        return _fitted(
            target,
            False,
            "it was tried on input like this and failed; it is listed to show what broke, never run",
            how,
            row.conditions,
        )
    _emit_event(MATCH_EVENT, _event_rows([row]), ())
    return _fitted(
        target,
        True,
        f"the input it was saved against has the structure of what you passed ({how}); stated conditions: "
        f"{row.conditions}. A structural match can be wrong.",
        how,
        row.conditions,
    )


def _run_target(target: str, /, *args: Any, **kwargs: Any) -> RunReport:
    """A function item's call, ``item(...)``: the tracked path (:func:`_tracked_run`). In the package it raises
    ``memory.helper_call`` as ``memory:run`` once per call, ``ok`` whatever the item did (its outcome is in
    ``memory.run_result``); a note's call is refused here."""
    if target.startswith("notes/"):
        raise WrongCall(
            call_text(f"memory.get({target!r})", args, kwargs),
            "a note cannot be called; it is text to read.",
            hint=f"Read it with print(memory.get({target!r})).",
        )
    return _tracked_run(target, args, kwargs)


#: An item's applicability check and its tracked call; in the package each raises ``memory.helper_call`` (under the
#: audit ids ``memory:match`` and ``memory:run``), bound at the end of this file.
_FIT: Any = _fit
_RUN_TARGET: Any = _run_target


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


def _emit(item: str, digest: str, outcome: str, event: str = CALL_EVENT) -> None:
    try:
        sys.audit(event, item, digest, outcome)
    except Exception:  # noqa: BLE001 - an audit hook never changes the call
        pass
    if event == CALL_EVENT and outcome != "ok":
        _note_failed(item)


#: The items whose call failed in this process (it raised, or refused its input), by the request they were called on
#: (a digest of its text). ``memory.ask`` sends them with its question, so the librarian never offers one of them
#: again in the same request: a use outcome of this request, kept in this process only, never stored.
_FAILED_HERE: dict[str, set] = {}


def _request_key() -> str:
    """The current request as :data:`_FAILED_HERE` keys it: a digest of its text ('' when it is not known here)."""
    text = _current_request()[0]
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16] if text else ""


def _note_failed(item: str) -> None:
    try:
        _FAILED_HERE.setdefault(_request_key(), set()).add(item)
    except Exception:  # noqa: BLE001 - keeping it never changes the call
        pass


def failed_here() -> list[str]:
    """The items whose call failed in this process on the current request, in id order."""
    if not _FAILED_HERE:
        return []  # nothing failed in this process: no need to ask which request this is
    try:
        return sorted(_FAILED_HERE.get(_request_key(), ()))
    except Exception:  # noqa: BLE001 - nothing known: nothing is held back
        return []


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
# The package's own calls each raise ``memory.helper_call`` once (module docstring): the lookups, and an item's
# tracked call and applicability check under the audit ids ``memory:run`` and ``memory:match``.
if __name__ == "memory":
    _install_call_finder(__name__)
    get = wrap_call(get, "memory:get", HELPER_EVENT)
    items = wrap_call(items, "memory:items", HELPER_EVENT)
    _RUN_TARGET = wrap_call(_run_target, "memory:run", HELPER_EVENT)
    _FIT = wrap_call(_fit, "memory:match", HELPER_EVENT)
