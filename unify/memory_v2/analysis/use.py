"""How one request used the memory library: a pure function of its transcript (memory v2.1, stage 1).

Standard library only, and no other module of this package is imported, so the file can be copied
into an offline analyser as it is. The harness computes the record at a request's ``finish``
(:func:`request_use` over the request's redacted transcript lines, the memory items at its pin, its
recorded actions, its ``memory.diff``, the export's root and the library's import surface at the pin)
and the offline funnel analyser recomputes it from an exported episode directory
(:func:`use_from_episode_dir`); both run the same code on the same bytes.

Everything is structural. Imports and calls come from each cell's syntax tree. Errors come only from
the runtime's structured result of the cell (``ExecutionResult.error``, ``session_id`` and
``session_created``), which the harness notes by tool call id as the result comes back
(``hooks.tool_result``, :func:`runtime_status`) and the record keeps, reduced to names, as
``cell_status``; frames are read by file path and function name, and a function name is kept only when
each dotted part is an identifier (else ``<other>``). The rendered tool message is never read for a
status, so nothing a cell prints can stand in for one, in either code projection.

Some cells have no structured result from the runtime; ``cells_without_metadata_by_cause`` counts them
by cause (``cells_without_metadata`` is their sum):

* ``empty``: the code tool's own plain result for an empty cell (status ``empty``; nothing ran);
* ``executor_error``: its plain result when the session executor itself raised (status ``error``, with
  the exception's class name when it is a builtin exception, else ``<other>``);
* ``tool_raised``: the tool call raised before returning (the harness notes the exception's class, as
  above);
* ``cancelled``, ``timeout``, ``step_cap``: the harness answered the call itself, as its own reply's
  opening words say (``Cancelled: ...``, ``Not run: ...``); ``other`` for anything else, such as a
  recording from before the field. These words are read only to name the cause, never for an outcome.

A cell whose outcome is unknown (``tool_raised`` and the harness-answered causes, ``executor_error``, or
an error too long to read, ``unread``) makes unknown the outcome of the items its code could reach, for
this request only (``items_outcome_unknown``): the items it imports, calls or references, those a
function it calls reaches (a ``def`` or ``class`` of the session whose body touched them), and every item
of a channel it uses dynamically or as a module object (``*``: every item). A cell too large or too
deep to parse reaches every item; one that does not compile, and a non-Python cell, reach none (a shell
cell's use of the library is not seen either way). ``outcomes_known`` is true when no item's outcome is
unknown. Text is read only for its structure: the traceback's header, frame and chaining lines and the
exception's type token, the diff's file headers, the harness's reply words above and, for legacy
recordings only, the index's channel headings and item lines. An exception message is never read for
meaning; a message deliberately written to imitate CPython's chain line, header and frame lines could
still forge a block. Nothing is keyed on a task.

Per memory item (``env/<channel>:<name>``, a public function of ``env/<channel>/__init__.py``):

* ``imported``: import statements that bind the item by name (``from env.<channel> import name``,
  aliases included, and ``from env.<channel> import *``, which binds the module's ``__all__``, or its
  public names when it has none, as at the pin); a name another channel re-exports
  (``from env.b import g`` in ``env/a/__init__.py``) resolves to the defining item ``env/b:g``;
* ``called``: call sites in cell code whose callee resolves to the item (a bound name or alias, an
  ``env.<channel>.name`` or ``<module alias>.name`` attribute, or a plain assignment of one of these);
  a static count of sites, not of executions;
* ``referenced``: other loads of the item (passed as a value, ``help(f)``, assigned);
* ``guarded``: those calls that sit inside a ``try`` body with an ``except`` clause (a refusal caught
  there leaves no traceback, so it cannot be counted);
* ``refused``: exceptions whose type is ``MemoryInputError`` and which left the item into cell code
  (see :func:`attribute_errors`); ``errored``: any other exception that left the item;
* ``modified_in_request``: the request's ``memory.diff`` changed a file of the item's channel (the
  scratch export the cells import from), so its refusals and errors are booked as ``refused_modified``
  and ``errored_modified`` instead: the code that raised may be the agent's edit, not the stored item;
* ``refused_then_accepted``: (unmodified) refusals followed, in a later cell of the same request, by an
  action on the item's channel that the environment recorded as ``ok`` (a channel-level proxy for "the
  environment accepted the input when it was handled directly"; the input itself is not compared).

Calls the syntax tree cannot resolve to one item (``getattr(module, name)(...)``, ``vars(module)``,
``importlib.import_module("env...")``, a module's ``__dict__``) are counted per channel in
``unknown_calls`` and never guessed; ``*`` is a dynamic access of the ``env`` package itself. Channel keys
are channels of the pin, ``*``, or ``?`` for a name the pin has no channel for (no text from cell code is
kept beyond item ids and channel names).

Names bound by a cell persist into later cells of the same execution session (the structured result's
``session_id``; a cell without a structured result from the runtime stays in the previous cell's
session) and are dropped when a
result reports a new session (``session_created``).

What the prompt showed is a structured record, not a reading of the prompt's text: whatever renders the
memory section calls :func:`record_shown` with the channel and item names it rendered and the exact text
it appended (kept only as a SHA-256, its UTF-8 byte count and the index's token estimate), and the record
keeps that as ``shown_record``. ``shown_channels`` and ``shown_items`` are its names (items and channels of
the pin) when a system prompt was recorded; ``memory_section_shown.prompt_confirmed`` says whether a
recorded system prompt ends with exactly that text. ``exposure_source`` says where the shown lists came
from: ``record``; ``legacy_text`` for a recording without the record, where the section is found by the
v2 index's opening line (:data:`HEADER_PREFIX`) and read by its ``## env.<channel>`` headings,
``- env.<channel>`` catalogue lines and ``- `name(...)` `` item lines; or ``unknown`` when neither is
there, and when the record is there but no system prompt was recorded (``prompt_confirmed`` false: nothing
can say whether the section reached the model, so nothing counts as shown).

What the cells asked the library helper to show counts as exposure too (``cell_exposure``), read from each
cell's syntax tree like the call sites, never from its output: under ``UNIFY_MEMORY_V2_SURFACING=catalogue``
the prompt names no channel or function, and the model looks them up. The helper is the export's ``memory``
module (``import memory [as m]``, ``from memory import catalog, find, describe``):

* ``memory.catalog()`` shows every channel of the pin (its lines and counts; whether it also lists the
  functions depends on the library's size, which the code does not show, so no function counts);
* ``memory.catalog(<channel>)``, with the channel a constant (``env.<ch>``, ``<ch>`` or ``env/<ch>``),
  shows that channel and its functions (every page is counted as shown: an approximation for a channel
  longer than one page);
* ``memory.describe(<function>)`` and ``help(<function>)`` show that function: a bound item, or for
  ``describe`` a constant naming it (``env/<ch>:<fn>``, ``env.<ch>.<fn>``, ``<ch>.<fn>`` or a name only one
  channel has); ``help(<channel module>)`` shows the channel and its functions;
* ``memory.find(<value>)`` shows functions that depend on the value at run time, so it is only counted.

``cell_exposure`` holds ``items`` and ``channels`` (a shown item's channel is shown too) and ``calls``, the
number of call sites of ``catalog``, ``find``, ``describe`` and ``help`` (on a memory object). An argument
the tree cannot resolve shows nothing (never guessed). The evidence store counts these as shown.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "HEADER_PREFIX",
    "VERSION",
    "attribute_errors",
    "dict_status",
    "env_channel",
    "failed_status",
    "library_surface",
    "memory_section",
    "modified_channels",
    "parse_traceback",
    "record_shown",
    "request_use",
    "roots_of",
    "runtime_status",
    "section_digest",
    "shown_in",
    "transcript_cells",
    "use_from_episode_dir",
]

VERSION = 6
SHOWN_VERSION = 1

#: The start of the v2 index's header (``unify.memory_v2.index.HEADER``); a test keeps the two in step.
#: Used only for legacy recordings, which have no ``shown_record``.
HEADER_PREFIX = "Memory library: candidates to check, not authority."

EXPOSURE_SOURCES = ("record", "legacy_text", "unknown")
#: A cell's status: its structured result had no error, had one, had one too large to read, the cell was
#: empty, or there was no structured result for it.
CELL_STATUSES = ("ok", "error", "unread", "empty", "unknown")
#: Why a cell has no structured result from the runtime (``cells_without_metadata_by_cause``).
CAUSES = (
    "empty",
    "executor_error",
    "tool_raised",
    "cancelled",
    "timeout",
    "step_cap",
    "other",
)
#: What a function or exception name that is not identifier-shaped (or not allowed) is kept as.
OTHER_NAME = "<other>"
#: The exception class names a cell status may keep: the builtin exceptions'.
EXCEPTION_NAMES = frozenset(
    n
    for n, v in vars(builtins).items()
    if isinstance(v, type) and issubclass(v, BaseException)
)
# The harness's own replies to a call it answered itself (``unify.common._async_tool.loop`` and
# ``loop_stop``), by opening words, in order: the cause they name.
_REPLY_CAUSES = (
    ("Cancelled: the step limit (", "step_cap"),
    ("Not run: the step limit (", "step_cap"),
    ("Not run: max_steps (", "step_cap"),
    ("Cancelled: the timeout (", "timeout"),
    ("Not run: the timeout (", "timeout"),
    ("Not run: timeout (", "timeout"),
    ("Cancelled: ", "cancelled"),
)

REFUSAL_TYPE = "MemoryInputError"

# Bounds: the record stays small whatever the request did.
MAX_CELLS = 2000
MAX_CODE_CHARS = 200_000
MAX_TRACEBACK_CHARS = 200_000
MAX_ITEMS_AT_PIN = 2000
MAX_ITEM_ROWS = 500
MAX_CELL_REFS = 20
MAX_SECTIONS = 4
MAX_CHANNEL_KEYS = 200
MAX_ACTIONS = 100_000
MAX_STAR_NAMES = 500
MAX_REEXPORTS = 500
MAX_ROOTS = 4
MAX_REEXPORT_HOPS = 8
MAX_DIFF_CHARS = 2_000_000
MAX_RENDERER_CHARS = 64
MAX_EXITS = 50
MAX_NAME_CHARS = 200
#: The library helper's module (the export's generated ``memory.py``, :mod:`..memory_helper`) and the
#: functions of it whose calls show the library (``cell_exposure``).
HELPER_MODULE = "memory"
HELPER_FUNCTIONS = ("catalog", "find", "describe")

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_IDENT_RE = re.compile(_IDENT)
_ITEM_ID = re.compile(rf"^env/({_IDENT}):({_IDENT})\Z")
_FRAME = re.compile(r'^  File "(?P<file>.*)", line \d+, in (?P<func>.+)$')
_TYPE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?::|$)")
_HEADER = "Traceback (most recent call last):"
# CPython's exception-group layout (traceback.TracebackException.format): the top-level group's lines
# start with "  | ", its members' with "    | ", and separators with "  +-+" or "    +".
_GROUP_HEADER = "  + Exception Group Traceback (most recent call last):"
_GROUP_LINE = "  | "
_MEMBER_LINE = "    | "
_SEPARATORS = ("  +-+", "    +")
_NESTED_GROUP = "Exception Group Traceback"
# The lines CPython's traceback module puts between chained exceptions (its own format, never a message).
_CHAIN = frozenset(
    s.strip()
    for s in (
        getattr(
            traceback,
            "_cause_message",
            "\nThe above exception was the direct cause of the following exception:\n\n",
        ),
        getattr(
            traceback,
            "_context_message",
            "\nDuring handling of the above exception, another exception occurred:\n\n",
        ),
    )
)
_NOT_NAME = re.compile(r"[^a-z0-9_]+")
_SHOWN_CHANNEL = re.compile(rf"^(?:## |- `?)env\.({_IDENT})\b")
_SHOWN_ITEM = re.compile(rf"^- `({_IDENT})\(")
_DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
_DIFF_TRUNCATED = re.compile(
    r"^\[memory\.diff truncated at \d+ of \d+ bytes; blob [0-9A-Za-z]+\]$",
)
_SAFE_LOAD = (ValueError, TypeError, RecursionError)


# --- the library at the pin ------------------------------------------------------------------------------


def _all_names(tree: ast.Module) -> list[str] | None:
    for node in tree.body:
        value = None
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__all__"
        ):
            value = node.value
        if isinstance(value, (ast.List, ast.Tuple)):
            return [
                e.value
                for e in value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return None


def _public_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name != "*":
                    names.append(a.asname or a.name.split(".")[0])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.append(n.id)
    return [n for n in names if not n.startswith("_")]


def library_surface(root: str | Path) -> dict:
    """The import surface of the library at *root* (the export, read before any cell runs):
    ``{"star": {channel: names a star import binds}, "reexports": {"env/a:g": "env/b:g"}}``.

    ``star`` is the module's ``__all__``, or its public top-level names without one. A re-export is a
    top-level ``from env.<b> import g [as h]`` (or ``from ..<b> import g``) in ``env/<a>/__init__.py``.
    """
    star: dict[str, list[str]] = {}
    reexports: dict[str, str] = {}
    for mod in sorted(Path(root).glob("env/*/__init__.py")):
        channel = mod.parent.name
        if not re.fullmatch(_IDENT, channel) or len(star) >= MAX_CHANNEL_KEYS:
            continue
        try:
            tree = ast.parse(mod.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        names = _all_names(tree)
        if names is None:
            names = _public_names(tree)
        star[channel] = sorted({n for n in names if re.fullmatch(_IDENT, n)})[
            :MAX_STAR_NAMES
        ]
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            parts = (node.module or "").split(".")
            if node.level == 0 and len(parts) == 2 and parts[0] == "env":
                source = parts[1]
            elif node.level == 2 and len(parts) == 1 and parts[0]:
                source = parts[0]
            else:
                continue
            for a in node.names:
                if a.name != "*" and len(reexports) < MAX_REEXPORTS:
                    reexports[f"env/{channel}:{a.asname or a.name}"] = (
                        f"env/{source}:{a.name}"
                    )
    return {"star": star, "reexports": dict(sorted(reexports.items()))}


class _Items:
    """The memory items at the request's pin, by channel, with the library's import surface."""

    def __init__(self, ids: Iterable[str], surface: Any = None) -> None:
        self.ids: list[str] = []
        self.by_channel: dict[str, list[str]] = {}
        for raw in sorted({i for i in ids if isinstance(i, str)}):
            m = _ITEM_ID.match(raw)
            if m is None:
                continue
            self.ids.append(raw)
            self.by_channel.setdefault(m.group(1), []).append(m.group(2))
        self.known = frozenset(self.ids)
        surface = surface if isinstance(surface, dict) else {}
        star = surface.get("star")
        self.star: dict[str, list[str]] | None = None
        if isinstance(star, dict):
            self.star = {
                k: sorted({n for n in v if isinstance(n, str)})
                for k, v in star.items()
                if isinstance(k, str) and isinstance(v, list)
            }
        rex = surface.get("reexports")
        self.reexports = (
            {k: v for k, v in rex.items() if isinstance(k, str) and isinstance(v, str)}
            if isinstance(rex, dict)
            else {}
        )
        self.channels = frozenset(self.by_channel) | frozenset(self.star or ())

    def item(self, channel: str, name: str) -> str | None:
        """The item ``<channel>.<name>`` resolves to (following re-exports), or None."""
        iid: str | None = f"env/{channel}:{name}"
        for _ in range(MAX_REEXPORT_HOPS):
            if iid is None or iid in self.known:
                return iid
            iid = self.reexports.get(iid)
        return None

    def star_names(self, channel: str) -> list[str]:
        if self.star is not None and channel in self.star:
            return list(self.star[channel])
        return list(self.by_channel.get(channel, []))


def env_channel(kind: Any, channel: Any) -> str | None:
    """The memory channel of an action (a copy of ``unify.memory_v2.episodes.env_channel``)."""
    if not isinstance(channel, str) or not channel:
        return None
    if kind == "tool" or ":" not in channel:
        return channel
    prefix, key = channel.split(":", 1)
    if prefix != kind:
        return None
    name = _NOT_NAME.sub("_", key.lower()).strip("_")
    return f"{kind}_{name}" if name else None


def modified_channels(
    diff: str,
    items: Iterable[str] | _Items,
) -> tuple[list[str], bool]:
    """(channels of the pin whose files the request's ``memory.diff`` changed, whether it was truncated).

    Read from the diff's ``diff --git a/<path> b/<path>`` headers only. A changed file directly under
    ``env/`` touches every channel, and so does a truncated diff (its later files are unknown).
    """
    its = items if isinstance(items, _Items) else _Items(items)
    if not isinstance(diff, str) or not diff:
        return [], False
    changed: set[str] = set()
    every = truncated = False
    for line in diff[:MAX_DIFF_CHARS].splitlines():
        m = _DIFF_GIT.match(line)
        if m is not None:
            for path in m.groups():
                parts = path.split("/")
                if parts[0] != "env":
                    continue
                if len(parts) >= 3:
                    changed.add(parts[1])
                else:
                    every = True
        elif _DIFF_TRUNCATED.match(line):
            truncated = True
    if len(diff) > MAX_DIFF_CHARS:
        truncated = True
    if every or truncated:
        changed |= set(its.channels)
    return sorted(changed & set(its.channels)), truncated


def roots_of(checkout: str | Path) -> list[str]:
    """The paths cells may see the export at: as given (its import path) and resolved (its bind)."""
    given = str(checkout)
    return _roots([given, os.path.realpath(given)])


def _roots(roots: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for r in roots:
        if isinstance(r, str) and r and r not in out and len(out) < MAX_ROOTS:
            out.append(r)
    return out


# --- cells from the transcript ---------------------------------------------------------------------------


def _fold(lines: Iterable[Any]) -> list[dict]:
    """The transcript's message lines, each tool result at its first position in its final version
    (as ``unify.memory_v2.integration.trajectory.fold``)."""
    out: list[dict] = []
    tool_at: dict[str, int] = {}
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        kind, msg = ln.get("type"), ln.get("message")
        if kind not in ("message", "message_update") or not isinstance(msg, dict):
            continue
        call_id = msg.get("tool_call_id") if msg.get("role") == "tool" else None
        key = str(call_id) if call_id is not None else None
        if key is not None and key in tool_at:
            out[tool_at[key]] = {**out[tool_at[key]], "message": msg}
            continue
        if kind != "message":
            continue
        out.append(dict(ln))
        if key is not None:
            tool_at[key] = len(out) - 1
    return out


def _arguments(fn: dict) -> dict:
    args = fn.get("arguments") or "{}"
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args)
    except _SAFE_LOAD:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def transcript_cells(lines: Iterable[Any]) -> list[dict]:
    """Every ``execute_code`` cell with a result, in result order (the episode's cell indices):
    ``{"index", "call", "code", "language"}``, ``call`` being the tool call's id.

    The rendered result is not read: what it shows depends on the code projection (the legacy one
    starts with a metadata block, the notebook one with what the cell printed), so nothing in it can be
    told apart from a cell's own output. A cell's status comes from :func:`runtime_status` instead.
    """
    pending: dict[Any, tuple[str, str]] = {}
    cells: list[dict] = []
    for row in _fold(lines):
        msg = row.get("message")
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if fn.get("name") != "execute_code":
                continue
            args = _arguments(fn)
            code = args.get("code")
            lang = args.get("language") or args.get("_language") or "python"
            pending[tc.get("id")] = ("" if code is None else str(code), str(lang))
        if msg.get("role") == "tool" and msg.get("tool_call_id") in pending:
            call = msg["tool_call_id"]
            code, lang = pending.pop(call)
            cells.append(
                {
                    "index": len(cells),
                    "call": str(call),
                    "code": code,
                    "language": lang,
                },
            )
    return cells


def _session_of(value: Any) -> int | str | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    return value if not isinstance(value, str) else value[:MAX_NAME_CHARS]


def _name_shape(name: Any) -> str:
    """*name* when it is a (dotted) identifier of at most :data:`MAX_NAME_CHARS` characters, else
    :data:`OTHER_NAME`: a frame's function name is kept only in that shape, so nothing else a traceback
    holds (or a block forged in a message) can land in the record."""
    if (
        isinstance(name, str)
        and 0 < len(name) <= MAX_NAME_CHARS
        and all(part.isidentifier() for part in name.split("."))
    ):
        return name
    return OTHER_NAME


def _exc_name(name: Any) -> str:
    """*name* when it is a builtin exception's class name, else :data:`OTHER_NAME`."""
    return name if isinstance(name, str) and name in EXCEPTION_NAMES else OTHER_NAME


def runtime_status(
    error: Any,
    session_id: Any = None,
    session_created: Any = None,
    *,
    items: Iterable[str] | _Items,
    roots: Iterable[str] = (),
) -> dict:
    """One cell's status from the runtime's structured result, as the use record keeps it.

    *error*, *session_id* and *session_created* are the fields of the ``ExecutionResult`` the code tool
    returned (the harness's own values, whatever the cell printed and whichever projection rendered
    it); *items* are the ids at the pin and *roots* the export's roots (:func:`roots_of`). The traceback
    is reduced at once to its exits from memory modules, so the result holds names only:
    ``{"status", "exits", "session", "fresh"}``, where ``status`` is ``ok``, ``error`` or ``unread`` (an
    error longer than :data:`MAX_TRACEBACK_CHARS`, whose outcome is then unknown) and each exit is
    ``[kind, channel, function]`` (``kind`` ``refused`` or ``errored``; see :func:`attribute_errors`).
    """
    its = items if isinstance(items, _Items) else _Items(items)
    exits: list[list[str]] = []
    if not isinstance(error, str) or not error:
        status = "ok"
    elif len(error) > MAX_TRACEBACK_CHARS:
        status = "unread"
    else:
        status = "error"
        exits = [list(e) for e in _exits(error, its, tuple(_roots(roots)))]
    return {
        "status": status,
        "exits": exits,
        "session": _session_of(session_id),
        "fresh": session_created is True,
    }


def _no_result(status: str, cause: str, exc: Any = None) -> dict:
    out = {
        "status": status,
        "exits": [],
        "session": None,
        "fresh": False,
        "cause": cause,
    }
    if cause in ("executor_error", "tool_raised"):
        out["exc"] = _exc_name(exc)
    return out


def _traceback_type(text: Any) -> str | None:
    """The class name (last dotted part) of the last exception a traceback shows, or None."""
    blocks = parse_traceback(text) if isinstance(text, str) else []
    kind = blocks[-1]["type"] if blocks else None
    return kind.rsplit(".", 1)[-1] if isinstance(kind, str) else None


def dict_status(
    value: Any,
    *,
    items: Iterable[str] | _Items,
    roots: Iterable[str] = (),
) -> dict:
    """One cell's status from the code tool's plain ``dict`` result (the legacy projection returns one,
    unwrapped, for an empty cell and when the session executor itself raised).

    A dict whose ``stdout`` is a list is a runtime result left unwrapped and is read as one
    (:func:`runtime_status`). Otherwise ``error`` None is an ``empty`` cell (nothing ran) and an ``error``
    string the executor's own failure: status ``error`` with cause ``executor_error`` and the exception's
    class name when it is a builtin exception (else ``<other>``); its traceback is the harness's, so it
    has no exits, and the outcome of what the cell's code reaches is unknown. Neither holds a session.
    """
    get = value.get if hasattr(value, "get") else (lambda k, d=None: d)
    error = get("error")
    if isinstance(get("stdout"), list):
        return runtime_status(
            error,
            get("session_id"),
            get("session_created"),
            items=items,
            roots=roots,
        )
    if error is None:
        return _no_result("empty", "empty")
    return _no_result("error", "executor_error", _traceback_type(error))


def failed_status(exc_name: Any) -> dict:
    """The status of a code cell whose tool call raised instead of returning (``tool_raised``): unknown,
    with the exception's class name when it is a builtin exception (else ``<other>``).
    """
    return _no_result("unknown", "tool_raised", exc_name)


def _kept_status(entry: Any, its: _Items) -> dict | None:
    """A status read back (from :func:`runtime_status`, :func:`dict_status`, :func:`failed_status` or a
    recorded ``cell_status`` entry), or None when it is not one of their shapes."""
    if not isinstance(entry, dict) or entry.get("status") not in CELL_STATUSES:
        return None
    status, cause = entry["status"], entry.get("cause")
    if cause is not None:
        expected = (
            "empty"
            if cause == "empty"
            else "error" if cause == "executor_error" else "unknown"
        )
        if cause not in CAUSES or status != expected:
            return None
        return _no_result(status, cause, entry.get("exc"))
    if status == "empty":
        return None
    exits: list[list[str]] = []
    raw = entry.get("exits")
    for e in raw if isinstance(raw, list) else []:
        if (
            isinstance(e, (list, tuple))
            and len(e) == 3
            and e[0] in ("refused", "errored")
            and e[1] in its.by_channel
            and isinstance(e[2], str)
            and (e[2] == OTHER_NAME or _name_shape(e[2]) == e[2])
            and len(exits) < MAX_EXITS
        ):
            exits.append([e[0], e[1], e[2]])
    return {
        "status": status,
        "exits": exits if status == "error" else [],
        "session": _session_of(entry.get("session")),
        "fresh": entry.get("fresh") is True,
    }


def _reply_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                return part["text"]
    return ""


def _reply_causes(lines: Iterable[Any]) -> dict[str, str]:
    """Per tool call id, the cause the harness's own reply names (``cancelled``, ``timeout``,
    ``step_cap``), from the reply's opening words only; calls with any other reply are left out. Read
    only for a cell without a structured result, and only to name why it has none."""
    out: dict[str, str] = {}
    for row in _fold(lines):
        msg = row.get("message")
        if msg.get("role") != "tool" or msg.get("tool_call_id") is None:
            continue
        head = _reply_text(msg.get("content")).lstrip()[:200]
        for prefix, cause in _REPLY_CAUSES:
            if head.startswith(prefix):
                out[str(msg["tool_call_id"])] = cause
                break
    return out


def _outcome_unknown(status: dict) -> bool:
    """Whether a cell's outcome is unknown: no result, an unread error, or the executor's own failure."""
    return status["status"] in ("unknown", "unread") or (
        status.get("cause") == "executor_error"
    )


def _statuses(
    cells: list[dict],
    cell_status: Any,
    its: _Items,
    causes: dict[str, str] | None = None,
) -> list[dict]:
    """Each cell's status, in cell order, from *cell_status* (by call id: a mapping, or a list of entries
    with ``call``); ``unknown`` where it has none, with the cause the harness's reply names (*causes*,
    else ``other``). A cell without a runtime result stays in the previous cell's session.
    """
    causes = causes or {}
    by_call: dict[str, Any] = {}
    if isinstance(cell_status, dict):
        by_call = {str(k): v for k, v in cell_status.items()}
    elif isinstance(cell_status, list):
        for e in cell_status:
            if isinstance(e, dict) and isinstance(e.get("call"), str):
                by_call.setdefault(e["call"], e)
    out: list[dict] = []
    session: int | str | None = None
    for cell in cells:
        kept = _kept_status(by_call.get(cell["call"]), its)
        if kept is None or (
            kept["status"] == "unknown" and kept.get("cause") != "tool_raised"
        ):
            kept = _no_result("unknown", causes.get(cell["call"], "other"))
        elif (
            kept["status"] == "ok" and "cause" not in kept and not cell["code"].strip()
        ):
            # an empty cell: the notebook projection wraps the code tool's plain result, the legacy
            # one hands it over as it is (``dict_status``); both read ``empty``
            kept = _no_result("empty", "empty")
        if "cause" in kept:
            kept["session"] = session
        session = kept["session"]
        out.append({"call": cell["call"], **kept})
    return out


def _system_prompts(lines: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("type") == "system_prompt":
            text = ln.get("content")
        elif ln.get("type") == "session_start":
            text = ln.get("system_prompt")
        else:
            continue
        if isinstance(text, str) and text:
            out.append(text)
    return out


def _sections(prompts: Iterable[str], header: str) -> tuple[int, list[str]]:
    seen: list[str] = []
    n = 0
    for text in prompts:
        n += 1
        at = text.rfind(header)
        if at < 0:
            continue
        section = text[at:]
        if section not in seen and len(seen) < MAX_SECTIONS:
            seen.append(section)
    return n, seen


def memory_section(prompts: Iterable[str], header: str = HEADER_PREFIX) -> dict:
    """The memory section the system prompts carried: ``{"shown", "sha256", "bytes", "est_tokens",
    "prompts", "distinct"}`` of the first one (the text from the header's last occurrence to the end).
    """
    n, seen = _sections(prompts, header)
    if not seen:
        return {
            "shown": False,
            "sha256": None,
            "bytes": 0,
            "est_tokens": 0,
            "prompts": n,
            "distinct": 0,
        }
    first = seen[0]
    return {
        "shown": True,
        "sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(),
        "bytes": len(first.encode("utf-8")),
        "est_tokens": math.ceil(len(first) / 4),
        "prompts": n,
        "distinct": len(seen),
    }


def shown_in(
    sections: Iterable[str],
    items: Iterable[str] | _Items,
) -> tuple[list[str], list[str]]:
    """(item ids, channels) the memory sections show, by structure and id.

    A channel is shown by a ``## env.<channel>`` heading or a ``- env.<channel>`` (or
    ``- `env.<channel>``...) catalogue line; an item by its own ``- `name(...)`` line under a shown
    channel's heading. Only items and channels of the pin count.
    """
    its = items if isinstance(items, _Items) else _Items(items)
    shown: set[str] = set()
    channels: set[str] = set()
    for section in sections:
        channel = None
        for line in section.splitlines():
            m = _SHOWN_CHANNEL.match(line)
            if m is not None:
                channel = m.group(1) if m.group(1) in its.channels else None
                if channel is not None:
                    channels.add(channel)
                continue
            if line.startswith("## "):
                channel = None
                continue
            m = _SHOWN_ITEM.match(line)
            if m is not None and channel is not None:
                iid = f"env/{channel}:{m.group(1)}"
                if iid in its.known:
                    shown.add(iid)
    return sorted(shown)[:MAX_ITEMS_AT_PIN], sorted(channels)[:MAX_CHANNEL_KEYS]


def section_digest(text: str) -> dict:
    """``{"sha256", "bytes", "est_tokens"}`` of a memory section's exact text (``sha256`` None when empty).

    ``est_tokens`` is the index's own estimate (characters / 4).
    """
    if not isinstance(text, str) or not text:
        return {"sha256": None, "bytes": 0, "est_tokens": 0}
    raw = text.encode("utf-8")
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "est_tokens": math.ceil(len(text) / 4),
    }


def record_shown(
    text: str,
    *,
    channels: Iterable[str],
    items: Iterable[str] = (),
    renderer: str,
) -> dict:
    """What a memory-section renderer put in the system prompt, as names and a digest.

    Call it where the section is rendered, with the exact *text* the harness appends to the system prompt
    (``""`` when it appends nothing), the *channels* the text names (``"x"`` for ``env/x``), the *items*
    whose own lines it carries (``"env/x:parse"``; leave empty for a catalogue of channels) and a short
    *renderer* name (``"index"``, ``"catalogue"``). Keep the result on the request run; the use record
    reads it instead of searching the prompt's text, so the section's wording can change freely.

    Only names and a digest are kept, never the text: ``{"version", "renderer", "channels", "items",
    "sha256", "bytes", "est_tokens"}``. Malformed names are dropped, each item's channel counts as shown,
    and an empty *text* shows nothing whatever names are passed. Deterministic and bounded.
    """
    text = text if isinstance(text, str) else ""
    ids = sorted(
        {i for i in items if isinstance(i, str) and _ITEM_ID.match(i)},
    )[:MAX_ITEMS_AT_PIN]
    names = {c for c in channels if isinstance(c, str) and _IDENT_RE.fullmatch(c)}
    names |= {_ITEM_ID.match(i).group(1) for i in ids}
    name = renderer if isinstance(renderer, str) else ""
    return {
        "version": SHOWN_VERSION,
        "renderer": name[:MAX_RENDERER_CHARS],
        "channels": sorted(names)[:MAX_CHANNEL_KEYS] if text else [],
        "items": ids if text else [],
        **section_digest(text),
    }


def _valid_shown(value: Any) -> dict | None:
    """A ``shown_record`` read back (from :func:`record_shown` or a recorded ``memory_use.json``), or None."""
    if not isinstance(value, dict) or value.get("version") != SHOWN_VERSION:
        return None
    sha, size = value.get("sha256"), value.get("bytes")
    if sha is not None and not (
        isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
    ):
        return None
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return None
    tokens = value.get("est_tokens")

    def names(key: str, pattern: re.Pattern) -> list[str]:
        got = value.get(key)
        if not isinstance(got, list):
            return []
        return sorted({v for v in got if isinstance(v, str) and pattern.fullmatch(v)})

    renderer = value.get("renderer")
    return {
        "version": SHOWN_VERSION,
        "renderer": renderer[:MAX_RENDERER_CHARS] if isinstance(renderer, str) else "",
        "channels": names("channels", _IDENT_RE)[:MAX_CHANNEL_KEYS],
        "items": names("items", _ITEM_ID)[:MAX_ITEMS_AT_PIN],
        "sha256": sha,
        "bytes": size,
        "est_tokens": (
            tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0
        ),
    }


def _ends_with(prompt: str, sha: str, size: int) -> bool:
    raw = prompt.encode("utf-8")
    return (
        len(raw) >= size and hashlib.sha256(raw[len(raw) - size :]).hexdigest() == sha
    )


def _exposure(
    prompts: list[str],
    shown: Any,
    its: _Items,
) -> tuple[str, dict, dict | None, list[str], list[str]]:
    """(exposure source, ``memory_section_shown``, the shown record kept, shown items, shown channels).

    A record with no system prompt recorded is ``unknown`` (``prompt_confirmed`` false, nothing shown):
    nothing says whether the section reached the model.
    """
    record = _valid_shown(shown)
    if record is not None and not prompts:
        section = {
            **memory_section(prompts),
            "prompt_confirmed": False,
            "source": "unknown",
        }
        return "unknown", section, record, [], []
    if record is not None:
        sent = record["sha256"] is not None
        section = {
            "shown": sent,
            "sha256": record["sha256"] if sent else None,
            "bytes": record["bytes"] if sent else 0,
            "est_tokens": record["est_tokens"] if sent else 0,
            "prompts": len(prompts),
            "distinct": 1 if sent else 0,
            "prompt_confirmed": (
                any(_ends_with(p, record["sha256"], record["bytes"]) for p in prompts)
                if sent
                else None
            ),
            "source": "record",
        }
        if not sent:
            return "record", section, record, [], []
        items = [i for i in record["items"] if i in its.known]
        channels = [c for c in record["channels"] if c in its.channels]
        return "record", section, record, items, channels
    _, sections = _sections(prompts, HEADER_PREFIX)
    section = {**memory_section(prompts), "prompt_confirmed": None}
    if not sections:
        return "unknown", {**section, "source": "unknown"}, None, [], []
    items, channels = shown_in(sections, its)
    return "legacy_text", {**section, "source": "legacy_text"}, None, items, channels


# --- imports and calls -----------------------------------------------------------------------------------

# A binding: ("item", item id) | ("module", channel) | ("package",) | ("dynamic", channel or "*") |
# ("reaches", item ids, channel keys): a function or class of the session whose body touched those |
# ("helper",): the library helper module | ("helper_fn", name): one of its HELPER_FUNCTIONS
_Binding = tuple

_ROW_KEYS = (
    "imported",
    "called",
    "referenced",
    "guarded",
    "refused",
    "errored",
    "refused_modified",
    "errored_modified",
    "refused_then_accepted",
)


class _Tally:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.module_imports: dict[str, int] = {}
        self.unknown: dict[str, int] = {}
        self.unattributed: dict[str, int] = {}
        # cell_exposure: what the cells asked the library helper (or help()) to show
        self.exposed_items: set[str] = set()
        self.exposed_channels: set[str] = set()
        self.helper_calls: dict[str, int] = dict.fromkeys(
            (*HELPER_FUNCTIONS, "help"),
            0,
        )

    def row(self, item: str) -> dict:
        r = self.items.get(item)
        if r is None:
            r = self.items[item] = {
                **dict.fromkeys(_ROW_KEYS, 0),
                "modified_in_request": False,
                "cells": [],
            }
        return r

    def bump(self, item: str, key: str, cell: int, n: int = 1) -> None:
        r = self.row(item)
        r[key] += n
        if cell not in r["cells"] and len(r["cells"]) < MAX_CELL_REFS:
            r["cells"].append(cell)

    @staticmethod
    def add(counts: dict[str, int], key: str) -> None:
        if key in counts or len(counts) < MAX_CHANNEL_KEYS:
            counts[key] = counts.get(key, 0) + 1


class _Cell(ast.NodeVisitor):
    """Walks one cell in statement order; *bindings* persist from earlier cells of the session.

    ``touched_items`` and ``touched_channels`` are what the cell's code could reach: the items it
    imports, calls or references, and the channel keys (``*`` for the package) it uses dynamically or as
    a module object, including through a function or class of the session whose body touched them.
    """

    def __init__(
        self,
        items: _Items,
        bindings: dict,
        tally: _Tally,
        cell: int,
    ) -> None:
        self.items, self.b, self.t, self.cell = items, bindings, tally, cell
        self.guard = 0
        self.touched_items: set[str] = set()
        self.touched_channels: set[str] = set()

    def _use(self, item: str, key: str) -> None:
        self.t.bump(item, key, self.cell)
        self.touched_items.add(item)

    def _reach(self, target: _Binding | None) -> None:
        """A load of *target* that names no single item: what it can reach is touched."""
        if target is None:
            return
        if target[0] in ("module", "dynamic"):
            self.touched_channels.add(self._key(target[1]))
        elif target[0] == "package":
            self.touched_channels.add("*")
        elif target[0] == "reaches":
            self.touched_items |= target[1]
            self.touched_channels |= target[2]

    # -- names ------------------------------------------------------------------------------------------
    def _key(self, channel: str) -> str:
        """A channel as a record key: a channel of the pin, ``*`` (the package), else ``?``."""
        return channel if channel == "*" or channel in self.items.channels else "?"

    def _unbind(self, name: str | None) -> None:
        if name:
            self.b.pop(name, None)

    def _bind(self, name: str, value: _Binding | None) -> None:
        if value is None:
            self.b.pop(name, None)
        else:
            self.b[name] = value

    def _store(self, target: ast.AST) -> None:
        """Every name a store target binds is no longer a memory binding."""
        for n in ast.walk(target):
            if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
                self._unbind(n.id)
            elif isinstance(n, (ast.MatchAs, ast.MatchStar)):
                self._unbind(n.name)
            elif isinstance(n, ast.MatchMapping):
                self._unbind(n.rest)
        # loads inside a target (``a[f(x)] = ...``) still count
        for n in ast.walk(target):
            if isinstance(n, (ast.Subscript, ast.Attribute)) and not isinstance(
                n.ctx,
                ast.Load,
            ):
                self.visit(n.value)
                if isinstance(n, ast.Subscript):
                    self.visit(n.slice)

    # -- resolution -------------------------------------------------------------------------------------
    def _is_import_call(self, fn: ast.AST) -> bool:
        if isinstance(fn, ast.Name):
            return fn.id in ("__import__", "import_module") and fn.id not in self.b
        return (
            isinstance(fn, ast.Attribute)
            and fn.attr == "import_module"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "importlib"
        )

    def resolve(self, node: ast.AST) -> _Binding | None:
        if isinstance(node, ast.Name):
            return self.b.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(node.value)
            if base is None:
                return None
            if base[0] == "helper":
                return (
                    ("helper_fn", node.attr) if node.attr in HELPER_FUNCTIONS else None
                )
            if base[0] == "package":
                return ("module", node.attr)
            if base[0] == "module":
                iid = self.items.item(base[1], node.attr)
                if iid is not None:
                    return ("item", iid)
                return ("dynamic", base[1]) if node.attr == "__dict__" else None
            if base[0] == "dynamic":
                return base
            return None
        if isinstance(node, ast.Call):
            fn = node.func
            if (
                isinstance(fn, ast.Name)
                and fn.id in ("getattr", "vars")
                and fn.id not in self.b
                and node.args
            ):
                base = self.resolve(node.args[0])
                if base is not None and base[0] in ("module", "dynamic"):
                    return ("dynamic", base[1])
                if base is not None and base[0] == "package":
                    return ("dynamic", "*")
                return None
            if (
                self._is_import_call(fn)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                parts = node.args[0].value.split(".")
                if parts[0] == "env":
                    return ("dynamic", parts[1] if len(parts) > 1 and parts[1] else "*")
            return None
        if isinstance(node, ast.Subscript):
            base = self.resolve(node.value)
            return base if base is not None and base[0] == "dynamic" else None
        return None

    # -- imports ----------------------------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            parts = alias.name.split(".")
            if alias.name == HELPER_MODULE:
                self._bind(alias.asname or HELPER_MODULE, ("helper",))
                continue
            if parts[0] != "env":
                self._unbind(alias.asname or parts[0])
                continue
            if len(parts) >= 2:
                self.t.add(self.t.module_imports, self._key(parts[1]))
            if alias.asname is None:
                self._bind("env", ("package",))
            elif len(parts) == 1:
                self._bind(alias.asname, ("package",))
            elif len(parts) == 2:
                self._bind(alias.asname, ("module", parts[1]))
            else:
                self._unbind(alias.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        parts = mod.split(".")
        if not node.level and mod == HELPER_MODULE:
            for alias in node.names:
                if alias.name == "*":
                    for name in HELPER_FUNCTIONS:
                        self._bind(name, ("helper_fn", name))
                elif alias.name in HELPER_FUNCTIONS:
                    self._bind(alias.asname or alias.name, ("helper_fn", alias.name))
                else:
                    self._unbind(alias.asname or alias.name)
            return
        if node.level or parts[0] != "env" or len(parts) > 2:
            for alias in node.names:
                if alias.name != "*":
                    self._unbind(alias.asname or alias.name)
            return
        if len(parts) == 1:  # from env import <channel>
            for alias in node.names:
                if alias.name == "*":
                    continue
                self.t.add(self.t.module_imports, self._key(alias.name))
                self._bind(alias.asname or alias.name, ("module", alias.name))
            return
        channel = parts[1]
        for alias in node.names:
            if alias.name == "*":
                for name in self.items.star_names(channel):
                    iid = self.items.item(channel, name)
                    if iid is None:
                        self._unbind(name)
                        continue
                    self._bind(name, ("item", iid))
                    self._use(iid, "imported")
                continue
            iid = self.items.item(channel, alias.name)
            target = alias.asname or alias.name
            if iid is None:
                self._unbind(target)
                continue
            self._bind(target, ("item", iid))
            self._use(iid, "imported")

    # -- assignments ------------------------------------------------------------------------------------
    def _assign(self, targets: list[ast.AST], value: ast.AST | None) -> None:
        if value is not None:
            self.visit(value)
        for target in targets:
            if isinstance(target, ast.Name):
                self._bind(
                    target.id,
                    self.resolve(value) if value is not None else None,
                )
            elif (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(value, (ast.Tuple, ast.List))
                and len(target.elts) == len(value.elts)
                and not any(isinstance(e, ast.Starred) for e in target.elts)
            ):
                for t, v in zip(target.elts, value.elts):
                    if isinstance(t, ast.Name):
                        self._bind(t.id, self.resolve(v))
                    else:
                        self._store(t)
            else:
                self._store(target)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assign(list(node.targets), node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is None:
            self._store(node.target)
        else:
            self._assign([node.target], node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._store(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._assign([node.target], node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            self._store(t)

    def _loop(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._store(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    visit_For = visit_AsyncFor = _loop

    def _with(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._store(item.optional_vars)
        for s in node.body:
            self.visit(s)

    visit_With = visit_AsyncWith = _with

    def _try(self, node: ast.Try) -> None:
        guarded = bool(node.handlers)
        self.guard += guarded
        try:
            for s in node.body:
                self.visit(s)
        finally:
            self.guard -= guarded
        for h in node.handlers:
            if h.type is not None:
                self.visit(h.type)
            self._unbind(h.name)
            for s in h.body:
                self.visit(s)
        for s in node.orelse + node.finalbody:
            self.visit(s)

    visit_Try = visit_TryStar = _try

    def visit_match_case(self, node: ast.match_case) -> None:
        self._store(node.pattern)
        if node.guard is not None:
            self.visit(node.guard)
        for s in node.body:
            self.visit(s)

    # -- scopes -----------------------------------------------------------------------------------------
    def _args(self, args: ast.arguments) -> list[str]:
        names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
        names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        return names

    def _scoped(self, unbind: list[str], body: list[ast.AST]) -> None:
        saved = dict(self.b)
        for n in unbind:
            self.b.pop(n, None)
        try:
            for s in body:
                self.visit(s)
        finally:
            self.b.clear()
            self.b.update(saved)

    def _body_reach(self, name: str, unbind: list[str], body: list[ast.AST]) -> None:
        """Walk a ``def`` or ``class`` body; *name* is then bound to what the body touched (so a later
        call of it, in any cell of the session, reaches that), or unbound when it touched nothing.
        """
        outer = self.touched_items, self.touched_channels
        self.touched_items, self.touched_channels = set(), set()
        try:
            self._scoped(unbind, body)
        finally:
            inner = self.touched_items, self.touched_channels
            self.touched_items = outer[0] | inner[0]
            self.touched_channels = outer[1] | inner[1]
        if inner[0] or inner[1]:
            self._bind(name, ("reaches", frozenset(inner[0]), frozenset(inner[1])))
        else:
            self._unbind(name)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        self._body_reach(node.name, self._args(node.args), list(node.body))

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        self._scoped(self._args(node.args), [node.body])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self._body_reach(node.name, [], list(node.body))

    def _comprehension(self, node: ast.AST) -> None:
        saved = dict(self.b)
        try:
            for gen in node.generators:
                self.visit(gen.iter)
                self._store(gen.target)
                for cond in gen.ifs:
                    self.visit(cond)
            if isinstance(node, ast.DictComp):
                self.visit(node.key)
                self.visit(node.value)
            else:
                self.visit(node.elt)
        finally:
            self.b.clear()
            self.b.update(saved)

    visit_ListComp = visit_SetComp = visit_GeneratorExp = visit_DictComp = (
        _comprehension
    )

    # -- uses -------------------------------------------------------------------------------------------
    def _show_channel(self, channel: str, items: bool) -> None:
        if channel in self.items.channels:
            self.t.exposed_channels.add(channel)
            if items:
                self.t.exposed_items.update(
                    f"env/{channel}:{n}" for n in self.items.by_channel.get(channel, [])
                )

    def _named_item(self, text: str) -> str | None:
        """The item a constant names as :func:`..memory_helper.describe` reads it, or None."""
        text = text.strip()
        m = _ITEM_ID.match(text)
        if m is not None:
            return self.items.item(m.group(1), m.group(2))
        parts = text.split(".")
        if parts[0] == "env":
            parts = parts[1:]
        if len(parts) == 2:
            return self.items.item(parts[0], parts[1])
        if len(parts) == 1:
            found = [
                f"env/{ch}:{text}"
                for ch, names in self.items.by_channel.items()
                if text in names
            ]
            return found[0] if len(found) == 1 else None
        return None

    def _exposure(self, name: str, node: ast.Call) -> None:
        """A call of the library helper's *name* (or of ``help``): what it shows (``cell_exposure``)."""
        arg = node.args[0] if node.args else None
        if arg is None and name == "catalog":
            arg = next((k.value for k in node.keywords if k.arg == "channel"), None)
        const = (
            arg.value
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            else None
        )
        target = self.resolve(arg) if arg is not None else None
        if name == "help":
            if target is None or target[0] not in ("item", "module"):
                return  # not a memory object: not counted
            self.t.helper_calls["help"] += 1
        else:
            self.t.helper_calls[name] += 1
        if name == "catalog":
            if arg is None:
                for channel in self.items.channels:
                    self._show_channel(channel, items=False)
            elif const is not None:
                self._show_channel(
                    const.strip().removeprefix("env.").removeprefix("env/").rstrip("/"),
                    items=True,
                )
        elif name in ("describe", "help"):
            iid = None
            if target is not None and target[0] == "item":
                iid = target[1]
            elif target is not None and target[0] == "module":
                self._show_channel(target[1], items=True)
            elif const is not None and name == "describe":
                iid = self._named_item(const)
            if iid is not None and iid in self.items.known:
                self.t.exposed_items.add(iid)
                self.t.exposed_channels.add(iid.split(":", 1)[0][len("env/") :])

    def visit_Call(self, node: ast.Call) -> None:
        target = self.resolve(node.func)
        if target is not None and target[0] == "helper_fn":
            self._exposure(target[1], node)
        elif (
            isinstance(node.func, ast.Name)
            and node.func.id == "help"
            and node.func.id not in self.b
        ):
            self._exposure("help", node)
        if target is not None and target[0] == "item":
            self._use(target[1], "called")
            if self.guard:
                self._use(target[1], "guarded")
        elif target is not None and target[0] == "dynamic":
            self.t.add(self.t.unknown, self._key(target[1]))
            self._reach(target)
        if target is None or target[0] not in ("item", "module", "package"):
            self.visit(node.func)
        for a in node.args:
            self.visit(a)
        for k in node.keywords:
            self.visit(k.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        target = self.resolve(node) if isinstance(node.ctx, ast.Load) else None
        if target is not None and target[0] == "item":
            self._use(target[1], "referenced")
            return
        if target is not None and target[0] in ("module", "package", "dynamic"):
            self._reach(target)  # a module object used as a value: any of its functions
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            target = self.b.get(node.id)
            if target is not None and target[0] == "item":
                self._use(target[1], "referenced")
            else:
                self._reach(target)
        else:
            self._unbind(node.id)


# --- errors ----------------------------------------------------------------------------------------------


def _chained(lines: list[str], i: int) -> bool:
    """Whether the header at *i* follows one of CPython's chain lines (blank, chain line, blank)."""
    return i >= 2 and lines[i - 1].strip() == "" and lines[i - 2].strip() in _CHAIN


def _plain_block(lines: list[str], i: int) -> tuple[dict, int]:
    """The block whose header is at *i*: its frames and type, and the index after its type line."""
    frames: list[tuple[str, str]] = []
    i += 1
    while i < len(lines) and lines[i].startswith(" "):
        m = _FRAME.match(lines[i])
        if m is not None:
            frames.append((m.group("file"), m.group("func")))
        i += 1
    m = _TYPE.match(lines[i]) if i < len(lines) else None
    return {"frames": frames, "type": m.group(1) if m else None}, i + 1


def _group(lines: list[str], i: int) -> tuple[list[dict], int]:
    """The exception group whose header is at *i*, then its first-level members, and the index after it.

    Each member's escaping block starts with the group's frames (the frames it left the cell through);
    a member that is itself a group is skipped.
    """
    gid = i
    frames: list[tuple[str, str]] = []
    gtype = None
    i += 1
    while i < len(lines) and lines[i].startswith(_GROUP_LINE.rstrip()):
        body = lines[i][len(_GROUP_LINE) :] if lines[i].startswith(_GROUP_LINE) else ""
        m = _FRAME.match(body)
        if m is not None:
            frames.append((m.group("file"), m.group("func")))
        elif body and not body.startswith(" ") and gtype is None:
            t = _TYPE.match(body)
            gtype = t.group(1) if t else None
        i += 1
    subs: list[list[str]] = []
    while i < len(lines) and lines[i].startswith(
        _SEPARATORS + (_MEMBER_LINE.rstrip(),),
    ):
        line = lines[i]
        if line.startswith(_SEPARATORS):
            subs.append([])
        elif subs:
            subs[-1].append(
                line[len(_MEMBER_LINE) :] if line.startswith(_MEMBER_LINE) else "",
            )
        i += 1
    blocks: list[dict] = [{"frames": frames, "type": gtype, "group": gid}]
    for sub in subs:
        while sub and not sub[-1].strip():
            sub.pop()
        if not sub or sub[0].startswith(_NESTED_GROUP):
            continue
        inner = _blocks(sub, groups=False)
        if not inner:
            t = _TYPE.match(sub[0])
            inner = [{"frames": [], "type": t.group(1) if t else None}]
        inner[-1] = {**inner[-1], "frames": frames + inner[-1]["frames"]}
        blocks.extend({**b, "member_of": gid} for b in inner)
    return blocks, i


def _blocks(lines: list[str], *, groups: bool) -> list[dict]:
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == _HEADER and (not blocks or _chained(lines, i)):
            block, i = _plain_block(lines, i)
            blocks.append(block)
            continue
        if groups and line == _GROUP_HEADER and (not blocks or _chained(lines, i)):
            found, i = _group(lines, i)
            blocks.extend(found)
            continue
        i += 1
    return blocks


def parse_traceback(text: str) -> list[dict]:
    """The exceptions a formatted traceback shows, oldest first: ``{"frames": [(file, function)], "type"}``.

    Only the traceback's structure is read: a block starts at the header line (the first one, or one
    after a chaining line of CPython's own format), its frames are the ``File "...", line N, in f`` lines,
    and its type is the dotted name the exception line starts with. An exception group adds its own block
    (with ``"group"``) and its first-level members' blocks (with ``"member_of"``), each member's escaping
    block starting with the group's frames. The message is never read for meaning.
    """
    if not isinstance(text, str) or len(text) > MAX_TRACEBACK_CHARS:
        return []
    return _blocks(text.splitlines(), groups=True)


def _module_channel(path: str, items: _Items, roots: tuple[str, ...]) -> str | None:
    """The channel whose ``env/<channel>/__init__.py`` *path* is: exactly under one of the export
    *roots* when they are known, else by its last three components."""
    path = path.replace("\\", "/")
    if roots:
        for root in roots:
            prefix = root.replace("\\", "/").rstrip("/") + "/"
            if path.startswith(prefix):
                parts = path[len(prefix) :].split("/")
                if len(parts) == 3 and parts[0] == "env" and parts[2] == "__init__.py":
                    return parts[1] if parts[1] in items.by_channel else None
        return None
    parts = path.split("/")
    if len(parts) >= 3 and parts[-1] == "__init__.py" and parts[-3] == "env":
        return parts[-2] if parts[-2] in items.by_channel else None
    return None


def _exits(
    text: str,
    its: _Items,
    roots: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    """``(kind, channel, function)`` per exception of a traceback that left a memory module (see
    :func:`attribute_errors`), at most :data:`MAX_EXITS`; a function name that is not a dotted
    identifier of at most :data:`MAX_NAME_CHARS` characters is kept as :data:`OTHER_NAME`.
    """
    out: list[tuple[str, str, str]] = []
    left_groups: set[int] = set()
    for block in parse_traceback(text):
        if block.get("member_of") in left_groups:
            continue
        frames = block["frames"]
        if not frames or _module_channel(frames[0][0], its, roots) is not None:
            continue
        for path, func in frames[1:]:
            channel = _module_channel(path, its, roots)
            if channel is None:
                continue
            kind = (block["type"] or "").rsplit(".", 1)[-1]
            out.append(
                (
                    "refused" if kind == REFUSAL_TYPE else "errored",
                    channel,
                    _name_shape(func),
                ),
            )
            if "group" in block:
                left_groups.add(block["group"])
            break
        if len(out) >= MAX_EXITS:
            break
    return out


def _outcomes(
    exits: Iterable[Iterable[str]],
    its: _Items,
    changed: frozenset[str],
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for kind, channel, func in exits:
        iid = f"env/{channel}:{func}"
        if iid not in its.known:
            out.append(("unattributed", channel))
        else:
            out.append((kind + ("_modified" if channel in changed else ""), iid))
    return out


def attribute_errors(
    text: str | None,
    items: Iterable[str] | _Items,
    *,
    roots: Iterable[str] = (),
    modified: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """``(outcome, item or channel)`` per exception of a cell's traceback that left a memory item.

    An exception left an item when its outermost frame is outside the memory modules (cell code) and
    some later frame is in ``env/<channel>/__init__.py`` (exactly under an export root when *roots* are
    given); the first such frame names the item (its function). An exception whose outermost frame is
    itself in a memory module was caught inside it, so it is not counted; a group that left an item is
    counted once, without its members. ``outcome`` is ``refused`` for a ``MemoryInputError`` and
    ``errored`` for any other type, each with ``_modified`` when the item's channel is in *modified*, and
    ``unattributed`` (with the channel) when that frame is not a public item.
    """
    its = items if isinstance(items, _Items) else _Items(items)
    if not text:
        return []
    return _outcomes(
        _exits(text, its, tuple(_roots(roots))),
        its,
        frozenset(modified),
    )


# --- the record ------------------------------------------------------------------------------------------


def _reached(its: _Items, items: set[str], channels: set[str]) -> set[str]:
    """The items of the pin *items* and *channels* reach: a channel key reaches each function of its
    channel and each name the channel re-exports; ``*`` reaches every item; ``?`` none.
    """
    if "*" in channels:
        return set(its.ids)
    out = {i for i in items if i in its.known}
    for channel in channels:
        out |= {f"env/{channel}:{n}" for n in its.by_channel.get(channel, ())}
        prefix = f"env/{channel}:"
        for name in its.reexports:
            if name.startswith(prefix):
                iid = its.item(channel, name[len(prefix) :])
                if iid is not None:
                    out.add(iid)
    return out


def _ok_after(actions: Iterable[Any]) -> dict[str, int]:
    """Per memory channel, the latest cell holding an action the environment recorded as ``ok``."""
    latest: dict[str, int] = {}
    for n, a in enumerate(actions):
        if n >= MAX_ACTIONS:
            break
        get = (
            a.get if isinstance(a, dict) else (lambda k, d=None, a=a: getattr(a, k, d))
        )
        cell = get("cell", -1)
        if isinstance(cell, bool) or not isinstance(cell, int) or get("status") != "ok":
            continue
        channel = env_channel(get("kind", "tool"), get("channel"))
        if channel is not None and cell > latest.get(channel, -1):
            latest[channel] = cell
    return latest


def request_use(
    lines: Iterable[Any],
    items: Iterable[str],
    actions: Iterable[Any] = (),
    *,
    memory_diff: str = "",
    export_roots: Iterable[str] = (),
    surface: Any = None,
    shown: Any = None,
    cell_status: Any = None,
) -> dict:
    """The request's ``memory_use`` record (see the module docstring); deterministic and bounded.

    *lines* are the request's transcript lines, *items* the item ids at its pin, *actions* its recorded
    actions (dicts or objects with ``cell``, ``kind``, ``channel`` and ``status``), *memory_diff* what it
    wrote into its export, *export_roots* the export's path(s) as the cells import it (:func:`roots_of`),
    *surface* the library's import surface at the pin (:func:`library_surface`), *shown* what the
    memory-section renderer recorded (:func:`record_shown`; None for a legacy recording) and
    *cell_status* each cell's status from the runtime's structured result (:func:`runtime_status`,
    :func:`dict_status`, :func:`failed_status`), by tool call id: a mapping, or the record's own
    ``cell_status`` list. A cell it does not cover is ``unknown`` (cause from the harness's reply, else
    ``other``), and so are the outcomes of the items that cell's code could reach, in this request only.
    """
    lines = list(lines)
    its = _Items(items, surface)
    roots = _roots(export_roots)
    changed, diff_truncated = modified_channels(memory_diff, its)
    tally = _Tally()
    sessions: dict[Any, dict] = {}
    cells = transcript_cells(lines)
    statuses = _statuses(cells[:MAX_CELLS], cell_status, its, _reply_causes(lines))
    unparsed = 0
    refusals: list[tuple[int, str]] = []
    # what the cells whose outcome is unknown could reach (every item past MAX_CELLS: not read)
    reach_items: set[str] = set()
    reach_channels: set[str] = {"*"} if len(cells) > MAX_CELLS else set()
    cells_unknown = 0
    for cell, status in zip(cells[:MAX_CELLS], statuses):
        idx = cell["index"]
        if status["fresh"]:
            sessions.pop(status["session"], None)
        bindings = sessions.setdefault(status["session"], {})
        visitor: _Cell | None = None
        # could have run, but its code was not read: it reaches every item
        opaque = False
        if str(cell["language"]).lower() in ("python", "py", "python3"):
            code = cell["code"]
            tree = None
            if len(code) > MAX_CODE_CHARS:
                unparsed += 1
                opaque = True
            else:
                try:
                    tree = ast.parse(code)
                except (SyntaxError, ValueError):  # does not compile: it never ran
                    unparsed += 1
                except (RecursionError, MemoryError):
                    unparsed += 1
                    opaque = True
            if tree is not None:
                visitor = _Cell(its, bindings, tally, idx)
                try:
                    for stmt in tree.body:
                        visitor.visit(stmt)
                except RecursionError:
                    unparsed += 1
                    opaque = True
        if _outcome_unknown(status):
            cells_unknown += 1
            if opaque:
                reach_channels.add("*")
            elif visitor is not None:
                reach_items |= visitor.touched_items
                reach_channels |= visitor.touched_channels
        for outcome, what in _outcomes(status["exits"], its, frozenset(changed)):
            if outcome == "unattributed":
                tally.add(tally.unattributed, what)
            else:
                tally.bump(what, outcome, idx)
                if outcome == "refused":
                    refusals.append((idx, what))
    latest_ok = _ok_after(actions)
    for idx, iid in refusals:
        channel = iid.split(":", 1)[0][len("env/") :]
        if latest_ok.get(channel, -1) > idx:
            tally.row(iid)["refused_then_accepted"] += 1
    for iid, row in tally.items.items():
        row["modified_in_request"] = iid.split(":", 1)[0][len("env/") :] in changed
    prompts = _system_prompts(lines)
    source, section, shown_record, shown_items, shown_channels = _exposure(
        prompts,
        shown,
        its,
    )
    rows = {k: tally.items[k] for k in sorted(tally.items)}
    truncated = (
        len(rows) > MAX_ITEM_ROWS
        or len(its.ids) > MAX_ITEMS_AT_PIN
        or len(cells) > MAX_CELLS
    )
    star = its.star or {}
    by_cause = dict.fromkeys(CAUSES, 0)
    for st in statuses:
        if st.get("cause") in by_cause:
            by_cause[st["cause"]] += 1
    unread = sum(st["status"] == "unread" for st in statuses)
    unknown_items = sorted(_reached(its, reach_items, reach_channels))
    return {
        "version": VERSION,
        "export_roots": roots,
        "items_at_pin": its.ids[:MAX_ITEMS_AT_PIN],
        "items_at_pin_count": len(its.ids),
        "surface": {
            "star": {k: star[k] for k in sorted(star)[:MAX_CHANNEL_KEYS]},
            "reexports": dict(sorted(its.reexports.items())[:MAX_REEXPORTS]),
        },
        "memory_section_shown": section,
        "exposure_source": source,
        "shown_record": shown_record,
        "shown_items": shown_items,
        "shown_channels": shown_channels,
        "modified_channels": changed,
        "memory_diff_truncated": diff_truncated,
        "cells": len(cells),
        "unparsed_cells": unparsed,
        "cell_status": statuses,
        "cells_without_metadata": sum(by_cause.values()),
        "cells_without_metadata_by_cause": by_cause,
        "cells_error_unread": unread,
        "cells_outcome_unknown": cells_unknown,
        "items_outcome_unknown": unknown_items[:MAX_ITEMS_AT_PIN],
        "items_outcome_unknown_count": len(unknown_items),
        "outcomes_known": not unknown_items,
        "items": dict(list(rows.items())[:MAX_ITEM_ROWS]),
        "module_imports": dict(sorted(tally.module_imports.items())),
        "unknown_calls": dict(sorted(tally.unknown.items())),
        "unattributed_errors": dict(sorted(tally.unattributed.items())),
        "cell_exposure": {
            "items": sorted(tally.exposed_items)[:MAX_ITEMS_AT_PIN],
            "channels": sorted(tally.exposed_channels)[:MAX_CHANNEL_KEYS],
            "calls": dict(tally.helper_calls),
        },
        "truncated": truncated,
    }


def _jsonl(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    out = []
    for ln in text.splitlines():
        try:
            row = json.loads(ln)
        except _SAFE_LOAD:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _decode_value(value: Any, blobs: Any, episode_id: str) -> Any:
    """Format 2 (episodes.RECORD_FORMAT), strictly and with the standard library only: a reference when
    ``__capped__`` is the dict's only key (its blob checked against its id), an escaped ``__literal__``
    unwrapped. Mirrors ``episodes._decode2``; a missing or corrupt blob raises ValueError.
    """
    if isinstance(value, dict) and len(value) == 1:
        ref = value.get("__capped__")
        if isinstance(ref, dict) and "blob" in ref:
            sha = str(ref["blob"])
            if not blobs.has(sha):
                raise ValueError(f"episode {episode_id}: blob {sha} missing")
            data = blobs.get(sha)
            if hashlib.sha256(data).hexdigest() != sha:
                raise ValueError(f"episode {episode_id}: blob {sha} corrupt")
            return json.loads(data.decode())
        if "__literal__" in value:
            return value["__literal__"]
    return value


def _sole_key_marker(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(value) == 1
        and ("__capped__" in value or "__literal__" in value)
    )


def _holds_reference(row: dict) -> bool:
    """Whether an actions row holds a format-2 marker (a sole-key __capped__ or __literal__ dict)."""
    values = [
        row.get("response"),
        *(row.get("args") or []),
        *(row.get("kwargs") or {}).values(),
    ]
    return any(_sole_key_marker(v) for v in values)


def _decode_row(row: dict, blobs: Any, episode_id: str) -> dict:
    row = dict(row)
    row["args"] = [_decode_value(x, blobs, episode_id) for x in row.get("args") or []]
    row["kwargs"] = {
        k: _decode_value(v, blobs, episode_id)
        for k, v in (row.get("kwargs") or {}).items()
    }
    row["response"] = _decode_value(row.get("response"), blobs, episode_id)
    return row


def use_from_episode_dir(
    path: str | Path,
    items: Iterable[str] | None = None,
    blobs: Any = None,
) -> dict:
    """:func:`request_use` over an exported episode directory (``transcript.jsonl``, ``actions.jsonl``,
    ``memory.diff``), with the export roots, import surface, shown record and cell statuses its
    ``memory_use.json`` recorded.

    *items* default to the ``items_at_pin`` recorded there (empty without one). A record in format 2
    (``UNIFY_MEMORY_V21``; ``record_format`` in ``meta.json``) stores large values as blob references, so its
    actions are resolved through *blobs* (a :class:`..blobs.BlobStore`); without it the call is refused rather
    than count references as values.
    """
    path = Path(path)
    actions = _jsonl(path / "actions.jsonl")
    try:
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, *_SAFE_LOAD):
        meta = None
    if not isinstance(meta, dict):
        if any(_holds_reference(r) for r in actions if isinstance(r, dict)):
            raise ValueError(
                "no readable meta.json, and the actions hold record references: cannot tell the record format",
            )
        meta = {}
    fmt = meta.get("record_format", 1)
    if isinstance(fmt, bool) or fmt not in (1, 2):
        raise ValueError(f"unknown record_format {fmt!r}")
    if fmt == 2:
        if blobs is None:
            raise ValueError(
                "record_format 2: pass blobs to resolve the record's references",
            )
        eid = str(meta.get("episode_id", ""))
        actions = [
            _decode_row(r, blobs, eid) if isinstance(r, dict) else r for r in actions
        ]
    try:
        recorded = json.loads((path / "memory_use.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, *_SAFE_LOAD):
        recorded = {}
    if not isinstance(recorded, dict):
        recorded = {}
    if items is None:
        items = recorded.get("items_at_pin") or []
    try:
        diff = (path / "memory.diff").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        diff = ""
    roots = recorded.get("export_roots")
    return request_use(
        _jsonl(path / "transcript.jsonl"),
        items,
        actions,
        memory_diff=diff,
        export_roots=roots if isinstance(roots, list) else (),
        surface=recorded.get("surface"),
        shown=recorded.get("shown_record"),
        cell_status=recorded.get("cell_status"),
    )
