"""How one request used the memory library: a pure function of its transcript (memory v2.1, stage 1).

Standard library only, and no other module of this package is imported, so the file can be copied
into an offline analyser as it is. The harness computes the record at a request's ``finish``
(:func:`request_use` over the request's transcript lines, the memory items at its pin and its recorded
actions) and the offline funnel analyser recomputes it from an exported episode directory
(:func:`use_from_episode_dir`); both run the same code.

Everything is structural. Imports and calls come from each cell's syntax tree; errors come from the
frames of the traceback the cell's result records (its ``error`` field), by file path and function
name. No message, output or other text is read for meaning, and nothing is keyed on a task.

Per memory item (``env/<channel>:<name>``, a public function of ``env/<channel>/__init__.py``):

* ``imported``: import statements that bind the item by name (``from env.<channel> import name``,
  aliases included, and ``from env.<channel> import *``);
* ``called``: call sites in cell code whose callee resolves to the item (a bound name or alias, an
  ``env.<channel>.name`` or ``<module alias>.name`` attribute, or a plain assignment of one of these);
  a static count of sites, not of executions;
* ``referenced``: other loads of the item (passed as a value, ``help(f)``, assigned);
* ``guarded``: those calls that sit inside a ``try`` body with an ``except`` clause (a refusal caught
  there leaves no traceback, so it cannot be counted);
* ``refused``: exceptions whose type is ``MemoryInputError`` and which left the item into cell code
  (see :func:`attribute_errors`); ``errored``: any other exception that left the item;
* ``refused_then_accepted``: refusals followed, in a later cell of the same request, by an action on the
  item's channel that the environment recorded as ``ok`` (a channel-level proxy for "the environment
  accepted the input when it was handled directly"; the input itself is not compared).

Calls the syntax tree cannot resolve to one item (``getattr(module, name)(...)``, ``vars(module)``,
``importlib.import_module("env...")``, a module's ``__dict__``) are counted per channel in
``unknown_calls`` and never guessed; ``*`` is a dynamic access of the ``env`` package itself. Channel keys
are channels of the pin, ``*``, or ``?`` for a name the pin has no channel for (no text from cell code is
kept beyond item ids and channel names).

``memory_section_shown`` describes the memory index in the request's system prompts: the text from the
index's fixed header to the end of the prompt (the harness appends the index last), as a SHA-256, its
UTF-8 byte count and the index's own token estimate (characters / 4).
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "HEADER_PREFIX",
    "VERSION",
    "attribute_errors",
    "env_channel",
    "memory_section",
    "parse_traceback",
    "request_use",
    "transcript_cells",
    "use_from_episode_dir",
]

VERSION = 1

#: The start of the index's header (``unify.memory_v2.index.HEADER``); a test keeps the two in step.
HEADER_PREFIX = "Memory library: candidates to check, not authority."

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

_ITEM_ID = re.compile(r"^env/([A-Za-z_][A-Za-z0-9_]*):([A-Za-z_][A-Za-z0-9_]*)\Z")
_FRAME = re.compile(r'^  File "(?P<file>.*)", line \d+, in (?P<func>.+)$')
_TYPE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)(?::|$)")
_HEADER = "Traceback (most recent call last):"
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


# --- the items -----------------------------------------------------------------------------------------


class _Items:
    """The memory items at the request's pin, by channel."""

    def __init__(self, ids: Iterable[str]) -> None:
        self.ids: list[str] = []
        self.by_channel: dict[str, list[str]] = {}
        for raw in sorted({i for i in ids if isinstance(i, str)}):
            m = _ITEM_ID.match(raw)
            if m is None:
                continue
            self.ids.append(raw)
            self.by_channel.setdefault(m.group(1), []).append(m.group(2))
        self.known = frozenset(self.ids)

    def item(self, channel: str, name: str) -> str | None:
        iid = f"env/{channel}:{name}"
        return iid if iid in self.known else None


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


# --- cells from the transcript ---------------------------------------------------------------------------


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


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
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _result_error(content: Any) -> str | None:
    """The ``error`` field of an ``execute_code`` result: the JSON object its content starts with."""
    raw = _text(content).lstrip()
    if not raw.startswith("{"):
        return None
    try:
        meta, _ = json.JSONDecoder().raw_decode(raw)
    except ValueError:
        return None
    err = meta.get("error") if isinstance(meta, dict) else None
    return err if isinstance(err, str) and err else None


def transcript_cells(lines: Iterable[Any]) -> list[dict]:
    """Every ``execute_code`` cell with a result, in result order (the episode's cell indices):
    ``{"index", "code", "language", "error"}``, where ``error`` is the traceback the result recorded.
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
            code, lang = pending.pop(msg["tool_call_id"])
            cells.append(
                {
                    "index": len(cells),
                    "code": code,
                    "language": lang,
                    "error": _result_error(msg.get("content")),
                },
            )
    return cells


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


def memory_section(prompts: Iterable[str], header: str = HEADER_PREFIX) -> dict:
    """The memory index the system prompts carried: ``{"shown", "sha256", "bytes", "est_tokens",
    "prompts", "distinct"}`` of the first one (the text from the header's last occurrence to the end).
    """
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


# --- imports and calls -----------------------------------------------------------------------------------

# A binding: ("item", item id) | ("module", channel) | ("package",) | ("dynamic", channel or "*")
_Binding = tuple


class _Tally:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.module_imports: dict[str, int] = {}
        self.unknown: dict[str, int] = {}
        self.unattributed: dict[str, int] = {}

    def row(self, item: str) -> dict:
        r = self.items.get(item)
        if r is None:
            r = self.items[item] = {
                "imported": 0,
                "called": 0,
                "referenced": 0,
                "guarded": 0,
                "refused": 0,
                "errored": 0,
                "refused_then_accepted": 0,
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
    """Walks one cell in statement order; *bindings* persist from earlier cells of the request."""

    def __init__(
        self,
        items: _Items,
        bindings: dict,
        tally: _Tally,
        cell: int,
    ) -> None:
        self.items, self.b, self.t, self.cell = items, bindings, tally, cell
        self.guard = 0

    # -- names ------------------------------------------------------------------------------------------
    def _key(self, channel: str) -> str:
        """A channel as a record key: a channel of the pin, ``*`` (the package), else ``?``."""
        return channel if channel == "*" or channel in self.items.by_channel else "?"

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
                for name in self.items.by_channel.get(channel, []):
                    iid = f"env/{channel}:{name}"
                    self._bind(name, ("item", iid))
                    self.t.bump(iid, "imported", self.cell)
                continue
            iid = self.items.item(channel, alias.name)
            target = alias.asname or alias.name
            if iid is None:
                self._unbind(target)
                continue
            self._bind(target, ("item", iid))
            self.t.bump(iid, "imported", self.cell)

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

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        self._scoped(self._args(node.args), list(node.body))
        self._unbind(node.name)

    visit_FunctionDef = visit_AsyncFunctionDef = _function

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d]:
            self.visit(d)
        self._scoped(self._args(node.args), [node.body])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self._scoped([], list(node.body))
        self._unbind(node.name)

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
    def visit_Call(self, node: ast.Call) -> None:
        target = self.resolve(node.func)
        if target is not None and target[0] == "item":
            self.t.bump(target[1], "called", self.cell)
            if self.guard:
                self.t.bump(target[1], "guarded", self.cell)
        elif target is not None and target[0] == "dynamic":
            self.t.add(self.t.unknown, self._key(target[1]))
        if target is None or target[0] not in ("item", "module", "package"):
            self.visit(node.func)
        for a in node.args:
            self.visit(a)
        for k in node.keywords:
            self.visit(k.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        target = self.resolve(node) if isinstance(node.ctx, ast.Load) else None
        if target is not None and target[0] == "item":
            self.t.bump(target[1], "referenced", self.cell)
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            target = self.b.get(node.id)
            if target is not None and target[0] == "item":
                self.t.bump(target[1], "referenced", self.cell)
        else:
            self._unbind(node.id)


# --- errors ----------------------------------------------------------------------------------------------


def parse_traceback(text: str) -> list[dict]:
    """The exceptions a formatted traceback shows, oldest first: ``{"frames": [(file, function)], "type"}``.

    Only the traceback's structure is read: a block starts at the header line (the first one, or one
    after a chaining line of CPython's own format), its frames are the ``File "...", line N, in f`` lines,
    and its type is the dotted name the exception line starts with. The message is never read.
    """
    if not isinstance(text, str) or len(text) > MAX_TRACEBACK_CHARS:
        return []
    lines = text.splitlines()
    blocks: list[dict] = []
    i, prev = 0, None
    while i < len(lines):
        line = lines[i]
        if line == _HEADER and (not blocks or prev in _CHAIN):
            frames: list[tuple[str, str]] = []
            i += 1
            while i < len(lines) and lines[i].startswith(" "):
                m = _FRAME.match(lines[i])
                if m is not None:
                    frames.append((m.group("file"), m.group("func")))
                i += 1
            m = _TYPE.match(lines[i]) if i < len(lines) else None
            blocks.append({"frames": frames, "type": m.group(1) if m else None})
            prev = None
            i += 1
            continue
        if line.strip():
            prev = line.strip()
        i += 1
    return blocks


def _module_channel(path: str, items: _Items) -> str | None:
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[-1] == "__init__.py" and parts[-3] == "env":
        return parts[-2] if parts[-2] in items.by_channel else None
    return None


def attribute_errors(
    text: str | None,
    items: Iterable[str] | _Items,
) -> list[tuple[str, str]]:
    """``(outcome, item or channel)`` per exception of a cell's traceback that left a memory item.

    An exception left an item when its outermost frame is outside the memory modules (cell code) and
    some later frame is in ``env/<channel>/__init__.py``; the first such frame names the item (its
    function). An exception whose outermost frame is itself in a memory module was caught inside it,
    so it is not counted. ``outcome`` is ``refused`` for a ``MemoryInputError``, ``errored`` for any
    other type, and ``unattributed`` (with the channel) when that frame is not a public item.
    """
    its = items if isinstance(items, _Items) else _Items(items)
    if not text:
        return []
    out: list[tuple[str, str]] = []
    for block in parse_traceback(text):
        frames = block["frames"]
        if not frames or _module_channel(frames[0][0], its) is not None:
            continue
        for path, func in frames[1:]:
            channel = _module_channel(path, its)
            if channel is None:
                continue
            iid = its.item(channel, func)
            if iid is None:
                out.append(("unattributed", channel))
            else:
                kind = (block["type"] or "").rsplit(".", 1)[-1]
                out.append(("refused" if kind == REFUSAL_TYPE else "errored", iid))
            break
    return out


# --- the record ------------------------------------------------------------------------------------------


def _action_rows(actions: Iterable[Any]) -> list[tuple[int, str | None, str]]:
    rows: list[tuple[int, str | None, str]] = []
    for a in actions:
        if len(rows) >= MAX_ACTIONS:
            break
        get = (
            a.get if isinstance(a, dict) else (lambda k, d=None, a=a: getattr(a, k, d))
        )
        cell = get("cell", -1)
        if isinstance(cell, bool) or not isinstance(cell, int):
            continue
        rows.append(
            (cell, env_channel(get("kind", "tool"), get("channel")), get("status")),
        )
    return rows


def request_use(
    lines: Iterable[Any],
    items: Iterable[str],
    actions: Iterable[Any] = (),
) -> dict:
    """The request's ``memory_use`` record (see the module docstring); deterministic and bounded.

    *lines* are the request's transcript lines, *items* the item ids at its pin, *actions* its recorded
    actions (dicts or objects with ``cell``, ``kind``, ``channel`` and ``status``).
    """
    lines = list(lines)
    its = _Items(items)
    tally = _Tally()
    bindings: dict = {}
    cells = transcript_cells(lines)
    unparsed = 0
    refusals: list[tuple[int, str]] = []
    for cell in cells[:MAX_CELLS]:
        idx = cell["index"]
        if str(cell["language"]).lower() not in ("python", "py", "python3"):
            continue
        code = cell["code"]
        try:
            if len(code) > MAX_CODE_CHARS:
                raise ValueError("cell too large")
            tree = ast.parse(code)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            unparsed += 1
            tree = None
        if tree is not None:
            visitor = _Cell(its, bindings, tally, idx)
            try:
                for stmt in tree.body:
                    visitor.visit(stmt)
            except RecursionError:
                unparsed += 1
        for outcome, what in attribute_errors(cell["error"], its):
            if outcome == "unattributed":
                tally.add(tally.unattributed, what)
            else:
                tally.bump(what, outcome, idx)
                if outcome == "refused":
                    refusals.append((idx, what))
    acts = _action_rows(actions)
    for idx, iid in refusals:
        channel = iid.split(":", 1)[0][len("env/") :]
        if any(c > idx and ch == channel and st == "ok" for c, ch, st in acts):
            tally.row(iid)["refused_then_accepted"] += 1
    rows = {k: tally.items[k] for k in sorted(tally.items)}
    truncated = len(rows) > MAX_ITEM_ROWS or len(its.ids) > MAX_ITEMS_AT_PIN
    if len(cells) > MAX_CELLS:
        truncated = True
    return {
        "version": VERSION,
        "items_at_pin": its.ids[:MAX_ITEMS_AT_PIN],
        "items_at_pin_count": len(its.ids),
        "memory_section_shown": memory_section(_system_prompts(lines)),
        "cells": len(cells),
        "unparsed_cells": unparsed,
        "items": dict(list(rows.items())[:MAX_ITEM_ROWS]),
        "module_imports": dict(sorted(tally.module_imports.items())),
        "unknown_calls": dict(sorted(tally.unknown.items())),
        "unattributed_errors": dict(sorted(tally.unattributed.items())),
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
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def use_from_episode_dir(path: str | Path, items: Iterable[str] | None = None) -> dict:
    """:func:`request_use` over an exported episode directory (``transcript.jsonl``, ``actions.jsonl``).

    *items* default to the ``items_at_pin`` its ``memory_use.json`` recorded (empty without one).
    """
    path = Path(path)
    if items is None:
        try:
            recorded = json.loads(
                (path / "memory_use.json").read_text(encoding="utf-8"),
            )
        except (FileNotFoundError, ValueError):
            recorded = {}
        items = (
            (recorded.get("items_at_pin") or []) if isinstance(recorded, dict) else []
        )
    return request_use(
        _jsonl(path / "transcript.jsonl"),
        items,
        _jsonl(path / "actions.jsonl"),
    )
