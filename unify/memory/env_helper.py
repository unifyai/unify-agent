"""The ``env`` package's catalogue: ``env.items`` lists its channels and functions, ``env.get`` opens one function.

In a cell::

    import env
    print(env.items())             # every channel (counts, summary, suspect flag) and function, bounded
    print(env.items("<channel>"))  # one channel's functions, in pages past the bound
    print(env.get("<function>"))   # one function's documentation: signature, input form, sections, inputs

``help(env.items)`` and ``help(env.get)`` say what each takes, returns and raises; ``env/__init__.pyi`` beside this
file is the whole contract as typed Python. Both calls check their arguments' types before they run, and every error
they raise says what was given, what was expected with an example call, and the nearest valid names or the right
call.

Each channel is a module of this package (``from env.<channel> import <function>``). The catalogue is the file the
harness wrote beside it (``.memory/catalog.json``, rendered from the library's commit with the harness's suspect
flags); these calls read only that file, make no network call and no call to the harness, and import only the
standard library and the contract runtime the harness ships beside the catalogue (``.memory/contract.py``). The
harness generates this file as the library's ``env/__init__.py``: edits are discarded.

``env.items()`` is bounded: at most :data:`CATALOG_MAX_TOKENS` estimated tokens (4 characters per token). Past that,
the overview prints the channel lines with their counts and points at ``env.items(channel)``, which lists one
channel's functions under the same bound, in pages (``page=2``, ...). Channels are in name order and functions in the
catalogue's order, so the same library always prints the same pages.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections.abc import Callable
from typing import Any

#: The catalogue and the contract runtime, relative to the library root (the parent of this package).
CATALOG_FILE = ".memory/catalog.json"
CONTRACT_FILE = ".memory/contract.py"
#: The library root: the export's root when this file is its ``env/__init__.py``.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: The module name the export's copy of the contract runtime is loaded under.
_CONTRACT_MODULE = "_env_contract"


def _load_contract() -> Any:
    """The contract runtime the harness wrote beside the catalogue, loaded by its path (once per process)."""
    import importlib.util

    found = sys.modules.get(_CONTRACT_MODULE)
    if found is not None:
        return found
    path = os.path.join(_ROOT, CONTRACT_FILE)
    spec = importlib.util.spec_from_file_location(_CONTRACT_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError("env: the library's contract runtime is missing; help() on a function still documents it")
    module = importlib.util.module_from_spec(spec)
    # registered before it runs: its dataclasses look their own module up by name
    sys.modules[_CONTRACT_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_CONTRACT_MODULE, None)
        raise
    return module


if __package__ and __package__.split(".")[0] == "unify":
    from unify.common.contract_runtime import (  # noqa: F401 - the calls' contract
        Ambiguous,
        ArgumentError,
        ContractError,
        NotFound,
        WrongCall,
        call_text,
        checked,
        nearest,
        render_pyi,
    )
else:  # the library's env/__init__.py: the runtime the harness wrote beside the catalogue
    _contract = _load_contract()
    Ambiguous = _contract.Ambiguous
    ArgumentError = _contract.ArgumentError
    ContractError = _contract.ContractError
    NotFound = _contract.NotFound
    WrongCall = _contract.WrongCall
    call_text = _contract.call_text
    checked = _contract.checked
    nearest = _contract.nearest
    render_pyi = _contract.render_pyi

__all__ = ["Catalogue", "Channel", "Function", "get", "items"]

# What one env.items() print may cost, in estimated tokens (ceil(characters / 4)); past it the view pages.
CATALOG_MAX_TOKENS = 1500
_CATALOG_MAX_CHARS = CATALOG_MAX_TOKENS * 4
_LINE_MAX_CHARS = 300  # one listed line in a paged view (a long summary is cut with "…")

SUSPECT_NOTE = "suspect: the environment changed since these were built; verify before use"
_EMPTY = "Memory library: empty (no channel holds a function yet). Do the work directly.\n"
_FOOTER = (
    "Import with `from env.<channel> import <function>`. `help(<function>)` or "
    '`env.get("env.<channel>.<function>")` shows a function\'s documentation and runnable '
    "example; `memory.find(value)` lists the functions built on inputs shaped like data you hold. Notes are "
    "in `env/<channel>/NOTES.md`; `README.md` in the library root holds the same list. Functions are "
    "candidates, not authority: check the example before relying on one. A MemoryInputError means the input "
    "differs from what the function was built from; then do the work directly. The library is a scratch "
    "copy: edits to it, or a new `proposals/<name>.md`, are discarded when the request ends but recorded for "
    "the next consolidation.\n"
)
_CHANNEL_TAIL = (
    "Import with `from env.<channel> import <function>`; `help(<function>)` or "
    '`env.get("env.<channel>.<function>")` shows one with its example.\n'
)
#: What every lookup error says to do next.
_BROWSE = "env.items() lists every channel and function."
_DOC_SECTIONS = (
    ("Args", "args"),
    ("Returns", "returns"),
    ("Raises", "raises"),
    ("Example", "example"),
    ("Use when", "use_when"),
    ("Don't use when", "dont_use_when"),
    ("Notes", "notes"),
)


# --- the typed results ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Channel:
    """One channel of the catalogue: the module ``env.<name>``, its function and note counts, its module's one-line
    summary, and whether the harness holds it suspect (the environment changed since its functions were built).
    ``print()`` shows its catalogue line."""

    name: str
    functions: int
    notes: int
    summary: str
    suspect: bool

    def __str__(self) -> str:
        return channel_line(
            {"channel": self.name, "functions": self.functions, "notes": self.notes, "summary": self.summary},
            self.suspect,
        )


@dataclasses.dataclass(frozen=True, repr=False)
class Function:
    """One function of the catalogue, as ``env.get`` returns it: its full name (``env.<channel>.<function>``), its
    channel, signature, one-line summary, input form (what its first parameter takes), effect and whether its channel
    is suspect. ``print()`` shows its documentation (``text``): the signature, summary, input form, effect, docstring
    sections and the shapes of the inputs it was built and checked on."""

    name: str
    channel: str
    signature: str
    summary: str
    input: str
    effect: str
    suspect: bool
    text: str

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


@dataclasses.dataclass(frozen=True, repr=False)
class Catalogue:
    """What ``env.items`` returns. ``print()`` shows the listing (``text``), at most :data:`CATALOG_MAX_TOKENS`
    estimated tokens a page. ``channels`` and ``functions`` hold the whole view as data, whatever the page shows:
    every channel and function for the overview, or the one channel and its functions. ``channel`` is the channel
    the view lists (None for the overview); ``page`` is this page of ``pages``."""

    text: str
    channels: tuple[Channel, ...]
    functions: tuple[Function, ...]
    channel: str | None
    page: int
    pages: int

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text


# --- the catalogue file --------------------------------------------------------------------------------------------


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def channel_line(row: dict, suspect: bool = False) -> str:
    """One channel of the catalogue (a ``channels`` row of ``.memory/catalog.json``): its module, function and note
    counts, summary and, when *suspect*, the drift flag."""
    counts = _plural(row["functions"], "function")
    if row["notes"]:
        counts += ", " + _plural(row["notes"], "note")
    line = f"- `env.{row['channel']}`: {counts}"
    if row["summary"]:
        line += f". {row['summary']}"
    if suspect:
        line += f" ({SUSPECT_NOTE})"
    return line


def _read(written: str) -> dict:
    """The catalogue beside this package; a file that cannot be read is a :class:`NotFound` naming no path."""
    try:
        with open(os.path.join(_ROOT, CATALOG_FILE), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise NotFound(
            written,
            f"the library's catalogue cannot be read ({type(exc).__name__}).",
            hint="help() on an imported function still shows its documentation.",
        ) from None
    if not isinstance(data, dict):
        raise NotFound(
            written,
            "the library's catalogue cannot be read (not a JSON object).",
            hint="help() on an imported function still shows its documentation.",
        )
    return data


def _channels(cat: dict) -> list[dict]:
    return sorted(
        (r for r in cat.get("channels", []) if isinstance(r, dict) and isinstance(r.get("channel"), str)),
        key=lambda r: r["channel"],
    )


def _functions(cat: dict) -> list[dict]:
    return [
        e
        for e in cat.get("functions", [])
        if isinstance(e, dict) and isinstance(e.get("module"), str) and isinstance(e.get("name"), str)
    ]


def _full_name(e: dict) -> str:
    return f"{e['module']}.{e['name']}"


def _suspect(cat: dict, channel: Any) -> bool:
    return any(c.get("suspect") and c.get("channel") == channel for c in _channels(cat))


def _channel_of(row: dict) -> Channel:
    return Channel(
        row["channel"],
        int(row.get("functions") or 0),
        int(row.get("notes") or 0),
        str(row.get("summary") or ""),
        bool(row.get("suspect")),
    )


# --- one function's documentation ----------------------------------------------------------------------------------


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
        what = f"columns {', '.join(cols)}" if cols else f"{len(s.get('types') or [])} columns, no header"
        return f"a {ext or 'delimited'} table with {what}"
    if fmt in ("json", "yaml", "jsonl"):
        tree = s.get("tree")
        if isinstance(tree, dict):
            return f"a {fmt} file with keys " + ", ".join(sorted(tree))
        return f"a {fmt} file of lists or records"
    if fmt == "xlsx":
        return "a workbook with sheets " + ", ".join(str(x.get("name")) for x in s.get("sheets") or [])
    return f"a {fmt} file"


def _indent(text: str) -> str:
    return "\n".join(("    " + ln) if ln.strip() else "" for ln in text.splitlines())


def _card(e: dict, cat: dict) -> str:
    """One function's documentation as ``print()`` shows it."""
    doc = e.get("doc") or {}
    out = [f"{e['module']}.{e.get('signature', e['name'])}", e.get("summary", "")]
    if doc.get("description"):
        out += ["", doc["description"]]
    form = e.get("input", "")
    meaning = (cat.get("input_forms") or {}).get(form, "")
    out.append("")
    if form:
        out.append(f"Input: {form}" + (f" ({meaning})" if meaning else ""))
    if e.get("effect"):
        out.append(f"Effect: {e['effect']}")
    for title, key in _DOC_SECTIONS:
        if doc.get(key):
            out += [f"{title}:", _indent(doc[key])]
    if _suspect(cat, e.get("channel")):
        out.append(f"Channel {e['module']} is {SUSPECT_NOTE}.")
    shapes = e.get("input_shapes")
    if shapes:
        out.append("Built and checked on inputs shaped as:")
        out += [f"    - {_shape_text(d)}" for d in shapes]
    if doc.get("example"):
        out.append(
            f"Example paths are relative to the library root ({_ROOT}); check the example before "
            "relying on the function.",
        )
    return "\n".join(out).rstrip() + "\n"


def _function_of(e: dict, cat: dict) -> Function:
    return Function(
        _full_name(e),
        str(e.get("channel") or ""),
        str(e.get("signature") or e["name"]),
        str(e.get("summary") or ""),
        str(e.get("input") or ""),
        str(e.get("effect") or ""),
        _suspect(cat, e.get("channel")),
        _card(e, cat),
    )


# --- the listing ---------------------------------------------------------------------------------------------------


def _function_line(e: dict) -> str:
    line = f"- `{e['module']}.{e.get('signature') or e['name']}`"
    if e.get("summary"):
        line += f": {e['summary']}"
    if e.get("input"):
        line += f" (input: {e['input']})"
    return line + "\n"


def _cut(line: str) -> str:
    """*line* (ending in a newline) at most :data:`_LINE_MAX_CHARS` characters."""
    if len(line) <= _LINE_MAX_CHARS:
        return line
    return line[: _LINE_MAX_CHARS - 2] + "…\n"


def _pages(lines: list[str], room: int) -> list[list[str]]:
    """*lines* packed in order into pages of at most *room* characters (every page holds one line at least)."""
    pages: list[list[str]] = [[]]
    used = 0
    for line in lines:
        if pages[-1] and used + len(line) > room:
            pages.append([])
            used = 0
        pages[-1].append(line)
        used += len(line)
    return pages


def _paged(head: str, lines: list[str], tail: str, page: int, call: str, written: str) -> tuple[str, int]:
    """(*head*, page *page* of *lines* and *tail*, at most :data:`_CATALOG_MAX_CHARS` characters; the number of
    pages). A page that is not the last says how to print the next (``env.items(<call>page=N)``)."""
    lines = [_cut(ln) for ln in lines]
    nav = "- ... page 999999 of 999999 (999999 more lines): " f"env.items({call}page=999999) prints the next\n"
    room = max(_CATALOG_MAX_CHARS - len(head) - len(tail) - len(nav), _LINE_MAX_CHARS)
    pages = _pages(lines, room)
    if not 1 <= page <= len(pages):
        raise NotFound(
            written,
            f"no page {page!r}; pages run from 1 to {len(pages)}.",
            expected=f"page: an int from 1 to {len(pages)}",
            example=f"env.items({call}page={len(pages)})",
        )
    body = "".join(pages[page - 1])
    if page < len(pages):
        rest = sum(len(p) for p in pages[page:])
        body += (
            f"- ... page {page} of {len(pages)} ({rest} more lines): "
            f"env.items({call}page={page + 1}) prints the next\n"
        )
    elif len(pages) > 1:
        body += f"- (page {page} of {len(pages)}, the last)\n"
    return head + body + tail, len(pages)


def _forms_line(cat: dict) -> str:
    forms = cat.get("input_forms") or {}
    if not forms:
        return ""
    return (
        "Input forms (what a function's first parameter takes): "
        + "; ".join(f"`{k}` {v}" for k, v in forms.items())
        + ".\n"
    )


def _channel_name(channel: str | Channel) -> str:
    if isinstance(channel, Channel):
        return channel.name
    name = channel.strip().strip("`")
    if name.startswith("env."):
        name = name[4:]
    elif name.startswith("env/"):
        name = name[4:]
    return name.rstrip("/")


def _overview(cat: dict, rows: list[dict], fns: list[dict], page: int, written: str) -> Catalogue:
    order = {r["channel"]: n for n, r in enumerate(rows)}
    fns = sorted(fns, key=lambda e: order.get(e.get("channel"), len(order)))  # stable: the catalogue's order
    channels = tuple(_channel_of(r) for r in rows)
    functions = tuple(_function_of(e, cat) for e in fns)
    head = (
        f"Memory library: {_plural(len(rows), 'channel')}, {_plural(len(fns), 'function')}, at {_ROOT} "
        "(first on the import path).\nChannels:\n"
    )
    lines = [channel_line(r, bool(r.get("suspect"))) + "\n" for r in rows]
    full = head + "".join(lines)
    if fns:
        full += "Functions:\n" + "".join(_function_line(e) for e in fns)
    full += _forms_line(cat) + _FOOTER
    if len(full) <= _CATALOG_MAX_CHARS and page == 1:
        return Catalogue(full, channels, functions, None, 1, 1)
    tail = (
        f"Functions: {len(fns)}, not listed here: the whole list is over this view's bound of "
        f"{CATALOG_MAX_TOKENS} tokens. Use memory.find(value) or env.items(channel) for the rest "
        '(e.g. env.items("env.<channel>")).\n' + _FOOTER
    )
    text, pages = _paged(head, lines, tail, page, "", written)
    return Catalogue(text, channels, functions, None, page, pages)


def _one_channel(cat: dict, row: dict, fns: list[dict], page: int, written: str) -> Catalogue:
    name = row["channel"]
    mine = [e for e in fns if e.get("channel") == name]
    head = (
        f"Memory library, channel `env.{name}` (at {_ROOT}):\n"
        + _cut(channel_line(row, bool(row.get("suspect"))) + "\n")
        + ("Functions:\n" if mine else "")
    )
    tail = _forms_line(cat) + _CHANNEL_TAIL
    text, pages = _paged(head, [_function_line(e) for e in mine], tail, page, f"{name!r}, ", written)
    return Catalogue(
        text,
        (_channel_of(row),),
        tuple(_function_of(e, cat) for e in mine),
        name,
        page,
        pages,
    )


@checked(name="env.items")
def items(channel: str | Channel | None = None, page: int = 1) -> Catalogue:
    """The environment catalogue: every channel of the ``env`` package (its function and note counts, summary and
    suspect flag) and every function (``env.<channel>.<signature>``, summary, input form), then how to use them; or,
    given a channel, that channel's functions.

    Args:
        channel: the channel to list (``"files"``, ``"env.files"``, ``"env/files"`` or a Channel from
            ``env.items().channels``); None lists every channel and function.
        page: the page of a view longer than one print's bound, from 1 (each page names the next).

    Returns:
        Catalogue: ``print()`` shows the listing, at most CATALOG_MAX_TOKENS estimated tokens a page;
        ``.channels`` and ``.functions`` hold the whole view as data.

    Raises:
        NotFound: no channel by that name (the nearest are listed), no such page, or the catalogue cannot be read.
        WrongCall: a function's name given where a channel is expected (open it with env.get).
        ArgumentError: an argument of the wrong type.

    Example:
        print(env.items())
    """
    written = call_text("env.items", [channel] if channel is not None else [], {"page": page} if page != 1 else {})
    cat = _read(written)
    rows = _channels(cat)
    fns = _functions(cat)
    if not rows:  # an empty library prints the same line for any view
        return Catalogue(_EMPTY, (), (), None, 1, 1)
    if channel is None:
        return _overview(cat, rows, fns, page, written)
    return _one_channel(cat, _resolve_channel(_channel_name(channel), rows, fns, written), fns, page, written)


def _resolve_channel(name: str, rows: list[dict], fns: list[dict], written: str) -> dict:
    by_name = {r["channel"]: r for r in rows}
    if name in by_name:
        return by_name[name]
    folded = [r for r in rows if r["channel"].casefold() == name.casefold()]
    if len(folded) == 1:
        return folded[0]
    # a function's name given for a channel: say so and give the call that opens it
    fn, _ = _match_function(name, fns)
    if fn is not None:
        raise WrongCall(
            written,
            "that is a function, not a channel.",
            hint=f"Open it with env.get({_full_name(fn)!r}).",
        )
    names = sorted(by_name)
    near = nearest(name, names)
    raise NotFound(
        written,
        f"no channel {name!r}.",
        expected="a channel's name",
        example=f"env.items({(near or names)[0]!r})",
        nearest=near,
        hint=_BROWSE,
    )


# --- one function ----------------------------------------------------------------------------------------------------


def _wanted(ref: str) -> str:
    """*ref* in the dotted form: ``env/<channel>:<function>``, ``<channel>:<function>`` and ``env.<channel>:<function>``
    become ``env.<channel>.<function>``; a call or signature after the name and backticks are dropped."""
    text = ref.strip().strip("`").strip()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text.replace("/", ".").replace(":", ".").strip(".")


def _match_function(wanted: str, fns: list[dict]) -> tuple[dict | None, list[str]]:
    """(the one function *wanted* names: its full name, ``<channel>.<function>`` or a plain name one channel has;
    the full names of every function a plain name several channels have names). (None, []) for none."""
    parts = wanted.split(".")
    if parts and parts[0] == "env" and len(parts) > 1:
        parts = parts[1:]
    if len(parts) == 2:
        exact = [e for e in fns if e.get("channel") == parts[0] and e["name"] == parts[1]]
        if exact:
            return exact[0], []
        folded = [
            e
            for e in fns
            if str(e.get("channel")).casefold() == parts[0].casefold() and e["name"].casefold() == parts[1].casefold()
        ]
        return (folded[0] if len(folded) == 1 else None), []
    if len(parts) == 1:
        by_name = [e for e in fns if e["name"] == parts[0]]
        if not by_name:
            by_name = [e for e in fns if e["name"].casefold() == parts[0].casefold()]
        if len(by_name) == 1:
            return by_name[0], []
        return None, [_full_name(e) for e in by_name]
    return None, []


@checked(name="env.get")
def get(ref: str | Function | Callable[..., Any]) -> Function:
    """One function of the environment catalogue, with its documentation.

    Args:
        ref: the function: its full name (``"env.files.read_rows"``), ``"files.read_rows"``, its item id
            (``"env/files:read_rows"``), its plain name when one channel has it (``"read_rows"``), the imported
            function itself, or a Function from ``env.items().functions``.

    Returns:
        Function: ``print()`` shows its signature, summary, input form, effect, docstring sections (Args, Returns,
        Raises, Example, ...) and the shapes of the inputs it was built and checked on; its fields hold the same.

    Raises:
        NotFound: no function by that name; the nearest names are listed.
        Ambiguous: a plain name several channels have; each full name is listed.
        WrongCall: a channel's name given where a function is expected (list it with env.items).
        ArgumentError: an argument of the wrong type.

    Example:
        print(env.get(env.items().functions[0]))
    """
    if isinstance(ref, Function):
        wanted, written = ref.name, call_text("env.get", [ref.name])
    elif isinstance(ref, str):
        wanted, written = _wanted(ref), call_text("env.get", [ref])
    else:  # the imported function: by the module it was imported from, else by its own name
        module = str(getattr(ref, "__module__", "") or "")
        name = str(getattr(ref, "__name__", "") or "")
        wanted = f"{module}.{name}" if module.startswith("env.") else name
        written = f"env.get({name or type(ref).__name__})"
    cat = _read(written)
    fns = _functions(cat)
    found, several = _match_function(wanted, fns)
    if several:
        raise Ambiguous(
            written,
            f"{wanted!r} names {len(several)} functions.",
            nearest=several,
            hint=f"Pass one of them, e.g. env.get({several[0]!r}).",
        )
    if found is not None:
        return _function_of(found, cat)
    if not fns:
        raise NotFound(
            written,
            f"no function {wanted!r}: the library holds no function yet.",
            hint=_BROWSE,
        )
    channels = {r["channel"] for r in _channels(cat)}
    bare = wanted[4:] if wanted.startswith("env.") else wanted
    if bare in channels:
        raise WrongCall(
            written,
            "that is a channel, not a function.",
            hint=f"List its functions with env.items({bare!r}).",
        )
    full = [_full_name(e) for e in fns]
    near = nearest(wanted, full)
    raise NotFound(
        written,
        f"no function {wanted!r} in the catalogue.",
        expected="a function's full name, <channel>.<function>, or its plain name when one channel has it",
        example=f"env.get({(near or full)[0]!r})",
        nearest=near,
        hint=_BROWSE,
    )


# --- the typed stub --------------------------------------------------------------------------------------------------

#: The stub's opening comment.
STUB_HEADER = (
    "env: the environment catalogue's calls and types (generated by the harness; edits are discarded).\n"
    "print() shows each result; its fields hold the same as data."
)


def stub() -> str:
    """``env/__init__.pyi``: the typed contract of :func:`items` and :func:`get` and their results, in one place."""
    return render_pyi(
        {"items": items, "get": get, "Catalogue": Catalogue, "Channel": Channel, "Function": Function},
        header=STUB_HEADER,
    )
