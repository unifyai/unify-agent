r"""``UNIFY_ESCAPE_DRIFT_CHECK``: refuse a stored literal retyped with one extra escape level.

A storage review copies code from the session's own cells into the source it
stores, typing it inside a JSON tool argument. On the lean-all office
attempts of 6 Oct (research artifact
``runtime-20261006/escaped-newline-v1/DIAGNOSIS.md``) it typed one escape
level too many in 16 of 41 writes that held an escape, against 0 of 74 task
cells: the cell's ``''.join(n+'\n' for n in qual)`` was stored as
``"".join(name + "\\n" ...)`` and ``r'\bERROR\b'`` as ``r"\\bERROR\\b"``.
The harness stores the source byte for byte, so the function wrote a literal
backslash-n into every later output file and its regex matched nothing.

While a storage review runs (:func:`enter`, from where the review is
started in ``code_act_actor``) this module keeps the string literals of the
reviewed session's Python code cells (``origin_capture._code_cells``). A
literal in the source being stored -- plain, raw, or the text of an
f-string -- has *drifted* when all of these hold:

- it holds a backslash and its value is in none of the cells' literals;
- removing one level of backslash escaping from it (:func:`one_level_less`)
  gives the value of a cell literal;
- that one-level-less value is not also written elsewhere in the same
  source (a function that escapes newlines on purpose writes both).

``FunctionManager.add_functions`` (and so ``patch_function``, which stores
through it) refuses a function with a drifted literal, naming both
spellings; the review can store it again. The comparison is only between
the write and the code of the same conversation: nothing about any request,
task or format is read. Offline, the rule flagged 15 of the 16 doubled
writes and 0 of the 49 others. Without the session's cells, or with the
switch off, nothing is checked.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import re
from dataclasses import dataclass
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple

MAX_NAMED = 5
"""The most drifted literals one refusal names."""

MAX_SHOWN_CHARS = 80
"""Characters of a literal's spelling shown in a refusal."""

_PREFIX = re.compile(r"([A-Za-z]*)['\"]")
# One level of escaping in a non-raw literal's value, as Python reads it.
_ESCAPE = re.compile(
    r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[0-7]{1,3}|[\\'\"abfnrtv])",
)
_SIMPLE = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
}


@dataclass(frozen=True)
class Literal:
    """A string literal's value, whether it was written raw, and its spelling."""

    value: str
    raw: bool
    text: str


def _is_raw(text: str) -> bool:
    match = _PREFIX.match(text)
    return bool(match) and "r" in match.group(1).lower()


def literals(code: str) -> List[Literal]:
    """The string literals of *code*: plain and raw strings, and each f-string's text parts."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []
    out: List[Literal] = []
    parts = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = ast.get_source_segment(code, node) or ""
        raw = _is_raw(text)
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.add(id(part))
                out.append(Literal(part.value, raw, text))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in parts
        ):
            text = ast.get_source_segment(code, node) or ""
            out.append(Literal(node.value, _is_raw(text), text))
    return out


def _unescape(match: "re.Match[str]") -> str:
    escape = match.group(1)
    if escape in _SIMPLE:
        return _SIMPLE[escape]
    if escape[0] in "xuU":
        return chr(int(escape[1:], 16))
    return chr(int(escape, 8))


def one_level_less(literal: Literal) -> str:
    """*literal*'s value with one level of backslash escaping removed.

    Raw: a doubled backslash becomes one. Otherwise each backslash escape
    Python reads in a string (``\\n``, ``\\t``, ``\\\\``, ``\\x41``, ...)
    becomes the character it stands for; any other backslash stays.
    """
    if literal.raw:
        return literal.value.replace("\\\\", "\\")
    return _ESCAPE.sub(_unescape, literal.value)


def cell_literals(cells: Sequence[str]) -> Mapping[str, str]:
    """Each literal value in *cells*, with the first spelling that wrote it."""
    seen: dict[str, str] = {}
    for code in cells:
        for literal in literals(code):
            seen.setdefault(literal.value, literal.text)
    return seen


def drifted(source: str, seen: Mapping[str, str]) -> List[Tuple[str, str]]:
    """``(spelling in source, spelling in the cells)`` for each drifted literal of *source*."""
    if not seen:
        return []
    found = literals(source)
    own = {literal.value for literal in found}
    hits: List[Tuple[str, str]] = []
    named = set()
    for literal in found:
        value = literal.value
        if "\\" not in value or value in seen:
            continue
        less = one_level_less(literal)
        if less == value or less not in seen or less in own:
            continue
        if literal.text in named:
            continue
        named.add(literal.text)
        hits.append((literal.text, seen[less]))
    return hits


def _shown(text: str) -> str:
    if len(text) <= MAX_SHOWN_CHARS:
        return text
    return text[: MAX_SHOWN_CHARS - 1] + "…"


def refusal(name: str, hits: Sequence[Tuple[str, str]]) -> str:
    """The store's refusal for *name*'s drifted literals: both spellings, nothing else."""
    pairs = [
        f"`{_shown(written)}` where "
        + ("the session's own code" if i == 0 else "it")
        + f" wrote `{_shown(cell)}`"
        for i, (written, cell) in enumerate(hits[:MAX_NAMED])
    ]
    listed = (
        pairs[0] if len(pairs) == 1 else ", ".join(pairs[:-1]) + " and " + pairs[-1]
    )
    more = len(hits) - MAX_NAMED
    if more > 0:
        listed += f" ({more} more like these)"
    return (
        f"'{name}' was not stored, because its source writes {listed} "
        "(one extra backslash at each escape)."
    )


# ---------------------------------------------------------------------------
# The reviewed session's cells, while its storage review runs
# ---------------------------------------------------------------------------

_SEEN: contextvars.ContextVar[Optional[Mapping[str, str]]] = contextvars.ContextVar(
    "unify_escape_drift_cells",
    default=None,
)


def session_cells(trajectory: Sequence[Mapping[str, Any]]) -> List[str]:
    """The code of the trajectory's Python cells, in order."""
    from .origin_capture import _code_cells

    return [
        code
        for _index, code, language, _output in _code_cells(trajectory)
        if language.lower() == "python"
    ]


def enter(
    trajectory: Optional[Sequence[Mapping[str, Any]]],
) -> "contextvars.Token[Optional[Mapping[str, str]]]":
    """Check writes against *trajectory*'s cells from now on (and in the tasks started now).

    With no cell literal in the trajectory, nothing is kept. Undo with :func:`leave`.
    """
    seen: Optional[Mapping[str, str]] = None
    if trajectory:
        try:
            seen = cell_literals(session_cells(trajectory)) or None
        except Exception:  # noqa: BLE001 - a check, never a reason to fail a review
            seen = None
    return _SEEN.set(seen)


def leave(token: "contextvars.Token[Optional[Mapping[str, str]]]") -> None:
    _SEEN.reset(token)


@contextlib.contextmanager
def reviewing(trajectory: Optional[Sequence[Mapping[str, Any]]]) -> Iterator[None]:
    """:func:`enter` for the block."""
    token = enter(trajectory)
    try:
        yield
    finally:
        leave(token)


def current() -> Optional[Mapping[str, str]]:
    """The cell literals writes are checked against now, or ``None``."""
    return _SEEN.get()


def check(name: str, source: str) -> None:
    """Raise ``ValueError`` with :func:`refusal` when *source* has a drifted literal."""
    seen = current()
    if not seen:
        return
    hits = drifted(source, seen)
    if hits:
        raise ValueError(refusal(name, hits))


__all__ = [
    "Literal",
    "cell_literals",
    "check",
    "current",
    "drifted",
    "enter",
    "leave",
    "literals",
    "one_level_less",
    "refusal",
    "reviewing",
    "session_cells",
]
