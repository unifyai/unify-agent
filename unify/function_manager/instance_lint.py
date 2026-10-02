"""Instance identifiers kept out of stored skills (``UNIFY_STORE_INSTANCE_LINT``).

A stored function or guidance entry is meant for later tasks of the same
kind. Storage reviews sometimes wrote the identifiers of the task in front of
them into what they stored: an aliased task id (``task-7a4cf12e``) in a
docstring, a playlist title quoted in the request (``"R&B Recommendation"``)
hard-coded in the code. No later task shares those values.

A top-level ``act()`` reads its request (the session's first user message)
once and keeps its *instance tokens* in the task context, which the task loop,
its tools and its storage reviews inherit; a sub-agent keeps the tokens of the
task it works for. Two kinds are taken:

* id-like tokens: UUIDs, ``word-<hex>`` aliases (and their hex part), hex runs
  of 8 or more with a digit, and runs of 6 or more digits that are not round
  numbers (``1000000`` is a limit, not an id);
* quoted instance strings: text of 12 to 200 characters between double,
  single or typographic quotes that reads like a name or title (a capitalised
  word after the first, a digit or a symbol such as ``&``); a lone identifier
  or dotted path (``request_demonstration``, ``primitives.x.y``) and plain
  lowercase prose are domain vocabulary and are not taken.

Words of the domain (an app name, ``grid``) are never tokens: only values of
this instance are. The checks are regular expressions over the request and an
``ast`` walk of the function; nothing is called.
"""

from __future__ import annotations

import ast
import contextvars
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Union

_UUID = re.compile(
    r"(?<![0-9a-z])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(?![0-9a-z])",
)
_ALIAS = re.compile(r"(?<![0-9a-z-])([a-z][a-z0-9]*)-([0-9a-f]{6,})(?![0-9a-z])")
_HEX = re.compile(r"(?<![0-9a-z])[0-9a-f]{8,}(?![0-9a-z])")
_DIGITS = re.compile(r"(?<![0-9a-z.])[0-9]{6,}(?![0-9a-z])")
_QUOTED = (
    re.compile(r'(?<![\w"])"([^"\n]{12,200})"(?![\w"])'),
    re.compile(r"(?<![\w'])'([^'\n]{12,200})'(?![\w'])"),
    re.compile(r"“([^“”\n]{12,200})”"),
    re.compile(r"‘([^‘’\n]{12,200})’"),
)
_IDENTIFIER_LIKE = re.compile(r"[A-Za-z_][\w.\-]*")
_SYMBOLS = re.compile(r"[&@#/+]")
_WS = re.compile(r"\s+")
# A round number (``1000000``, ``250000``) is a limit or a size, not an id.
_ROUND = re.compile(r"[0-9]{1,2}0+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

# Shapes that name a task instance in any function name, whatever the request.
_NAME_ALIAS = re.compile(r"(?:^|_)(task_?[0-9a-f]{6,})(?:_|$)")
_NAME_UUID = re.compile(
    r"[0-9a-f]{8}_?[0-9a-f]{4}_?[0-9a-f]{4}_?[0-9a-f]{4}_?[0-9a-f]{12}",
)
# The same shapes in free text (docstrings, guidance).
_TEXT_ALIAS = re.compile(r"(?<![0-9a-z])task[-_][0-9a-f]{6,}(?![0-9a-z])")


@dataclass(frozen=True)
class InstanceTokens:
    """What identifies the current task instance, lowercased."""

    ids: tuple = ()
    quoted: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.ids or self.quoted)


_CURRENT: contextvars.ContextVar[Optional[InstanceTokens]] = contextvars.ContextVar(
    "unify_instance_tokens",
    default=None,
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_STORE_INSTANCE_LINT", False))


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _texts(request: Any) -> Iterator[str]:
    if isinstance(request, str):
        yield request
    elif isinstance(request, dict):
        for value in request.values():
            yield from _texts(value)
    elif isinstance(request, (list, tuple)):
        for value in request:
            yield from _texts(value)


def _reads_like_a_name(text: str) -> bool:
    if _IDENTIFIER_LIKE.fullmatch(text):
        return False
    words = text.split()
    return (
        _has_digit(text)
        or bool(_SYMBOLS.search(text))
        or any(word[:1].isupper() for word in words[1:])
    )


def tokens_of(request: Any) -> InstanceTokens:
    """The instance tokens of a request (a string, or messages/dicts of strings)."""
    ids: List[str] = []
    quoted: List[str] = []

    def add(bucket: List[str], value: str) -> None:
        if value not in bucket:
            bucket.append(value)

    for text in _texts(request):
        lowered = text.lower()
        for match in _UUID.finditer(lowered):
            add(ids, match.group(0))
        for match in _ALIAS.finditer(lowered):
            if _has_digit(match.group(2)):
                add(ids, match.group(0))
                add(ids, match.group(2))
        for match in _HEX.finditer(lowered):
            if _has_digit(match.group(0)) and not _ROUND.fullmatch(match.group(0)):
                add(ids, match.group(0))
        for match in _DIGITS.finditer(lowered):
            if not _ROUND.fullmatch(match.group(0)):
                add(ids, match.group(0))
        for pattern in _QUOTED:
            for match in pattern.finditer(text):
                value = _WS.sub(" ", match.group(1)).strip()
                if len(value) >= 12 and _reads_like_a_name(value):
                    add(quoted, value.lower())
    # Longest first, so an alias is reported before its hex part.
    ids.sort(key=len, reverse=True)
    return InstanceTokens(ids=tuple(ids), quoted=tuple(quoted))


def enter(request: Any) -> Optional[contextvars.Token]:
    """Keep *request*'s tokens for the current task, unless a task already set them."""
    if not enabled() or _CURRENT.get() is not None:
        return None
    return _CURRENT.set(tokens_of(request))


def leave(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _CURRENT.reset(token)
    except ValueError:
        # Reset from another context (a handle cleaned up elsewhere).
        pass


def current() -> InstanceTokens:
    return _CURRENT.get() or InstanceTokens()


def _slug(text: str) -> str:
    return _NON_ALNUM.sub("_", text.lower()).strip("_")


def _id_in(token: str, text: str) -> bool:
    return re.search(rf"(?<![0-9a-z]){re.escape(token)}(?![0-9a-z])", text) is not None


def _found_in_text(text: str, tokens: InstanceTokens) -> Optional[str]:
    """The first instance token *text* contains, or a task-alias/UUID shape."""
    lowered = _WS.sub(" ", text.lower())
    for token in tokens.ids:
        if _id_in(token, lowered):
            return token
    for value in tokens.quoted:
        if value in lowered:
            return value
    for pattern in (_TEXT_ALIAS, _UUID):
        match = pattern.search(lowered)
        if match:
            return match.group(0)
    return None


def name_problem(name: str, tokens: Optional[InstanceTokens] = None) -> Optional[str]:
    """Why *name* identifies a task instance, or ``None``."""
    tokens = current() if tokens is None else tokens
    lowered = name.lower()
    for token in tokens.ids:
        slug = _slug(token)
        hit = (
            re.search(rf"(?<![0-9]){slug}(?![0-9])", lowered)
            if slug.isdigit()
            else slug in lowered
        )
        if hit:
            return (
                f"its name contains {token!r}, an identifier from this task's request"
            )
    for value in tokens.quoted:
        slug = _slug(value)
        if len(slug) >= 12 and slug in lowered:
            return f"its name spells {value!r}, a value quoted in this task's request"
    for pattern in (_NAME_ALIAS, _NAME_UUID):
        match = pattern.search(lowered)
        if match:
            return f"its name contains {match.group(0).strip('_')!r}, the shape of a task id"
    return None


def _docstring_node(node: ast.AST) -> Optional[ast.AST]:
    body = getattr(node, "body", None) or []
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[0].value
    return None


def code_problem(
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    tokens: Optional[InstanceTokens] = None,
) -> Optional[str]:
    """Why the code of *node* (literals and defaults, not its docstring) hard-codes the instance."""
    tokens = current() if tokens is None else tokens
    if not tokens:
        return None
    digits = {token for token in tokens.ids if token.isdigit()}
    docstrings = {
        id(doc)
        for inner in ast.walk(node)
        if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for doc in [_docstring_node(inner)]
        if doc is not None
    }
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Constant) or id(inner) in docstrings:
            continue
        value = inner.value
        if isinstance(value, str):
            lowered = _WS.sub(" ", value.lower())
            for token in tokens.ids:
                if _id_in(token, lowered):
                    return (
                        f"its code hard-codes {token!r}, an identifier from this "
                        f"task's request"
                    )
            for quoted in tokens.quoted:
                if quoted in lowered:
                    return (
                        f"its code hard-codes {quoted!r}, a value quoted in this "
                        f"task's request"
                    )
        elif isinstance(value, int) and not isinstance(value, bool):
            if str(value) in digits:
                return (
                    f"its code hard-codes {value}, an identifier from this task's "
                    f"request"
                )
    return None


def text_warning(
    what: str,
    text: str,
    tokens: Optional[InstanceTokens] = None,
) -> Optional[str]:
    """A warning when *text* (a docstring, a guidance entry) names this task instance."""
    tokens = current() if tokens is None else tokens
    found = _found_in_text(text or "", tokens)
    if found is None:
        return None
    return (
        f"{what} names {found!r}, which identifies this task instance and no "
        f"later task shares; describe the kind of input instead and drop the "
        f"identifier (patch it out)"
    )


def refusal(name: str, problem: str) -> str:
    return (
        f"'{name}' was not stored, because {problem}, and no later task will "
        f"share it. Take the value as a parameter (or drop it) and name the "
        f"function for what it does, then add it again."
    )


def join_warnings(warnings: Iterable[Optional[str]]) -> Optional[str]:
    kept = [w for w in warnings if w]
    return "; ".join(kept) if kept else None
