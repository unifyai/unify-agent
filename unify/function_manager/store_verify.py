"""Verify before store (``UNIFY_STORE_VERIFY=package.module:factory``): a function is stored only after it has
passed on a task it was not written from.

A storage review writes a function from one finished session, so it cannot tell which of the values it saw are
the procedure and which are that one request's details (a phone number, a date format, a folder). Stored as they
are, such values make the function do the previous request's work on the next one. Values that the session's
credentials made up (a login token, a password) are worse: they are wrong everywhere else. This switch adds a
gate between the review and the library:

1. **Static checks** (this module, nothing environment-specific): a literal in the function body that matches a
   part of the session's request that differs in a sibling request (a request of the same kind the environment
   knows) is refused and named, and the review is asked for a parameter; so is a literal equal to one of the
   session's credentials or shaped like a token, a write to a global, and a credential parameter (one the
   environment names, such as ``access_token``) that does not default to ``None``.
2. **A run on a held-out task**: the environment's verifier (the factory's object) runs the function, with the
   arguments the review chose, on a fresh copy of a sibling task's world and judges the outcome with the
   environment's own check. A pass is recorded under the sha256 of the function's exact source.
3. **The gate**: ``add_functions`` refuses a function whose exact source has no recorded pass, naming why.

The review reaches (1) and (2) through ``FunctionManager.check_function`` (the tool
``FunctionManager_check_function``, offered to the review only while the switch is set). The switch needs
``UNIFY_STORE_ADMISSION`` too, so that the only writer is a review the environment admitted: with it, in-session
writes are withheld, so every stored function has passed through the gate.

A verifier is any object with

``held_out(name) -> HeldOut | mapping``
    whether a held-out task is available now and, if so, the texts the static checks compare (the session's
    request, the sibling requests), the held-out task's request (shown to the review so it can choose the
    arguments), the parameter names the environment treats as credentials, and the session's credentials
    (compared only, never shown);
``run(candidate, call_kwargs) -> Verdict | mapping``
    load the :class:`Candidate` against the held-out world (``candidate.load(primitives, extra_globals)``), call
    it with ``call_kwargs`` and judge the outcome.

With the switch unset nothing here is imported by the storage path, no tool is added and every prompt is the
shipped one.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

MAX_RUN_CHECKS = 4
"""Run checks (``check_function`` with ``call_kwargs``) allowed per process, that is per session and its review."""

MIN_STRING_LITERAL = 3
"""Shorter string literals (separators, single letters) are never compared with the requests."""

LOG_METHODS = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
        "print",
    },
)

_TOKEN_RE = re.compile(r"[^\s\"'`“”‘’,;()\[\]{}<>]+")
_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*")
_STRFTIME = {
    "%Y": "YYYY",
    "%y": "YY",
    "%m": "MM",
    "%d": "DD",
    "%H": "HH",
    "%I": "HH",
    "%M": "MM",
    "%S": "SS",
    "%b": "MON",
    "%B": "MONTH",
    "%j": "DDD",
}
_FORMAT_FIELD = re.compile(r"\{[^{}]*\}")
_SPLIT = "\x00"


class StoreVerifyError(RuntimeError):
    """The verifier cannot be loaded, or the switch is set without store admission."""


# ---------------------------------------------------------------------------
# What a verifier receives and returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOut:
    """What the environment says about the held-out task for the current review."""

    available: bool
    reason: str = ""
    source_texts: tuple[str, ...] = ()
    sibling_texts: tuple[str, ...] = ()
    held_out_text: str = ""
    credential_params: tuple[str, ...] = ()
    secrets: tuple[str, ...] = field(default=(), repr=False)
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "HeldOut":
        if isinstance(value, HeldOut):
            return value
        if not isinstance(value, Mapping):
            return cls(
                False,
                f"the verifier answered {type(value).__name__}, not a held-out description",
            )

        def texts(key: str) -> tuple[str, ...]:
            raw = value.get(key) or ()
            if isinstance(raw, str):
                raw = (raw,)
            return tuple(str(t) for t in raw if t is not None)

        return cls(
            available=value.get("available") is True,
            reason=str(value.get("reason") or ""),
            source_texts=texts("source_texts"),
            sibling_texts=texts("sibling_texts"),
            held_out_text=str(value.get("held_out_text") or ""),
            credential_params=texts("credential_params"),
            secrets=texts("secrets"),
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True)
class Verdict:
    """The verifier's judgement of one run on the held-out task."""

    ok: bool
    reason: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "Verdict":
        if isinstance(value, Verdict):
            return value
        if not isinstance(value, Mapping):
            return cls(
                False,
                f"the verifier answered {type(value).__name__}, not a verdict",
            )
        return cls(
            ok=value.get("ok") is True,
            reason=str(value.get("reason") or ""),
            details=dict(value.get("details") or {}),
        )


@dataclass(frozen=True)
class Candidate:
    """A function the review wants to store, as the verifier sees it.

    ``load(primitives, extra_globals)`` returns the raw function loaded the way a search loads it (its stored
    callees injected), in a scratch namespace whose ``primitives`` (and any ``extra_globals``) the verifier
    supplies, so every environment call the function makes reaches the held-out world instead of the session's.
    ``effects`` are the effect labels of the environment methods it names (``read``, ``write``,
    ``destructive``).
    ``signature`` describes a source declaration without executing it. Defaults are literal
    values and annotations are declared text, not resolved runtime types. ``(...)`` means the
    declaration or a default/annotation is unsupported or unknown. It is not a runtime callable
    or safety certificate.
    """

    name: str
    source: str
    signature: str
    depends_on: tuple[str, ...]
    effects: tuple[str, ...]
    loader: Callable[[Any, Optional[Mapping[str, Any]]], Callable[..., Any]] = field(
        repr=False,
        compare=False,
    )

    @property
    def sha256(self) -> str:
        return source_sha256(self.source)

    def load(
        self,
        primitives: Any,
        extra_globals: Optional[Mapping[str, Any]] = None,
    ) -> Callable[..., Any]:
        return self.loader(primitives, extra_globals)


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The process's verifier, run-check budget and recorded passes
# ---------------------------------------------------------------------------

_lock = threading.RLock()
_verifier: Any = None
_loaded_spec: Optional[str] = None
_run_checks = 0
_passes: dict[str, dict] = {}


def spec() -> str:
    from unify.settings import SETTINGS

    return str(getattr(SETTINGS, "UNIFY_STORE_VERIFY", "") or "").strip()


def enabled() -> bool:
    return bool(spec())


def verifier() -> Any:
    """The verifier the factory named by ``UNIFY_STORE_VERIFY`` returned (created once per process)."""
    global _verifier, _loaded_spec
    current = spec()
    if not current:
        raise StoreVerifyError("UNIFY_STORE_VERIFY is not set")
    with _lock:
        if _verifier is not None and _loaded_spec == current:
            return _verifier
        from unify.settings import SETTINGS

        if not str(getattr(SETTINGS, "UNIFY_STORE_ADMISSION", "") or "").strip():
            raise StoreVerifyError(
                "UNIFY_STORE_VERIFY needs UNIFY_STORE_ADMISSION: without it the session "
                "itself could store functions that never passed the check",
            )
        module_name, _, attr = current.partition(":")
        if not module_name or not attr:
            raise StoreVerifyError(
                f"UNIFY_STORE_VERIFY {current!r} is not 'package.module:factory'",
            )
        try:
            target: Any = importlib.import_module(module_name)
            for part in attr.split("."):
                target = getattr(target, part)
            made = target()
        except Exception as exc:
            raise StoreVerifyError(
                f"UNIFY_STORE_VERIFY {current!r} failed: {type(exc).__name__}: {exc}",
            ) from exc
        for method in ("held_out", "run"):
            if not callable(getattr(made, method, None)):
                raise StoreVerifyError(
                    f"the verifier from {current!r} has no {method}()",
                )
        _verifier, _loaded_spec = made, current
        return made


def held_out(name: str) -> HeldOut:
    try:
        return HeldOut.coerce(verifier().held_out(name))
    except StoreVerifyError:
        raise
    except Exception as exc:
        return HeldOut(
            False,
            f"the verifier failed: {type(exc).__name__}: {str(exc)[:300]}",
        )


def take_run_check() -> Optional[int]:
    """Use one run check; the number used, or ``None`` when the budget is spent."""
    global _run_checks
    with _lock:
        if _run_checks >= MAX_RUN_CHECKS:
            return None
        _run_checks += 1
        return _run_checks


def run_checks_left() -> int:
    with _lock:
        return MAX_RUN_CHECKS - _run_checks


def record_pass(sha256: str, entry: Mapping[str, Any]) -> None:
    with _lock:
        _passes[sha256] = dict(entry)


def passed(sha256: str) -> Optional[dict]:
    with _lock:
        found = _passes.get(sha256)
        return dict(found) if found is not None else None


def reset() -> None:
    """Forget the verifier, the budget and the passes (tests)."""
    global _verifier, _loaded_spec, _run_checks
    with _lock:
        _verifier = None
        _loaded_spec = None
        _run_checks = 0
        _passes.clear()


# ---------------------------------------------------------------------------
# Static check (a): literals that are details of one request
# ---------------------------------------------------------------------------


def strftime_pattern(fmt: str) -> str:
    """``%Y-%m-%d`` as the request would write it: ``YYYY-MM-DD``."""
    out = fmt
    for code, shown in _STRFTIME.items():
        out = out.replace(code, shown)
    return out


def _has_strftime(text: str) -> bool:
    return any(code in text for code in _STRFTIME)


def _format_string_pieces(text: str) -> list[str]:
    """A ``str.format`` template's literal pieces, a date field (``{:%Y-%m-%d}``) rebuilt as its pattern."""

    def field_text(match: re.Match) -> str:
        inner = match.group(0)[1:-1]
        _, colon, spec_text = inner.partition(":")
        if colon and _has_strftime(spec_text):
            return strftime_pattern(spec_text)
        return _SPLIT

    rebuilt = _FORMAT_FIELD.sub(field_text, text)
    return [p for p in rebuilt.split(_SPLIT)]


def _joined_pieces(node: ast.JoinedStr) -> list[str]:
    """An f-string's literal pieces: text parts joined, a date field rebuilt as its pattern (``YYYY-MM-DD``),
    any other field a break."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            spec_node = value.format_spec
            spec_text = ""
            if isinstance(spec_node, ast.JoinedStr) and all(
                isinstance(v, ast.Constant) for v in spec_node.values
            ):
                spec_text = "".join(str(v.value) for v in spec_node.values)
            parts.append(
                strftime_pattern(spec_text) if _has_strftime(spec_text) else _SPLIT,
            )
        else:
            parts.append(_SPLIT)
    return "".join(parts).split(_SPLIT)


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    names += [p.arg for p in (a.vararg, a.kwarg) if p is not None]
    return set(names)


def _is_log_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "print"
    if isinstance(func, ast.Attribute):
        return func.attr in LOG_METHODS
    return False


def _refers_to(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(n, ast.Name) and n.id in names for n in ast.walk(node))


@dataclass(frozen=True)
class Literal:
    text: str
    line: int
    kind: str  # "string", "number", "pattern"


def body_literals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Literal]:
    """The function's constants that could be a request's details: string and number literals of its body and
    of its parameter defaults, and the date patterns its format strings build. Left out: the docstring, the
    messages of print/log calls, ``raise`` and ``assert``, and constants compared with a parameter (a check of an
    input against the values it may take)."""
    params = _parameters(node)
    skip: set[int] = set()
    found: list[Literal] = []
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        skip.update(id(n) for n in ast.walk(body[0]))
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and _is_log_call(n):
            for arg in (*n.args, *(k.value for k in n.keywords)):
                skip.update(id(x) for x in ast.walk(arg))
        elif isinstance(n, ast.Raise):
            for part in (n.exc, n.cause):
                if part is not None:
                    skip.update(id(x) for x in ast.walk(part))
        elif isinstance(n, ast.Assert) and n.msg is not None:
            skip.update(id(x) for x in ast.walk(n.msg))
        elif isinstance(n, ast.Compare):
            sides = [n.left, *n.comparators]
            if any(isinstance(s, ast.Name) and s.id in params for s in sides):
                for side in sides:
                    if not _refers_to(side, params):
                        skip.update(id(x) for x in ast.walk(side))
    joined_inner: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.JoinedStr):
            for inner in ast.walk(n):
                if inner is not n:
                    joined_inner.add(id(inner))
    defaults = [
        *node.args.defaults,
        *[d for d in node.args.kw_defaults if d is not None],
    ]
    targets: list[ast.AST] = [*body, *defaults]
    for top in targets:
        for n in ast.walk(top):
            if id(n) in skip:
                continue
            line = getattr(n, "lineno", 0)
            if isinstance(n, ast.JoinedStr):
                if id(n) in joined_inner:
                    continue
                for piece in _joined_pieces(n):
                    if piece.strip():
                        found.append(Literal(piece, line, "pattern"))
                continue
            if id(n) in joined_inner or not isinstance(n, ast.Constant):
                continue
            value = n.value
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                found.append(Literal(str(value), line, "number"))
            elif isinstance(value, str):
                found.append(Literal(value, line, "string"))
                if _has_strftime(value):
                    found.append(Literal(strftime_pattern(value), line, "pattern"))
                if "{" in value and "}" in value:
                    for piece in _format_string_pieces(value):
                        if piece.strip() and piece != value:
                            found.append(Literal(piece, line, "pattern"))
    return found


def request_tokens(text: str) -> list[str]:
    """A request's words and values, lower-cased, stripped of quotes, brackets and trailing punctuation."""
    tokens = []
    for raw in _TOKEN_RE.findall(text or ""):
        token = raw.rstrip(".:!?").lower()
        if token:
            tokens.append(token)
    return tokens


def _bare(token: str) -> str:
    return token.lstrip("$€£#")


def differing_spans(source: str, sibling: str) -> list[tuple[list[str], list[str]]]:
    """The spans of ``source`` that a sibling request writes differently: (source tokens, sibling tokens)."""
    a, b = request_tokens(source), request_tokens(sibling)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [
        (a[i1:i2], b[j1:j2])
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal" and i2 > i1
    ]


def _literal_in_token(literal: str, kind: str, token: str) -> bool:
    token = _bare(token)
    if kind == "number":
        return literal == token
    if len(literal) >= MIN_STRING_LITERAL and literal in token:
        return True
    return len(token) >= 4 and not token.isalpha() and token in literal


def _literal_in_span(literal: str, kind: str, tokens: Sequence[str]) -> bool:
    joined = " ".join(tokens)
    if kind == "number":
        return any(literal == _bare(t) for t in tokens)
    return literal in joined or any(_literal_in_token(literal, kind, t) for t in tokens)


def varying_literals(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_texts: Sequence[str],
    sibling_texts: Sequence[str],
) -> list[Literal]:
    """Literals of the function that match a part of one request which another request of the same kind writes
    differently (and which that other version does not contain): values of that one request. Every ordered pair of
    the session's request and the sibling requests is compared, so a literal copied from a sibling (the held-out
    request, for example) is refused as well as one copied from the session's own request.
    """
    texts = [t for t in (*source_texts, *sibling_texts) if t]
    spans = [
        span
        for i, first in enumerate(texts)
        for j, second in enumerate(texts)
        if i != j
        for span in differing_spans(first, second)
    ]
    found: list[Literal] = []
    seen: set[str] = set()
    for literal in body_literals(node):
        text = literal.text.strip().lower()
        kind = literal.kind
        if kind != "number" and len(text) < MIN_STRING_LITERAL:
            continue
        if text in seen:
            continue
        for source_span, sibling_span in spans:
            if any(
                _literal_in_token(text, kind, t) for t in source_span
            ) and not _literal_in_span(
                text,
                kind,
                sibling_span,
            ):
                seen.add(text)
                found.append(literal)
                break
    return found


# ---------------------------------------------------------------------------
# Static check (b): credentials
# ---------------------------------------------------------------------------


def _entropy(text: str) -> float:
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return -sum(c / len(text) * math.log2(c / len(text)) for c in counts.values())


def looks_like_token(text: str) -> bool:
    """A JWT, or a long unbroken string of mixed letters and digits with high entropy (a key, a token)."""
    if _JWT_RE.search(text):
        return True
    if len(text) < 24 or any(ch.isspace() for ch in text):
        return False
    if not (any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text)):
        return False
    return _entropy(text) >= 3.5


def _all_string_constants(node: ast.AST) -> list[tuple[str, int]]:
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out.append((n.value, getattr(n, "lineno", 0)))
    return out


def credential_problems(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    secrets: Iterable[str] = (),
    credential_params: Iterable[str] = (),
) -> list[str]:
    """What in the function would carry this session's credentials into the library, one message each (a secret
    is never quoted)."""
    problems: list[str] = []
    known = [s for s in (str(x) for x in secrets) if len(s) >= 4]
    flagged: set[int] = set()
    for text, line in _all_string_constants(node):
        if any(
            secret == text or (len(secret) >= 6 and secret in text) for secret in known
        ):
            problems.append(
                f"the string literal on line {line} is one of this session's credentials (a password or an "
                f"access token); a stored function must log in itself and hold no credential",
            )
            flagged.add(line)
        elif looks_like_token(text) and line not in flagged:
            problems.append(
                f"the string literal on line {line} looks like a token or key; a stored function must log in "
                f"itself and hold no credential",
            )
            flagged.add(line)
    for n in ast.walk(node):
        if isinstance(n, ast.Global):
            problems.append(
                f"`global {', '.join(n.names)}` (line {n.lineno}): a stored function must not keep state "
                f"(such as a token) in module globals between calls",
            )
    wanted = set(credential_params)
    if wanted:
        a = node.args
        positional = [*a.posonlyargs, *a.args]
        pos_defaults = [None] * (len(positional) - len(a.defaults)) + list(a.defaults)
        pairs = list(zip(positional, pos_defaults)) + list(
            zip(a.kwonlyargs, a.kw_defaults),
        )
        for param, default in pairs:
            if param.arg not in wanted:
                continue
            if not (isinstance(default, ast.Constant) and default.value is None):
                problems.append(
                    f"the parameter `{param.arg}` is a credential here: it must default to None, because the "
                    f"check calls the function without it; the function must log in itself when it is None",
                )
    return problems


def static_problems(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    held: HeldOut,
) -> list[str]:
    """Every static refusal for a function against the held-out description, one message each."""
    problems = []
    for literal in varying_literals(node, held.source_texts, held.sibling_texts):
        problems.append(
            f"the literal {literal.text!r} (line {literal.line}) is a detail of one request that another "
            f"request of the same kind states differently; make it a parameter and pass the value the request gives",
        )
    problems += credential_problems(
        node,
        secrets=held.secrets,
        credential_params=held.credential_params,
    )
    return problems


def doctrine() -> str:
    """The one sentence the storage review's doctrine gains while the switch is set."""
    return (
        "`FunctionManager_add_functions` stores a function only after that exact source has passed "
        "`FunctionManager_check_function` on a held-out task of the same kind: call it first without "
        "`call_kwargs` (static checks, and the held-out task's request), then with the `call_kwargs` that "
        "request needs; any value that differs between such requests must be a parameter, and the function "
        "must log in itself."
    )


__all__ = [
    "Candidate",
    "HeldOut",
    "MAX_RUN_CHECKS",
    "StoreVerifyError",
    "Verdict",
    "body_literals",
    "credential_problems",
    "differing_spans",
    "doctrine",
    "enabled",
    "held_out",
    "looks_like_token",
    "passed",
    "record_pass",
    "request_tokens",
    "reset",
    "run_checks_left",
    "source_sha256",
    "static_problems",
    "strftime_pattern",
    "take_run_check",
    "varying_literals",
    "verifier",
]
