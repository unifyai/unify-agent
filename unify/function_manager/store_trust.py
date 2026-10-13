"""Helpers for recorded calls of stored functions: argument hashes, failure reasons and caller faults.

The function cases (:mod:`unify.function_manager.store_cases`) hash a call's arguments
(:func:`input_hash`), keep a failure as ``Type: message`` (:func:`failure_reason`) and do not hold a
failure against the function when the caller caused it (:func:`caller_fault`): arguments that do not fit
the function's signature (read from its stored source by :func:`source_signature`), or an unfilled
placeholder such as ``"{{access_token}}"``, ``"${API_KEY}"``, ``"$token"`` or ``"<password>"`` passed for
a parameter whose name marks it as a credential (:func:`looks_like_placeholder`, :func:`credential_key`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from typing import Any, Mapping, Optional, Sequence

REASON_LIMIT = 300
"""Characters of a failure's ``Type: message`` that are kept."""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable(value: Any) -> Any:
    """A JSON stand-in for a value JSON cannot hold: its repr, or its type when the repr is an address."""
    text = repr(value)
    return f"<{type(value).__name__}>" if " at 0x" in text else text


def input_hash(arguments: Mapping[str, Any]) -> str:
    """The sha256 of a call's arguments bound to parameter names (``f(1)`` and ``f(x=1)`` hash alike)."""
    try:
        text = json.dumps(dict(arguments), sort_keys=True, default=_stable)
    except Exception:  # noqa: BLE001 - unsortable keys and the like
        text = _stable(dict(arguments))
    return sha256(text)


def failure_reason(error: Any) -> str:
    """``Type: message`` of an exception, or the last line of a traceback, truncated."""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        lines = [line.strip() for line in str(error).strip().splitlines()]
        text = next((line for line in reversed(lines) if line), "failed")
    return text if len(text) <= REASON_LIMIT else text[: REASON_LIMIT - 3] + "..."


_CREDENTIAL_NAME = re.compile(
    r"token|key|secret|passw(?:or)?d|auth|credential",
    re.IGNORECASE,
)
_CREDENTIAL_WORD = re.compile(
    r"(?:\w*(?:token|secret|passw(?:or)?d|credential)"
    r"|(?:api|access|auth|client|master|private|secret|session|signing)?key"
    r"|auth(?:ori[sz]ation|entication)?)s?",
    re.IGNORECASE,
)
_WORDS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def credential_key(name: Any) -> bool:
    """Whether a key or parameter ``name`` marks the value it holds as a credential, word by word.

    The words of :data:`_CREDENTIAL_NAME` (``token``, ``key``, ``secret``, ``password``, ``auth``,
    ``credential``) as whole words of the name (split at separators and camel case, a plural allowed), plus
    compounds ending in one (``accesstoken``, ``apikey``, ``Authorization``): ``access_token``, ``apiKey``
    and ``X-Auth-Token`` match; ``author``, ``monkey`` and ``keyword``, which the substring rule for
    shown arguments also withholds, do not. Values are replaced under these names in recorded data, where
    a word that only contains one is usually data a function computes on.
    """
    words = _WORDS.findall(str(name))
    return any(_CREDENTIAL_WORD.fullmatch(word) for word in words)


_TEMPLATE = re.compile(r"^(?:\{\{.*\}\}|\$\{.*\})$", re.DOTALL)
"""``{{access_token}}``, ``${API_KEY}``: a template that was never filled, whatever it names."""
_NAMED_SLOT = re.compile(
    r"^(?:\$([A-Za-z_][\w.]*)|%([A-Za-z_]\w*)%|<([^<>]+)>|(your[\s_-][\w\s-]*))$",
    re.IGNORECASE,
)
"""``$access_token``, ``%TOKEN%``, ``<api key>``, ``YOUR_PASSWORD``: a slot, when what it names is a credential."""
_PLACEHOLDER_WORD = re.compile(
    r"^(?:placeholder|redacted|changeme|undefined|null|none|todo|x{3,}|\.{3}|\*{3,})$",
    re.IGNORECASE,
)


class _Default:
    """Stands in for a default value :func:`source_signature` does not evaluate."""

    def __repr__(self) -> str:
        return "..."


_HAS_DEFAULT = _Default()


def looks_like_placeholder(parameter: str, value: Any) -> bool:
    """Whether ``value``, passed for ``parameter``, is an unfilled stand-in for a credential.

    Only a parameter whose name marks it as a credential is judged, and only a string that is an unfilled
    template (``{{x}}``, ``${x}``), a slot naming a credential (``$access_token``, ``%TOKEN%``,
    ``<api key>``, ``YOUR_PASSWORD``), a placeholder word (``placeholder``, ``redacted``, ``xxx``...), or
    the parameter's own name. A real secret that merely starts with ``$`` is not a slot unless what
    follows names a credential.
    """
    if not isinstance(value, str) or not _CREDENTIAL_NAME.search(str(parameter)):
        return False
    text = value.strip().strip("\"'").strip()
    if not text:
        return False
    if _TEMPLATE.match(text) or _PLACEHOLDER_WORD.match(text):
        return True
    if text.lower() == str(parameter).lower():
        return True
    slot = _NAMED_SLOT.match(text)
    return bool(slot) and bool(
        _CREDENTIAL_NAME.search(next(g for g in slot.groups() if g is not None)),
    )


def source_signature(source: Any, name: str) -> Optional[inspect.Signature]:
    """The signature of the function ``name`` defined in ``source``, read without running it (defaults
    stand in as markers); ``None`` when the source does not define it."""
    import ast

    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return None
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    if node is None:
        return None
    Parameter = inspect.Parameter
    spec = node.args
    positional = [*spec.posonlyargs, *spec.args]
    first_default = len(positional) - len(spec.defaults)
    params = [
        Parameter(
            arg.arg,
            (
                Parameter.POSITIONAL_ONLY
                if i < len(spec.posonlyargs)
                else Parameter.POSITIONAL_OR_KEYWORD
            ),
            default=_HAS_DEFAULT if i >= first_default else Parameter.empty,
        )
        for i, arg in enumerate(positional)
    ]
    if spec.vararg is not None:
        params.append(Parameter(spec.vararg.arg, Parameter.VAR_POSITIONAL))
    params += [
        Parameter(
            arg.arg,
            Parameter.KEYWORD_ONLY,
            default=Parameter.empty if default is None else _HAS_DEFAULT,
        )
        for arg, default in zip(spec.kwonlyargs, spec.kw_defaults)
    ]
    if spec.kwarg is not None:
        params.append(Parameter(spec.kwarg.arg, Parameter.VAR_KEYWORD))
    try:
        return inspect.Signature(params)
    except ValueError:
        return None


def caller_fault(
    signature: Optional[inspect.Signature],
    name: str,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> Optional[str]:
    """Why a failure of this call would be the caller's, not the function's; ``None`` when it would not.

    The arguments must bind to the function's signature, and no credential parameter may hold a
    placeholder (:func:`looks_like_placeholder`).
    """
    if signature is None:
        named = dict(kwargs)
    else:
        try:
            bound = signature.bind(*args, **kwargs)
        except TypeError as exc:
            return (
                f"the arguments do not fit the signature of `{name}{signature}`: {exc}"
            )
        named = {}
        for param, value in bound.arguments.items():
            kind = signature.parameters[param].kind
            if kind is inspect.Parameter.VAR_KEYWORD and isinstance(value, Mapping):
                named.update(value)
            else:
                named[param] = value
    for param, value in named.items():
        if looks_like_placeholder(param, value):
            return (
                f"`{param}` was given an unfilled placeholder instead of a real value; "
                f"pass the actual value from your session"
            )
    return None


__all__ = [
    "caller_fault",
    "credential_key",
    "failure_reason",
    "input_hash",
    "looks_like_placeholder",
    "sha256",
    "source_signature",
]
