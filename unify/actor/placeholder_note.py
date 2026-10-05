"""``UNIFY_PLACEHOLDER_NOTE``: say when an ``execute_function`` argument is a stand-in for a credential.

``execute_function`` passes ``call_kwargs`` to the function exactly as
written: a value cannot name a session variable, whatever ``state_mode`` or
session the call runs in. In an AppWorld HIGH cell the model logged in, held
the token in the session variable ``access_token``, and called a stored
function through the tool with ``{"access_token": "{{access_token}}"}``
twice; the function received those 16 characters and both calls failed with
401. In the 124 AppWorld attempt logs written from 1 to 5 October, 21
``execute_function`` calls passed a credential-named argument a stand-in,
and all 21 failed (401, 422, or the function's own empty-token ValueError):
12 that ``looks_like_placeholder`` already recognises (6 unfilled
templates, 5 slots such as ``$spotify_token`` or ``<hidden-token>``, 1 the
parameter's own name) and 9 it does not (5 empty or blank strings, 2
``unknown``, 1 ``x``, 1 ``token``). None of the 21 calls whose credential
argument returned held one. No argument that is not credential-named held
a template.

With the switch on, the tool's result (returned or failed) carries a short
note naming each such argument and the text it received, and saying that
values are passed as written and session variables are not substituted; the
call itself runs unchanged. The tool's description says the same in one
sentence. Only a value :func:`stand_in` classifies is ever echoed, so a real
credential never appears in the note. Off: no note, and the description as
shipped.

A credential-named argument is a stand-in when
:func:`unify.function_manager.store_trust.looks_like_placeholder` says so
(templates, credential slots, placeholder words, the parameter's own name),
or, for a parameter that clearly names a credential (``access_token``,
``password``, ``api_key``; not ``sort_key`` or ``author``), when it is one of
the further stand-ins those logs show: an empty or blank string, a single
character, ``unknown``, or a bare credential noun (``token``, ``password``,
``api_key``...). ``None`` is not text and is left alone: an optional
credential may be ``None``. The detector the store's trust records use is
not widened, so those records stay as they are.

The core tool surface (``UNIFY_TOOL_SURFACE=core``) is not covered:
``functions.run`` takes Python keyword arguments in a cell, where a session
variable is passed by naming it, so the confusion this note answers does not
arise there.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

#: The longest stand-in text echoed in full.
_ECHO_CHARS = 60

#: Stand-ins seen in the logs that ``looks_like_placeholder`` does not catch.
_UNKNOWN = frozenset({"unknown"})
_CREDENTIAL_NOUNS = frozenset(
    {
        "token",
        "tokens",
        "accesstoken",
        "authtoken",
        "bearertoken",
        "bearer",
        "apikey",
        "key",
        "secret",
        "password",
        "passwd",
        "credential",
        "credentials",
        "auth",
    },
)

DOC_SENTENCE = (
    "Values are literals: session variables are not substituted, so a\n"
    "value held in a session variable is passed by calling the function\n"
    "from ``execute_code``."
)

#: A parameter name that clearly names a credential, for the stand-ins
#: ``looks_like_placeholder`` does not judge (its own name rule also takes
#: ``sort_key``, ``keyword`` and ``author``, where an empty string or a single
#: character is ordinary data).
_CREDENTIAL_PARAMETER = re.compile(
    r"token|secret|passw(?:or)?d|credential|auth(?!or)"
    r"|(?:api|access|client|master|private|secret|session|signing)[\W_]*key",
    re.IGNORECASE,
)

# The end of ``call_kwargs``'s description in the ``execute_function`` docs.
_DOC_ANCHOR = re.compile(
    r"(?P<indent>[ \t]*)which fails type validation at the callee\)\.",
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_PLACEHOLDER_NOTE", False))


def stand_in(parameter: Any, value: Any) -> bool:
    """Whether ``value``, passed for ``parameter``, stands in for a credential rather than being one."""
    from unify.function_manager.store_trust import (
        _CREDENTIAL_NAME,
        looks_like_placeholder,
    )

    if not isinstance(value, str) or not _CREDENTIAL_NAME.search(str(parameter)):
        return False
    if looks_like_placeholder(str(parameter), value):
        return True
    if not _CREDENTIAL_PARAMETER.search(str(parameter)):
        return False
    text = value.strip().strip("\"'").strip()
    if len(text) <= 1:
        return True
    folded = re.sub(r"[\s_-]+", "", text).lower()
    return text.lower() in _UNKNOWN or folded in _CREDENTIAL_NOUNS


def _shown(value: str) -> str:
    if not value.strip():
        return "an empty string" if not value else "a blank string"
    text = value if len(value) <= _ECHO_CHARS else value[: _ECHO_CHARS - 1] + "…"
    return f"the literal text `{text}`"


def note(call_kwargs: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The note for a call with these ``call_kwargs``; ``None`` when off or no argument is a stand-in."""
    if not enabled() or not isinstance(call_kwargs, Mapping):
        return None
    found = [
        f"`{name}` received {_shown(value)}"
        for name, value in call_kwargs.items()
        if stand_in(name, value)
    ]
    if not found:
        return None
    return (
        "; ".join(found) + ". `call_kwargs` values are passed as written and "
        "session variables are not substituted; to pass a variable, call the "
        "function from `execute_code`."
    )


def correct_doc(doc: str) -> str:
    """``execute_function``'s docs with :data:`DOC_SENTENCE` after ``call_kwargs``'s description."""

    def _add(match: "re.Match[str]") -> str:
        indent = match.group("indent")
        lines = DOC_SENTENCE.split("\n")
        return match.group(0) + "".join(f"\n{indent}{line}" for line in lines)

    return _DOC_ANCHOR.sub(_add, doc, count=1)


__all__ = ["DOC_SENTENCE", "correct_doc", "enabled", "note", "stand_in"]
