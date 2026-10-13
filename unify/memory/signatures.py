"""The one rule for printed identifiers: what counts as an identifier in a request, an applicability signature or any
content a mechanism compares, and how it is stripped. Identifiers are never keys: every signature and every content
comparison strips them first, with this rule and no other.

A token is an identifier by its form alone (:func:`is_identifier`), never by a word list:

* a UUID;
* a run of 8 or more hex characters that holds at least one digit (``0f3a9c7e1b``, ``ab12cd34ef``);
* a token of 6 or more characters holding 3 or more digits (``run_000123``, ``case-2041``), except a word followed
  by a width or version of at most three digits (``sha256``, ``float128``), which names a format, not a thing.

Ordinary words that carry a number (``utf8``, ``float64``, ``base64``, ``int32``, ``x2``, ``step3``) are kept.

**Aliases the request prints as ids** (:func:`printed_ids`). A request can label a value as an identifier by its own
structure: a ``label: value`` or ``label = value`` field whose label is an identifier label (a label word, alone or as
the last part of a ``snake``, ``kebab`` or ``camel`` name, out of :data:`ID_LABELS`: ``id``, ``ids``,
``identifier``, ``alias``, ``task``, ``ref``, ``ticket``), or a ``#`` directly before a token; a value of at least
:data:`MIN_PRINTED` characters.
The labelled value is an identifier of that request whatever its form (``task: lunar-otter`` makes ``lunar-otter``
one), and :func:`strip_identifiers` with ``request=`` strips it wherever it appears. The label set names the
structure a request uses to say "this is an id"; it is never used to match or rank content.
"""

from __future__ import annotations

import re

ID_MARK = "<id>"
#: Field labels by which a request marks its value as an identifier (module docstring).
ID_LABELS = frozenset({"id", "ids", "identifier", "alias", "task", "ref", "ticket"})
#: A labelled value shorter than this is not taken (a one- or two-character value would strip ordinary text).
MIN_PRINTED = 3

_UUID = re.compile(
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
)
_TOKEN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?")
_HEX = re.compile(r"[0-9A-Fa-f]{8,}")
_WIDTH = re.compile(r"[A-Za-z]+[0-9]{1,3}")
# ``label: value`` or ``label = value`` (optionally quoted), the label one name; ``#value``
_FIELD = re.compile(
    r"""(?P<label>[A-Za-z][A-Za-z0-9_-]{0,40})["']?\s*[:=]\s*["']?(?P<value>[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}[A-Za-z0-9]|[A-Za-z0-9])""",
)
_HASH = re.compile(r"(?<![A-Za-z0-9&])#(?P<value>[A-Za-z0-9][A-Za-z0-9_-]*)")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def is_identifier(token: str) -> bool:
    """Whether *token*, on its own, has an identifier's form (module docstring)."""
    t = str(token)
    if _UUID.fullmatch(t):
        return True
    if _HEX.fullmatch(t) and any(c.isdigit() for c in t):
        return True
    if len(t) >= 6 and sum(c.isdigit() for c in t) >= 3 and not _WIDTH.fullmatch(t):
        return bool(_TOKEN.fullmatch(t))
    return False


def _label_parts(label: str) -> list[str]:
    return [p.lower() for p in re.split(r"[_-]", _CAMEL.sub("_", label)) if p]


def printed_ids(request: str) -> set[str]:
    """The values *request* labels as identifiers by its own structure (module docstring)."""
    out: set[str] = set()
    text = str(request or "")
    for m in _FIELD.finditer(text):
        parts = _label_parts(m.group("label"))
        if parts and parts[-1] in ID_LABELS:
            out.add(m.group("value"))
    for m in _HASH.finditer(text):
        out.add(m.group("value"))
    return {v for v in out if len(v) >= MIN_PRINTED}


def strip_identifiers(text: str, request: str | None = None) -> str:
    """*text* with every identifier replaced by :data:`ID_MARK`: UUIDs, tokens of an identifier's form, and, with
    *request*, every value that request labels as an identifier (:func:`printed_ids`), wherever it appears."""
    out = _UUID.sub(ID_MARK, str(text))
    labelled = sorted(printed_ids(request) if request else set(), key=len, reverse=True)
    for value in labelled:
        out = re.sub(
            r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])",
            ID_MARK,
            out,
        )
    return _TOKEN.sub(lambda m: ID_MARK if is_identifier(m.group(0)) else m.group(0), out)
