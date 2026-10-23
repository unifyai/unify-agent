"""Docstring sections, parsed; pure.

:func:`parse` splits a cleaned docstring (``ast.get_docstring``/``inspect.getdoc``) into its first-line summary,
its description, its sections and the ``Effect:``/``Input:`` line fields. A section starts at a line ``<Name>:``
with no indentation (text may follow on the same line); its body is the indented lines after it. ``Examples:`` is
read as ``Example:`` (and ``Arguments:``/``Parameters:`` as ``Args:``, ``Return:`` as ``Returns:``). The ``When to
use`` section (:mod:`.when_to_use`) is a section too, so it never runs into the one before it.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field

#: The section names :func:`parse` reads (``When to use`` is :mod:`.when_to_use`'s).
SECTIONS = ("Args", "Returns", "Raises", "Example", "Use when", "Don't use when", "Notes", "When to use")
LINE_FIELDS = ("Effect", "Input")
_ALIASES = {
    "Examples": "Example",
    "Don’t use when": "Don't use when",
    "Arguments": "Args",
    "Parameters": "Args",
    "Return": "Returns",
}
_NAMES = sorted(
    set(SECTIONS) | set(LINE_FIELDS) | set(_ALIASES),
    key=len,
    reverse=True,
)
_HEADER = re.compile(
    r"^(" + "|".join(re.escape(n) for n in _NAMES) + r"):[ \t]*(.*)$",
)
_ARG = re.compile(r"^\*{0,2}([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)\n]*\))?\s*:\s*(.*)$")


@dataclass
class Docstring:
    summary: str = ""
    description: str = ""
    sections: dict[str, str] = field(default_factory=dict)  # canonical name -> body
    args: list[tuple[str, str]] = field(default_factory=list)  # (parameter, text)
    lines: dict[str, str] = field(default_factory=dict)  # Effect / Input -> value


def _body(lines: list[str]) -> str:
    return textwrap.dedent("\n".join(lines)).strip("\n").rstrip()


def _args(body: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ln in body.splitlines():
        m = None if ln[:1] in (" ", "\t") else _ARG.match(ln)
        if m is not None:
            out.append((m.group(1), m.group(2).strip()))
        elif out and ln.strip():
            name, text = out[-1]
            out[-1] = (name, (text + " " + ln.strip()).strip())
    return out


def parse(doc: str) -> Docstring:
    """The sections of a cleaned docstring (``ast.get_docstring``/``inspect.getdoc``); never raises."""
    lines = (doc or "").strip("\n").splitlines()
    out = Docstring(summary=lines[0].strip() if lines else "")
    bodies: dict[str, list[str]] = {}
    current, description = "", []
    for ln in lines[1:]:
        m = _HEADER.match(ln)
        if m is None:
            (description if not current else bodies[current]).append(ln)
            continue
        name = _ALIASES.get(m.group(1), m.group(1))
        if name in LINE_FIELDS:
            out.lines.setdefault(name, m.group(2).strip())
            current = ""
            continue
        current = name
        bodies.setdefault(name, [])
        if bodies[name]:
            bodies[name].append("")
        if m.group(2).strip():
            # the inline text is the body's first line, at the body's indentation
            bodies[name].append("    " + m.group(2).strip())
    out.description = _body(description).strip()
    out.sections = {k: _body(v) for k, v in bodies.items() if _body(v).strip()}
    out.args = _args(out.sections.get("Args", ""))
    return out
