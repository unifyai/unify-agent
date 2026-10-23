"""The "when to use" section every stored function and note carries, and the README every library folder holds; pure.

The writer writes both; the gate checks their shape only (library mode, G1, :mod:`.library_checks`), never their
wording. A function's docstring holds the section as one more docstring section::

    When to use:
        Saved for: what the source request asked for, summarised.
        Recognise: how to tell that a new request is that situation.
        Not for: the requests that look alike but need something else.
        Example request: one request it fits, in the words a requester might use.

A note holds the same four lines under a heading of its body, ``## When to use`` (any heading level), up to the
next heading; a line may start with a list marker (``- ``) and a label may be bold (``**Saved for:**``). Each
part's text may go on over the lines after its label. Every part is present once, with text holding a letter or
a digit.

A note also holds a ``## Rule`` section with text (:func:`rule_problems`): the part of it ``memory.get`` serves.

Folders: every package directory ``memory/<package>/`` and every topic directory ``notes/<topic>/`` holds a
``README.md`` the writer keeps: what the folder holds and how its items relate. The gate checks that each folder
a pass changes holds one with some text.
"""

from __future__ import annotations

import re

#: The docstring section's name (``When to use:``) and the note body's heading text (``## When to use``).
SECTION = "When to use"
#: The four parts, each a labelled line of the section, in the order the writer is asked to write them.
PARTS = ("Saved for", "Recognise", "Not for", "Example request")
#: Spellings taken for a part's label, besides its own.
_ALIASES = {"Recognize": "Recognise"}
#: The file each library folder holds.
README = "README.md"

_LABEL = re.compile(
    r"^[ \t]*(?:[-*+][ \t]+)?(?:\*\*|__)?("
    + "|".join(re.escape(p) for p in sorted({*PARTS, *_ALIASES}, key=len, reverse=True))
    + r")[ \t]*:(?:\*\*|__)?[ \t]*(.*)$",
)
_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.*?)[ \t#]*$")
_WORD = re.compile(r"[^\W_]")
_PACKAGE_DIR = re.compile(r"^memory/([a-z][a-z0-9_]*)/")
_TOPIC_DIR = re.compile(r"^notes/([a-z0-9][a-z0-9_-]*)/")

SHAPE = (
    f"a `{SECTION}:` section with four labelled lines: "
    + ", ".join(f"`{p}:`" for p in PARTS)
    + ", each followed by its text"
)


def parts(body: str) -> tuple[dict[str, str], list[str]]:
    """The section *body*'s parts (label -> text, continuation lines joined) and the labels given more than once."""
    found: dict[str, list[str]] = {}
    twice: list[str] = []
    current: str | None = None
    for line in (body or "").splitlines():
        m = _LABEL.match(line)
        if m is not None:
            label = _ALIASES.get(m.group(1), m.group(1))
            if label in found:
                twice.append(label)
                current = None
                continue
            current = label
            found[label] = [m.group(2).strip()]
        elif current is not None and line.strip():
            found[current].append(line.strip())
    return {k: " ".join(t for t in v if t) for k, v in found.items()}, twice


def section_problems(body: str | None, where: str) -> list[str]:
    """What the section's shape lacks (empty when it holds every part); *where* names the section in each reason."""
    if body is None:
        return [f"has no {where} ({SHAPE})"]
    found, twice = parts(body)
    out = [f"{where} gives `{label}:` more than once" for label in dict.fromkeys(twice)]
    for label in PARTS:
        if label not in found:
            out.append(f"{where} has no `{label}:` line")
        elif not _WORD.search(found[label]):
            out.append(f"{where} has an empty `{label}:` line")
    return out


def docstring_section(doc: str) -> str | None:
    """The ``When to use:`` section of a cleaned docstring, dedented (None when it has none)."""
    from .docstrings import parse

    sections = parse(doc or "").sections
    return sections.get(SECTION) if SECTION in sections else _empty_section(doc)


def _empty_section(doc: str) -> str | None:
    """'' when the docstring has the section's header with nothing under it (the parser drops empty sections)."""
    for line in (doc or "").splitlines():
        if line.rstrip() == f"{SECTION}:":
            return ""
    return None


def function_problems(doc: str) -> list[str]:
    """What a function docstring's ``When to use:`` section lacks."""
    return section_problems(docstring_section(doc), f"`{SECTION}:` section")


def note_section(text: str) -> str | None:
    """The body of a note's ``When to use`` heading, up to the next heading (None when it has none)."""
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m is not None and m.group(2).strip().rstrip(":").strip().lower() == SECTION.lower():
            body = []
            for nxt in lines[i + 1 :]:
                if _HEADING.match(nxt):
                    break
                body.append(nxt)
            return "\n".join(body)
    return None


def note_problems(text: str) -> list[str]:
    """What a note's ``When to use`` section lacks."""
    return section_problems(note_section(text), f"`## {SECTION}` section")


#: The heading of a note's rule: what ``memory.get`` serves of a note besides its front matter
#: (:func:`.library_helper._note_rule`); the rest of the note is read at its path.
RULE = "Rule"


def rule_problems(text: str) -> list[str]:
    """What a note's ``## Rule`` section lacks: missing, or without a letter or a digit before the next ``## ``."""
    lines = (text or "").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == f"## {RULE}".lower()), None)
    if start is None:
        return [f"has no `## {RULE}` section (what memory.get serves of a note besides its front matter)"]
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    return [] if _WORD.search("\n".join(body)) else [f"has an empty `## {RULE}` section"]


def folder(path: str) -> str | None:
    """The library folder *path* (relative, POSIX) lies in: ``memory/<package>`` or ``notes/<topic>``; else None."""
    m = _PACKAGE_DIR.match(path) or _TOPIC_DIR.match(path)
    if m is None:
        return None
    return m.group(0).rstrip("/")


def readme_of(folder_path: str) -> str:
    return f"{folder_path}/{README}"


def readme_problem(text: bytes | None) -> str | None:
    """Why a folder's README does not count (None when it does): missing, or without a letter or a digit."""
    if text is None:
        return "has no README.md"
    if not _WORD.search(text.decode("utf-8", "replace")):
        return "has an empty README.md"
    return None
