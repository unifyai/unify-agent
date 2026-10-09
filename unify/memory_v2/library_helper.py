"""The memory library's helpers. Task 3 completes this module; it is copied as ``memory/__init__.py``.

Standard library only, with no relative import: the harness imports it as a module, and every copy of the
library runs it as its package init.
"""

from __future__ import annotations

INDEX_FILE = "INDEX.md"
LINKS_FILE = "links.json"
ITEMS_FILE = ".memory/items.json"
FIND_FILE = ".memory/find.json"
MATCHER_FILE = ".memory/matcher.py"
SHAPES_FILE = ".memory/shapes.py"
ITEM_LINE = "- `"


def split_index(text: str) -> tuple[str, list[tuple[str, str]]]:
    """An index as its head (before the first ``## `` heading) and its sections (heading, body)."""
    head: list[str] = []
    sections: list[list[str]] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            sections.append([line[3:].rstrip("\n"), ""])
        elif sections:
            sections[-1][1] += line
        else:
            head.append(line)
    return "".join(head), [(h, b) for h, b in sections]


def section_key(heading: str) -> str:
    """``memory.<package>`` or ``notes/<topic>``: a heading without its summary."""
    return heading.split(" — ", 1)[0].strip()


def wanted_key(name: object) -> str:
    """The section a name means: ``"office"`` and ``"memory/office"`` both mean ``memory.office``."""
    n = str(name).strip().strip("/")
    if n.startswith(("memory.", "notes/")):
        return n
    if n.startswith("memory/"):
        return "memory." + n[len("memory/") :].replace("/", ".")
    return "memory." + n


def find_section(text: str, name: object) -> str:
    """The full section of the index *text* that *name* means; LookupError naming the sections otherwise."""
    key = wanted_key(name)
    _, sections = split_index(text)
    for heading, body in sections:
        if section_key(heading) == key:
            return f"## {heading}\n{body}"
    keys = ", ".join(section_key(h) for h, _ in sections) or "(none)"
    raise LookupError(f"memory: no index section {key!r}; the sections are: {keys}")


def item_count(body: str) -> int:
    """The item lines of a section body."""
    return sum(1 for line in body.splitlines() if line.startswith(ITEM_LINE))
