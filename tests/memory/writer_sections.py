"""What a writer adds to every item and folder it writes, for the memory test worlds (:mod:`unify.memory.when_to_use`).

The library gate refuses a new or changed function or note without its "when to use" section, and the items of a
changed folder without a README.md. A world that builds a pass's candidate adds them as a writer does:
:func:`with_section` to a module's public functions, :func:`note_with_section` to a note and :func:`with_readmes`
to the files of a candidate. Each leaves what already has one as it is.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping

from unify.memory import when_to_use

PARTS = {
    "Saved for": "the work the source request asked for, in general terms.",
    "Recognise": "a request asking for that work on input of the same shape.",
    "Not for": "input of another shape, or other work on the same input.",
    "Example request": "Do that work on this input.",
}
README_TEXT = "What this folder holds and how its items relate.\n"


def docstring_lines(indent: str) -> str:
    """The section as docstring lines at *indent*, ending with a newline."""
    body = "".join(f"{indent}    {label}: {text}\n" for label, text in PARTS.items())
    return f"{indent}{when_to_use.SECTION}:\n{body}"


def note_lines() -> str:
    return f"## {when_to_use.SECTION}\n" + "".join(f"- {label}: {text}\n" for label, text in PARTS.items())


def with_section(source: str) -> str:
    """*source* with the section added to each top-level public function whose triple-quoted docstring lacks one."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node)
        first = node.body[0]
        indent = " " * first.col_offset
        if doc is None:
            if first.lineno == node.lineno:
                continue  # a one-line definition: left as it is
            # a function with no docstring gets one: a summary line and the section
            edits.append((first.lineno - 1, None, indent, node.name))
            continue
        if when_to_use.docstring_section(doc) is not None:
            continue
        line = lines[first.end_lineno - 1]
        close = first.end_col_offset - 3
        if line[close : close + 3] not in ('"""', "'''"):
            continue
        edits.append((first.end_lineno - 1, close, indent, node.name))
    for row, close, indent, name in sorted(edits, key=lambda e: e[0], reverse=True):
        if close is None:
            summary = name.replace("_", " ").capitalize() + "."
            lines.insert(row, f'{indent}"""{summary}\n\n' + docstring_lines(indent) + f'{indent}"""\n')
            continue
        line = lines[row]
        lines[row] = line[:close].rstrip() + "\n\n" + docstring_lines(indent) + indent + line[close:]
    return "".join(lines)


def note_with_section(text: str) -> str:
    """A note's *text* with the section appended to its body when it has none."""
    if when_to_use.note_section(text) is not None:
        return text
    return text.rstrip("\n") + "\n\n" + note_lines()


def with_readmes(files: Mapping[str, str | None], have: Mapping[str, object] = {}) -> dict[str, str | None]:
    """*files* (path -> text, ``None`` deleted) with a README.md for each library folder they write in that neither
    they nor *have* (the files already there) hold."""
    out = dict(files)
    for path, text in files.items():
        if text is None:
            continue
        folder = when_to_use.folder(path)
        if folder is None:
            continue
        readme = when_to_use.readme_of(folder)
        if readme not in out and readme not in have:
            out[readme] = README_TEXT
    return out


def as_written(files: Mapping[str, str | None], have: Mapping[str, object] = {}) -> dict[str, str | None]:
    """*files* as a writer leaves them: each module's functions and each note with its section, each folder with
    its README.md."""
    out: dict[str, str | None] = {}
    for path, text in files.items():
        kind = None if text is None else _kind(path)
        if kind == "module":
            text = with_section(text)
        elif kind == "note":
            text = note_with_section(text)
        out[path] = text
    return with_readmes(out, have)


def _kind(path: str) -> str | None:
    from unify.memory.layout import classify

    return classify(path)
