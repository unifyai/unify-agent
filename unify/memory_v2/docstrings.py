"""The lean docstring standard of environment functions (v2.1 stage 3), parsed and checked; pure.

A stored function documents itself well enough that the working model never reads its source: ``help(fn)``
and ``memory.describe(fn)`` show exactly this docstring. The standard, for every new or changed environment
function (the gate checks it under G1 and runs the examples under G3)::

    Read the accounts-payable ledger into rows.          <- Summary: the first line

    Args:
        path: the ledger CSV file (a path).              <- one entry per parameter; the first names its
        strict: refuse rows with an empty amount.           input form (:data:`.manifest.INPUT_KINDS`)

    Returns:
        A list of dicts, one per row, keyed by column.

    Raises:
        MemoryInputError: when the file is not a CSV with the ledger's columns.

    Example:
        >>> rows = read_ledger("env/worktree_workspace/tests/data/ledger.csv")
        >>> rows[0]["vendor_id"]
        'V-17'

    Use when: ...                                         <- optional
    Don't use when: ...                                   <- optional
    Notes: ...                                            <- optional

    Effect: read
    Input: path

A section starts at a line ``<Name>:`` with no indentation (text may follow on the same line); its body is
the indented lines after it. ``Examples:`` is accepted for ``Example:``. Every ``>>>`` example runs as a
doctest with the library root as the working directory and the function's module as its globals, so a
recorded input copied under ``env/<channel>/tests/`` is reached by its path from the root.

Refusals explain themselves: each ``raise MemoryInputError(...)`` in the function passes a message whose
literal text (string constants and the constant parts of f-strings, concatenations and ``format``
templates) has at least :data:`MIN_REFUSAL_CHARS` characters, saying what was expected and what to do
instead. A message built only from variables cannot be measured and is accepted. The check reads the
function's own code, so it covers every refusal path, not only the ones its tests exercise.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field

REQUIRED_SECTIONS = ("Args", "Returns", "Raises", "Example")
OPTIONAL_SECTIONS = ("Use when", "Don't use when", "Notes")
LINE_FIELDS = ("Effect", "Input")
ERROR_NAME = "MemoryInputError"
MIN_REFUSAL_CHARS = 24

# The catalogue's key for each section (``.memory/catalog.json``).
SECTION_KEYS = {
    "Args": "args",
    "Returns": "returns",
    "Raises": "raises",
    "Example": "example",
    "Use when": "use_when",
    "Don't use when": "dont_use_when",
    "Notes": "notes",
}
_ALIASES = {"Examples": "Example", "Don’t use when": "Don't use when"}
_NAMES = sorted(
    set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS) | set(LINE_FIELDS) | set(_ALIASES),
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


def params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The parameter names of *fn*, in order (``*args`` and ``**kwargs`` by their names)."""
    a = fn.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg is not None:
        names.append(a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg is not None:
        names.append(a.kwarg.arg)
    return names


def has_example(d: Docstring) -> bool:
    return ">>>" in d.sections.get("Example", "")


def problems(doc: str, names: list[str], input_form: str | None) -> list[str]:
    """What the docstring lacks of the standard, as reason fragments (``[]`` when it complies).

    *names* are the function's parameters; *input_form* its declared form (None: not declared, which G1
    refuses on its own).
    """
    d = parse(doc)
    out: list[str] = []
    if not d.summary:
        out.append("has no one-line summary (its first line)")
    if names:
        if "Args" not in d.sections:
            out.append(
                "has no Args: section (one `name: description` line per parameter)",
            )
        else:
            described = [n for n, _ in d.args]
            missing = [n for n in names if n not in described]
            if missing:
                out.append(f"Args: does not describe parameter(s) {', '.join(missing)}")
            first = dict(d.args).get(names[0], "")
            if (
                input_form
                and names[0] in described
                and not re.search(rf"\b{re.escape(input_form)}\b", first, re.I)
            ):
                out.append(
                    f"Args: the entry of the first parameter `{names[0]}` does not name its "
                    f"input form `{input_form}`",
                )
    if "Returns" not in d.sections:
        out.append("has no Returns: section")
    if "Raises" not in d.sections:
        out.append(f"has no Raises: section naming {ERROR_NAME} and when it is raised")
    elif ERROR_NAME not in d.sections["Raises"]:
        out.append(f"Raises: does not name {ERROR_NAME} and when it is raised")
    if not has_example(d):
        out.append("has no Example: section with a `>>>` example")
    return out


def _names_error(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == ERROR_NAME
    return isinstance(node, ast.Attribute) and node.attr == ERROR_NAME


def _literal(node: ast.expr) -> str | None:
    """The literal text of a message expression, or None when it holds none (a bare variable, say)."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        parts = [_literal(node.left), _literal(node.right)]
        if all(p is None for p in parts):
            return None
        return "".join(p or "" for p in parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        return _literal(node.func.value)
    return None


def refusal_problems(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Each ``raise MemoryInputError`` in *fn* without a measurable message of :data:`MIN_REFUSAL_CHARS`."""
    out: list[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        call = node.exc if isinstance(node.exc, ast.Call) else None
        if not _names_error(call.func if call is not None else node.exc):
            continue
        if call is None or (not call.args and not call.keywords):
            out.append(
                f"raises {ERROR_NAME} without a message (line {node.lineno}); say what was "
                "expected and what to do instead",
            )
            continue
        if not call.args:
            continue
        text = _literal(call.args[0])
        if text is not None and len(text.strip()) < MIN_REFUSAL_CHARS:
            out.append(
                f"raises {ERROR_NAME} with a {len(text.strip())}-character message (line "
                f"{node.lineno}); say what was expected and what to do instead (at least "
                f"{MIN_REFUSAL_CHARS} characters of text)",
            )
    return out


def as_catalog(d: Docstring) -> dict:
    """The parsed docstring as the catalogue stores it: only what is present."""
    out: dict = {}
    if d.description:
        out["description"] = d.description
    for name, key in SECTION_KEYS.items():
        if d.sections.get(name):
            out[key] = d.sections[name]
    if d.args:
        out["arg_list"] = [{"name": n, "text": t} for n, t in d.args]
    return out


def describe_standard() -> str:
    """The standard as Sol's brief states it, from the constants above."""
    required = ", ".join(f"`{n}:`" for n in REQUIRED_SECTIONS)
    optional = ", ".join(f"`{n}:`" for n in OPTIONAL_SECTIONS)
    return (
        f"the one-line summary, then the sections {required}, each a line `<Name>:` without "
        "indentation followed by indented text. Args: has one `name: description` line per parameter, "
        "and the first parameter's line names its input form. Raises: names "
        f"{ERROR_NAME} and when it is raised. Example: holds at least one `>>>` doctest example that "
        "runs on a recorded input copied as a test fixture under env/<channel>/tests/ (the gate runs "
        "every example as a doctest, read-only, with /memory as the working directory and the "
        "module's names in scope; a failing example refuses the function). Optional sections: "
        f"{optional}. Each `raise {ERROR_NAME}(...)` passes a message of at least "
        f"{MIN_REFUSAL_CHARS} characters saying what was expected and what to do instead"
    )
