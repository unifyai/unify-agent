"""The lean docstring standard of library functions, parsed and checked; pure.

A stored function documents itself well enough that the working model never reads its source: ``help(fn)`` and
its card show exactly this docstring. The standard, for every new or changed function (the gate reads it; it runs
nothing)::

    Read the accounts-payable ledger into rows.          <- Summary: the first line

    Args:
        path: the ledger CSV file (a path).              <- one entry per parameter
        strict: refuse rows with an empty amount.

    Returns:
        A list of dicts, one per row, keyed by column.

    Raises:
        MemoryInputError: when the file is not a CSV with the ledger's columns.

    Example:
        >>> read_ledger("env/worktree_workspace/tests/data/ledger.csv")[0]["vendor_id"]
        'V-17'

    Use when: ...                                         <- optional
    Don't use when: ...                                   <- optional
    Notes: ...                                            <- optional

    Effect: read
    Input: path

A section starts at a line ``<Name>:`` with no indentation (text may follow on the same line); its body is
the indented lines after it. ``Examples:`` is accepted for ``Example:`` (and ``Arguments:``/``Parameters:``
for ``Args:``, ``Return:`` for ``Returns:``). Every ``>>>`` example runs as a doctest with the library root as
the working directory and the function's module as its globals, so a recorded input copied under
``env/<channel>/tests/`` is reached by its path from the root.

The examples must be real (:func:`example_problems`): the section parses as doctest examples, none is
skipped (``+SKIP``), at least one example is a call of the function itself with an argument, made once, whose
value (or an item, attribute or method of it) is the shown output, and that output has a letter or digit
beyond ``...`` (so ``fn(x) == fn(x)``, ``(fn(x), 3)[1]``, ``isinstance(fn(x), list)`` and ``[...]`` do not
count); the examples read at least one fixture under ``env/<channel>/tests/`` (a string constant naming the
path), and that call's arguments hold the fixture's path or a name an earlier example bound to it. The gate also checks that the fixture exists, that it has the shape of an input
the function was admitted on (:mod:`.memory_helper`), and, when the examples run, that at least one example
ran and the function was called.

Refusals explain themselves: each ``raise MemoryInputError(...)`` in the function passes a message whose
literal text (string constants and the constant parts of f-strings, concatenations and ``format``
templates) has at least :data:`MIN_REFUSAL_CHARS` characters, saying what was expected and what to do
instead. A message built only from variables cannot be measured and is accepted. The check reads the
function's own code, so it covers every refusal path, not only the ones its tests exercise.
"""

from __future__ import annotations

import ast
import doctest
import re
import textwrap
from dataclasses import dataclass, field

REQUIRED_SECTIONS = ("Args", "Returns", "Raises", "Example")
OPTIONAL_SECTIONS = ("Use when", "Don't use when", "Notes")
#: The library mode's section (:mod:`.when_to_use`): parsed as a section so that it never runs into the one before it.
LIBRARY_SECTIONS = ("When to use",)
LINE_FIELDS = ("Effect", "Input")
ERROR_NAME = "MemoryInputError"
_ALIASES = {
    "Examples": "Example",
    "Don’t use when": "Don't use when",
    "Arguments": "Args",
    "Parameters": "Args",
    "Return": "Returns",
}
_NAMES = sorted(
    set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS) | set(LIBRARY_SECTIONS) | set(LINE_FIELDS) | set(_ALIASES),
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


def examples(d: Docstring) -> list[doctest.Example]:
    """The doctest examples of the Example section, as doctest itself parses them (``[]`` if none)."""
    try:
        return doctest.DocTestParser().get_examples(d.sections.get("Example", ""))
    except ValueError:  # malformed (an inconsistent indentation, say)
        return []


def _tree(source: str) -> ast.AST | None:
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def _names_fn(node: ast.AST, name: str) -> bool:
    """Whether *node* is a call of *name* (by name or as an attribute, ``env.c.name``)."""
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == name)
        or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
    )


def _chain_root(node: ast.AST, name: str) -> ast.AST:
    """The root of a chain of items, attributes and method calls (``fn(x)[0].total``, ``fn(x).keys()``)."""
    while True:
        if _names_fn(node, name):
            return node
        if isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        else:
            return node


def _shown_call(source: str, name: str) -> ast.Call | None:
    """The call of *name* whose value the example shows: the example is one expression, the call itself or
    an item, attribute or method of its result, holding exactly one call of *name*, with an argument.
    None for an assignment, a comparison (``fn(x) == fn(x)``), a discarded result (``(fn(x), 3)[1]``),
    a wrapped one (``isinstance(fn(x), list)``) or a call without arguments."""
    tree = _tree(source)
    if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None
    expr = tree.body[0].value
    root = _chain_root(expr, name)
    if not _names_fn(root, name) or not (root.args or root.keywords):
        return None
    if sum(1 for n in ast.walk(expr) if _names_fn(n, name)) != 1:
        return None
    return root


def _shows_result(want: str) -> bool:
    """An expected output that shows something: a letter or digit once ``...`` is removed (not ``...``,
    ``[...]``, ``{...}`` or ``'...'``)."""
    return any(c.isalnum() for c in want.replace("...", ""))


def _fixture_names(
    found: list[doctest.Example],
    prefix: str,
) -> tuple[set[str], list[ast.AST]]:
    """The names bound (by an earlier example's assignment) to a value built from a fixture path under
    *prefix*, and each example's parsed source."""
    tainted: set[str] = set()
    trees: list[ast.AST] = []

    def reaches(node: ast.AST) -> bool:
        return any(
            (
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.startswith(prefix)
            )
            or (isinstance(n, ast.Name) and n.id in tainted)
            for n in ast.walk(node)
        )

    for e in found:
        tree = _tree(e.source)
        trees.append(tree)
        for n in ast.walk(tree) if tree is not None else ():
            value = getattr(n, "value", None)
            if (
                isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
                and value is not None
            ):
                if reaches(value):
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    tainted |= {
                        t.id
                        for tg in targets
                        for t in ast.walk(tg)
                        if isinstance(t, ast.Name)
                    }
    return tainted, trees


def _passes_fixture(call: ast.Call, prefix: str, tainted: set[str]) -> bool:
    """Whether the call's arguments hold a fixture path under *prefix* or a name bound to one."""
    parts = [*call.args, *(k.value for k in call.keywords)]
    return any(
        (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith(prefix)
        )
        or (isinstance(n, ast.Name) and n.id in tainted)
        for part in parts
        for n in ast.walk(part)
    )


def fixture_paths(d: Docstring, channel: str) -> list[str]:
    """The paths under ``env/<channel>/tests/`` the examples name as string constants, sorted."""
    prefix = f"env/{channel}/tests/"
    out: set[str] = set()
    for e in examples(d):
        tree = _tree(e.source)
        for n in ast.walk(tree) if tree is not None else ():
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, str)
                and n.value.startswith(prefix)
                and ".." not in n.value.split("/")
                and "\n" not in n.value
            ):
                out.add(n.value)
    return sorted(out)


def example_problems(d: Docstring, name: str | None, channel: str | None) -> list[str]:
    """Why the Example section is not a real example of *name* on a fixture of *channel* (``[]``: it is)."""
    found = examples(d)
    if not found:
        return ["has no Example: section with a `>>>` example"]
    out: list[str] = []
    if any(e.options.get(doctest.SKIP) for e in found):
        out.append("Example: an example is skipped (`+SKIP`); every example must run")
    shown = (
        [
            (i, call)
            for i, e in enumerate(found)
            if e.exc_msg is None  # a refusal shows no result
            and _shows_result(e.want)
            and (call := _shown_call(e.source, name)) is not None
        ]
        if name is not None
        else []
    )
    if name is not None and not shown:
        out.append(
            f"Example: no example calls `{name}(...)` with an argument and shows its result (an "
            "expected output with a letter or digit, not only `...`; the example is the call itself, or "
            "an item or attribute of its result, made once)",
        )
    paths = fixture_paths(d, channel) if channel is not None else []
    if channel is not None and not paths:
        out.append(
            f"Example: no example reads a recorded input from a fixture under env/{channel}/tests/",
        )
    if name is not None and shown and paths:
        prefix = f"env/{channel}/tests/"
        tainted, _ = _fixture_names(found, prefix)
        if not any(_passes_fixture(call, prefix, tainted) for _, call in shown):
            out.append(
                f"Example: the example showing `{name}(...)`'s result does not pass it the recorded "
                "fixture (name the fixture's path, or a name bound to it, in the call's arguments)",
            )
    return out


def problems(
    doc: str,
    names: list[str],
    input_form: str | None,
    *,
    name: str | None = None,
    channel: str | None = None,
) -> list[str]:
    """What the docstring lacks of the standard, as reason fragments (``[]`` when it complies).

    *names* are the function's parameters; *input_form* its declared form (None: not declared, which G1
    refuses on its own); *name* and *channel* (the gate gives both) turn on the example's call and
    fixture checks.
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
    out += example_problems(d, name, channel)
    return out


