"""One example call of a library function, quoted from its tests: what the index, ``memory.show`` and the
docstring's ``Example:`` line show of how to call it.

A function's examples are the calls its tests make (:mod:`.example_calls`). The harness picks one that returns on
values loaded from recorded fixtures (:func:`pick_example`) and quotes its source text from the test file
(:func:`call_text`): the call expression at the recorded line, whitespace runs collapsed to one space, comments
dropped, cut past :data:`EXAMPLE_MAX_CHARS`. Only a call written in the test file itself is quoted (a call made
inside a helper module under ``tests/`` would show the helper's parameters, not the test's values).

The item record keeps the picked call (``{"test", "line", "call"}``). Whatever shows it re-quotes the call from the
tree first (:func:`verified_example`): a test edited since gives other text at that line, and then nothing is shown,
so a stale call is never quoted.

A function's docstring names one of its tests on its ``Example:`` line (``memory/<package>/tests/test_x.py::
test_name``); no doctest and no ``Input:`` line (:func:`docstring_problems`). Pure, standard library only.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath

from . import docstrings
from .example_calls import CallKey, ExampleCall

EXAMPLE_MAX_CHARS = 160
CUT = " …"
#: A test id as a docstring's Example: line names it: a test file under a package's ``tests/`` and a test function
#: (inside classes, each named), without parameter brackets.
TEST_ID = re.compile(
    r"^memory/[A-Za-z_][A-Za-z0-9_]*/tests/(?:[A-Za-z0-9_]+/)*test_[A-Za-z0-9_]*\.py"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*)+$",
)
TEST_ID_FORM = "memory/<package>/tests/test_<name>.py::test_<case>"
_INPUT_LINE = re.compile(r"^[ \t]*Input:", re.M)
ReadFile = Callable[[str], "bytes | None"]


def _names_fn(node: ast.AST, name: str) -> bool:
    """A call of *name*, by name or as an attribute (``fill.flood_fill(...)``)."""
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == name)
        or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
    )


def _without_comments(segment: str) -> str:
    """*segment* with its ``#`` comments removed (a multi-line call may carry them between arguments)."""
    if "#" not in segment:
        return segment
    try:
        comments = [
            t
            for t in tokenize.generate_tokens(io.StringIO(segment).readline)
            if t.type == tokenize.COMMENT
        ]
    except (tokenize.TokenError, SyntaxError):
        return segment
    lines = segment.splitlines(keepends=True)
    for t in reversed(comments):
        row = t.start[0] - 1
        lines[row] = lines[row][: t.start[1]] + lines[row][t.end[1] :]
    return "".join(lines)


def quote_call(source: str, line: int, name: str) -> str | None:
    """The source text of the call of *name* at *line* of *source*, on one line, cut past
    :data:`EXAMPLE_MAX_CHARS` with ``" …"``; None when no such call is there.

    A call spanning several lines is found by any of its lines: one that starts at *line* is preferred, then the
    shortest span (the innermost call), then the first in the source."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    found = [
        n
        for n in ast.walk(tree)
        if _names_fn(n, name) and n.lineno <= line <= (n.end_lineno or n.lineno)
    ]
    if not found:
        return None
    node = min(
        found,
        key=lambda n: (
            n.lineno != line,
            (n.end_lineno or n.lineno) - n.lineno,
            n.lineno,
            n.col_offset,
        ),
    )
    segment = ast.get_source_segment(source, node)
    if not segment:
        return None
    text = " ".join(_without_comments(segment).split())
    if len(text) > EXAMPLE_MAX_CHARS:
        text = text[: EXAMPLE_MAX_CHARS - len(CUT)].rstrip() + CUT
    return text


def _safe_rel(rel: object) -> str | None:
    """*rel* when it is a plain relative path under ``memory/`` (no ``..``, no absolute path), else None."""
    if not isinstance(rel, str) or not rel or "\0" in rel or "\\" in rel:
        return None
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or p.parts[:1] != ("memory",):
        return None
    return p.as_posix()


def tree_reader(tree: Path) -> ReadFile:
    """A reader of the files of the library copy at *tree*: a file's bytes by tree-relative path, None for a
    missing file, a link, a path outside the tree or one not under ``memory/``."""
    root = Path(tree)

    def read(rel: str) -> bytes | None:
        safe = _safe_rel(rel)
        if safe is None:
            return None
        path = root / safe
        try:
            if path.is_symlink() or not path.is_file():
                return None
            if not path.resolve().is_relative_to(root.resolve()):
                return None
            return path.read_bytes()
        except OSError:
            return None

    return read


def _quote_from(read: ReadFile, test_file: str, line: int, name: str) -> str | None:
    data = read(test_file)
    if data is None:
        return None
    try:
        source = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return quote_call(source, line, name)


def call_text(tree: Path, test_file: str, line: int, name: str) -> str | None:
    """The call of *name* at *line* of the test file *test_file* (tree-relative) of the library copy at *tree*,
    quoted as :func:`quote_call` does; None when the file or the call is not there."""
    return _quote_from(tree_reader(tree), test_file, line, name)


def file_of_test(test: str) -> str:
    """The file of a test id (``memory/<pkg>/tests/test_x.py::test_y`` gives ``memory/<pkg>/tests/test_x.py``)."""
    return test.split("::", 1)[0]


def _function_name(item: str) -> str:
    return item.split(":", 1)[1] if ":" in item else item


def pick_example(
    item: str,
    calls: list[ExampleCall],
    grounded: set[CallKey],
    named: str | None,
    tree: Path,
) -> dict | None:
    """``{"test", "line", "call"}``: the example call of *item* to show. Among its calls that returned and whose
    arguments are grounded in recorded fixtures (*grounded*), written in their test's own file: the first of the
    test *named* by the docstring's ``Example:`` line, else the first by (test, indices, ordinal). A call whose
    text cannot be quoted from *tree* is passed over. None when no call qualifies."""
    name = _function_name(item)
    eligible = sorted(
        (
            c
            for c in calls
            if c.item == item
            and c.outcome == "returned"
            and c.key in grounded
            and c.file == file_of_test(c.key.test)
        ),
        key=lambda c: c.key,
    )
    if named is not None:
        eligible = [c for c in eligible if c.key.test == named] + [
            c for c in eligible if c.key.test != named
        ]
    read = tree_reader(tree)
    for c in eligible:
        text = _quote_from(read, c.file, c.line, name)
        if text is not None:
            return {"test": c.key.test, "line": c.line, "call": text}
    return None


def verified_example(
    item: str,
    example: object,
    read: ReadFile,
) -> dict | None:
    """The recorded *example* (``{"test", "line", "call"}``) of *item* when its test file still quotes the same
    call at that line, else None: a stale example is never shown."""
    if not isinstance(example, Mapping):
        return None
    test, line, call = example.get("test"), example.get("line"), example.get("call")
    if (
        not isinstance(test, str)
        or not TEST_ID.match(test)
        or not isinstance(line, int)
        or isinstance(line, bool)
        or not isinstance(call, str)
        or not call
    ):
        return None
    if _quote_from(read, file_of_test(test), line, _function_name(item)) != call:
        return None
    return {"test": test, "line": line, "call": call}


def examples_of(
    records: Mapping[str, dict] | None,
    read: ReadFile,
) -> Callable[[str], dict | None]:
    """An item's shown example: its record's ``example``, re-checked against the tree (:func:`verified_example`);
    None without records, without a recorded example, or when the record is stale."""
    cache: dict[str, dict | None] = {}

    def get(item: str) -> dict | None:
        if item not in cache:
            rec = records.get(item) if records else None
            cache[item] = (
                verified_example(item, rec.get("example"), read)
                if isinstance(rec, Mapping)
                else None
            )
        return cache[item]

    return get


def _code(text: str) -> str:
    """*text* as a markdown code span (a longer fence when it holds a backtick)."""
    if "`" not in text:
        return f"`{text}`"
    return f"`` {text} ``"


def example_text(example: Mapping) -> str:
    """How the index shows an example: ``` `<call>` (<test id>) ```."""
    return f"{_code(example['call'])} ({example['test']})"


def example_of(
    records: Mapping[str, dict] | None,
    read: ReadFile,
) -> Callable[[str], str | None]:
    """:func:`.library_index.render_index`'s ``example_of``: an item's shown example as index text, or None."""
    shown = examples_of(records, read)

    def get(item: str) -> str | None:
        ex = shown(item)
        return example_text(ex) if ex is not None else None

    return get


def docstring_example(doc: str) -> str | None:
    """The one test id the docstring's ``Example:`` section names (backticks around it allowed); None when the
    section is missing, holds anything else (a doctest, say) or more than one line."""
    body = docstrings.parse(doc or "").sections.get("Example", "")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) != 1:
        return None
    test = lines[0].strip("`").strip()
    return test if TEST_ID.match(test) else None


def docstring_problems(
    item: str,
    doc: str,
    tests: list[str],
    calls: list[ExampleCall],
    grounded: set[CallKey],
) -> list[str]:
    """Why *item*'s docstring does not meet the library rule (``[]``: it does): its ``Example:`` line names one
    test of its listed test files (*tests*) that makes a call of it returning on fixture values (*calls* that
    returned and are in *grounded*), and it has no ``Input:`` line."""
    out: list[str] = []
    named = docstring_example(doc)
    if named is None:
        out.append(
            f"{item} docstring has no Example: line naming one of its tests ({TEST_ID_FORM})",
        )
    elif file_of_test(named) not in set(tests):
        out.append(
            f"{item} docstring Example: names {named}, which is not a test of its listed test files",
        )
    elif not any(
        c.item == item
        and c.key.test == named
        and c.outcome == "returned"
        and c.key in grounded
        for c in calls
    ):
        out.append(
            f"{item} docstring Example: names {named}, which makes no call of it that returns on fixture values",
        )
    if _INPUT_LINE.search(doc or ""):
        out.append(
            f"{item} docstring has an Input: line; a function's inputs are shown by its tests' calls, so remove "
            "the line",
        )
    return out
