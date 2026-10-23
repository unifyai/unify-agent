"""One example call of a library function, quoted from its tests: what the index, an item's card and the
docstring's ``Example:`` line show of how to call it.

An item record may keep one example call of its function (``{"test", "line", "call"}``), picked by a gate before
rev 2.1 from the calls its tests made; the gate now runs no test, so it picks none. Whatever shows a kept call
re-quotes it from the tree first (:func:`verified_example`, :func:`call_text`: the call expression at the recorded
line, whitespace runs collapsed to one space, comments dropped, cut past :data:`EXAMPLE_MAX_CHARS`): a test edited
since gives other text at that line, and then nothing is shown, so a stale call is never quoted. Pure, standard
library only.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath


EXAMPLE_MAX_CHARS = 160
CUT = " …"
#: A test id as a docstring's Example: line names it: a test file under a package's ``tests/`` and a test function
#: (inside classes, each named), without parameter brackets.
TEST_ID = re.compile(
    r"^memory/[A-Za-z_][A-Za-z0-9_]*/tests/(?:[A-Za-z0-9_]+/)*test_[A-Za-z0-9_]*\.py"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*)+$",
)
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


