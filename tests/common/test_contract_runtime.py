"""Symbolic: the contract runtime: typed arguments checked at the call, and errors that teach.

The runtime is standard library only (the library package carries it byte for byte), so these tests load it as a
plain module. The error texts are the approved memory API's own examples
(docs/design/memory-api-items-20261013.md in the research repository).
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Annotated, Any, Literal

import pytest

from unify.common import contract_runtime as cr


def test_the_error_texts_are_the_designs_examples():
    miss = cr.NotFound(
        "memory.get('stencil.md')",
        "no item named 'stencil.md'.",
        expected="an item id, a function name, or a note slug",
        example="memory.get('notes/grid/stencil.md')",
        nearest=["notes/grid/stencil.md", "notes/grid/paint_stencil.md"],
        hint="Browse with memory.items('notes/grid').",
    )
    assert str(miss) == (
        "memory.get('stencil.md'): no item named 'stencil.md'.\n"
        "Expected an item id, a function name, or a note slug, e.g. memory.get('notes/grid/stencil.md').\n"
        "Nearest: notes/grid/stencil.md, notes/grid/paint_stencil.md. Browse with memory.items('notes/grid')."
    )
    wrong = cr.WrongCall(
        "memory.items('notes/grid/stencil.md')",
        "that is an item, not a topic.",
        hint="Open it with memory.get('notes/grid/stencil.md').",
    )
    assert str(wrong) == (
        "memory.items('notes/grid/stencil.md'): that is an item, not a topic. "
        "Open it with memory.get('notes/grid/stencil.md')."
    )
    # each kind is also the builtin a caller catches, and keeps its parts
    assert isinstance(miss, LookupError) and isinstance(wrong, LookupError)
    assert isinstance(cr.ArgumentError("f()", "x."), TypeError)
    assert miss.nearest == ("notes/grid/stencil.md", "notes/grid/paint_stencil.md")
    assert miss.call == "memory.get('stencil.md')" and wrong.hint.startswith("Open it")


def test_nearest_puts_the_same_last_segment_first():
    pool = ["notes/grid/paint_stencil.md", "notes/grid/stencil.md", "notes/maps/legend.md", "memory.grid.cells:rows"]
    assert cr.nearest("stencil.md", pool)[0] == "notes/grid/stencil.md"
    assert cr.nearest("row", pool)[0] == "memory.grid.cells:rows"
    assert cr.nearest("zzzz", pool) == []


@cr.checked(name="memory.items")
def items(
    topic: str | None = None,
    kind: Literal["function", "note"] | None = None,
    limit: Annotated[int, cr.Ge(1)] = 20,
) -> list[str]:
    """Browse the library.

    Example:
        memory.items('notes/grid', kind='note')
    """
    return [str(topic), str(kind), str(limit)]


def test_a_wrong_type_names_the_parameter_the_type_and_the_example():
    assert items("notes/grid", kind="note") == ["notes/grid", "note", "20"]
    with pytest.raises(cr.ArgumentError) as info:
        items("notes/grid", kind="notes")
    assert str(info.value) == (
        "memory.items('notes/grid', kind='notes'): kind is 'notes' (str).\n"
        "Expected kind: 'function' | 'note' | None, e.g. memory.items('notes/grid', kind='note')."
    )
    with pytest.raises(cr.ArgumentError, match=r"limit is 0 \(int\)\.\nExpected limit: int >= 1"):
        items(limit=0)
    with pytest.raises(cr.ArgumentError, match=r"limit is True \(bool\)"):
        items(limit=True)


def test_a_wrong_arity_or_keyword_is_an_argument_error_with_the_signature():
    with pytest.raises(cr.ArgumentError) as info:
        items(topic="a", colour="red")
    text = str(info.value)
    assert text.startswith("memory.items(topic='a', colour='red'): got an unexpected keyword argument 'colour'.")
    assert "Expected memory.items(topic: 'str | None' = None" in text or "Expected memory.items(topic" in text
    assert text.endswith("e.g. memory.items('notes/grid', kind='note').")
    with pytest.raises(TypeError):  # still a TypeError for a caller that catches one
        items("a", "note", 3, 4)


def test_conforms_covers_the_forms_a_contract_uses():
    assert cr.conforms(1, float) and not cr.conforms(True, float) and not cr.conforms(True, int)
    assert cr.conforms(["a"], list[str]) and not cr.conforms(["a", 1], list[str]) and not cr.conforms("ab", list[str])
    assert cr.conforms({"a": 1}, dict[str, int]) and not cr.conforms({1: 1}, dict[str, int])
    assert cr.conforms(("a", 1), tuple[str, int]) and cr.conforms(("a", "b"), tuple[str, ...])
    assert cr.conforms(None, int | None) and not cr.conforms("x", int | None)
    assert cr.conforms("", str) and not cr.conforms("", Annotated[str, cr.MinLen(1)])
    assert cr.conforms(len, Any) and cr.conforms(print, Any)
    assert cr.type_text(Annotated[str, cr.MinLen(1)]) == "str non-empty"
    assert cr.type_text(tuple[str, ...]) == "tuple[str, ...]"


def test_an_async_call_is_checked_too():
    @cr.checked(name="memory.ask")
    async def ask(request: str) -> str:
        """Ask.

        Example:
            memory.ask('rotate a grid')
        """
        return request

    assert asyncio.run(ask("x")) == "x"
    with pytest.raises(cr.ArgumentError, match=r"request is 3 \(int\)"):
        asyncio.run(ask(3))


def test_call_text_cuts_long_values():
    text = cr.call_text("memory.get", ["x" * 500])
    assert len(text) < 80 and text.endswith("…)")


@dataclasses.dataclass(frozen=True)
class Fit:
    """Whether an item applies to the inputs."""

    ok: bool
    why: str

    def explain(self, verbose: bool = False) -> str:
        """The reason, longer with *verbose*."""
        return self.why


def test_the_stub_reads_as_typed_python():
    stub = cr.render_pyi({"items": items, "Fit": Fit}, header="memory: the library's contract")
    assert stub.startswith("# memory: the library's contract\n")
    assert (
        "def items(topic: str | None = None, kind: Literal['function', 'note'] | None = None, "
        "limit: int = 20) -> list[str]:"
    ) in stub
    assert "@dataclass(frozen=True)\nclass Fit:" in stub and "    ok: bool\n    why: str" in stub
    assert "    def explain(self, verbose: bool = False) -> str:" in stub
    compile(stub, "<stub>", "exec")  # it is valid Python


def test_the_runtime_is_standard_library_only():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(cr.__file__).read_text())
    imported = {
        (n.module or "").split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
    } | {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported <= {"__future__", "dataclasses", "difflib", "functools", "inspect", "types", "typing", "collections"}
