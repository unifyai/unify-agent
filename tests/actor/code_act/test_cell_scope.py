"""Symbolic: every name a cell binds is there for the next cell, as in a notebook.

A cell runs as the body of ``async def __exec_wrapper()``. Only the names the
wrapper declares ``global`` reach the session; any other name the cell binds
is a local of the wrapper and is gone when it returns. With
``UNIFY_CELL_SCOPE_FIX`` on (the default) the declaration is every name
Python's own symbol table finds the cell's scope binding: under ``if``,
``for``, ``while``, ``with``, ``try``, ``except``, ``match``, a walrus
(also one inside a comprehension), ``del``, and annotated assignments (whose
annotation is dropped, because an annotated name cannot be declared global).
Off, it is the names bound by top-level assignments, imports, defs and
classes only, as shipped. Nested scopes keep their own names either way.

Cells run through the real ``SessionExecutor``, in the sandboxed worker
(skipped, saying so, where bubblewrap is missing). Nothing here reaches a
model.
"""

from __future__ import annotations

import contextlib

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.settings import ProductionSettings, SETTINGS


def has(name: str) -> str:
    """Cell code whose value says whether *name* is bound (no ``globals()``
    in the restricted builtins)."""
    return (
        f"try:\n    {name}\n    __found = True\n"
        "except NameError:\n    __found = False\n__found"
    )


def last_line(error) -> str:
    return str(error).strip().splitlines()[-1] if error else ""


@pytest.fixture(params=[pytest.param("worker", marks=needs_bwrap)])
def where(request):
    request.getfixturevalue("world")
    return request.param


@pytest.fixture(params=[True, False], ids=["fix_on", "fix_off"])
def fix(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_SCOPE_FIX", request.param)
    return request.param


class FakeFiles:
    async def search(self, query, limit=3):
        return {"hits": [f"{query}-{i}" for i in range(limit)]}


class FakePrimitives:
    def __init__(self) -> None:
        self.files = FakeFiles()


class FakeEnvironment:
    def __init__(self) -> None:
        self._instance = FakePrimitives()

    def get_instance(self):
        return self._instance


@contextlib.asynccontextmanager
async def executor(environments=None):
    ex = SessionExecutor(environments=environments or {}, timeout=60)
    try:
        yield ex
    finally:
        await ex.close()


async def run(ex, code, mode="stateful"):
    return await ex.execute(
        code=code,
        state_mode=mode,
        session_id=None if mode == "stateless" else 0,
    )


# (setup cell, next cell, its stdout with the fix, how its error starts without)
PERSISTS = {
    "if": (
        "if True:\n    y = 1",
        "print(y)",
        "1",
        "NameError: name 'y' is not defined",
    ),
    "for": (
        "for i in range(3):\n    acc = i",
        "print(acc, i)",
        "2 2",
        "NameError: name 'acc' is not defined",
    ),
    "while": (
        "n = 0\nwhile n < 3:\n    n += 1\n    last = n",
        "print(last)",
        "3",
        "NameError: name 'last' is not defined",
    ),
    "with": (
        "import io\nwith io.StringIO('a') as fh:\n    t = fh.read()",
        "print(t, fh.closed)",
        "a True",
        "NameError: name 't' is not defined",
    ),
    "try": (
        "try:\n    z = 5\nexcept Exception:\n    pass",
        "print(z)",
        "5",
        "NameError: name 'z' is not defined",
    ),
    "except": (
        "try:\n    1 / 0\nexcept ZeroDivisionError as exc:\n    msg = str(exc)",
        "print(msg)",
        "division by zero",
        "NameError: name 'msg' is not defined",
    ),
    "import_in_try": (
        "try:\n    import fractions\nexcept ImportError:\n    fractions = None",
        "print(fractions.Fraction(1, 2))",
        "1/2",
        "NameError: name 'fractions' is not defined",
    ),
    "walrus": ("(w := 7)", "print(w)", "7", "NameError: name 'w' is not defined"),
    "walrus_in_comprehension": (
        "[(sq := k * k) for k in range(4)]",
        "print(sq)",
        "9",
        "NameError: name 'sq' is not defined",
    ),
    "def_in_if": (
        "if True:\n    def helper():\n        return 3",
        "print(helper())",
        "3",
        "NameError: name 'helper' is not defined",
    ),
    "match": (
        "match [1, 2]:\n    case [first, *rest]:\n        pass",
        "print(first, rest)",
        "1 [2]",
        "NameError: name 'first' is not defined",
    ),
    # Without the fix the setup cell itself is a SyntaxError ("annotated
    # name 'total' can't be global").
    "annotated": (
        "total: int = 4",
        "print(total)",
        "4",
        "NameError: name 'total' is not defined",
    ),
    # The shadowing guard still renames the binding, wherever it is.
    "shadowing_guard": (
        "if True:\n    primitives = 3",
        "print(_primitives_local)",
        "3",
        "NameError: name '_primitives_local' is not defined",
    ),
}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("case", sorted(PERSISTS))
async def test_a_name_bound_anywhere_in_a_cell_reaches_the_next(case, where, fix):
    setup, probe, printed, error_off = PERSISTS[case]
    async with executor() as ex:
        first = await run(ex, setup)
        second = await run(ex, probe)
    if fix:
        assert first["error"] is None, first["error"]
        assert second["error"] is None, second["error"]
        assert parts_to_text(second["stdout"]).strip() == printed
    else:
        # (Python may add a suggestion: "Did you mean: ...?")
        assert last_line(second["error"]).startswith(error_off)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_del_removes_an_earlier_cells_name(where, fix):
    async with executor() as ex:
        await run(ex, "x = 1")
        deleted = await run(ex, "del x")
        after = await run(ex, has("x"))
    if fix:
        assert deleted["error"] is None, deleted["error"]
        assert after["result"] is False
    else:
        assert last_line(deleted["error"]) == (
            "UnboundLocalError: cannot access local variable 'x' where it is "
            "not associated with a value"
        )
        assert after["result"] is True


# A cell that calls the injected primitives, and what it prints.
USES_PRIMITIVES = "r = await primitives.files.search('q', limit=1)\nprint(r['hits'])"
USED = "['q-0']\n"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_del_cannot_remove_the_injected_primitives(where, fix):
    async with executor({"primitives": FakeEnvironment()}) as ex:
        deleted = await run(ex, "del primitives")
        used = await run(ex, USES_PRIMITIVES)
        # The same cell goes on to use them after the deletion fails.
        same_cell = await run(
            ex,
            "try:\n    del primitives\nexcept NameError:\n    pass\n" + USES_PRIMITIVES,
        )
    if fix:
        # The shadowing guard renames the deletion too: the cell never bound
        # a `_primitives_local`, so there is nothing to delete.
        assert last_line(deleted["error"]).startswith(
            "NameError: name '_primitives_local' is not defined",
        )
        assert parts_to_text(same_cell["stdout"]) == USED, same_cell["error"]
    else:
        assert last_line(deleted["error"]) == (
            "UnboundLocalError: cannot access local variable 'primitives' "
            "where it is not associated with a value"
        )
        assert last_line(same_cell["error"]).startswith("UnboundLocalError")
    assert parts_to_text(used["stdout"]) == USED, used["error"]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_del_of_a_shadowing_primitives_removes_only_the_cells_copy(
    where,
    fix,
):
    async with executor({"primitives": FakeEnvironment()}) as ex:
        bound = await run(ex, "primitives = 1")
        deleted = await run(ex, "del primitives")
        local = await run(ex, has("_primitives_local"))
        used = await run(ex, USES_PRIMITIVES)
    assert bound["error"] is None, bound["error"]
    if fix:
        assert deleted["error"] is None, deleted["error"]
        assert local["result"] is False
    else:
        assert last_line(deleted["error"]) == (
            "UnboundLocalError: cannot access local variable 'primitives' "
            "where it is not associated with a value"
        )
        assert local["result"] is True
    assert parts_to_text(used["stdout"]) == USED, used["error"]


NESTED = {
    "comprehension": ("[q for q in range(3)]", "q"),
    "generator": ("sum(g for g in range(3))", "g"),
    "function": ("def f(a):\n    b = a\n    return b\nf(1)", "b"),
    "lambda": ("(lambda p: p)(1)", "p"),
    "class_body": ("class C:\n    cv = 1", "cv"),
}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize("case", sorted(NESTED))
async def test_a_nested_scopes_names_stay_in_it(case, where, fix):
    code, name = NESTED[case]
    async with executor() as ex:
        ran = await run(ex, code)
        probe = await run(ex, has(name))
    assert ran["error"] is None, ran["error"]
    assert probe["result"] is False


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_global_statement_the_cell_writes_still_works(where, fix):
    async with executor() as ex:
        top = await run(ex, "global g1\ng1 = 5")
        inner = await run(ex, "def setk():\n    global k\n    k = 1\nsetk()")
        both = await run(ex, "print(g1, k)")
    assert top["error"] is None, top["error"]
    assert inner["error"] is None, inner["error"]
    assert parts_to_text(both["stdout"]) == "5 1\n"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_except_as_name_is_deleted_at_the_handlers_end(where):
    # Module (and Jupyter) semantics: the handler unbinds its name, also an
    # earlier cell's binding of it.
    async with executor() as ex:
        await run(ex, "exc = 'earlier'")
        handled = await run(
            ex,
            "try:\n    1 / 0\nexcept ZeroDivisionError as exc:\n    pass",
        )
        probe = await run(ex, has("exc"))
    assert handled["error"] is None, handled["error"]
    assert probe["result"] is False


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_read_only_and_stateless_cells_still_keep_nothing(where, fix):
    async with executor() as ex:
        await run(ex, "x = 1")
        what_if = await run(
            ex,
            "x = 2\nif True:\n    y = 3\nprint(x, y)",
            mode="read_only",
        )
        scratch = await run(ex, "if True:\n    s = 1\nprint(s)", mode="stateless")
        again = await run(ex, has("s"), mode="stateless")
        kept = await run(ex, "print(x)")
        y = await run(ex, has("y"))
        s = await run(ex, has("s"))
    assert parts_to_text(what_if["stdout"]) == "2 3\n", what_if["error"]
    assert parts_to_text(scratch["stdout"]) == "1\n", scratch["error"]
    assert again["result"] is False
    assert parts_to_text(kept["stdout"]) == "1\n"
    assert y["result"] is False
    assert s["result"] is False


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_last_expression_is_still_the_cells_result(where, fix):
    async with executor() as ex:
        product = await run(ex, "a = 2\nif True:\n    b = 3\na * b")
        alone = await run(ex, "if True:\n    c = 1\nc")
        printed = await run(ex, "print('no value')")
    assert product["result"] == 6
    assert alone["result"] == 1
    assert printed["result"] is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "code",
    ["x = (", "x = 1\nglobal x", "nonlocal q\nq = 1", "from os import *"],
)
async def test_a_cell_that_does_not_compile_reports_as_before(
    code,
    where,
    monkeypatch,
):
    errors = {}
    for on in (True, False):
        monkeypatch.setattr(SETTINGS, "UNIFY_CELL_SCOPE_FIX", on)
        async with executor() as ex:
            res = await run(ex, code)
        assert res["result"] is None
        errors[on] = res["error"]
    assert "SyntaxError" in errors[False]
    assert errors[True] == errors[False]


def test_the_switch_is_on_by_default_and_turns_off():
    assert ProductionSettings().UNIFY_CELL_SCOPE_FIX is True
    for off in ("0", "false", "False", "off"):
        assert (
            ProductionSettings(UNIFY_CELL_SCOPE_FIX=off).UNIFY_CELL_SCOPE_FIX is False
        )
    assert ProductionSettings(UNIFY_CELL_SCOPE_FIX="1").UNIFY_CELL_SCOPE_FIX is True
