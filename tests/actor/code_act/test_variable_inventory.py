"""Symbolic: ``UNIFY_VARIABLE_INVENTORY=on`` lists the session's variables.

On Continual-ARC 88% of the code behind accepted answers held the request's
input as a pasted literal, and with ``UNIFY_STATEFUL_CELLS`` half of the
computation cells built on an earlier variable: the model works from what
it remembers is on the bench. With the switch on, an ``execute_code``
result ends with one line naming the variables the session's cells have
bound, each with its type and a short shape (the value itself for a scalar
or a short string), most recently bound first: at most 12 names and 400
characters. The line comes only when the names or their shapes changed
since the last one shown, never when there are none, and never for a cell
that keeps nothing (stateless, read-only). Harness globals are left out.
It is the tool's own result, so a budget footer still comes after it, and
no earlier message changes. Off, results are as shipped.

The transport is scripted (``tests/cache_discipline_helpers.py``): nothing
leaves the process. Each test that drives a session bounds every wait.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import re
import time
import types

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import core_world  # noqa: F401
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor import notebook_cells
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import ExecutionResult, TextPart
from unify.settings import SETTINGS

SESSION_BOUND_S = 2.0
LABEL = "[variables] "
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PANDAS = importlib.util.find_spec("pandas") is not None


def _inventory_cls():
    from unify.actor.execution.worker_child import Inventory

    return Inventory


def _describe(value):
    from unify.actor.execution.worker_child import describe_value

    return describe_value(value)


def _cell_ns(source: str, ns: dict | None = None) -> tuple[dict, dict]:
    """Run *source* as a cell would run in *ns*: snapshot, exec, return both."""
    ns = {} if ns is None else ns
    ns.setdefault("__name__", "__sandbox_test__")
    before = _inventory_cls().snapshot(ns)
    exec(compile(source, "<string>", "exec"), ns)
    return ns, before


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_VARIABLE_INVENTORY", "on")


# ── the switch ───────────────────────────────────────────────────────────


def test_the_switch_is_validated():
    from unify.settings import ProductionSettings

    assert ProductionSettings().UNIFY_VARIABLE_INVENTORY == ""
    assert (
        ProductionSettings(UNIFY_VARIABLE_INVENTORY="off").UNIFY_VARIABLE_INVENTORY
        == ""
    )
    assert (
        ProductionSettings(UNIFY_VARIABLE_INVENTORY=" On ").UNIFY_VARIABLE_INVENTORY
        == "on"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_VARIABLE_INVENTORY="yes")


# ── shapes ───────────────────────────────────────────────────────────────


def _fn_ns() -> dict:
    ns = {"__name__": "__sandbox_test__"}
    exec(
        compile(
            "def solve(grid, k=2, *rest, **opts):\n    return grid\n"
            "async def fetch(url):\n    return url\n"
            "square = lambda x: x * x\n"
            "class Point:\n    pass\n"
            "where = Point()\n",
            "<string>",
            "exec",
        ),
        ns,
    )
    return ns


@pytest.mark.parametrize(
    "value, expected",
    [
        (3, "int = 3"),
        (-2.5, "float = -2.5"),
        (True, "bool = True"),
        (None, "None"),
        ("abc", "str = 'abc'"),
        ("a\nb", "str = 'a\\nb'"),
        ("x" * 100, "str[100]"),
        (b"\x00" * 7, "bytes[7]"),
        ([1, 2, 3], "list[3]"),
        ([], "list[0]"),
        ([[0, 1], [1, 0], [2, 2]], "list[3x2]"),
        ([[1], [1, 2]], "list[2]"),
        ([(1, 2), (3, 4)], "list[2x2]"),
        ((1, 2), "tuple[2]"),
        ({"a": 1, "b": [2]}, "dict[2 keys]"),
        ({}, "dict[0 keys]"),
        ({1, 2}, "set[2]"),
        (frozenset(), "frozenset[0]"),
        (10**100, "int (333 bits)"),
        (1.0e300, "float = 1e+300"),
    ],
)
def test_a_value_is_its_type_and_a_short_shape(value, expected):
    assert _describe(value) == expected


def test_a_subclass_shows_its_own_name():
    from collections import Counter, OrderedDict, defaultdict

    assert _describe(Counter("aab")) == "Counter[2 keys]"
    assert _describe(defaultdict(list)) == "defaultdict[0 keys]"
    assert _describe(OrderedDict(a=1)) == "OrderedDict[1 keys]"


def test_a_long_scalar_is_truncated():
    text = _describe(0.1234567890123456789 * 1e-300)
    assert text.startswith("float = ") and len(text) <= len("float = ") + 24


def test_model_definitions_are_named_by_kind_and_parameters():
    ns = _fn_ns()
    assert _describe(ns["solve"]) == "function(grid, k, *rest, **opts)"
    assert _describe(ns["fetch"]) == "async function(url)"
    assert _describe(ns["square"]) == "function(x)"
    assert _describe(ns["Point"]) == "class"
    assert _describe(ns["where"]) == "Point"


def test_a_value_is_never_repr_d_whole():
    class Loud:
        def __repr__(self):  # pragma: no cover - the test fails if called
            raise AssertionError("repr called")

        def __len__(self):  # pragma: no cover
            raise AssertionError("len called")

    assert _describe(Loud()) == "Loud"
    # A big list costs a row scan, not a repr.
    big = [[0] * 30 for _ in range(30)]
    started = time.perf_counter()
    for _ in range(200):
        assert _describe(big) == "list[30x30]"
    assert time.perf_counter() - started < 1.0


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy is not installed")
def test_numpy_arrays_show_shape_and_dtype():
    import numpy as np

    assert _describe(np.zeros((3, 4), dtype=np.int64)) == "ndarray[3x4] int64"
    assert _describe(np.arange(5, dtype=np.float64)) == "ndarray[5] float64"
    assert _describe(np.int64(7)) == "int64 = 7"


@pytest.mark.skipif(not HAS_PANDAS, reason="pandas is not installed")
def test_pandas_frames_show_their_shape():
    import pandas as pd

    frame = pd.DataFrame({"a": range(10), "b": range(10)})
    assert _describe(frame) == "DataFrame[10x2]"
    assert _describe(frame["a"]) == "Series[10] int64"


# ── what is left out ─────────────────────────────────────────────────────


def test_harness_globals_are_left_out():
    from unify.actor.execution.worker_child import Reply, Request

    harness = {
        "primitives": object(),
        "request": Request("[1, 2]"),
        "reply": Reply(),
        "record": object(),
        "agents": object(),
        "display": print,
        "functions": object(),
        "guidance": object(),
        "steering": object(),
    }
    ns = {"__name__": "__sandbox_test__", **harness}
    inv = _inventory_cls()()
    ns, before = _cell_ns(
        "import json\n"
        "import collections as coll\n"
        "from os import path\n"
        "_private = 1\n"
        "loads = json.loads\n"
        "kept = [1, 2]\n"
        # A cell that rebinds harness names: still the harness's names.
        "primitives = 5\n" "display = 6\n",
        ns,
    )
    text = inv.after_cell(ns, before)
    assert text == "[variables] kept: list[2]"


def test_names_the_harness_installed_are_left_out():
    inv = _inventory_cls()()
    stored = types.SimpleNamespace(run=1)
    ns = {"__name__": "__sandbox_test__"}
    before = inv.snapshot(ns)
    ns["helper_ns"] = stored  # installed by the harness during the cell
    ns["mine"] = 1
    assert inv.after_cell(ns, before, installed={"helper_ns": stored}) == (
        "[variables] mine: int = 1"
    )


def test_a_stored_function_is_left_out():
    """A library function the harness compiled into the namespace (its code
    is labelled ``<function:NAME>``) is not a variable the model bound."""
    ns = {"__name__": "__sandbox_test__"}
    inv = _inventory_cls()()
    before = inv.snapshot(ns)
    exec(compile("def stored(x):\n    return x\n", "<function:stored>", "exec"), ns)
    exec(compile("def mine(x):\n    return x\n", "<string>", "exec"), ns)
    assert inv.after_cell(ns, before) == "[variables] mine: function(x)"


# ── order, limits and when it is shown ───────────────────────────────────


def test_most_recently_bound_first_then_by_name():
    inv = _inventory_cls()()
    ns, before = _cell_ns("b = 1\na = 2\n")
    assert inv.after_cell(ns, before) == "[variables] a: int = 2; b: int = 1"
    ns, before = _cell_ns("c = 'x'\n", ns)
    assert inv.after_cell(ns, before) == (
        "[variables] c: str = 'x'; a: int = 2; b: int = 1"
    )
    # Rebinding a name moves it to the front.
    ns, before = _cell_ns("b = 3\n", ns)
    assert inv.after_cell(ns, before) == (
        "[variables] b: int = 3; c: str = 'x'; a: int = 2"
    )
    # So does a change of shape in place.
    ns, before = _cell_ns("rows = []\n", ns)
    inv.after_cell(ns, before)
    ns, before = _cell_ns("c = 'y'\n", ns)
    inv.after_cell(ns, before)
    ns, before = _cell_ns("rows.append(1)\n", ns)
    text = inv.after_cell(ns, before)
    assert text.startswith("[variables] rows: list[1]; c: str = 'y'; b: int = 3")


def test_at_most_twelve_names_then_how_many_more():
    inv = _inventory_cls()()
    source = "".join(f"v{i:02d} = {i}\n" for i in range(15))
    ns, before = _cell_ns(source)
    text = inv.after_cell(ns, before)
    shown = [part.split(":")[0] for part in text[len(LABEL) :].split("; ")]
    assert shown[:12] == [f"v{i:02d}" for i in range(12)]
    assert shown[12] == "…and 3 more"
    assert len(shown) == 13


def test_at_most_four_hundred_characters():
    inv = _inventory_cls()()
    source = "".join(f"{'n' * 60}_{i} = {i}\n" for i in range(8))
    ns, before = _cell_ns(source)
    text = inv.after_cell(ns, before)
    assert len(text) <= 400
    assert re.search(r"; …and (\d+) more$", text)
    kept = text[len(LABEL) :].split("; ")[:-1]
    more = int(re.search(r"(\d+) more$", text).group(1))
    assert len(kept) + more == 8
    assert all(entry.endswith(f"= {i}") for i, entry in enumerate(kept))


def test_shown_only_when_names_or_shapes_change():
    inv = _inventory_cls()()
    ns, before = _cell_ns("grid = [[0, 1], [1, 0]]\n")
    assert inv.after_cell(ns, before) == "[variables] grid: list[2x2]"
    # Nothing new bound: nothing shown.
    ns, before = _cell_ns("print(grid)\n", ns)
    assert inv.after_cell(ns, before) is None
    # A value change that keeps the shape: nothing shown.
    ns, before = _cell_ns("grid[0][0] = 5\n", ns)
    assert inv.after_cell(ns, before) is None
    # Rebound to an equal shape: still nothing new to say.
    ns, before = _cell_ns("grid = [[1, 1], [1, 1]]\n", ns)
    assert inv.after_cell(ns, before) is None
    # A new shape is shown.
    ns, before = _cell_ns("grid.append([2, 2])\n", ns)
    assert inv.after_cell(ns, before) == "[variables] grid: list[3x2]"
    # A deletion changes the set.
    ns, before = _cell_ns("k = 1\n", ns)
    assert inv.after_cell(ns, before) == "[variables] k: int = 1; grid: list[3x2]"
    ns, before = _cell_ns("del k\n", ns)
    assert inv.after_cell(ns, before) == "[variables] grid: list[3x2]"


def test_nothing_when_there_are_no_variables():
    inv = _inventory_cls()()
    ns, before = _cell_ns("print(1)\n")
    assert inv.after_cell(ns, before) is None
    ns, before = _cell_ns("x = 1\n", ns)
    assert inv.after_cell(ns, before) == "[variables] x: int = 1"
    ns, before = _cell_ns("del x\n", ns)
    assert inv.after_cell(ns, before) is None
    # Bound again after the empty state: shown again.
    ns, before = _cell_ns("x = 1\n", ns)
    assert inv.after_cell(ns, before) == "[variables] x: int = 1"


# ── the session, in process and in the worker ────────────────────────────

CELLS = [
    ("print('hello')", "stateful"),
    ("grid = [[0, 1], [1, 0]]\nk = 2", "stateful"),
    ("print(len(grid))", "stateful"),
    ("for i in range(2):\n    last = i", "stateful"),
    ("tmp = 1", "stateless"),
    ("print(grid)", "read_only"),
    ("def flip(g):\n    return [row[::-1] for row in g]", "stateful"),
    ("out = flip(grid)", "stateful"),
    ("grid.append([3, 3])", "stateful"),
    ("del k", "stateful"),
    ("raise ValueError('boom')", "stateful"),
    ("partial = 1\nraise ValueError('after binding')", "stateful"),
]

EXPECTED = [
    None,
    "[variables] grid: list[2x2]; k: int = 2",
    None,
    "[variables] i: int = 1; last: int = 1; grid: list[2x2]; k: int = 2",
    None,
    None,
    "[variables] flip: function(g); i: int = 1; last: int = 1; "
    "grid: list[2x2]; k: int = 2",
    "[variables] out: list[2x2]; flip: function(g); i: int = 1; last: int = 1; "
    "grid: list[2x2]; k: int = 2",
    "[variables] grid: list[3x2]; out: list[2x2]; flip: function(g); i: int = 1; "
    "last: int = 1; k: int = 2",
    "[variables] grid: list[3x2]; out: list[2x2]; flip: function(g); i: int = 1; "
    "last: int = 1",
    None,
    "[variables] partial: int = 1; grid: list[3x2]; out: list[2x2]; "
    "flip: function(g); i: int = 1; last: int = 1",
]


async def _drive(ex: SessionExecutor, bound: float = SESSION_BOUND_S) -> list:
    seen = []
    for code, mode in CELLS:
        res = await asyncio.wait_for(
            ex.execute(
                code=code,
                state_mode=mode,
                session_id=None if mode == "stateless" else 0,
                inventory=True,
            ),
            bound,
        )
        seen.append(res.get("inventory"))
    return seen


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_the_worker_reports_the_same(on, world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_BIND_REQUEST", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_REPLY_CHANNEL", "code+text")
    from unify.common._async_tool import bound_request

    token = bound_request.bind(True)
    bound_request.current().text = "[[1, 2]]"
    ex = SessionExecutor()
    try:
        await asyncio.wait_for(
            ex.execute(code="1", state_mode="stateful", session_id=0),
            10,
        )  # starts the worker
        # A stateless cell starts a worker of its own.
        assert await _drive(ex, bound=15) == EXPECTED
        assert ex.python_session(session_id=0)._worker is not None
    finally:
        bound_request.unbind(token)
        await ex.close()


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_nothing_unless_asked(on):
    ex = SessionExecutor()
    try:
        res = await ex.execute(code="x = 1", state_mode="stateful", session_id=0)
        assert "inventory" not in res
        # Asked later: the binding is new to what has been shown.
        res = await ex.execute(
            code="y = 2",
            state_mode="stateful",
            session_id=0,
            inventory=True,
        )
        assert res["inventory"] == "[variables] y: int = 2"
    finally:
        await ex.close()


# ── how the result shows it ──────────────────────────────────────────────

LINE = "[variables] grid: list[2x2]; k: int = 2"


def _texts(blocks: list[dict]) -> str:
    return "".join(b.get("text", "") for b in blocks)


def test_the_result_ends_with_the_line():
    base = dict(
        stdout=[TextPart(text="hello\n")],
        stderr=[],
        result=None,
        error=None,
        state_mode="stateful",
        session_id=0,
        duration_ms=3,
    )
    off = ExecutionResult(**base).to_llm_content()
    on = ExecutionResult(**base, inventory=LINE).to_llm_content()
    assert on[: len(off)] == off
    assert on[len(off) :] == [{"type": "text", "text": "\n" + LINE}]
    # A notebook cell: the last line, after the traceback.
    nb_base = {**base, "error": "Traceback ...\nValueError: x\n"}
    nb_off = notebook_cells._as_cell_result(ExecutionResult(**nb_base))
    nb_on = notebook_cells._as_cell_result(ExecutionResult(**nb_base, inventory=LINE))
    assert _texts(nb_on.to_llm_content()) == (
        _texts(nb_off.to_llm_content()) + "\n" + LINE
    )
    # A cell with no output at all is just the line.
    bare = notebook_cells._as_cell_result(
        ExecutionResult(stdout=[], stderr=[], inventory=LINE),
    )
    assert bare.to_llm_content() == [{"type": "text", "text": LINE}]


def test_off_the_result_is_as_shipped():
    out = ExecutionResult(stdout=[TextPart(text="a\n")], state_mode="stateful")
    assert out.inventory is None
    assert "[variables]" not in json.dumps(out.to_llm_content())


# ── the actor ────────────────────────────────────────────────────────────

ACT_CELLS = ["grid = [[1, 2], [3, 4]]", "print(grid)", "total = sum(map(sum, grid))"]


async def _act(actor) -> tuple[list[str], list]:
    replies = [h.completion(calls=[("execute_code", {"code": c})]) for c in ACT_CELLS]
    replies.append(h.completion(content="10"))
    with h.scripted(replies) as provider:
        handle = await actor.act(
            "Add up the numbers and reply with the total.",
            persist=False,
            can_store=False,
            clarification_enabled=False,
        )
        result = await asyncio.wait_for(handle.result(), SESSION_BOUND_S * 3)
    assert result == "10"
    tools = [
        json.dumps(m["content"]) if not isinstance(m["content"], str) else m["content"]
        for m in handle._client.messages
        if m.get("role") == "tool"
    ]
    return tools, provider.requests


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("projection", ["", "notebook"])
async def test_an_act_on_the_core_surface_lists_the_workers_variables(
    on,
    core_world,  # noqa: F811
    monkeypatch,
    projection,
):
    """The core surface's sandbox holds ``functions`` and ``guidance``; its
    cells run in the worker, which lists what they bound and not those."""
    from tests.actor.code_act.core_world import new_actor

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_PROJECTION", projection)
    actor = new_actor()
    try:
        tools, _requests = await _act(actor)
    finally:
        await actor.close()
    assert "[variables] grid: list[2x2]" in tools[0]
    assert "[variables]" not in tools[1]
    assert "[variables] total: int = 10; grid: list[2x2]" in tools[2]
    assert not re.search(r"\b(functions|guidance):", " ".join(tools))


def test_the_tools_and_prompt_are_unchanged_by_the_switch(monkeypatch):
    from unify.actor import prompt_builders as pb
    from unify.actor.code_act_actor import CodeActActor
    from unify.common.llm_helpers import method_to_schema

    def surface() -> tuple[dict, str]:
        actor = CodeActActor()
        tools = actor.get_tools("act")
        schemas = {
            name: method_to_schema(getattr(tool, "fn", tool), name)
            for name, tool in tools.items()
        }
        prompt = pb.build_code_act_prompt(
            environments=actor.environments,
            tools=dict(tools),
            can_store=True,
        )
        return schemas, prompt

    monkeypatch.setattr(SETTINGS, "UNIFY_VARIABLE_INVENTORY", "")
    off = surface()
    monkeypatch.setattr(SETTINGS, "UNIFY_VARIABLE_INVENTORY", "on")
    assert surface() == off
