"""Symbolic and sandboxed: what the code-cell tool asks for, shows and exemplifies.

In the Python-tool-mode ARC LOW runs (af8958e5d) 54-66% of the model's cells
were narration: a printed sentence, a comment or ``None``, with the reasoning
in ``execute_code``'s required ``thought`` ("shown to the user as the
rationale for this step") and no reasoning tokens. Each result came back as a
JSON envelope of session metadata before what the cell printed, and nothing
showed a turn that computes in a cell, reads the output and replies. Three
off-by-default switches (unify/actor/code_cells.py):

* ``UNIFY_CODE_ONLY_CELLS``: ``code`` is the only required argument and there
  is no ``thought``; without primitives, no ``include_parent_chat_context``.
* ``UNIFY_PLAIN_CELL_OUTPUT``: a result reads as a notebook cell's: stdout,
  stderr, ``Out: <repr>``, the traceback.
* ``UNIFY_CODE_EXAMPLE_TURN``: one worked turn, on no domain, in the tool's
  description.

The model, where there is one, is the scripted transport of
tests/cache_discipline_helpers.py; the last tests' cells run in the real
sandboxed worker.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.helpers import _handle_project
from unify.actor import code_cells, core_surface
from unify.common.llm_helpers import method_to_schema
from unify.settings import ProductionSettings, SETTINGS

SWITCHES = (
    "UNIFY_CODE_ONLY_CELLS",
    "UNIFY_PLAIN_CELL_OUTPUT",
    "UNIFY_CODE_EXAMPLE_TURN",
)


def _actor(environments=None, **kwargs):
    from unify.actor.code_act_actor import CodeActActor

    return CodeActActor(environments=environments or [], **kwargs)


def _code_tool(actor, *, core=False):
    tools = actor.get_tools("act")
    if core:
        tools = core_surface.core_tools(tools, steering=False)
    tool = tools["execute_code"]
    return method_to_schema(
        getattr(tool, "fn", tool),
        "execute_code",
        expose_context_control=True,
    )["function"]


@pytest.fixture(params=["", "sandboxed"])
def workspace(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE", request.param)
    return request.param


def test_off_by_default():
    defaults = ProductionSettings()
    for name in SWITCHES:
        assert getattr(defaults, name) is False, name
        assert getattr(ProductionSettings(**{name: "1"}), name) is True, name
        assert getattr(SETTINGS, name) is False, name


# ── UNIFY_CODE_ONLY_CELLS ────────────────────────────────────────────────────


@pytest.mark.parametrize("core", [False, True])
def test_off_the_cell_tool_is_as_shipped(workspace, core):
    tool = _code_tool(_actor(), core=core)
    params = tool["parameters"]
    assert params["required"] == ["thought"]
    assert params["properties"]["code"] == {"type": ["string", "null"]}
    assert "Shown to the user" in params["properties"]["thought"]["description"]
    assert "include_parent_chat_context" in params["properties"]


@pytest.mark.parametrize("core", [False, True])
def test_on_code_is_the_only_required_argument(workspace, core, monkeypatch):
    shipped = _code_tool(_actor(), core=core)
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    tool = _code_tool(_actor(), core=core)
    params = tool["parameters"]
    assert params["required"] == ["code"]
    assert list(params["properties"])[0] == "code"
    assert params["properties"]["code"] == {"type": "string"}
    assert "thought" not in params["properties"]
    assert "include_parent_chat_context" not in params["properties"]
    # Everything else is as shipped: the other arguments and the description.
    kept = {
        k: v
        for k, v in shipped["parameters"]["properties"].items()
        if k not in ("thought", "code", "include_parent_chat_context")
    }
    assert {k: v for k, v in params["properties"].items() if k != "code"} == kept
    assert tool["description"] == shipped["description"]
    assert ("language" in params["properties"]) == (workspace == "sandboxed")


def test_with_primitives_the_parent_context_flag_stays(monkeypatch):
    from unify.actor.environments import ActorEnvironment

    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    tool = _code_tool(_actor([ActorEnvironment()]))
    assert "include_parent_chat_context" in tool["parameters"]["properties"]
    assert tool["parameters"]["required"] == ["code"]


def test_composes_with_stateful_cells_and_the_trim(workspace, monkeypatch):
    for name in ("UNIFY_CODE_ONLY_CELLS", "UNIFY_STATEFUL_CELLS", "UNIFY_PROMPT_TRIM"):
        monkeypatch.setattr(SETTINGS, name, True)
    for core in (False, True):
        params = _code_tool(_actor(), core=core)["parameters"]
        expected = ["code", "language"] if workspace else ["code"]
        assert list(params["properties"]) == expected
        assert params["required"] == ["code"]


def test_execute_function_is_unchanged(monkeypatch):
    def schema():
        tool = _actor(function_manager=_fm()).get_tools("act")["execute_function"]
        return method_to_schema(getattr(tool, "fn", tool), "execute_function")

    shipped = schema()
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    assert schema() == shipped


def _fm():
    from unify.function_manager.function_manager import FunctionManager

    return FunctionManager(include_primitives=False)


def test_the_switch_reaches_only_the_actor_built_with_it(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    on = _actor()
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", False)
    off = _actor()
    assert "thought" in _code_tool(off)["parameters"]["properties"]
    on_tool = on.get_tools("act")["execute_code"]
    assert (
        "thought"
        not in method_to_schema(
            getattr(on_tool, "fn", on_tool),
            "execute_code",
        )[
            "function"
        ]["parameters"]["properties"]
    )


def _cell(code: str, **extra):
    return lambda: h.completion(calls=[("execute_code", {"code": code, **extra})])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_a_thought_is_refused_and_a_cell_without_one_runs(monkeypatch):
    """In process: the call that still passes ``thought`` is refused naming
    the tool's parameters; the next one, without it, runs."""
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_ONLY_CELLS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    actor = _actor(can_store=False)
    replies = (
        _cell("print('A' * 2)", thought="Announcing the step."),
        _cell("print('B' * 2)"),
        lambda: h.completion(content="done"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Print two letters.", persist=False)
            result = await asyncio.wait_for(handle.result(), 90)
    finally:
        await actor.close()
    assert result == "done"
    tool_messages = [
        json.dumps(m["content"])
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "thought" in tool_messages[0] and "code" in tool_messages[0]
    assert "AA" not in tool_messages[0]
    assert "BB" in tool_messages[1]


# ── UNIFY_PLAIN_CELL_OUTPUT: the description ────────────────────────────────


@pytest.mark.parametrize("core", [False, True])
def test_the_output_section_describes_the_plain_output(workspace, core, monkeypatch):
    shipped = _code_tool(_actor(), core=core)["description"]
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", True)
    on = _code_tool(_actor(), core=core)["description"]
    assert "An ExecutionResult with" in shipped
    assert "An ExecutionResult with" not in on and "session_created" not in on
    assert "``Out: <repr>``" in on and "``[stderr]``" in on
    # The handle sentence stays where the shipped section had it.
    assert ("steerable handle" in on) == ("steerable handle" in shipped)
    if workspace:
        assert "``Out:`` is the exit status" in " ".join(on.split())
    # Only the Output section (and the bash bullet's word) changed.
    before = shipped.split("Output\n------")[0].replace("``result``", "``Out:``")
    assert on.split("Output\n------")[0] == before


def test_the_plain_description_composes_with_stateful_cells(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STATEFUL_CELLS", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_PLAIN_CELL_OUTPUT", True)
    on = _code_tool(_actor())["description"]
    assert "session_id" not in on and "``Out: <repr>``" in on


# ── UNIFY_CODE_EXAMPLE_TURN ──────────────────────────────────────────────────

# Words of the benchmarks the switch was found on; the turn must not use any.
BENCHMARK_WORDS = re.compile(
    r"\b(arc|grid|puzzle|demo|submit|task id|appworld|scienceworld|crafter)\b",
    re.I,
)


@pytest.mark.parametrize("core", [False, True])
def test_the_description_ends_with_one_worked_turn(workspace, core, monkeypatch):
    shipped = _code_tool(_actor(), core=core)["description"]
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_EXAMPLE_TURN", True)
    on = _code_tool(_actor(), core=core)["description"]
    assert on.startswith(shipped.rstrip())
    turn = on[len(shipped.rstrip()) :].strip()
    assert turn == code_cells.EXAMPLE_TURN
    assert turn.count("Example turn") == 1
    assert "reply in the requester's format" in turn
    assert "No cell is needed to take an action" in turn
    assert not BENCHMARK_WORDS.search(turn), BENCHMARK_WORDS.search(turn)


def test_the_example_runs_as_written():
    """The two cells of the example compute what it says they show."""
    blocks = re.findall(r"::\n\n((?:    .*\n?)+)", code_cells.EXAMPLE_TURN)
    namespace: dict = {}
    for block in blocks:
        exec("\n".join(line[4:] for line in block.splitlines()), namespace)
    assert namespace["total"] == pytest.approx(3 * 1.25 + 2 * 4.0)


def test_core_keeps_the_turn_after_the_steering_section_is_removed(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CODE_EXAMPLE_TURN", True)
    on = _code_tool(_actor(), core=True)["description"]
    assert "Steering while the block runs" not in on
    assert on.rstrip().endswith("defines is taken by replying.")


# ── all of them, in the real sandboxed worker ───────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
@pytest.mark.parametrize("surface", ["", "core"])
async def test_cells_in_the_sandboxed_worker_read_as_a_notebook(
    world,  # noqa: F811
    monkeypatch,
    surface,
):
    """Lean prompt, worker Python, the lean fixes and the three switches on:
    the tool asks for ``code`` only; the results carry what the cells printed,
    their last values and the traceback, and no session metadata."""
    from unify import db
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    if surface == "core":
        (world["state"] / "store.sqlite").unlink()
        db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", surface)
    for name in ("UNIFY_STATEFUL_CELLS", "UNIFY_PROMPT_TRIM", *SWITCHES):
        monkeypatch.setattr(SETTINGS, name, True)
    actor = _actor(
        function_manager=FunctionManager(include_primitives=False),
        guidance_manager=GuidanceManager(),
        can_store=False,
    )
    replies = (
        _cell('items = [("pen", 3, 1.25), ("pad", 2, 4.0)]'),
        _cell(
            "import sys\nprint('isolated', sys.flags.isolated)\n"
            "total = sum(q * p for _, q, p in items)\ntotal",
        ),
        _cell("import sys\nprint('warn', file=sys.stderr)\n{'n': len(items)}"),
        _cell("items[5]"),
        lambda: h.completion(content="11.75"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("What do the items cost?", persist=False)
            result = await asyncio.wait_for(handle.result(), 150)
    finally:
        await actor.close()
        if surface == "core":
            db.reset_store()
    assert result == "11.75"
    tool = next(
        t
        for t in provider.requests[0]["tools"]
        if t["function"]["name"] == "execute_code"
    )["function"]
    assert tool["parameters"]["required"] == ["code"]
    assert set(tool["parameters"]["properties"]) == {"code", "language"}
    assert code_cells.EXAMPLE_TURN.splitlines()[0] in tool["description"]
    replies_seen = [
        m["content"]
        for m in provider.requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    texts = [
        "".join(b["text"] for b in c if b.get("type") == "text") for c in replies_seen
    ]
    assert texts[0] == "(no output)"
    # The cells ran in the worker (``python -I``) and kept ``items``.
    assert texts[1] == "isolated 1\nOut: 11.75"
    assert texts[2] == "[stderr]\nwarn\nOut: {'n': 2}", texts[2]
    assert "IndexError" in texts[3] and not texts[3].startswith("{"), texts[3]
    for text in texts:
        for word in ("session_created", "duration_ms", "state_mode", "--- stdout"):
            assert word not in text, text
