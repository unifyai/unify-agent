"""Symbolic and sandboxed: ``UNIFY_SHORTLIST_CALLABLE_FIRST``: the library shortlist leads with what can be called.

Without the reuse upgrade, 69% of the Python-tool-mode ARC LOW returning
visits opened by reading stored prose guidance, and a stored function was
called on 9% of them; the shortlist listed functions and guidance by
similarity alone, guidance first when it ranked higher, a function as a bare
signature, and a guidance entry without the functions it links. With the
switch on the functions come first, each as a call with its own parameter
names, "already loaded; call directly" where the harness loaded it (the core
surface with ``UNIFY_CORE_BIND_LISTED``; on the JSON tools the switch loads
the listed functions as a FunctionManager read does), and a guidance line
names the functions it links. Nothing is called or forced.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import new_actor as _actor
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.helpers import _handle_project
from unify import db
from unify.actor import library_shortlist as ls
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = 'def double(x: int) -> int:\n    """Double a number."""\n    return x * 2\n'
TASK = "Double the number 5 and report it."

FUNCTIONS = [
    {
        "name": "double",
        "argspec": "(x: int) -> int",
        "docstring": "Double a number.",
        "_similarity": 0.5,
    },
    {
        "name": "scale",
        "argspec": "(x, factor=2, *, round_to=None)",
        "docstring": "Scale a number.",
        "_similarity": 0.4,
    },
]
GUIDANCE = [
    {
        "guidance_id": 7,
        "title": "Doubling",
        "content": "Multiply by two.",
        "_similarity": 0.9,
    },
]


class _Library:
    """A function and a guidance ranking, guidance ranked first."""

    def __init__(self, functions=FUNCTIONS, guidance=GUIDANCE, links=None):
        self._functions, self._guidance = list(functions), list(guidance)
        self.links = links if links is not None else {7: ["double"]}
        self.asked: list = []

    def _shortlist_rows(self, text, k):
        return [dict(r) for r in self._functions]

    def _gated_shortlist_rows(self, threshold, k):
        return [
            {**r, "similar_request": 0.31, "usage_calls": 2} for r in self._functions
        ]

    def _linked_function_names(self, ids):
        self.asked.append(list(ids))
        return {i: self.links.get(i, []) for i in ids}


class _Guidance(_Library):
    def _shortlist_rows(self, text, k):
        return [dict(r) for r in self._guidance]


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_CALLABLE_FIRST", True)


def test_off_by_default_and_the_list_is_as_shipped():
    assert ProductionSettings().UNIFY_SHORTLIST_CALLABLE_FIRST is False
    shipped = ls.shortlist_block(_Library(), _Guidance(), TASK)
    lines = shipped.splitlines()
    assert lines[0] == ls._HEADER
    # Ranked by similarity: the guidance entry first, as shipped.
    assert lines[1].startswith("- guidance 7 `Doubling`")
    assert lines[2] == "- function `double(x: int) -> int`: Double a number."


def test_functions_lead_with_their_calls(on):
    gm = _Guidance()
    block = ls.shortlist_block(_Library(), gm, TASK, surface="core")
    assert block.splitlines() == [
        ls._HEADER_CALLABLE_FIRST,
        '- function `await functions.run("double", x=...)`: Double a number.',
        '- function `await functions.run("scale", x=..., factor=2, round_to=None)`: '
        "Scale a number.",
        "- guidance 7 `Doubling` (for `double`): Multiply by two.",
    ]
    assert gm.asked == [[7]]
    assert ls.shortlisted_names(block) == {
        "functions": ["double", "scale"],
        "guidance": ["7"],
    }


def test_a_loaded_function_is_called_directly(on):
    asked: list = []

    def bind(names):
        asked.append(list(names))
        return {"double": False, "scale": True}

    block = ls.shortlist_block(_Library(), _Guidance(), TASK, bind=bind, surface="core")
    assert asked == [["double", "scale"]]
    assert block.splitlines()[1:3] == [
        "- function `double(x=...)` (already loaded; call directly): Double a number.",
        "- function `await scale(x=..., factor=2, round_to=None)` "
        "(already loaded; call directly): Scale a number.",
    ]


def test_on_the_json_tools_an_unloaded_function_is_an_execute_function_call(on):
    block = ls.shortlist_block(_Library(), _Guidance(), TASK, surface="json")
    assert block.splitlines()[1] == (
        '- function `execute_function("double", {"x": ...})`: Double a number.'
    )


def test_guidance_without_links_reads_as_shipped(on):
    block = ls.shortlist_block(_Library(), _Guidance(links={}), TASK)
    assert block.splitlines()[-1] == "- guidance 7 `Doubling`: Multiply by two."
    # Guidance only: the header and nothing to call.
    only = ls.shortlist_block(_Library(functions=[]), _Guidance(links={}), TASK)
    assert only.splitlines() == [
        ls._HEADER_CALLABLE_FIRST,
        "- guidance 7 `Doubling`: Multiply by two.",
    ]


def test_the_gated_list_leads_with_calls_too(on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    block = ls.shortlist_block(
        _Library(),
        _Guidance(),
        TASK,
        gate=0.2,
        bind=lambda names: {"double": False},
        surface="core",
    )
    lines = block.splitlines()
    assert lines[0].endswith(" Functions first:") and "similar_request" in lines[0]
    assert lines[1] == (
        "- function `double(x=...)` (already loaded; call directly): Double a "
        "number. [similar_request 0.31 · used 2×]"
    )
    assert lines[2].startswith('- function `await functions.run("scale"')


@pytest.mark.parametrize(
    "argspec,call",
    [
        (
            "(grid, marker_color, motif, anchor)",
            "grid=..., marker_color=..., motif=..., anchor=...",
        ),
        (
            "(a: int, b: str = 'x', *rest, k: int, **kw) -> list",
            "a=..., b='x', *rest, k=...",
        ),
        ("(x, /, y=2)", "..., y=2"),
        ("()", ""),
        ("", ""),
    ],
)
def test_call_arguments_use_the_real_parameter_names(argspec, call):
    assert ls.call_arguments(argspec) == call


def test_an_unparsable_signature_still_lists_the_function(on):
    lib = _Library(
        functions=[
            {"name": "odd", "argspec": "(x y)", "docstring": "?", "_similarity": 0.5},
        ],
    )
    block = ls.shortlist_block(lib, _Guidance(links={}), TASK, surface="core")
    assert '- function `await functions.run("odd", ...)`: ?' in block


def test_the_guidance_manager_names_the_functions_an_entry_links(unify_home):
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    db.reset_store()
    try:
        fm = FunctionManager(include_primitives=False)
        fm.add_functions(implementations=[DOUBLE])
        fid = db.query_one("SELECT function_id FROM functions WHERE name = 'double'")[
            "function_id"
        ]
        gm = GuidanceManager()
        gid = gm.add_guidance(title="Doubling", content="x2", function_ids=[fid])[
            "details"
        ]["guidance_id"]
        assert gm._linked_function_names([gid]) == {gid: ["double"]}
        assert gm._linked_function_names([]) == {}
    finally:
        db.reset_store()


# ── a listed function is called directly in the first cell ──────────────────


def _pin_ranking(monkeypatch):
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    def ranked(self, text, k):
        row = db.query_one(
            "SELECT function_id, name, argspec, docstring FROM functions "
            "WHERE name = 'double'",
        )
        return [{**row, "_similarity": 0.9}] if row else []

    monkeypatch.setattr(FunctionManager, "_shortlist_rows", ranked)
    monkeypatch.setattr(GuidanceManager, "_shortlist_rows", lambda self, t, k: [])


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
@pytest.mark.parametrize("surface", ["json", "json-worker", "core"])
async def test_a_listed_function_runs_by_name_in_the_first_cell(
    world,  # noqa: F811
    monkeypatch,
    surface,
):
    (world["state"] / "store.sqlite").unlink()
    db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_CALLABLE_FIRST", True)
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_WORKSPACE_PYTHON",
        "" if surface == "json" else "worker",
    )
    if surface == "core":
        monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
        monkeypatch.setattr(SETTINGS, "UNIFY_CORE_BIND_LISTED", True)
    _pin_ranking(monkeypatch)
    actor = _actor(can_store=False)
    actor.function_manager.add_functions(implementations=[DOUBLE])
    replies = (
        lambda: h.completion(
            calls=[
                ("execute_code", {"thought": "Use it.", "code": "print(double(5))"}),
            ],
        ),
        lambda: h.completion(content="10"),
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(TASK, persist=False)
            result = await asyncio.wait_for(handle.result(), 150)
    finally:
        await actor.close()
    try:
        requests = h.session_requests(provider.requests)
        first = next(
            m["content"] for m in requests[0]["messages"] if m["role"] == "user"
        )
        assert (
            "- function `double(x=...)` (already loaded; call directly): "
            "Double a number." in first
        ), first
        tool = [
            json.dumps(m["content"])
            for m in requests[-1]["messages"]
            if m.get("role") == "tool"
        ]
        assert "10" in tool[0] and "NameError" not in tool[0], tool
        # Loaded as a read loads it: the call is recorded, no search hit.
        usage = db.query_one(
            "SELECT usage_calls, usage_search_hits FROM functions WHERE name = 'double'",
        )
        assert usage["usage_search_hits"] == 0
        if surface != "json-worker":  # recorded there with UNIFY_FUNCTION_HELPERS
            assert usage["usage_calls"] == 1
        assert requests[0]["tool_choice"] == "auto"
        assert result == "10"
    finally:
        db.reset_store()
