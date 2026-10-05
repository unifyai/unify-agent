"""Symbolic: under the core surface a stored function is usable as listed.

The AppWorld core-surface screen (e836a5aee, HIGH r1) showed the library
shortlist in 11 sessions and none of them called a stored function: the
model rewrote listed and found functions instead, and no call failed. Three
things differ from the JSON surface, which made 0.38-0.71 stored-function
calls per task: a listed function is not callable until something reads it
and the list does not say how to call it; the prompt's ``functions`` line has
no example call; and a guidance read shows the functions it links only as
bare ``function_ids``. Two off-by-default switches address the first two;
they inform, and force nothing:

* ``UNIFY_CORE_BIND_LISTED``: the shortlist's functions are bound at task
  start exactly as ``functions.get`` binds one (no search hit counted), and
  either shortlist header says how to call one.
* ``UNIFY_CORE_CALL_EXAMPLE``: the ``functions`` index line shows one call.

The model is a scripted transport (tests/cache_discipline_helpers.py);
cells run in the real sandboxed worker, skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor as _actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify import db
from unify.actor import core_surface
from unify.actor import library_shortlist as ls
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = 'def double(x: int) -> int:\n    """Double a number."""\n    return x * 2\n'
TWICE = (
    'async def double_twice(x: int) -> int:\n    """Double a number twice."""\n'
    "    return x * 4\n"
)
TASK = "Double the number 5 and report it."
EXAMPLE = 'Example: `total = await functions.run("sum_invoice_lines", invoice_id=7)`'


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code})],
    )


def _tool_replies(request: dict) -> list[str]:
    return [
        json.dumps(m["content"]) for m in request["messages"] if m.get("role") == "tool"
    ]


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


async def _act(actor, replies, request=TASK):
    with h.scripted(replies) as provider:
        handle = await actor.act(request, persist=False)
        result = await asyncio.wait_for(handle.result(), 120)
    return result, h.session_requests(provider.requests)


def _pin_ranking(monkeypatch):
    """The shortlist ranks ``double`` first, without embeddings.

    The ranking (embedding similarity) is tested in test_library_shortlist;
    these tests are about what the list does once it names a function, so
    the ranking is pinned to the stored row and needs no embedding model.
    """
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


def _usage(name: str) -> dict:
    return db.query_one(
        "SELECT usage_calls, usage_search_hits FROM functions WHERE name = ?",
        (name,),
    )


# ── the settings ─────────────────────────────────────────────────────────────


def test_the_switches_are_off_by_default():
    defaults = ProductionSettings()
    for name in (
        "UNIFY_CORE_BIND_LISTED",
        "UNIFY_CORE_CALL_EXAMPLE",
    ):
        assert getattr(defaults, name) is False, name
        assert getattr(ProductionSettings(**{name: "1"}), name) is True, name


# ── the shortlist's text, with and without a binder ─────────────────────────


class _Library:
    """The two shortlist rankings of a function manager, and a guidance one."""

    def __init__(self, functions, guidance=()):
        self._functions = list(functions)
        self._guidance = list(guidance)

    def _shortlist_rows(self, text, k):
        return [dict(r) for r in (self._functions or self._guidance)]

    def _gated_shortlist_rows(self, threshold, k):
        return [
            {**r, "similar_request": 0.31, "usage_calls": 2} for r in self._functions
        ]


ROWS = [
    {
        "name": "double",
        "argspec": "(x: int) -> int",
        "docstring": "Double a number.",
        "_similarity": 0.9,
    },
    {
        "name": "double_twice",
        "argspec": "(x: int) -> int",
        "docstring": "Double a number twice.",
        "_similarity": 0.8,
    },
]
GUIDANCE_ROWS = [
    {
        "guidance_id": 1,
        "title": "Doubling",
        "content": "Multiply by two.",
        "_similarity": 0.7,
    },
]


@pytest.mark.parametrize("gate", [None, 0.2])
def test_a_binder_binds_the_listed_functions_and_the_header_says_how_to_call(gate):
    fm = _Library(ROWS)
    gm = _Library([], GUIDANCE_ROWS)
    shipped = ls.shortlist_block(fm, gm, TASK, gate=gate)
    asked: list = []

    def bind(names):
        asked.append(list(names))
        return {"double": False, "double_twice": True}

    on = ls.shortlist_block(fm, gm, TASK, gate=gate, bind=bind)
    header = ls._GATED_HEADER_CALL if gate else ls._HEADER_CALL
    shipped_header = ls._GATED_HEADER if gate else ls._HEADER
    assert shipped.splitlines()[0] == shipped_header
    assert on.splitlines()[0] == header and ls.CALL_FORM in header
    assert '`await functions.run("name", arg=...)`' in header
    # Only the listed functions are bound, once; guidance is never passed.
    assert asked == [["double", "double_twice"]]
    # The lines keep their signatures; an async def is marked after it.
    shipped_lines, on_lines = shipped.splitlines()[1:], on.splitlines()[1:]
    assert on_lines[0] == shipped_lines[0]
    assert on_lines[1] == shipped_lines[1].replace(
        "`double_twice(x: int) -> int`",
        "`double_twice(x: int) -> int` (async)",
    )
    assert on_lines[1] != shipped_lines[1]
    assert ls.shortlisted_names(on)["functions"] == ["double", "double_twice"]


def test_without_anything_bound_the_list_is_as_shipped():
    fm = _Library(ROWS)
    gm = _Library([], GUIDANCE_ROWS)
    shipped = ls.shortlist_block(fm, gm, TASK)

    def fails(names):
        raise RuntimeError("store is gone")

    asked: list = []
    assert ls.shortlist_block(fm, gm, TASK, bind=fails) == shipped
    assert ls.shortlist_block(fm, gm, TASK, bind=lambda names: {}) == shipped
    # A list of guidance only binds nothing and says nothing about calls.
    guidance_only = ls.shortlist_block(_Library([]), gm, TASK)
    assert guidance_only.splitlines()[0] == ls._HEADER
    assert (
        ls.shortlist_block(
            _Library([]),
            gm,
            TASK,
            bind=lambda names: asked.append(names) or {},
        )
        == guidance_only
    )
    assert asked == []


# ── the shortlist's functions are callable in the first cell ────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
@pytest.mark.parametrize("bind", [False, True])
async def test_a_listed_function_is_callable_in_the_first_cell_only_when_bound(
    core_world,
    monkeypatch,
    bind,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_BIND_LISTED", bind)
    _pin_ranking(monkeypatch)
    actor = _actor(can_store=False)
    actor.function_manager.add_functions(implementations=[DOUBLE])
    replies = (_cell("print(double(5))"), lambda: h.completion(content="done"))
    try:
        result, requests = await _act(actor, replies)
    finally:
        await actor.close()
    assert result == "done"
    first = _first_user(requests[0])
    assert "- function `double(x: int) -> int`: Double a number." in first
    (reply,) = _tool_replies(requests[-1])
    if bind:
        assert ls._HEADER_CALL in first
        assert "10" in reply and "NameError" not in reply, reply
        # Bound as a read binds, and the call is recorded; no search hit.
        assert _usage("double") == {"usage_calls": 1, "usage_search_hits": 0}
    else:
        assert ls._HEADER in first and ls.CALL_FORM not in first
        assert "NameError" in reply, reply
        assert _usage("double")["usage_search_hits"] == 0
    # Nothing is forced: the first request's tools and choice are as shipped.
    assert requests[0]["tool_choice"] == "auto"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_binding_changes_only_the_list_in_the_first_request(
    core_world,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    _pin_ranking(monkeypatch)
    firsts = {}
    for bind in (False, True):
        monkeypatch.setattr(SETTINGS, "UNIFY_CORE_BIND_LISTED", bind)
        db.clear()
        actor = _actor(can_store=False)
        actor.function_manager.add_functions(implementations=[DOUBLE])
        try:
            _r, requests = await _act(actor, (lambda: h.completion(content="done"),))
        finally:
            await actor.close()
        firsts[bind] = requests[0]
    off, on = firsts[False], firsts[True]
    assert on["tools"] == off["tools"]
    assert on["messages"][0] == off["messages"][0]
    assert ls._HEADER in _first_user(off) and ls._HEADER_CALL in _first_user(on)
    assert _first_user(on) == _first_user(off).replace(ls._HEADER, ls._HEADER_CALL)


# ── the index line's example ────────────────────────────────────────────────


def test_the_functions_index_line_shows_a_call_only_with_the_switch():
    shipped = core_surface.PromptSurface().index()
    on = core_surface.PromptSurface(call_example=True).index()
    assert EXAMPLE not in shipped
    flat = " ".join(on.split())
    assert EXAMPLE in flat and "`sum_invoice_lines(invoice_id=7)`" in flat
    anchor = "callable by name in later cells."
    assert flat == " ".join(shipped.split()).replace(
        anchor,
        anchor
        + " "
        + EXAMPLE
        + ", or once found, `sum_invoice_lines(invoice_id=7)`; a stored "
        "function that does a step saves rewriting it.",
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
@_handle_project
async def test_the_core_prompt_carries_the_example_with_the_switch(
    core_world,
    monkeypatch,
):
    systems = {}
    for on in (False, True):
        monkeypatch.setattr(SETTINGS, "UNIFY_CORE_CALL_EXAMPLE", on)
        actor = _actor(can_store=False)
        try:
            _r, requests = await _act(actor, (lambda: h.completion(content="done"),))
        finally:
            await actor.close()
        systems[on] = requests[0]["messages"][0]["content"]
    assert EXAMPLE not in " ".join(systems[False].split())
    assert EXAMPLE in " ".join(systems[True].split())
