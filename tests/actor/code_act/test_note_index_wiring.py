"""Symbolic: ``UNIFY_NOTE_INDEX`` reaches a top-level task's first message, and the functions it attaches are bound, never called.

A note is written, with the function it links, while handling one request;
a later, similar request finds it by that request. The embedder is a concept
fake through the real cache; the provider is scripted, so nothing leaves the
process. The core-surface case runs cells in the real sandboxed worker and
is skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify import db
from unify.actor import code_act_actor as caa
from unify.actor import note_index as ni
from unify.function_manager import task_origin
from unify.settings import SETTINGS

WRITTEN_FOR = "Total my card payments for May and report the sum."
SIMILAR = "Total my card payments for June and report the sum."
UNRELATED = "Plan a week of vegetarian dinners for two."
EXPLODE = (
    "def explode_payments(x):\n"
    '    """Total card payments."""\n'
    "    raise RuntimeError('the harness called a stored function')\n"
)
DOUBLE = 'def double(x: int) -> int:\n    """Double a number."""\n    return x * 2\n'

CONCEPTS = [
    {"card", "payments", "total", "sum"},
    {"vegetarian", "dinners", "week"},
    {"double", "number"},
]


def _vector(text: str) -> np.ndarray:
    words = re.findall(r"[a-z]+", text.lower())
    v = np.array(
        [0.05] + [sum(w in group for w in words) for group in CONCEPTS],
        dtype=np.float32,
    )
    return v / np.linalg.norm(v)


@pytest.fixture
def embed_calls(monkeypatch, tmp_path):
    """Every embed() call, answered by the concept fake through the real cache."""
    from unify.common import embeddings

    calls: list[list[str]] = []
    fake = embeddings.Embedder(
        "note-index-test-concepts",
        lambda texts: np.stack([_vector(t) for t in texts]),
    )
    monkeypatch.setattr(embeddings, "embedder", lambda: fake)
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))
    real = embeddings.embed

    def spy(texts):
        calls.append(list(texts))
        return real(texts)

    monkeypatch.setattr(embeddings, "embed", spy)
    return calls


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_NOTE_INDEX", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_ORIGIN", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _seed(actor, source, name, request=WRITTEN_FOR):
    """Under *request*: the function and a note that links it."""

    def write():
        actor.function_manager.add_functions(implementations=source)
        row = db.query_one("SELECT function_id FROM functions WHERE name = ?", (name,))
        actor.guidance_manager.add_guidance(
            title="Card payments",
            content="Amounts are negative; sum their absolute values.",
            function_ids=[int(row["function_id"])],
        )

    _in_task(request, write)


def _usage(name: str) -> int:
    return int(
        db.query_one("SELECT usage_calls FROM functions WHERE name = ?", (name,))[
            "usage_calls"
        ]
        or 0,
    )


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


async def _act(actor, task, replies):
    try:
        with h.scripted(list(replies)) as provider:
            handle = await actor.act(task, persist=False)
            result = await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    return result, h.session_requests(provider.requests)


def _done():
    return lambda: h.completion(content="done")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_on_the_section_is_in_the_first_message_and_its_function_is_not_called(
    on,
    embed_calls,
):
    actor = caa.CodeActActor()
    _seed(actor, EXPLODE, "explode_payments")
    result, requests = await _act(actor, SIMILAR, [_done()] * 8)
    assert result == "done"
    first = _first_user(requests[0])
    assert ni.HEADER.strip() in first
    section = first[first.index(ni.HEADER.strip()) :]
    assert re.search(r"^### Note \d+: Card payments$", section, re.M)
    assert "Amounts are negative; sum their absolute values." in section
    assert "- `explode_payments(x)` (loaded): Total card payments." in section
    # Bound as a read binds it (no search hit), and never run by the harness.
    assert _usage("explode_payments") == 0
    # One embedding call at the task start, over the request and the origin.
    at_start = [texts for texts in embed_calls if SIMILAR in texts]
    assert len(at_start) == 1 and WRITTEN_FOR in at_start[0]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
@pytest.mark.parametrize("solved", [False, True])
async def test_on_a_note_whose_writer_the_checker_did_not_accept_is_labelled(
    on,
    embed_calls,
    monkeypatch,
    solved,
):
    # The checker's outcome is kept in the request log (a listing switch keeps it).
    monkeypatch.setattr(SETTINGS, "UNIFY_ORIGIN_PROVENANCE", True)
    actor = caa.CodeActActor()
    _seed(actor, EXPLODE, "explode_payments")
    assert _in_task(WRITTEN_FOR, lambda: task_origin.record_outcome(solved))
    _result, requests = await _act(actor, SIMILAR, [_done()] * 8)
    first = _first_user(requests[0])
    heading = re.search(r"^### Note \d+: Card payments.*$", first, re.M).group(0)
    assert heading.endswith(ni.FAILED_WRITER) is (not solved)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_on_a_sub_agent_gets_no_section(on, embed_calls):
    actor = caa.CodeActActor()
    _seed(actor, EXPLODE, "explode_payments")
    token = task_origin.enter(UNRELATED)  # the caller's task
    try:
        _result, requests = await _act(actor, SIMILAR, [_done()] * 8)
    finally:
        task_origin.leave(token)
    assert ni.HEADER.strip() not in json.dumps(requests[0]["messages"])
    assert not [texts for texts in embed_calls if SIMILAR in texts]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_off_no_section_and_no_embedding(embed_calls):
    assert SETTINGS.UNIFY_NOTE_INDEX is False
    _result, requests = await _act(caa.CodeActActor(), SIMILAR, [_done()] * 8)
    assert ni.HEADER.strip() not in json.dumps(requests[0]["messages"])
    assert not [texts for texts in embed_calls if SIMILAR in texts]


def _cell(code: str):
    return lambda: h.completion(
        calls=[("execute_code", {"thought": "Next step.", "code": code})],
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_core_an_attached_function_is_callable_in_the_first_cell(
    core_world,
    on,
    embed_calls,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CORE_BIND_LISTED", False)
    actor = new_actor(can_store=False)
    _seed(actor, DOUBLE, "double", request="Double the number 4.")
    result, requests = await _act(
        actor,
        "Double the number 5.",
        (_cell("print(double(5))"), _done()),
    )
    assert result == "done"
    first = _first_user(requests[0])
    assert "- `double(x: int) -> int` (loaded): Double a number." in first
    replies = [
        json.dumps(m["content"])
        for m in requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert "10" in replies[-1] and "NameError" not in replies[-1], replies
    # The model's call is the only one.
    assert _usage("double") == 1
    assert requests[0]["tool_choice"] == "auto"
