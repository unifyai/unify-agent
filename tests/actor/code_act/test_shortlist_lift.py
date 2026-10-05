"""Symbolic: ``UNIFY_SHORTLIST_LIFT`` ranks the shortlist by lift over recent requests.

On the 5 Oct Continual-ARC paper-protocol run (lean-all, LOW), the ungated
shortlist's first entry was the same generic guidance entry in every list:
embedding similarity to the full request is mostly similarity to the
stream's shared rules text, and prose resembles that text more than any
function does. With the switch an entry scores its similarity to this
request less its mean similarity to the last k top-level requests, from
cached vectors only, so an entry close to every request gives way to one
close to this request; a floor drops entries that match this request
clearly less than recent ones. With fewer than k earlier requests the list
is as shipped. Embeddings here come from a fake embedder over a few
concepts, cached as the real ones are; requests are captured at unillm's
transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.actor import shortlist_lift
from unify.common import embeddings
from unify.function_manager import task_origin
from unify.settings import ProductionSettings, SETTINGS

PREAMBLE = (
    "You are working through a stream of table puzzles. Each instance gives a "
    "puzzle id and an input table; puzzles recur with fresh tables. Reply with "
    "the output table rows on the last line."
)


def _visit(puzzle: str, hint: str) -> str:
    return f"{PREAMBLE}\n\nNew instance. Puzzle id: {puzzle}\nThe table: {hint}\n"


EARLIER = [
    _visit("p-55e10", "mirror it"),
    _visit("p-71c2f", "sort its rows"),
    _visit("p-a04d9", "flip it"),
]
ROTATE = _visit("p-3d61a", "rotate it a quarter turn")
ROTATE_AGAIN = _visit("p-3d61a", "rotate it a quarter turn, again")
UNRELATED = "Plan a week of vegetarian dinners for two and write the shopping list."

GENERIC_TITLE = "Table puzzles"
GENERIC = (
    "Read each puzzle table and its rows, then reply with the output table "
    "rows for the instance."
)
ROTATE_DOC = "Rotate a table a quarter turn."

CONCEPTS = [
    {"puzzle", "puzzles", "table", "tables", "rows", "instance", "reply", "output"},
    {"rotate", "rotation", "turn", "quarter"},
    {"mirror", "flip", "reflect"},
    {"sort", "order"},
    {"vegetarian", "dinners", "shopping", "week"},
]
MODEL = "memsurf-test-concepts"


def _vector(text: str) -> np.ndarray:
    words = re.findall(r"[a-z]+", text.lower())
    v = np.array(
        [0.05] + [sum(w in group for w in words) for group in CONCEPTS],
        dtype=np.float32,
    )
    return v / np.linalg.norm(v)


@pytest.fixture
def computed(monkeypatch, tmp_path):
    """The texts the fake embedder computed (each a model call), through the real cache."""
    calls: list[list[str]] = []

    def compute(texts):
        calls.append(list(texts))
        return np.stack([_vector(t) for t in texts])

    fake = embeddings.Embedder(MODEL, compute)
    monkeypatch.setattr(embeddings, "embedder", lambda: fake)
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))
    return calls


@pytest.fixture
def switches(monkeypatch):
    def set_(*, lift="", gate="", origin=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_LIFT", lift)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _rotate_source() -> str:
    return (
        "def rotate_table(table):\n"
        f'    """{ROTATE_DOC}"""\n'
        "    return [list(row) for row in zip(*table[::-1])]\n"
    )


def _seed(actor, *, under=None):
    """A generic guidance entry and a specific function, recorded under *under* when given.

    With request records on, the stream's earlier requests are logged first.
    """
    for request in EARLIER:
        _in_task(request, lambda: None)

    def write():
        actor.function_manager.add_functions(implementations=_rotate_source())
        return int(
            actor.guidance_manager.add_guidance(title=GENERIC_TITLE, content=GENERIC)[
                "details"
            ]["guidance_id"],
        )

    return _in_task(under, write) if under else write()


async def _act(task, *, seed=None):
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    try:
        with h.scripted([lambda: h.completion(content="done")] * 8) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests)


def _first_user(requests) -> str:
    return next(m["content"] for m in requests[0]["messages"] if m["role"] == "user")


def _block(text: str) -> str | None:
    for header in (ls._HEADER, ls._LIFT_HEADER):
        if header in text:
            return text[text.index(header) :].split("\n\n", 1)[0]
    return None


def _entries(block: str | None) -> list[str]:
    return [ln for ln in (block or "").splitlines() if ln.startswith("- ")]


async def _stream(tasks, *, seed=None):
    """Each task as its own top-level act() on one store; the first user message of each."""
    out = []
    for i, task in enumerate(tasks):
        out.append(_first_user(await _act(task, seed=seed if i == 0 else None)))
    return out


# ── UNIFY_SHORTLIST_LIFT ─────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lift_puts_the_entry_close_to_this_request_before_the_generic_one(
    switches,
    computed,
):
    switches()
    shipped = await _stream([*EARLIER, ROTATE], seed=_seed)
    raw = _entries(_block(shipped[-1]))
    assert raw[0].startswith("- guidance ") and GENERIC_TITLE in raw[0]
    assert raw[1].startswith("- function `rotate_table(table)`")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lift_ranks_by_lift_once_k_earlier_requests_are_cached(
    switches,
    computed,
):
    switches(lift="recent:3")
    firsts = await _stream([*EARLIER, ROTATE], seed=_seed)
    # The first three task starts have fewer than 3 earlier requests: as shipped.
    for first in firsts[:3]:
        assert ls._HEADER in first and ls._LIFT_HEADER not in first
    block = _block(firsts[-1])
    assert block.startswith(ls._LIFT_HEADER)
    lines = _entries(block)
    assert lines[0].startswith("- function `rotate_table(table)`: " + ROTATE_DOC)
    assert any(GENERIC_TITLE in line for line in lines[1:])


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lift_embeds_nothing_the_shipped_ranking_does_not(
    switches,
    computed,
    monkeypatch,
    tmp_path,
):
    from unify import db

    tasks = [*EARLIER, ROTATE, ROTATE_AGAIN]
    switches()
    await _stream(tasks, seed=_seed)
    shipped = [text for call in computed for text in call]
    # The same stream on an empty store, request log and vector cache, lift on.
    computed.clear()
    db.clear()
    task_origin.request_log_path().unlink(missing_ok=True)
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "lift-embeddings.sqlite"))
    switches(lift="recent:2")
    firsts = await _stream(tasks, seed=_seed)
    lifted = [text for call in computed for text in call]
    assert sum(ls._LIFT_HEADER in first for first in firsts) == 3
    assert sorted(lifted) == sorted(shipped)


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lift_floor_drops_entries_that_stand_out_for_no_request(
    switches,
    computed,
):
    # The rotate function stands out for ROTATE only; for a request like the
    # earlier ones nothing does, and a floor above 0 then lists nothing.
    switches(lift="recent:3:0.05")
    firsts = await _stream(
        [*EARLIER, ROTATE, _visit("p-9e2b7", "mirror it")],
        seed=_seed,
    )
    rotate = _entries(_block(firsts[3]))
    assert rotate == [rotate[0]] and rotate[0].startswith("- function `rotate_table")
    assert _block(firsts[4]) is None


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_lift_leaves_the_current_request_out_of_its_own_baseline(
    switches,
    computed,
):
    switches(lift="recent:3:0.05")
    # ROTATE was seen before: its own earlier start must not cancel its lift.
    firsts = await _stream([*EARLIER, ROTATE, ROTATE], seed=_seed)
    assert _entries(_block(firsts[-1]))[0].startswith("- function `rotate_table")
    kept = shortlist_lift.recent_hashes(MODEL, "")
    assert len(kept) == 4  # a repeat moves to the end; it is not kept twice
    assert kept[0] == embeddings.text_hash(ls.request_text(ROTATE))


def test_lift_is_not_logged_inside_a_task(switches):
    switches(lift="recent")
    outer = shortlist_lift.enter()
    try:
        assert outer is not None
        assert shortlist_lift.enter() is None  # a sub-agent's task start
    finally:
        shortlist_lift.leave(outer)
    switches()
    assert shortlist_lift.enter() is None  # off: nothing is marked


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_lift_refuses_to_start_without_an_embedding_shortlist(
    switches,
    computed,
    monkeypatch,
):
    switches(lift="recent")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", False)
    with pytest.raises(ValueError, match="UNIFY_LIBRARY_SHORTLIST"):
        await _act(ROTATE)
    switches(lift="recent", gate="similar_request:0.2", origin=True)
    with pytest.raises(ValueError, match="UNIFY_SHORTLIST_GATE"):
        await _act(ROTATE)


def test_lift_setting_parses_and_refuses():
    assert ProductionSettings(UNIFY_SHORTLIST_LIFT="recent").shortlist_lift() == (
        4,
        None,
    )
    assert ProductionSettings(
        UNIFY_SHORTLIST_LIFT="Recent:8:-0.08",
    ).shortlist_lift() == (8, -0.08)
    assert ProductionSettings(UNIFY_SHORTLIST_LIFT="").shortlist_lift() is None
    for bad in ("recent:0", "lift", "recent:x", "recent:4:nan"):
        with pytest.raises(ValueError, match="UNIFY_SHORTLIST_LIFT"):
            ProductionSettings(UNIFY_SHORTLIST_LIFT=bad)


def test_lift_floors_per_embedder():
    assert shortlist_lift.floor_for(embeddings.OPENROUTER.model, None) == -0.08
    assert shortlist_lift.floor_for(embeddings.OPENROUTER.model, 0.02) == 0.02
    assert shortlist_lift.floor_for(embeddings.LOCAL.model, None) == -0.03
    assert shortlist_lift.floor_for("unknown-model", None) == float("-inf")
