"""Symbolic: ``UNIFY_SHORTLIST_GATE`` lists stored functions by request similarity, or nothing.

The retrieval-matching audit (research artifact retrieval-matching-audit-v1,
5 Oct) rebuilt 2,277 task starts with a non-empty library. The embedding
shortlist (``UNIFY_LIBRARY_SHORTLIST`` alone) has no threshold: it listed
4.5 entries on ARC queries with no correct entry, 4.3 on ScienceWorld and 5.0
on AppWorld dev with a frozen library never used there, a list on 100% of
them, and no cosine threshold separated useful entries from useless ones.
Comparing the new request with the requests an entry was stored under
(``similar_request``) can be thresholded: at 0.175, with stream-wide weights
and whole identifiers, AppWorld repeat visits found last time's entry 88% of
the time while ARC got a list on 5% of no-correct-entry queries and AppWorld
first visits on 1%. With the gate the shortlist lists only stored functions
whose ``similar_request`` passes the threshold, computes no embedding,
leaves the activation ranking out and shows the score and call count as
evidence. Requests are captured at unillm's transport, so nothing leaves the
process.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.function_manager import task_origin
from unify.settings import ProductionSettings, SETTINGS

GATE = "similar_request:0.175"
PREAMBLE = (
    "You are working through a stream of table puzzles. Each instance gives a "
    "puzzle id and an input table; puzzles recur with fresh tables. Reply with "
    "the output table as rows of integers on the last line of your reply."
)


def _visit(puzzle: str, table: list[list[int]]) -> str:
    rows = "\n".join(" ".join(str(v) for v in row) for row in table)
    size = f"{len(table)}x{len(table[0])}"
    return f"{PREAMBLE}\n\nNew instance. Puzzle id: {puzzle}\nInput ({size}):\n{rows}\n"


FIRST = _visit("p-3d61a", [[1, 0, 2], [0, 1, 0]])
# The same puzzle again: only the id's neighbours, the size and the numbers differ.
AGAIN = _visit("p-3d61a", [[5, 5], [0, 5], [5, 0]])
OTHERS = [
    ("triple", 3, _visit("p-55e10", [[3, 3], [3, 0]])),
    ("quadruple", 4, _visit("p-71c2f", [[2, 0], [0, 2], [2, 2]])),
    ("quintuple", 5, _visit("p-a04d9", [[7, 1, 7]])),
]
UNRELATED = "Plan a week of vegetarian dinners for two and write the shopping list."


def _source(name: str, factor: int, doc: str = "") -> str:
    doc = doc or f"Multiply a number by {factor}."
    return f'def {name}(x: int) -> int:\n    """{doc}"""\n    return {factor} * x\n'


@pytest.fixture
def switches(monkeypatch):
    def set_(*, gate=GATE, shortlist=True, origin=True, builtin=False):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", shortlist)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SNAPSHOT", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", builtin)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")

    return set_


@pytest.fixture
def embed_calls(monkeypatch):
    """Every embed() call; fake unit vectors, so nothing reaches a model."""
    calls: list[list[str]] = []

    def spy(texts):
        calls.append(list(texts))
        vectors = np.ones((len(texts), 4), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    import unify.common.embeddings as embeddings
    import unify.common.semantic_search as semantic_search

    monkeypatch.setattr(embeddings, "embed", spy)
    monkeypatch.setattr(semantic_search, "embed", spy)
    return calls


def _in_task(request, fn):
    token = task_origin.enter(request)
    try:
        return fn()
    finally:
        task_origin.leave(token)


def _seed(actor, *, with_origin=True):
    fm = actor.function_manager
    store = lambda request, source: (  # noqa: E731
        _in_task(request, lambda: fm.add_functions(implementations=source))
        if with_origin
        else fm.add_functions(implementations=source)
    )
    store(FIRST, _source("double", 2, "Double a number.\nMore lines."))
    for name, factor, request in OTHERS:
        store(request, _source(name, factor))
    actor.guidance_manager.add_guidance(
        title="Doubling",
        content="Double the number with 2 * x.",
    )


async def _act(task, *, seed=None, embed_calls=None, outer_task=None):
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    token = task_origin.enter(outer_task) if outer_task else None
    try:
        with h.scripted([lambda: h.completion(content="done")] * 8) as provider:
            before = len(embed_calls) if embed_calls is not None else 0
            handle = await actor.act(task, persist=False)
            at_start = (embed_calls or [])[before:]
            await asyncio.wait_for(handle.result(), 60)
    finally:
        task_origin.leave(token)
        await actor.close()
    return h.session_requests(provider.requests), at_start


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


def _block(text: str) -> str | None:
    if ls._GATED_HEADER not in text:
        return None
    return text[text.index(ls._GATED_HEADER) :].split("\n\n", 1)[0]


LINE = re.compile(
    r"^- function `double\(x: int\) -> int`: Double a number\. "
    r"\[similar_request (\d\.\d\d) · used 0×\]$",
)


# ── through the actor ────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_function_stored_for_the_same_puzzle_is_listed_without_embedding(
    switches,
    embed_calls,
):
    switches()
    requests, at_start = await _act(AGAIN, seed=_seed, embed_calls=embed_calls)
    first = _first_user(requests[0])
    block = _block(first)
    assert block is not None, first
    lines = block.splitlines()[1:]
    assert len(lines) == 1, lines
    match = LINE.match(lines[0])
    assert match, lines[0]
    assert 0.175 <= float(match.group(1)) < 1
    # After the snapshot line, before the request; no guidance; no embedding.
    assert first.startswith(
        "Library at task start: 4 stored functions, 1 guidance entry.\n\n"
        + ls._GATED_HEADER,
    )
    assert first.endswith(f"\n\n---\n\n{AGAIN}")
    assert "guidance" not in block.split("\n", 1)[1]
    assert at_start == []
    assert requests[0]["tool_choice"] == "auto"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_same_request_scores_one(switches, embed_calls):
    switches()
    requests, _ = await _act(FIRST, seed=_seed, embed_calls=embed_calls)
    # Listed first, whatever else passes the gate.
    line = _block(_first_user(requests[0])).splitlines()[1]
    assert LINE.match(line).group(1) == "1.00"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_unrelated_request_gets_no_list(switches, embed_calls):
    switches()
    requests, at_start = await _act(UNRELATED, seed=_seed, embed_calls=embed_calls)
    assert _first_user(requests[0]) == (
        "Library at task start: 4 stored functions, 1 guidance entry."
        f"\n\n---\n\n{UNRELATED}"
    )
    assert at_start == []


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_function_without_an_origin_record_is_never_listed(
    switches,
    embed_calls,
):
    switches()
    requests, _ = await _act(
        FIRST,
        seed=lambda actor: _seed(actor, with_origin=False),
        embed_calls=embed_calls,
    )
    assert _block(_first_user(requests[0])) is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_built_in_guidance_is_never_listed(switches, embed_calls):
    switches(builtin=True)
    requests, _ = await _act(AGAIN, seed=_seed, embed_calls=embed_calls)
    block = _block(_first_user(requests[0]))
    assert block is not None
    assert all(l.startswith("- function ") for l in block.splitlines()[1:])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_sub_agents_task_gets_no_list(switches, embed_calls):
    switches()
    requests, _ = await _act(
        AGAIN,
        seed=_seed,
        embed_calls=embed_calls,
        outer_task="the caller's task",
    )
    assert _block(_first_user(requests[0])) is None


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_without_the_gate_the_shortlist_embeds_as_shipped(switches, embed_calls):
    # The spy sees the embedding shortlist's calls, so its silence above counts.
    switches(gate="")
    requests, at_start = await _act(AGAIN, seed=_seed, embed_calls=embed_calls)
    assert at_start
    assert ls._HEADER in _first_user(requests[0])
    assert _block(_first_user(requests[0])) is None


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "kwargs, names",
    [
        ({"shortlist": False}, "UNIFY_LIBRARY_SHORTLIST"),
        ({"origin": False}, "UNIFY_TASK_ORIGIN"),
    ],
)
async def test_an_actor_refuses_a_gate_nothing_could_pass(switches, kwargs, names):
    switches(**kwargs)
    actor = caa.CodeActActor()
    try:
        with pytest.raises(ValueError, match=names):
            await actor.act(AGAIN, persist=False)
    finally:
        await actor.close()


# ── ranking ──────────────────────────────────────────────────────────────


class _Scores:
    def __init__(self, scores):
        self.scores = scores

    def score(self, row):
        return self.scores.get(row["name"])


def _row(name, function_id, calls=0):
    return {"name": name, "function_id": function_id, "usage_calls": calls}


def test_rows_rank_by_score_then_calls_then_newest_and_at_most_five():
    rows = [
        _row("low", 1),
        _row("old_unused", 2),
        _row("new_unused", 3),
        _row("used", 4, calls=7),
        _row("best", 5),
        _row("no_origin", 6),
        _row("f7", 7),
        _row("f8", 8),
        _row("f9", 9),
    ]
    marker = _Scores(
        {
            "low": 0.1,
            "old_unused": 0.5,
            "new_unused": 0.5,
            "used": 0.5,
            "best": 0.9,
            "f7": 0.3,
            "f8": 0.2,
            "f9": 0.175,
        },
    )
    kept = ls.gate_rows(rows, marker, 0.175)
    assert [r["name"] for r in kept] == [
        "best",
        "used",
        "new_unused",
        "old_unused",
        "f7",
    ]
    assert kept[0]["similar_request"] == 0.9
    assert "similar_request" not in rows[4]  # copies, the rows are untouched
    assert [r["name"] for r in ls.gate_rows(rows, marker, 0.175, k=9)][-2:] == [
        "f8",
        "f9",
    ]
    assert ls.gate_rows(rows, marker, 0.95) == []


def test_the_line_shows_the_score_and_the_call_count():
    row = {
        "name": "double",
        "argspec": "(x: int) -> int",
        "docstring": "Double a number.\nMore.",
        "similar_request": 0.31,
        "usage_calls": 4,
    }
    assert ls._gated_function_line(row) == (
        "- function `double(x: int) -> int`: Double a number. "
        "[similar_request 0.31 · used 4×]"
    )


def test_the_gated_rows_use_neither_embeddings_nor_the_activation_ranking(
    switches,
    monkeypatch,
):
    from unify.function_manager import function_manager as fm_module
    from unify.function_manager.function_manager import FunctionManager

    switches()

    def refuse(*args, **kwargs):
        raise AssertionError("not on the gated path")

    monkeypatch.setattr(fm_module, "rank_by_similarity", refuse)
    monkeypatch.setattr(FunctionManager, "_activation_rank", refuse)
    fm = FunctionManager()
    _in_task(FIRST, lambda: fm.add_functions(implementations=_source("double", 2)))
    rows = _in_task(FIRST, lambda: fm._gated_shortlist_rows(0.175, 5))
    assert [(r["name"], r["similar_request"]) for r in rows] == [("double", 1.0)]
    assert "metadata" not in rows[0]


def test_the_header_asks_nothing_and_names_no_benchmark():
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    words = re.findall(r"[a-z]+", ls._GATED_HEADER.lower())
    for word in ("must", "always", "first", "before", "try", "should"):
        assert word not in words
    assert not BENCHMARK_WORDS.search(ls._GATED_HEADER)


# ── the setting ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, stored, threshold",
    [
        ("", "", None),
        ("similar_request:0.175", "similar_request:0.175", 0.175),
        (" Similar_Request:.2 ", "similar_request:0.2", 0.2),
        ("similar_request:1", "similar_request:1", 1.0),
    ],
)
def test_the_setting_parses(value, stored, threshold):
    settings = ProductionSettings(UNIFY_SHORTLIST_GATE=value)
    assert settings.UNIFY_SHORTLIST_GATE == stored
    assert settings.shortlist_gate_threshold() == threshold


@pytest.mark.parametrize(
    "value",
    [
        "0.2",
        "similar_request",
        "similar_request:0",
        "similar_request:1.5",
        "similar_request:nan",
        "similar_request:x",
        "cosine:0.7",
    ],
)
def test_the_setting_refuses_other_values(value):
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_SHORTLIST_GATE=value)


def test_the_setting_defaults_off():
    assert ProductionSettings().UNIFY_SHORTLIST_GATE == ""
