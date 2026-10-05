"""Symbolic: ``UNIFY_EVIDENCE_LIST``, the shortlist as a list of evidence (retrieval Option A).

On the 5 Oct Continual-ARC paper-protocol run the shipped shortlist, ranked
by the embedding of the whole request, put one generic note written after a
failed visit first in every list, gave entries from the same task no higher
score than others, always listed five entries, and never said why one was
there. The evidence list answers "seen before?" by keys (same request, rare
shared identifier, similar wording; no embedding), lists at most k
"possibly related" entries by meaning under a header that claims nothing,
compares the request whole (no masking, no score against recent requests),
shows each card's record, and is silent when nothing qualifies.
Functions and notes are linked many to many, so a card is a function with
its notes or a note with the functions it guides. Embeddings come from a
concept fake; requests are captured at unillm's transport, so nothing leaves
the process.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import evidence_list as ev
from unify.function_manager import entry_record, task_origin
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
    _visit("p-b81e3", "sort its rows by size"),
    _visit("p-c90f4", "mirror it twice"),
]
ROTATE = _visit("p-3d61a", "rotate it a quarter turn")
ROTATE_AGAIN = _visit("p-3d61a", "rotate it a quarter turn, again")
NEW_PUZZLE = _visit("p-e7d22", "rotate each row")
UNRELATED = "Plan a week of vegetarian dinners for two and write the shopping list."
REWORDED = "Give me the rotation of this matrix."

GENERIC_TITLE = "Table puzzles"
GENERIC = "Read each puzzle table and its rows, then reply with the output table rows."
ROTATE_DOC = "Rotate a table a quarter turn."
ROTATE_NOTE = "Rotating tables: pass the table as given; turn direction is clockwise."

CONCEPTS = [
    {"puzzle", "puzzles", "table", "tables", "rows", "instance", "reply", "output"},
    {"rotate", "rotation", "rotating", "turn", "quarter"},
    {"mirror", "flip", "reflect"},
    {"sort", "order"},
    {"vegetarian", "dinners", "shopping", "week"},
]
FLOOR = 0.5


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

    def compute(texts):
        return np.stack([_vector(t) for t in texts])

    fake = embeddings.Embedder("evidence-test-concepts", compute)
    monkeypatch.setattr(embeddings, "embedder", lambda: fake)
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))
    real = embeddings.embed

    def spy(texts):
        calls.append(list(texts))
        return real(texts)

    monkeypatch.setattr(embeddings, "embed", spy)
    return calls


@pytest.fixture
def switches(monkeypatch):
    def set_(*, evidence=f"related:1:{FLOOR}", record=True, origin=True, lift=""):
        monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY_SHORTLIST", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_EVIDENCE_LIST", evidence)
        monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", record)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_LIFT", lift)
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_GATE", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_SHORTLIST_RELATED", "")
        monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", origin)
        monkeypatch.setattr(SETTINGS, "UNIFY_TRY_FIRST", False)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_IDENTIFIERS", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_SIMILAR_REQUEST_CORPUS", "stream")
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


def _flip_source() -> str:
    return (
        "def flip_table(table):\n"
        '    """Flip a table top to bottom."""\n'
        "    return [list(row) for row in table[::-1]]\n"
    )


def _function_id(actor, name: str) -> int:
    rows = actor.function_manager.filter_functions(filter=f"name == '{name}'")
    return int(rows[0]["function_id"])


def _seed(actor):
    """Five other puzzles logged; under ROTATE a function and its linked note; a generic note under the first."""
    for request in EARLIER:
        _in_task(request, lambda: None)
    _in_task(
        EARLIER[0],
        lambda: actor.guidance_manager.add_guidance(
            title=GENERIC_TITLE,
            content=GENERIC,
        ),
    )

    def write():
        actor.function_manager.add_functions(implementations=_rotate_source())
        fid = _function_id(actor, "rotate_table")
        actor.guidance_manager.add_guidance(
            title="Rotating tables",
            content=ROTATE_NOTE,
            function_ids=[fid],
        )

    _in_task(ROTATE, write)


async def _act(task, *, seed=None, outer_task=None):
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    token = task_origin.enter(outer_task) if outer_task else None
    try:
        with h.scripted([lambda: h.completion(content="done")] * 8) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        task_origin.leave(token)
        await actor.close()
    requests = h.session_requests(provider.requests)
    return next(m["content"] for m in requests[0]["messages"] if m["role"] == "user")


def _tier(first: str, header: str) -> str:
    if header not in first:
        return ""
    return first[first.index(header) :].split("\n\n", 1)[0]


# ── seen before ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_repeat_lists_the_function_with_its_note_as_seen_before(
    switches,
    embed_calls,
):
    switches()
    first = await _act(ROTATE_AGAIN, seed=_seed)
    seen = _tier(first, ev.SEEN_HEADER)
    lines = seen.splitlines()
    assert lines[0] == ev.SEEN_HEADER
    heads = [ln for ln in lines if ln.startswith("- ")]
    assert heads[0] == "- function `rotate_table(table)`: " + ROTATE_DOC
    # The note it links is in the same card, with its status beside its first line.
    assert lines[2] == (
        "  with guidance 2 `Rotating tables` (unverified: written in a session "
        "whose outcome is unknown): " + ROTATE_NOTE
    )
    assert lines[3] == (
        "  why listed: stored while handling a request that also named "
        "`p-3d61a` (1 earlier request named it); that session's outcome is unknown"
    )
    assert lines[4] == (
        "  record: unverified: stored in a session whose outcome is unknown; "
        "not called yet"
    )
    # The note is shown once, in the function's card, not as a card of its own.
    assert len(heads) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_an_accepted_session_and_a_call_show_in_the_record(
    switches,
    embed_calls,
):
    switches()

    def seed(actor):
        _seed(actor)
        _in_task(ROTATE, lambda: task_origin.record_outcome(True))
        _in_task(
            ROTATE,
            lambda: entry_record.record_use(
                entry_record.FUNCTION,
                "rotate_table",
                entry_record.CALL,
            ),
        )

    first = await _act(ROTATE_AGAIN, seed=seed)
    seen = _tier(first, ev.SEEN_HEADER).splitlines()
    assert seen[3].endswith("the checker accepted that session's answer")
    assert seen[4] == (
        "  record: verified: stored in a session whose answer was accepted; "
        "called in 1 session (1 accepted)"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_an_unrelated_request_gets_no_list(switches, embed_calls):
    switches()
    first = await _act(UNRELATED, seed=_seed)
    assert ev.SEEN_HEADER not in first
    assert ev.RELATED_HEADER not in first
    assert "Library at task start: 1 stored function, 2 guidance entries." in first


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_reworded_request_gets_the_function_as_possibly_related_only(
    switches,
    embed_calls,
):
    switches()
    first = await _act(REWORDED, seed=_seed)
    assert ev.SEEN_HEADER not in first
    related = _tier(first, ev.RELATED_HEADER).splitlines()
    assert related[0] == ev.RELATED_HEADER
    assert related[1].startswith(
        "- function `rotate_table(table)`: Use this when you need to rotate "
        "table. " + ROTATE_DOC,
    )
    assert "(statement from its name and docstring)" in related[1]
    assert related[2] == "  with guidance 2 `Rotating tables`"
    assert related[3].startswith("  record: unverified")
    assert len([ln for ln in related if ln.startswith("- ")]) == 1


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_sub_agent_gets_no_list(switches, embed_calls):
    switches()
    first = await _act(ROTATE_AGAIN, seed=_seed, outer_task="the caller's task")
    assert ev.SEEN_HEADER not in first
    assert ev.RELATED_HEADER not in first


# ── units: cards, tiers, one embedding call ──────────────────────────────


def _rows():
    functions = [
        {
            "function_id": 1,
            "name": "rotate_table",
            "argspec": "(table)",
            "docstring": ROTATE_DOC,
            "metadata": {},
        },
        {
            "function_id": 2,
            "name": "flip_table",
            "argspec": "(table)",
            "docstring": "Flip a table.",
            "metadata": {},
        },
    ]
    notes = [
        {
            "guidance_id": 7,
            "title": "Table turns",
            "content": "Turn or flip: pick by the cue.",
            "metadata": {},
        },
        {
            "guidance_id": 8,
            "title": "Rotation direction",
            "content": "Clockwise.",
            "metadata": {},
        },
        {
            "guidance_id": 9,
            "title": "Standalone",
            "content": "No function.",
            "metadata": {},
        },
    ]
    # Many to many: note 7 guides both functions; rotate_table has notes 7 and 8.
    links = [(1, 7), (2, 7), (1, 8)]
    return functions, notes, links


def test_cards_render_both_sides_of_many_to_many_links():
    lib = ev.build_library(*_rows())
    rotate = ev.card_for(lib, ("function", "rotate_table"))
    assert [row["guidance_id"] for _, row in rotate.linked] == [7, 8]
    note = ev.card_for(lib, ("guidance", "7"))
    assert [row["name"] for _, row in note.linked] == ["rotate_table", "flip_table"]
    alone = ev.card_for(lib, ("guidance", "9"))
    assert alone.linked == []
    text = ev.render([], [(note, "Use this when: Table turns.", False, "")], {})
    assert "- guidance 7 `Table turns`: Use this when: Table turns." in text
    assert "  guides function `rotate_table(table)`" in text
    assert "  guides function `flip_table(table)`" in text


def test_an_entry_an_earlier_card_shows_is_not_listed_again():
    lib = ev.build_library(*_rows())
    cards = ev.choose_cards(
        lib,
        [("function", "rotate_table"), ("guidance", "7"), ("guidance", "9")],
        5,
    )
    assert [card.key() for card in cards] == [
        ("function", "rotate_table"),
        ("guidance", "9"),
    ]


def _select(lib, request, *, k, embed, matcher=None):
    return ev.select(
        lib,
        request,
        request,
        None,
        [],
        {},
        k=k,
        floor=FLOOR,
        threshold=ev.DEFAULT_THRESHOLD,
        embed=embed,
        matcher=matcher,
    )


def test_one_embedding_call_per_task_start_and_none_without_the_related_tier():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.stack([_vector(t) for t in texts])

    lib = ev.build_library(*_rows())
    listing = _select(lib, REWORDED, k=1, embed=embed)
    assert len(calls) == 1 and calls[0][0] == REWORDED
    # The note on rotation is closest in meaning; its card shows the function it guides.
    ((card, *_),) = listing.related
    assert card.key() == ("guidance", "8")
    assert [row["name"] for _, row in card.linked] == ["rotate_table"]
    calls.clear()
    assert _select(lib, REWORDED, k=0, embed=embed).related == []
    assert calls == []


def test_the_matcher_is_replaceable_as_a_whole():
    class Nothing(ev.Matcher):
        def seen(self, lib, current_text, current_key, logged, uses, *, threshold):
            return []

        def related_scores(self, lib, keys, request, *, embed):
            return {("guidance", "9"): 1.0}

    lib = ev.build_library(*_rows())
    listing = _select(lib, REWORDED, k=1, embed=None, matcher=Nothing())
    assert listing.seen == []
    assert [card.key() for card, *_ in listing.related] == [("guidance", "9")]
    assert ev.MATCHERS["keys"] is ev.KeysAndStatements


def test_the_request_is_compared_whole():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return np.stack([_vector(t) for t in texts])

    lib = ev.build_library(*_rows())
    _select(lib, ROTATE_AGAIN, k=1, embed=embed)
    assert calls[0][0] == ROTATE_AGAIN


# ── refusals and settings ────────────────────────────────────────────────


def test_the_switch_parses_and_refuses_bad_values():
    assert ProductionSettings(UNIFY_EVIDENCE_LIST="on").evidence_list() == (1, None)
    assert ProductionSettings(UNIFY_EVIDENCE_LIST="related:2:0.4").evidence_list() == (
        2,
        0.4,
    )
    assert ProductionSettings().evidence_list() is None
    for bad in ("yes", "related:3", "related:1:2"):
        with pytest.raises(ValueError):
            ProductionSettings(UNIFY_EVIDENCE_LIST=bad)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"UNIFY_LIBRARY_SHORTLIST": False}, "UNIFY_LIBRARY_SHORTLIST"),
        ({"UNIFY_TASK_ORIGIN": False}, "UNIFY_TASK_ORIGIN"),
        ({"UNIFY_ENTRY_RECORD": False}, "UNIFY_ENTRY_RECORD"),
        ({"UNIFY_SHORTLIST_LIFT": "recent:4"}, "UNIFY_SHORTLIST_LIFT"),
    ],
)
def test_the_list_refuses_to_start_without_what_it_reads(
    switches,
    monkeypatch,
    change,
    message,
):
    switches()
    for name, value in change.items():
        monkeypatch.setattr(SETTINGS, name, value)
    with pytest.raises(ValueError, match=message):
        ev.require_prerequisites()


def test_the_texts_name_no_benchmark_and_ask_for_no_example_check():
    from tests.actor.code_act.test_prompt_generality import _findings

    texts = {
        "SEEN_HEADER": ev.SEEN_HEADER,
        "RELATED_HEADER": ev.RELATED_HEADER,
        "entry_record.REVIEW_SECTION": entry_record.REVIEW_SECTION,
    }
    assert _findings(texts) == []
    for text in texts.values():
        assert not re.search(r"\b(must|always|never)\b", text, re.IGNORECASE)
