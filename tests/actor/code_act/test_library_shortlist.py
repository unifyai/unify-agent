"""Symbolic: the library entries closest to a task are listed in its first message.

Without a discovery gate the model searches the libraries only when it
chooses to, and optional-only access is known to be under-used. So the
harness ranks the stored functions and
guidance entries against the request by embedding similarity (no model call)
and lists the closest five, one line each, in the first user message; the
model decides whether to read, call or search. Nothing is forced, nothing is
repeated later, and the ranking counts no search hit. Requests are captured
at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.agents.binding import PROMPT_SECTION

TASK = "List the files in the workspace."
HEADER = ls._HEADER
# The shared agent record's section, which the first message carries.
RECORD = PROMPT_SECTION


def _answer_at_once():
    return [lambda: h.completion(content="done")] * 8


async def _act(replies=None, *, seed=None, task=TASK):
    actor = caa.CodeActActor()
    if seed is not None:
        seed(actor)
    try:
        with h.scripted(replies or _answer_at_once()) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests), actor


def _first_user(request: dict) -> str:
    return next(m["content"] for m in request["messages"] if m["role"] == "user")


def _add_function(actor, name="list_names", doc="List the names in a directory."):
    actor.function_manager.add_functions(
        implementations=[
            f'def {name}(path):\n    """{doc}"""\n'
            "    import os\n    return sorted(os.listdir(path))",
        ],
    )


def _add_guidance(actor):
    actor.guidance_manager.add_guidance(
        title="Listing files",
        content="List a directory with os.listdir and sort the names.",
    )


def _seed(actor):
    _add_function(actor)
    _add_guidance(actor)


def _shortlist(text: str) -> str | None:
    if HEADER not in text:
        return None
    block = text[text.index(HEADER) :]
    return block.split("\n\n", 1)[0]


# ── on ───────────────────────────────────────────────────────────────────


@pytest.mark.requires_provider_key
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_first_message_lists_the_closest_function_and_guidance():
    on, actor = await _act(seed=_seed)
    first = _first_user(on[0])
    block = _shortlist(first)
    assert block is not None
    lines = block.splitlines()[1:]
    assert any(
        l.startswith("- function `list_names(path)`: List the names in a directory.")
        for l in lines
    )
    assert any(l.startswith("- guidance ") and "`Listing files`" in l for l in lines)
    # After the snapshot line, before the request; nothing else changes.
    assert first.startswith(
        "Library at task start: 1 stored function, 1 guidance entry.\n\n" + HEADER,
    )
    assert first.endswith(f"\n\n---\n\n{TASK}")
    assert on[0]["tool_choice"] == "auto"


@pytest.mark.requires_provider_key
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_list_is_written_once_and_never_repeated():
    on, _ = await _act(
        [
            lambda: h.completion(calls=[("execute_code", {"code": "print(1)"})]),
            *_answer_at_once(),
        ],
        seed=_seed,
    )
    assert len(on) >= 2
    for request in on:
        holders = [
            m
            for m in request["messages"]
            if HEADER in json.dumps(m.get("content"), default=str)
        ]
        assert [m["role"] for m in holders] == ["user"]
        assert holders[0]["content"] == _first_user(on[0])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_an_empty_library_adds_nothing():
    on, _ = await _act()
    assert _first_user(on[0]) == (
        "Library at task start: 0 stored functions, 0 guidance entries.\n\n"
        f"{RECORD}\n\n---\n\n{TASK}"
    )


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_ranking_counts_no_search_hit():

    def seed(actor):
        _add_function(actor)

    _, actor = await _act(seed=seed)
    rows = actor.function_manager._rows("name = 'list_names'")
    assert rows and int(rows[0].get("usage_search_hits") or 0) == 0


def test_at_most_five_entries_closest_first_and_no_primitives():
    class FM:
        def _shortlist_rows(self, text, k):
            return [
                {
                    "name": f"f{i}",
                    "argspec": "(x)",
                    "docstring": f"doc {i}",
                    "_similarity": s,
                }
                for i, s in enumerate([0.9, 0.2, 0.7, 0.0, 0.5, 0.6])
            ][:k]

    class GM:
        def _shortlist_rows(self, text, k):
            return [
                {
                    "guidance_id": 3,
                    "title": "T",
                    "content": "C\nmore",
                    "_similarity": 0.8,
                },
            ]

    block = ls.shortlist_block(FM(), GM(), "anything")
    lines = block.splitlines()
    assert lines[0] == HEADER
    assert lines[1:] == [
        "- function `f0(x)`: doc 0",
        "- guidance 3 `T`: C",
        "- function `f2(x)`: doc 2",
        "- function `f4(x)`: doc 4",
        "- function `f1(x)`: doc 1",
    ]
    assert len(lines) - 1 == ls.K
    assert ls.shortlisted_names(block)["guidance"] == ["3"]
    # entries with nothing in common (similarity 0) are left out
    assert all("f3" not in l for l in lines)


def test_the_managers_rank_stored_entries_only():
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager()
    rows = fm._shortlist_rows("list the names in a directory", 5)
    assert all(
        not r.get("is_primitive")
        and not str(r.get("name", "")).startswith("primitives.")
        for r in rows
    )


def test_a_failed_ranking_gives_no_list():
    class Broken:
        def _shortlist_rows(self, text, k):
            raise RuntimeError("no embeddings")

    assert ls.shortlist_block(Broken(), Broken(), "anything") is None


def test_the_header_asks_nothing_and_names_no_benchmark():
    from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS

    text = HEADER.lower()
    for word in ("must", "always", "first", "before", "try"):
        assert word not in text.replace(",", " ").split()
    assert not BENCHMARK_WORDS.search(HEADER)
