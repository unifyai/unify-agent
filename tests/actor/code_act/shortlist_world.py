"""The concept-fake world the shortlist tests share.

A stream of table puzzles whose requests repeat the same rules text, a
generic guidance entry and a specific function, and a fake embedder over a
few concepts whose vectors go through the real embedding cache. Requests are
captured at unillm's transport, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.common import embeddings
from unify.function_manager import task_origin

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
    if ls._HEADER in text:
        return text[text.index(ls._HEADER) :].split("\n\n", 1)[0]
    return None


def _entries(block: str | None) -> list[str]:
    return [ln for ln in (block or "").splitlines() if ln.startswith("- ")]


async def _stream(tasks, *, seed=None):
    """Each task as its own top-level act() on one store; the first user message of each."""
    out = []
    for i, task in enumerate(tasks):
        out.append(_first_user(await _act(task, seed=seed if i == 0 else None)))
    return out
