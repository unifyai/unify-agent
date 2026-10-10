"""Memory v2.1 r5 (r4 §2, §5): the general-notes sentence, and the solution-first episode overview."""

import asyncio

from unify.memory_v2 import batch_map as bm
from unify.memory_v2 import prompts_v21 as pv
from unify.memory_v2.episodes import Action, Cell, Episode
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import Turns, _call, _sol


def _ep(cells, actions):
    return Episode(
        episode_id="e1",
        started_at="2026-10-10T00:00:00Z",
        ended_at="2026-10-10T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="none",
        memory_main="abc",
        worktree_before=None,
        worktree_after=None,
        request=["solve it"],
        transcript=[],
        cells=list(cells),
        actions=list(actions),
    )


CELLS = [
    Cell(0, "grid = [[1, 2], [3, 4]]\nprint(grid)", "[[1, 2], [3, 4]]"),
    Cell(
        1,
        "def flip(g):\n    return g[::-1]\nprint(flip(grid))",
        "",
        "NameError: boom",
    ),
    Cell(2, "def flip(g):\n    return [r[::-1] for r in g]\nans = flip(grid)", "ok"),
]
ACTIONS = [
    Action(
        0,
        "env",
        "request_demos",
        [],
        {},
        "Demos received",
        "ok",
        "read",
        None,
        "dialogue",
    ),
    Action(
        2,
        "env",
        "submit",
        [[[2, 1], [4, 3]]],
        {},
        "Your submission was accepted",
        "ok",
        "write",
        None,
        "dialogue",
    ),
]


def test_overview_is_end_first_then_an_index_with_handles():
    view, end = bm.solution_first(_ep(CELLS, ACTIONS))
    head, index = view.split("== steps, newest first ==")
    assert end[:2] == [
        "cell:2",
        "action:1",
    ]  # the last cell without an error, then the last action
    assert "[r[::-1] for r in g]" in head and "submit" in head
    lines = index.strip().splitlines()
    assert lines[0].startswith("cell:2") and "defines flip" in lines[0]
    assert lines[1].startswith("cell:1") and "  error  " in lines[1]
    assert any(ln.startswith("action:1") and "env.submit" in ln for ln in lines)
    assert (
        sum(ln.startswith(("cell:", "action:", "observation:")) for ln in lines) >= 5
    )  # every step listed


def test_overview_without_cells():
    view, end = bm.solution_first(_ep([], ACTIONS[:1]))
    assert (
        "== end ==" in view
        and "action:0" in view
        and "cell:" not in view.split("== steps")[0]
    )
    view, end = bm.solution_first(_ep([], []))
    assert end == [] and "(no cell, action or observation was recorded)" in view


def test_the_overview_credits_the_end_parts_it_shows_in_full(tmp_path):
    ep = _ep(CELLS, ACTIONS)
    model = Turns([_call("o", "read_episode", {"episode": "e1", "part": "overview"})])
    _, _, sol = _sol(tmp_path, model, v21=True, max_calls=5)
    sol.load = {"e1": ep}.__getitem__
    out = asyncio.run(sol.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    assert model.outputs["o"].startswith("== end ==")
    assert (
        out.coverage["parts_read"] >= 1
    )  # cell:2 / action:1 are required parts here and were shown whole


def test_the_notes_sentence_is_in_write_and_passes_the_guards():
    text = pv.write_brief_now()
    assert "in general terms, so it applies beyond the episodes it came from" in text
    assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []
    assert "in general terms, so it applies" not in pv.curate_brief_now()
