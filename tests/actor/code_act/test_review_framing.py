"""Symbolic: ``UNIFY_REVIEW_FRAMING=unified`` and ``UNIFY_CURATION_DOCTRINE=compose``.

On ARC LOW (2 Oct analysis) the forked storage review added 0 functions in
145 forks, against 21 in 52 solved-with-code episodes for the standalone
review. The fork is the actor's own conversation, whose system prompt says a
*dedicated review* extracts skills; the appended message then switches it to
a "skill librarian" whose first instruction is "Often nothing is", and,
unlike the standalone review, it never gets the closing instruction to
store. ``unified`` frames the review as the agent's own curation step in
both prompts and gives the fork that instruction; ``compose`` replaces the
rulebook's discouragements with rules for small, composed, behaviour-named
units that a patch must not break. Requests are captured at unillm's
transport (``tests/cache_discipline_helpers.py``).
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import prompt_builders as pb
from unify.settings import ProductionSettings, SETTINGS

OFTEN_NOTHING = "Often nothing is"


@pytest.fixture
def framing(monkeypatch):
    def set_(*, unified: bool = False, compose: bool = False, fork: bool = False):
        monkeypatch.setattr(
            SETTINGS,
            "UNIFY_REVIEW_FRAMING",
            "unified" if unified else "",
        )
        monkeypatch.setattr(
            SETTINGS,
            "UNIFY_CURATION_DOCTRINE",
            "compose" if compose else "",
        )
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", fork)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", fork)

    return set_


FORK_REPLIES = (
    h.REVIEW_REPLIES[0],
    h.REVIEW_REPLIES[1],
    lambda: h.completion(content="Nothing worth storing."),
)


async def _forked_review_requests() -> list[dict]:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        _summary, _, requests = await h.scenario_review(
            FORK_REPLIES,
            actor=actor,
            tools=h.session_tools(actor),
        )
    finally:
        await actor.close()
    return requests


def _review(requests: list[dict]) -> dict:
    """The standalone review's first request (the session's come before it)."""
    return next(
        r
        for r in requests
        if r["messages"][-1]["content"] == caa._DEFAULT_STORAGE_REVIEW_INSTRUCTIONS
    )


def _system(request: dict) -> str:
    return request["messages"][0]["content"]


# ── the forked review ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unified_the_fork_is_the_agents_own_curation_step(framing):
    framing(unified=True, fork=True)
    requests = await _forked_review_requests()
    appended = requests[2]["messages"][-1]["content"]
    assert appended.startswith(
        "## Curating The Library\n\nThe task above is finished. This is the "
        "curation step that follows it: you did this work",
    )
    assert "skill librarian" not in appended
    assert OFTEN_NOTHING not in appended
    # The closing instruction the standalone review gets, after the result.
    assert appended.endswith(caa._REVIEW_CLOSING_UNIFIED)
    assert appended.index("## Final Result") < appended.index("## Now")
    assert "store any reusable functions and compositional guidance" in appended
    # The rulebook itself is unchanged.
    assert caa._STORAGE_TWO_STORES in appended
    assert "## Building The Library" not in appended


@pytest.mark.asyncio
async def test_shipped_the_fork_opens_as_the_skill_librarian(framing):
    framing(fork=True)
    requests = await _forked_review_requests()
    appended = requests[2]["messages"][-1]["content"]
    assert appended.startswith(caa._REVIEW_FORK_ROLE)
    assert OFTEN_NOTHING in appended
    assert "## Now" not in appended


@pytest.mark.asyncio
async def test_compose_the_fork_carries_the_doctrine_without_the_discouragement(
    framing,
    monkeypatch,
):
    framing(compose=True, fork=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    requests = await _forked_review_requests()
    appended = requests[2]["messages"][-1]["content"]
    assert OFTEN_NOTHING not in appended
    assert caa._STORAGE_COMPOSE_DOCTRINE in appended
    assert "patch a broader existing entry" not in appended
    assert "Do not move a fix into a broader entry" in appended
    assert caa._STORAGE_COMPOSE_STEP_3 in appended
    assert "most trajectories warrant function changes at most" not in appended
    assert (
        appended.index("## Two Stores")
        < appended.index("## Building The Library")
        < appended.index("### Update before you add")
    )


# ── the standalone review ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unified_the_standalone_review_speaks_to_the_same_agent(framing):
    framing(unified=True)
    _summary, _, requests = await h.scenario_review()
    system = _system(_review(requests))
    assert system.startswith("You are the agent that just completed the task below.")
    assert OFTEN_NOTHING not in system
    assert "## Completed Trajectory" in system


@pytest.mark.asyncio
async def test_compose_the_standalone_review_carries_the_doctrine(framing):
    framing(compose=True)
    _summary, _, requests = await h.scenario_review()
    system = _system(_review(requests))
    assert system.startswith("You are a skill librarian.")
    assert OFTEN_NOTHING not in system
    assert caa._STORAGE_COMPOSE_DOCTRINE in system
    assert caa._STORAGE_COMPOSE_STEP_3 in system


@pytest.mark.asyncio
async def test_off_the_standalone_review_is_upstreams(framing):
    framing()
    _summary, _, requests = await h.scenario_review()
    golden = json.loads(h.GOLDEN.read_text())["review"]
    assert [h.request_bytes(r) for r in requests] == golden


def test_the_compose_doctrine_names_no_benchmark_or_instance():
    text = (caa._STORAGE_COMPOSE_DOCTRINE + caa._STORAGE_COMPOSE_STEP_3).lower()
    for word in (
        "arc",
        "appworld",
        "scienceworld",
        "alfworld",
        "grid",
        "demonstration",
    ):
        assert f" {word}" not in text


# ── the actor's own prompt ───────────────────────────────────────────────


def _prompt(persist: bool) -> str:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    return pb.build_code_act_prompt(
        environments={},
        tools=dict(actor.get_tools("act")),
        can_store=True,
        persist=persist,
    )


@pytest.mark.parametrize("persist", [False, True])
def test_unified_the_actor_is_told_it_curates_after_the_task(framing, persist):
    framing(unified=True)
    prompt = _prompt(persist)
    assert "dedicated review extracts" not in prompt
    assert "dedicated skill-consolidation process" not in prompt
    assert "curation step that follows the task" in prompt
    if persist:
        assert "curate the libraries from your\ntrajectory yourself" in prompt
    else:
        assert "you also curate the libraries from your full" in prompt


@pytest.mark.parametrize("persist", [False, True])
def test_shipped_the_actor_is_told_a_dedicated_review_runs(framing, persist):
    framing()
    prompt = _prompt(persist)
    assert "a dedicated review extracts functions" in prompt
    assert "dedicated skill-consolidation process" in prompt
    assert "curation step" not in prompt


def test_the_settings_accept_only_their_values():
    assert ProductionSettings(
        UNIFY_REVIEW_FRAMING=" Unified ",
    ).UNIFY_REVIEW_FRAMING == ("unified")
    assert ProductionSettings(
        UNIFY_CURATION_DOCTRINE="compose",
    ).UNIFY_CURATION_DOCTRINE == ("compose")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_REVIEW_FRAMING="librarian")
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_CURATION_DOCTRINE="more")
