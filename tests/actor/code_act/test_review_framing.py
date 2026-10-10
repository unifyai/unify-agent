"""Symbolic: the review is framed as the agent's own curation step (``UNIFY_REVIEW_FRAMING=unified``, baked).

On ARC LOW (2 Oct analysis) the forked storage review added 0 functions in
145 forks, against 21 in 52 solved-with-code episodes for the standalone
review. The fork is the actor's own conversation, whose system prompt says a
*dedicated review* extracts skills; the appended message then switches it to
a "skill librarian" whose first instruction is "Often nothing is", and,
unlike the standalone review, it never gets the closing instruction to
store. ``unified`` frames the forked review as the agent's own curation step
and gives the fork that instruction (the core lean prompt says only that a
review follows); ``compose`` replaces the
rulebook's discouragements with rules for small, composed, behaviour-named
units that a patch must not break. Requests are captured at unillm's
transport (``tests/cache_discipline_helpers.py``).
"""

from __future__ import annotations


import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa

OFTEN_NOTHING = "Often nothing is"


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
async def test_unified_the_fork_is_the_agents_own_curation_step():
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
