"""Symbolic: the forked gate and reviews carry the origin-link, outcome and generalise notes.

``UNIFY_REVIEW_GATE_FORK``, ``UNIFY_REVIEW_FORK`` and ``UNIFY_REVIEW_FORK_CORE``
ask the gate and the storage review at the end of the session's own
conversation, with everything the standalone prompts carry in one appended
user message. The notes other switches add to those prompts must reach the
appended message too, in the standalone order and after the session's cached
prefix: ``UNIFY_REVIEW_OUTCOME``'s question (gate and review) and
``UNIFY_REVIEW_GENERALISE``'s functions stored for similar requests (review).
Requests are captured at unillm's transport, so nothing leaves the process;
the core review's cells would run in the real sandboxed worker.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    world,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_review_fork_core import (
    _is_review as _is_core_review,
    _session_and_review as _core_session_and_review,
)
from tests.actor.code_act.test_review_outcome import (  # noqa: F401 (fixture)
    VERDICTS,
    _kept,
    _session,
    switches,
)
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import review_outcome as ro
from unify.settings import SETTINGS

_GENERALISE = "## Functions Stored For Similar Requests\n\nstored-for-similar\n\n"
_SESSION_SYSTEM = "You are a scripted actor."


def _dumps(messages: list[dict]) -> list[str]:
    return [json.dumps(m, default=str) for m in messages]


@pytest.fixture
def forks(switches, monkeypatch):  # noqa: F811
    def set_(*, gate: bool) -> None:
        switches()
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE_FORK", gate)
        monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
        monkeypatch.setattr(caa, "_library_counts", lambda *_a, **_k: (1, 0))
        monkeypatch.setattr(
            caa,
            "_review_generalise_note",
            lambda *_a, **_k: _GENERALISE,
        )

    return set_


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_forked_gate_asks_the_outcome_and_its_judgement_is_kept(forks):
    forks(gate=True)
    note, requests = await _session(
        VERDICTS["confirmation"][0],
        "unused",
        gate_reply='{"review": false, "reason": "one-off", "answer_outcome": "confirmed"}',
    )
    assert note["type"] == "storage_review_skipped"
    gate = requests[-1]
    question = gate["messages"][-1]["content"]
    assert question.startswith("## Library Review Gate")
    assert ro.GATE_SECTION in question
    assert question.index(ro.GATE_SECTION) < question.index("## Final reply")
    # The session's last request is the gate's prefix, byte for byte.
    last = requests[-2]
    n = len(last["messages"])
    assert _dumps(gate["messages"])[:n] == _dumps(last["messages"])
    assert _kept() == (True, "review")


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_the_forked_review_carries_the_generalise_and_outcome_notes(forks):
    forks(gate=False)
    note, requests = await _session(
        VERDICTS["rejection"][0],
        'Nothing worth storing.\n{"answer_outcome": "rejected"}',
    )
    assert note["type"] == "storage_review_complete"
    review, last = requests[-1], requests[-2]
    assert review["messages"][0] == {"role": "system", "content": _SESSION_SYSTEM}
    n = len(last["messages"])
    assert _dumps(review["messages"])[:n] == _dumps(last["messages"])
    message = review["messages"][-1]["content"]
    generalise = message.index(_GENERALISE)
    outcome = message.index(ro.REVIEW_SECTION)
    final = message.index("## Final Result\n\n")
    assert generalise < outcome < final
    assert _kept() == (False, "review")


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_the_core_forked_review_carries_the_generalise_and_outcome_notes(
    core_world,  # noqa: F811
    switches,  # noqa: F811
    monkeypatch,
):
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_FORK_CORE", True)
    monkeypatch.setattr(caa, "_review_generalise_note", lambda *_a, **_k: _GENERALISE)
    session = (lambda: h.completion(content="8"),)
    review = (lambda: h.completion(content='Nothing.\n{"answer_outcome": "unknown"}'),)
    _result, requests, _stored, _closes = await _core_session_and_review(
        session,
        review,
    )
    (rev,) = [r for r in requests if _is_core_review(r)]
    last = [r for r in requests if not _is_core_review(r)][-1]
    n = len(last["messages"])
    assert _dumps(rev["messages"])[:n] == _dumps(last["messages"])
    rulebook = rev["messages"][-1]["content"]
    generalise = rulebook.index(_GENERALISE)
    outcome = rulebook.index(ro.REVIEW_SECTION)
    assert generalise < outcome < rulebook.index("## Final Result\n\n")
    assert rulebook.endswith("## Final Result\n\n8")
