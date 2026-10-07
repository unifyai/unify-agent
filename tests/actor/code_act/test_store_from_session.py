"""Symbolic: ``UNIFY_STORE_FROM_SESSION`` reaches the storage review.

The review is told it may store a function the session ran by its name, and
its writes resolve names against the session's own trajectory. The provider
is scripted; nothing leaves the process.
"""

from __future__ import annotations

import contextlib

import pytest

from tests.actor.code_act import test_outcome_channel as toc
from tests.actor.code_act.test_outcome_channel import switches  # noqa: F401 (fixture)
from unify.function_manager import session_source as ss


def _carrying_note(requests):
    return [r for r in requests if ss.REVIEW_NOTE.strip() in toc._review_text(r)]


@pytest.mark.asyncio
async def test_on_the_review_is_told_and_resolves_against_the_session(
    monkeypatch,
    switches,
):
    entered = []
    real = ss.reviewing

    @contextlib.contextmanager
    def spy(trajectory):
        entered.append(list(trajectory or []))
        with real(trajectory):
            yield

    monkeypatch.setattr(ss, "reviewing", spy)
    _note, requests, _handle, _ = await toc._persistent_review()
    assert _carrying_note(requests)
    assert len(entered) == 1
    assert any("Send the email to Kim" in str(m.get("content")) for m in entered[0])
