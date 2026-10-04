"""Symbolic: ``UNIFY_CURATION_DOCTRINE=minimal``: a storage rulebook without the office assistant.

Every storage review request carries about 3.9k tokens of rulebook written
for a colleague product: preserving user notifications in stored functions,
weekly recurring deliverables, specialist sub-agents, trials of candidate
models inside the review, PHASE/SKIP/SOFT_FAIL logging markers and a
distillation essay. ``minimal`` keeps the compose rules and what storage
mechanically needs (what can be stored and how it runs, dependencies, what
guidance is for) and drops the rest.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import SETTINGS

_DROPPED = (
    "### Preserving user-facing communication points",
    "## Recurring Deliverables",
    "## Sub-Agent Delegation Patterns",
    "### Model choice is part of distillation",
    "### The distillation dial",
    "### Expressive logging in stored functions",
    "PHASE",
    "send_notification",
)


def _sections(monkeypatch, doctrine: str) -> str:
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", doctrine)
    return caa._storage_doctrine_sections()


def test_minimal_keeps_what_storage_needs(monkeypatch):
    text = _sections(monkeypatch, "minimal")
    assert text.startswith(caa._STORAGE_MINIMAL_DOCTRINE)
    assert caa._STORAGE_COMPOSE_DOCTRINE in text
    flat = " ".join(text.split())
    for kept in (
        "`FunctionManager_add_functions`",
        "`dependencies`",
        "`query_llm(...)`",
        "`run_coro_sync(factory)`",
        "`function_ids`",
    ):
        assert kept in flat, kept


@pytest.mark.parametrize("dropped", _DROPPED)
def test_minimal_drops_the_office_assistant(monkeypatch, dropped):
    assert dropped not in _sections(monkeypatch, "minimal")
    assert dropped in _sections(monkeypatch, "compose")


def test_minimal_is_at_least_three_quarters_shorter(monkeypatch):
    compose = _sections(monkeypatch, "compose")
    minimal = _sections(monkeypatch, "minimal")
    assert len(minimal) < len(compose) / 4


def test_minimal_keeps_the_compose_instructions_and_opening(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "compose")
    compose = (caa._storage_base_instructions(), caa._review_fork_role())
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "minimal")
    assert (caa._storage_base_instructions(), caa._review_fork_role()) == compose


def test_compose_and_shipped_are_unchanged(monkeypatch):
    shipped = _sections(monkeypatch, "")
    assert shipped.startswith(caa._STORAGE_WHAT_CAN_BE_STORED)
    assert caa._STORAGE_RECURRING_DELIVERABLE in shipped
    assert caa._STORAGE_COMPOSE_DOCTRINE not in shipped


def test_minimal_names_no_benchmark():
    words = set(re.findall(r"[a-z]+", caa._STORAGE_MINIMAL_DOCTRINE.lower()))
    for word in (
        "arc",
        "appworld",
        "scienceworld",
        "crafter",
        "grid",
        "demo",
        "example",
    ):
        assert word not in words


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_sent_review_carries_the_minimal_rulebook(monkeypatch):
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "minimal")
    actor = CodeActActor()
    try:
        with h.scripted(h.ACTOR_REPLIES) as provider:
            handle = await actor.act("List the files in the workspace.", persist=False)
            await asyncio.wait_for(handle.result(), 60)
            await asyncio.wait_for(handle._completion_event.wait(), 60)
    finally:
        await actor.close()
    reviews = [
        r
        for r in provider.requests
        if caa._STORAGE_MINIMAL_DOCTRINE in str(r["messages"][0]["content"])
    ]
    assert reviews
    assert "## Recurring Deliverables" not in str(reviews[0]["messages"][0]["content"])


@pytest.mark.parametrize(
    "value, expected",
    [("minimal", "minimal"), ("MINIMAL", "minimal"), ("compose", "compose"), ("", "")],
)
def test_the_setting_parses(value, expected):
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_CURATION_DOCTRINE=value).UNIFY_CURATION_DOCTRINE
        == expected
    )
