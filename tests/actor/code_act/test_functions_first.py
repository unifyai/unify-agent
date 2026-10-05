"""Symbolic: ``UNIFY_CURATION_DOCTRINE=functions_first``.

On the 5 Oct ARC LOW runs, 80 of 192 repeat visits had a solved earlier
visit and no stored function to reuse: the review kept a note, the gate
called the task a one-off, or the answer had been typed out. Where functions
were stored, 12 of 13 for one rule were helpers that left the rule's
deciding value as a parameter, and 4 of the 5 wrong reuses passed that
parameter wrongly; repeat visits stored siblings for the same rule.

``functions_first`` keeps the minimal rulebook and replaces the compose
rules with a doctrine that asks first which functions a trajectory
supports (a root function composed of small ones, varying values as
parameters, settled decisions in the code, no instance literals,
generalise rather than duplicate) and stores linked guidance as well.
It informs; it forces no write.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.settings import ProductionSettings, SETTINGS


def _sections(monkeypatch, doctrine: str) -> str:
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", doctrine)
    return caa._storage_doctrine_sections()


# ── the doctrine ─────────────────────────────────────────────────────────


def test_functions_first_replaces_the_compose_rules_on_the_minimal_rulebook(
    monkeypatch,
):
    text = _sections(monkeypatch, "functions_first")
    assert text.startswith(caa._STORAGE_MINIMAL_WHAT)
    assert caa._STORAGE_FUNCTIONS_FIRST_DOCTRINE in text
    assert caa._STORAGE_COMPOSE_DOCTRINE not in text
    assert caa._STORAGE_MINIMAL_GUIDANCE not in text
    assert caa._STORAGE_FUNCTIONS_FIRST_GUIDANCE in text
    # The minimal rulebook's mechanics stay.
    flat = " ".join(text.split())
    for kept in (
        "`FunctionManager_add_functions`",
        "`dependencies`",
        "`query_llm(...)`",
        "`run_coro_sync(factory)`",
        "`function_ids`",
    ):
        assert kept in flat, kept


def test_functions_first_states_the_preference_plainly(monkeypatch):
    flat = " ".join(_sections(monkeypatch, "functions_first").split())
    for rule in (
        "the first question of this review is which functions the trajectory supports",
        "store that procedure as a root function",
        "Compose it from small functions",
        "Parametrise what varies; decide what was decided",
        "not handed to the caller as a parameter to guess",
        "No instance literals",
        "Generalise rather than duplicate",
        "rather than adding a sibling under a new name",
        "Guidance with the functions",
        "Guidance on its own is right only when nothing in the work could be "
        "written as a function",
    ):
        assert rule in flat, rule
    # The minimal rulebook's "no guidance needed" line is not carried over.
    assert "needs no guidance entry" not in flat


def test_functions_first_has_its_own_step_3_and_keeps_the_compose_frame(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "minimal")
    minimal_steps = caa._storage_base_instructions()
    minimal_role = caa._review_fork_role()
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "functions_first")
    steps = caa._storage_base_instructions()
    assert caa._STORAGE_FUNCTIONS_FIRST_STEP_3 in steps
    assert caa._STORAGE_COMPOSE_STEP_3 not in steps
    # Steps 1, 2, 4 and 5 and the opening are the compose/minimal ones.
    head = minimal_steps[: minimal_steps.index("3. Decide")]
    tail = minimal_steps[minimal_steps.index("4. **Delete") :]
    assert steps == head + caa._STORAGE_FUNCTIONS_FIRST_STEP_3 + tail
    assert caa._review_fork_role() == minimal_role


def test_functions_first_keeps_the_compose_update_first_order(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "minimal")
    minimal = caa._storage_update_first_note()
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "functions_first")
    assert caa._storage_update_first_note() == minimal
    assert "Do not move a fix into a broader entry" in minimal


@pytest.mark.parametrize("doctrine", ["", "compose", "minimal"])
def test_the_other_doctrines_are_unchanged(monkeypatch, doctrine):
    text = _sections(monkeypatch, doctrine)
    assert caa._STORAGE_FUNCTIONS_FIRST_DOCTRINE not in text
    assert caa._STORAGE_FUNCTIONS_FIRST_GUIDANCE not in text
    assert caa._STORAGE_FUNCTIONS_FIRST_STEP_3 not in caa._storage_base_instructions()
    if doctrine == "minimal":
        assert text.startswith(caa._STORAGE_MINIMAL_DOCTRINE)
        assert caa._STORAGE_COMPOSE_DOCTRINE in text


def test_functions_first_names_no_benchmark_and_no_example_check():
    text = (
        caa._STORAGE_FUNCTIONS_FIRST_DOCTRINE
        + caa._STORAGE_FUNCTIONS_FIRST_GUIDANCE
        + caa._STORAGE_FUNCTIONS_FIRST_STEP_3
    )
    words = set(re.findall(r"[a-z]+", text.lower()))
    for word in (
        "arc",
        "appworld",
        "scienceworld",
        "crafter",
        "grid",
        "puzzle",
        "demo",
        "demonstration",
        "example",
        "examples",
        "verify",
    ):
        assert word not in words, word


@pytest.mark.parametrize(
    "value, expected",
    [
        ("functions_first", "functions_first"),
        ("FUNCTIONS_FIRST", "functions_first"),
        ("minimal", "minimal"),
        ("", ""),
    ],
)
def test_the_doctrine_setting_parses(value, expected):
    assert (
        ProductionSettings(UNIFY_CURATION_DOCTRINE=value).UNIFY_CURATION_DOCTRINE
        == expected
    )


def test_the_doctrine_setting_refuses_other_values():
    with pytest.raises(ValueError, match="functions_first"):
        ProductionSettings(UNIFY_CURATION_DOCTRINE="functions")


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_the_sent_review_carries_the_functions_first_rulebook(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "functions_first")
    actor = caa.CodeActActor()
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
        if caa._STORAGE_FUNCTIONS_FIRST_DOCTRINE in str(r["messages"][0]["content"])
    ]
    assert reviews
    assert caa._STORAGE_COMPOSE_DOCTRINE not in str(
        reviews[0]["messages"][0]["content"],
    )
