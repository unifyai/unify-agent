"""Symbolic: ``UNIFY_CURATION_DOCTRINE=balanced``.

On the 5 Oct Continual-ARC paper-protocol run notes were written as
single-instance anecdotes, one note gathered three tasks' rules, and one
puzzle ended with six entries. ``balanced`` keeps the minimal rulebook and
puts functions and guidance on equal footing, holding a note to a
function's discipline: reusable, general, conditioned, one subject, a
superset rather than a sibling. It informs; it forces no write.
"""

from __future__ import annotations

import re

import pytest

from unify.actor import code_act_actor as caa
from unify.settings import ProductionSettings, SETTINGS


def _sections(monkeypatch, doctrine: str) -> str:
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", doctrine)
    return caa._storage_doctrine_sections()


def test_balanced_replaces_the_compose_rules_on_the_minimal_rulebook(monkeypatch):
    text = _sections(monkeypatch, "balanced")
    assert text.startswith(caa._STORAGE_MINIMAL_WHAT)
    assert caa._STORAGE_BALANCED_DOCTRINE in text
    assert caa._STORAGE_BALANCED_GUIDANCE in text
    for gone in (
        caa._STORAGE_COMPOSE_DOCTRINE,
        caa._STORAGE_FUNCTIONS_FIRST_DOCTRINE,
        caa._STORAGE_MINIMAL_GUIDANCE,
        caa._STORAGE_FUNCTIONS_FIRST_GUIDANCE,
    ):
        assert gone not in text


def test_balanced_asks_notes_to_be_reusable_supersets(monkeypatch):
    flat = " ".join(_sections(monkeypatch, "balanced").split())
    assert (
        "Notes, like functions, are reusable, general-purpose and distilled as supersets."
        in flat
    )
    assert "A superset, not a sibling." in flat
    assert "One subject." in flat
    assert "Guidance comes second" not in flat


def test_balanced_step_three_and_other_doctrines_unchanged(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "balanced")
    assert caa._STORAGE_BALANCED_STEP_3 in caa._storage_base_instructions()
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "functions_first")
    assert caa._STORAGE_BALANCED_STEP_3 not in caa._storage_base_instructions()
    assert caa._STORAGE_BALANCED_DOCTRINE not in caa._storage_doctrine_sections()


def test_balanced_parses_and_asks_nothing_with_must(monkeypatch):
    assert (
        ProductionSettings(UNIFY_CURATION_DOCTRINE="balanced").UNIFY_CURATION_DOCTRINE
        == "balanced"
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_CURATION_DOCTRINE="even")
    text = caa._STORAGE_BALANCED_DOCTRINE + caa._STORAGE_BALANCED_STEP_3
    assert not re.search(r"\b(must|always)\b", text, re.IGNORECASE)
