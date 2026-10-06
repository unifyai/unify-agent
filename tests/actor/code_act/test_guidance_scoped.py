"""Symbolic: ``UNIFY_GUIDANCE_SCOPED`` asks the storage review to keep each lesson in its own entry.

On the 5 Oct Continual-ARC paper-protocol run the entry written after the
first (failed) instance was later grown with other tasks' rules, so one
generic note carried several tasks' lessons under the first task's origin.
With the switch the storage rulebook asks that a lesson from the trajectory
go into a guidance entry of its own rather than be appended to one written
while handling other tasks. Last, every MEMORY-SURFACE switch together with
the lean profile.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.test_listing_notes import switches  # noqa: F401 (fixture)
from tests.actor.code_act.shortlist_world import (  # noqa: F401 (fixture)
    EARLIER,
    ROTATE,
    _seed,
    _stream,
    _block,
    computed,
)
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import library_shortlist as ls
from unify.settings import SETTINGS

# ── UNIFY_GUIDANCE_SCOPED ────────────────────────────────────────────────


@pytest.mark.parametrize("doctrine", ["", "minimal", "functions_first"])
def test_scoped_guidance_rule_is_in_the_rulebook_only_when_on(
    switches,
    monkeypatch,
    doctrine,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", doctrine)
    switches()
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_SCOPED", False)
    off = caa._storage_doctrine_sections()
    assert "Keep guidance scoped" not in off
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_SCOPED", True)
    on = caa._storage_doctrine_sections()
    assert on.replace(caa._GUIDANCE_SCOPED, "") == off
    assert caa._GUIDANCE_SCOPED in on
    text = caa._GUIDANCE_SCOPED.lower()
    assert "do not append this task's lesson" in text
    for word in ("must", "always", "example", "arc", "puzzle"):
        assert word not in text.split()


# ── every MEMORY-SURFACE switch with the lean profile ─────────────────────


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_all_switches_with_the_lean_profile(switches, computed, monkeypatch):
    switches(origin=True, provenance=True, lesson=True, usage=True)
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_SCOPED", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROMPT_PROFILE", "lean")
    monkeypatch.setattr(SETTINGS, "UNIFY_CURATION_DOCTRINE", "minimal")
    monkeypatch.setattr(SETTINGS, "UNIFY_BATCH_WAKE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_REVIEW_GATE", True)
    firsts = await _stream([*EARLIER, ROTATE], seed=lambda a: _seed(a, under=ROTATE))
    block = _block(firsts[-1])
    assert block.startswith(ls._HEADER)
    lines = block.splitlines()[1:]
    function = lines.index(next(ln for ln in lines if ln.startswith("- function")))
    assert lines[function].startswith("- function `rotate_table(table)`")
    assert lines[function].endswith("[not called yet]")
    assert lines[function + 1] == (
        "  origin: stored while handling this same request; that session's "
        "outcome is unknown"
    )
    guidance = lines.index(next(ln for ln in lines if ln.startswith("- guidance")))
    assert lines[guidance].endswith(
        "(unverified: written in a session whose outcome is unknown)",
    )
    assert lines[guidance + 1].startswith("  origin: written while handling")
