"""S9: the gate, on what a writer really leaves (a raw candidate: nothing adds a section for it), refuses a note with no
``## Rule`` section and refuses the removal of a library item; a note with its Rule section lands (the control).

Design: change c (the gate refuses a note without ``## Rule``), f and the writer brief's rule 4 ("This pass never
removes library items"; retiring is a decision, never a file removal), G5 ("no fewer items"); spec R9.2; audit W7,
W29/H1-2 (G5 only labels), H6-2/H6-3 (the "refused" tests check a stub list and every fixture adds the section) and
guard J5 (the gate behaviour table on raw candidates).

Two levels: the whole system (a scripted writer's pass through the real service, gate and library), and the real
``Gate.merge`` on raw candidates for the table of cases.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory import library_example_world as LW
from tests.memory.library_example_world import library_example_world  # noqa: F401 (fixture)
from tests.system import sysworld as SW

NOTE = "notes/text/words.md"
README = "notes/text/README.md"


def _raw(mem, files: dict, base: str = "main") -> str:
    """Commit *files* (None deletes) on *base* exactly as given: no section or README is added."""
    with mem.temp_checkout(base) as wt:
        for rel, text in files.items():
            if text is None:
                (wt / rel).unlink()
                continue
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


def _note_item() -> dict:
    return {"item": NOTE, "kind": "note", "source_episodes": ["e1"]}


def _land(world, files: dict, man: dict, pass_id: str):
    mem, _ev, _blobs, gate = world
    return gate().merge(mem.head(), _raw(mem, files), man, pass_id, "write", None, "0.01")


def _landed_note(world):
    res = _land(world, {NOTE: SW.NOTE_TEXT, README: SW.NOTE_README}, LW.manifest({}, _note_item()), "p-good")
    assert res.passed and NOTE in res.items_merged, (res.reasons, res.refused)
    return res


def _on_main(mem, rel: str) -> bool:
    return mem.run("ls-tree", "--name-only", "main", "--", rel).strip() == rel


def test_gate_lands_a_note_with_its_rule_section_on_a_raw_candidate(library_example_world):  # noqa: F811
    _landed_note(library_example_world)
    assert _on_main(library_example_world.mem, NOTE)


def test_gate_refuses_a_note_without_a_rule_section_on_a_raw_candidate(library_example_world):  # noqa: F811
    other = "notes/text/other.md"
    res = _land(library_example_world, {other: SW.NOTE_NO_RULE, README: SW.NOTE_README},
                LW.manifest({}, {"item": other, "kind": "note", "source_episodes": ["e1"]}), "p-norule")
    assert other not in res.items_merged and not _on_main(library_example_world.mem, other), (res.passed, res.reasons)


def test_gate_refuses_a_declared_deletion_of_a_library_item(library_example_world):  # noqa: F811
    _landed_note(library_example_world)
    res = _land(library_example_world, {NOTE: None}, {**LW.manifest({}), "items": [], "deleted": [NOTE]}, "p-del")
    assert not res.passed and _on_main(library_example_world.mem, NOTE), (res.passed, res.reasons)


def test_gate_refuses_an_undeclared_removal_of_a_library_item(library_example_world):  # noqa: F811
    _landed_note(library_example_world)
    res = _land(library_example_world, {NOTE: None}, {**LW.manifest({}), "items": []}, "p-undeclared")
    assert not res.passed and _on_main(library_example_world.mem, NOTE), (res.passed, res.reasons)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_a_writer_pass_cannot_land_a_note_without_a_rule_section(system, mode):
    s = system
    s.writer.plan = "land_no_rule"
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")
    assert s.writer.passes, "no pass ran for the group"
    assert not s.library_main_has(SW.NOTE_ID), "a note with no Rule section landed"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_a_writer_pass_cannot_remove_a_library_item(system, mode):
    s = system
    plans = iter(["land", "delete"])
    s.writer.plan = lambda ids: next(plans, "keep")
    await s.accepted_words(["stone"], "reverse")
    await s.accepted_words(["glass"], "upper")  # pass 1: the note lands (with its Rule section)
    assert s.library_main_has(SW.NOTE_ID), [p.get("items") for p in s.pass_results()]
    await s.accepted_words(["moss"], "title")  # pass 2: the writer removes it and declares it deleted
    assert len(s.writer.passes) >= 2, s.writer.passes
    assert s.library_main_has(SW.NOTE_ID), "a pass removed a library item"
