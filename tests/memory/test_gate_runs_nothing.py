"""Rev 2.1: the gate runs no code. No item, test, recipe, reader, suite or recorded episode is run to check, classify
or rank anything; what a run would have checked is recorded on the labels of the items it concerns (``not run:
<why>``), or noted for the pass.

The execution machinery behind the old checks (the gate's pytest runs, the behaviour probes, the recipe and procedure
runs, the example-call records, the held-out reruns and the stage-5 draws) is deleted; these tests hold what is left:
labels, notes and the structural checks.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.memory import sandbox_run
from unify.memory.gate import CHECKS, RED_TEST_HINT, Gate, GateResult, _Run
from unify.memory.gate_policy import NOT_RUN_GATE, not_run_reason
from unify.memory.library_checks import derive_effect
from unify.memory.procedures import Cover
from tests.memory import library_example_world as lw
from tests.memory.library_example_world import library_example_world  # noqa: F401 - a fixture

ITEM = "memory.grid.fill:flood_fill"
OTHER = "memory.grid.fill:other"
MODULE = "memory/grid/fill.py"
TEST = "memory/grid/tests/test_fill.py"
FILL = "def flood_fill(grid):\n    return grid\n"
OTHER_P = "def other(x):\n    return x\n"
OTHER_C = "def other(x):\n    return [x]\n"


@pytest.fixture
def no_box(monkeypatch):
    """Any confined run fails the test: the gate must start none."""

    def refuse(*a, **k):
        raise AssertionError("the gate started a confined run")

    monkeypatch.setattr(sandbox_run, "_spawn_box", refuse)


def _run(tmp_path, *, tests: list[str], changed_test: bool) -> _Run:
    """A G3 run over one package whose module the pass changes (``other`` is edited)."""
    p, c = tmp_path / "p", tmp_path / "c"
    for root, other in ((p, OTHER_P), (c, OTHER_C)):
        (root / "memory/grid/tests").mkdir(parents=True)
        (root / MODULE).write_text(FILL + "\n\n" + other)
        (root / TEST).write_text("from memory.grid.fill import other\n\n\ndef test_other():\n    assert other(1)\n")
    item = SimpleNamespace(item=OTHER, kind="function", tests=tests)
    run = _Run(GateResult(True, {k: True for k in CHECKS}), SimpleNamespace(items=[item]), {}, "p", "c", tmp_path)
    run.p_tree, run.c_tree = p, c
    run.p_files = {MODULE: ("100644", "a"), TEST: ("100644", "t")}
    run.c_files = {MODULE: ("100644", "b"), TEST: ("100644", "t2" if changed_test else "t")}
    run.changed = [MODULE] + ([TEST] if changed_test else [])
    run.p_bodies = {ITEM: ("function", FILL, True), OTHER: ("function", OTHER_P, True)}
    run.c_bodies = {ITEM: ("function", FILL, True), OTHER: ("function", OTHER_C, True)}
    return run


def test_g3_labels_a_listed_test_and_every_stored_function_in_scope_as_not_run(tmp_path, no_box):
    run = _run(tmp_path, tests=[TEST], changed_test=True)
    Gate._g3(Gate.__new__(Gate), run)
    assert run.res.passed and run.res.reasons == []
    assert run.label_findings[OTHER] == [
        f"not run: G3 {TEST}: {NOT_RUN_GATE}",
        f"not run: G3 behaviour on its recorded calls: {NOT_RUN_GATE}",
    ]
    # an unchanged function of the changed module is in scope too: nothing runs it either
    assert run.label_findings[ITEM] == [f"not run: G3 behaviour on its recorded calls: {NOT_RUN_GATE}"]
    assert f"note: suite memory/grid/tests not run: {NOT_RUN_GATE}" in run.notes


def test_an_edited_function_without_a_new_or_changed_test_is_labelled_with_the_hint(tmp_path, no_box):
    run = _run(tmp_path, tests=[TEST], changed_test=False)
    Gate._g3(Gate.__new__(Gate), run)
    assert run.res.passed  # labels only: a quality finding refuses nothing
    assert run.label_findings[OTHER] == [f"G3: {OTHER} has no new or changed test. {RED_TEST_HINT}"]
    assert "The gate does not run it" in RED_TEST_HINT


def test_a_listed_test_missing_from_the_candidate_still_refuses(tmp_path, no_box):
    run = _run(tmp_path, tests=["memory/grid/tests/test_gone.py"], changed_test=False)
    run.changed.append("memory/grid/tests/test_gone.py")
    Gate._g3(Gate.__new__(Gate), run)
    assert not run.res.passed
    assert "G3: listed test memory/grid/tests/test_gone.py is not in the candidate" in run.res.reasons


def test_the_not_run_reason_says_nothing_of_purity():
    assert not_run_reason("def f(x):\n    return x + 1\n") == NOT_RUN_GATE
    assert not_run_reason("def f(state, env):\n    return env.mail.read()\n") == NOT_RUN_GATE


def test_a_whole_check_and_merge_start_no_confined_run(library_example_world, no_box):  # noqa: F811
    mem, ev, blobs, gate = library_example_world
    files, fixtures = lw.files_and_fixtures(blobs.get)
    res = gate().check(mem.head(), lw.candidate(mem, files), lw.manifest(fixtures))
    assert res.passed, res.reasons
    labels = res.verification[lw.ITEM_ID]["label_findings"]
    assert f"not run: G3 {lw.TEST_FILE}: {NOT_RUN_GATE}" in labels, labels
    assert any(r.startswith("not run: G2 examples: ") for r in labels), labels
    assert res.levels[lw.ITEM_ID] == "unchecked"
    merged = gate().merge(mem.head(), lw.candidate(mem, files), lw.manifest(fixtures), "p1", "write", None, "0")
    assert merged.passed and mem.head() == merged.merged, merged.reasons


def test_an_effect_comes_from_episode_covers_only():
    episodes = {"w": SimpleNamespace(actions=[SimpleNamespace(kind="tool", effect="write")])}
    assert derive_effect([Cover("w", "episode", runner="tool")], episodes.get) == "write"
    assert derive_effect([Cover("w", "episode", runner="worktree")], episodes.get) == "write"
    assert derive_effect([Cover("w", "cell", index=0)], episodes.get) is None
    assert derive_effect([], episodes.get) is None


def test_a_dialogue_procedure_reads():
    assert derive_effect([Cover("d", "episode", runner="dialogue", action=0)], lambda eid: None) == "read"


def test_the_writer_has_no_tool_that_re_runs_recorded_cells():
    """Rev 2.1: the writer's replay tool re-ran an episode's recorded cells with a candidate in place, a re-execution of
    a capture, which no mechanism may do. It is gone, with the manifest's ``replaces`` field it read; the writer still
    runs code of its own in its box while it writes."""
    from unify.common import tool_registry
    from unify.memory import agent_profiles, manifest, tool_handlers

    assert "replay" not in tool_registry.registry()
    assert "replay" not in tool_handlers.HANDLED
    writer = next(p for p in agent_profiles.AGENT_PROFILES if p["agent"] == "writer" and p["mode"] == "write")
    assert "replay" not in writer["in_code"]
    assert "replaces" not in manifest.LIBRARY_ITEM_FIELDS
    with pytest.raises(manifest.ManifestError):
        manifest.parse_manifest(
            {"items": [{"item": "memory.a.b:f", "kind": "function", "source_episodes": ["e1"],
                        "tests": ["memory/a/tests/test_b.py"], "replaces": []}]},  # fmt: skip
        )
