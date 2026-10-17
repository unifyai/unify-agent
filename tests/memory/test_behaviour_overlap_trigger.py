"""CURATE's overlap candidates by behaviour (``UNIFY_MEMORY_LIBRARY_OVERLAP=behaviour``): the switch, the rule set
of overlap_candidates, a kept pair staying settled until a definition changes, the other reasons unchanged, and what
the curate brief says."""

from __future__ import annotations

import shutil

import pytest

from tests.memory.test_behaviour_overlap import FILES, GRID_SHAPE, _id
from tests.memory.test_layout import _tree
from unify.memory import curate as cu
from unify.memory import library_prompts as lp
from unify.memory import overlap as ov
from unify.memory.curate import CurateState
from unify.memory.integration import switch
from unify.memory.writer_pass import PassConfig
from unify.settings import SETTINGS

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

NOTES = {
    "notes/grid/a.md": "---\ntitle: A\ndescription: a\nuses: [memory.grid.ops:double]\n---\nA.\n",
    "notes/grid/b.md": "---\ntitle: B\ndescription: b\nuses: [memory.grid.ops:double]\n---\nB.\n",
}
INPUTS = {
    _id("double"): [[["grid", [[1, 2], [3, 4]]]]],
    _id("twice"): [[["grid", [[7]]]]],
    _id("triple"): [[["grid", [[1, 2], [3, 4]]]]],
    _id("width"): [[["grid", [[1, 2, 3]]]]],
}
SHAPES = {item: {"shapes": [{**GRID_SHAPE, "param": "grid"}]} for item in INPUTS}
# the structural rules pair every one of these: one shape and one fixture in common
FIXTURES = {item: {"memory/grid/tests/data/g.json"} for item in INPUTS}


def test_the_setting_defaults_to_structure_and_parses_both_rule_sets():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_OVERLAP == "structure"
    assert PassConfig().overlap == "structure"
    assert switch.PARSERS[switch.LIBRARY_OVERLAP] is switch.parse_library_overlap
    assert switch.parse_library_overlap("") == "structure"
    assert switch.parse_library_overlap(" Behaviour ") == "behaviour"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_LIBRARY_OVERLAP"):
        switch.parse_library_overlap("both")

    class S:
        UNIFY_MEMORY_LIBRARY_OVERLAP = "behaviour"

    assert switch.library_overlap(S()) == "behaviour"
    with pytest.raises(ValueError):
        ov.overlap_candidates("not read", rules="names")


@needs_bwrap
def test_behaviour_rules_keep_only_pairs_that_do_the_same_and_the_notes_rule(tmp_path):
    root = _tree(tmp_path / "lib", {**FILES, **NOTES})
    structural = ov.overlap_candidates(root, shapes=SHAPES, fixtures=FIXTURES)
    pairs = {
        tuple(c["items"])
        for c in structural["candidates"]
        if c["rule"] == "shapes_and_fixtures"
    }
    assert len(pairs) == 6  # every pair of the four functions
    got = ov.overlap_candidates(
        root, shapes=SHAPES, fixtures=FIXTURES, rules="behaviour", inputs=INPUTS
    )
    assert [(c["rule"], c["items"]) for c in got["candidates"]] == [
        ("behaviour", [_id("double"), _id("twice")]),
        ("same_uses", ["notes/grid/a.md", "notes/grid/b.md"]),
    ]
    assert got["truncated"] == []
    assert all(c["fingerprint"].startswith("overlap:") for c in got["candidates"])
    # a function without recorded inputs is never a candidate
    none = ov.overlap_candidates(root, shapes=SHAPES, rules="behaviour", inputs={})
    assert [c["rule"] for c in none["candidates"]] == ["same_uses"]


def _state(overlap, bodies):
    return CurateState(
        commit="c" * 40,
        overlap=overlap,
        suspects={},
        index_tokens=100,
        bodies=bodies,
    )


@needs_bwrap
def test_a_kept_pair_stays_settled_until_either_definition_changes(tmp_path):
    root = _tree(tmp_path / "lib", FILES)
    first = ov.overlap_candidates(root, shapes=SHAPES, rules="behaviour", inputs=INPUTS)
    st = _state(first, {})
    due = cu.due(st, set())
    assert list(due.values()) == [
        f"overlap (behaviour): {_id('double')}, {_id('twice')}"
    ]
    # the curate pass keeps twice with a reason: that reason is settled
    st.fired = due
    seen = set(cu.settled(st, set(), {_id("twice"): "a different job"}))
    assert seen == set(due)
    # a later library with the same two definitions and one more function: nothing new is due
    more = dict(FILES)
    more["memory/grid/extra.py"] = (
        '"""More."""\n\n\ndef half(grid):\n    return [[v // 2 for v in r] for r in grid]\n'
    )
    root2 = _tree(tmp_path / "lib2", more)
    inputs2 = {**INPUTS, "memory.grid.extra:half": [[["grid", [[4]]]]]}
    shapes2 = {**SHAPES, "memory.grid.extra:half": SHAPES[_id("double")]}
    again = ov.overlap_candidates(
        root2, shapes=shapes2, rules="behaviour", inputs=inputs2
    )
    assert cu.due(_state(again, {}), seen) == {}
    # twice's definition changes (same behaviour, new body): the pair is due again
    changed = dict(FILES)
    changed["memory/grid/ops.py"] = FILES["memory/grid/ops.py"].replace(
        "v + v", "v * 2"
    )
    root3 = _tree(tmp_path / "lib3", changed)
    later = ov.overlap_candidates(
        root3, shapes=SHAPES, rules="behaviour", inputs=INPUTS
    )
    assert list(cu.due(_state(later, {}), seen).values()) == [
        f"overlap (behaviour): {_id('double')}, {_id('twice')}",
    ]


def test_suspects_and_the_index_fire_as_before_with_no_overlap():
    st = CurateState(
        commit="c" * 40,
        overlap={"version": 2, "candidates": [], "truncated": []},
        suspects={
            _id("double"): {
                "reasons": ["raised in 2 episodes"],
                "episodes": ["e1", "e2"],
            }
        },
        index_tokens=10**6,
        bodies={_id("double"): "def double(grid): ..."},
    )
    lines = sorted(cu.due(st, set()).values())
    assert len(lines) == 2
    assert lines[0].startswith("index: ") and lines[1].startswith(
        "retired by its outcomes: "
    )


@pytest.mark.parametrize("in_code", ["on", "off"])
def test_the_curate_brief_says_what_its_candidates_are(monkeypatch, in_code):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", in_code)
    plain = lp.curate_brief_now()
    assert plain == lp.curate_brief_now(overlap="structure")
    assert lp.OVERLAPS_WHAT["structure"] in plain
    by_behaviour = lp.curate_brief_now(overlap="behaviour")
    assert lp.OVERLAPS_WHAT["behaviour"] in by_behaviour
    assert lp.OVERLAPS_WHAT["structure"] not in by_behaviour
    assert (
        by_behaviour.replace(
            lp.OVERLAPS_WHAT["behaviour"], lp.OVERLAPS_WHAT["structure"]
        )
        == plain
    )
