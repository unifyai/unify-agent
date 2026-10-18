"""Functions that do the same job, settled when they are stored (``UNIFY_MEMORY_LIBRARY_OVERLAP=at-write``): the
switch, the nearest functions in behaviour, the decisions a manifest makes, the store pass's check, and the briefs.
"""

from __future__ import annotations

import shutil

import pytest

from tests.memory.test_behaviour_overlap import FILES, GRID_SHAPE, _id
from tests.memory.test_layout import _tree
from tests.memory.test_writer_pass import Turns, _writer
from unify.memory import behaviour_overlap as bo
from unify.memory import library_prompts as lp
from unify.memory import shape_rows
from unify.memory.gate import ParentSnapshot
from unify.memory.integration import switch
from unify.memory.writer_pass import PassConfig
from unify.settings import SETTINGS

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

SHAPES = {
    _id(n): {"shapes": [{**GRID_SHAPE, "param": "grid"}]}
    for n in ("double", "twice", "triple", "double_small", "width")
}
INPUTS = {
    _id("double"): [[["grid", [[1, 2], [3, 4]]]]],
    _id("twice"): [[["grid", [[7]]]]],
    _id("triple"): [[["grid", [[1, 2], [3, 4]]]]],
    _id("double_small"): [[["grid", [[1, 1]]]]],
    _id("width"): [[["grid", [[1, 2, 3]]]]],
}


def test_the_setting_defaults_to_structure_and_parses_at_write():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_OVERLAP == "structure"
    assert PassConfig().overlap == "structure"
    assert switch.PARSERS[switch.LIBRARY_OVERLAP] is switch.parse_library_overlap
    assert switch.parse_library_overlap("") == "structure"
    assert switch.parse_library_overlap(" AT-WRITE ") == "at-write"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_LIBRARY_OVERLAP"):
        switch.parse_library_overlap("behaviour")

    class S:
        UNIFY_MEMORY_LIBRARY_OVERLAP = "at-write"

    assert switch.library_overlap(S()) == "at-write"


@needs_bwrap
def test_the_nearest_functions_put_the_same_job_first_and_leave_out_what_never_agrees(
    tmp_path,
):
    root = _tree(tmp_path / "lib", FILES)
    found, cut = bo.neighbours(root, [_id("twice")], SHAPES, INPUTS)
    assert cut == []
    rows = found[_id("twice")]
    # double does the same; double_small agrees on its own input but refuses twice's; triple and width never agree
    assert [(r["item"], r["verdict"]) for r in rows] == [
        (_id("double"), bo.SAME),
        (_id("double_small"), bo.NO_CASE),
    ]
    assert rows[0]["own_cases"] == [1, 1] and rows[0]["agree"] == 2
    assert (
        bo.neighbours(root, [_id("twice")], SHAPES, INPUTS, k=1)[0][_id("twice")]
        == rows[:1]
    )
    text = bo.full_result(
        found,
        {
            _id("double"): "def double(g):\n    return 1",
            _id("twice"): "def twice(g):\n    return 1",
        },
    )
    assert text[0].startswith(f"Functions nearest {_id('twice')} in behaviour")
    assert any(
        line.startswith(f"- {_id('double')}: the same wherever both return")
        for line in text
    )
    assert "  -def double(g):" in text and "  +def twice(g):" in text
    assert bo.decision_line(_id("twice"), rows[0]).startswith(
        f"same job: {_id('twice')} returns the same as {_id('double')}",
    )


def test_a_manifest_decides_by_keep_drop_change_delete_or_alias():
    new, old, other_new = "memory.p.m:new", "memory.p.m:old", "memory.p.m:fresh"
    pending = [(new, old, False), (new, other_new, True), (other_new, new, True)]
    base = {"items": [{"item": new}, {"item": other_new}]}
    # the undecided pairs come back sorted, whatever the order of pending
    assert bo.undecided(pending, base) == sorted(
        [(new, old), (new, other_new), (other_new, new)],
    )
    assert bo.undecided(pending, {**base, "kept": {new: "a different job"}}) == []
    assert bo.undecided(pending, {**base, "kept": {new: "  "}}) == bo.undecided(
        pending,
        base,
    )
    assert bo.undecided(pending, {**base, "kept": {old: "keep the old one"}}) == sorted(
        [(new, other_new), (other_new, new)],
    )
    # the pass drops its function, or changes the existing one instead
    assert bo.undecided(pending, {"items": [{"item": other_new}]}) == [(other_new, new)]
    assert bo.undecided(pending, {"items": [{"item": new}, {"item": old}]}) == [
        (new, other_new),
    ]
    # two functions the pass adds are not decided by listing both
    assert bo.undecided([(new, other_new, True)], base) == [(new, other_new)]
    assert (
        bo.undecided(pending, {**base, "deleted": [old], "aliases": {other_new: new}})
        == []
    )
    assert bo.undecided(pending, None) == []


TEST_SOURCE = (
    "from memory.grid.ops import twice\n\n\n"
    "def test_twice():\n    assert twice([[7]]) == [[14]]\n"
)


@needs_bwrap
def test_the_store_pass_check_shows_the_same_job_and_finish_waits_for_a_decision(
    tmp_path,
    monkeypatch,
):
    # the parent has no twice (a private helper has its body): the pass adds twice and its test
    parent_ops = FILES["memory/grid/ops.py"].replace(
        "def twice(grid):",
        "def _unused(grid):",
    )
    parent = _tree(tmp_path / "parent", {**FILES, "memory/grid/ops.py": parent_ops})
    tree = _tree(
        tmp_path / "tree",
        {**FILES, "memory/grid/tests/test_ops.py": TEST_SOURCE},
    )
    _, ev, writer = _writer(
        tmp_path / "w",
        Turns([]),
        library_mode=True,
        overlap="at-write",
    )
    digests = bo.body_digests(tree)
    ev.write_function_inputs(
        _id("double"),
        digests[_id("double")],
        INPUTS[_id("double")],
    )
    ev.write_function_inputs(
        _id("triple"),
        digests[_id("triple")],
        INPUTS[_id("triple")],
    )
    # the shapes are the catalogue's business; here every function takes the same kind of grid
    monkeypatch.setattr(
        shape_rows,
        "shapes_at",
        lambda *a, **k: {i: SHAPES[i] for i in (_id("double"), _id("triple"))},
    )
    monkeypatch.setattr(
        shape_rows,
        "arg_descriptors",
        lambda calls, tree, **k: [{**GRID_SHAPE, "param": "grid"}] if calls else [],
    )
    manifest = {
        "items": [
            {
                "item": _id("twice"),
                "kind": "function",
                "tests": ["memory/grid/tests/test_ops.py"],
            },
        ],
    }
    assert writer._same_job_on()
    lines, full, pending = writer._same_job(
        tree,
        ParentSnapshot("0" * 40, {}, parent),
        manifest,
    )
    assert pending == [(_id("twice"), _id("double"), False)]
    assert lines == [
        bo.decision_line(_id("twice"), {"item": _id("double"), "agree": 2}),
    ]
    assert any(line.startswith(f"- {_id('double')}: the same") for line in full)
    kept = {**manifest, "kept": {_id("twice"): "kept apart on purpose"}}
    assert writer._same_job(tree, ParentSnapshot("0" * 40, {}, parent), kept)[0] == []
    assert bo.undecided(pending, kept) == []
    # a function the pass does not change is never compared
    assert writer._same_job(tree, ParentSnapshot("0" * 40, {}, tree), manifest) == (
        [],
        [],
        [],
    )
    writer._role = "curate"
    assert not writer._same_job_on()


@pytest.mark.parametrize("in_code", ["on", "off"])
def test_the_briefs_with_at_write(monkeypatch, in_code):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", in_code)
    plain = lp.write_brief_now()
    assert plain == lp.write_brief_now(overlap="structure")
    assert (
        lp.write_brief_now(overlap="at-write")
        == plain.rstrip("\n") + "\n\n" + lp.SAME_JOB_SECTION
    )
    cur = lp.curate_brief_now()
    assert lp.OVERLAPS_WHAT["structure"] in cur
    at_write = lp.curate_brief_now(overlap="at-write")
    assert (
        at_write.replace(lp.OVERLAPS_WHAT["at-write"], lp.OVERLAPS_WHAT["structure"])
        == cur
    )


def test_a_check_whose_comparison_raises_keeps_no_earlier_pairs(tmp_path, monkeypatch):
    tree = _tree(tmp_path / "tree", FILES)
    _, _, writer = _writer(
        tmp_path / "w",
        Turns([]),
        library_mode=True,
        overlap="at-write",
    )
    writer._same_job_pending = [(_id("twice"), _id("double"), False)]

    called = []

    def fails(*a, **k):
        called.append(1)
        raise RuntimeError("no box")

    monkeypatch.setattr(writer, "_same_job", fails)
    monkeypatch.setattr(writer.gate, "preview", lambda *a, **k: [])
    manifest = {"items": [{"item": _id("twice"), "kind": "function"}]}
    reasons: list[str] = []
    writer._check(manifest, tree, ParentSnapshot("0" * 40, {}, tree), reasons)
    assert called and writer._same_job_pending == []
