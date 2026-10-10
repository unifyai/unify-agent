"""Overlap candidates (spec v2.1 §10.3): shapes and covers, anti-unified bodies, notes with the same uses."""

from __future__ import annotations

from unify.memory_v2 import overlap as ov
from tests.memory_v2.test_layout import LIB, _tree

TOKENS = "memory.text.parse:tokens"
SPLIT_ID = "memory.text.split:split_words"
SPLIT_PATH = "memory/text/split.py"
#: tokens under another name, parameter and docstring: the same job
SPLIT = '''"""Splitting."""


def split_words(text):
    """Break a line into its parts."""
    return text.split()
'''
#: two functions with one name and unrelated bodies
TOTAL_A = '"""Sums."""\n\n\ndef total(rows):\n    """Add the amounts."""\n    return sum(r["n"] for r in rows)\n'
TOTAL_B = (
    '"""Files."""\n\n\ndef total(path):\n    """Read the lines."""\n'
    "    with open(path, encoding='utf-8') as fh:\n        return fh.read().splitlines()\n"
)
NOTE_A = "---\ntitle: A\ndescription: One.\nuses: [memory.text.dates:parse_date, memory.text.parse:tokens]\n---\nA.\n"
NOTE_B = "---\ntitle: B\ndescription: Two.\nuses: [memory.text.parse:tokens, memory.text.dates:parse_date]\n---\nB.\n"
NOTE_C = (
    "---\ntitle: C\ndescription: Three.\nuses: [memory.text.parse:tokens]\n---\nC.\n"
)
EXTRA = {
    SPLIT_PATH: SPLIT,
    "memory/calc/a.py": TOTAL_A,
    "memory/files/b.py": TOTAL_B,
    "notes/text/a.md": NOTE_A,
    "notes/text/b.md": NOTE_B,
    "notes/text/c.md": NOTE_C,
}
SHAPE_1 = {"kind": "text", "lines": 1}
SHAPE_2 = {"kind": "text", "lines": 2}


def _lib(root, **changes):
    return _tree(root, {**LIB, **EXTRA, **changes})


def _by(res, rule):
    return {tuple(c["items"]): c for c in res["candidates"] if c["rule"] == rule}


def test_identical_bodies_under_other_names_are_a_candidate(tmp_path):
    res = ov.overlap_candidates(_lib(tmp_path))
    c = _by(res, "antiunify")[(TOKENS, SPLIT_ID)]
    assert (c["kept_share"], c["holes"]) == (1.0, 0)
    assert c["reasons"][0] == "bodies anti-unify: kept share 1.00, 0 hole(s)"
    assert "their parameters differ" in c["reasons"]
    assert "def " in c["generalisation"] and c["fingerprint"].startswith("overlap:")


def test_names_never_decide(tmp_path):
    res = ov.overlap_candidates(_lib(tmp_path))
    pair = ("memory.calc.a:total", "memory.files.b:total")
    assert all(tuple(c["items"]) != pair for c in res["candidates"])


def test_shapes_and_covers_need_both(tmp_path):
    shapes = {
        TOKENS: {"shapes": [SHAPE_1]},
        SPLIT_ID: {"shapes": [SHAPE_1, SHAPE_2]},
        "memory.calc.a:total": {
            "shapes": [SHAPE_1],
        },  # a shared shape and no shared cover: not a candidate
    }
    covers = [
        (TOKENS, "e1", 0),
        (SPLIT_ID, "e1", 0),
        (SPLIT_ID, "e2", 3),
        ("memory.calc.a:total", "e9", 1),
        (
            "memory.text.parse:gone",
            "e1",
            0,
        ),  # a deleted item's recorded covers are ignored
    ]
    res = ov.overlap_candidates(_lib(tmp_path), shapes=shapes, covers=covers)
    found = _by(res, "shapes_and_covers")
    assert list(found) == [(TOKENS, SPLIT_ID)]
    assert found[(TOKENS, SPLIT_ID)]["reasons"] == [
        "recorded input shapes: 1 shared of 2",
        "covers: 1 shared of 2",
        f"covers: every cover of {TOKENS} is a cover of {SPLIT_ID}",
    ]


def test_typed_covers_overlap_too(tmp_path):
    cell = '{"index": 0, "type": "cell"}'
    shapes = {
        "memory.calc.a:total": {"shapes": [SHAPE_1]},
        "memory.files.b:total": {"shapes": [SHAPE_1]},
    }
    typed = [("memory.calc.a:total", "e5", cell), ("memory.files.b:total", "e5", cell)]
    res = ov.overlap_candidates(_lib(tmp_path), shapes=shapes, typed=typed)
    (c,) = _by(res, "shapes_and_covers").values()
    assert c["items"] == ["memory.calc.a:total", "memory.files.b:total"]
    assert c["reasons"][-1] == (
        "covers: every cover of memory.calc.a:total is a cover of memory.files.b:total (the same covers)"
    )


def test_notes_using_the_same_functions(tmp_path):
    found = _by(ov.overlap_candidates(_lib(tmp_path)), "same_uses")
    assert list(found) == [("notes/text/a.md", "notes/text/b.md")]
    assert found[("notes/text/a.md", "notes/text/b.md")]["reasons"] == [
        "the notes use the same 2 function(s): memory.text.dates:parse_date, memory.text.parse:tokens",
    ]


def test_deterministic_and_fingerprinted_by_content(tmp_path):
    shapes = {TOKENS: {"shapes": [SHAPE_1]}, SPLIT_ID: {"shapes": [SHAPE_1]}}
    covers = [(TOKENS, "e1", 0), (SPLIT_ID, "e1", 0)]
    first = ov.overlap_candidates(_lib(tmp_path / "a"), shapes=shapes, covers=covers)
    again = ov.overlap_candidates(
        _lib(tmp_path / "b"),
        shapes=dict(reversed(shapes.items())),
        covers=covers[::-1],
    )
    assert ov.render_overlap(first) == ov.render_overlap(again)
    fp = _by(first, "shapes_and_covers")[(TOKENS, SPLIT_ID)]["fingerprint"]
    commented = SPLIT.replace(
        "    return text.split()",
        "    # split on any whitespace\n    return text.split()",
    )
    same = ov.overlap_candidates(
        _lib(tmp_path / "c", **{SPLIT_PATH: commented}),
        shapes=shapes,
        covers=covers,
    )
    assert _by(same, "shapes_and_covers")[(TOKENS, SPLIT_ID)]["fingerprint"] == fp
    edited = SPLIT.replace("text.split()", "text.split(' ')")
    other = ov.overlap_candidates(
        _lib(tmp_path / "d", **{SPLIT_PATH: edited}),
        shapes=shapes,
        covers=covers,
    )
    assert _by(other, "shapes_and_covers")[(TOKENS, SPLIT_ID)]["fingerprint"] != fp


def test_bounds_are_named_never_silent(tmp_path):
    assert ov.overlap_candidates(_lib(tmp_path / "a"))["truncated"] == []
    assert ov.overlap_candidates(_lib(tmp_path / "b"), max_pairs=1)["truncated"] == [
        "antiunify",
    ]
