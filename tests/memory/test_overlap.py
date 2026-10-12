"""Overlap candidates: shapes and input fixtures, anti-unified bodies, notes with the same uses."""

from __future__ import annotations

from unify.memory import overlap as ov
from tests.memory.test_layout import LIB, _tree

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
DATA_1 = "memory/text/tests/data/e1_line.json"
DATA_2 = "memory/text/tests/data/e2_line.json"
DATA_9 = "memory/calc/tests/data/e9_rows.json"


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


def _records(fixtures: dict) -> dict:
    """Item records whose verification lists each item's input fixtures, as the gate records them."""
    return {
        item: {"item": item, "verification": {"examples": {"fixtures": paths}}}
        for item, paths in fixtures.items()
    }


def test_shapes_and_fixtures_need_both(tmp_path):
    shapes = {
        TOKENS: {"shapes": [SHAPE_1]},
        SPLIT_ID: {"shapes": [SHAPE_1, SHAPE_2]},
        "memory.calc.a:total": {
            "shapes": [SHAPE_1],
        },  # a shared shape and no shared fixture: not a candidate
    }
    records = _records(
        {
            TOKENS: [DATA_1],
            SPLIT_ID: [DATA_1, DATA_2],
            "memory.calc.a:total": [DATA_9],
            "memory.text.parse:gone": [DATA_1],  # a deleted item's fixtures are left out
        },
    )
    res = ov.overlap_candidates(
        _lib(tmp_path),
        shapes=shapes,
        fixtures=ov.fixture_signatures(records),
    )
    assert res["version"] == ov.OVERLAP_VERSION == 2
    found = _by(res, "shapes_and_fixtures")
    assert list(found) == [(TOKENS, SPLIT_ID)]
    assert found[(TOKENS, SPLIT_ID)]["reasons"] == [
        "recorded argument shapes: 1 shared of 2",
        "input fixtures: 1 shared of 2",
        f"input fixtures: every input fixture of {TOKENS} is one of {SPLIT_ID}",
    ]


def test_shapes_match_whatever_parameter_they_were_passed_as(tmp_path):
    shapes = {
        "memory.calc.a:total": {"shapes": [{**SHAPE_1, "param": "rows"}]},
        "memory.files.b:total": {"shapes": [{**SHAPE_1, "param": "path"}]},
    }
    fixtures = {
        "memory.calc.a:total": [DATA_1],
        "memory.files.b:total": [DATA_1],
    }
    res = ov.overlap_candidates(_lib(tmp_path), shapes=shapes, fixtures=fixtures)
    (c,) = _by(res, "shapes_and_fixtures").values()
    assert c["items"] == ["memory.calc.a:total", "memory.files.b:total"]
    assert c["reasons"][-1] == (
        "input fixtures: every input fixture of memory.calc.a:total is one of memory.files.b:total "
        "(the same fixtures)"
    )


def test_fixture_sets_come_only_from_the_examples_record():
    records = {
        TOKENS: {"verification": {"examples": {"fixtures": [DATA_1, "", 3]}}},
        SPLIT_ID: {"verification": {"tests": ["memory/text/tests/test_split.py"]}},
        "memory.calc.a:total": {"verification": None},
        "memory.files.b:total": {"verification": {"examples": {"fixtures": []}}},
    }
    assert ov.fixture_signatures(records) == {TOKENS: {DATA_1}}


def test_notes_using_the_same_functions(tmp_path):
    found = _by(ov.overlap_candidates(_lib(tmp_path)), "same_uses")
    assert list(found) == [("notes/text/a.md", "notes/text/b.md")]
    assert found[("notes/text/a.md", "notes/text/b.md")]["reasons"] == [
        "the notes use the same 2 function(s): memory.text.dates:parse_date, memory.text.parse:tokens",
    ]


def test_deterministic_and_fingerprinted_by_content(tmp_path):
    shapes = {TOKENS: {"shapes": [SHAPE_1]}, SPLIT_ID: {"shapes": [SHAPE_1]}}
    fixtures = {TOKENS: [DATA_1, DATA_2], SPLIT_ID: [DATA_2, DATA_1]}
    first = ov.overlap_candidates(
        _lib(tmp_path / "a"),
        shapes=shapes,
        fixtures=fixtures,
    )
    again = ov.overlap_candidates(
        _lib(tmp_path / "b"),
        shapes=dict(reversed(shapes.items())),
        fixtures={k: v[::-1] for k, v in reversed(fixtures.items())},
    )
    assert ov.render_overlap(first) == ov.render_overlap(again)
    fp = _by(first, "shapes_and_fixtures")[(TOKENS, SPLIT_ID)]["fingerprint"]
    commented = SPLIT.replace(
        "    return text.split()",
        "    # split on any whitespace\n    return text.split()",
    )
    same = ov.overlap_candidates(
        _lib(tmp_path / "c", **{SPLIT_PATH: commented}),
        shapes=shapes,
        fixtures=fixtures,
    )
    assert _by(same, "shapes_and_fixtures")[(TOKENS, SPLIT_ID)]["fingerprint"] == fp
    edited = SPLIT.replace("text.split()", "text.split(' ')")
    other = ov.overlap_candidates(
        _lib(tmp_path / "d", **{SPLIT_PATH: edited}),
        shapes=shapes,
        fixtures=fixtures,
    )
    assert _by(other, "shapes_and_fixtures")[(TOKENS, SPLIT_ID)]["fingerprint"] != fp


def test_bounds_are_named_never_silent(tmp_path):
    assert ov.overlap_candidates(_lib(tmp_path / "a"))["truncated"] == []
    assert ov.overlap_candidates(_lib(tmp_path / "b"), max_pairs=1)["truncated"] == [
        "antiunify",
    ]
