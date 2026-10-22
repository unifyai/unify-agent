"""Typed covers: the forms :func:`unify.memory.procedures.parse_cover` reads and those it refuses. The runners that
re-did a procedure's recorded job are gone: the gate runs no code (rev 2.1)."""

import pytest

from unify.memory import procedures as pr


def test_parse_cover_forms_and_refusals():
    assert pr.parse_cover(["e1", 3]) == ("e1", 3)
    assert pr.parse_cover({"episode": "e1", "type": "cell", "cell": 2}) == pr.Cover(
        "e1",
        "cell",
        index=2,
    )
    c = pr.parse_cover(
        {"episode": "e1", "type": "episode", "runner": "tool", "params": {"n": 1}},
    )
    assert (c.runner, c.params, c.action) == ("tool", {"n": 1}, None)
    for bad, why in [
        ({"episode": "e1", "type": "episode", "runner": "shell"}, "goes to drafts"),
        (
            {"episode": "e1", "type": "episode", "runner": "dialogue"},
            "names the action",
        ),
        ({"episode": "e1", "type": "cell"}, "cell index"),
        ({"episode": "../x", "type": "diff"}, "episode id"),
        ({"episode": "e1", "type": "diff", "extra": 1}, "unknown cover keys"),
        (["e1", True], "a cover is"),
    ]:
        with pytest.raises(ValueError, match=why):
            pr.parse_cover(bad)
