"""Symbolic: ``UNIFY_FUNCTION_PATCH`` fixes a stored guidance entry by replacing excerpts.

``patch_guidance(id_or_title, old, new, why)`` -- or ``edits=[{old, new},
...]`` for several changes, applied in order and all or none -- replaces each
``old``, which must match once (the matching ladder is tested in
``tests/common/test_patch_ladder.py``), through one ``update_guidance``, so
the entry keeps its id, title and ``function_ids`` and the replaced version
goes to ``guidance_history`` with ``why``, one row per call. Built-in entries
are refused. With the switch off the method refuses and changes nothing. No
model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import SETTINGS

CONTENT = (
    "1. Log in to Venmo with the stored credentials.\n"
    "2. Top up the balance.\n"
    "3. Pay the contact.\n"
)


def _mark() -> int:
    return int(
        db.query_one("SELECT MAX(history_id) AS m FROM guidance_history")["m"] or 0,
    )


def _history_since(mark: int) -> list[dict]:
    rows = db.query(
        "SELECT * FROM guidance_history WHERE history_id > ? ORDER BY history_id",
        (mark,),
    )
    for row in rows:
        row["previous"] = db.loads(row["previous"])
    return rows


def _row(gid: int) -> dict:
    row = db.query_one("SELECT * FROM guidance WHERE guidance_id = ?", (gid,))
    return db.decode(dict(row), db.GUIDANCE_JSON_COLUMNS)


@pytest.fixture
def patch_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)


def _add(gm: GuidanceManager, title: str = "Pay on Venmo", **kw) -> int:
    return gm.add_guidance(title=title, content=CONTENT, **kw)["details"]["guidance_id"]


@_handle_project
@pytest.mark.parametrize("by", ["id", "digits", "title"])
def test_a_unique_excerpt_is_patched_in_place_with_history(patch_on, by):
    gm = GuidanceManager()
    gid = _add(gm, function_ids=[7])
    before = _row(gid)
    mark = _mark()
    key = {"id": gid, "digits": f" {gid} ", "title": "Pay on Venmo"}[by]
    out = gm.patch_guidance(
        id_or_title=key,
        old="2. Top up the balance.\n",
        new="2. Pay from the linked card; a top-up is not needed.\n",
        why="topping up first failed the task",
    )
    assert out == {
        "outcome": "guidance patched",
        "details": {
            "guidance_id": gid,
            "edits": [{"match": "exact", "replaced": 1}],
        },
    }
    after = _row(gid)
    assert after["content"] == CONTENT.replace(
        "Top up the balance.",
        "Pay from the linked card; a top-up is not needed.",
    )
    assert after["title"] == before["title"]
    assert after["function_ids"] == [7]
    assert after["created_at"] == before["created_at"]
    rows = _history_since(mark)
    assert len(rows) == 1
    assert rows[0]["guidance_id"] == gid
    assert rows[0]["reason"] == "topping up first failed the task"
    assert rows[0]["previous"]["content"] == CONTENT


@_handle_project
@pytest.mark.parametrize(
    "old, count",
    [("Top up the wallet", "0 times"), ("the", "3 times")],
)
def test_zero_or_several_matches_change_nothing(patch_on, old, count):
    gm = GuidanceManager()
    gid = _add(gm)
    mark = _mark()
    with pytest.raises(ValueError) as exc:
        gm.patch_guidance(id_or_title=gid, old=old, new="x", why="w")
    assert f"`old` occurs {count} in the content of guidance {gid}" in str(exc.value)
    assert " | " in str(exc.value)
    assert _row(gid)["content"] == CONTENT
    assert _history_since(mark) == []


@_handle_project
def test_an_ambiguous_title_is_refused_with_the_ids(patch_on):
    gm = GuidanceManager()
    a, b = _add(gm), _add(gm)
    with pytest.raises(ValueError, match=f"guidance_ids {a}, {b}"):
        gm.patch_guidance(id_or_title="Pay on Venmo", old="3.", new="4.", why="w")
    assert _row(a)["content"] == _row(b)["content"] == CONTENT


@_handle_project
def test_built_in_and_missing_entries_are_refused(patch_on):
    gm = GuidanceManager()
    builtin = db.query_one("SELECT guidance_id, title FROM builtin_guidance LIMIT 1")
    assert builtin is not None
    for key in (builtin["guidance_id"], builtin["title"]):
        with pytest.raises(ValueError, match="built-in"):
            gm.patch_guidance(id_or_title=key, old="a", new="b", why="w")
    with pytest.raises(ValueError, match="No guidance found with guidance_id 999999"):
        gm.patch_guidance(id_or_title=999999, old="a", new="b", why="w")
    with pytest.raises(ValueError, match="No stored guidance is titled 'nope'"):
        gm.patch_guidance(id_or_title="nope", old="a", new="b", why="w")
    gid = _add(gm)
    with pytest.raises(ValueError, match="say `why`"):
        gm.patch_guidance(id_or_title=gid, old="3.", new="4.", why="")


@_handle_project
def test_off_the_method_refuses_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    gm = GuidanceManager()
    gid = _add(gm)
    mark = _mark()
    with pytest.raises(ValueError, match="UNIFY_FUNCTION_PATCH is off"):
        gm.patch_guidance(id_or_title=gid, old="3.", new="4.", why="w")
    assert _row(gid)["content"] == CONTENT
    assert _history_since(mark) == []


@_handle_project
def test_a_batch_is_one_update_with_one_history_row(patch_on):
    gm = GuidanceManager()
    gid = _add(gm, function_ids=[7])
    mark = _mark()
    out = gm.patch_guidance(
        id_or_title=gid,
        why="the top-up step failed and the payment needs a note",
        edits=[
            {"old": "2. Top up the balance.\n", "new": ""},
            {"old": "3. Pay the contact.", "new": "2. Pay the contact."},
            {
                "old": "2. Pay the contact.\n",
                "new": "2. Pay the contact.\n3. Add a note.\n",
            },
        ],
    )
    assert out["details"]["edits"] == [{"match": "exact", "replaced": 1}] * 3
    assert _row(gid)["content"] == (
        "1. Log in to Venmo with the stored credentials.\n"
        "2. Pay the contact.\n"
        "3. Add a note.\n"
    )
    assert _row(gid)["function_ids"] == [7]
    rows = _history_since(mark)
    assert len(rows) == 1
    assert rows[0]["previous"]["content"] == CONTENT
    assert rows[0]["reason"] == "the top-up step failed and the payment needs a note"


@_handle_project
def test_when_edit_two_of_three_fails_nothing_is_stored(patch_on):
    gm = GuidanceManager()
    gid = _add(gm)
    mark = _mark()
    with pytest.raises(ValueError) as exc:
        gm.patch_guidance(
            id_or_title=gid,
            why="w",
            edits=[
                {"old": "1. Log in", "new": "1. Sign in"},
                {"old": "Top up the wallet", "new": "Skip the top-up"},
                {"old": "3. Pay", "new": "3. Send"},
            ],
        )
    message = str(exc.value)
    assert message.startswith(
        "Edit 2 of 3 (matched in the text as edit 1 left it; no edit was kept): "
        f"`old` occurs 0 times in the content of guidance {gid}",
    )
    assert "The closest text is line 2" in message
    assert "   2 | 2. Top up the balance." in message
    assert _row(gid)["content"] == CONTENT
    assert _history_since(mark) == []


@_handle_project
def test_prose_with_crlf_and_trailing_spaces_is_matched_and_keeps_its_endings(
    patch_on,
):
    gm = GuidanceManager()
    crlf = CONTENT.replace("\n", "  \r\n")
    gid = gm.add_guidance(title="CRLF", content=crlf)["details"]["guidance_id"]
    out = gm.patch_guidance(
        id_or_title=gid,
        old="2. Top up the balance.\n3. Pay the contact.\n",
        new="2. Pay the contact.\n",
        why="w",
    )
    assert out["details"]["edits"] == [{"match": "trailing_whitespace", "replaced": 1}]
    assert _row(gid)["content"] == (
        "1. Log in to Venmo with the stored credentials.  \r\n2. Pay the contact.\r\n"
    )


@_handle_project
def test_replace_all_aliases_and_ambiguity(patch_on):
    gm = GuidanceManager()
    gid = _add(gm)
    mark = _mark()
    with pytest.raises(ValueError) as exc:
        gm.patch_guidance(id_or_title=gid, old_string="the", new_string="a", why="w")
    assert (
        f"`old` occurs 3 times in the content of guidance {gid} (at lines 1, 2, 3)"
        in str(exc.value)
    )
    assert "set `replace_all`" in str(exc.value)
    assert _history_since(mark) == []
    out = gm.patch_guidance(
        id_or_title=gid,
        old_string="the",
        new_string="a",
        replace_all=True,
        why="w",
    )
    assert out["details"]["edits"] == [{"match": "exact", "replaced": 3}]
    assert _row(gid)["content"] == CONTENT.replace("the", "a")
    assert len(_history_since(mark)) == 1


@_handle_project
def test_with_the_switch_off_a_batch_is_refused_too(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", False)
    gm = GuidanceManager()
    gid = _add(gm)
    mark = _mark()
    with pytest.raises(ValueError, match="UNIFY_FUNCTION_PATCH is off"):
        gm.patch_guidance(
            id_or_title=gid,
            why="w",
            edits=[{"old": "3.", "new": "4."}],
        )
    assert _row(gid)["content"] == CONTENT
    assert _history_since(mark) == []
