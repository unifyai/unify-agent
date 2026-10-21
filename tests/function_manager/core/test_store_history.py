"""Symbolic: ``UNIFY_FUNCTION_PATCH`` keeps every overwritten version.

``add_functions(overwrite=True)`` updates the stored row in place, and
``update_guidance`` does the same for a note, so as shipped the version that
was replaced is gone. With the switch on, the row as it was is appended to
``function_history`` / ``guidance_history`` with a reason and a timestamp
before each overwrite, in the same transaction. Refreshes of stale reasons
are not overwrites and write nothing; a refused overwrite writes nothing;
deleting or clearing never removes history. With the switch off nothing is
written and the stored rows are exactly as shipped. No model is called.

History outlives the per-test store reset (it is append-only), so each test
reads only the rows written after it started.
"""

from __future__ import annotations

from datetime import datetime


from tests.helpers import _handle_project
from unify import db
from unify.function_manager.function_manager import (
    DEFAULT_OVERWRITE_REASON,
    FunctionManager,
)
from unify.guidance_manager.guidance_manager import (
    DEFAULT_UPDATE_REASON,
    GuidanceManager,
)

V1 = "def scale(x: int) -> int:\n    return x * 2\n"
V2 = "def scale(x: int) -> int:\n    return x * 3\n"
V3 = "def scale(x: int) -> int:\n    return x * 4\n"


def _FM() -> FunctionManager:
    return FunctionManager(include_primitives=False)


def _mark(table: str) -> int:
    row = db.query_one(f"SELECT MAX(history_id) AS m FROM {table}")
    return int(row["m"] or 0)


def _since(table: str, mark: int) -> list[dict]:
    rows = db.query(
        f"SELECT * FROM {table} WHERE history_id > ? ORDER BY history_id",
        (mark,),
    )
    for row in rows:
        row["previous"] = db.loads(row["previous"])
    return rows


def _stored(name: str) -> dict:
    return dict(db.query_one("SELECT * FROM functions WHERE name = ?", (name,)))


# --------------------------------------------------------------------------- #
#  Functions                                                                   #
# --------------------------------------------------------------------------- #


@_handle_project
def test_an_overwrite_keeps_the_previous_row_with_its_reason():
    fm = _FM()
    mark = _mark("function_history")
    fm.add_functions(implementations=V1, preconditions={"scale": {"ok": True}})
    before = _stored("scale")
    assert _since("function_history", mark) == []  # adding is not an overwrite

    result = fm.add_functions(implementations=V2, overwrite=True)
    assert result == {"scale": "updated"}

    rows = _since("function_history", mark)
    assert len(rows) == 1
    row = rows[0]
    assert row["function_id"] == before["function_id"]
    assert row["name"] == "scale"
    assert row["reason"] == DEFAULT_OVERWRITE_REASON
    datetime.fromisoformat(row["replaced_at"])
    assert row["previous"]["implementation"] == V1
    assert row["previous"]["precondition"] == {"ok": True}
    assert row["previous"]["created_at"] == before["created_at"]
    # The full row is kept, decoded, not only the source.
    assert set(row["previous"]) == set(before)
    assert _stored("scale")["implementation"] == V2


@_handle_project
def test_each_overwrite_adds_a_row_and_nothing_removes_them():
    fm = _FM()
    mark = _mark("function_history")
    fm.add_functions(implementations=V1)
    fm.add_functions(implementations=V2, overwrite=True)
    fm.add_functions(implementations=V3, overwrite=True)
    rows = _since("function_history", mark)
    assert [r["previous"]["implementation"] for r in rows] == [V1, V2]

    fm.delete_function(function_id=_stored("scale")["function_id"])
    assert len(_since("function_history", mark)) == 2
    fm.clear()
    db.clear()
    assert len(_since("function_history", mark)) == 2


@_handle_project
def test_a_skipped_add_writes_no_history():
    fm = _FM()
    mark = _mark("function_history")
    fm.add_functions(implementations=V1)
    assert fm.add_functions(implementations=V2) == {"scale": "skipped: already exists"}
    assert _since("function_history", mark) == []


@_handle_project
def test_a_refused_overwrite_writes_no_history(monkeypatch):
    fm = _FM()
    mark = _mark("function_history")
    fm.add_functions(implementations=V1)
    broken = "def scale(x: int) -> int:\n    return undefined_helper(x)\n"
    result = fm.add_functions(
        implementations=broken,
        overwrite=True,
        raise_on_error=False,
    )
    assert result["scale"].startswith("error: ")
    assert _since("function_history", mark) == []
    assert _stored("scale")["implementation"] == V1


@_handle_project
def test_refreshing_stale_reasons_is_not_an_overwrite():
    fm = _FM()
    mark = _mark("function_history")
    fm.add_functions(
        implementations=[
            "def helper() -> int:\n    return 1\n",
            "def caller() -> int:\n    return helper()\n",
        ],
    )
    fm.delete_function(
        function_id=_stored("helper")["function_id"],
        delete_dependents=False,
    )
    fm.reconcile_dependencies()
    assert _stored("caller")["stale_reasons"] != "[]"
    assert _since("function_history", mark) == []


def _overwrite_sequence(fm: FunctionManager) -> dict:
    fm.add_functions(implementations=V1, preconditions={"scale": {"ok": True}})
    fm.add_functions(implementations=V2, overwrite=True, dependencies=[])
    row = _stored("scale")
    row.pop("created_at")
    return row


# --------------------------------------------------------------------------- #
#  Guidance                                                                    #
# --------------------------------------------------------------------------- #


def _guidance(gid: int) -> dict:
    return dict(
        db.query_one("SELECT * FROM guidance WHERE guidance_id = ?", (gid,)),
    )


@_handle_project
def test_a_guidance_update_keeps_the_previous_row():
    gm = GuidanceManager()
    mark = _mark("guidance_history")
    gid = gm.add_guidance(title="Pay on Venmo", content="Step 1. Log in.")["details"][
        "guidance_id"
    ]
    before = _guidance(gid)
    gm.update_guidance(guidance_id=gid, content="Step 1. Log in first.")
    rows = _since("guidance_history", mark)
    assert len(rows) == 1
    assert rows[0]["guidance_id"] == gid
    assert rows[0]["title"] == "Pay on Venmo"
    assert rows[0]["reason"] == DEFAULT_UPDATE_REASON
    datetime.fromisoformat(rows[0]["replaced_at"])
    assert rows[0]["previous"]["content"] == "Step 1. Log in."
    assert rows[0]["previous"]["created_at"] == before["created_at"]
    assert set(rows[0]["previous"]) == set(before)

    gm.delete_guidance(guidance_id=gid)
    gm.clear()
    assert len(_since("guidance_history", mark)) == 1


@_handle_project
def test_reconciling_guidance_is_not_an_update():
    fm = _FM()
    gm = GuidanceManager()
    mark = _mark("guidance_history")
    fm.add_functions(implementations=V1)
    fid = _stored("scale")["function_id"]
    gid = gm.add_guidance(title="t", content="c", function_ids=[fid])["details"][
        "guidance_id"
    ]
    db.execute("DELETE FROM functions WHERE function_id = ?", (fid,))
    out = gm.reconcile_dependencies()
    assert out["details"]["stale_guidance_ids"] == [gid]
    assert _since("guidance_history", mark) == []
