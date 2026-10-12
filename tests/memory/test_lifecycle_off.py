"""With UNIFY_MEMORY_LIBRARY off, the item lifecycle changes nothing: no note, no lifecycle, v2's use record and
library copies."""

from __future__ import annotations

import json

from tests.memory.integration.test_consolidate import (
    FakeWriter,
    _events,
    _record,
    _run,
    _stores,
)
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import _tree
from unify.memory import library_export as lx
from unify.memory import lifecycle
from unify.memory.integration import consolidate
from unify.memory.integration.request import memory_use


def test_library_off_writes_no_notes_and_never_runs_the_lifecycle(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeWriter())
    called = []
    monkeypatch.setattr(
        lifecycle,
        "consolidate_records",
        lambda *a, **k: called.append(1),
    )
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert stores.evidence.pass_exists(out.pass_id)
    assert called == [] and stores.memory.run("for-each-ref", "refs/notes/items") == ""
    assert not any(e.get("phase") == "records" for e in _events(stores))


def test_library_off_use_record_and_copies_are_unchanged(tmp_path):
    root = _tree(tmp_path / "lib")
    assert lx.generated_library_mode(root) == lx.generated_library_mode(
        root,
        records=None,
    )
    items = json.loads(lx.generated_library_mode(root)[".memory/items.json"])["items"]
    assert all(
        r["status_reason"] is None and r["verification"] is None and r["use"] is None
        for r in items.values()
    )
    assert "layout" not in memory_use(_ep(), ["env/x:parse"])
