"""Stage 7: per-item admission's record in the evidence store (the reduction itself: test_import_scope_library_mode)."""

import sqlite3

from unify.memory.evidence import EvidenceStore


def _write(root, files):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_an_old_evidence_store_gains_the_per_item_columns(tmp_path):
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(str(path))
    db.execute(
        "CREATE TABLE passes(pass_id TEXT PRIMARY KEY, kind TEXT, channel TEXT, parent TEXT, "
        "candidate TEXT, passed INTEGER, reasons TEXT, usd TEXT, patch_blob TEXT)",
    )
    db.execute("INSERT INTO passes VALUES('old','k',NULL,'p','c',1,'[]','0',NULL)")
    db.commit()
    db.close()
    ev = EvidenceStore(path)
    ev.record_pass(
        {
            "pass_id": "new",
            "passed": 1,
            "merged": "m",
            "items_merged": '["env/venmo:me"]',
            "items_refused": "{}",
        },
    )
    rows = ev.db.execute(
        "SELECT pass_id, merged, items_merged FROM passes ORDER BY pass_id",
    ).fetchall()
    assert rows == [("new", "m", '["env/venmo:me"]'), ("old", None, None)]
