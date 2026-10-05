"""Symbolic: with ``UNIFY_ENTRY_RECORD``, ``UNIFY_EVIDENCE_LIST`` and ``UNIFY_SEARCH_IDENTIFIERS`` off, the store and the review's texts are as before.

The prompts are covered by ``test_switches_off_equivalence``. This adds what
those goldens cannot see: the store a session writes (its tables, their
schemas and every stored row), the request log (none), the rows and reads a
search returns, and the storage review's rulebook and notes. The golden
``memory_a_off_golden.json`` was recorded with this file's
:func:`snapshot` on the commit these switches were built on (34ba472ac);
run it with ``RECORD_MEMORY_A_GOLDEN=1`` and ``-s`` to print it again
(between ``GOLDEN-BEGIN`` and ``GOLDEN-END`` lines).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from unify.settings import SETTINGS

GOLDEN = Path(__file__).with_name("memory_a_off_golden.json")
REQUEST = "Rename the files in batch inv_2024q3 to ISO dates and zip them."
# Columns that hold the time of the write.
_CLOCK = {"created_at", "usage_last_called_at", "replaced_at", "usage_recent_calls"}


def _fake_embed(texts):
    import numpy as np

    out = []
    for text in texts:
        seed = int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


def _source(name: str, doc: str) -> str:
    return f'def {name}(x: str) -> str:\n    """{doc}"""\n    return x\n'


def snapshot() -> dict:
    """Store writes, reads and review texts with every switch at its default, as plain data."""
    from unify import db
    from unify.actor import code_act_actor as caa
    from unify.function_manager import task_origin
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    fm, gm = FunctionManager(), GuidanceManager()
    token = task_origin.enter(REQUEST)
    try:
        fm.add_functions(
            implementations=[
                _source("rename_batch", "Rename a batch."),
                _source("zip_batch", "Zip a batch."),
            ],
        )
        ids = {r["name"]: int(r["function_id"]) for r in fm.filter_functions()}
        gid = int(
            gm.add_guidance(
                title="Batch jobs",
                content="Rename, then zip.",
                function_ids=[ids["rename_batch"], ids["zip_batch"]],
            )["details"]["guidance_id"],
        )
        gm.update_guidance(guidance_id=gid, content="Rename, then zip, then check.")
        read = gm.get_guidance(guidance_id=gid).model_dump(mode="json")
        searched = [
            g.model_dump(mode="json")
            for g in gm.search(references={"content": "zip inv_2024q3"}, k=5)
        ]
        found = fm.search_functions(
            query="rename inv_2024q3",
            n=5,
            include_implementations=False,
        )
        fm.add_functions(
            implementations=_source("zip_batch", "Zip a batch, again."),
            overwrite=True,
        )
        fm.delete_function(function_id=ids["rename_batch"], delete_dependents=False)
        _, review_note, gate_note = caa._origin_link_notes(
            [],
            outcome=None,
            answer=None,
            lessons=False,
        )
    finally:
        task_origin.leave(token)
    tables = {
        row["name"]: row["sql"]
        for row in db.query(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name",
        )
        if not str(row["name"]).startswith("sqlite_")
    }

    def rows(table: str) -> list:
        return [
            {k: v for k, v in r.items() if k not in _CLOCK}
            for r in db.query(f"SELECT * FROM {table} ORDER BY rowid")
        ]

    return {
        "tables": tables,
        "functions": rows("functions"),
        "guidance": rows("guidance"),
        "request_log_exists": task_origin.request_log_path().exists(),
        "read": read,
        "searched": searched,
        "found": [{k: v for k, v in r.items() if k not in _CLOCK} for r in found],
        "rulebook": {
            "doctrine": caa._storage_doctrine_sections(),
            "instructions": caa._storage_base_instructions(),
            "review_note": review_note,
            "gate_note": gate_note,
        },
    }


@pytest.fixture
def all_off(monkeypatch, tmp_path):
    """Every lane switch at its default, on a fresh store of its own (no table another test made)."""
    from tests.actor.code_act.test_switches_off_equivalence import NEW_SWITCHES
    from unify import db
    import unify.common.embeddings as embeddings
    import unify.common.semantic_search as semantic_search

    for name, value in NEW_SWITCHES.items():
        if hasattr(SETTINGS, name):
            monkeypatch.setattr(SETTINGS, name, value)
    monkeypatch.setattr(embeddings, "embed", _fake_embed)
    monkeypatch.setattr(semantic_search, "embed", _fake_embed)
    monkeypatch.setenv("UNIFY_STORE_PATH", str(tmp_path / "store.sqlite"))
    monkeypatch.setenv("UNIFY_HOME", str(tmp_path / "home"))
    db.reset_store()
    yield
    db.reset_store()


def test_the_store_reads_and_review_texts_are_as_before(all_off):
    got = json.loads(json.dumps(snapshot(), default=str, sort_keys=True))
    record = os.environ.get("RECORD_MEMORY_A_GOLDEN")
    if record:
        print(
            "GOLDEN-BEGIN\n"
            + json.dumps(got, indent=1, sort_keys=True)
            + "\nGOLDEN-END",
        )
        return
    assert got == json.loads(GOLDEN.read_text())
