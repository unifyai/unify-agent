"""Symbolic: ``UNIFY_BUILTIN_GUIDANCE=0``: the built-in catalogue is neither seeded nor read.

As shipped, every store is seeded with the Agent Skills snapshot (14 entries
such as docx, pptx and algorithmic-art) and every guidance read returns them
beside what the assistant stored. Hermes and OpenClaw start from an empty
skill library, so a learning comparison counts only what each harness learned;
with the switch off Unify's guidance reads see only its stored entries. The
embedder is replaced by a deterministic one, so nothing leaves the process.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from unify import db
from unify.common import semantic_search
from unify.guidance_manager import builtins
from unify.guidance_manager.builtins import load_snapshot, stable_guidance_id
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import ProductionSettings, SETTINGS
from tests.helpers import _handle_project


def _fake_embed(texts):
    vectors = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        v = np.random.default_rng(seed).normal(size=16)
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors)


@pytest.fixture(autouse=True)
def _offline_embed(monkeypatch):
    monkeypatch.setattr(semantic_search, "embed", _fake_embed)


def _a_builtin() -> dict:
    entry = next(iter(load_snapshot().values()))
    return {"guidance_id": stable_guidance_id(entry["title"]), **entry}


def _store(gm: GuidanceManager) -> list[int]:
    return [
        gm.add_guidance(title=title, content=f"Procedure: {title.lower()}.")["details"][
            "guidance_id"
        ]
        for title in ("Pay a contact on Venmo", "Rename downloads by date")
    ]


@_handle_project
def test_on_reads_include_the_built_in_catalogue(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", True)
    gm = GuidanceManager()
    _store(gm)
    builtin_count = db.query_one("SELECT COUNT(*) AS n FROM builtin_guidance")["n"]
    assert builtin_count > 0
    assert gm._num_items() == 2 + builtin_count
    rows = gm.search(references={"title": "make a slide deck"}, k=50)
    assert any(row.is_builtin for row in rows)


@_handle_project
def test_off_every_read_sees_only_the_stored_entries(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    gm = GuidanceManager()
    ids = _store(gm)
    builtin = _a_builtin()

    rows = gm.search(references={"title": builtin["title"]}, k=50)
    assert sorted(row.guidance_id for row in rows) == sorted(ids)
    assert not any(row.is_builtin for row in gm.search(references=None, k=50))
    assert sorted(row.guidance_id for row in gm.filter()) == sorted(ids)
    assert gm.filter(filter="is_builtin = 1") == []
    assert gm._num_items() == 2
    with pytest.raises(Exception):
        gm.get_guidance(guidance_id=builtin["guidance_id"])
    assert gm.get_guidance(guidance_id=ids[0]).title == "Pay a contact on Venmo"


@_handle_project
def test_off_the_store_keeps_rows_an_earlier_process_seeded(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", True)
    GuidanceManager()
    before = db.query_one("SELECT COUNT(*) AS n FROM builtin_guidance")["n"]
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    gm = GuidanceManager()
    assert gm._num_items() == 0
    assert db.query_one("SELECT COUNT(*) AS n FROM builtin_guidance")["n"] == before


def test_off_nothing_is_seeded(monkeypatch):
    seeded = []
    monkeypatch.setattr(builtins, "_SEEDED_FOR", set())
    monkeypatch.setattr(builtins, "seed_builtin_guidance", lambda **_: seeded.append(1))
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    builtins.ensure_seeded()
    assert seeded == [] and builtins._SEEDED_FOR == set()
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", True)
    builtins.ensure_seeded()
    assert seeded == [1]


@_handle_project
def test_off_a_built_in_title_is_not_a_read_only_entry(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    gm = GuidanceManager()
    title = _a_builtin()["title"]
    with pytest.raises(ValueError, match="No stored guidance is titled"):
        gm.patch_guidance(id_or_title=title, old="a", new="b", why="test")
    out = gm.add_guidance(title=title, content="My own procedure.")
    gid = out["details"]["guidance_id"]
    assert gm.get_guidance(guidance_id=gid).content == "My own procedure."


def test_the_setting_parses_booleans():
    assert ProductionSettings().UNIFY_BUILTIN_GUIDANCE is True
    assert (
        ProductionSettings(UNIFY_BUILTIN_GUIDANCE="0").UNIFY_BUILTIN_GUIDANCE is False
    )
    assert (
        ProductionSettings(UNIFY_BUILTIN_GUIDANCE="false").UNIFY_BUILTIN_GUIDANCE
        is False
    )
