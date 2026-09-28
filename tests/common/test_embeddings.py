"""Symbolic: vectors are computed once per text and embedder, then served from
the local cache, and either embedder ranks rows by meaning.

The ``embedders`` fixture runs a test once through OpenRouter and once through
the local model, which downloads its weights on the first run.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from unify.common import embeddings
from unify.common.embeddings import LOCAL, OPENROUTER, embed, embedder
from unify.common.semantic_search import rank_by_similarity
from unify.settings import SETTINGS


@pytest.fixture(params=[False, True], ids=["openrouter", "local"])
def embedders(request, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", request.param)


def _functions() -> list[dict]:
    return [
        {
            "id": 1,
            "name": "celsius_to_fahrenheit",
            "docstring": "Convert a temperature from Celsius to Fahrenheit.",
        },
        {
            "id": 2,
            "name": "send_invoice_email",
            "docstring": "Email an invoice PDF to a customer.",
        },
        {
            "id": 3,
            "name": "resize_image",
            "docstring": "Scale an image file to the given dimensions and save it.",
        },
    ]


def test_flag_selects_the_embedder(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", False)
    assert embedder() is OPENROUTER
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", True)
    assert embedder() is LOCAL


def test_repeated_texts_come_from_the_cache(embedders, monkeypatch):
    texts = [
        "Rotate the signing key before the audit.",
        "Water the office plants on Fridays.",
        "Rotate the signing key before the audit.",
    ]
    first = embed(texts)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert (first[0] == first[2]).all()

    def refuse(texts: list[str]) -> np.ndarray:
        raise AssertionError(f"re-embedded {texts}")

    chosen = embedder()
    monkeypatch.setattr(embeddings, "embedder", lambda: replace(chosen, compute=refuse))
    assert (embed(texts) == first).all()


def test_embedders_never_share_vectors(monkeypatch):
    text = ["Archive the quarterly board minutes."]
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", False)
    remote = embed(text)
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_EMBEDDINGS", True)
    local = embed(text)
    assert remote.shape == (1, 1536)
    assert local.shape == (1, 384)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("change a thermometer reading to the US scale", "celsius_to_fahrenheit"),
        ("mail a bill to a client", "send_invoice_email"),
        ("make a picture smaller", "resize_image"),
    ],
)
def test_rows_rank_by_meaning(embedders, query, expected):
    ranked = rank_by_similarity(
        _functions(),
        {"name": query, "docstring": query},
        limit=3,
        id_field="id",
    )
    assert ranked[0]["name"] == expected
    similarities = [row["_similarity"] for row in ranked]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in similarities)


def test_blank_references_return_the_newest_rows():
    ranked = rank_by_similarity(
        _functions(),
        {"name": "   "},
        limit=2,
        id_field="id",
    )
    assert [row["id"] for row in ranked] == [3, 2]
    assert [row["_similarity"] for row in ranked] == [0.0, 0.0]
