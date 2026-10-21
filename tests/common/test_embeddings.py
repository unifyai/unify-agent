"""Symbolic: vectors are computed once per text, then served from the local cache.

The embedder posts to OpenRouter's embeddings endpoint, stubbed here: the
request is built and the reply read as the real call does, and the stub's
vectors are a fixed function of each text, listed out of order as the real
endpoint may list them. Nothing leaves the process.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import httpx
import numpy as np
import pytest

from unify.common import embeddings
from unify.common.embeddings import OPENROUTER, embed, embedder


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [float(b) + 1.0 for b in digest[:16]]


@pytest.fixture
def endpoint(monkeypatch):
    """OpenRouter's embeddings endpoint, stubbed; the requests it got."""
    sent: list[dict] = []

    def post(url, *, headers, json, timeout):
        sent.append({"url": url, "headers": headers, "json": json})
        data = [{"index": i, "embedding": _vector(t)} for i, t in enumerate(json["input"])]
        return httpx.Response(
            200,
            json={"data": data[::-1], "usage": {"prompt_tokens": 7, "cost": 0.0001}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(embeddings.httpx, "post", post)
    return sent


def test_the_embedder_is_openrouters():
    assert embedder() is OPENROUTER
    assert OPENROUTER.model == "openai/text-embedding-3-small@openrouter/2048"


def test_a_request_goes_to_openrouter_with_the_provider_key(endpoint):
    rows: list[dict] = []
    vectors = embed(["Rotate the signing key."], on_call=rows.append)
    (request,) = endpoint
    assert request["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert request["headers"]["Authorization"].startswith("Bearer ")
    assert request["json"]["model"] == "openai/text-embedding-3-small"
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert rows == [
        {"model": "openai/text-embedding-3-small", "texts": 1, "prompt_tokens": 7, "cost": 0.0001},
    ]


def test_repeated_texts_come_from_the_cache(endpoint, monkeypatch):
    texts = [
        "Rotate the signing key before the audit.",
        "Water the office plants on Fridays.",
        "Rotate the signing key before the audit.",
    ]
    first = embed(texts)
    assert (first[0] == first[2]).all()
    # the reply listed out of order is put back in order
    expected = np.asarray(_vector(texts[1]), dtype=np.float32)
    assert np.allclose(first[1], expected / np.linalg.norm(expected))

    def refuse(batch: list[str]) -> np.ndarray:
        raise AssertionError(f"re-embedded {batch}")

    monkeypatch.setattr(embeddings, "embedder", lambda: replace(OPENROUTER, compute=refuse))
    assert (embed(texts) == first).all()
