"""Symbolic: ``UNIFY_EMBED_URL`` sends the OpenRouter embedding requests elsewhere.

A deployment whose model calls go through a proxy (to track their cost, or
because the process has no other way out) needs the embedding requests to go
the same way. The request is otherwise unchanged. The HTTP post is recorded,
not sent, so nothing leaves the process.
"""

from __future__ import annotations

import httpx

from unify.common import embeddings
from unify.settings import SETTINGS


def _recording(monkeypatch) -> list[str]:
    urls: list[str] = []

    def post(url, **kwargs):
        urls.append(url)
        data = [
            {"index": i, "embedding": [1.0, 0.0]}
            for i, _ in enumerate(kwargs["json"]["input"])
        ]
        return httpx.Response(
            200,
            json={"data": data},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(embeddings.httpx, "post", post)
    return urls


def test_default_posts_to_openrouter(monkeypatch):
    urls = _recording(monkeypatch)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "")
    embeddings._openrouter(["a text"])
    assert urls == ["https://openrouter.ai/api/v1/embeddings"]


def test_the_switch_names_the_endpoint(monkeypatch):
    urls = _recording(monkeypatch)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/v1/embeddings")
    vectors = embeddings._openrouter(["a text", "another"])
    assert urls == ["http://127.0.0.1:9/v1/embeddings"]
    assert vectors.shape == (2, 2)
