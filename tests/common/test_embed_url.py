"""Symbolic: ``UNIFY_EMBED_URL`` sends the OpenRouter embedding requests elsewhere.

A deployment whose model calls go through a proxy (to track their cost, or
because the process has no other way out) needs the embedding requests to go
the same way. The request is otherwise unchanged. The HTTP post is recorded,
not sent, so nothing leaves the process.
"""

from __future__ import annotations

import httpx
import pytest

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


def test_without_a_route_nothing_is_posted_and_the_refusal_names_the_settings(monkeypatch):
    """No default endpoint: a provider is never called directly, outside the model client and the run's proxy."""
    urls = _recording(monkeypatch)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "  ")
    with pytest.raises(embeddings.EmbeddingRouteRefused, match="UNIFY_EMBED_URL.*UNIFY_LOCAL_EMBEDDINGS"):
        embeddings._openrouter(["a text"])
    assert urls == []


def test_the_switch_names_the_endpoint(monkeypatch):
    urls = _recording(monkeypatch)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", "http://127.0.0.1:9/v1/embeddings")
    vectors = embeddings._openrouter(["a text", "another"])
    assert urls == ["http://127.0.0.1:9/v1/embeddings"]
    assert vectors.shape == (2, 2)


@pytest.mark.parametrize(
    "url",
    [
        "https://openrouter.ai/api/v1/embeddings",  # a provider, directly
        "http://10.0.0.5:9/v1/embeddings",  # another host
        "http://localhost:9/v1/embeddings",  # a name, not the loopback address
        "http://127.0.0.1/v1/embeddings",  # no port
        "http://user:pw@127.0.0.1:9/v1/embeddings",  # pragma: allowlist secret
        "http://127.0.0.1:9/v1/embeddings?k=v",
        "http://127.0.0.1:9/v1/embeddings#f",
        "http://127.0.0.1:9/v1 embeddings",
        "http://127.0.0.1:notaport/v1",
    ],
)
def test_only_a_loopback_route_with_a_port_is_taken_and_the_refusal_never_quotes_it(monkeypatch, url):
    """The embedder sends the process's key to the route, so only the run's loopback proxy lane is accepted."""
    urls = _recording(monkeypatch)
    monkeypatch.setattr(SETTINGS, "UNIFY_EMBED_URL", url)
    with pytest.raises(embeddings.EmbeddingRouteRefused) as refused:
        embeddings._openrouter(["a text"])
    assert urls == [] and "UNIFY_EMBED_URL" in str(refused.value)
    assert url not in str(refused.value) and "pw" not in str(refused.value)


def test_the_loopback_route_is_taken_as_written():
    url = "http://127.0.0.1:18200/lane/v1/embeddings"
    assert embeddings.embed_route(f"  {url} ") == url
