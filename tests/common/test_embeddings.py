"""The embeddings provider and its cache, with no network.

The OpenRouter request is exercised against a stubbed ``requests.post``; every
other test embeds through the deterministic ``fake_embeddings`` provider.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import requests
from pydantic import SecretStr
from unillm.settings import SETTINGS as UNILLM_SETTINGS

from unify.common import embeddings
from unify.common.embeddings import EmbeddingsUnavailable

pytestmark = pytest.mark.no_unify_context


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture
def openrouter(monkeypatch, tmp_path):
    """The real provider with a placeholder key and a recording ``requests.post``."""
    monkeypatch.setenv("UNIFY_EMBED_CACHE", str(tmp_path / "embeddings.sqlite"))
    monkeypatch.delenv("UNIFY_EMBED_MODEL", raising=False)
    monkeypatch.setattr(
        UNILLM_SETTINGS,
        "OPENROUTER_API_KEY",
        SecretStr("sk-or-placeholder"),
    )
    requests_sent: list[dict[str, Any]] = []

    def post(url, *, headers, json, timeout):
        requests_sent.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout},
        )
        # Rows come back out of order; the provider sorts them by index.
        data = [
            {"index": index, "embedding": [float(index), 1.0]}
            for index in range(len(json["input"]))
        ]
        return _Response({"data": list(reversed(data))})

    monkeypatch.setattr(requests, "post", post)
    embeddings.reset()
    yield requests_sent
    embeddings.reset()


def test_openrouter_request_embeds_through_text_embedding_3_small(openrouter):
    first, second = embeddings.embed_many(["first", "second"])

    assert first.tolist() == [0.0, 1.0]
    assert second.tolist() == [1.0, 1.0]
    [sent] = openrouter
    assert sent["url"] == "https://openrouter.ai/api/v1/embeddings"
    assert sent["json"] == {
        "model": "openai/text-embedding-3-small",
        "input": ["first", "second"],
    }
    assert sent["headers"] == {"Authorization": "Bearer sk-or-placeholder"}
    assert sent["timeout"]


def test_requests_are_batched_by_input_count(openrouter, monkeypatch):
    monkeypatch.setattr(embeddings, "MAX_BATCH_INPUTS", 2)

    vectors = embeddings.embed_many(["a", "b", "c"])

    assert [sent["json"]["input"] for sent in openrouter] == [["a", "b"], ["c"]]
    # Each batch indexes from zero; results keep the order of the inputs.
    assert [vector.tolist() for vector in vectors] == [
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]


def test_missing_key_makes_embeddings_unavailable(no_embedding_key):
    with pytest.raises(EmbeddingsUnavailable, match="OPENROUTER_API_KEY"):
        embeddings.embed("pay a bill")


def test_failed_request_makes_embeddings_unavailable(openrouter, monkeypatch):
    def unreachable(*args, **kwargs):
        raise requests.ConnectionError("openrouter.ai unreachable")

    monkeypatch.setattr(requests, "post", unreachable)

    with pytest.raises(EmbeddingsUnavailable, match="unreachable") as raised:
        embeddings.embed("pay a bill")
    assert "sk-or-placeholder" not in str(raised.value)


def test_a_model_off_openrouter_is_unavailable(openrouter, monkeypatch):
    monkeypatch.setenv("UNIFY_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    with pytest.raises(EmbeddingsUnavailable, match="@openrouter"):
        embeddings.embed("pay a bill")
    assert openrouter == []


def test_repeated_texts_are_served_from_the_cache(fake_embeddings):
    alpha, beta, alpha_again = embeddings.embed_many(["alpha", "beta", "alpha"])

    assert fake_embeddings.calls == [["alpha", "beta"]]
    assert np.array_equal(alpha, alpha_again)
    assert not np.array_equal(alpha, beta)

    embeddings.embed("beta")
    # A new process starts with an empty memory but the same file on disk.
    embeddings.reset()
    assert np.array_equal(embeddings.embed("alpha"), alpha)
    assert fake_embeddings.calls == [["alpha", "beta"]]


def test_the_cache_is_keyed_by_model(fake_embeddings, monkeypatch):
    embeddings.embed("alpha")
    monkeypatch.setenv("UNIFY_EMBED_MODEL", "test/other-concepts")
    embeddings.embed("alpha")

    assert fake_embeddings.calls == [["alpha"], ["alpha"]]


def test_inputs_are_cut_to_the_endpoint_limit(fake_embeddings):
    long_text = "é" * embeddings.MAX_INPUT_BYTES

    embeddings.embed(long_text)
    embeddings.embed(long_text + " a tail beyond the limit")

    [[sent]] = fake_embeddings.calls
    assert len(sent.encode("utf-8")) <= embeddings.MAX_INPUT_BYTES
    assert long_text.startswith(sent)
