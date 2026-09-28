"""Text embeddings for semantic search, kept on this machine.

A vector is computed once per model and text and stored under a hash of the
text in ``UNIFY_EMBED_CACHE``, else ``<UNIFY_HOME>/embeddings.sqlite``, so
every comparison runs here and only text never embedded before reaches a
model. The file is a cache: deleting it costs only the recomputation, and
any number of stores can share one.

Two embedders produce the vectors:

- ``openai/text-embedding-3-small`` through OpenRouter, the default, on the
  same ``OPENROUTER_API_KEY`` as the LLM. It reads the first 2,048 tokens of
  a text.
- ``BAAI/bge-small-en-v1.5`` in process, when ``UNIFY_LOCAL_EMBEDDINGS`` is
  set. It needs no network once its weights are in the Hugging Face cache,
  but reads only English and the first 512 tokens of a text.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import numpy as np
import tiktoken
import unillm

from .. import db
from ..settings import SETTINGS


@dataclass(frozen=True)
class Embedder:
    """A model that turns texts into vectors.

    ``model`` labels its vectors in the cache with everything that shapes
    them (the model, how it is reached and how much of a text it reads), so
    vectors made any other way are never compared with them.
    """

    model: str
    compute: Callable[[list[str]], np.ndarray]


_OPENROUTER_MODEL = "openai/text-embedding-3-small"
# One vector of a whole long procedure averages its purpose, which comes
# first, away with its appendices, so a text is embedded from its first 2,048
# tokens. The API caps a request at 300,000 tokens, which 128 such texts fit.
_OPENROUTER_MAX_TOKENS = 2048
_OPENROUTER_BATCH = 128


def _fit_openrouter_input(encoding: tiktoken.Encoding, text: str) -> str:
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= _OPENROUTER_MAX_TOKENS:
        return text
    return encoding.decode(tokens[:_OPENROUTER_MAX_TOKENS])


def _openrouter(texts: list[str]) -> np.ndarray:
    encoding = tiktoken.get_encoding("cl100k_base")
    inputs = [_fit_openrouter_input(encoding, text) for text in texts]
    headers = {
        "Authorization": f"Bearer {unillm.SETTINGS.OPENROUTER_API_KEY.get_secret_value()}",
    }
    vectors: list[list[float]] = []
    for start in range(0, len(inputs), _OPENROUTER_BATCH):
        response = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers=headers,
            json={
                "model": _OPENROUTER_MODEL,
                "input": inputs[start : start + _OPENROUTER_BATCH],
            },
            timeout=120,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return np.asarray(vectors, dtype=np.float32)


_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
# A pinned commit keeps the vectors stable, and a file cached under a commit
# resolves without contacting the Hub.
_LOCAL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"  # pragma: allowlist secret
_LOCAL_FILES = (
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_LOCAL_MAX_TOKENS = 512
_LOCAL_BATCH = 32
_local_lock = threading.Lock()


@cache
def _local_model() -> tuple[Any, Any]:
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer

    paths = [
        hf_hub_download(_LOCAL_MODEL, name, revision=_LOCAL_REVISION)
        for name in _LOCAL_FILES
    ]
    folder = Path(paths[0]).parent
    return (
        AutoTokenizer.from_pretrained(folder),
        AutoModel.from_pretrained(folder).eval(),
    )


def _local(texts: list[str]) -> np.ndarray:
    import torch

    # The model loads once per process, and its tokenizer cannot be shared
    # between threads.
    with _local_lock, torch.inference_mode():
        tokenizer, model = _local_model()
        batches = []
        for start in range(0, len(texts), _LOCAL_BATCH):
            inputs = tokenizer(
                texts[start : start + _LOCAL_BATCH],
                padding=True,
                truncation=True,
                max_length=_LOCAL_MAX_TOKENS,
                return_tensors="pt",
            )
            # BGE represents a text by the final hidden state of its [CLS] token.
            batches.append(model(**inputs).last_hidden_state[:, 0].numpy())
    return np.concatenate(batches)


OPENROUTER = Embedder(
    f"{_OPENROUTER_MODEL}@openrouter/{_OPENROUTER_MAX_TOKENS}",
    _openrouter,
)
LOCAL = Embedder(f"{_LOCAL_MODEL}@{_LOCAL_REVISION}/{_LOCAL_MAX_TOKENS}", _local)


def embedder() -> Embedder:
    """The embedder in use: ``LOCAL`` when ``UNIFY_LOCAL_EMBEDDINGS`` is set."""
    return LOCAL if SETTINGS.UNIFY_LOCAL_EMBEDDINGS else OPENROUTER


def _connect() -> sqlite3.Connection:
    explicit = os.environ.get("UNIFY_EMBED_CACHE", "").strip()
    path = Path(explicit) if explicit else db.store_home() / "embeddings.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vectors (model TEXT NOT NULL,"
        " text_hash TEXT NOT NULL, vector BLOB NOT NULL,"
        " PRIMARY KEY (model, text_hash))",
    )
    return conn


def _stored(
    conn: sqlite3.Connection,
    model: str,
    hashes: Sequence[str],
) -> dict[str, np.ndarray]:
    rows = conn.execute(
        "SELECT text_hash, vector FROM vectors WHERE model = ?"
        f" AND text_hash IN ({', '.join('?' for _ in hashes)})",
        (model, *hashes),
    )
    return {
        text_hash: np.frombuffer(vector, dtype=np.float32) for text_hash, vector in rows
    }


def embed(texts: Sequence[str]) -> np.ndarray:
    """Unit vectors for ``texts``, one row per text, in order.

    Texts already in the cache never reach the model. The rest are embedded
    in one pass, stored, and read back, so a text that two processes embed at
    once still resolves to the one stored vector.
    """
    chosen = embedder()
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    with closing(_connect()) as conn:
        found = _stored(conn, chosen.model, hashes)
        missing = {
            digest: text for digest, text in zip(hashes, texts) if digest not in found
        }
        if missing:
            vectors = chosen.compute(list(missing.values()))
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO vectors (model, text_hash, vector)"
                    " VALUES (?, ?, ?)",
                    [
                        (chosen.model, digest, vector.tobytes())
                        for digest, vector in zip(missing, vectors)
                    ],
                )
            found = _stored(conn, chosen.model, hashes)
    return np.stack([found[digest] for digest in hashes])


__all__ = ["Embedder", "LOCAL", "OPENROUTER", "embed", "embedder"]
