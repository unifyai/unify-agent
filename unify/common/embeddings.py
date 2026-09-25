"""Text embeddings for semantic search over the skill libraries.

Embeddings come from ``openai/text-embedding-3-small`` through OpenRouter's
embeddings endpoint, authenticated with the ``OPENROUTER_API_KEY`` that
unillm resolves for the LLM calls (the environment, ``.env``, or Secret
Manager). ``UNIFY_EMBED_MODEL`` selects another ``<model>@openrouter``; no
local backend is bundled.

Every vector is cached by ``(model, text)`` in a SQLite file
(``UNIFY_EMBED_CACHE``, defaulting to ``embeddings.sqlite`` beside the store)
so embedding the same text again is a lookup, across processes and runs.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import requests
from unillm.settings import SETTINGS as UNILLM_SETTINGS

from unify import db

DEFAULT_MODEL = "openai/text-embedding-3-small@openrouter"
OPENROUTER_SUFFIX = "@openrouter"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

# text-embedding-3-small accepts at most 8191 tokens per input. A byte-level
# BPE token spans at least one byte, so a head of 8000 UTF-8 bytes always
# fits; the head of an entry is where its title and summary sit anyway.
MAX_INPUT_BYTES = 8000
# One request carries at most this many inputs and UTF-8 bytes, well inside
# the endpoint's per-request limits.
MAX_BATCH_INPUTS = 256
MAX_BATCH_BYTES = 200_000
REQUEST_TIMEOUT = (10, 60)

Provider = Callable[[list[str]], list[list[float]]]

_lock = threading.RLock()
_memory: dict[tuple[str, str], np.ndarray] = {}
_cache_conn: sqlite3.Connection | None = None
_cache_path: str | None = None


class EmbeddingsUnavailable(RuntimeError):
    """Texts cannot be embedded right now: no key, or the endpoint failed."""


def configured_model() -> str:
    return os.environ.get("UNIFY_EMBED_MODEL", "").strip() or DEFAULT_MODEL


def _cache_file() -> str:
    explicit = os.environ.get("UNIFY_EMBED_CACHE", "").strip()
    if explicit:
        return explicit
    return str(db.store_home() / "embeddings.sqlite")


def _cache() -> sqlite3.Connection:
    global _cache_conn, _cache_path
    path = _cache_file()
    if _cache_conn is not None and _cache_path == path:
        return _cache_conn
    if _cache_conn is not None:
        _cache_conn.close()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        " model TEXT NOT NULL, text_hash TEXT NOT NULL, dim INTEGER NOT NULL,"
        " vector BLOB NOT NULL, PRIMARY KEY (model, text_hash))",
    )
    _cache_conn, _cache_path = conn, path
    return conn


def _head(text: str) -> str:
    """The part of ``text`` that is embedded: at most ``MAX_INPUT_BYTES``."""
    return text.encode("utf-8")[:MAX_INPUT_BYTES].decode("utf-8", errors="ignore")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _batches(texts: list[str]) -> Iterable[list[str]]:
    batch: list[str] = []
    size = 0
    for text in texts:
        length = len(text.encode("utf-8"))
        if batch and (
            len(batch) >= MAX_BATCH_INPUTS or size + length > MAX_BATCH_BYTES
        ):
            yield batch
            batch, size = [], 0
        batch.append(text)
        size += length
    if batch:
        yield batch


def _openrouter(name: str, texts: list[str]) -> list[list[float]]:
    api_key = UNILLM_SETTINGS.OPENROUTER_API_KEY.get_secret_value()
    if not api_key:
        raise EmbeddingsUnavailable(
            "no OPENROUTER_API_KEY in the environment, .env or Secret Manager",
        )
    vectors: list[list[float]] = []
    for batch in _batches(texts):
        try:
            response = requests.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": name, "input": batch},
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EmbeddingsUnavailable(
                f"OpenRouter embeddings request failed: {exc}",
            ) from exc
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise EmbeddingsUnavailable(
                f"OpenRouter returned no embeddings: {payload.get('error')!r}",
            )
        rows = sorted(rows, key=lambda row: row["index"])
        vectors.extend([float(x) for x in row["embedding"]] for row in rows)
    return vectors


def _provider(model: str) -> Provider:
    if not model.endswith(OPENROUTER_SUFFIX):
        raise EmbeddingsUnavailable(
            f"UNIFY_EMBED_MODEL={model!r} is not an OpenRouter model "
            f"(<model>{OPENROUTER_SUFFIX}); no other embedding backend is available",
        )
    name = model[: -len(OPENROUTER_SUFFIX)]
    return lambda texts: _openrouter(name, texts)


def embed_many(texts: Iterable[str], model: str | None = None) -> list[np.ndarray]:
    """Embed several texts, serving cached vectors and batching the rest.

    Raises :class:`EmbeddingsUnavailable` when a text is not cached and the
    provider cannot embed it.
    """
    model = model or configured_model()
    heads = [_head(str(text)) for text in texts]
    results: list[np.ndarray | None] = [None] * len(heads)
    missing: dict[str, list[int]] = {}
    with _lock:
        conn = _cache()
        for index, text in enumerate(heads):
            key = (model, _hash(text))
            vector = _memory.get(key)
            if vector is None:
                row = conn.execute(
                    "SELECT vector FROM embeddings WHERE model = ? AND text_hash = ?",
                    key,
                ).fetchone()
                if row is not None:
                    vector = np.frombuffer(row[0], dtype="<f4")
                    _memory[key] = vector
            if vector is None:
                missing.setdefault(text, []).append(index)
            else:
                results[index] = vector
    if missing:
        ordered = list(missing)
        fetched = _provider(model)(ordered)
        with _lock:
            conn = _cache()
            for text, values in zip(ordered, fetched, strict=True):
                vector = np.asarray(values, dtype="<f4")
                key = (model, _hash(text))
                _memory[key] = vector
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings (model, text_hash, dim, vector)"
                    " VALUES (?, ?, ?, ?)",
                    (model, key[1], len(vector), vector.tobytes()),
                )
                for index in missing[text]:
                    results[index] = vector
    return [vector for vector in results if vector is not None]


def embed(text: str, model: str | None = None) -> np.ndarray:
    """Embed one text."""
    return embed_many([text], model=model)[0]


def reset() -> None:
    """Drop in-memory state so a new cache path or model takes effect."""
    global _cache_conn, _cache_path
    with _lock:
        _memory.clear()
        if _cache_conn is not None:
            _cache_conn.close()
        _cache_conn, _cache_path = None, None


__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingsUnavailable",
    "MAX_INPUT_BYTES",
    "configured_model",
    "embed",
    "embed_many",
    "reset",
]
