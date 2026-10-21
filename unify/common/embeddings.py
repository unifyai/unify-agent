"""Text embeddings for semantic search, kept on this machine.

A vector is computed once per model and text and stored under a hash of the
text in ``<UNIFY_HOME>/embeddings.sqlite``, so every comparison runs here and
only text never embedded before reaches the model. The file is a cache:
deleting it costs only the recomputation.

The vectors are ``openai/text-embedding-3-small``'s, posted to OpenRouter on
the same ``OPENROUTER_API_KEY`` as the LLM. It reads the first 2,048 tokens of
a text.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Sequence

import httpx
import numpy as np
import tiktoken
import unillm

from .. import db


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
_OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


# One vector of a whole long procedure averages its purpose, which comes
# first, away with its appendices, so a text is embedded from its first 2,048
# tokens. The API caps a request at 300,000 tokens, which 128 such texts fit.
_OPENROUTER_MAX_TOKENS = 2048
_OPENROUTER_BATCH = 128

# Who is told of each request the embedder sends (``embed``'s ``on_call``), per thread.
_calls = threading.local()


def _tell(row: dict) -> None:
    """Tell this thread's ``on_call`` of one request sent to the model; it never fails the embedding."""
    on_call = getattr(_calls, "on_call", None)
    if on_call is None:
        return
    try:
        on_call(row)
    except Exception:  # noqa: BLE001 - an observer never changes an embedding
        pass


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
        batch = inputs[start : start + _OPENROUTER_BATCH]
        # a request that fails after it was sent may still be billed: it is told with its cost unknown
        row = {"model": _OPENROUTER_MODEL, "texts": len(batch), "prompt_tokens": None, "cost": None}
        try:
            response = httpx.post(
                _OPENROUTER_URL,
                headers=headers,
                json={"model": _OPENROUTER_MODEL, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            body = response.json()
            usage = body.get("usage") if isinstance(body, dict) else None
            if isinstance(usage, dict):
                row["prompt_tokens"] = usage.get("prompt_tokens")
                row["cost"] = usage.get("cost")
        finally:
            _tell(row)
        data = sorted(body["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return np.asarray(vectors, dtype=np.float32)


OPENROUTER = Embedder(
    f"{_OPENROUTER_MODEL}@openrouter/{_OPENROUTER_MAX_TOKENS}",
    _openrouter,
)


def embedder() -> Embedder:
    """The embedder in use."""
    return OPENROUTER


def _connect() -> sqlite3.Connection:
    path = db.store_home() / "embeddings.sqlite"
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


def embed(
    texts: Sequence[str],
    *,
    on_call: Callable[[dict], None] | None = None,
) -> np.ndarray:
    """Unit vectors for ``texts``, one row per text, in order.

    Texts already in the cache never reach the model. The rest are embedded
    in one pass, stored, and read back, so a text that two processes embed at
    once still resolves to the one stored vector.

    ``on_call`` is told of each request sent to the provider, sent or failed, as
    ``{"model", "texts", "prompt_tokens", "cost"}``: the token count and the
    cost the reply reports, ``None`` where it reports none, so a caller can
    book what it spent.
    """
    chosen = embedder()
    hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    with closing(_connect()) as conn:
        found = _stored(conn, chosen.model, hashes)
        missing = {
            digest: text for digest, text in zip(hashes, texts) if digest not in found
        }
        if missing:
            _calls.on_call = on_call
            try:
                vectors = chosen.compute(list(missing.values()))
            finally:
                _calls.on_call = None
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


__all__ = ["Embedder", "OPENROUTER", "embed", "embedder"]
