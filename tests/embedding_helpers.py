"""A deterministic stand-in for the embedding endpoint.

Words map onto a few concept axes, so synonyms and paraphrases land on the
same axis ("bill", "invoice", "owe" and "settle" are all payment), while every
other word hashes onto an axis of its own. That is just enough geometry for a
test to tell ranking by meaning from ranking by shared words, with no network
and the same vectors on every run.
"""

from __future__ import annotations

import hashlib
import re

CONCEPTS: dict[str, tuple[str, ...]] = {
    "payment": (
        "pay",
        "pays",
        "paying",
        "payment",
        "bill",
        "bills",
        "invoice",
        "invoices",
        "owe",
        "owed",
        "settle",
        "settles",
    ),
    "energy": ("utility", "utilities", "electricity", "electric", "power", "gas"),
    "organisation": ("company", "vendor", "supplier", "business", "firm"),
    "message": ("email", "mail", "message", "letter", "send", "sends", "write"),
    "records": ("document", "documents", "record", "records", "knowledge", "file"),
    "lookup": ("search", "searches", "find", "look", "lookup", "locate"),
    "bookkeeping": ("ledger", "books", "accounts", "reconcile", "balance"),
    "baking": ("bake", "bread", "loaf", "loaves", "sourdough", "dough"),
    "forecast": ("budget", "forecast", "forecasts", "quarterly", "expenses"),
}
DIMENSIONS = 512

_AXIS = {word: axis for axis, words in enumerate(CONCEPTS.values()) for word in words}
_WORD = re.compile(r"[a-z]+")


def concept_vector(text: str) -> list[float]:
    """Bag of concepts: one count per concept axis or hashed word axis."""
    vector = [0.0] * DIMENSIONS
    for word in _WORD.findall(text.lower()):
        axis = _AXIS.get(word)
        if axis is None:
            digest = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            axis = len(CONCEPTS) + digest % (DIMENSIONS - len(CONCEPTS))
        vector[axis] += 1.0
    return vector


class FakeEmbeddings:
    """An embedding provider that records every batch it is asked for."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [concept_vector(text) for text in texts]

    @property
    def embedded(self) -> list[str]:
        return [text for call in self.calls for text in call]
