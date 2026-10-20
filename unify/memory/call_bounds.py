"""One model request's worst case, per model: the tier-1 catalogue-max bound of the pinned public catalogue (cbb212a9,
the same source as the programme guard's ``catalogue_max_bound`` and the kc3hf68se guard; e.g. 0.65625 USD and
13.125 USD for the two models the kits run), in ``call_bounds.json`` beside this file (written by the research repo's
``artifacts/orchestrator/REDESIGN-BUILD-20261020/builder-a/make_call_bounds.py``). A model the catalogue cannot bound
has no entry and is refused: fail closed.

Standard library only.
"""

from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Iterable

#: The table and the catalogue it was computed from.
TABLE = Path(__file__).with_name("call_bounds.json")
CATALOGUE_SHA256 = "cbb212a90a75ba8b7326ba14dd15eb3548dde6923846c005c2f71eb22a50beae"


class NoBound(ValueError):
    """A model the pinned catalogue does not bound: nothing that needs its worst case may run."""


@lru_cache(maxsize=1)
def _table() -> dict[str, Decimal]:
    doc = json.loads(TABLE.read_text(encoding="utf-8"))
    if doc.get("catalogue_sha256") != CATALOGUE_SHA256:
        raise NoBound(f"{TABLE.name} was not computed from catalogue {CATALOGUE_SHA256[:8]}")
    return {str(k): Decimal(str(v)) for k, v in (doc.get("bounds") or {}).items()}


def catalogue_id(model: str) -> str:
    """*model* as the catalogue names it: the route suffix (``@openrouter``) off."""
    return str(model or "").strip().split("@", 1)[0]


def call_worst_case_usd(model: str) -> Decimal:
    """One request's worst case for *model*; raises :class:`NoBound` when the catalogue does not bound it."""
    key = catalogue_id(model)
    bound = _table().get(key)
    if bound is None or bound <= 0:
        raise NoBound(f"model {key!r} has no worst case in catalogue {CATALOGUE_SHA256[:8]}")
    return bound


def worst_of(models: Iterable[str]) -> Decimal:
    """The largest worst case over *models* (each must be bounded)."""
    bounds = [call_worst_case_usd(m) for m in models]
    if not bounds:
        raise NoBound("no model given")
    return max(bounds)
