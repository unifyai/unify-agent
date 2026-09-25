"""Query-time activation for the stored-function library.

A function's standing in the library is not a curator's verdict but a
memory trace: every use strengthens it, disuse decays it, and retrieval
ranks on similarity *and* standing — the way a colleague's own procedures
come to mind. Three properties fall out, and they are the design:

* **Deletion is unnecessary for hygiene.** What a crowded library costs is
  scope — search slots, schema surface, attention — not storage. A function
  whose activation has decayed below the scope threshold stops surfacing
  (and stops costing anything) while its row remains, so a revival is a
  lookup rather than a re-derivation. Nothing here mutates rows on a
  timer: activation is computed at query time from the usage trace, so
  "dropping out of scope" is passive arithmetic, not a background job.

* **Creation is an activation event** (the newborn grace). A brand-new
  function has no usage; if standing counted only calls, novelty would be
  a death spiral — never surfaced, never called, decayed. Storing the
  function IS its first use, giving it a window to earn its second.

* **Decay is relative to each function's own rhythm.** A single global
  half-life murders the quarterly-report function every quarter. Here the
  decay clock scales with the median interval between a function's own
  uses: a skill exercised every 90 days is still "recent" at day 60,
  while a daily skill dormant for a month has plainly lapsed.

Every number lives in :class:`ActivationSettings` — a decision, never a
call-site constant. Ranking caps the activation term (``similarity_floor``)
so semantic similarity dominates and standing acts as the tiebreaker;
without the cap, usage-weighted retrieval feeds usage counts and an
entrenched mediocre function shadows a better newcomer forever. The other
half of that guard lives in the supersede path, which transfers the old
function's usage trace to its replacement, so an upgrade inherits standing
instead of restarting the popularity contest from zero.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable, Sequence

from pydantic import BaseModel, Field


class ActivationSettings(BaseModel):
    """Knobs for the memory-trace arithmetic.

    Attributes:
        similarity_floor: Fraction of the ranking score that similarity
            keeps regardless of standing. 0.7 means activation can swing at
            most 30% of the score — a tiebreaker, never the headline.
        newborn_floor: Activation a freshly created, never-called function
            holds while its creation is still "recent" — the grace that
            lets novelty surface long enough to earn a first call.
        strength_saturation_calls: Call count at which usage strength
            saturates; log-scaled below it, flat above.
        rhythm_default_days: Decay clock for functions with fewer than two
            calls (no observable rhythm yet).
        rhythm_min_days / rhythm_max_days: Clamp on the observed median
            inter-use interval, so one burst or one long gap cannot set a
            pathological clock.
        rhythm_slack: Multiplier on the rhythm before decay bites — 3.0
            means a function is at ~e⁻¹ standing after three of its own
            typical intervals without use.
        scope_threshold: Activation below which a function stops surfacing
            in search (out of scope). Its row persists; explicit lookups
            and dormant-inclusive searches still reach it.
        recent_calls_kept: How many recent call timestamps the usage trace
            retains for rhythm estimation (the total count is unbounded).
    """

    enabled: bool = Field(
        True,
        description=(
            "Master switch over the whole subsystem: off means no usage "
            "recording, no activation ranking, no scope dropout — the store "
            "behaves exactly as it did before activation existed."
        ),
    )
    similarity_floor: float = Field(0.7, ge=0.0, le=1.0)
    newborn_floor: float = Field(0.5, ge=0.0, le=1.0)
    strength_saturation_calls: int = Field(20, ge=1)
    rhythm_default_days: float = Field(14.0, gt=0.0)
    rhythm_min_days: float = Field(1.0, gt=0.0)
    rhythm_max_days: float = Field(180.0, gt=0.0)
    rhythm_slack: float = Field(3.0, gt=0.0)
    scope_threshold: float = Field(0.05, ge=0.0, le=1.0)
    recent_calls_kept: int = Field(32, ge=2)
    search_overfetch_factor: int = Field(
        3,
        ge=1,
        description=(
            "Search fetches this multiple of the requested n before the "
            "activation pass ranks and scope-filters, so dropped rows do "
            "not leave the caller short."
        ),
    )
    search_overfetch_cap: int = Field(
        50,
        ge=1,
        description="Absolute ceiling on the overfetched candidate pool.",
    )


def _parse_when(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def _rhythm_days(
    call_times: Sequence[datetime],
    settings: ActivationSettings,
) -> float:
    """The function's own cadence: median gap between consecutive uses."""
    if len(call_times) < 2:
        return settings.rhythm_default_days
    ordered = sorted(call_times)
    gaps = [
        (b - a).total_seconds() / 86400.0
        for a, b in zip(ordered, ordered[1:])
        if (b - a).total_seconds() > 0
    ]
    if not gaps:
        return settings.rhythm_default_days
    gaps.sort()
    mid = len(gaps) // 2
    median = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2.0
    return min(max(median, settings.rhythm_min_days), settings.rhythm_max_days)


def activation(
    *,
    now: datetime,
    created_at: object,
    call_count: int,
    recent_calls: Iterable[object] = (),
    settings: ActivationSettings,
) -> float:
    """Standing in [0, 1] from the usage trace, computed at query time.

    ``recency × strength``, where recency decays against the function's own
    rhythm and strength log-saturates with total calls. Creation counts as
    the first activation event, and a never-called function's strength is
    the newborn floor rather than zero — grace, not immortality: with no
    second use, the same decay that retires veterans retires it too.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    calls = [t for t in (_parse_when(c) for c in recent_calls) if t is not None]
    events = list(calls)
    created = _parse_when(created_at)
    if created is not None:
        events.append(created)
    if not events:
        # No trace at all (legacy row): treat as a newborn created now, so
        # pre-activation libraries surface until real usage data accrues.
        return settings.newborn_floor
    last = max(events)
    age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    rhythm = _rhythm_days(calls, settings)
    recency = math.exp(-age_days / (settings.rhythm_slack * rhythm))
    n = max(0, int(call_count))
    strength = math.log1p(n) / math.log1p(settings.strength_saturation_calls)
    strength = min(1.0, strength)
    floor = settings.newborn_floor
    return recency * (floor + (1.0 - floor) * strength)


def rank_score(
    similarity: float,
    standing: float,
    settings: ActivationSettings,
) -> float:
    """Similarity dominates; standing is the capped tiebreaker."""
    floor = settings.similarity_floor
    return similarity * (floor + (1.0 - floor) * max(0.0, min(1.0, standing)))


def in_scope(standing: float, settings: ActivationSettings) -> bool:
    """Whether the function still comes to mind at all."""
    return standing >= settings.scope_threshold


def merged_usage(
    old: dict | None,
    new: dict | None,
    settings: ActivationSettings,
) -> dict:
    """Combine two usage traces — the supersede inheritance.

    Calls sum, recent timestamps union (bounded), last-called takes the
    max: the replacement stands where its predecessor stood.
    """
    old = dict(old or {})
    new = dict(new or {})
    calls = int(old.get("calls") or 0) + int(new.get("calls") or 0)
    recents = [
        t
        for t in (
            _parse_when(x)
            for x in [
                *(old.get("recent_calls") or []),
                *(new.get("recent_calls") or []),
            ]
        )
        if t is not None
    ]
    recents = sorted(set(recents))[-settings.recent_calls_kept :]
    lasts = [
        t
        for t in (
            _parse_when(old.get("last_called_at")),
            _parse_when(new.get("last_called_at")),
        )
        if t is not None
    ]
    return {
        "calls": calls,
        "last_called_at": max(lasts).isoformat() if lasts else None,
        "recent_calls": [t.isoformat() for t in recents],
        "search_hits": int(old.get("search_hits") or 0)
        + int(new.get("search_hits") or 0),
    }
