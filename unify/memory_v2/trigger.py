"""When consolidation runs, and which recorded episodes a pass covers.

Two modes:

* ``"batched"`` (the default): a pass fires once the experience recorded since the last batched pass
  reaches ``experience_budget`` tokens (E, default 150,000; :mod:`.experience` counts each trajectory's
  content recorded once, never provider usage). One :class:`PassRequest` of kind ``"batched"`` covers every
  channel with new evidence: ``channel=None`` and every episode since the last pass. Nothing is keyed on a
  stream, a session or its end. An optional idle trigger (off by default) fires on :meth:`Trigger.idle`
  when at least ``idle_min`` trajectories are new and ``idle_after_s`` seconds have passed since the
  timestamp the caller gave the last :meth:`Trigger.after_episode`. A queued drift channel rides with the
  next batched pass (it does not fire one on its own). The writer's budget of one pass is E x ``usd_per_token``
  (:meth:`Trigger.pass_budget_usd`); the caller passes it to ``PassConfig.max_usd``. Under
  ``UNIFY_MEMORY_LIBRARY=on`` E is ``UNIFY_MEMORY_LIBRARY_E`` (default 100,000).

  **Admission** (optional, the episode-end fork on): an episode enters a batch only once its fork has settled
  (``admission.settled(eid)``, :class:`.integration.fork.Admission`). The others are held: the pass request lists
  them in ``held``, and :meth:`Trigger.mark_done` hands them to ``admission.set_rolled`` so the next pass sees them
  again, whatever came after them. A pass is due when the experience of the episodes it admits reaches E. This keys
  on each episode's own signal, never on the stream.
* ``"d6"`` (the earlier per-channel cadence, kept for offline comparison): after each episode, an incremental pass per
  channel the episode touched (or a drift channel queued), plus a maintenance pass every
  ``maintenance_every`` requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from .evidence import EvidenceStore

MODES = ("batched", "d6")
EXPERIENCE_BUDGET = 150_000
# With the library mode on, E is 100k recorded tokens; the default above stays the earlier mode's.
EXPERIENCE_BUDGET_LIBRARY = 100_000
USD_PER_TOKEN = Decimal("0.00000073")
# The cursor of the batched mode in the evidence store's cursor table (no channel key starts with "@").
BATCHED_CURSOR = "@batched"


@dataclass
class PassRequest:
    kind: str
    channel: str | None
    episodes: list[str]
    lift: bool
    experience_tokens: int | None = (
        None  # batched passes: the experience the pass covers
    )
    #: batched passes under admission: pending episodes whose fork has not settled; they roll to the next pass
    held: list[str] = field(default_factory=list)


def admit_settled(
    staging_root: Path | str,
    episode_ids: list[str],
) -> tuple[list[str], list[str]]:
    """The admission rule, as a pure function: ``(admitted, waiting)`` in the given order. An episode is settled once
    its staging dir under *staging_root* (``<state>/memory-staging``) holds its status file (``fork.json`` with any
    status, ``"timeout"`` and ``"lost"`` included). An episode with no staging dir had no fork and is admitted (with
    the fork on, the dir is reserved before the episode is recorded). It reads files only; recording ``"lost"`` or
    ``"timeout"`` for a fork that can no longer settle itself is :func:`.integration.fork.settled`'s work."""
    from .staging import STATUS_FILES

    admitted, waiting = [], []
    root = Path(staging_root)
    for eid in episode_ids:
        d = root / eid
        if not os.path.isdir(d) or os.path.islink(d):
            admitted.append(eid)
        elif any(os.path.lexists(d / name) for name in STATUS_FILES):
            admitted.append(eid)
        else:
            waiting.append(eid)
    return admitted, waiting


class Admission(Protocol):
    def settled(self, eid: str) -> bool: ...

    def rolled(self) -> list[str]: ...

    def set_rolled(self, eids: list[str]) -> None: ...


class Trigger:
    def __init__(
        self,
        evidence: EvidenceStore,
        maintenance_every: int = 25,
        lift_min: int = 2,
        *,
        mode: str = "batched",
        experience_budget: int = EXPERIENCE_BUDGET,
        usd_per_token: Decimal | str = USD_PER_TOKEN,
        idle_after_s: float | None = None,
        idle_min: int = 3,
        admission: Admission | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"trigger mode must be one of {MODES}, got {mode!r}")
        if int(experience_budget) <= 0:
            raise ValueError("experience_budget must be a positive number of tokens")
        self.ev, self.every, self.lift_min = evidence, maintenance_every, lift_min
        self.mode = mode
        self.budget = int(experience_budget)
        self.usd_per_token = Decimal(str(usd_per_token))
        self.idle_after_s, self.idle_min = idle_after_s, int(idle_min)
        self._drift: set[str] = set()
        self._last_at: float | None = None
        self.admission = admission

    def queue_drift(self, channel: str) -> None:
        self._drift.add(channel)

    def pass_budget_usd(self) -> Decimal:
        """The writer's budget of one batched pass: E x USD per token, a Decimal (pass as ``max_usd``)."""
        return Decimal(self.budget) * self.usd_per_token

    # -- batched ----------------------------------------------------------------------------------------
    def pending(self) -> list[tuple[str, int]]:
        """(episode id, experience tokens) recorded since the last batched pass, after the episodes earlier passes
        rolled over (under admission)."""
        rows = self.ev.experience_since(self.ev.cursor(BATCHED_CURSOR))
        if self.admission is None:
            return rows
        seen = {e for e, _ in rows}
        carried = []
        for eid in self.admission.rolled():
            if eid in seen:
                continue
            try:
                tokens = self.ev.experience_of(eid)[0]
            except KeyError:
                tokens = 0
            carried.append((eid, tokens))
            seen.add(eid)
        return carried + rows

    def _batched(self) -> PassRequest:
        rows, held = self.pending(), []
        if self.admission is not None:
            ready = [self.admission.settled(e) for e, _ in rows]
            held = [e for (e, _), ok in zip(rows, ready) if not ok]
            rows = [r for r, ok in zip(rows, ready) if ok]
        eids = [e for e, _ in rows]
        return PassRequest(
            "batched",
            None,
            eids,
            len(eids) >= self.lift_min,
            sum(t for _, t in rows),
            held,
        )

    def after_episode(
        self,
        episode_id: str,
        at: float | None = None,
    ) -> list[PassRequest]:
        """The passes due once *episode_id* is indexed; *at* (seconds, any clock) feeds the idle trigger."""
        if at is not None:
            self._last_at = float(at)
        if self.mode == "d6":
            return self._d6(episode_id)
        self.ev.seq_of(episode_id)  # KeyError for an episode that was never indexed
        req = self._batched()
        return [req] if req.episodes and req.experience_tokens >= self.budget else []

    def idle(self, now: float) -> list[PassRequest]:
        """The optional idle trigger (batched mode, ``idle_after_s`` set; else nothing)."""
        if self.mode != "batched" or self.idle_after_s is None or self._last_at is None:
            return []
        if float(now) - self._last_at < self.idle_after_s:
            return []
        req = self._batched()
        return [req] if len(req.episodes) >= self.idle_min else []

    # -- d6 ---------------------------------------------------------------------------------------------
    def _d6(self, episode_id: str) -> list[PassRequest]:
        seq = self.ev.seq_of(episode_id)
        channels = set(self.ev.channels_of(episode_id)) | self._drift
        reqs = []
        for ch in sorted(channels):
            pending = self.ev.episode_ids_since(ch, self.ev.cursor(ch))
            if pending or ch in self._drift:
                reqs.append(
                    PassRequest(
                        "incremental",
                        ch,
                        pending,
                        len(pending) >= self.lift_min,
                    ),
                )
        if seq % self.every == 0:
            reqs.append(PassRequest("maintenance", None, [], False))
        return reqs

    def mark_done(self, req: PassRequest) -> None:
        if req.kind == "batched":
            covered = [*req.episodes, *req.held]
            if covered:
                cursor = self.ev.cursor(BATCHED_CURSOR)
                self.ev.set_cursor(
                    BATCHED_CURSOR,
                    max(cursor, *(self.ev.seq_of(e) for e in covered)),
                )
            if self.admission is not None:
                self.admission.set_rolled(list(req.held))
            self._drift.clear()
            return
        if req.channel is not None:
            if req.episodes:
                self.ev.set_cursor(
                    req.channel,
                    max(self.ev.seq_of(e) for e in req.episodes),
                )
            self._drift.discard(req.channel)
