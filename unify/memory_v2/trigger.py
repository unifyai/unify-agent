"""When consolidation runs (spec §F1, D6; the lead's batched cadence of 8 Oct).

Two modes:

* ``"batched"`` (the default): a pass fires once the experience recorded since the last batched pass
  reaches ``experience_budget`` tokens (E, default 150,000; :mod:`.experience` counts each trajectory's
  content recorded once, never provider usage). One :class:`PassRequest` of kind ``"batched"`` covers every
  channel with new evidence: ``channel=None`` and every episode since the last pass. Nothing is keyed on a
  stream, a session or its end. An optional idle trigger (off by default) fires on :meth:`Trigger.idle`
  when at least ``idle_min`` trajectories are new and ``idle_after_s`` seconds have passed since the
  timestamp the caller gave the last :meth:`Trigger.after_episode`. A queued drift channel rides with the
  next batched pass (it does not fire one on its own). The Sol budget of one pass is E x ``usd_per_token``
  (:meth:`Trigger.pass_budget_usd`); the caller passes it to ``PassConfig.max_usd``.
* ``"d6"`` (v0 D6 as frozen; kept for offline comparison): after each episode, an incremental pass per
  channel the episode touched (or a drift channel queued), plus a maintenance pass every
  ``maintenance_every`` requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .evidence import EvidenceStore

MODES = ("batched", "d6")
EXPERIENCE_BUDGET = 150_000
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

    def queue_drift(self, channel: str) -> None:
        self._drift.add(channel)

    def pass_budget_usd(self) -> Decimal:
        """The Sol budget of one batched pass: E x USD per token, a Decimal (pass as ``max_usd``)."""
        return Decimal(self.budget) * self.usd_per_token

    # -- batched ----------------------------------------------------------------------------------------
    def pending(self) -> list[tuple[str, int]]:
        """(episode id, experience tokens) recorded since the last batched pass."""
        return self.ev.experience_since(self.ev.cursor(BATCHED_CURSOR))

    def _batched(self) -> PassRequest:
        rows = self.pending()
        eids = [e for e, _ in rows]
        return PassRequest(
            "batched",
            None,
            eids,
            len(eids) >= self.lift_min,
            sum(t for _, t in rows),
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
        return [req] if req.experience_tokens >= self.budget else []

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
            if req.episodes:
                self.ev.set_cursor(
                    BATCHED_CURSOR,
                    max(self.ev.seq_of(e) for e in req.episodes),
                )
            self._drift.clear()
            return
        if req.channel is not None:
            if req.episodes:
                self.ev.set_cursor(
                    req.channel,
                    max(self.ev.seq_of(e) for e in req.episodes),
                )
            self._drift.discard(req.channel)
