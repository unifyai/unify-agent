"""Signals and the job-level promotion rule (spec §8, D7/D8/D15)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .evidence import EvidenceStore
from .gitio import Repo

REGIME_SOURCES: dict[str, frozenset[str]] = {
    "dense": frozenset(
        {"checker", "environment", "provenance", "recurrence", "reader"},
    ),
    "sparse": frozenset(
        {"checker", "environment", "provenance", "recurrence", "reader"},
    ),
    "implicit": frozenset({"environment", "provenance", "recurrence", "reader"}),
    "none": frozenset({"environment"}),
}
_SUPPORT = {("checker", "pass"), ("provenance", "support")}
_SUPPORT_READER = {("reader", "accept")}
_CONTRARY = {
    ("checker", "fail"),
    ("provenance", "correct"),
    ("reader", "correct"),
    ("reader", "re_ask"),
    ("recurrence", "re_ask"),
}


class SignalMasked(ValueError):
    pass


@dataclass
class Signal:
    signal_id: str
    episode_id: str
    source: str
    label: str
    ts: str
    refers_to: str = ""
    regime: str = "dense"
    revealed: bool = True
    reveal_p: str | None = None


def post_signal(
    sig: Signal,
    episodes: Repo,
    commit_sha: str,
    evidence: EvidenceStore,
) -> None:
    try:
        ep_regime = evidence.regime_of(sig.episode_id)
    except KeyError:
        raise ValueError(f"unknown episode {sig.episode_id}") from None
    if sig.regime != ep_regime:
        raise SignalMasked(
            f"signal regime {sig.regime} differs from episode regime {ep_regime}",
        )
    if ep_regime not in REGIME_SOURCES:
        raise SignalMasked(f"unknown regime {ep_regime}")
    if sig.source not in REGIME_SOURCES[ep_regime]:
        raise SignalMasked(
            f"{sig.source} signals are not observable in the {ep_regime} regime",
        )
    episodes.add_note(commit_sha, json.dumps(asdict(sig), sort_keys=True))
    evidence.add_signal(sig)


def job_item_status(
    item: str,
    evidence: EvidenceStore,
    *,
    reader_promotes: bool,
) -> str:
    support_kinds = _SUPPORT | (_SUPPORT_READER if reader_promotes else set())
    supporting = set()
    for eid in evidence.item_episodes(item):
        for s in evidence.signals_for(eid):
            if (s.source, s.label) in _CONTRARY:
                return "hidden"
            if (s.source, s.label) in support_kinds:
                supporting.add(eid)
    return "promotable" if len(supporting) >= 2 else "insufficient"


def tainted_items(eid: str, evidence: EvidenceStore) -> list[str]:
    return evidence.items_citing(eid)
