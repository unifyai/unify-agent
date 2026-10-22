"""Signals and the job-level promotion rule.

A signal may also say when it arrived (``arrived_at``), when its episode ended (``episode_ended_at``), whether it
came within the episode or after it (``timing``, derived from the two, :func:`timing_of`) and what it bears on
(``about``: ``action:<n>``, ``item:<id>`` or ``capture:<id>``). These fields are set only under
an inferred regime; a signal without them is written as before, byte for byte (:func:`note_text`).

Under that switch an episode's regime is never defaulted: :func:`infer_regime` infers it from the signals that
arrive for that episode, by their source and timing, with the run's own history (:func:`run_prior`) as the prior
when nothing has arrived yet. It never reads a world, a request or a stream position.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter
from collections.abc import Iterable
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
    # The bed showed the actor this same verdict (a checker signal posted from a
    # ``{"checker": ...}`` line). Only such checker signals may be used by the writer, the gate and the
    # lifecycle; a grader the actor never sees (``{"outcome": ...}``) stays False.
    visible_to_actor: bool = False
    # (empty otherwise): when the harness recorded the signal, when its episode
    # ended, ``in_episode`` or ``after_episode`` (:func:`timing_of`), and what it bears on (:data:`ABOUT`).
    arrived_at: str = ""
    episode_ended_at: str = ""
    timing: str = ""
    about: str = ""


#: When a signal arrived relative to its episode's end.
IN_EPISODE = "in_episode"
AFTER_EPISODE = "after_episode"
#: What a signal may say it bears on: an action of its episode, a library item or a captured procedure.
ABOUT = re.compile(r"action:[0-9]{1,9}|item:[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,199}|capture:[0-9a-f]{16}")
#: The fields only an inferred-regime signal carries; a signal without them is noted as before.
TIMING_FIELDS = ("arrived_at", "episode_ended_at", "timing", "about")


def note_text(sig: Signal) -> str:
    """The signal's note line: ``visible_to_actor`` appears only when True, and the timing fields only when set, so
    every other signal's note is byte-identical to the build without them."""
    row = asdict(sig)
    if not row["visible_to_actor"]:
        del row["visible_to_actor"]
    for key in TIMING_FIELDS:
        if not row[key]:
            del row[key]
    return json.dumps(row, sort_keys=True)


def _instant(ts: str) -> dt.datetime:
    """An ISO time (``Z`` read as UTC) as an aware datetime; ValueError when it is not one."""
    if not isinstance(ts, str) or not ts:
        raise ValueError(f"not an ISO time: {ts!r}"[:120])
    t = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if t.tzinfo is None:
        raise ValueError(f"an ISO time without a zone: {ts!r}"[:120])
    return t


def timing_of(arrived_at: str, episode_ended_at: str) -> str:
    """``in_episode`` when the signal arrived at or before its episode ended, else ``after_episode``; both times
    are ISO times with a zone (ValueError otherwise)."""
    return IN_EPISODE if _instant(arrived_at) <= _instant(episode_ended_at) else AFTER_EPISODE


def valid_about(about: object) -> bool:
    """Whether *about* is empty or one ref of :data:`ABOUT`."""
    return about == "" or (isinstance(about, str) and ABOUT.fullmatch(about) is not None)


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
    episodes.add_note(commit_sha, note_text(sig))
    evidence.add_signal(sig)


def tainted_items(eid: str, evidence: EvidenceStore) -> list[str]:
    return evidence.items_citing(eid)


# --- the inferred regime -------------------------------------------------------

#: The regimes, in the order a tie in the run's history is broken (the one that observes more first).
REGIMES = ("dense", "sparse", "implicit", "none")
#: Signals that correct the actor after the fact: the implicit regime's own evidence.
CORRECTIONS = frozenset({("reader", "correct"), ("reader", "re_ask"), ("recurrence", "re_ask")})
#: How an episode's regime was inferred: from its own signals or implicit evidence, from the run's history (nothing
#: of its own arrived yet), or from nothing at all.
BASIS_SIGNALS = "signals"
BASIS_PRIOR = "prior"
BASIS_NOTHING = "nothing"
REGIME_BASES = (BASIS_SIGNALS, BASIS_PRIOR, BASIS_NOTHING)


def _per_action(s: Signal) -> bool:
    return f"{s.episode_id}/actions/" in str(s.refers_to or "")


def infer_regime(
    signals: Iterable[Signal],
    *,
    implicit_evidence: bool = False,
    prior: str | None = None,
) -> tuple[str, str]:
    """``(regime, basis)`` of one episode from the signals that arrived for it, by source and timing only:

    * ``dense``: a verdict the actor saw (a visible ``checker`` signal) came within the episode after an answered
      action: one that names the action it judged, or two or more within the episode;
    * ``sparse``: an outcome came for the episode as a whole and no per-action verdict did: one visible verdict
      within it, a visible verdict after it ended, or a ``provenance`` support or correction;
    * ``implicit``: only errors, stated conditions (*implicit_evidence*: the episode's use record shows an error
      or a ``holds`` result) or corrections (:data:`CORRECTIONS`) arrived;
    * otherwise nothing arrived: the run's history (*prior*, :func:`run_prior`) when it has one, else ``none``.

    A verdict the actor never saw (a hidden grader's) decides nothing, as it is never evidence. *basis* says which
    rule gave the regime (:data:`REGIME_BASES`)."""
    sigs = list(signals)
    seen = [s for s in sigs if s.source == "checker" and s.visible_to_actor is True]
    within = [s for s in seen if s.timing != AFTER_EPISODE]
    if any(_per_action(s) for s in within) or len(within) >= 2:
        return "dense", BASIS_SIGNALS
    if seen or any(s.source == "provenance" for s in sigs):
        return "sparse", BASIS_SIGNALS
    if implicit_evidence or any((s.source, s.label) in CORRECTIONS for s in sigs):
        return "implicit", BASIS_SIGNALS
    if prior in REGIMES and prior != "none":
        return prior, BASIS_PRIOR
    return "none", BASIS_NOTHING


def run_prior(inferred: Iterable[tuple[str, str]]) -> str | None:
    """The run's prior regime: the most common regime among its earlier episodes whose own signals gave one
    (*inferred*: ``(regime, basis)`` per episode; a regime that came from the prior itself is never counted, so the
    prior never feeds itself), ties going to the regime that observes more (:data:`REGIMES` order); None when no
    episode has one."""
    counts = Counter(r for r, basis in inferred if basis == BASIS_SIGNALS and r in REGIMES)
    if not counts:
        return None
    top = max(counts.values())
    return next(r for r in REGIMES if counts.get(r) == top)
