"""Outcome timing, inferred regimes and late verdicts (``UNIFY_MEMORY_REGIME=inferred``).

With the switch on, no episode's regime is defaulted. When a request is recorded, its regime is inferred from the
signals that arrived for it within the request (:func:`..signals.infer_regime`: by source and timing only), from
the implicit evidence its use record holds (an error, a stated condition's result), and, when nothing arrived yet,
from the run's own history (:func:`..signals.run_prior`); the evidence store keeps the regime with what it rests on
and logs every change (:meth:`..evidence.EvidenceStore.record_regime`).

A verdict may also arrive after its episode ended: a later request's session takes a checker line naming the
earlier episode by its id (``{"checker": {"label": "fail"}, "episode": "<id>"}``, never by position), and the
request posts it when it is recorded (:func:`post_late`). The earlier episode's regime is inferred again with the
new signal (a change is logged), the signal is noted with ``timing: after_episode``, and the captures judged on that
episode are judged again (:func:`..captures.rejudge_episode`). Item records are recomputed from the evidence at
the next pass (:func:`.consolidate.after_passes`), which the attention ledger makes due: every signal's arrival is a
row of :data:`ARRIVALS_FILE` (ids and labels only), from which the ledger creates its ``late_evidence`` entries.

Nothing here reads a world, a request's text or a stream position; a signal is attached to an episode by its id.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..signals import (
    AFTER_EPISODE,
    IN_EPISODE,
    Signal,
    infer_regime,
    post_signal,
    run_prior,
    timing_of,
    valid_about,
)

#: Every signal's arrival, one JSON object per line, under the harness-only state directory.
ARRIVALS_FILE = "signal-arrivals.jsonl"
#: An episode id, in the build's one grammar for episode ids (a request names its episode
#: ``%Y%m%dT%H%M%S-<8 hex>``; a segment ``<episode>.s<n>``).
EPISODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
#: The checker labels a late verdict may carry.
LABELS = ("pass", "fail")


def now() -> str:
    """UTC now, ISO-8601 with milliseconds and a ``Z``."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _n(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def implicit_evidence(memory_use: Any) -> bool:
    """Whether an episode's use record holds implicit evidence: a library call that raised (an error), or a
    stated result condition's result (``holds`` true or false) or a correction on a call. Counts and flags only."""
    use = memory_use if isinstance(memory_use, dict) else {}
    rows = use.get("items") if isinstance(use.get("items"), dict) else {}
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        runtime = row.get("runtime") if isinstance(row.get("runtime"), dict) else {}
        if _n(row.get("errored")) or _n(runtime.get("error")):
            return True
        for o in row.get("outcomes") or []:
            if isinstance(o, dict) and (
                isinstance(o.get("holds"), bool) or o.get("outcome") == "error" or o.get("corrected") is True
            ):
                return True
    return False


def infer(evidence: Any, eid: str, signals: Iterable[Signal], memory_use: Any) -> tuple[str, str]:
    """``(regime, basis)`` of episode *eid* from *signals* (every signal that arrived for it) and its use record,
    with the run's other episodes as the prior (:func:`..signals.run_prior`)."""
    return infer_regime(
        signals,
        implicit_evidence=implicit_evidence(memory_use),
        prior=run_prior(evidence.inferred_regimes(exclude=eid)),
    )


def timed(sig: Signal, arrived_at: str, episode_ended_at: str, about: str = "") -> Signal:
    """*sig* with its arrival, its episode's end, its timing (:func:`..signals.timing_of`) and what it is about."""
    sig.arrived_at = arrived_at
    sig.episode_ended_at = episode_ended_at
    sig.timing = timing_of(arrived_at, episode_ended_at)
    sig.about = about
    return sig


def log_arrival(state_dir: Path | str, sig: Signal, *, regime_before: str | None, regime_after: str) -> None:
    """Append one arrival row (ids, labels and times only) to :data:`ARRIVALS_FILE`, with ``O_APPEND`` and an
    ``fsync``."""
    row = {
        "v": 1,
        "signal_id": sig.signal_id,
        "episode_id": sig.episode_id,
        "source": sig.source,
        "label": sig.label,
        "visible_to_actor": bool(sig.visible_to_actor),
        "refers_to": sig.refers_to,
        "arrived_at": sig.arrived_at,
        "episode_ended_at": sig.episode_ended_at,
        "timing": sig.timing,
        "about": sig.about,
        "regime_before": regime_before,
        "regime_after": regime_after,
        "at": now(),
    }
    path = Path(state_dir) / ARRIVALS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def read_arrivals(state_dir: Path | str) -> list[dict]:
    """Every arrival row, oldest first; a torn or malformed line is skipped."""
    path = Path(state_dir) / ARRIVALS_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("v") == 1:
            out.append(row)
    return out


def late_entry(raw_label: Any, episode: Any, about: Any, *, current: str, exists: Any) -> tuple[dict | None, str]:
    """A late checker line checked: ``(entry, "")`` with ``{"episode", "label", "about", "at"}``, or ``(None, why)``.
    *episode* must be another recorded episode's id (*exists* tells), *about* empty or one ref of
    :data:`..signals.ABOUT`. The answer never quotes the line."""
    if raw_label not in LABELS:
        return None, 'a checker line is exactly {"label": "pass"} or {"label": "fail"}'
    if not isinstance(episode, str) or not EPISODE_ID.fullmatch(episode):
        return None, "episode must be a recorded episode's id"
    if episode == current:
        return None, "that is this request's own episode: send the line without episode"
    about = "" if about is None else about
    if not valid_about(about):
        return None, "about must be action:<n>, item:<id> or capture:<16 hex digits>"
    if not exists(episode):
        return None, "no recorded episode has that id"
    return {"episode": episode, "label": raw_label, "about": about, "at": now()}, ""


def post_late(
    stores: Any,
    state_dir: Path | str,
    entry: dict,
    n: int,
    from_episode: str,
    *,
    lookup: Any,
) -> dict:
    """Post one late verdict (:func:`late_entry`) on its earlier episode: the episode's regime inferred again with
    it (a change logged with the signal's id), the signal noted on the episode's commit with ``timing``
    ``after_episode`` (a verdict the bed showed the actor, so ``visible_to_actor``), its arrival row written, and the
    captures judged on that episode judged again. ``about: action:<n>`` makes it refer to that action of its episode;
    otherwise it refers to the episode as a whole. Returns ``{"signal_id", "episode_id", "regime_before",
    "regime_after", "captures"}``."""
    from .. import captures

    eid = entry["episode"]
    ep = lookup.episode(eid)
    sha, _ = stores.evidence.episode_ref(eid)
    about = entry.get("about") or ""
    refers_to = eid
    if about.startswith("action:"):
        refers_to = f"{eid}/actions/{int(about.split(':', 1)[1])}"
    sig = Signal(
        f"{eid}.late.{from_episode}.{n}",
        eid,
        "checker",
        entry["label"],
        entry["at"],
        refers_to=refers_to,
        visible_to_actor=True,
    )
    timed(sig, entry["at"], str(getattr(ep, "ended_at", "") or ""), about)
    if sig.timing != AFTER_EPISODE:
        raise ValueError("a late verdict arrived before its episode ended")
    known = list(stores.evidence.signals_for(eid))
    regime, basis = infer(stores.evidence, eid, [*known, sig], getattr(ep, "memory_use", None))
    before = stores.evidence.record_regime(eid, regime, basis, entry["at"], signal_id=sig.signal_id)
    sig.regime = regime
    post_signal(sig, stores.episodes, sha, stores.evidence)
    log_arrival(state_dir, sig, regime_before=before, regime_after=regime)
    changed = captures.rejudge_episode(
        captures.store_dir(state_dir),
        eid,
        stores.evidence.signals_for(eid),
        ep,
        regime,
    )
    return {
        "signal_id": sig.signal_id,
        "episode_id": eid,
        "regime_before": before,
        "regime_after": regime,
        "captures": changed,
    }


__all__ = [
    "AFTER_EPISODE",
    "ARRIVALS_FILE",
    "IN_EPISODE",
    "implicit_evidence",
    "infer",
    "late_entry",
    "log_arrival",
    "now",
    "post_late",
    "read_arrivals",
    "timed",
]
