"""Outcome-driven trust per signal regime: what one use of a stored procedure counts as, its track record per kind of
signal, and the status and evidence level its outcomes give. Deterministic and pure: no model call, no file, no
environment text.

A *use* is one call of a stored procedure (a library item or a captured attempt) in one episode. What it counts as
depends on the regime the episode declared (:data:`.signals.REGIME_SOURCES`):

* ``dense`` (a verdict after each action): the verdict the actor saw after the action the use answered. Answered and
  accepted is a clean use that carries a verdict; answered and rejected is a failed use, even when the episode later
  succeeded another way.
* ``sparse`` (an outcome at the end): the episode-end outcome plus the intermediate signals (errors, refusals, the
  procedure's own ``holds`` condition). A use in an episode that ended positive is a clean use that carries a positive
  end; an answered use in an episode that ended contrary is a failed use.
* ``implicit``: implicit signals only. A use that returned without error, whose ``holds`` was not false, with no
  correction after it and no disagreement, is clean; an error, ``holds`` false, a correction after an answered use, or
  disagreement with the actor's own final result is a failed use.
* ``none``: as ``implicit``, without corrections, which that regime cannot observe.

In every regime an error and a false ``holds`` are failed uses, and a refusal (``MemoryInputError``: an input the
procedure was not built for) is never a failure. A use whose result was not handed back never gets a verdict and is
never failed by the episode's end. A verdict the actor never saw is never given here: the caller passes only visible
verdicts (:func:`signal_outcome`).

**Statuses** (:func:`decide`), from the uses of the current version only:

* ``retired``: failed uses in at least :data:`.lifecycle.MIN_EPISODES` episodes;
* ``adopted``: clean uses in at least :data:`.lifecycle.K_STABLE` episodes and no failed use. Where any clean use is in
  a ``dense`` or ``sparse`` episode, at least one clean use must carry an accepted verdict or a positive end. The
  basis is ``verdicts`` when one does, else ``implicit``; an ``implicit`` basis never claims correctness;
* ``on-trial`` otherwise. An ``adopted`` item stays adopted until the retirement rule fires; one episode never decides.

**Evidence levels.** The gate sets ``unchecked`` and ``reproduces-its-case`` (for a capture, the harness's own
reproduction of its recorded case does). Outcomes only move the top of the scale (:func:`apply_outcome`, the one
decider, which :func:`.lifecycle.decide_status` delegates to): an item adopted on the basis of verdicts is raised to
``confirmed-by-outcomes``, and one that is retired, or no longer adopted on verdicts, drops back to
``reproduces-its-case``. Outcomes in the ``implicit`` and ``none`` regimes never confirm.

The status and level names are defined here, in one place, until the shared constants module lands; the mapping from
the lifecycle's statuses is :data:`FROM_LIFECYCLE`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

# --- names ----------------------------------------------------------------------------------------------------

#: Statuses and evidence levels: the one vocabulary (:mod:`.item_states`); ``FROM_LIFECYCLE`` maps the lifecycle's old
#: status names (``deprecated`` is the curator's retirement).
from .item_states import (  # noqa: E402
    ADOPTED,
    CONFIRMED_BY_OUTCOMES,
    FROM_LIFECYCLE,  # noqa: F401 - read as trust.FROM_LIFECYCLE
    ON_TRIAL,
    REPRODUCES_ITS_CASE,
    RETIRED,
    STATUSES,  # noqa: F401 - read as trust.STATUSES
    UNCHECKED,  # noqa: F401 - read as trust.UNCHECKED
)

#: What an adoption rests on.
BASIS_VERDICTS = "verdicts"
BASIS_IMPLICIT = "implicit"

REGIMES = ("dense", "sparse", "implicit", "none")
OUTCOMES = ("returned", "refused", "error")
VERDICTS = ("accepted", "rejected")
END_OUTCOMES = ("positive", "contrary")

CLEAN = "clean"
FAILED = "failed"
NEUTRAL = "neutral"


def _thresholds() -> tuple[int, int]:
    from .lifecycle import K_STABLE, MIN_EPISODES

    return K_STABLE, MIN_EPISODES


# --- one use --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Use:
    """One call of a stored procedure in one episode, with the signals that bear on it.

    *outcome*: ``returned``, ``refused`` (``MemoryInputError``) or ``error``. *answered*: its result was the value
    handed back. *verdict*: the visible verdict after the answered action (``accepted``/``rejected``), else None.
    *end_outcome*: the episode-end outcome (``positive``/``contrary``), else None. *holds*: the procedure's stated
    result condition on this call (None: none stated). *corrected*: a correction or re-ask followed the use.
    *agreed*: agreement with another procedure, a second run or the actor's own final result (None: nothing to
    compare).

    *answered* None means the record does not say whether the result was handed back (a use record without result
    digests). Such a use is attributed as the lifecycle always has: a contrary signal in its episode
    (*episode_negative*, in a regime with signals) fails it, and it never needs a verdict to be clean. *unknown*: a
    cell with an unknown outcome could reach it, so it is never clean (an error still counts).
    """

    episode: str
    regime: str
    outcome: str = "returned"
    answered: bool | None = False
    verdict: str | None = None
    end_outcome: str | None = None
    holds: bool | None = None
    corrected: bool = False
    agreed: bool | None = None
    episode_negative: bool = False
    unknown: bool = False

    def row(self) -> dict:
        return asdict(self)


def use_from_row(row: Mapping[str, Any]) -> Use | None:
    """A :class:`Use` from its JSON row (:meth:`Use.row`), or None for a malformed row."""
    if not isinstance(row, Mapping):
        return None
    episode, regime = row.get("episode"), row.get("regime")
    outcome = row.get("outcome", "returned")
    if not isinstance(episode, str) or regime not in REGIMES or outcome not in OUTCOMES:
        return None
    verdict = row.get("verdict") if row.get("verdict") in VERDICTS else None
    end = row.get("end_outcome") if row.get("end_outcome") in END_OUTCOMES else None
    holds = row.get("holds") if isinstance(row.get("holds"), bool) else None
    agreed = row.get("agreed") if isinstance(row.get("agreed"), bool) else None
    return Use(
        episode,
        regime,
        outcome,
        answered=row.get("answered") if isinstance(row.get("answered"), bool) else None,
        verdict=verdict,
        end_outcome=end,
        holds=holds,
        corrected=row.get("corrected") is True,
        agreed=agreed,
        episode_negative=row.get("episode_negative") is True,
        unknown=row.get("unknown") is True,
    )


def judge(use: Use) -> tuple[str, str | None, bool]:
    """``(clean | failed | neutral, why, supported)`` for one use under its episode's regime (module docstring).
    *why* names the failing signal (``error``, ``holds_false``, ``rejected``, ``contrary_end``, ``corrected``,
    ``disagreed``) or why a use does not count; *supported* is True for a clean use that carries an accepted verdict
    or a positive end."""
    regime = use.regime
    if use.outcome == "refused":
        return NEUTRAL, "refused", False
    if use.outcome == "error":
        return FAILED, "error", False
    if use.holds is False:
        return FAILED, "holds_false", False
    if use.answered is None:
        # not attributed: as the lifecycle always counted it
        if use.episode_negative and regime in ("dense", "sparse", "implicit"):
            return FAILED, "negative", False
        if use.unknown:
            return NEUTRAL, "unknown outcome", False
        return CLEAN, None, False
    if use.unknown:
        return NEUTRAL, "unknown outcome", False
    if regime == "dense":
        if use.answered and use.verdict == "rejected":
            return FAILED, "rejected", False
        return CLEAN, None, use.answered and use.verdict == "accepted"
    if regime == "sparse":
        if use.answered and use.end_outcome == "contrary":
            return FAILED, "contrary_end", False
        if use.end_outcome == "contrary":
            return NEUTRAL, "not answered in an episode that ended contrary", False
        return CLEAN, None, use.end_outcome == "positive"
    # implicit and none: implicit signals only, never a verdict
    if regime == "implicit" and use.corrected and use.answered:
        return FAILED, "corrected", False
    if use.agreed is False:
        return FAILED, "disagreed", False
    return CLEAN, None, False


# --- the track record -----------------------------------------------------------------------------------------


def empty_track() -> dict:
    return {
        "verdicts": {"accepted": 0, "rejected": 0},
        "end_outcomes": {"positive": 0, "contrary": 0},
        "intermediate": {"errors": 0, "refusals": 0, "holds_true": 0, "holds_false": 0},
        "implicit": {
            "ran_ok": 0,
            "agreed": 0,
            "disagreed": 0,
            "corrected": 0,
            "reused_episodes": 0,
        },
    }


def track(uses: Iterable[Use]) -> dict:
    """The track record per kind of signal: verdicts (answered uses only), end outcomes (answered uses only),
    intermediate signals, and implicit signals, with ``reused_episodes`` the distinct episodes in which a call
    returned. Signals a regime cannot observe are not counted (a verdict outside ``dense``, an end outcome outside
    ``dense`` and ``sparse``, a correction in ``none``)."""
    out = empty_track()
    ran: set[str] = set()
    for u in uses:
        inter, imp = out["intermediate"], out["implicit"]
        if u.outcome == "refused":
            inter["refusals"] += 1
            continue
        if u.outcome == "error":
            inter["errors"] += 1
            continue
        ran.add(u.episode)
        imp["ran_ok"] += 1
        if u.holds is True:
            inter["holds_true"] += 1
        elif u.holds is False:
            inter["holds_false"] += 1
        if u.agreed is True:
            imp["agreed"] += 1
        elif u.agreed is False:
            imp["disagreed"] += 1
        if u.corrected and u.regime != "none":
            imp["corrected"] += 1
        if u.answered and u.regime == "dense" and u.verdict in VERDICTS:
            out["verdicts"][u.verdict] += 1
        if u.answered and u.regime in ("dense", "sparse") and u.end_outcome in END_OUTCOMES:
            out["end_outcomes"][u.end_outcome] += 1
    out["implicit"]["reused_episodes"] = len(ran)
    return out


# --- status and level -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    status: str
    basis: str | None  # verdicts | implicit, for an adopted item; else None
    rule: str | None  # the failing signal that retired it
    reason: str | None
    episodes: tuple[str, ...]


def decide(uses: Iterable[Use], prior: str | None = None) -> Decision:
    """The status of a procedure from the uses of its current version (module docstring). *prior* is its status so
    far, in these names (:data:`FROM_LIFECYCLE` maps the lifecycle's): ``adopted`` stays until retirement fires.

    Retirement is checked in this order, each over :data:`.lifecycle.MIN_EPISODES` episodes: errors (rule
    ``errors``); one failing signal (``negative_signals`` for a use the record does not attribute, else the signal's
    name: ``rejected``, ``contrary_end``, ``corrected``, ``holds_false``, ``disagreed``); then failed uses of any kind
    together (``failed_uses``)."""
    k_stable, min_episodes = _thresholds()
    rows = [(u, *judge(u)) for u in uses]
    failed: dict[str, list[str]] = {}
    for u, kind, why, _ in rows:
        if kind == FAILED:
            eps = failed.setdefault(why or "failed", [])
            if u.episode not in eps:
                eps.append(u.episode)
    failed_eps = sorted({e for eps in failed.values() for e in eps})
    for why in _RETIRE_ORDER:
        eps = sorted(failed.get(why, []))
        if len(eps) >= min_episodes:
            rule, reason = _RULE[why]
            return Decision(RETIRED, None, rule, reason.format(n=len(eps)), tuple(eps))
    if len(failed_eps) >= min_episodes:
        parts = ", ".join(f"{_WHY[k]} in {len(failed[k])}" for k in sorted(failed))
        return Decision(
            RETIRED,
            None,
            "failed_uses",
            f"failed uses in {len(failed_eps)} episodes ({parts})",
            tuple(failed_eps),
        )
    clean = sorted({u.episode for u, kind, _, _ in rows if kind == CLEAN} - set(failed_eps))
    supported = sorted({u.episode for u, kind, _, s in rows if kind == CLEAN and s} - set(failed_eps))
    # the verdict requirement applies to attributed uses in regimes that give verdicts or end outcomes
    needs_support = any(
        kind == CLEAN and u.answered is not None and u.regime in ("dense", "sparse")
        for u, kind, _, _ in rows
    )
    basis = BASIS_VERDICTS if supported else BASIS_IMPLICIT
    # an adoption with no verdict-giving regime behind it says only what implicit signals show
    implicit_only = basis == BASIS_IMPLICIT and not any(
        kind == CLEAN and u.regime in ("dense", "sparse") for u, kind, _, _ in rows
    )
    adopted_reason = f"used without problems in {len(clean)} episodes" if implicit_only else None
    if prior == ADOPTED:
        return Decision(ADOPTED, basis, None, adopted_reason, tuple(clean))
    if not failed_eps and len(clean) >= k_stable:
        if needs_support and not supported:
            return Decision(
                ON_TRIAL,
                None,
                None,
                f"clean uses in {len(clean)} episodes, none yet with an accepted verdict or a positive end",
                (),
            )
        return Decision(ADOPTED, basis, None, adopted_reason, tuple(clean))
    return Decision(ON_TRIAL, None, None, None, ())


_RETIRE_ORDER = ("error", "negative", "rejected", "contrary_end", "corrected", "holds_false", "disagreed")
_RULE = {
    "error": ("errors", "it raised in {n} episodes"),
    "negative": ("negative_signals", "negative signals followed its use in {n} episodes"),
    "rejected": ("rejected", "its result was rejected in {n} episodes"),
    "contrary_end": ("contrary_end", "it answered in {n} episodes that ended contrary"),
    "corrected": ("corrected", "a correction followed its answer in {n} episodes"),
    "holds_false": ("holds_false", "its stated result condition was false in {n} episodes"),
    "disagreed": ("disagreed", "its result disagreed with the final result in {n} episodes"),
}
_WHY = {
    "error": "errors",
    "negative": "negative signals",
    "holds_false": "its stated result condition false",
    "rejected": "a rejected result",
    "contrary_end": "an answer in an episode that ended contrary",
    "corrected": "a correction after its answer",
    "disagreed": "disagreement with the final result",
    "failed": "failures",
}


@dataclass(frozen=True)
class Outcome:
    """What outcomes make of a procedure: its status (and the rule, reason and episodes behind it), what an adoption
    rests on, and its evidence level."""

    status: str
    level: str | None
    basis: str | None
    rule: str | None
    reason: str | None
    episodes: tuple[str, ...]


def apply_outcome(level: str | None, prior: str | None, uses: Iterable[Use]) -> Outcome:
    """The one decider of what outcomes do to a procedure (a library item or a capture): its status from its uses
    (:func:`decide`) and its evidence level. *level* is the level the gate (or the capture's reproduction) set; this
    never recomputes it. Outcomes only raise it to ``confirmed-by-outcomes`` (adopted on the basis of verdicts) and
    lower ``confirmed-by-outcomes`` back to ``reproduces-its-case`` when the item is retired or is no longer adopted
    on verdicts. Outcomes in the ``implicit`` and ``none`` regimes never confirm."""
    d = decide(uses, prior)
    new = level
    if d.status == ADOPTED and d.basis == BASIS_VERDICTS:
        new = CONFIRMED_BY_OUTCOMES
    elif level == CONFIRMED_BY_OUTCOMES:
        new = REPRODUCES_ITS_CASE  # outcomes no longer confirm it: the reproduction alone stands
    return Outcome(d.status, new, d.basis, d.rule, d.reason, d.episodes)


def record_line(t: Mapping[str, Any]) -> str:
    """One line saying what the track record holds, never claiming more: verdicts when there are any, otherwise the
    episodes it ran in without errors; never the word for correctness."""
    v = t.get("verdicts") or {}
    acc, rej = int(v.get("accepted") or 0), int(v.get("rejected") or 0)
    if acc or rej:
        return f"its record has verdicts: {acc} accepted, {rej} rejected"
    ran = int((t.get("implicit") or {}).get("reused_episodes") or 0)
    if ran:
        return f"its record has no verdicts; it ran without errors in {ran} episodes"
    return "it has not been used since it was stored"


def status_text(status: str, basis: str | None) -> str:
    """The status as the index shows it: ``adopted (implicit)`` for an adoption without verdicts."""
    if status == ADOPTED and basis == BASIS_IMPLICIT:
        return f"{ADOPTED} (implicit)"
    return status


# --- what a status rests on ---------------------------------------------------

#: What an item's status rests on, by the signals its uses carry: verdicts after its answered calls, outcomes at the
#: end of its episodes, implicit signals only (stated conditions, corrections), or no outcome signal at all.
BASIS_END_OUTCOMES = "end_outcomes"
BASIS_NO_OUTCOME_SIGNALS = "no_outcome_signals"
STATUS_BASES = (BASIS_VERDICTS, BASIS_END_OUTCOMES, BASIS_IMPLICIT, BASIS_NO_OUTCOME_SIGNALS)


def status_basis(uses: Iterable[Use]) -> str:
    """What a status decided from *uses* rests on (:data:`STATUS_BASES`): ``verdicts`` when any use carries a
    verdict, else ``end_outcomes`` when any carries an end outcome, else ``implicit`` when any carries a stated
    condition's result or a correction, else ``no_outcome_signals`` (no use, or uses that only returned, raised or
    refused). A contrary signal on a use its record does not attribute (``episode_negative``) counts as its regime
    observes it: a verdict in ``dense``, an end outcome in ``sparse``, a correction in ``implicit``. An error is
    counted as a failed use, never as an outcome signal."""
    us = list(uses)
    negative = {u.regime for u in us if u.answered is None and u.episode_negative}
    if any(u.verdict in VERDICTS for u in us) or "dense" in negative:
        return BASIS_VERDICTS
    if any(u.end_outcome in END_OUTCOMES for u in us) or "sparse" in negative:
        return BASIS_END_OUTCOMES
    if any(u.holds is not None or u.corrected for u in us) or "implicit" in negative:
        return BASIS_IMPLICIT
    return BASIS_NO_OUTCOME_SIGNALS


def use_counts(uses: Iterable[Use]) -> dict:
    """How the uses were judged (:func:`judge`): ``{"clean", "failed", "neutral"}`` counts."""
    out = {CLEAN: 0, FAILED: 0, NEUTRAL: 0}
    for u in uses:
        out[judge(u)[0]] += 1
    return out


def basis_text(status: str, basis: str, counts: Mapping[str, int]) -> str:
    """The status as a card states it, by what it rests on: never a claim of correctness the signals do not give.
    ``no_outcome_signals`` says no outcome signal has arrived for any use; the others say how many uses were clean and
    failed and how they were judged."""
    clean, failed = int(counts.get(CLEAN) or 0), int(counts.get(FAILED) or 0)
    if basis == BASIS_NO_OUTCOME_SIGNALS:
        used = clean + failed + int(counts.get(NEUTRAL) or 0)
        tail = f" ({used} use{'s' if used != 1 else ''} without one)" if used else " (it has not been used)"
        return f"{status}: no outcome signal has arrived for any use of this item{tail}"
    how = {
        BASIS_VERDICTS: "judged by the verdict after each answered call",
        BASIS_END_OUTCOMES: "all judged at episode end",
        BASIS_IMPLICIT: "judged by implicit signals only (errors, stated conditions, corrections), never by an outcome",
    }[basis]
    return f"{status}: {clean} clean use{'s' if clean != 1 else ''}, {failed} failed, {how}"


# --- signals to outcomes --------------------------------------------------------------------------------------


def signal_outcome(
    signals: Iterable[Any],
    regime: str
) -> dict:
    """``{verdict, end_outcome, corrected}`` from an episode's signals, as the regime may observe them
    (:data:`.signals.REGIME_SOURCES`): a ``checker`` verdict only when the run allows visible verdicts and the
    signal itself was visible to the actor; a contrary signal wins over a positive one. In ``dense`` the checker
    gives the verdict; in ``sparse`` it gives the end outcome, as does provenance; corrections are the reader's
    ``correct`` and ``re_ask``, the recurrence's ``re_ask`` and the provenance's ``correct``."""
    from .signals import REGIME_SOURCES

    allowed = REGIME_SOURCES.get(regime, frozenset())
    verdict = end = None
    corrected = False
    for s in signals:
        source, label = getattr(s, "source", None), getattr(s, "label", None)
        if source not in allowed:
            continue
        if source == "checker":
            if not ((getattr(s, "visible_to_actor", False) is True)):
                continue
            if regime == "dense":
                verdict = "rejected" if label == "fail" or verdict == "rejected" else (
                    "accepted" if label == "pass" else verdict
                )
            elif regime == "sparse":
                end = "contrary" if label == "fail" or end == "contrary" else (
                    "positive" if label == "pass" else end
                )
        elif source == "provenance" and regime in ("dense", "sparse"):
            end = "contrary" if label == "correct" or end == "contrary" else (
                "positive" if label == "support" else end
            )
        if (source, label) in (
            ("reader", "correct"),
            ("reader", "re_ask"),
            ("recurrence", "re_ask"),
            ("provenance", "correct"),
        ):
            corrected = True
    if regime == "none":
        verdict, end, corrected = None, None, False
    return {"verdict": verdict, "end_outcome": end, "corrected": corrected}
