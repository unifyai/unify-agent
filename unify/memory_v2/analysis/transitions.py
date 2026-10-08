"""Transition tables over recorded dialogue actions (spec §F2, §I2): (action, next observation) pairs.

A dialogue action is the action an agent's reply carried (``args[0]``, else the method and arguments)
paired with the observation that came back (``response``). :func:`transitions` turns exported episodes
(or Action lists) into :class:`Transition` rows that also carry the observation the action was taken
in (``before``). The helpers check knowledge against every recorded pair, never a sample:

* :func:`table` groups pairs by a key (by default the action) and lists the distinct results;
* :func:`conflicts` names keys whose results differ, i.e. where a deterministic rule keyed that way
  would be contradicted by a recording;
* :func:`check` runs a predictor over every in-scope transition and reports each disagreement;
* :func:`fixture` writes the pairs as JSON-able rows a test file can embed.

Consistency is by exact equality of the values the caller's functions return (a parsed observation,
say), never by words. Imports only the standard library, so it runs in the consolidation sandbox as
``memlab.analysis.transitions``. Tests in the memory library cannot import memlab (the gate mounts only
``/memory``); copy the pairs they need into a fixture file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

_LABEL = re.compile(r"^\s*([^:\n]{1,40}?)\s*:")


@dataclass(frozen=True)
class Transition:
    episode: str
    index: int  # the action's action_index (what a cover cites)
    channel: str
    action: str
    before: Any  # the observation the action was taken in (None if unknown)
    after: Any  # the observation that came back


def _get(a: Any, name: str, default: Any = None) -> Any:
    return a.get(name, default) if isinstance(a, dict) else getattr(a, name, default)


def action_text(a: Any) -> str:
    """The action a dialogue row records: its first argument if text, else method plus arguments."""
    args = _get(a, "args") or []
    if args and isinstance(args[0], str) and len(args) == 1 and not _get(a, "kwargs"):
        return args[0]
    return _canon([_get(a, "method"), list(args), _get(a, "kwargs") or {}])


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=False)


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return _canon(value)


def from_actions(
    episode: str,
    actions: Iterable[Any],
    first_observation: Any = None,
) -> list[Transition]:
    """The recorded dialogue pairs of one episode, in order.

    *actions* are :class:`~memlab.episodes.Action` objects or exported action dicts (``index`` is used
    when present, else the position). Only ``kind == "dialogue"`` rows with status ``ok`` and a
    response count. ``before`` is the previous pair's observation on the same channel, or
    *first_observation* for the first.
    """
    out: list[Transition] = []
    last: dict[str, Any] = {}
    for pos, a in enumerate(actions):
        if (_get(a, "kind") or "tool") != "dialogue":
            continue
        if _get(a, "status") != "ok" or _get(a, "response") is None:
            continue
        ch = _get(a, "channel")
        idx = _get(a, "index", pos)
        out.append(
            Transition(
                episode,
                int(idx if idx is not None else pos),
                ch,
                action_text(a),
                last.get(ch, first_observation),
                _get(a, "response"),
            ),
        )
        last[ch] = _get(a, "response")
    return out


def transitions(episodes: Iterable[dict]) -> list[Transition]:
    """Every dialogue pair of exported episode rows (``/inputs/episodes/<id>.json``), in order.

    The first pair's ``before`` is the episode's first request message, the initial observation.
    """
    out: list[Transition] = []
    for row in episodes:
        req = row.get("request") or []
        out += from_actions(
            row["episode_id"],
            row.get("actions") or [],
            req[0] if req else None,
        )
    return out


def group(
    ts: Iterable[Transition],
    key: Callable[[Transition], Any] = lambda t: t.action,
) -> dict[Any, list[Transition]]:
    out: dict[Any, list[Transition]] = {}
    for t in ts:
        out.setdefault(_hashable(key(t)), []).append(t)
    return out


def table(
    ts: Iterable[Transition],
    key: Callable[[Transition], Any] = lambda t: t.action,
    value: Callable[[Transition], Any] = lambda t: t.after,
) -> dict[Any, dict[str, list[tuple[str, int]]]]:
    """key -> {canonical value -> [(episode, index), ...]}: what each key was recorded to give."""
    out: dict[Any, dict[str, list[tuple[str, int]]]] = {}
    for k, rows in group(ts, key).items():
        slot = out.setdefault(k, {})
        for t in rows:
            slot.setdefault(_canon(value(t)), []).append((t.episode, t.index))
    return out


def conflicts(
    ts: Iterable[Transition],
    key: Callable[[Transition], Any] = lambda t: t.action,
    value: Callable[[Transition], Any] = lambda t: t.after,
) -> dict[Any, dict[str, list[tuple[str, int]]]]:
    """The keys of :func:`table` with more than one distinct value (a rule keyed so is contradicted)."""
    return {k: v for k, v in table(ts, key, value).items() if len(v) > 1}


@dataclass
class Check:
    checked: int = 0
    out_of_scope: int = 0
    contradictions: list[tuple[str, int, Any, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """At least one in-scope transition, and none contradicts the predictor."""
        return self.checked > 0 and not self.contradictions


def check(
    ts: Iterable[Transition],
    predict: Callable[[Transition], Any],
    observe: Callable[[Transition], Any],
    in_scope: Callable[[Transition], bool] | None = None,
) -> Check:
    """Compare ``predict(t)`` with ``observe(t)`` by equality on every in-scope transition.

    A predictor that raises on an in-scope transition is a contradiction (its prediction is recorded
    as ``"raised <ExceptionType>"``).
    """
    res = Check()
    for t in ts:
        if in_scope is not None and not in_scope(t):
            res.out_of_scope += 1
            continue
        res.checked += 1
        try:
            p = predict(t)
        except Exception as exc:  # noqa: BLE001 - a failing predictor is a finding
            p = f"raised {type(exc).__name__}"
        o = observe(t)
        if p != o:
            res.contradictions.append((t.episode, t.index, p, o))
    return res


def fixture(ts: Iterable[Transition]) -> list[dict]:
    """JSON-able rows (episode, index, channel, action, before, after) for a test fixture file."""
    return [
        {
            "episode": t.episode,
            "index": t.index,
            "channel": t.channel,
            "action": t.action,
            "before": t.before,
            "after": t.after,
        }
        for t in ts
    ]


def observation_shape(obs: Any) -> str:
    """A structural summary of one observation, for fingerprints.

    Parsed values (and text that parses as JSON) give their key tree; other text gives the sequence of
    its line labels (the text before a ``:`` within the first 40 characters, else ``_``), with
    consecutive repeats collapsed. Values, numbers and free text never appear.
    """
    from .shapes import tree

    if obs is None:
        return "none"
    if not isinstance(obs, str):
        return "tree:" + _canon(tree(obs))
    s = obs.strip()
    if s[:1] in ("{", "["):
        try:
            return "tree:" + _canon(tree(json.loads(s)))
        except (ValueError, RecursionError):
            pass
    labels: list[str] = []
    for line in s.splitlines():
        if not line.strip():
            continue
        m = _LABEL.match(line)
        lab = (m.group(1).strip() + ":") if m else "_"
        if not labels or labels[-1] != lab:
            labels.append(lab)
    return "text:" + "|".join(labels[:32])
