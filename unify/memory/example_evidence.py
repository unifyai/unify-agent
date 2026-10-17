"""Cross-episode evidence of a library function's examples, from the provenance of the fixtures its tests load.

A function's examples are the calls its tests make on it (:mod:`.example_calls`). Each argument of an example call
gets one grounding (:func:`grounding`):

* ``copied``: its digest equals the digest of a view of a fixture with provenance (:func:`.fixtures.value_views`),
  or it is a path naming such a fixture;
* ``derived``: not copied, but replacing values of a fixture the test loads changed it in the matched call (the
  test's own code is the transformation; the perturbation check finds the link);
* ``literal``: anything else, including values without a digest that the perturbation cannot show changing.

The function's examples count as cross-episode evidence when the copied or derived arguments of the calls it
accepted carry values from at least :data:`MIN_EXAMPLE_EPISODES` episodes, together with the episodes of its
verified typed ``episode`` covers (:func:`evidence_check`). The rule check (:func:`rule_problems`) reads the same
groundings.

Pure: no git, no store, no clock. Reasons name items, episode ids, fixture paths, lines and counts, never a
recorded value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .example_calls import CallKey, ExampleCall
from .fixtures import View
from .library_prompts import EPISODES_FIX

#: Examples need recorded values from at least this many episodes.
MIN_EXAMPLE_EPISODES = 1


@dataclass(frozen=True)
class Grounding:
    kind: str  # "copied" | "derived" | "literal"
    fixtures: tuple[str, ...]  # paths
    episodes: tuple[str, ...]


LITERAL = Grounding("literal", (), ())


def _episodes_of(views: dict[str, list[View]]) -> set[str]:
    return {v.episode for vs in views.values() for v in vs}


def grounding(
    call: ExampleCall,
    views: dict[str, dict[str, list[View]]],
    derived: dict[str, set[str]],
) -> dict[str, Grounding]:
    """Each argument of *call* (by parameter name) with its grounding. *views*: fixture path -> its views
    (:func:`.fixtures.value_views`; a fixture without provenance has none). *derived*: parameter name -> the
    fixtures whose perturbation changed that argument in this call."""
    out: dict[str, Grounding] = {}
    for arg in call.args:
        fixtures: set[str] = set()
        episodes: set[str] = set()
        if arg.sha is not None:
            for path in sorted(views):
                hit = views[path].get(arg.sha)
                if hit:
                    fixtures.add(path)
                    episodes |= {v.episode for v in hit}
        if arg.path is not None and views.get(arg.path):
            fixtures.add(arg.path)
            episodes |= _episodes_of(views[arg.path])
        if fixtures:
            out[arg.name] = Grounding(
                "copied",
                tuple(sorted(fixtures)),
                tuple(sorted(episodes)),
            )
            continue
        changed = sorted(p for p in derived.get(arg.name, set()) if isinstance(p, str))
        if changed:
            eps: set[str] = set()
            for p in changed:
                eps |= _episodes_of(views.get(p, {}))
            out[arg.name] = Grounding("derived", tuple(changed), tuple(sorted(eps)))
            continue
        out[arg.name] = LITERAL
    return out


def groundings(
    calls: list[ExampleCall],
    views: dict[str, dict[str, list[View]]],
    derived: dict[CallKey, dict[str, set[str]]],
) -> dict[CallKey, dict[str, Grounding]]:
    """:func:`grounding` of every call, by call key (*derived*: the perturbation check's links per call)."""
    return {c.key: grounding(c, views, derived.get(c.key, {})) for c in calls}


def _grounded(g: Grounding) -> bool:
    return g.kind in ("copied", "derived")


def _accepted(calls: Iterable[ExampleCall]) -> list[ExampleCall]:
    return sorted((c for c in calls if c.outcome == "returned"), key=lambda c: c.key)


def evidence_check(
    item: str,
    calls: list[ExampleCall],
    ground: dict[CallKey, dict[str, Grounding]],
    typed_episodes: set[str],
) -> tuple[list[str], dict]:
    """The cross-episode rule on *item*'s example *calls*: (reasons, record fields ``episodes`` and ``fixtures``).

    The episodes are those of the copied or derived arguments of the calls the function accepted (returned on),
    plus *typed_episodes* (its verified typed ``episode`` covers). One episode giving two fixtures counts once. A
    refused or raised call's arguments count for nothing.
    """
    episodes: set[str] = set(typed_episodes)
    fixtures: set[str] = set()
    for c in _accepted(calls):
        for g in ground.get(c.key, {}).values():
            if _grounded(g):
                episodes |= set(g.episodes)
                fixtures |= set(g.fixtures)
    fields = {"episodes": sorted(episodes), "fixtures": sorted(fixtures)}
    if len(episodes) >= MIN_EXAMPLE_EPISODES:
        return [], fields
    ids = ", ".join(sorted(episodes)) or "none"
    return [
        f"[examples:episodes] {item}'s calls that return use recorded values from {len(episodes)} episode(s) "
        f"({ids}); {EPISODES_FIX}",
    ], fields


def rule_episodes(
    ground: dict[CallKey, dict[str, Grounding]],
    calls: list[ExampleCall] | None = None,
) -> set[str]:
    """The episodes of the copied or derived arguments in *ground*; with *calls*, of the accepted calls only."""
    keep = None if calls is None else {c.key for c in calls if c.outcome == "returned"}
    return {
        e
        for key, params in ground.items()
        if keep is None or key in keep
        for g in params.values()
        if _grounded(g)
        for e in g.episodes
    }


#: In the library layout, a function that replaces a value it computed under a condition needs examples from this many
#: episodes, like every other function: one recorded episode is enough, and the check with unseen values (mandatory
#: for every function) is what catches a function that only reproduces its recorded outputs. The earlier layout keeps
#: the gate's ``RULE_EPISODES`` for its covers.
LIBRARY_RULE_EPISODES = 1


def rule_problems(
    item: str,
    *,
    path: str,
    rule_line: int | None,
    unchecked: bool,
    episodes: set[str],
    need: int | None = None,
) -> tuple[list[str], list[str]]:
    """The rule check on an added or changed function (reasons, notes): a function that replaces a value it
    computed from its input under a condition (*rule_line* of *path*) needs examples from *need* episodes
    (default: :data:`LIBRARY_RULE_EPISODES`)."""
    if need is None:
        need = LIBRARY_RULE_EPISODES
    if unchecked:
        return [], [f"G2 rule check: {item} is too large to analyse; not checked"]
    if rule_line and len(episodes) < need:
        return [
            f"{item} replaces a value it computed from its input under a condition (line {rule_line} of "
            f"{path}); a function that encodes such a rule needs examples from at least {need} "
            + ("episode" if need == 1 else "episodes"),
        ], []
    return [], []
