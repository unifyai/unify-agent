from __future__ import annotations

from ..episodes import Cell
from .cells import CallSite
from .provenance import def_use_edges


def backward_slice(cells: list[Cell], target: int) -> set[int]:
    edges = def_use_edges(cells)
    keep, frontier = {target}, [target]
    while frontier:
        cur = frontier.pop()
        for a, b in edges:
            if b == cur and a not in keep:
                keep.add(a)
                frontier.append(a)
    return keep


def generic_methods(
    episode_sites: list[list[CallSite]],
    share: float = 0.2,
) -> set[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for sites in episode_sites:
        for key in {(s.channel, s.method) for s in sites}:
            counts[key] = counts.get(key, 0) + 1
    n = max(len(episode_sites), 1)
    return {k for k, v in counts.items() if v / n >= share}


def prelude(sites: list[CallSite], generic: set[tuple[str, str]]) -> list[CallSite]:
    out = []
    for s in sites:
        if (s.channel, s.method) not in generic:
            break
        out.append(s)
    return out
