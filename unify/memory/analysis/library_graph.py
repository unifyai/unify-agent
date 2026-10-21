"""The library's shape graph (memory hygiene, stage 4): functions linked by the input shapes they cover.

Input: a mapping from each library item (a string) to the set of recorded input-shape signatures (strings) it
covers (the caller builds it from the evidence store; this module never reads evidence, outcomes or task labels).
A cover that is a bare string, or holds anything but strings, is refused (``TypeError``). Output, pure and
deterministic, in one :class:`ShapeGraph`:

* **edges**: items that share at least one signature, weighted by the Jaccard similarity of their sets
  (``|A & B| / |A | B|``);
* **clusters**: connected components over edges of at least a threshold weight ("functions over the same kind
  of data"); an item with no such edge is its own cluster;
* **subsumptions**: ``narrow`` covers a non-empty subset of what ``wide`` covers, so ``narrow`` is a retirement
  candidate if ``wide`` also matches its outputs and passes its tests (the differential runner and the gate
  decide that, not this module). Equal sets are listed once, in name order, with ``equal=True``.

Pairs come from an inverted index (signature -> items), item by item: an item's co-covering items are counted
over its own signatures only, so a pair's shared count is exact once its first item is done.

Bounds: at most ``max_items`` items and ``max_signatures`` signatures per item are read (in sorted order); at
most ``max_pairs`` pairs are kept and ``max_work`` index steps spent; whatever stops early is named in
``truncated`` (``items``, ``signatures``, ``pairs``, ``work``). Edges and clusters are then a partial view (every
listed weight is still exact), but **a truncated graph lists no subsumptions**: an unread signature or an
uncounted pair could make one false, and a subsumption is a retirement candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

MAX_ITEMS = 2_000
MAX_SIGNATURES = 10_000
MAX_PAIRS = 200_000
MAX_WORK = 5_000_000  # inverted-index steps


@dataclass(frozen=True)
class Edge:
    a: str
    b: str  # a < b
    weight: float  # Jaccard similarity
    shared: int
    union: int


@dataclass(frozen=True)
class Subsumption:
    narrow: str
    wide: str
    shared: int  # |narrow|, all of which wide covers
    equal: bool


@dataclass(frozen=True)
class ShapeGraph:
    nodes: tuple[tuple[str, int], ...]  # (item, number of signatures read), by item
    edges: tuple[Edge, ...]
    clusters: tuple[tuple[str, ...], ...]
    subsumptions: tuple[Subsumption, ...]  # empty whenever the graph is truncated
    truncated: tuple[
        str,
        ...,
    ]  # what stopped early: "items", "signatures", "pairs", "work"

    @property
    def complete(self) -> bool:
        return not self.truncated


def _bounded(
    covers: Mapping[str, Iterable[str]],
    max_items: int,
    max_signatures: int,
) -> tuple[dict[str, frozenset[str]], list[str]]:
    cut: list[str] = []
    if not isinstance(covers, Mapping):
        raise TypeError("covers must map each item to its covered signatures")
    if any(not isinstance(item, str) for item in covers):
        raise TypeError("items must be strings")
    if len(covers) > max_items:
        cut.append("items")
    out: dict[str, frozenset[str]] = {}
    for item in sorted(covers)[:max_items]:
        cover = covers[item]
        if isinstance(cover, (str, bytes)) or not isinstance(cover, Iterable):
            raise TypeError(
                f"the cover of {item[:80]!r} must be a collection of signatures",
            )
        sigs = set(cover)
        if any(not isinstance(sig, str) for sig in sigs):
            raise TypeError(
                f"the cover of {item[:80]!r} must hold only string signatures",
            )
        ordered = sorted(sigs)
        if len(ordered) > max_signatures and "signatures" not in cut:
            cut.append("signatures")
        out[item] = frozenset(ordered[:max_signatures])
    return out, cut


def _pairs(
    sets: Mapping[str, frozenset[str]],
    max_pairs: int,
    max_work: int,
) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Shared-signature counts of co-covering pairs (a < b), item by item through an inverted index."""
    index: dict[str, list[str]] = {}
    for item in sorted(sets):
        for sig in sets[item]:
            index.setdefault(sig, []).append(item)  # each list is in item order
    shared: dict[tuple[str, str], int] = {}
    work = 0
    for a in sorted(sets):
        counts: dict[str, int] = {}
        for sig in sets[a]:
            items = index[sig]
            work += len(items)
            if work > max_work:
                return shared, ["work"]
            for b in items:
                if b > a:
                    counts[b] = counts.get(b, 0) + 1
        if len(shared) + len(counts) > max_pairs:
            return shared, ["pairs"]  # every pair kept so far is exact
        for b in sorted(counts):
            shared[(a, b)] = counts[b]
    return shared, []


def _edges(
    sets: Mapping[str, frozenset[str]],
    shared: Mapping[tuple[str, str], int],
) -> tuple[Edge, ...]:
    out = []
    for (a, b), n in shared.items():
        union = len(sets[a]) + len(sets[b]) - n
        out.append(Edge(a, b, n / union, n, union))
    return tuple(sorted(out, key=lambda e: (-e.weight, e.a, e.b)))


def _clusters(
    items: Iterable[str],
    edges: Iterable[Edge],
    threshold: float,
) -> tuple[tuple[str, ...], ...]:
    parent = {i: i for i in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        if e.weight >= threshold:
            ra, rb = find(e.a), find(e.b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for i in sorted(parent):
        groups.setdefault(find(i), []).append(i)
    return tuple(
        sorted((tuple(g) for g in groups.values()), key=lambda g: (-len(g), g)),
    )


def _subsumptions(
    sets: Mapping[str, frozenset[str]],
    shared: Mapping[tuple[str, str], int],
) -> tuple[Subsumption, ...]:
    out = []
    for (a, b), n in shared.items():
        na, nb = len(sets[a]), len(sets[b])
        if n == na == nb:
            out.append(Subsumption(a, b, n, True))
        elif n == na:
            out.append(Subsumption(a, b, n, False))
        elif n == nb:
            out.append(Subsumption(b, a, n, False))
    return tuple(sorted(out, key=lambda s: (s.narrow, s.wide)))


def shape_graph(
    covers: Mapping[str, Iterable[str]],
    *,
    threshold: float = 0.5,
    min_weight: float = 0.0,
    max_items: int = MAX_ITEMS,
    max_signatures: int = MAX_SIGNATURES,
    max_pairs: int = MAX_PAIRS,
    max_work: int = MAX_WORK,
) -> ShapeGraph:
    """Nodes, edges (weight >= *min_weight*, heaviest first), clusters at *threshold* and subsumptions.

    Subsumptions are listed only when nothing was truncated (module docstring).
    """
    sets, cut = _bounded(covers, max_items, max_signatures)
    shared, cut_pairs = _pairs(sets, max_pairs, max_work)
    cut += cut_pairs
    edges = _edges(sets, shared)
    return ShapeGraph(
        nodes=tuple((i, len(sets[i])) for i in sorted(sets)),
        edges=tuple(e for e in edges if e.weight >= min_weight),
        clusters=_clusters(sets, edges, threshold),
        subsumptions=() if cut else _subsumptions(sets, shared),
        truncated=tuple(cut),
    )
