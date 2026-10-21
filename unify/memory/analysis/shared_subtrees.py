"""Shared subtrees across a library's functions (memory hygiene, stage 4): candidates for an extracted helper.

A subtree is an expression or statement of at least ``min_size`` nodes; two subtrees are the same when they are
equal **up to renaming of local names** (each subtree's locals numbered by first appearance within it, so
``grid`` in one function and ``args[0]``'s binding ``g`` in another match). Free names, attributes, constants
and operators must be equal. Structure only: no words are compared beyond the AST's own labels.

Candidates are subtrees found in at least ``min_functions`` functions:

* **maximal**: a subtree that only ever occurs as the same child of a larger shared subtree (one with the same
  functions and occurrences) is not listed beside it;
* **non-overlapping**: within one function, nested occurrences of the same subtree count once (the outer one);
* **ranked by a compression estimate** (Stitch-style, greedy): extracting a helper of ``s`` nodes with ``p``
  parameters (locals read but not bound inside it) used at ``u`` sites saves
  ``u * s - (s + 2 + p) - u * (2 + p)`` nodes (the sites' code, less the helper's definition and its calls).
  The greedy pass picks the best, removes occurrences that overlap what it picked, re-scores, and repeats;
  near-linear: groups are hash buckets of the renaming-invariant key, used positions a bitmap per function, and
  the candidates a heap re-scored lazily (scores only fall, so a popped score that is still current is the best).

Locations are the original source lines, and the example is the first occurrence's original source.

Bounds: functions over the anti-unification caps (:data:`.fn_antiunify.MAX_NODES`, :data:`.fn_antiunify.MAX_DEPTH`)
are skipped and named; at most :data:`MAX_LIBRARY_NODES` nodes are read (functions past it are skipped,
``truncated``); subtrees over ``max_size`` nodes are not candidates; at most ``max_groups`` qualifying groups are
ranked (the largest by ``u * s``, ``truncated`` when more); the greedy pass stops, ``truncated``, after
``max_work`` occurrence checks and marked positions. Deterministic. Standard library only.

Known limits: a helper's parameter count is the locals it reads but does not bind, so values it binds and the
code after it reads (outputs) are not counted; a subtree that only occurs inside a larger shared one is dropped
in advance even if the larger one is later not picked.
"""

from __future__ import annotations

import ast
import heapq
from dataclasses import dataclass
from typing import Mapping

from .fn_antiunify import _render, _Term, function_defs, normalise

MAX_LIBRARY_NODES = 400_000
MAX_GROUPS = 2_000
MAX_GREEDY_WORK = 5_000_000  # occurrence checks and marked positions in the greedy pass


@dataclass(frozen=True)
class Occurrence:
    function: str
    lineno: int | None
    end_lineno: int | None
    start: int  # pre-order position of the subtree in the function's normalised tree


@dataclass(frozen=True)
class SharedSubtree:
    key: str  # the renaming-invariant structural key (hex)
    node_type: str  # the AST node type at the subtree's root
    kind: str  # "expr" or "stmt"
    size: int
    functions: tuple[str, ...]
    occurrences: tuple[Occurrence, ...]
    params: int  # locals read but not bound inside: an extracted helper's parameters
    compression: int  # nodes saved by extracting it (see the module docstring)
    example: str | None  # the first occurrence's original source


@dataclass(frozen=True)
class SharedReport:
    candidates: tuple[SharedSubtree, ...]
    skipped: tuple[str, ...]  # functions over the bounds or past the library cap
    truncated: bool


@dataclass
class _Occ:
    function: str
    term: _Term
    parent: _Term | None

    @property
    def end(self) -> int:
        return self.term.start + self.term.size


def _subtree(t: _Term) -> list[_Term]:
    out, stack = [], [t]
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(cur.children())
    return out


def _params(t: _Term) -> int:
    """Locals of *t* read but not bound inside it."""
    bound: set[str] = set()
    for s in _subtree(t):
        if s.own and not (
            isinstance(s.node, ast.Name) and isinstance(s.node.ctx, ast.Load)
        ):
            bound.update(s.own)
    return len(set(t.locs or ()) - bound)


def compression(uses: int, size: int, params: int) -> int:
    """Nodes saved by extracting a *size*-node helper with *params* parameters used at *uses* sites."""
    return uses * size - (size + 2 + params) - uses * (2 + params)


def _non_overlapping(occs: list[_Occ]) -> list[_Occ]:
    out: list[_Occ] = []
    last: dict[str, int] = {}
    for o in sorted(occs, key=lambda o: (o.function, o.term.start)):
        if o.term.start >= last.get(o.function, -1):
            out.append(o)
            last[o.function] = o.end
    return out


def shared_subtrees(
    fns: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    *,
    min_size: int = 8,
    min_functions: int = 3,
    max_size: int = 400,
    top: int = 20,
    max_groups: int = MAX_GROUPS,
    max_work: int = MAX_GREEDY_WORK,
) -> SharedReport:
    """Subtrees shared by at least *min_functions* of *fns* (name -> function), best compression first."""
    skipped: list[str] = []
    truncated = False
    read = 0
    groups: dict[bytes, list[_Occ]] = {}
    sizes: dict[str, int] = {}
    for name in sorted(fns):
        norm = normalise(fns[name])
        if norm is None or read + norm.size > MAX_LIBRARY_NODES:
            truncated = truncated or norm is not None
            skipped.append(name)
            continue
        read += norm.size
        sizes[name] = norm.size
        stack: list[tuple[_Term, _Term | None]] = [(norm.term, None)]
        while stack:
            t, parent = stack.pop()
            if (
                t.category in ("expr", "stmt")
                and min_size <= t.size <= max_size
                and t.alpha is not None
                and parent is not None  # the function itself is not a helper
            ):
                groups.setdefault(t.alpha, []).append(_Occ(name, t, parent))
            stack.extend((c, t) for c in reversed(t.children()) if c.size >= min_size)
    # qualifying groups: enough distinct functions after removing nested repeats
    live: dict[bytes, list[_Occ]] = {}
    for key, occs in groups.items():
        kept = _non_overlapping(occs)
        if len({o.function for o in kept}) >= min_functions:
            live[key] = kept
    # maximal: drop a group whose every occurrence is the same child of one larger qualifying group
    dropped = set()
    for key, occs in live.items():
        parents = {o.parent.alpha for o in occs}
        if len(parents) == 1:
            (p,) = parents
            if p is not None and p in live and len(live[p]) == len(occs):
                dropped.add(key)
    ranked = sorted(
        (k for k in live if k not in dropped),
        key=lambda k: (-len(live[k]) * live[k][0].term.size, k),
    )
    if len(ranked) > max_groups:
        ranked, truncated = ranked[:max_groups], True
    params = {k: _params(live[k][0].term) for k in ranked}
    # greedy: best compression first, re-scored after each pick without the occurrences it covered. Scores only
    # fall as positions are used, so a heap of possibly stale scores is re-scored lazily: a popped entry whose
    # score is still current is the best (ties: larger, then smaller key, as before).
    used: dict[str, bytearray] = (
        {}
    )  # per function: 1 at each pre-order position a pick covers
    work = 0

    def current(k: bytes) -> tuple[int, int, list[_Occ]] | None:
        nonlocal work
        occs = []
        for o in live[k]:
            mask = used.get(o.function)
            work += 1 + (o.term.size if mask is not None else 0)
            if mask is None or 1 not in mask[o.term.start : o.end]:
                occs.append(o)
        if len({o.function for o in occs}) < min_functions:
            return None
        size = occs[0].term.size
        gain = compression(len(occs), size, params[k])
        return (gain, size, occs) if gain > 0 else None

    heap: list[tuple[int, int, bytes]] = []
    for k in ranked:
        cur = current(k)
        if cur is not None:
            heap.append((-cur[0], -cur[1], k))
    heapq.heapify(heap)
    chosen: list[SharedSubtree] = []
    while heap and len(chosen) < top:
        if work > max_work:
            truncated = True
            break
        neg_gain, neg_size, k = heapq.heappop(heap)
        cur = current(k)
        if cur is None:
            continue
        gain, size, occs = cur
        if (gain, size) != (-neg_gain, -neg_size):
            heapq.heappush(heap, (-gain, -size, k))
            continue
        for o in occs:
            mask = used.get(o.function)
            if mask is None:
                mask = used[o.function] = bytearray(sizes[o.function])
            mask[o.term.start : o.end] = b"\x01" * (o.end - o.term.start)
            work += o.end - o.term.start
        t = occs[0].term
        chosen.append(
            SharedSubtree(
                key=k.hex(),
                node_type=t.cls,
                kind=t.category,
                size=t.size,
                functions=tuple(sorted({o.function for o in occs})),
                occurrences=tuple(
                    Occurrence(
                        o.function,
                        getattr(o.term.orig, "lineno", None),
                        getattr(o.term.orig, "end_lineno", None),
                        o.term.start,
                    )
                    for o in occs
                ),
                params=params[k],
                compression=gain,
                example=_render(t.orig) if t.orig is not None else None,
            ),
        )
    return SharedReport(tuple(chosen), tuple(skipped), truncated)


def shared_subtrees_source(source: str, **kw) -> SharedReport:
    """:func:`shared_subtrees` over the top-level functions of a module's *source*."""
    return shared_subtrees(function_defs(source), **kw)
