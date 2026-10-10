"""Overlap candidates for CURATE (spec v2.1 §10.3), found deterministically from structure and recordings.

Three rules. None of them reads a name, a docstring, a description or any other text:

* ``shapes_and_covers``: two functions whose recorded input shapes match (at least one shape descriptor in
  common) and whose covers overlap (at least one recorded input admitted for both: an action, or a typed cover
  such as an actor cell). When every cover of one is a cover of the other, the reason says so: the narrower one
  may be retired behind an alias of the wider one. Matching shapes alone are not overlap (two readers of one file
  can do different jobs), nor are shared covers without a shared shape.
* ``antiunify``: two functions whose bodies anti-unify (:mod:`.analysis.fn_antiunify`: the most specific
  generalisation of their normalised ASTs, locals renamed, docstrings dropped) with a kept share of at least
  :data:`ANTIUNIFY_MIN_SHARE`. The generalisation, holes as parameters, is the proposed merged body.
* ``same_uses``: notes whose ``uses:`` lists name the same set of functions.

Each candidate carries a fingerprint: a hash of its rule, its items and their current content (each function's
unparsed definition, each note's bytes), so formatting and comments in code never change it. CURATE's trigger
(:mod:`.curate`) counts a candidate once per fingerprint.

Bounds are explicit, never silent: the shape and cover graphs keep :mod:`.analysis.library_graph`'s bounds, and at
most *max_pairs* function pairs are anti-unified, in sorted order. What stopped early is named in the result's
``truncated``. Pure: it reads files and never imports or runs library code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path

from . import layout
from .analysis.fn_antiunify import antiunify, function_defs
from .analysis.library_graph import shape_graph

OVERLAP_VERSION = 1
#: Spec §10.3: two bodies "anti-unify" at this kept share or more. In the hygiene study the real near-duplicates
#: kept 0.7-1.0 and unrelated functions under 0.15 (test_hygiene_analysis.py); offline replay calibrates it.
ANTIUNIFY_MIN_SHARE = 0.7
#: Function pairs anti-unified per call, in sorted order (77 functions, the largest v2 library, make 2,926).
MAX_ANTIUNIFY_PAIRS = 5000
RULES = ("shapes_and_covers", "antiunify", "same_uses")


def _canon(obj: object) -> str:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def shape_signatures(rows: Mapping[str, Mapping]) -> dict[str, set[str]]:
    """item -> the canonical JSON of each recorded input-shape descriptor (rows of :func:`.shape_rows.shapes_at`)."""
    return {
        item: {_canon(d) for d in (row.get("shapes") or [])}
        for item, row in rows.items()
    }


def cover_signatures(
    covers: Iterable[tuple[str, str, int]],
    typed: Iterable[tuple[str, str, str]] = (),
) -> dict[str, set[str]]:
    """item -> one string per recorded cover: ``action <episode> <index>`` or ``typed <episode> <cover JSON>``."""
    out: dict[str, set[str]] = {}
    for item, eid, idx in covers:
        out.setdefault(item, set()).add(f"action {eid} {int(idx)}")
    for item, eid, cover in typed:
        out.setdefault(item, set()).add(f"typed {eid} {cover}")
    return out


def by_shapes_and_covers(
    shapes: Mapping[str, set[str]],
    covers: Mapping[str, set[str]],
    functions: set[str],
) -> tuple[list[dict], list[str]]:
    """Pairs of *functions* with a shared shape and a shared cover, and what the two graphs left out."""

    def keep(m: Mapping[str, set[str]]) -> dict[str, set[str]]:
        return {i: set(s) for i, s in m.items() if i in functions and s}

    by_shape = shape_graph(keep(shapes))
    by_cover = shape_graph(keep(covers))
    shared_shapes = {(e.a, e.b): e for e in by_shape.edges}
    within = {(s.narrow, s.wide): s for s in by_cover.subsumptions}
    out = []
    for e in sorted(by_cover.edges, key=lambda e: (e.a, e.b)):
        s = shared_shapes.get((e.a, e.b))
        if s is None:
            continue
        reasons = [
            f"recorded input shapes: {s.shared} shared of {s.union}",
            f"covers: {e.shared} shared of {e.union}",
        ]
        for narrow, wide in ((e.a, e.b), (e.b, e.a)):
            sub = within.get((narrow, wide))
            if sub is not None:
                same = " (the same covers)" if sub.equal else ""
                reasons.append(
                    f"covers: every cover of {narrow} is a cover of {wide}{same}",
                )
                break
        out.append(
            {"rule": "shapes_and_covers", "items": [e.a, e.b], "reasons": reasons},
        )
    cut = [f"shapes:{t}" for t in by_shape.truncated] + [
        f"covers:{t}" for t in by_cover.truncated
    ]
    return out, cut


def _definitions(tree: Path, lib: layout.Library) -> dict[str, tuple[object, dict]]:
    """item id -> (its definition, its module's top-level definitions by name), for every function item."""
    modules: dict[str, dict] = {}
    out: dict[str, tuple[object, dict]] = {}
    for f in lib.functions:
        if f.module not in modules:
            source = (
                (Path(tree) / f.path).read_bytes().decode("utf-8", errors="replace")
            )
            modules[f.module] = function_defs(source)
        defs = modules[f.module]
        if f.name in defs:
            out[f.item_id] = (defs[f.name], defs)
    return out


def by_antiunify(
    tree: Path,
    lib: layout.Library,
    *,
    min_share: float = ANTIUNIFY_MIN_SHARE,
    max_pairs: int = MAX_ANTIUNIFY_PAIRS,
) -> tuple[list[dict], list[str]]:
    """Pairs of functions whose bodies anti-unify at *min_share* or more, and ``["antiunify"]`` past *max_pairs*.

    Sibling calls are inlined (the *library* of :func:`.analysis.fn_antiunify.antiunify`) only for two functions of
    one module, whose siblings are the same.
    """
    defs = _definitions(tree, lib)
    pairs = list(combinations(sorted(defs), 2))
    out = []
    for a, b in pairs[:max_pairs]:
        (fa, siblings), (fb, _) = defs[a], defs[b]
        same_module = a.split(":", 1)[0] == b.split(":", 1)[0]
        g = antiunify([fa, fb], library=siblings if same_module else None)
        if g is None or g.kept_share < min_share:
            continue
        reasons = [
            f"bodies anti-unify: kept share {g.kept_share:.2f}, {len(g.holes)} hole(s)",
        ]
        if g.wrapper:
            reasons.append(
                "one calls the other: keep or inline the wrapper rather than merge two peers",
            )
        if g.helper_driven:
            reasons.append(
                "most of what is kept is a shared helper's code: reuse the helper rather than merge",
            )
        if not g.same_signature:
            reasons.append("their parameters differ")
        if g.bounded:
            reasons.append(
                "the comparison's work bound was reached: the kept share is a lower bound",
            )
        out.append(
            {
                "rule": "antiunify",
                "items": [a, b],
                "reasons": reasons,
                "kept_share": round(g.kept_share, 4),
                "holes": len(g.holes),
                "generalisation": g.source,
            },
        )
    return out, (["antiunify"] if len(pairs) > max_pairs else [])


def by_same_uses(lib: layout.Library) -> list[dict]:
    """Groups of two or more readable notes whose ``uses:`` name the same set of functions."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for n in lib.notes:
        if not n.error and n.uses:
            groups.setdefault(tuple(sorted(set(n.uses))), []).append(n.item_id)
    return [
        {
            "rule": "same_uses",
            "items": sorted(notes),
            "reasons": [
                f"the notes use the same {len(uses)} function(s): {', '.join(uses)}",
            ],
        }
        for uses, notes in sorted(groups.items())
        if len(notes) >= 2
    ]


def fingerprint(
    rule: str,
    items: list[str],
    tree: Path,
    bodies: Mapping[str, str],
) -> str:
    """``overlap:<sha256>`` of the rule, the items and their content (a function's unparsed definition, a note's
    bytes): unchanged by formatting or comments in code, changed by any edit of what the items are.
    """
    h = hashlib.sha256(_canon([OVERLAP_VERSION, rule, sorted(items)]).encode("utf-8"))
    for item in sorted(items):
        if layout.NOTE_ID.match(item):
            try:
                data = (Path(tree) / item).read_bytes()
            except OSError:
                data = b""
        else:
            data = bodies.get(item, "").encode("utf-8")
        h.update(hashlib.sha256(data).digest())
    return "overlap:" + h.hexdigest()


def overlap_candidates(
    tree: Path,
    *,
    shapes: Mapping[str, Mapping] | None = None,
    covers: Iterable[tuple[str, str, int]] = (),
    typed: Iterable[tuple[str, str, str]] = (),
    min_share: float = ANTIUNIFY_MIN_SHARE,
    max_pairs: int = MAX_ANTIUNIFY_PAIRS,
) -> dict:
    """Every overlap candidate of the library at *tree* (spec §10.3).

    *shapes* are the commit's shape rows (:func:`.shape_rows.shapes_at`), *covers* the evidence store's
    ``(item, episode, action)`` covers and *typed* its typed covers ``(item, episode, cover JSON)``. Rows of items
    the tree no longer holds are ignored. Returns ``{"version", "candidates", "truncated"}``, the candidates in
    rule order, then by items.
    """
    tree = Path(tree)
    lib = layout.discover(tree)
    functions = {f.item_id for f in lib.functions}
    bodies = layout.function_bodies(tree)
    found, cut = by_shapes_and_covers(
        shape_signatures(shapes or {}),
        cover_signatures(covers, typed),
        functions,
    )
    merged, cut_au = by_antiunify(tree, lib, min_share=min_share, max_pairs=max_pairs)
    candidates = [
        {**c, "fingerprint": fingerprint(c["rule"], c["items"], tree, bodies)}
        for c in found + merged + by_same_uses(lib)
    ]
    candidates.sort(key=lambda c: (RULES.index(c["rule"]), c["items"]))
    return {
        "version": OVERLAP_VERSION,
        "candidates": candidates,
        "truncated": cut + cut_au,
    }


def render_overlap(result: Mapping) -> str:
    """``overlap.json``: sorted keys, one trailing newline."""
    return json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
