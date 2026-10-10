"""Memory hygiene, the shape graph (stage 4, build step 2): Jaccard edges, clusters and subsumption candidates over
each item's covered input-shape signatures (synthetic signature sets; the caller supplies real ones from evidence).
"""

from __future__ import annotations

import random
import time

import pytest

from unify.memory_v2.analysis.library_graph import shape_graph

COVERS = {
    "env/env:a": {"s1", "s2", "s3"},
    "env/env:b": {"s1", "s2", "s3", "s4"},
    "env/env:c": {"s1", "s2"},
    "env/env:d": {"s9"},
    "env/env:e": set(),
}


def test_jaccard_edges_link_items_over_shared_signatures():
    g = shape_graph(COVERS)
    edges = {(e.a, e.b): e for e in g.edges}
    assert edges[("env/env:a", "env/env:b")].weight == pytest.approx(3 / 4)
    assert edges[("env/env:a", "env/env:b")].shared == 3
    assert edges[("env/env:a", "env/env:c")].weight == pytest.approx(2 / 3)
    assert not any("env/env:d" in k or "env/env:e" in k for k in edges)
    assert g == shape_graph(dict(reversed(list(COVERS.items()))))
    assert {(e.a, e.b) for e in shape_graph(COVERS, min_weight=0.7).edges} == {
        ("env/env:a", "env/env:b"),
    }


def test_clusters_cut_at_the_threshold():
    assert shape_graph(COVERS, threshold=0.6).clusters == (
        ("env/env:a", "env/env:b", "env/env:c"),
        ("env/env:d",),
        ("env/env:e",),
    )
    assert ("env/env:a", "env/env:b") in shape_graph(COVERS, threshold=0.7).clusters


def test_subsumption_candidates_by_covered_signatures():
    subs = {(s.narrow, s.wide): s for s in shape_graph(COVERS).subsumptions}
    assert ("env/env:c", "env/env:a") in subs and ("env/env:a", "env/env:b") in subs
    assert ("env/env:c", "env/env:b") in subs
    assert ("env/env:b", "env/env:a") not in subs
    # nothing is subsumed by covering no signature, and an empty cover subsumes nothing
    assert not any("env/env:e" in k for k in subs)
    eq = shape_graph({"x": {"s"}, "y": {"s"}}).subsumptions
    assert len(eq) == 1 and eq[0].equal and (eq[0].narrow, eq[0].wide) == ("x", "y")


def test_shape_graph_bundles_nodes_edges_clusters_and_subsumptions():
    g = shape_graph(COVERS, threshold=0.6)
    assert dict(g.nodes)["env/env:b"] == 4 and dict(g.nodes)["env/env:e"] == 0
    assert g.complete and g.truncated == ()


def test_a_pair_cap_truncates_and_lists_no_subsumption():
    many = {f"i{n:02d}": {"shared"} for n in range(50)}
    g = shape_graph(many, max_pairs=100)
    assert g.truncated == ("pairs",) and len(g.edges) <= 100
    assert g.subsumptions == ()  # all 50 covers are equal, but the graph is incomplete
    # every pair it does list is exact
    assert all(e.weight == 1.0 and e.shared == 1 for e in g.edges)


def test_a_signature_cap_never_reports_a_false_subsumption():
    # narrow holds a signature past the per-item cap that wide lacks: read in full, narrow is not within wide
    covers = {"narrow": {"a", "b", "z"}, "wide": {"a", "b", "c"}}
    full = shape_graph(covers)
    assert not any(s.narrow == "narrow" for s in full.subsumptions)
    cut = shape_graph(covers, max_signatures=2)
    assert "signatures" in cut.truncated and cut.subsumptions == ()


def test_an_item_cap_is_reported():
    g = shape_graph({f"i{n}": {"s"} for n in range(5)}, max_items=3)
    assert "items" in g.truncated and len(g.nodes) == 3 and g.subsumptions == ()


def test_dense_co_cover_is_bounded_in_time():
    # 2,000 items sharing 300 signatures: about 6e8 pair increments unbounded; the work cap stops it early
    dense = {f"i{n:04d}": {f"s{k}" for k in range(300)} for n in range(2000)}
    t0 = time.monotonic()
    g = shape_graph(dense)
    assert time.monotonic() - t0 < 30
    assert set(g.truncated) & {"work", "pairs"} and g.subsumptions == ()


def test_covers_are_type_checked():
    with pytest.raises(TypeError):
        shape_graph({"a": "s1s2"})  # a string is not a set of signatures
    with pytest.raises(TypeError):
        shape_graph({"a": {"s1", 2}})
    with pytest.raises(TypeError):
        shape_graph({1: {"s1"}})


def test_random_covers_match_a_brute_force_reference():
    rng = random.Random(7)
    for _ in range(30):
        covers = {
            f"f{i}": {f"s{rng.randint(0, 9)}" for _ in range(rng.randint(0, 6))}
            for i in range(rng.randint(2, 9))
        }
        g = shape_graph(covers)
        ref_edges = set()
        ref_subs = set()
        names = sorted(covers)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                n = len(covers[a] & covers[b])
                if n:
                    ref_edges.add((a, b, n))
                if covers[a] and covers[a] == covers[b]:
                    ref_subs.add((a, b, True))
                elif covers[a] and covers[a] < covers[b]:
                    ref_subs.add((a, b, False))
                elif covers[b] and covers[b] < covers[a]:
                    ref_subs.add((b, a, False))
        assert {(e.a, e.b, e.shared) for e in g.edges} == ref_edges
        assert {(s.narrow, s.wide, s.equal) for s in g.subsumptions} == ref_subs
