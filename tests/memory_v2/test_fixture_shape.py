"""The examples' fixture check compares shapes exactly, with no information floor (v2.1 re-review N1).

``find`` needs a named key before it says anything (:func:`unify.memory_v2.memory_helper.compare`), which is
right for retrieval and wrong for a fixture: a headerless table, a raw grid or a list of scalars has no named
key, yet a fixture of exactly its shape is the recorded input. :func:`same_shape` compares the whole signature
(named and unnamed key paths with their leaf types) and a file's format, delimiter, header, columns and
column types; lengths and counts are left out. The gate accepts a fixture whose shape is the same as a
recorded input's, or a keyed structural match (:func:`unify.memory_v2.gate.fixture_fits`).

Seeded generators: random value schemas (each instantiated twice with fresh values, and once with one leaf
type changed) and random headerless tables.
"""

import json
import random

from unify.memory_v2.gate import fixture_fits
from unify.memory_v2.memory_helper import (
    compare,
    file_shape,
    same_shape,
    shape_key,
    value_shape,
)

LEAVES = ("int", "float", "str", "bool")
KEYS = ("id", "grid", "rows", "name", "score", "items", "cells", "at")


def _schema(rng: random.Random, depth: int = 0):
    roll = rng.random()
    if depth >= 3 or roll < 0.35:
        return rng.choice(LEAVES)
    if roll < 0.7:
        return ("list", _schema(rng, depth + 1))
    keys = rng.sample(KEYS, rng.randint(1, 4))
    return ("dict", {k: _schema(rng, depth + 1) for k in sorted(keys)})


def _value(rng: random.Random, schema):
    if schema == "int":
        return rng.randint(-1000, 1000)
    if schema == "float":
        return rng.randint(-1000, 1000) + 0.5
    if schema == "str":
        return "".join(rng.choice("abcdefgh") for _ in range(rng.randint(1, 6)))
    if schema == "bool":
        return rng.random() < 0.5
    if schema[0] == "list":
        return [_value(rng, schema[1]) for _ in range(rng.randint(1, 5))]
    return {k: _value(rng, sub) for k, sub in schema[1].items()}


def _leaves(schema, path=()):
    if isinstance(schema, str):
        yield path
    elif schema[0] == "list":
        yield from _leaves(schema[1], path + (0,))
    else:
        for k, sub in schema[1].items():
            yield from _leaves(sub, path + (k,))


def _retyped(schema, path):
    if not path:
        return next(t for t in LEAVES if t != schema)
    if schema[0] == "list":
        return ("list", _retyped(schema[1], path[1:]))
    return ("dict", {**schema[1], path[0]: _retyped(schema[1][path[0]], path[1:])})


def _wrapped(schema):
    """A top-level value is a dict or a list (what value_shape shapes)."""
    return schema if not isinstance(schema, str) else ("list", schema)


def test_same_schema_values_have_the_same_shape_and_a_retyped_leaf_does_not():
    rng = random.Random(20261008)
    floorless = 0
    for _ in range(400):
        schema = _wrapped(_schema(rng))
        a, b = value_shape(_value(rng, schema)), value_shape(_value(rng, schema))
        assert a is not None and b is not None
        assert same_shape(a, b), (schema, a, b)
        assert fixture_fits(a, b)
        if compare(a, b) is None:
            floorless += 1  # find's floor would have refused this identical shape
        path = rng.choice(list(_leaves(schema)))
        c = value_shape(_value(rng, _retyped(schema, path)))
        assert not same_shape(a, c), (schema, path)
    assert (
        floorless >= 40
    )  # the generator reaches the low-information shapes N1 is about


def _table(rng: random.Random, types: list[str], rows: int, header: bool) -> bytes:
    def cell(t):
        if t == "int":
            return str(rng.randint(0, 999))
        if t == "float":
            return f"{rng.randint(0, 999)}.{rng.randint(1, 9)}"
        if t == "date":
            return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        return rng.choice(["ab", "cd", "ef"])

    lines = [",".join(f"c{i}" for i in range(len(types)))] if header else []
    lines += [",".join(cell(t) for t in types) for _ in range(rows)]
    return ("\n".join(lines) + "\n").encode()


def test_headerless_tables_of_the_same_column_types_have_the_same_shape():
    rng = random.Random(8)
    for _ in range(200):
        width = rng.randint(2, 6)
        types = [rng.choice(["int", "float", "date", "str"]) for _ in range(width)]
        if all(t == "str" for t in types):
            types[0] = "int"  # an all-text first row reads as a header
        recorded = file_shape("rec.csv", _table(rng, types, rng.randint(2, 30), False))
        fixture = file_shape(
            "fixture.csv",
            _table(rng, types, rng.randint(2, 5), False),
        )
        assert (
            recorded["shape"]["header"] is False and fixture["shape"]["header"] is False
        )
        assert compare(fixture, recorded) is None  # no named key: find says nothing
        assert same_shape(fixture, recorded) and fixture_fits(fixture, recorded)
        headed = file_shape("fixture.csv", _table(rng, types, 2, True))
        assert headed["shape"]["header"] is True
        assert not same_shape(headed, recorded) and not fixture_fits(headed, recorded)
        other = list(types)
        other[rng.randrange(width)] = "str" if types[0] != "str" else "int"
        if other != types and not all(t == "str" for t in other):
            moved = file_shape("fixture.csv", _table(rng, other, 3, False))
            assert not fixture_fits(moved, recorded)


def test_the_three_shapes_of_the_review():
    """N1's probes: each fixture has exactly the recorded shape and fits; find's floor would refuse each."""
    grid_rec, grid_fx = value_shape([[0, 1], [1, 0]]), value_shape(b"[[1, 0], [0, 1]]")
    list_rec, list_fx = value_shape([4, 5, 6]), value_shape("[1, 2, 3]")
    csv_rec = file_shape("rec.csv", b"5,6\n7,8\n")
    csv_fx = file_shape("pairs.csv", b"1,2\n3,4\n")
    tail_rec = file_shape("", b"5,6\n7,8\n")  # a recorded shell output tail
    for fx, rec in (
        (grid_fx, grid_rec),
        (list_fx, list_rec),
        (csv_fx, csv_rec),
        (csv_fx, tail_rec),
    ):
        assert compare(fx, rec) is None
        assert same_shape(fx, rec) and fixture_fits(fx, rec)
    assert not fixture_fits(
        file_shape("f.csv", b"a,b\n1,2\n"),
        csv_rec,
    )  # headed vs headerless
    assert not fixture_fits(value_shape("[[1.5, 0.5]]"), grid_rec)  # float cells
    assert not fixture_fits(value_shape("[1, 2, 3]"), grid_rec)  # a list is not a grid


def test_shape_key_leaves_out_lengths_counts_and_encodings():
    a = file_shape("a.csv", b"\xef\xbb\xbfx,y\n1,2\n" + b"3,4\n" * 50)
    b = file_shape("b.csv", b"x,y\n1,2\n")
    assert shape_key(a) == shape_key(b)
    assert json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True)
