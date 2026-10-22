"""The library-mode manifest takes a fixed set of keys at each level and refuses everything else with a reason:
a function declares no input form and no action cover, lists the tests that hold its examples, and declares field
types per fixture field. Nothing the writer writes is silently dropped.
"""

import ast
import dataclasses
import inspect
import textwrap

import pytest

from unify.memory import manifest as mf
from unify.memory.manifest import (
    LIBRARY_ITEM_FIELDS,
    LIBRARY_MANIFEST_FIELDS,
    Manifest,
    ManifestError,
    parse_manifest,
)
from tests.memory.test_manifest_templates import READ_ELSEWHERE

ID = "memory.acct.users:user_id"
TEST = "memory/acct/tests/test_users.py"
DATA = "memory/acct/tests/data/users.json"


def _item(**extra) -> dict:
    return {"item": ID, "kind": "function", "source_episodes": ["e1"], "tests": [TEST], **extra}


def _refusal(manifest: dict) -> str:
    with pytest.raises(ManifestError) as exc:
        parse_manifest(manifest)
    return str(exc.value)


def test_an_input_form_is_refused():
    assert _refusal({"items": [_item(input="observation")]}) == (
        "memory.acct.users:user_id: input is no longer read: a function takes any arguments and its examples are "
        "the calls its tests make on fixtures; remove input"
    )


def test_an_action_cover_is_refused_and_typed_covers_are_kept():
    assert _refusal({"items": [_item(covers=[{"episode": "e1", "type": "diff"}, ["e1", 0]])]}) == (
        "memory.acct.users:user_id: the action cover ['e1', 0] is no longer read: list the tests that call it on "
        "fixtures made with fixture (covers takes only cell, diff and episode covers)"
    )
    got = parse_manifest(
        {"items": [_item(covers=[{"episode": "e1", "type": "cell", "cell": 2}])]},
    ).items[0]
    assert got.covers == [] and [(c.episode, c.type, c.index) for c in got.typed_covers] == [("e1", "cell", 2)]


def test_an_unknown_item_field_is_refused():
    assert _refusal({"items": [_item(examples=["memory/acct/tests/test_users.py::test_one"])]}) == (
        "memory.acct.users:user_id: unknown field 'examples'; an item takes item, kind, source_episodes, tests, "
        "covers, field_types, adopts, role, reader, recipe, cases"
    )


def test_an_unknown_manifest_field_is_refused():
    assert _refusal({"items": [_item()], "summry": "typo"}) == (
        "unknown manifest field 'summry'; the manifest takes items, support, deleted, unlisted, deleted_tests, "
        "skeleton, fixtures, tests_changed, renames, pool, why, summary, kept, retired, aliases"
    )
    # the manifest-wide adopts keeps its own reason
    assert _refusal({"items": [], "adopts": []}) == (
        "adopts belongs on each item that came from a proposal, not on the manifest"
    )


def test_field_types_keys_types_and_kinds_are_checked():
    declared = {f"{DATA}#rows[].month": "month", f"{DATA}##3": "percentage", f"{DATA}#a.b": "hour"}
    got = parse_manifest({"items": [_item(field_types=declared)]}).items[0]
    assert got.field_types == declared and got.input is None
    for key in (
        "rows[].month",
        f"{DATA}#",
        "memory/other/tests/data/users.json#month",
        "memory/acct/tests/users.json#month",
        "memory/acct/tests/data/#month",
        "memory/acct/tests/data/../x.json#month",
        "/memory/acct/tests/data/users.json#month",
    ):
        assert _refusal({"items": [_item(field_types={key: "month"})]}) == (
            f"memory.acct.users:user_id: field_types key {key!r} is not <fixture path>#<field> with the path "
            "under memory/<package>/tests/data/"
        ), key
    assert _refusal({"items": [_item(field_types={f"{DATA}#month": "weekday"})]}) == (
        "memory.acct.users:user_id: unknown semantic type 'weekday'; the fixed list is month, day_of_month, hour, "
        "probability, percentage, nonneg_count, nonneg_money, currency_code"
    )
    note = {"item": "notes/acct/users.md", "kind": "note", "field_types": {f"{DATA}#month": "month"}}
    assert _refusal({"items": [note]}) == "notes/acct/users.md: only functions declare field types"


def test_a_function_lists_the_tests_that_hold_its_examples():
    for tests in ([], None):
        item = _item(tests=tests)
        if tests is None:
            del item["tests"]
        assert _refusal({"items": [item]}) == (
            "memory.acct.users:user_id: a function lists the tests that hold its examples (tests is empty)"
        )
    note = {"item": "notes/acct/users.md", "kind": "note", "source_episodes": ["e1"]}
    assert parse_manifest({"items": [note]}).items[0].tests == []


def _item_keys_read() -> set[str]:
    """The keys ``_parse_library_mode`` reads from an item: ``raw.get("…")`` and ``raw["…"]``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(mf._parse_library_mode)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "raw"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "raw"
            and isinstance(node.slice, ast.Constant)
        ):
            keys.add(node.slice.value)
    return keys


def test_the_field_constants_cover_every_key_the_harness_reads():
    read = _item_keys_read()
    assert {"item", "kind", "tests", "covers", "field_types", "adopts", "source_episodes"} <= read
    assert read <= set(LIBRARY_ITEM_FIELDS)
    assert "input" not in LIBRARY_ITEM_FIELDS
    parsed_top = {f.name for f in dataclasses.fields(Manifest)}
    assert parsed_top | set(READ_ELSEWHERE) | {"fixtures", "items"} <= set(LIBRARY_MANIFEST_FIELDS)
    assert "adopts" not in LIBRARY_MANIFEST_FIELDS
