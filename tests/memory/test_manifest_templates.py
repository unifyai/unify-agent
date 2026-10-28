"""Every manifest field a brief shows a model is one the harness reads, at the level the brief shows it: item
fields are fields of a parsed item, and a top-level field is either parsed by the manifest parser or read by the
module named here. A field the brief shows that nothing reads would be silently dropped.
"""

import dataclasses
import re

import pytest

from unify.memory import library_prompts as lp
from unify.memory.manifest import (
    LABEL_ITEM_FIELDS,
    LIBRARY_ITEM_FIELDS,
    Manifest,
    ManifestError,
    ManifestItem,
    parse_manifest,
)

STAGED = "/inputs/staging/e2/candidates/fill.py"

#: Top-level fields the manifest parser leaves to another reader, with the module that reads them.
READ_ELSEWHERE = {
    "tests_changed": "library_checks.py",
    "renames": "library_checks.py",
    "pool": "pool.py",
    "why": "writer_pass.py",
    "summary": "writer_pass.py",
}


def _template(brief: str) -> str:
    """The manifest template: the JSON object on the line that opens with ``{"items": [`` (rev 2 follows it on the same
    line with the optional top-level keys)."""
    start = brief.index('{"items": [')
    line = brief[start : brief.index("\n", start)]
    return line[: line.rindex("}") + 1]


def _levels(template: str) -> tuple[set[str], set[str]]:
    items = template[template.index("[{") : template.index("}]") + 2]
    item_keys = set(re.findall(r'"([a-z_]+)":', items))
    top_keys = set(re.findall(r'"([a-z_]+)":', template.replace(items, "[]"))) - {
        "items",
    }
    return item_keys, top_keys


def _briefs() -> dict[str, str]:
    return {
        "write": lp.write_brief_now(helper=True, similar=True),
        "write plain": lp.write_brief_now(),
    }


def test_every_field_a_brief_shows_is_read_at_its_level():
    from pathlib import Path

    import unify.memory as pkg

    item_fields = {f.name for f in dataclasses.fields(ManifestItem)}
    parsed_top = {f.name for f in dataclasses.fields(Manifest)}
    root = Path(pkg.__file__).parent
    # rev 2: an item may also state its label fields, which the gate parses (labels=True) into ``stated``
    label_fields = set(LABEL_ITEM_FIELDS)
    assert "stated" in item_fields
    for name, brief in _briefs().items():
        item_keys, top_keys = _levels(_template(brief))
        assert item_keys and item_keys <= item_fields | label_fields, (name, item_keys - item_fields - label_fields)
        assert item_keys <= set(LIBRARY_ITEM_FIELDS) | label_fields, (
            name,
            item_keys - set(LIBRARY_ITEM_FIELDS) - label_fields,
        )
        for key in top_keys - parsed_top:
            assert key in READ_ELSEWHERE, (name, key)
            assert f'"{key}"' in (root / READ_ELSEWHERE[key]).read_text(), (name, key)


#: A function item with every key the brief templates show, plus the documented ``field_types``; the values are
#: valid for the library-mode parser.
FIXTURE = "memory/acct/tests/data/users.json"
ITEM_VALUES = {
    "item": "memory.acct.users:user_id",
    "kind": "function",
    "source_episodes": ["e1", "e2"],
    "tests": ["memory/acct/tests/test_users.py"],
    "covers": [{"episode": "e1", "type": "diff"}],
    "adopts": [{"ref": STAGED, "how": "generalised"}],
    "field_types": {f"{FIXTURE}#rows[].month": "month", f"{FIXTURE}##3": "percentage"},
}
#: The label fields a template shows, with values the gate's parser (labels=True) takes, and what it keeps of each.
LABEL_VALUES = {
    "situation": "reading a user id from an account response",
    "inputs": ["response.user_id"],
    "constants": [{"value": "the user_id key", "source": "e1/action:0"}],
    "effects": "reads",
}
LABEL_KEPT = {**LABEL_VALUES, "constants": ["the user_id key (source: e1/action:0)"]}


def _round_trip(name: str) -> None:
    from unify.memory.procedures import parse_cover

    item_keys, _ = _levels(_template(_briefs()[name]))
    keys = item_keys | {"field_types"}
    labels = keys & set(LABEL_ITEM_FIELDS)
    keys -= labels
    assert keys <= set(LIBRARY_ITEM_FIELDS), (name, keys - set(LIBRARY_ITEM_FIELDS))
    item = {k: ITEM_VALUES[k] for k in sorted(keys)}
    stated = {k: LABEL_VALUES[k] for k in sorted(labels)}
    man = parse_manifest(
        {"items": [{**item, **stated}], "support": [], "why": "w", "summary": "s"},
        labels=True,
    )
    got = man.items[0]
    assert (got.item, got.kind, got.source_episodes, got.tests) == (
        item["item"],
        "function",
        item["source_episodes"],
        item["tests"],
    )
    assert (
        got.typed_covers == [parse_cover(c) for c in item["covers"]]
        and got.covers == []
    )
    assert got.field_types == item["field_types"] and not hasattr(got, "input")
    assert got.stated == {k: LABEL_KEPT[k] for k in labels}
    if "adopts" in item:
        assert got.adopts == ["staging:e2/candidates/fill.py"]
        assert got.adopt_how == {"staging:e2/candidates/fill.py": "generalised"}
    for extra in ("examples", "path", "signature", "inputs"):
        with pytest.raises(ManifestError) as exc:
            parse_manifest({"items": [{**item, extra: []}]})
        assert str(exc.value) == (
            f"memory.acct.users:user_id: unknown field {extra!r}; an item takes item, kind, source_episodes, "
            "tests, covers, field_types, adopts, role, reader, recipe, cases"
        )


def test_the_write_template_round_trips_through_the_parser():
    """Every item key the WRITE template shows, and the field_types key its bullets document, parse into the item
    as given; a key outside the item fields is refused, never dropped."""
    _round_trip("write")
    _round_trip("write plain")


def test_no_brief_keeps_a_placeholder_and_write_brief_takes_no_input_forms():
    import inspect

    for name, brief in _briefs().items():
        assert re.search(r"<<[^<>]*>>", brief) is None, name
    assert "input_forms" not in inspect.signature(lp.write_brief).parameters
    assert "<<input_forms>>" not in inspect.getsource(lp)


def test_the_write_brief_names_the_semantic_types_field_types_takes():
    from unify.memory.manifest import SEMANTIC_TYPES, describe_semantic_types

    # rev 2: the brief names `field_types` and points to /inputs/README.md, which carries the types (constant per run)
    brief = _briefs()["write"]
    assert "`field_types`" in brief and "(form: /inputs/README.md)" in brief
    readme = lp.input_pointer_files()[lp.INPUTS_README_FILE]
    assert describe_semantic_types() in readme
    assert all(name in readme for name in SEMANTIC_TYPES)


def test_the_write_brief_states_the_material_and_the_reach_as_facts():
    from unify.memory import writer_pass

    for brief in (
        lp.write_brief_now(helper=True, similar=True),
        lp.write_brief_now(),
    ):
        # rev 2 wording (tools are called in code): the same two facts, the material and the reach
        assert "You can also build from any recorded episode with no proposal behind it." in brief
        assert "At any point you can open any recorded episode, in this batch or not" in brief
        assert "`tools.overview` and `tools.read_episode` take any recorded id" in brief
    assert "`tools.similar` finds others with matching code" in lp.write_brief_now(
        similar=True,
    )
    with_helpers = lp.write_brief_now(helper=True)
    # the wall clock is a fact; there is no finishing window and no time allowance for helpers
    assert "the helpers you start work within the same wall clock" in with_helpers
    for brief in (with_helpers, lp.write_brief_now()):
        assert "finishing window" not in brief and "last quarter" not in brief
    assert not hasattr(writer_pass, "finish_window_s") and not hasattr(
        writer_pass,
        "FINISH_WINDOW_SHARE",
    )
    _template(
        with_helpers,
    )  # the manifest template still parses out of the changed brief
