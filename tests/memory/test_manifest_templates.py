"""Every manifest field a brief shows a model is one the harness reads, at the level the brief shows it: item
fields are fields of a parsed item, and a top-level field is either parsed by the manifest parser or read by the
module named here. A field the brief shows that nothing reads would be silently dropped.
"""

import dataclasses
import re

from unify.memory import library_prompts as lp
from unify.memory.manifest import Manifest, ManifestItem, parse_manifest
from tests.memory.test_pool_start import STAGED

#: Top-level fields the manifest parser leaves to another reader, with the module that reads them.
READ_ELSEWHERE = {
    "tests_changed": "library_checks.py",
    "renames": "library_checks.py",
    "pool": "pool.py",
    "why": "writer_pass.py",
    "summary": "writer_pass.py",
    "kept": "curate.py",
    "retired": "curate.py",
    "aliases": "curate.py",
}


def _template(brief: str) -> str:
    start = brief.index('{"items": [')
    return brief[start : brief.index("}\n", start) + 1]


def _levels(template: str) -> tuple[set[str], set[str]]:
    items = template[template.index("[{") : template.index("}]") + 2]
    item_keys = set(re.findall(r'"([a-z_]+)":', items))
    top_keys = set(re.findall(r'"([a-z_]+)":', template.replace(items, "[]"))) - {
        "items",
    }
    return item_keys, top_keys


def _briefs() -> dict[str, str]:
    return {
        "write": lp.write_brief_now(staging=True, pool=True, helper=True, similar=True),
        "write plain": lp.write_brief_now(),
        "curate": lp.curate_brief_now(),
    }


def test_every_field_a_brief_shows_is_read_at_its_level():
    from pathlib import Path

    import unify.memory as pkg

    item_fields = {f.name for f in dataclasses.fields(ManifestItem)}
    parsed_top = {f.name for f in dataclasses.fields(Manifest)}
    root = Path(pkg.__file__).parent
    for name, brief in _briefs().items():
        item_keys, top_keys = _levels(_template(brief))
        assert item_keys and item_keys <= item_fields, (name, item_keys - item_fields)
        for key in top_keys - parsed_top:
            assert key in READ_ELSEWHERE, (name, key)
            assert f'"{key}"' in (root / READ_ELSEWHERE[key]).read_text(), (name, key)


def test_the_write_template_round_trips_through_the_parser():
    item = {
        "item": "memory.acct.users:user_id",
        "kind": "function",
        "input": "text",
        "source_episodes": ["e1"],
        "tests": ["memory/acct/tests/test_users.py"],
        "covers": [["e1", 0]],
        "adopts": [STAGED],
    }
    item_keys, _ = _levels(_template(_briefs()["write"]))
    assert set(item) == item_keys
    man = parse_manifest(
        {"items": [item], "support": [], "why": "w", "summary": "s"},
        library_mode=True,
    )
    got = man.items[0]
    assert (got.item, got.kind, got.input, got.source_episodes, got.tests) == (
        item["item"],
        "function",
        "text",
        ["e1"],
        ["memory/acct/tests/test_users.py"],
    )
    assert got.adopts == ["staging:e2/candidates/fill.py"] and got.covers == [("e1", 0)]


def test_the_fork_and_the_helpers_are_told_the_function_contract_the_gate_enforces():
    """A function takes any arguments and refuses other shapes with MemoryInputError; its cases are recorded values,
    so the writer can copy them into fixtures. No text names a first-parameter input form any more.
    """
    from unify.memory import curate, helpers
    from unify.memory.integration import fork
    from unify.memory.manifest import INPUT_KINDS

    texts = {
        "fork": fork.fork_instruction(),
        "helper": helpers.helper_brief(False),
        "helper with similar": helpers.helper_brief(True),
        "curate": curate.curate_system(),
    }
    for name, text in texts.items():
        assert "MemoryInputError" in text and "any arguments" in text, name
        assert "first parameter" not in text and "recorded form" not in text, name
        for form in INPUT_KINDS:
            assert f"`{form}`" not in text, (name, form)
    assert lp.CANDIDATE_CONTRACT in texts["fork"]
    assert "Each value must appear in this conversation as it is" in texts["fork"]
    for name in ("helper", "helper with similar"):
        assert (
            lp.CANDIDATE_CONTRACT.replace("recorded here", "recorded in the episodes")
            in texts[name]
        )
        assert "each value as the records hold it" in texts[name]
