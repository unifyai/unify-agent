"""The prompts, and everything else the actor or the writer reads in library mode, checked for conflicting
instructions. Each rule lives in its role's prompts only, and no retired rule survives.
"""

from __future__ import annotations



from unify.memory import library_prompts as pv
from tests.memory import prompt_guards as guards


def _prompts():
    return {
        "guide": pv.GUIDE_LIBRARIAN,
        "write": pv.write_brief_now(),
    }


def _surfaces(tmp_path):
    from tests.memory.test_layout import _tree
    from unify.memory import catalogue, library_export, repair
    from unify.memory.gate import CHECKS, GateResult
    from unify.memory.integration import prompt

    out = dict(_prompts())
    root = _tree(tmp_path / "lib")
    catalogue.write_files(root, library_export.generated_library_mode(root))
    out["section"] = prompt.render_memory_section(root)[0]
    from unify.common.tool_registry import allow_list, registry

    for name in allow_list("writer", frozenset({"helper", "similar"})):
        out[f"tool:{name}"] = registry()[name].description
    out["repair"] = repair.repair_message(
        1,
        "/inputs/gate/result-0.md",
        GateResult(False, {c: True for c in CHECKS}),
    )
    return out


def test_no_v2_rule_survives(tmp_path):
    for name, text in _surfaces(tmp_path).items():
        low = text.lower()
        for retired in pv.V2_RETIRED:
            # a retired file name is matched in its own case: v2's NOTES.md is not the fork's staged notes.md
            # (lessons), which rev 2's brief names
            if retired.endswith(".md"):
                assert retired not in text, (name, retired)
            else:
                assert retired.lower() not in low, (name, retired)
        assert guards.benchmark_words(text) == [], name
        assert guards.example_checks(text) == [], name


#: Rules rev 2's in-code prompts do not state: CURATE's (no curator), the mechanisms rev 2 removed (recorded
#: replays, drawn inputs, the literal rule) and ``memory.find`` (the library's calls are ask, items and get), and the
#: status meanings, which the v2 guide stated and the in-code guide leaves to the library's own help.
REV2_NOT_STATED = frozenset({
    "status meanings",
    "find matches structure only",
    "procedures are replayed",
    "drawn inputs appended",
    "literal rule",
    "merge keeps aliases",
    "standing tests green",
    "retired fixtures stay used",
    "two episodes decide",
})
#: Rules the in-code guide leaves to the library's own help (``help(memory)``): stated there.
IN_LIBRARY_HELP = frozenset({"fits compares structure only"})


def test_each_rule_lives_in_its_roles_prompts_only():
    from unify.memory import library_export as lx

    texts = _prompts()
    help_text = lx.stub_source(librarian=True).decode() + lx.package_source(librarian=True).decode()
    assert len({r[0] for r in pv.RULES}) == len(pv.RULES)
    for rule, _spec, owners, phrase in pv.RULES:
        if rule in REV2_NOT_STATED:
            assert all(phrase not in text for text in texts.values()), rule
            continue
        if rule in IN_LIBRARY_HELP:
            assert phrase in help_text and all(phrase not in text for text in texts.values()), rule
            continue
        for role, text in texts.items():
            if role in owners:
                assert phrase in text, (rule, role)
            else:
                assert phrase not in text, (rule, role)


def test_the_write_brief_points_to_where_an_outcome_is_and_never_to_a_reading_list():
    """The batch map names no reading list; the brief points to the parts where each episode ended."""
    brief = " ".join(pv.write_brief_now().split())
    assert "required_parts" not in brief and "neither covered" not in brief
    # rev 2's wording: the brief points to where each episode ended
    assert "a row's `end_parts` show where the episode ended" in brief


