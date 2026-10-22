"""The prompts, and everything else the actor or the writer reads in library mode, checked for conflicting
instructions. Each rule lives in its role's prompts only, and no retired rule survives.
"""

from __future__ import annotations


import pytest

from unify.memory import library_prompts as pv


def _prompts():
    return {
        "guide": pv.GUIDE_LIBRARY,
        "write": pv.write_brief_now(),
        "curate": pv.curate_brief_now(),
    }


def _integrated():
    for name in (
        "unify.memory.repair",
        "unify.memory.drafts",
        "unify.memory.library_export",
        "unify.memory.library_index",
        "unify.memory.code_lint",
        "unify.memory.curate",
        "tests.memory.test_layout",
    ):
        pytest.importorskip(name)


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
    _integrated()
    for name, text in _surfaces(tmp_path).items():
        low = text.lower()
        for retired in pv.V2_RETIRED:
            assert retired.lower() not in low, (name, retired)
        assert pv.benchmark_words(text) == [], name
        assert pv.example_checks(text) == [], name


def test_each_rule_lives_in_its_roles_prompts_only():
    _integrated()
    texts = _prompts()
    assert len({r[0] for r in pv.RULES}) == len(pv.RULES)
    for rule, _spec, owners, phrase in pv.RULES:
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


