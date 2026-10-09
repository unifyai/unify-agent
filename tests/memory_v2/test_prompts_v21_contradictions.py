"""Spec §12.4 (P7 Task 2): the three prompts, and everything else the actor or the writer reads under v2.1, checked
for conflicting instructions. Each rule lives in its role's prompts only, and no retired v2 rule survives.

The surfaces come from P2 (repair and drafts messages), P3 (the rendered section and generated files) and P6
(the CURATE inputs); those tests run once they are integrated."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unify.memory_v2 import prompts_v21 as pv
from unify.memory_v2.integration.switch import v21_conflicts
from unify.memory_v2.sol_pass import sol_tools


def _prompts():
    return {
        "guide": pv.GUIDE_V21,
        "write": pv.write_brief_now(),
        "curate": pv.curate_brief_now(),
    }


def _integrated():
    for name in (
        "unify.memory_v2.repair",
        "unify.memory_v2.drafts",
        "unify.memory_v2.library_export",
        "unify.memory_v2.library_index",
        "unify.memory_v2.code_lint",
        "unify.memory_v2.curate",
        "tests.memory_v2.test_layout",
    ):
        pytest.importorskip(name)


def _surfaces(tmp_path):
    from tests.memory_v2.test_layout import _tree
    from unify.memory_v2 import catalogue, drafts, library_export, repair
    from unify.memory_v2.gate import CHECKS, GateResult
    from unify.memory_v2.integration import prompt

    out = dict(_prompts())
    root = _tree(tmp_path / "lib")
    catalogue.write_files(root, library_export.generated_v21(root))
    out["section"] = prompt.render_memory_v21(root)[0]
    for tool in sol_tools(v21=True):
        out[f"tool:{tool['function']['name']}"] = tool["function"]["description"]
    out["repair"] = repair.repair_message(
        1,
        "/inputs/gate/result-0.md",
        GateResult(False, {c: True for c in CHECKS}),
    )
    out["drafts"] = drafts.drafts_message(
        [{"pass_id": "e1.p0", "items": ["memory.text.dates:parse_date"]}],
        True,
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


def test_the_v21_tool_descriptions_hold_no_retired_rule():
    for tool in sol_tools(v21=True):
        text = tool["function"]["description"]
        for retired in pv.V2_RETIRED:
            assert retired.lower() not in text.lower(), (
                tool["function"]["name"],
                retired,
            )
        assert pv.benchmark_words(text) == [] and pv.example_checks(text) == []


def test_the_v21_check_description_names_no_channel_and_says_what_it_runs():
    """P2 Amendment B: under v2.1 check() also runs the items' own tests and the drawn inputs."""
    check = next(t for t in sol_tools(v21=True) if t["function"]["name"] == "check")[
        "function"
    ]["description"]
    assert "channel" not in check and "static checks" in check
    assert "not the tests" not in check
    assert "the items' own tests and the recorded inputs the gate draws" in check
    v2 = next(t for t in sol_tools() if t["function"]["name"] == "check")["function"][
        "description"
    ]
    assert (
        "covers and their channels" in v2 and "not the tests" in v2
    )  # v2 is unchanged


def test_v2_settings_that_contradict_v21_are_named_only_with_v21_on():
    bad = SimpleNamespace(
        UNIFY_MEMORY_V21="on",
        UNIFY_MEMORY_V2_DOCSTRINGS="on",
        UNIFY_MEMORY_V2_SURFACING="catalogue",
        UNIFY_MEMORY_V2_QA_REPLAY="on",
        UNIFY_MEMORY_V2_QA_FIXTURE_SIZE="on",
    )
    assert v21_conflicts(bad) == [
        "UNIFY_MEMORY_V2_DOCSTRINGS=on",
        "UNIFY_MEMORY_V2_SURFACING=catalogue",  # pragma: allowlist secret (a setting)
        "UNIFY_MEMORY_V2_QA_REPLAY=on",
        "UNIFY_MEMORY_V2_QA_FIXTURE_SIZE=on",
    ]
    assert (
        v21_conflicts(SimpleNamespace(**{**vars(bad), "UNIFY_MEMORY_V21": "off"})) == []
    )
    assert (
        v21_conflicts(
            SimpleNamespace(UNIFY_MEMORY_V21="on", UNIFY_MEMORY_V2_SURFACING="index"),
        )
        == []
    )
