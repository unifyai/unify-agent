"""The prompts, and everything else the actor or the writer reads in library mode, checked for conflicting
instructions. Each rule lives in its role's prompts only, and no retired rule survives.
"""

from __future__ import annotations

from unify.memory import library_prompts as pv
from tests.memory import prompt_guards as guards


#: Each rule the prompts state: (rule, topic, owners, the phrase that states it). The phrase must be in every
#: owner's text and in no other prompt.
RULES: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    (
        "fits compares structure only",
        "how an item's applicability check matches",
        ("guide",),
        "It compares the structure of what you",
    ),
    (
        "an input refusal means do the step yourself",
        "input refusals",
        ("guide",),
        "Then do that step with your own code",
    ),
    (
        "what to store is the writer's decision",
        "what to store",
        ("write",),
        "What you store is your decision",
    ),
    (
        "start from the actor's code",
        "where code comes from",
        ("write",),
        "Start from the working model's own code",
    ),
    (
        "outcomes are judged from the records",
        "outcome evidence",
        ("write",),
        "Judge outcomes from the records",
    ),
    ("failure-only work is a note", "notes", ("write",), "becomes a note"),
    (
        "a pass never removes library items",
        "pass roles",
        ("write",),
        "A pass never removes library items",
    ),
    ("trust order", "trusted sources", ("write",), "in this order of trust"),
    (
        "exact trusted assertion",
        "test assertions",
        ("write",),
        "equals an exact trusted value",
    ),
    (
        "no self-oracle",
        "test assertions",
        ("write",),
        "Never compute an expected value with the function's own expression",
    ),
    ("deterministic tests", "qa determinism", ("write",), "never assert on the clock"),
    ("no credentials", "credentials", ("write",), "Never store credentials"),
    ("why trailer", "commit trailer", ("write",), '"why"'),
    ("repair rounds", "repair rounds", ("write",), "you get a repair round"),
    ("drafts", "drafts", ("write",), "kept as a draft"),
)

#: Instructions and surfaces of the earlier memory design that are retired: none may appear in any text the actor or
#: the writer reads in library mode.
V2_RETIRED = (
    "do not guess",
    "Do not write job functions",
    "Do not write workflow notes",
    "Tend the library too",
    "env/<channel>",
    "memory_channels",
    "library.json",
    "previous_gate.json",
    "unify_memory_testkit",
    "memlab",
    "NOTES.md",
    "workflows/",
    "catalog()",
    "check the example first",
    "candidates, not authority",
    "recorded for the next consolidation",
    "proposals/",
    "memory.diff",
)


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
        for retired in V2_RETIRED:
            # a retired file name is matched in its own case: v2's NOTES.md is not the fork's staged notes.md
            # (lessons), which rev 2's brief names
            if retired.endswith(".md"):
                assert retired not in text, (name, retired)
            else:
                assert retired.lower() not in low, (name, retired)
        assert guards.benchmark_words(text) == [], name
        assert guards.example_checks(text) == [], name


#: Rules the in-code guide leaves to the library's own help (``help(memory)``): stated there.
IN_LIBRARY_HELP = frozenset({"fits compares structure only"})


def test_each_rule_lives_in_its_roles_prompts_only():
    from unify.memory import library_export as lx

    texts = _prompts()
    help_text = lx.stub_source(librarian=True).decode() + lx.package_source(librarian=True).decode()
    assert len({r[0] for r in RULES}) == len(RULES)
    for rule, _spec, owners, phrase in RULES:
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


