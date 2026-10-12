"""The memory prompts: one role each, every part present, no benchmark words, no example checking."""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from unify.memory import library_prompts as pv
from unify.memory.integration import prompt as v2_prompt

BRIEF_HEADINGS = {
    "write": [
        "Role",
        "Inputs",
        "Tools",
        "The record",
        "How to work",
        "What to write",
        "Rules",
        "The gate",
        "Repair",
        "Limits",
        "Done",
    ],
    "curate": [
        "Role",
        "Inputs",
        "Tools",
        "The record",
        "What to do",
        "Rules",
        "The gate",
        "Repair",
        "Limits",
        "Done",
    ],
}
ENV = {"PYTHONPATH": "/memory"}
OPTIONAL_TOOLS = frozenset({"pool", "similar", "delegate", "replay"})


def _limits(**over):
    values = dict(
        view_bytes=8000,
        cell_s=60.0,
        file_bytes=1024**2,
        total_bytes=20 * 1024**2,
        entries=2000,
        step_guard=500,
        check_floor_s=60.0,
        cells_per_turn=8,
        readers_per_turn=16,
    )
    values.update(over)
    return pv.Limits(**values)


def _write(limits=None, **over):
    """The WRITE brief with every optional input and tool, so each write-only phrase is present."""
    values = dict(
        limits=limits or _limits(),
        mutation_min=Decimal("0.6"),
        sample_k=8,
        lint_min=4,
        pytest_env=ENV,
        staging=True,
        pool=True,
        tools="".join(
            f"- `{t}`: the {t} tool.\n" for t in sorted(OPTIONAL_TOOLS)
        ).rstrip("\n"),
        tool_names=OPTIONAL_TOOLS,
        pool_kinds="`candidate` (a function with its cases)",
    )
    values.update(over)
    return pv.write_brief(**values)


def _curate(limits=None, **over):
    values = dict(
        limits=limits or _limits(),
        pytest_env=ENV,
        overlaps_path="/inputs/curate/overlaps.json",
        suspects_path="/inputs/curate/suspects.json",
        trigger_path="/inputs/curate/trigger.json",
    )
    values.update(over)
    return pv.curate_brief(**values)


def _texts():
    return {"guide": pv.GUIDE_LIBRARY, "write": _write(), "curate": _curate()}


def _headings(text):
    return re.findall(r"^# (.+)$", text, re.M)


def test_each_brief_has_its_parts_in_order():
    assert _headings(_write()) == BRIEF_HEADINGS["write"]
    assert _headings(_curate()) == BRIEF_HEADINGS["curate"]


def test_the_guide_has_every_part_and_takes_extra_lines_only_before_the_files_line():
    for part, phrase in pv.GUIDE_PARTS:
        assert phrase in pv.GUIDE_LIBRARY, part
    assert pv.GUIDE_LIBRARY.startswith(
        "### Memory Library\n\n",
    ) and pv.GUIDE_LIBRARY.endswith(
        "as usual.\n",
    )
    assert pv.guide_library_mode() == pv.GUIDE_LIBRARY
    more = pv.guide_library_mode(("- one more line.\n",))
    assert more == pv.GUIDE_LIBRARY.replace(
        pv.FILES_LINE,
        "- one more line.\n" + pv.FILES_LINE,
    )
    assert pv.GUIDE_LIBRARY.count(pv.FILES_LINE) == 1


def test_numbers_come_from_the_values_given():
    limits = _limits(view_bytes=1234, cell_s=45.0, step_guard=321, entries=77)
    text = _write(
        limits=limits,
        mutation_min=Decimal("0.7"),
        sample_k=11,
        lint_min=6,
    )
    for s in (
        "at most 1234 bytes",
        "at most 45 s",
        "after 321 model and check calls",
        "77 files and directories",
        "at least 70% of small mutations",
        "appends 11 more recorded inputs",
        "text of 6 or more characters",
    ):
        assert s in text, s
    assert "<<" not in text and ">>" not in text
    assert "after 321 model and check calls" in _curate(limits=limits)
    assert '"PYTHONPATH": "/x"' in _write(pytest_env={"PYTHONPATH": "/x"})


def test_an_unfilled_placeholder_is_refused():
    with pytest.raises(ValueError, match="unfilled placeholders"):
        pv._fill("a <<b>> c", {})


def test_the_vocabulary_check_is_live():
    for word in (
        "apis",
        "venmo",
        "ARC",
        "grid",
        "office",
        "payroll",
        "Crafter",
        "ALFWorld",
        "AppWorld",
        "ScienceWorld",
        "TravelPlanner",
        "submission",
        "CORRECT",
    ):
        assert pv.benchmark_words(f"use the {word} tool"), word
    assert pv.benchmark_words("keep it small, correct and easy to navigate") == []
    assert pv.benchmark_words("search the archive of records") == []


def test_no_benchmark_vocabulary_anywhere():
    for name, text in _texts().items():
        assert pv.benchmark_words(text) == [], name
    assert pv.benchmark_words(" ".join(pv.INPUT_FORMS_LIBRARY.values())) == []


def test_no_example_verification_instruction_and_the_check_is_live():
    assert pv.example_checks(
        v2_prompt.GUIDE,
    )  # v2's "check the example first" is caught
    for bad in (
        "Verify your function against the training examples.",
        "Test it on the task's example pairs first.",
        "check each example before you answer",
        "Run the given examples through it.",
    ):
        assert pv.example_checks(bad), bad
    for name, text in _texts().items():
        assert pv.example_checks(text) == [], name


def test_no_role_carries_another_roles_instructions():
    t = {k: v.lower() for k, v in _texts().items()}
    for marker in pv.WRITE_ONLY:
        assert (
            marker in t["write"]
            and marker not in t["curate"]
            and marker not in t["guide"]
        ), marker
    for marker in pv.CURATE_ONLY:
        assert (
            marker in t["curate"]
            and marker not in t["write"]
            and marker not in t["guide"]
        ), marker
    for marker in pv.WRITER_ONLY:
        assert (
            marker in t["write"] and marker in t["curate"] and marker not in t["guide"]
        ), marker


def test_input_forms_match_the_manifest():
    from unify.memory.manifest import INPUT_KINDS

    assert list(pv.INPUT_FORMS_LIBRARY) == list(INPUT_KINDS)


def test_rendered_from_the_constants_that_enforce_them():
    """Needs repair, library_index, code_lint with MUTATION_MIN, and curate: runs once they are integrated."""
    repair = pytest.importorskip("unify.memory.repair")
    code_lint = pytest.importorskip("unify.memory.code_lint")
    INDEX_VIEW_TOKENS = pytest.importorskip(
        "unify.memory.library_index",
    ).INDEX_VIEW_TOKENS
    pytest.importorskip("unify.memory.curate")
    from unify.memory import qa, views
    from unify.memory.gate import _PYTEST_ENV

    from unify.memory import writer_pass

    text = pv.write_brief_now()
    assert f"at most {views.VIEW_BYTES} bytes" in text
    # the briefs state no check or repair-round counts (the pass enforces none)
    assert "times per pass" not in text and "while the pass" not in text
    assert f"appends {qa.SAMPLE_K} more recorded inputs" in text
    assert f"text of {code_lint.LINT_MIN} or more characters" in text
    assert f"at least {pv._pct(qa.MUTATION_MIN)}% of small mutations" in text
    assert f"after {writer_pass.STEP_GUARD} model and check calls" in text
    assert pv._env(_PYTEST_ENV) in text
    assert INDEX_VIEW_TOKENS and repair  # imported: the constants exist
    assert (
        f"after {writer_pass.STEP_GUARD} model and check calls" in pv.curate_brief_now()
    )


def test_no_read_only_instruction():
    """Read-only is enforced by the mount; no text tells anyone not to write the library."""
    for name, text in _texts().items():
        assert "never write" not in text.lower(), name
        assert "The library is read-only" not in text, name
    assert "generated by the harness" in _write()
    assert pv.GUIDE_LIBRARY.endswith(
        "\n\nWrite your own code and files in your session as usual.\n",
    )


def test_the_guide_sentences_are_kept():
    for name, t in _texts().items():
        assert "does not run the tests" not in t, name
    # the two GUIDE sentences still awaiting a decision are kept
    assert (
        "Before you write code for a step, look in the index for an item that does it."
        in pv.GUIDE_LIBRARY
    )
    assert "Using the library is optional: use an item when it fits" in pv.GUIDE_LIBRARY
