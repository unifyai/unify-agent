"""The v2.1 prompts (spec §12): one role each, every part present, no benchmark words, no example checking."""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from unify.memory_v2 import prompts_v21 as pv
from unify.memory_v2.integration import prompt as v2_prompt

BRIEF_HEADINGS = {
    "write": [
        "Role",
        "Inputs",
        "Tools",
        "What to read",
        "What to write",
        "Rules",
        "The gate",
        "Repair",
        "Done",
    ],
    "curate": ["Role", "Inputs", "Tools", "What to do", "Rules", "Repair", "Done"],
}
ENV = {"PYTHONPATH": "/memory"}


def _write(**over):
    values = dict(
        view_bytes=8000,
        max_checks=5,
        repair_rounds=2,
        mutation_min=Decimal("0.6"),
        sample_k=8,
        lint_min=4,
        pytest_env=ENV,
        semantic_types="month (integer 1–12)",
    )
    values.update(over)
    return pv.write_brief(**values)


def _curate(**over):
    values = dict(
        view_bytes=8000,
        max_checks=5,
        repair_rounds=2,
        pytest_env=ENV,
        index_tokens=4000,
        overlaps_path="/inputs/curate/overlaps.json",
        suspects_path="/inputs/curate/suspects.json",
        trigger_path="/inputs/curate/trigger.json",
    )
    values.update(over)
    return pv.curate_brief(**values)


def _texts():
    return {"guide": pv.GUIDE_V21, "write": _write(), "curate": _curate()}


def _headings(text):
    return re.findall(r"^# (.+)$", text, re.M)


def test_each_brief_has_its_parts_in_order():
    assert _headings(_write()) == BRIEF_HEADINGS["write"]
    assert _headings(_curate()) == BRIEF_HEADINGS["curate"]


def test_the_guide_has_every_part_and_takes_extra_lines_only_before_the_files_line():
    for part, phrase in pv.GUIDE_PARTS:
        assert phrase in pv.GUIDE_V21, part
    assert pv.GUIDE_V21.startswith("### Memory Library\n\n") and pv.GUIDE_V21.endswith(
        "as usual.\n",
    )
    assert pv.guide_v21() == pv.GUIDE_V21
    more = pv.guide_v21(("- one more line.\n",))
    assert more == pv.GUIDE_V21.replace(
        pv.FILES_LINE,
        "- one more line.\n" + pv.FILES_LINE,
    )
    assert pv.GUIDE_V21.count(pv.FILES_LINE) == 1


def test_numbers_come_from_the_values_given():
    text = _write(
        view_bytes=1234,
        max_checks=7,
        repair_rounds=3,
        mutation_min=Decimal("0.7"),
        sample_k=11,
        lint_min=6,
    )
    for s in (
        "at most 1234 bytes",
        "at most 7 times per pass",
        "up to 3 of them",
        "at least 70% of small mutations",
        "appends 11 more recorded inputs",
        "text of 6 or more characters",
    ):
        assert s in text, s
    assert "<<" not in text and ">>" not in text
    assert "over its budget of 999 tokens" in _curate(index_tokens=999)
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
    assert pv.benchmark_words(" ".join(pv.INPUT_FORMS_V21.values())) == []


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
    from unify.memory_v2.manifest import INPUT_KINDS

    assert list(pv.INPUT_FORMS_V21) == list(INPUT_KINDS)


def test_rendered_from_the_constants_that_enforce_them():
    """Needs P2's repair, P3's library_index, P4's code_lint and MUTATION_MIN and P6's curate: runs once integrated."""
    repair = pytest.importorskip("unify.memory_v2.repair")
    code_lint = pytest.importorskip("unify.memory_v2.code_lint")
    INDEX_VIEW_TOKENS = pytest.importorskip(
        "unify.memory_v2.library_index",
    ).INDEX_VIEW_TOKENS
    pytest.importorskip("unify.memory_v2.curate")
    from unify.memory_v2 import qa, views
    from unify.memory_v2.gate import _PYTEST_ENV

    text = pv.write_brief_now()
    assert f"at most {views.VIEW_BYTES} bytes" in text
    # r5 S6: the v2.1 briefs state no check or repair-round counts (the pass enforces none)
    assert "times per pass" not in text and "while the pass budget lasts" not in text
    assert f"appends {qa.SAMPLE_K} more recorded inputs" in text
    assert f"text of {code_lint.LINT_MIN} or more characters" in text
    assert f"at least {pv._pct(qa.MUTATION_MIN)}% of small mutations" in text
    assert pv._env(_PYTEST_ENV) in text
    assert f"over its budget of {INDEX_VIEW_TOKENS} tokens" in pv.curate_brief_now()


def test_amendment_e_no_read_only_instruction():
    """Read-only is enforced by the mount (P3 Amendment C); no text tells anyone not to write the library."""
    for name, text in _texts().items():
        assert "never write" not in text.lower(), name
        assert "The library is read-only" not in text, name
    assert "are generated by the harness.\n" in _write()
    assert pv.GUIDE_V21.endswith(
        "\n\nWrite your own code and files in your session as usual.\n",
    )


def test_amendment_b_observations_and_what_check_runs():
    text = _write()
    assert "`request`, `observation:<i>`, `cell:<i>`" in text
    assert "its request, every observation" in text
    for name, t in _texts().items():
        assert "does not run the tests" not in t, name
    assert "the items' own tests and the recorded inputs the gate draws" in text
    assert "the items' own tests and the recorded inputs the gate draws" in _curate()
    # the two GUIDE sentences that wait for the lead's decision are kept
    assert (
        "Before you write code for a step, look in the index for an item that does it."
        in pv.GUIDE_V21
    )
    assert "Using the library is optional: use an item when it fits" in pv.GUIDE_V21
