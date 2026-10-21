"""The memory prompts: one role each, every part present, no benchmark words, no example checking."""

from __future__ import annotations

import re
from decimal import Decimal

import pytest

from unify.memory import library_prompts as pv

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


def _contract_texts():
    """The texts that state the function contract outside the writer's briefs: the fork's instruction and its
    format block, both helper briefs, the curator's system brief, and the actor's two guides.
    """
    from unify.memory import curate, helpers
    from unify.memory.integration import fork

    return {
        "contract": pv.CANDIDATE_CONTRACT,
        "fork": fork.fork_instruction(),
        "fork format": fork.FORMAT_EXAMPLE,
        "helper": helpers.helper_brief(False),
        "helper with similar": helpers.helper_brief(True),
        "curate": curate.curate_system(),
        "guide": pv.GUIDE_LIBRARY,
        "pointer guide": pv.GUIDE_POINTER,
    }


def test_the_contract_texts_carry_no_benchmark_words_and_no_example_checking():
    for name, text in _contract_texts().items():
        assert pv.benchmark_words(text) == [], name
        assert pv.example_checks(text) == [], name


def test_the_fork_format_block_follows_the_contract_it_states():
    """The format's candidate defines MemoryInputError in its module and raises it on another shape; its cases
    return the expected values."""
    import json

    from unify.memory.integration import fork, fork_worker

    blocks, _ = fork_worker.parse_reply(fork.FORMAT_EXAMPLE)
    by_path = {b["path"]: b["content"] for b in blocks}
    ns: dict = {}
    exec(compile(by_path["candidates/double.py"], "double.py", "exec"), ns)
    for case in json.loads(by_path["cases/double.json"]):
        assert ns["double"](*case["inputs"]) == case["expected"]
    with pytest.raises(ns["MemoryInputError"]):
        ns["double"]("21")


#: How the library briefs described a first parameter's recorded input forms, before a function took any arguments.
FORMER_INPUT_FORMS = (
    "a path to a work-tree file",
    "the text of a file or of an output",
    "the raw bytes of a file",
    "an observation value, as recorded",
    "the object through which the request's tools are called",
)


