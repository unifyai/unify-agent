"""Symbolic: what the actor reads names no benchmark.

The harness is general-purpose; benchmarks test whether its behaviour
transfers, so their names and actions must not reach what the model reads.
This lint renders the actor's system prompt (plain, with a structured answer,
with an environment's namespaces), its one model-API tool, the in-code tools'
help and the loop's own messages, and fails on any benchmark or dataset name,
or a benchmark's own vocabulary, as a whole word.

The same texts must not tell the agent to check its work against example
inputs and outputs it was given (4 Oct), except the general example-check tool
itself (14 Oct).
"""

from __future__ import annotations

import re

from unify.actor import prompt_builders as pb
from unify.common import tool_registry as R
from unify.common._async_tool import cache_discipline

# Benchmark and dataset names (with or without a separator), and words that
# belong to one benchmark's protocol: ARC's grids and demonstrations.
BENCHMARK_WORDS = re.compile(
    r"\b("
    r"arc|arc[\s_-]?agi|appworld|app[\s_-]world|science[\s_-]?world|"
    r"alf[\s_-]?world|travel[\s_-]?planner|web[\s_-]?arena|swe|swe[\s_-]?bench|"
    r"grids?|demos?|demonstrations?|demonstrate[sd]?"
    r")\b",
    re.IGNORECASE,
)

# Instructions to check, verify or reproduce given examples or input/output
# pairs. A sentence fails when it names examples or known input/output values
# and a check in the same sentence, or uses one of the fixed phrasings.
EXAMPLE_CHECK_PHRASES = re.compile(
    r"\b("
    r"every example|(worked|given|provided|sample) examples?|"
    r"inputs?\s+(and|/|->|→)\s+(the\s+)?(exact\s+)?outputs?|"
    r"(input|output)[\s/-]*pairs?|expected outputs?|known results?"
    r")\b",
    re.IGNORECASE,
)
EXAMPLE_WORDS = re.compile(
    r"\b(examples?|(inputs?|outputs?)\b.{0,40}\bpairs)\b",
    re.IGNORECASE,
)
CHECK_WORDS = re.compile(
    r"\b(check(s|ed|ing)?|verif(y|ies|ied|ying)|reproduc(e|es|ed|ing)|"
    r"match(es|ed|ing)?|compare[sd]?)\b",
    re.IGNORECASE,
)

def _benchmark_words(texts: dict[str, str]) -> list[str]:
    found = []
    for source, text in texts.items():
        for match in BENCHMARK_WORDS.finditer(text):
            context = text[max(0, match.start() - 50) : match.end() + 50]
            found.append(f"{source}: {match.group(0)!r} in {context!r}")
    return found


def _sentences(text: str) -> list[str]:
    """``text`` in sentences, with line wrapping undone."""
    flat = " ".join(text.split())
    return [s for s in re.split(r"(?<=[.!?;:])\s+|\s+-\s+", flat) if s]


def _checks_against_examples(sentence: str) -> bool:
    return bool(
        EXAMPLE_CHECK_PHRASES.search(sentence)
        or (EXAMPLE_WORDS.search(sentence) and CHECK_WORDS.search(sentence)),
    )


def _example_checks(texts: dict[str, str]) -> list[str]:
    found = []
    for source, text in texts.items():
        text = " ".join(text.split())
        for sentence in _sentences(text):
            if _checks_against_examples(sentence):
                found.append(f"{source}: {sentence!r}")
    return found


def _findings(texts: dict[str, str]) -> list[str]:
    """Benchmark words, and instructions to check against given examples."""
    return _benchmark_words(texts) + _example_checks(texts)


# ── the lint itself ──────────────────────────────────────────────────────


def test_the_pattern_catches_names_and_actions_as_whole_words():
    caught = [
        "ARC",
        "arc-agi",
        "AppWorld",
        "ScienceWorld",
        "science world",
        "ALFWorld",
        "TravelPlanner",
        "WebArena",
        "SWE-bench",
        "grid",
        "demos",
        "demonstration",
    ]
    for word in caught:
        assert BENCHMARK_WORDS.search(f"solve the {word} task"), word
    for text in ("search", "archive", "gridlock", "demonstrably", "answer"):
        assert not BENCHMARK_WORDS.search(text), text


def test_the_pattern_catches_instructions_to_check_against_examples():
    caught = [
        # What the switches said before 4 Oct.
        "write a program that produces it, check it against every example "
        "or known result you have, and answer with its output.",
        "When a check fails, fix the program's logic and re-run it (never "
        "hard-code the expected outputs).",
        "Run a stored function on inputs you already have, and check its "
        "output against the evidence you have (worked examples, expected "
        "formats, earlier feedback).",
        "Record an observed input and output in the docstring.",
        # Other phrasings.
        "Verify the function on the examples before storing it.",
        "Make sure the program reproduces each example.",
        "Compare the result with the given input/output pairs.",
        "Test it on the provided examples.",
    ]
    for text in caught:
        assert _example_checks({"t": text}), text
    for text in (
        "Confirm the outcome from real evidence (return values, a re-read).",
        "For example, store `compute_summary(week)`; check that it returns.",
        "Store code that ran and worked.",
        "Each write is refused unless its names resolve and it loads.",
    ):
        assert not _example_checks({"t": text}), text


# ── the actor ────────────────────────────────────────────────────────────


def _actor_texts() -> dict[str, str]:
    from tests.function_manager.primitives.test_environment_namespaces import weather_surface
    from unify.actor.environments import registered_environments
    from unify.function_manager.primitives import register_environment
    from unify.function_manager.primitives.environment import clear_environment_namespaces

    texts = {
        "prompt": pb.build_code_act_prompt(environments={}),
        "prompt, structured": pb.build_code_act_prompt(environments={}, structured=True),
        "execute_code": R.EXECUTE_CODE.description,
        "compression fork": cache_discipline.compression_fork_instruction(),
    }
    for name in R.allow_list("actor"):
        texts[f"tools.help({name!r})"] = R.help_text(name, "actor")
    clear_environment_namespaces()
    register_environment(weather_surface(), source="tests:weather_surface")
    try:
        envs = registered_environments()
        texts["prompt, environment"] = pb.build_code_act_prompt(
            environments={e.namespace: e for e in envs},
        )
    finally:
        clear_environment_namespaces()
    return texts


#: The general example-check tool (the lead's 14 Oct decision: any domain, explicit or implicit examples) is the
#: one text whose subject is checking against examples; it still names no benchmark.
EXAMPLE_CHECK_TOOL = "tools.help('check_examples')"


def test_what_the_actor_reads_names_no_benchmark():
    texts = _actor_texts()
    assert EXAMPLE_CHECK_TOOL in texts
    assert _benchmark_words(texts) == []
    assert _example_checks({k: v for k, v in texts.items() if k != EXAMPLE_CHECK_TOOL}) == []
