"""Guards the prompt tests apply to every text an agent reads: no benchmark or environment names, no verdict text, no
instruction to check a task's examples, no economy cues, and a bound on inlined lists. They live with the tests: the
harness never reads them.
"""

from __future__ import annotations

import re

#: Words no writer brief or first message may hold: economy cues and the retired reading-quota machinery.
BANNED_WORDS = (
    "cheaply",
    "budget",
    "coverage",
    "required_parts",
    "experience_tokens",
    "covered or dismissed",
)

#: The most bytes an inlined list may hold (with ``library_prompts.MAX_INLINE_LIST`` items). The Tools section,
#: rendered once per run from the tool registry, is stable text and is not such a list.
MAX_INLINE_LIST_BYTES = 1000

#: Names and terms of particular benchmarks and environments (case-insensitive, whole words).
_BENCHMARK = re.compile(
    r"\b(apis|venmo|arc|grids?|office|payroll|crafter|alfworld|appworld|scienceworld|travelplanner|webshop|"
    r"minecraft|sokoban|submission)\b",
    re.IGNORECASE,
)
#: An environment's verdict text, which the harness never matches: upper case only.
_VERDICT = re.compile(r"\bCORRECT\b")

#: Instructions to check a task's examples. A function docstring's ``Example:`` line names one of its tests, not a
#: task's example, and matches none of these.
_EXAMPLE_CHECKS = (
    re.compile(
        r"\b(check|verify|validate|confirm|test|try|run)\w*\b[^.\n]{0,60}?\bexamples?\b",
        re.I,
    ),
    re.compile(r"\bexamples?\b[^.\n]{0,20}\bfirst\b", re.I),
    re.compile(
        r"\b(training|train|worked|given|task'?s?|request'?s?|input[- ]output)\s+(examples?|pairs?)\b",
        re.I,
    ),
)


def benchmark_words(text: str) -> list[str]:
    return [m.group(0) for m in _BENCHMARK.finditer(text)] + [m.group(0) for m in _VERDICT.finditer(text)]


def example_checks(text: str) -> list[str]:
    return [m.group(0) for rx in _EXAMPLE_CHECKS for m in rx.finditer(text)]
