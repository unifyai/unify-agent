"""No text an agent reads steers what is stored, in either direction.

Agents decide what to store; the harness states what each tool does. No prompt, brief, card, tool text or help topic
discourages, bars or blocks keeping work-, request- or instance-specific artifacts (for reuse, memory or transfer),
and none urges keeping them either. The scan covers every string the memory agents' briefs and the actor's memory
texts are built from (module constants, docstrings the package's ``help`` shows, card templates, tool texts) and the
rendered briefs themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Modules whose strings an agent reads: briefs, guides, cards, tool texts and help topics.
MODULES = (
    "unify/memory/library_prompts.py",
    "unify/memory/librarian.py",
    "unify/memory/integration/fork.py",
    "unify/memory/helpers.py",
    "unify/memory/kinship.py",
    "unify/memory/library_helper.py",
    "unify/memory/library_ask.py",
    "unify/memory/land_capture.py",
    "unify/memory/attention_tools.py",
    "unify/memory/manifest.py",
    "unify/memory/when_to_use.py",
    "unify/common/tool_registry.py",
    "unify/agents/run_record.py",
)

#: Wording that steers against keeping specific work (generalise it, it holds only for one case, it is not reusable).
AGAINST = (
    "not reusable",
    "only for that case",
    "may hold only",
    "instance-gated",
    "generalise it",
    "generalise them",
    "generalize it",
    "in general terms",
    "in general words",
    "in general form",
    "task-specific",
    "episode-specific",
    "instance-specific",
    "request-specific",
    "specific to this",
    "hardcode",
    "hard-code",
    "beyond any one",
    "beyond this request",
    "not its scope",
    "derive it",
    "wrong on the next request",
    "a constant only where",
    "values that vary become parameters",
    "refuse shapes",
)
#: Wording that steers towards keeping it (per task, for future instances of this task, whatever is likely to repeat).
TOWARDS = (
    "per task",
    "for future instances",
    "instances of this task",
    "keyed by task",
    "task id",
    "similar requests later",
    "similar work later",
    "likely to repeat",
    "store anything",
)


def _strings(path: str) -> list[tuple[int, str]]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _hits(text: str) -> list[str]:
    low = " ".join(text.lower().split())
    return [p for p in (*AGAINST, *TOWARDS) if p in low]


@pytest.mark.parametrize("path", MODULES)
def test_no_string_an_agent_reads_steers_storage(path):
    found = [f"{path}:{line}: {_hits(text)}" for line, text in _strings(path) if _hits(text)]
    assert not found, found


def test_no_rendered_brief_steers_storage():
    from unify.memory import helpers, kinship, librarian
    from unify.memory import library_prompts as lp
    from unify.memory.integration import fork

    texts = {
        "write": lp.write_brief_now(helper=False),
        "write+helper": lp.write_brief_now(helper=True),
        "helper": helpers.helper_brief(similar=True),
        "fork": fork.INSTRUCTION,
        "segment": fork.SEGMENT_INSTRUCTION,
        "librarian": librarian.BRIEF_EVIDENCE,
        "judge": kinship.BRIEF,
        "actor guide": lp.GUIDE_LIBRARIAN,
    }
    found = {name: _hits(text) for name, text in texts.items() if _hits(text)}
    assert not found, found


def test_the_lists_catch_what_they_are_for():
    assert _hits("Generalise it: values that vary become parameters") and _hits("for future instances of this task")
    assert not _hits("Read the request's text in code.")
