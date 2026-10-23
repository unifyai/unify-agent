"""No text an agent reads steers what is stored, in either direction.

Agents decide what to store; the harness states what each tool does. No prompt, brief, card, tool text or help topic
discourages, bars or blocks keeping work-, request- or instance-specific artifacts (for reuse, memory or transfer),
and none urges keeping them either. The scan covers every string the memory agents' briefs and the actor's memory
texts are built from (module constants, docstrings the package's ``help`` shows, card templates, tool texts) and every
text as rendered (:mod:`tests.memory.agent_texts`: every role's brief, opening and first message, ``tools.list()`` and
every ``tools.help`` per role, every refusal, the land_capture reply, the judge's card and the gate's lines as real
merges write them). Besides the fixed phrases, a list of paraphrases (:data:`PARAPHRASES`) is matched as words; a hit
passes only where a reviewed phrase of :data:`ALLOWED` covers it, saying what the tool or record mechanically does.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Modules whose strings an agent reads: briefs, guides, cards, catalogue texts, serving and trust labels, gate
#: messages, tool texts and help topics.
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
    "unify/memory/library_export.py",
    "unify/memory/serving.py",
    "unify/memory/trust.py",
    "unify/memory/library_checks.py",
    "unify/memory/writer_pass.py",
    "unify/common/tool_registry.py",
    "unify/agents/run_record.py",
    "unify/agents/run_record_tools.py",
    "unify/memory/gate.py",
    "unify/memory/gate_policy.py",
    "unify/memory/captures.py",
    "unify/memory/library_index.py",
    "unify/memory/capture_groups.py",
    "unify/memory/agent_profiles.py",
    "unify/memory/messaging.py",
    "unify/memory/attention.py",
    "unify/memory/writer_tests.py",
    "unify/memory/item_label.py",
    "unify/memory/qa.py",
    "unify/memory/integration/prompt.py",
    "unify/memory/integration/hooks.py",
    "unify/memory/integration/in_episode.py",
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


#: Paraphrases of the steering above, matched as words: wording that urges keeping work for reuse or generalising it,
#: deriving every value, or keeping failures as knowledge. "reused", "derived" and "re-derives" describe what a tool or
#: the gate mechanically does and are not matched.
PARAPHRASES = re.compile(
    r"\b(reusable|reuse|(?:could|can|may) be reused|generali[sz](?:e|es|ing)|derive|every value|every parameter|"
    r"is knowledge|knowledge too)\b",
    re.IGNORECASE,
)
#: Reviewed phrases holding a paraphrase that say what a tool or a record does, never what to keep: (phrase, why).
ALLOWED = (
    ("every value whole", "check_examples shows each example's values whole: what the tool prints"),
    ("Generalises: ", "a card's link line naming the items an item generalises: a relation the writer stated"),
)


def _paraphrases(text: str) -> list[str]:
    """The paraphrase hits of *text* that no reviewed phrase covers."""
    out = []
    for m in PARAPHRASES.finditer(text):
        lo = max(0, m.start() - 80)
        window = text[lo : m.end() + 80]
        if not any(phrase in window and _covers(text, phrase, m) for phrase, _ in ALLOWED):
            out.append(" ".join(text[max(0, m.start() - 40) : m.end() + 40].split()))
    return out


def _covers(text: str, phrase: str, m: re.Match) -> bool:
    start = text.find(phrase, max(0, m.start() - len(phrase)))
    while start != -1 and start <= m.start():
        if start + len(phrase) >= m.end():
            return True
        start = text.find(phrase, start + 1)
    return False


def _strings(path: str) -> list[tuple[int, str]]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return [(n.lineno, n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _hits(text: str) -> list[str]:
    low = " ".join(text.lower().split())
    return [p for p in (*AGAINST, *TOWARDS) if p in low]


@pytest.mark.parametrize("path", MODULES)
def test_no_string_an_agent_reads_steers_storage(path):
    found = [
        f"{path}:{line}: {_hits(text) + _paraphrases(text)}"
        for line, text in _strings(path)
        if _hits(text) or _paraphrases(text)
    ]
    assert not found, found


def test_no_rendered_text_steers_storage(tmp_path):
    from tests.memory.agent_texts import rendered

    texts = rendered(tmp_path)
    assert any(k.startswith("rendered:write:") for k in texts) and "rendered:gate" in texts
    found = {name: _hits(text) + _paraphrases(text) for name, text in texts.items() if _hits(text) or _paraphrases(text)}
    assert not found, found


def test_every_reviewed_phrase_still_covers_a_hit(tmp_path):
    from tests.memory.agent_texts import rendered

    every = list(rendered(tmp_path).values()) + [t for path in MODULES for _, t in _strings(path)]
    for phrase, _ in ALLOWED:
        assert any(phrase in t and PARAPHRASES.search(phrase) for t in every), phrase


def test_the_lists_catch_what_they_are_for():
    assert _hits("Generalise it: values that vary become parameters") and _hits("for future instances of this task")
    assert not _hits("Read the request's text in code.")
    for steering in (
        "for code that ran and worked and could be reused",
        "Where code looks reusable, look for it",
        "whether code you are about to generalise occurs",
        "reads from it every value the job needs",
        "the whole job with every parameter read from it",
        "derive every value the job needs from the request",
        "A failed capture is knowledge too",
    ):
        assert _paraphrases(steering), steering
    for mechanics in ("the gate re-derives every fixture", "a reader with the same code is reused", "its derived fields"):
        assert not _paraphrases(mechanics), mechanics
