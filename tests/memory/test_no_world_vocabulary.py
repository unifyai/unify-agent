"""No world's vocabulary in what any agent reads: the guard grep over every text the harness gives a model.

The general harness names no world (no benchmark, no environment, no kind of task it was tuned on). This test
runs one case-insensitive whole-word grep (:data:`GUARD`) over every text an agent can read:

* the exported ``memory`` package, file by file, whole (the actor reads its help and can read its source):
  ``memory/__init__.py`` (with ``memory.ask``), its typed stub, its contract runtime, the shape matcher, and the
  v2 helper;
* every string of the modules that hold the tool specs and docstrings (``tool_registry``) and every role's
  prompt (:data:`GUARD_MODULES`): docstrings, prompt texts, tool descriptions, refusal texts;
* the prompts, tool texts, refusals, replies and gate lines as rendered for each role and switch combination
  (:func:`_rendered`, :mod:`tests.memory.agent_texts`), so text composed at run time is read too.

A hit passes only where a line of the reviewed allow-list (:data:`ALLOW_FILE`) covers it: a phrase holding the
word, for a scope of sources, with the reason it is generic English. Every line must still cover a hit, so the
list cannot grow stale or wide. A module that gains model-facing text joins :data:`GUARD_MODULES`.
"""

from __future__ import annotations

import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
#: The guard grep, case-insensitive, whole words.
GUARD = re.compile(
    r"\b(grid|grids|puzzle|demo|demos|task|tasks|airline|booking|crafter|arc|poker|hand|game|colleague|office|"
    r"stencil|colour|color|payment|payments|currency|currencies|money|tax|taxes|invoice|invoices|demonstration|"
    r"demonstrations|submit|submits|submitted|submission|submissions)\b",
    re.IGNORECASE,
)
#: The reviewed allow-list: ``scope<TAB>phrase<TAB>reason`` per line (``#`` starts a comment).
ALLOW_FILE = Path(__file__).with_name("no_world_vocabulary_allowlist.tsv")
#: The modules whose every string is model-facing text (tool specs, docstrings, role prompts and briefs).
GUARD_MODULES = (
    "unify/common/tool_registry.py",
    "unify/memory/library_prompts.py",
    "unify/memory/writer_pass.py",
    "unify/memory/qa.py",
    "unify/memory/helpers.py",
    "unify/memory/librarian.py",
    "unify/memory/library_ask.py",
    "unify/memory/agent_profiles.py",
    "unify/memory/messaging.py",
    "unify/memory/integration/fork.py",
    "unify/memory/integration/prompt.py",
    "unify/memory/integration/hooks.py",
    # the standing-note digest the actor's first message carries (rev 2.4)
    "unify/memory/standing_notes.py",
    "unify/actor/prompt_builders.py",
    # the cards, catalogue and search rows an agent reads about each item, and the status texts they carry
    "unify/memory/library_export.py",
    "unify/memory/library_search.py",
    "unify/memory/serving.py",
    "unify/memory/trust.py",
    # the redesign's agent-facing texts (REDESIGN-BUILD-PLAN-20261020): the run record, its tools and the fork's relay
    # (the run record), the writer service's notices, metering's labels, the capture trace, the attention ledger's
    # refusals and tools, the writer's stated tests and item labels
    "unify/agents/run_record.py",
    "unify/agents/run_record_tools.py",
    "unify/agents/fork_relay.py",
    "unify/memory/integration/fork_worker.py",
    "unify/memory/writer_service.py",
    "unify/memory/metering.py",
    "unify/memory/capture_trace.py",
    "unify/memory/attention.py",
    "unify/memory/attention_tools.py",
    "unify/memory/writer_tests.py",
    "unify/memory/item_label.py",
    # the cards and help(memory) of the library helper, the land_capture texts and replies, the manifest's texts, the
    # gate's label lines, the captures' and the capture groups' texts, the index header, the judge's brief, the ask
    # call's texts and the when-to-use shapes
    "unify/memory/library_helper.py",
    "unify/memory/land_capture.py",
    "unify/memory/manifest.py",
    "unify/memory/library_checks.py",
    "unify/memory/gate.py",
    "unify/memory/gate_policy.py",
    "unify/memory/captures.py",
    "unify/memory/library_index.py",
    "unify/memory/capture_groups.py",
    "unify/memory/kinship.py",
    "unify/memory/when_to_use.py",
)


@dataclass(frozen=True)
class Hit:
    source: str
    word: str
    start: int
    end: int
    context: str


@dataclass(frozen=True)
class Allowed:
    scope: str
    phrase: str
    reason: str

    def covers(self, source: str, text: str, start: int, end: int) -> bool:
        if self.scope != "*" and not source.startswith(self.scope):
            return False
        lo = max(0, start - len(self.phrase))
        for m in re.finditer(re.escape(self.phrase), text[lo : end + len(self.phrase)]):
            if lo + m.start() <= start and end <= lo + m.end():
                return True
        return False


def allow_list() -> list[Allowed]:
    out = []
    for n, line in enumerate(ALLOW_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        assert len(parts) == 3 and all(
            p.strip() for p in parts
        ), f"{ALLOW_FILE.name}:{n}: scope, phrase, reason"
        scope, phrase, reason = parts
        assert GUARD.search(
            phrase
        ), f"{ALLOW_FILE.name}:{n}: the phrase holds no guarded word"
        out.append(Allowed(scope, phrase, reason))
    return out


def _strings(source: str) -> str:
    """Every string literal of a module (f-string parts included), one per line, as written."""
    middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    return "\n".join(
        tok.string
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.STRING or (middle is not None and tok.type == middle)
    )


def _package() -> dict[str, str]:
    from unify.memory import library_export

    return {
        "package:memory/__init__.py": library_export.package_source(
            librarian=True
        ).decode("utf-8"),
        "package:memory/__init__.py (no librarian)": library_export.package_source().decode(
            "utf-8"
        ),
        "package:memory/__init__.pyi": library_export.stub_source(
            librarian=True
        ).decode("utf-8"),
        "package:memory/_contract.py": library_export.CONTRACT_SOURCE.read_text(
            encoding="utf-8"
        ),
        "package:matcher": library_export.matcher_source().decode("utf-8"),
        "package:memory_helper.py": (ROOT / "unify/memory/memory_helper.py").read_text(
            encoding="utf-8"
        ),
    }


def _modules() -> dict[str, str]:
    return {
        f"module:{m}": _strings((ROOT / m).read_text(encoding="utf-8"))
        for m in GUARD_MODULES
    }


def _rendered(monkeypatch, tmp: Path) -> dict[str, str]:
    """Every text a memory agent is shown, as composed at run time (:func:`tests.memory.agent_texts.rendered`)."""
    from tests.memory.agent_texts import rendered

    return rendered(tmp)


def texts(tmp: Path, monkeypatch) -> dict[str, str]:
    return {**_package(), **_modules(), **_rendered(monkeypatch, tmp)}


def hits(texts_by_source: dict[str, str]) -> list[Hit]:
    out = []
    for source, text in texts_by_source.items():
        for m in GUARD.finditer(text):
            lo, hi = max(0, m.start() - 40), min(len(text), m.end() + 40)
            out.append(Hit(source, m.group(0), m.start(), m.end(), text[lo:hi]))
    return out


def test_no_text_an_agent_reads_names_a_world(tmp_path, monkeypatch):
    allowed = allow_list()
    by_source = texts(tmp_path, monkeypatch)
    # every source above was read: each guarded module, each agent's tools, opening and help for each tool it has
    from unify.common import tool_registry as R

    every = frozenset({"pool", "similar", "helper"})
    assert all(by_source.get(f"module:{m}") for m in GUARD_MODULES)
    for agent in R.AGENTS:
        assert by_source.get(f"rendered:tools:{agent}") and by_source.get(f"rendered:preamble:{agent}"), agent
        assert all(by_source.get(f"rendered:help:{agent}:{n}") for n in R.allow_list(agent, every)), agent
    used: set[Allowed] = set()
    leaks = []
    for h in hits(by_source):
        cover = {
            a
            for a in allowed
            if a.covers(h.source, by_source[h.source], h.start, h.end)
        }
        if not cover:
            leaks.append(f"{h.source}: {h.word!r} in {h.context!r}")
        used |= cover
    assert not leaks, "world vocabulary outside the allow-list:\n" + "\n".join(
        leaks[:200]
    )
    stale = [f"{a.scope}\t{a.phrase}" for a in allowed if a not in used]
    assert (
        not stale
    ), "allow-list lines that cover nothing (remove them):\n" + "\n".join(stale)


def test_the_allow_list_is_well_formed():
    allowed = allow_list()
    assert len(allowed) == len({(a.scope, a.phrase) for a in allowed})
    for a in allowed:
        assert a.scope == "*" or a.scope.split(":", 1)[0] in (
            "package",
            "module",
            "rendered",
        ), a


@pytest.mark.parametrize(
    "text, words",
    [
        ("Read notes/grid/stencil.md first.", ["grid", "stencil"]),
        ("the ARC demo pairs", ["ARC", "demo"]),
        ("a task id", ["task"]),
        ("arcs, tasked, handled, colours and griddle", []),
    ],
)
def test_the_guard_matches_whole_words_in_any_case(text, words):
    assert [m.group(0) for m in GUARD.finditer(text)] == words


# On a keyless host: print every hit and every unused allow-list line, to review the allow-list.
if __name__ == "__main__":
    import sys
    import tempfile

    class _Patch:
        def setattr(self, obj, name, value):
            setattr(obj, name, value)

    with tempfile.TemporaryDirectory() as d:
        found = texts(Path(d), _Patch())
        allowed = allow_list() if ALLOW_FILE.exists() else []
        seen: set[Allowed] = set()
        for h in hits(found):
            cover = {
                a
                for a in allowed
                if a.covers(h.source, found[h.source], h.start, h.end)
            }
            seen |= cover
            state = "ok" if cover else "LEAK"
            sys.stdout.write(f"{state}\t{h.source}\t{h.word}\t{h.context!r}\n")
        for a in allowed:
            if a not in seen:
                sys.stdout.write(f"STALE\t{a.scope}\t{a.phrase}\n")
        sys.stdout.write(f"SOURCES\t{len(found)}\n")
