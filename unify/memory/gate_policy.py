"""What the gate does with each finding, stated once (:data:`GATE`): which findings refuse, which only label, and
that the gate runs nothing.

The rules: no capture or function is executed or re-run to classify, dedupe, rank or test it (that fails for impure
items and for worlds that cannot be replayed, and a statically pure one is no exception); a write-time quality check
labels, never rejects; confinement, security, credential and record-keeping checks stay hard. So:

* **Refusals** (:data:`GATE`'s ``refusals``, :func:`hard`): safety (G6), the pass's declared change set and manifest
  (G1 findings no item owns), the provenance of recorded evidence (fixtures re-derived from the recording's bytes,
  source episodes that exist), a new or changed note without its ``## Rule`` section, a pass with fewer items than its
  parent or a dropped alias (G5: retiring an item is a state change that keeps it on record, so nothing deletes an
  item), and what breaks the library itself (code that does not parse, listed files missing, the gate's own kit in a
  test).
* **Labels**: every other finding about an item is recorded on its label, and a finding about the pass is a note.
* **Execution**: none (``runs`` is empty). An item's tests, its recipe, its recorded calls and its package's suites
  are not run by the gate, and its label says ``not run: <why>`` (:data:`NOT_RUN_GATE`). Running code is the writer's
  own choice during its pass (``tools.test``, or its tests in its cells). Nothing classifies an item as pure or impure,
  acting or not: that cannot be told reliably, and no label, level, status or rank rests on it.

The writer's brief (:mod:`.library_prompts`), the fixture, land_capture and check tool texts, the land_capture reply
and the gate's own label lines are built from :data:`GATE`, so what they say the gate does is what :func:`hard`
decides. Everything here reads syntax only: nothing is imported or run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from unify.common.tool_registry import (
    GATE_CASES_TEXT as CASES_TEXT,
    GATE_CHECK_RETURNS as CHECK_RETURNS,
    GATE_FIXTURE_TEXT as FIXTURE_TEXT,
    GATE_LAND_TEXT as LAND_GATE_TEXT,
)

from .when_to_use import RULE

# CASES_TEXT, CHECK_RETURNS, FIXTURE_TEXT and LAND_GATE_TEXT: what the writer's tool texts say of the gate, kept with
# the registry's tool texts (which import no harness module), so a tool text and :data:`GATE` cannot disagree.

#: Why the gate runs no item, test or recipe: the same words on every label that says ``not run``.
NOT_RUN_GATE = (
    "the gate runs no code; an item's tests run only when you run them (tools.test, or pytest in your cells)"
)


def not_run_reason(source: Any) -> str:
    """Why the gate did not run an item (:data:`NOT_RUN_GATE`): the same words for every item. Its *source* is not
    read: whether code is pure cannot be told reliably, so nothing classifies it."""
    return NOT_RUN_GATE


@dataclass(frozen=True)
class Refusal:
    """One kind of finding that refuses: how the writer's brief names it (*words*), the check it belongs to (None:
    any check), and how the gate's own reason text marks it: one of *marks* in the reason (empty: every finding of the
    check), or, with *unowned*, a finding no manifest item owns."""

    words: str
    check: str | None = None
    marks: tuple[str, ...] = ()
    unowned: bool = False

    def covers(self, check: str, reason: str, owners: list[str]) -> bool:
        if self.check is not None and check != self.check:
            return False
        if self.unowned:
            return not owners
        return not self.marks or any(m in reason for m in self.marks)


#: The mark of a note's missing or empty ``## Rule`` section, as :func:`.when_to_use.rule_problems` words it.
RULE_MARK = f"`## {RULE}` section"
#: The marks of the other refusals, as the gate's checks word their reasons.
FIXTURE_MARK = "fixture provenance"
NO_SOURCE_MARK = "names no source episode"
UNKNOWN_EPISODE_MARK = "cites unknown episode"
NO_PARSE_MARK = "does not parse"
MISSING_MARK = "is not in the candidate"
KIT_MARK = "[library_mode:plain]"


@dataclass(frozen=True)
class GatePolicy:
    """The gate's decisions as one description: what refuses (*refusals*, in the order the brief names them), what
    each check only labels (*labels*, by check), and what the gate runs (*runs*: nothing)."""

    refusals: tuple[Refusal, ...]
    labels: tuple[tuple[str, str], ...]
    runs: tuple[str, ...] = ()

    def hard(self, check: str, reason: str, owners: Iterable[str]) -> bool:
        """Whether a finding of *check* with *reason* about *owners* refuses (else it labels its items, or notes the
        pass)."""
        found = list(owners)
        return any(r.covers(check, str(reason), found) for r in self.refusals)

    def refusal_words(self) -> list[str]:
        return [r.words for r in self.refusals]


GATE = GatePolicy(
    refusals=(
        Refusal("safety (G6)", "G6"),
        Refusal(
            "a finding about the pass as a whole (a manifest that does not parse, a change it leaves undeclared, a "
            "change outside `memory/` and `notes/` or to the generated files)",
            "G1",
            unowned=True,
        ),
        Refusal("a fixture that did not come through `tools.fixture` unchanged", "G1", (FIXTURE_MARK,)),
        Refusal(
            "an item that names no source episode, or cites one that is not recorded",
            "G1",
            (NO_SOURCE_MARK, UNKNOWN_EPISODE_MARK),
        ),
        Refusal(f"a new or changed note without a `## {RULE}` section with text", "G1", (RULE_MARK,)),
        Refusal("a pass with fewer items than the library it started from, or a dropped alias (G5)", "G5"),
        Refusal("code that does not parse", None, (NO_PARSE_MARK,)),
        Refusal("a listed item, test or case file missing from the candidate", None, (MISSING_MARK,)),
        Refusal("a test that imports the gate's own test kit", None, (KIT_MARK,)),
    ),
    labels=(
        (
            "G1",
            "layout and provenance (the links, each new or changed function and note's `When to use` section and each "
            "changed folder's README.md)",
        ),
        (
            "G2",
            "evidence (each function's listed tests name it, a procedure's recipe and cases, and the source episodes "
            "of a procedure; an item's support is the episodes it comes from and the uses whose outcomes confirmed "
            "it, never a run of the gate's)",
        ),
        (
            "G3",
            "tests (plain pytest, an exact assertion against a trusted value, and no test lost or weakened without a "
            "reason, fewer tests included; the gate runs none of them: they state what the item claims)",
        ),
        ("G4", "the index (a note when the index has grown past its size limit)"),
    ),
)


def hard(check: str, reason: str, owners: Iterable[str]) -> bool:
    """Whether a finding refuses (:data:`GATE`)."""
    return GATE.hard(check, reason, owners)


#: The gate's G2 label for a procedure from an episode that failed, as the gate writes it (it lands with the label).
FAILED_SOURCE_LABEL = (
    "{item}: its source episode {episodes} failed; the procedure lands with this label"
)
#: The gate's G2 label for a procedure that names no recipe.
NO_RECIPE_LABEL = (
    "{item} is a procedure without a recipe: a recipe is a function of its module, named in the entry's recipe, "
    "that is handed the request as the harness hands it and calls the procedure"
)
