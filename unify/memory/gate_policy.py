"""The rev 2.1 gate: every check labels, and the gate runs no code at all.

The rules: no capture or function is executed or re-run to classify, dedupe, rank or test it (that fails for impure
items and for worlds that cannot be replayed, and a statically pure one is no exception); a write-time quality check
labels, never rejects; confinement, security and credential checks stay hard. So:

* **Labels only** (:func:`hard`): a finding about an item is recorded on its label, a finding about the pass is a
  note, and nothing refuses a merge, except the hard checks: G6 (safety: key-shaped strings, unreadable items), the
  pass's declared change set and manifest (G1 findings that no item owns: scope, undeclared or deleted items, the
  manifest itself), the provenance of recorded evidence (fixtures re-derived from the recording, cited episodes),
  and what breaks the library itself (syntax, imports that fail, missing paths, the gate's own kit).
* **Execution**: none. An item's tests, its recipe, its recorded calls and its package's suites are not run by the
  gate, and its label says ``not run: <why>`` (:data:`NOT_RUN_GATE`, :func:`not_run_reason`). Running code is a test
  the writer chooses: it states what it expects and runs it with ``tools.test`` during its pass. Nothing classifies an
  item as pure or impure, acting or not: that cannot be told reliably, and no label, level, status or rank rests on it.

Everything here reads syntax only: nothing is imported or run.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Why the gate runs no item, test or recipe (rev 2.1).
NOT_RUN_GATE = "the gate runs no code (rev 2.1); the writer's tests are the ones it states and runs with tools.test"


def not_run_reason(source: Any) -> str:
    """Why the gate did not run an item (:data:`NOT_RUN_GATE`): the same words for every item. Its *source* is not
    read: whether code is pure cannot be told reliably, so nothing classifies it."""
    return NOT_RUN_GATE


def hard(check: str, reason: str, owners: Iterable[str]) -> bool:
    """Whether a finding stays a refusal under labels only (module docstring)."""
    from .gate import _HARD_MARKS

    text = str(reason)
    if check == "G6" or any(m in text for m in _HARD_MARKS):
        return True
    if check == "G1":
        if not list(owners):
            return True  # the pass's declared change set and its manifest
        return any(m in text for m in ("fixture provenance", "source episode", "cites unknown episode", "file "))
    return False
