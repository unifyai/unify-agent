"""Value parsing for the UNIFY_MEMORY_V2 settings (unify/settings.py calls these validators).

The contract (online build, spec §F1 and D23):

- ``UNIFY_MEMORY_V2``: ``on``, ``off`` or empty (empty and ``off`` mean off).
- ``UNIFY_MEMORY_V2_E``: the experience budget E in tokens, a positive int; empty means 150000. A pass
  becomes due once the evidence since the last pass reaches E, and its USD cap is E times the allowance.
- ``UNIFY_MEMORY_V2_SOL_MODEL``: the model that runs consolidation passes; empty means
  ``openai/gpt-6-sol``. Sol's reasoning effort has no switch: it is the actor's effort for the run.
- ``UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS``: USD Sol may spend per token of experience, a positive
  plain decimal string; empty means ``0.00000073``.
- ``UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD``: a non-negative plain decimal string, or empty for no guard. When
  set, no further pass starts once the run's committed Sol USD plus the next pass's cap would exceed it.

Stage-5 test checks in the consolidation gate (memory v2.1; :mod:`unify.memory_v2.qa`), each off by default
and read only while ``UNIFY_MEMORY_V2`` is on:

- ``UNIFY_MEMORY_V2_QA_FIXTURES``: ``on``, ``strict``, ``off`` or empty. Seeded random draws of recorded inputs
  of each new or changed function's family; ``strict`` also refuses a drawn input its own tests never read.
- ``UNIFY_MEMORY_V2_QA_MUTATION``: ``on``, ``off`` or empty. Mutation testing of each new or changed function.
- ``UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL``: the minimum share of output-changing mutants the tests must kill, a
  plain decimal string from 0 to 1; empty means ``0.5``.
- ``UNIFY_MEMORY_V2_QA_DETERMINISM``: ``on``, ``off`` or empty. Pinned clock and randomness; each new or changed
  test file runs twice.
- ``UNIFY_MEMORY_V2_QA_REPLAY``: ``on``, ``off`` or empty. Environment functions are tested through the
  recorded replay, never a stand-in.
- ``UNIFY_MEMORY_V2_QA_FIXTURE_SIZE``: ``on``, ``off`` or empty. A bound on test files, blob references and
  truncation markers.

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any

SWITCH = "UNIFY_MEMORY_V2"
EXPERIENCE_BUDGET = "UNIFY_MEMORY_V2_E"
SOL_MODEL = "UNIFY_MEMORY_V2_SOL_MODEL"
SOL_ALLOWANCE = "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS"
SOL_RUN_GUARD = "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD"

QA_FIXTURES = "UNIFY_MEMORY_V2_QA_FIXTURES"
QA_MUTATION = "UNIFY_MEMORY_V2_QA_MUTATION"
QA_MIN_KILL = "UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL"
QA_DETERMINISM = "UNIFY_MEMORY_V2_QA_DETERMINISM"
QA_REPLAY = "UNIFY_MEMORY_V2_QA_REPLAY"
QA_FIXTURE_SIZE = "UNIFY_MEMORY_V2_QA_FIXTURE_SIZE"

EXPERIENCE_BUDGET_DEFAULT = 150000
SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
SOL_ALLOWANCE_DEFAULT = "0.00000073"
QA_MIN_KILL_DEFAULT = "0.5"

#: A plain decimal: digits, optionally a point and more digits. No sign, exponent, separator or name.
_PLAIN_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_DIGITS = re.compile(r"[0-9]+")


def _stripped(v: Any) -> str:
    return str(v if v is not None else "").strip()


def parse_switch(v: Any) -> str:
    """``on``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value != "on":
        raise ValueError(f"{SWITCH} must be empty, 'off' or 'on', not {v!r}")
    return value


def parse_experience_budget(v: Any) -> int:
    """A positive token count, written as digits only."""
    refusal = (
        f"{EXPERIENCE_BUDGET} must be a positive whole number of tokens, not {v!r}"
    )
    if isinstance(v, bool):
        raise ValueError(refusal)
    if isinstance(v, int):
        value = v
    else:
        text = _stripped(v)
        if text == "":
            return EXPERIENCE_BUDGET_DEFAULT
        if not _DIGITS.fullmatch(text):
            raise ValueError(refusal)
        value = int(text)
    if value <= 0:
        raise ValueError(refusal)
    return value


def parse_sol_model(v: Any) -> str:
    value = _stripped(v) or SOL_MODEL_DEFAULT
    if any(c.isspace() for c in value):
        raise ValueError(f"{SOL_MODEL} must be a model id without spaces, not {v!r}")
    return value


def _plain_decimal(name: str, text: str, v: Any, *, positive: bool) -> str:
    if not _PLAIN_DECIMAL.fullmatch(text):
        raise ValueError(
            f"{name} must be a plain decimal USD amount (digits and at most one point, "
            f"no sign or exponent), not {v!r}",
        )
    if positive and Decimal(text) <= 0:
        raise ValueError(f"{name} must be above zero, not {v!r}")
    return text


def parse_sol_allowance(v: Any) -> str:
    """USD per token of experience: a positive plain decimal string."""
    text = _stripped(v) or SOL_ALLOWANCE_DEFAULT
    return _plain_decimal(SOL_ALLOWANCE, text, v, positive=True)


def parse_sol_run_guard(v: Any) -> str:
    """The run's Sol USD guard: a non-negative plain decimal string, or ``""`` for none."""
    text = _stripped(v)
    if text == "":
        return ""
    return _plain_decimal(SOL_RUN_GUARD, text, v, positive=False)


def _on_off(name: str):
    def parse(v: Any) -> str:
        """``on``, or ``""`` for off (empty or ``off``)."""
        value = _stripped(v).lower()
        if value in ("", "off"):
            return ""
        if value != "on":
            raise ValueError(f"{name} must be empty, 'off' or 'on', not {v!r}")
        return value

    return parse


parse_qa_mutation = _on_off(QA_MUTATION)
parse_qa_determinism = _on_off(QA_DETERMINISM)
parse_qa_replay = _on_off(QA_REPLAY)
parse_qa_fixture_size = _on_off(QA_FIXTURE_SIZE)


def parse_qa_fixtures(v: Any) -> str:
    """``on``, ``strict``, or ``""`` for off (empty or ``off``)."""
    value = _stripped(v).lower()
    if value in ("", "off"):
        return ""
    if value not in ("on", "strict"):
        raise ValueError(
            f"{QA_FIXTURES} must be empty, 'off', 'on' or 'strict', not {v!r}",
        )
    return value


def parse_qa_min_kill(v: Any) -> str:
    """A share from 0 to 1: a plain decimal string."""
    text = _stripped(v) or QA_MIN_KILL_DEFAULT
    if not _PLAIN_DECIMAL.fullmatch(text) or Decimal(text) > 1:
        raise ValueError(
            f"{QA_MIN_KILL} must be a plain decimal from 0 to 1 (digits and at most one point), not {v!r}",
        )
    return text


#: The validator for each setting (unify/settings.py ``parse_memory_v2``).
PARSERS = {
    SWITCH: parse_switch,
    EXPERIENCE_BUDGET: parse_experience_budget,
    SOL_MODEL: parse_sol_model,
    SOL_ALLOWANCE: parse_sol_allowance,
    SOL_RUN_GUARD: parse_sol_run_guard,
    QA_FIXTURES: parse_qa_fixtures,
    QA_MUTATION: parse_qa_mutation,
    QA_MIN_KILL: parse_qa_min_kill,
    QA_DETERMINISM: parse_qa_determinism,
    QA_REPLAY: parse_qa_replay,
    QA_FIXTURE_SIZE: parse_qa_fixture_size,
}
