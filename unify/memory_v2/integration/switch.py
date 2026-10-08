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

The v2.1 surfacing switches (lane S1). Each default restores the behaviour of the v2 screen build
(``9deefbfd1``) exactly, so a paired v2 vs v2.1 comparison runs on one build:

- ``UNIFY_MEMORY_V2_SURFACING``: ``index`` (default; empty means it): the system prompt ends with the v2
  per-function index and the export line, the export holds only the commit's files, no input shapes are
  recorded or frozen, and Sol's first message carries the index. ``catalogue``: a constant guide paragraph
  in the prompt (the same bytes for the whole run, from the first request with a non-empty library; the
  channels, counts and suspect flags are what ``memory.catalog()`` prints in the cell); the generated
  ``README.md``, ``.memory/catalog.json``, ``.memory/shapes.py`` and ``memory.py`` in every export; input
  shapes recorded at each merge and frozen per commit; Sol's first message carries the README and its brief
  says never to write those files.
- ``UNIFY_MEMORY_V2_DOCSTRINGS``: ``off`` (default; empty means it) or ``on``: the gate's lean docstring
  standard (G1) and examples run (G3), with the standard and the check list in Sol's brief. The online
  driver and the offline replay (``memory_v2_offline/cadence_replay.py``) both build the gate from this
  switch (:func:`surfacing_options`), so a replay enforces exactly what Sol is told.
- ``UNIFY_MEMORY_V2_SOFT_BUDGET``: ``off`` (default; empty means it): G4 refuses an index over 4,000
  estimated tokens, as in v2. ``on``: G4 only notes that hygiene is due past the budget, measured on the
  library's surface (the README and channel lines under ``catalogue``, the index under ``index``).

Not switched (a declared safety fix in every mode): the gate refuses bytecode, native code, start-up hooks
and root entries other than ``env/``, ``workflows/`` and the test kit before extraction
(:func:`unify.memory_v2.manifest.unsafe_path`), and changes to the paths the harness reserves for its
generated catalogue (:func:`unify.memory_v2.catalogue.reserved`).

Money stays a decimal string as written (never a float), and exponent forms are refused, so a value is
read the same way by every consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

SWITCH = "UNIFY_MEMORY_V2"
EXPERIENCE_BUDGET = "UNIFY_MEMORY_V2_E"
SOL_MODEL = "UNIFY_MEMORY_V2_SOL_MODEL"
SOL_ALLOWANCE = "UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS"
SOL_RUN_GUARD = "UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD"
SURFACING = "UNIFY_MEMORY_V2_SURFACING"
DOCSTRINGS = "UNIFY_MEMORY_V2_DOCSTRINGS"
SOFT_BUDGET = "UNIFY_MEMORY_V2_SOFT_BUDGET"

EXPERIENCE_BUDGET_DEFAULT = 150000
SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
SOL_ALLOWANCE_DEFAULT = "0.00000073"
SURFACING_VALUES = ("index", "catalogue")
SURFACING_DEFAULT = "index"
ON_OFF = ("off", "on")

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


def parse_surfacing(v: Any) -> str:
    """``index`` (also for empty) or ``catalogue``."""
    value = _stripped(v).lower() or SURFACING_DEFAULT
    if value not in SURFACING_VALUES:
        raise ValueError(
            f"{SURFACING} must be empty, 'index' or 'catalogue', not {v!r}"[:200],
        )
    return value


def _on_off(name: str, v: Any) -> str:
    value = _stripped(v).lower() or "off"
    if value not in ON_OFF:
        raise ValueError(f"{name} must be empty, 'off' or 'on', not {v!r}"[:200])
    return value


def parse_docstrings(v: Any) -> str:
    """``off`` (also for empty) or ``on``."""
    return _on_off(DOCSTRINGS, v)


def parse_soft_budget(v: Any) -> str:
    """``off`` (also for empty) or ``on``."""
    return _on_off(SOFT_BUDGET, v)


@dataclass(frozen=True)
class SurfacingOptions:
    """The three v2.1 surfacing switches, parsed; the defaults are the v2 screen build's behaviour."""

    surfacing: str = SURFACING_DEFAULT
    docstrings: bool = False
    soft_budget: bool = False

    @property
    def catalogue(self) -> bool:
        return self.surfacing == "catalogue"

    def gate_kwargs(self) -> dict[str, Any]:
        """The :class:`unify.memory_v2.gate.Gate` keyword arguments these switches set (Sol's brief follows
        the gate it is given, so this is the one place both are configured)."""
        return {
            "surfacing": self.surfacing,
            "docstring_standard": self.docstrings,
            "soft_budget": self.soft_budget,
        }

    def as_dict(self) -> dict[str, str]:
        """The settings as written (for a run's provenance)."""
        return {
            SURFACING: self.surfacing,
            DOCSTRINGS: "on" if self.docstrings else "off",
            SOFT_BUDGET: "on" if self.soft_budget else "off",
        }


def surfacing_options(settings: Any) -> SurfacingOptions:
    """The switches from *settings* (``unify.settings.SETTINGS``, or any object; a missing one is its default),
    each through its validator, so an invalid value is refused here as at settings load.
    """
    return SurfacingOptions(
        surfacing=parse_surfacing(getattr(settings, SURFACING, "")),
        docstrings=parse_docstrings(getattr(settings, DOCSTRINGS, "")) == "on",
        soft_budget=parse_soft_budget(getattr(settings, SOFT_BUDGET, "")) == "on",
    )


#: The validator for each setting (unify/settings.py ``parse_memory_v2``).
PARSERS = {
    SWITCH: parse_switch,
    EXPERIENCE_BUDGET: parse_experience_budget,
    SOL_MODEL: parse_sol_model,
    SOL_ALLOWANCE: parse_sol_allowance,
    SOL_RUN_GUARD: parse_sol_run_guard,
    SURFACING: parse_surfacing,
    DOCSTRINGS: parse_docstrings,
    SOFT_BUDGET: parse_soft_budget,
}
