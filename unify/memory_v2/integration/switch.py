"""Value parsing for the UNIFY_MEMORY_V2 settings (unify/settings.py calls these validators)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

#: field -> (accepted values after lower-casing; the value empty and ``off`` map to)
_CHOICES = {
    "UNIFY_MEMORY_V2": (("on",), ""),
    "UNIFY_MEMORY_V2_TRIGGER": (("d6", "batched"), "d6"),
}
_SOL_MODEL_DEFAULT = "openai/gpt-6-sol"
_SOL_BUDGET_DEFAULT = "2.50"


def parse_choice(name: str, v: Any) -> str:
    accepted, empty = _CHOICES[name]
    value = str(v or "").strip().lower()
    if value == "" or (value == "off" and name == "UNIFY_MEMORY_V2"):
        return empty
    if value not in accepted:
        allowed = ", ".join(repr(a) for a in accepted)
        raise ValueError(f"{name} must be empty or one of {allowed}, not {v!r}")
    return value


def parse_sol(name: str, v: Any) -> str:
    value = str(v if v is not None else "").strip()
    if name == "UNIFY_MEMORY_V2_SOL_MODEL":
        value = value or _SOL_MODEL_DEFAULT
        if any(c.isspace() for c in value):
            raise ValueError(f"{name} must be a model id without spaces, not {v!r}")
        return value
    value = value or _SOL_BUDGET_DEFAULT
    try:
        amount = Decimal(value)
    except InvalidOperation:
        amount = None
    if amount is None or not amount.is_finite() or amount < 0:
        raise ValueError(f"{name} must be a non-negative decimal USD amount, not {v!r}")
    return value
