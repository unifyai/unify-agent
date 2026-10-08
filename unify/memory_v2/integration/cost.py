"""Per-call model cost by purpose (integration Task 22): decimal strings, ``unknown`` when unpriced.

Two sources, one row shape (:class:`..episodes.CostRow`):

* :class:`CostListener`, registered once per process with ``unillm.add_llm_event_listener``
  (:func:`install`), records the request's own calls while it is active (the CLI runs one request per
  process). The purpose comes from the call's origin tag: Sol's calls (origin ``memory_v2.sol...``) are
  skipped, because the pass records its own through :func:`recording_turn`; an origin containing
  ``embed`` is ``embedding``; every other call is ``actor``.
* :func:`recording_turn` wraps Sol's model turn and appends one ``sol`` row per turn.

Money is a plain decimal string (never an exponent). A cost that is missing, not a finite non-negative
number, or in any other form (``"x+unknown"``) is ``"unknown"``; an unknown cost is never zero. Token
counts the provider did not report stay ``None``.
"""

from __future__ import annotations

import threading
from decimal import Decimal, InvalidOperation
from typing import Any

from unify.common.llm_meter import decimal_string, to_decimal_usd

from ..episodes import CostRow
from ..sol_pass import ModelTurn

UNKNOWN = "unknown"
SOL_ORIGIN = "memory_v2.sol"

__all__ = [
    "SOL_ORIGIN",
    "UNKNOWN",
    "CostListener",
    "install",
    "money",
    "recording_turn",
]


def money(value: Any) -> str:
    """*value* as a plain non-negative decimal string, or ``"unknown"``.

    Numbers go through :func:`unify.common.llm_meter.to_decimal_usd` (a float via its shortest repr); a
    string must parse as a finite decimal. Exponent forms are rewritten without the exponent.
    """
    if isinstance(value, str):
        try:
            amount: Decimal | None = Decimal(value.strip())
        except InvalidOperation:
            return UNKNOWN
    else:
        amount = to_decimal_usd(value)
    if amount is None or not amount.is_finite() or amount < 0:
        return UNKNOWN
    return decimal_string(amount) or UNKNOWN


def _tokens(response: Any) -> tuple[int | None, int | None]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None, None

    def count(key: str) -> int | None:
        v = usage.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None

    return count("prompt_tokens"), count("completion_tokens")


def _purpose(origin: str) -> str | None:
    if origin.startswith(SOL_ORIGIN):
        return None
    return "embedding" if "embed" in origin else "actor"


class CostListener:
    """Records one :class:`CostRow` per model call while active; never raises into the model call."""

    def __init__(self) -> None:
        self.rows: list[CostRow] = []
        self._active = False
        self._lock = threading.Lock()
        self.failed = 0  # events that could not be read (counted, never raised)

    def activate(self) -> None:
        with self._lock:
            self.rows = []
            self._active = True

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def __call__(self, event: Any) -> None:
        if not self._active:
            return
        try:
            origin = getattr(event, "origin", None)
            purpose = _purpose(origin if isinstance(origin, str) else "")
            if purpose is None:
                return
            request = getattr(event, "request", None)
            model = request.get("model") if isinstance(request, dict) else None
            prompt, completion = _tokens(getattr(event, "response", None))
            row = CostRow(
                purpose,
                model if isinstance(model, str) and model else UNKNOWN,
                prompt,
                completion,
                money(getattr(event, "provider_cost", None)),
            )
        except Exception:  # noqa: BLE001 - metering never changes a call
            with self._lock:
                self.failed += 1
            return
        with self._lock:
            if self._active:
                self.rows.append(row)


_LISTENER: CostListener | None = None
_INSTALL_LOCK = threading.Lock()


def install() -> CostListener:
    """The process's listener, registered with unillm on the first call; idempotent."""
    global _LISTENER
    with _INSTALL_LOCK:
        if _LISTENER is None:
            import unillm

            listener = CostListener()
            unillm.add_llm_event_listener(listener)
            _LISTENER = listener
        return _LISTENER


def recording_turn(turn: ModelTurn, rows: list[CostRow], model: str) -> ModelTurn:
    """*turn*, appending one ``sol`` row per call to *rows*; a raising call appends ``unknown`` and re-raises.

    The turn's own return value is passed through unchanged, so the pass's spend accounting is the same
    with or without the wrapper.
    """

    async def recorded(messages: list[dict], tools: list[dict]) -> tuple[dict, str]:
        try:
            msg, usd = await turn(messages, tools)
        except (
            BaseException
        ):  # cancellation included: the call may still have cost money
            rows.append(CostRow("sol", model, None, None, UNKNOWN))
            raise
        rows.append(CostRow("sol", model, None, None, money(usd)))
        return msg, usd

    return recorded
