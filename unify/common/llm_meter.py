"""Per-run LLM token accounting split by purpose.

Every actor LLM client is created with a ``purpose`` — ``planning`` (the
CodeAct loop and its librarian) — carried in the client's ``origin`` tag. A
single unillm event listener reads that tag back from each ``LLMEvent`` and
adds the call's usage to the :class:`RunMeter` bound to the current context,
so a task run can report how many tokens went where. The listener never
raises and never changes a request.

Money is summed as :class:`~decimal.Decimal` and reported as a decimal
string. A call whose provider cost is not reported (unillm reports none for
cache hits, streaming and errors) is counted as unknown, never as zero, so
a purpose's cost is unknown (``None``) as soon as one of its calls is; the
sum of the reported costs is kept beside it.
"""

from __future__ import annotations

import contextvars
import math
import threading
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import unillm

from unify.common.llm_client import LLMPurpose, purpose_from_origin

PURPOSES: tuple[LLMPurpose, ...] = ("planning",)


def to_decimal_usd(cost: Any) -> Optional[Decimal]:
    """A reported cost as an exact :class:`Decimal`; ``None`` when unknown.

    A float is read through its shortest repr (``0.1`` is ``Decimal("0.1")``,
    not the binary value's 55 digits), so summing reported costs never adds
    float error. ``None``, a bool, and a value that is not a finite number
    are unknown.
    """
    if cost is None or isinstance(cost, bool):
        return None
    if isinstance(cost, Decimal):
        value = cost
    else:
        if isinstance(cost, float) and not math.isfinite(cost):
            return None
        try:
            value = Decimal(str(cost))
        except (InvalidOperation, ValueError, TypeError):
            return None
    return value if value.is_finite() else None


def decimal_string(value: Optional[Decimal]) -> Optional[str]:
    """*value* as a plain decimal string (never an exponent); ``None`` stays ``None``."""
    return None if value is None else format(value, "f")


@dataclass
class RunMeter:
    """Prompt/completion tokens and cost per purpose for one run.

    Read the cost with :meth:`cost_usd` (``None`` when any call's cost was
    unknown), :meth:`known_cost_usd` and :meth:`unknown_cost_calls`;
    :meth:`snapshot` reports it as decimal strings. The float ``cost`` view
    is deprecated.
    """

    tokens: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            purpose: {"prompt": 0, "completion": 0} for purpose in PURPOSES
        },
    )
    calls: Dict[str, int] = field(default_factory=lambda: {p: 0 for p in PURPOSES})
    _known_cost: Dict[str, Decimal] = field(
        default_factory=lambda: {p: Decimal(0) for p in PURPOSES},
        repr=False,
    )
    _unknown_cost_calls: Dict[str, int] = field(
        default_factory=lambda: {p: 0 for p in PURPOSES},
        repr=False,
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(
        self,
        purpose: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost: Any = None,
    ) -> None:
        """Count one call. *cost* is the provider's charge in USD (a number,
        a numeric string or a ``Decimal``); ``None`` means it was not reported."""
        if purpose not in self.tokens:
            purpose = "planning"
        charge = to_decimal_usd(cost)
        with self._lock:
            self.tokens[purpose]["prompt"] += int(prompt_tokens or 0)
            self.tokens[purpose]["completion"] += int(completion_tokens or 0)
            self.calls[purpose] += 1
            if charge is None:
                self._unknown_cost_calls[purpose] += 1
            else:
                self._known_cost[purpose] += charge

    def known_cost_usd(self, purpose: str) -> Decimal:
        """The sum of the costs reported for *purpose*'s calls."""
        with self._lock:
            return self._known_cost.get(purpose, Decimal(0))

    def unknown_cost_calls(self, purpose: str) -> int:
        """How many of *purpose*'s calls had no reported cost."""
        with self._lock:
            return self._unknown_cost_calls.get(purpose, 0)

    def cost_usd(self, purpose: str) -> Optional[Decimal]:
        """*purpose*'s cost; ``None`` (unknown) if any of its calls' was."""
        with self._lock:
            if self._unknown_cost_calls.get(purpose, 0):
                return None
            return self._known_cost.get(purpose, Decimal(0))

    def total_cost_usd(self) -> Optional[Decimal]:
        """The run's cost over every purpose; ``None`` if any call's was unknown."""
        with self._lock:
            if any(self._unknown_cost_calls.values()):
                return None
            return sum(self._known_cost.values(), Decimal(0))

    @property
    def cost(self) -> Dict[str, float]:
        """Deprecated float view: the reported costs per purpose.

        Unknown costs count as nothing here, as they always did; use
        :meth:`cost_usd`, which keeps them unknown.
        """
        warnings.warn(
            "RunMeter.cost is deprecated: it is a float and counts unknown "
            "costs as zero. Use cost_usd(), known_cost_usd() and "
            "unknown_cost_calls().",
            DeprecationWarning,
            stacklevel=2,
        )
        with self._lock:
            return {k: float(v) for k, v in self._known_cost.items()}

    def snapshot(self) -> Dict[str, Any]:
        """Tokens, calls and cost. Costs are decimal strings: ``cost`` is
        ``None`` for a purpose with an unknown cost, ``cost_known`` sums the
        reported ones and ``cost_unknown_calls`` counts the rest."""
        with self._lock:
            return {
                "tokens": {k: dict(v) for k, v in self.tokens.items()},
                "calls": dict(self.calls),
                "cost": {
                    k: (None if self._unknown_cost_calls[k] else decimal_string(v))
                    for k, v in self._known_cost.items()
                },
                "cost_known": {
                    k: decimal_string(v) for k, v in self._known_cost.items()
                },
                "cost_unknown_calls": dict(self._unknown_cost_calls),
            }

    def total(self, purpose: str) -> int:
        with self._lock:
            entry = self.tokens.get(purpose) or {}
            return int(entry.get("prompt", 0)) + int(entry.get("completion", 0))


current_run_meter: contextvars.ContextVar[Optional[RunMeter]] = contextvars.ContextVar(
    "current_run_meter",
    default=None,
)

_listener: Optional[Any] = None
_install_lock = threading.Lock()


def _usage_from_event(event: Any) -> tuple[int, int]:
    response = getattr(event, "response", None)
    if not isinstance(response, dict):
        return 0, 0
    usage = response.get("usage") or {}
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(
        usage.get("completion_tokens") or 0,
    )


def _on_llm_event(event: Any) -> None:
    meter = current_run_meter.get()
    if meter is None:
        return
    prompt_tokens, completion_tokens = _usage_from_event(event)
    purpose = purpose_from_origin(getattr(event, "origin", None)) or "planning"
    meter.add(
        purpose,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=getattr(event, "provider_cost", None),
    )


def install_run_metering() -> None:
    """Register the process-wide listener once; idempotent."""
    global _listener
    with _install_lock:
        if _listener is None:
            _listener = unillm.add_llm_event_listener(_on_llm_event)


def new_run_meter() -> RunMeter:
    """Create a meter and make sure the listener that feeds it is installed."""
    install_run_metering()
    return RunMeter()


def handle_run_stats(handle: Any) -> Dict[str, Any]:
    """Token accounting a run handle exposes for its execution row.

    Handles that carry ``run_stats`` report it verbatim; a bare loop handle
    that only carries ``run_meter`` reports the meter's token split.
    """
    stats = getattr(handle, "run_stats", None)
    out: Dict[str, Any] = dict(stats) if isinstance(stats, dict) else {}
    meter = getattr(handle, "run_meter", None)
    if isinstance(meter, RunMeter) and not out.get("tokens"):
        out["tokens"] = meter.snapshot()["tokens"]
    return out


__all__ = [
    "PURPOSES",
    "RunMeter",
    "current_run_meter",
    "decimal_string",
    "handle_run_stats",
    "install_run_metering",
    "new_run_meter",
    "to_decimal_usd",
]
