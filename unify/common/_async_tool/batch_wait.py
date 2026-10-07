"""The loop's batch-wake hold, and the accounting of model turns the loop
cancels after sending them.

When a turn calls several tools and the first one returns, a loop that woke
the model at once would start its next turn on part of the batch and cancel
it when a sibling lands; the provider has already received, generated and
billed the cancelled turn. Batch wake makes waiting the loop's rule: whenever
calls are running, a hold covers all of them (the model declares nothing),
its clock starting only when the first result is held
(``from_first_result``), so a slow batch with nothing landed waits as long as
it runs and a landed result waits at most ``HOLD_CEILING_SECONDS``. Only tool
results wait for it: a new message, a clarification request or a progress
notification (from the user, the environment or another agent) still wakes
the model at once. The loop also stops cancelling a sent turn when a result
or a progress notification lands during it.

The accounting is always on and changes no request: every model turn the loop
cancels after dispatch is counted on the loop's runtime state and published as
a ``ToolLoopCancelledTurn`` event (``unify/events/types/tool_loop.py``), once
when it is cancelled and once more if unillm later reports what the provider
charged for it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional, Set

from ...logger import LOGGER as logger

# The kind of event published for each cancelled turn (an EventBus type).
CANCELLED_TURN_EVENT = "ToolLoopCancelledTurn"


# The longest a landed result is held back while other calls still run
# (seconds), counted from the first result held.
HOLD_CEILING_SECONDS = 15.0


@dataclass
class BatchHold:
    """The running calls the next turn waits for.

    ``owed`` records that a result landed while the hold kept the model
    waiting, so the turn it earned is granted when the hold ends.
    ``from_first_result`` marks the batch-wake hold: its time starts when the
    first result is held (``hold``), not when it is installed, so it never
    ends while there is nothing to wake the model with.
    """

    tasks: Set[asyncio.Task] = field(default_factory=set)
    until: Optional[float] = None
    owed: bool = False
    from_first_result: bool = False
    seconds: float = 0.0

    @property
    def declared(self) -> bool:
        return bool(self.tasks)

    def install(
        self,
        tasks: Set[asyncio.Task],
        seconds: float,
        *,
        from_first_result: bool = False,
    ) -> None:
        self.tasks = set(tasks)
        self.seconds = seconds
        self.until = None if from_first_result else time.monotonic() + seconds
        self.owed = False
        self.from_first_result = from_first_result

    def holds(self, landed: Set[asyncio.Task]) -> bool:
        """Whether the results of *landed* wait for the hold to end."""
        return self.declared

    def hold(self) -> None:
        """A result landed and waits for the hold to end: it is owed a turn,
        and a ``from_first_result`` hold's time starts now."""
        self.owed = True
        if self.until is None and self.from_first_result:
            self.until = time.monotonic() + self.seconds

    def active(self, pending: Set[asyncio.Task]) -> bool:
        """Still holding: a declared call is running and the time is not up."""
        if not (self.declared and self.tasks & pending):
            return False
        if self.until is None:
            return self.from_first_result
        return time.monotonic() < self.until

    def remaining(self) -> float:
        if self.until is None:
            return 0.0
        return max(0.0, self.until - time.monotonic())

    def time_left(self) -> Optional[float]:
        """How long the hold may still wait; ``None`` while a
        ``from_first_result`` hold has held nothing, so has no deadline."""
        if self.until is None and self.from_first_result:
            return None
        return self.remaining()

    def release(self) -> bool:
        """End the hold; return whether a held result still owes a turn."""
        owed = self.owed
        self.tasks = set()
        self.until = None
        self.owed = False
        self.from_first_result = False
        self.seconds = 0.0
        return owed


# ── cancelled-turn accounting ───────────────────────────────────────────────


def _decimal_usd(cost: Any) -> Optional[str]:
    if cost is None or isinstance(cost, bool):
        return None
    try:
        value = Decimal(str(cost))
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    return format(value, "f")


def _usage_from_response(response: Any) -> tuple[Optional[int], ...]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return None, None, None

    def _int(value: Any) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    details = usage.get("prompt_tokens_details")
    cached = _int(details.get("cached_tokens")) if isinstance(details, dict) else None
    return (
        _int(usage.get("prompt_tokens")),
        cached,
        _int(usage.get("completion_tokens")),
    )


@dataclass
class TurnMeter:
    """One dispatched model turn, watched in case the loop cancels it.

    unillm keeps a cancelled request running and, when the provider answers,
    emits an LLM event carrying what it charged. ``metered`` scopes a hook to
    the turn's own task, so that event is matched to this turn without
    correlating requests.
    """

    loop_id: str
    label: Optional[str]
    step_index: int
    on_event: Callable[[dict], None]
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: bool = False
    cause: Optional[str] = None
    cancelled_turns: int = 0
    pending_tools: int = 0

    def payload(self, phase: str, **extra: Any) -> dict:
        return {
            "turn_id": self.turn_id,
            "phase": phase,
            "loop_id": self.loop_id,
            "hierarchy_label": self.label,
            "step_index": self.step_index,
            "cause": self.cause or "unknown",
            "cancelled_turns": self.cancelled_turns,
            "pending_tools": self.pending_tools,
            **extra,
        }

    def mark_cancelled(self, cause: str, cancelled_turns: int, pending: int) -> None:
        self.cancelled = True
        self.cause = cause
        self.cancelled_turns = cancelled_turns
        self.pending_tools = pending
        self.on_event(self.payload("cancelled"))

    def observe(self, event: Any) -> None:
        """An LLM event from this turn's call path (unillm's hook)."""
        # unillm also reports the cancelled call itself, with no response;
        # only the provider's answer, which arrives later, carries a charge.
        response = getattr(event, "response", None)
        if not self.cancelled or not isinstance(response, dict):
            return
        prompt, cached, completion = _usage_from_response(response)
        self.on_event(
            self.payload(
                "billed",
                provider_cost_usd=_decimal_usd(getattr(event, "provider_cost", None)),
                prompt_tokens=prompt,
                cached_prompt_tokens=cached,
                completion_tokens=completion,
            ),
        )


async def metered(coro: Any, meter: TurnMeter) -> Any:
    """Await *coro* (one model turn) with *meter* hooked to its LLM events.

    The hook is set inside the turn's own task, so it covers this call path
    only, and it hands every event on to whichever scoped hook was active
    before, so other consumers see exactly what they saw without it.
    """
    import unillm

    previous = unillm.get_llm_event_hook()

    def _hook(event: Any) -> None:
        try:
            meter.observe(event)
        except Exception as exc:  # accounting never fails a call
            logger.error(f"cancelled-turn accounting failed: {exc!r}")
        if previous is not None:
            previous(event)

    with unillm.llm_event_hook_scope(_hook):
        return await coro


def publish(payload: dict) -> None:
    """Publish one ``ToolLoopCancelledTurn`` event; never raises."""
    try:
        from unify.events.event_bus import EVENT_BUS, Event

        loop = asyncio.get_running_loop()
        loop.create_task(
            EVENT_BUS.publish(Event(type=CANCELLED_TURN_EVENT, payload=payload)),
        )
    except Exception as exc:
        logger.debug(f"cancelled-turn event not published: {exc!r}")


def note_cancelled(
    runtime_state: Any,
    meter: TurnMeter,
    cause: str,
    pending: int,
) -> None:
    """Count a turn the loop is about to cancel and publish its first phase."""
    runtime_state.cancelled_turns += 1
    by_cause = runtime_state.cancelled_turns_by_cause
    by_cause[cause] = by_cause.get(cause, 0) + 1
    meter.mark_cancelled(cause, runtime_state.cancelled_turns, pending)


def record(runtime_state: Any, payload: dict) -> None:
    """Add a ``billed`` phase's charge to the loop's running total."""
    if payload.get("phase") != "billed":
        return
    usd = payload.get("provider_cost_usd")
    if usd is None:
        return
    runtime_state.cancelled_turns_priced += 1
    runtime_state.cancelled_turns_usd = format(
        Decimal(runtime_state.cancelled_turns_usd) + Decimal(usd),
        "f",
    )
