"""
Test driver for ConversationManager stepping.

Provides deterministic event-by-event testing without relying on background
tasks or the async event loop. This keeps test-only code out of production
modules while providing a clean stepping API.

Usage:
    driver = CMStepDriver(conversation_manager)
    result = await driver.step(some_event)
    assert driver.cm.in_flight_actions  # access underlying CM state
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import TYPE_CHECKING

from unify.legacy.conversation_manager.domains.event_handlers import EventHandler
from unify.legacy.conversation_manager.events import (
    Event,
    UnifyMessageSent,
    ActorHandleStarted,
)

# Events a step reports back to the test as its observable output: outbound
# chat messages and action starts.
_OUTPUT_EVENT_TYPES = (UnifyMessageSent, ActorHandleStarted)

if TYPE_CHECKING:
    from unify.legacy.conversation_manager.conversation_manager import (
        ConversationManager,
    )


@dataclass(frozen=True)
class StepResult:
    """Result of processing a single event step."""

    input_event: Event
    llm_requested: bool
    llm_ran: bool
    output_events: list[Event]
    llm_step_count: int = 0  # Number of LLM thinking steps taken


# Context variable to track LLM run requests during stepping
_step_llm_requests: contextvars.ContextVar[list[tuple[float, bool]] | None] = (
    contextvars.ContextVar("_step_llm_requests", default=None)
)


class CMStepDriver:
    """
    Test driver for stepping through ConversationManager events.

    Provides deterministic event-by-event testing by:
    - Recording LLM run requests during event handling
    - Running the LLM immediately (instead of via debouncer)
    - Capturing and applying published output events to local state

    Attributes are proxied to the underlying ConversationManager, so you can
    access cm.chat_history, cm.in_flight_actions, etc. directly through the driver.
    """

    def __init__(self, cm: "ConversationManager"):
        self._cm = cm
        # Track all tool calls made across all steps for test assertions
        self.all_tool_calls: list[str] = []

    @property
    def cm(self) -> "ConversationManager":
        """Access the underlying ConversationManager."""
        return self._cm

    def __getattr__(self, name: str):
        """Proxy attribute access to the underlying ConversationManager."""
        return getattr(self._cm, name)

    async def step(
        self,
        event: Event,
        *,
        publish: bool = False,
        run_llm: bool = True,
    ) -> StepResult:
        """Process one event deterministically and return produced output events.

        This bypasses the normal async event-driven flow by:
        - Recording any requested LLM runs during event handling
        - Running the LLM immediately (if requested and ``run_llm`` is True)
        - Capturing and applying any published output events to local state

        Args:
            event: Input event to process.
            publish: Whether to forward published events to the broker.
            run_llm: Whether to execute the LLM when the event handler requests
                it.  Set to ``False`` for infrastructure-only tests that verify
                event handling (thread indexing, role tagging, etc.) without
                introducing nondeterminism from the LLM deciding to respond.

        Returns:
            StepResult with output events produced during this step.
        """
        published_events: list[Event] = []
        output_events: list[Event] = []
        llm_requested = False
        llm_ran = False

        original_publish = self._cm.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                published_events.append(evt)
            # Don't forward to the real broker. The post-processing loop below
            # handles all published events locally via EventHandler.handle_event().
            # Forwarding would cause double-handling because wait_for_events()
            # (running as a background task) also subscribes to these channels.
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)

        # Patch request_llm_run to use our context var
        original_request = self._cm.request_llm_run

        async def patched_request(
            delay=0,
            is_user_origin=False,
            **kwargs,
        ) -> None:
            requests = _step_llm_requests.get()
            if requests is not None:
                requests.append((delay, is_user_origin))
                return
            # Fall back to normal behavior if not in step context
            self._cm._pending_llm_requests.append(
                (delay, is_user_origin),
            )

        try:
            self._cm.event_broker.publish = publish_wrapper
            self._cm.request_llm_run = patched_request

            await EventHandler.handle_event(
                event,
                self._cm,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            if llm_requested and run_llm:
                llm_ran = True
                tool_names = await self._cm._run_llm()
                # Track tool calls for test assertions
                self.all_tool_calls.extend(tool_names)

                # Await any pending steering tasks (e.g., async ask_*)
                # so their events flow through our patches while active.
                pending = set(self._cm._pending_steering_tasks)
                if pending:
                    await asyncio.wait(pending, timeout=120)

            # Apply any published events to local state so callers can inspect state
            # without depending on background broker subscribers.
            for evt in published_events:
                if isinstance(evt, _OUTPUT_EVENT_TYPES):
                    output_events.append(evt)
                await EventHandler.handle_event(
                    evt,
                    self._cm,
                )
        finally:
            self._cm.event_broker.publish = original_publish
            self._cm.request_llm_run = original_request
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=llm_requested,
            llm_ran=llm_ran,
            output_events=output_events,
        )

    async def step_until_wait(
        self,
        event: Event,
        *,
        max_steps: int = 5,
        publish: bool = False,
    ) -> StepResult:
        """Process an event and keep running LLM until it calls 'wait'.

        This gives the LLM continuous control until it explicitly decides to
        stop by calling the 'wait' tool.

        Args:
            event: Input event to process.
            max_steps: Maximum LLM steps to prevent infinite loops (default 5).
            publish: Whether to forward published events to the broker.

        Returns:
            StepResult with all output events produced across all steps.
        """
        all_output_events: list[Event] = []
        llm_ran = False

        original_publish = self._cm.event_broker.publish

        async def publish_wrapper(channel: str, message: str) -> int:
            try:
                evt = Event.from_json(message)
            except Exception:
                evt = None
            if evt is not None:
                if isinstance(evt, _OUTPUT_EVENT_TYPES):
                    all_output_events.append(evt)
                # Handle the event locally for deterministic state updates.
                # Don't forward to the real broker — wait_for_events() (running
                # as a background task) subscribes to these channels and would
                # process the same event a second time, causing duplicate chat
                # sends, duplicate notifications, and uncontrolled LLM runs.
                await EventHandler.handle_event(
                    evt,
                    self._cm,
                )
            return 0

        step_requests: list[tuple[float, bool]] = []
        token = _step_llm_requests.set(step_requests)

        # Patch request_llm_run to use our context var
        original_request = self._cm.request_llm_run

        async def patched_request(
            delay=0,
            is_user_origin=False,
            **kwargs,
        ) -> None:
            requests = _step_llm_requests.get()
            if requests is not None:
                requests.append((delay, is_user_origin))
                return
            self._cm._pending_llm_requests.append(
                (delay, is_user_origin),
            )

        try:
            self._cm.event_broker.publish = publish_wrapper
            self._cm.request_llm_run = patched_request

            # First, handle the incoming event
            await EventHandler.handle_event(
                event,
                self._cm,
            )

            llm_requested = bool(step_requests)
            step_requests.clear()

            # Run LLM in a loop until 'wait' is called or max_steps reached
            step_count = 0
            while llm_requested and step_count < max_steps:
                llm_ran = True
                tool_names = await self._cm._run_llm()
                step_count += 1

                # Track tool calls for test assertions
                self.all_tool_calls.extend(tool_names)

                # Await any pending steering tasks (e.g., async ask_*)
                # so their events flow through our publish_wrapper and
                # request_llm_run patch while they're still active.
                pending = set(self._cm._pending_steering_tasks)
                if pending:
                    await asyncio.wait(pending, timeout=120)

                # Check if another LLM run was requested (e.g., by event handlers
                # processing events from background tasks)
                llm_requested = bool(step_requests)
                step_requests.clear()

                # Stop if 'wait' was called AND no new requests came in
                if "wait" in tool_names and not llm_requested:
                    break

        finally:
            self._cm.event_broker.publish = original_publish
            self._cm.request_llm_run = original_request
            _step_llm_requests.reset(token)

        return StepResult(
            input_event=event,
            llm_requested=True,
            llm_ran=llm_ran,
            output_events=all_output_events,
            llm_step_count=step_count,
        )
