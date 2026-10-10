"""Wire unillm LLM events to the EventBus.

Every completed LLM call becomes one ``LLM`` event carrying the full request
and response. The listener is registered once, during ``unify.init()``, and
stays active for the lifetime of the process.

The request unillm reports is the one it sent, transport credentials included
(``api_key``, an ``Authorization`` header, a gateway URL with a password), so
the event carries a redacted copy (:func:`unify.common.redact_request.redact_llm_request`)
and never the original: whatever subscribes to or keeps the event cannot see a
key, and the request that was sent is not touched.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import unillm

if TYPE_CHECKING:
    from unillm import LLMEvent

_HOOK_INSTALLED = False


def _llm_event_to_eventbus(event: "LLMEvent") -> None:
    """Publish a unillm ``LLMEvent`` as an ``LLM`` event.

    unillm calls this synchronously after each LLM call, so the publish is
    scheduled as a task rather than awaited. With no event loop running in
    the calling thread there is nothing to schedule onto and the event is
    dropped.
    """
    from ..common.redact_request import redact_llm_request
    from .event_bus import EVENT_BUS, Event
    from .types.llm import LLMPayload

    try:
        request = redact_llm_request(event.request)
    except Exception:
        # Never publish what could not be redacted; keep only the model.
        sent = event.request if isinstance(event.request, dict) else {}
        model = sent.get("model")
        request = {
            "model": model if isinstance(model, str) else None,
            "redaction_failed": True,
        }
    payload = LLMPayload(
        request=request,
        response=event.response,
        provider_cost=event.provider_cost,
    )
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(EVENT_BUS.publish(Event(type="LLM", payload=payload)))


def install_llm_event_hook() -> None:
    """Register the listener with unillm, once per process.

    unillm listeners are process-wide, so the registration works whichever
    thread performs it: ``unify.init()`` may run in a worker thread while LLM
    calls happen on the main async context. Registration is additive, so
    other consumers (metering, benchmark harnesses) register alongside this
    one in any order without displacing it.
    """
    global _HOOK_INSTALLED

    if _HOOK_INSTALLED:
        return
    unillm.add_llm_event_listener(_llm_event_to_eventbus)
    _HOOK_INSTALLED = True
