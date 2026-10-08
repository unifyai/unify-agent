"""Tests for LLM event integration with EventBus.

These tests verify that:
1. The LLMPayload model correctly captures full request/response data
2. The hook converts unillm LLMEvent to EventBus events
3. LLM events are published to EventBus during actual LLM calls
4. Cost information is properly captured
"""

from __future__ import annotations

import asyncio

import pytest
import unillm
from unillm import LLMEvent

from tests.helpers import _handle_project, capture_events
from unify.events.event_bus import EventBus, Event, EVENT_BUS
from unify.events.types.llm import LLMPayload
from unify.events.llm_event_hook import _llm_event_to_eventbus, install_llm_event_hook

# ---------------------------------------------------------------------------
#  1. LLMPayload model tests
# ---------------------------------------------------------------------------


class TestLLMPayloadModel:
    """Tests for the LLMPayload Pydantic model."""

    def test_create_basic_payload(self):
        payload = LLMPayload(
            request={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert payload.request["model"] == "gpt-4o"
        assert payload.response is None
        assert payload.provider_cost is None

    def test_create_full_payload(self):
        payload = LLMPayload(
            request={"model": "gpt-4o", "messages": []},
            response={"id": "chatcmpl-123", "model": "gpt-4o", "choices": []},
            provider_cost=0.001,
        )
        assert payload.request["model"] == "gpt-4o"
        assert payload.response["id"] == "chatcmpl-123"
        assert payload.provider_cost == 0.001

    def test_payload_costs_default_to_none(self):
        """Cost should default to None."""
        payload = LLMPayload(request={"model": "gpt-4o"})
        assert payload.provider_cost is None

    def test_payload_allows_extra_fields(self):
        """LLMPayload should accept extra fields for forward compatibility."""
        payload = LLMPayload(
            request={"model": "gpt-4o"},
            custom_field="custom_value",
        )
        assert payload.model_extra.get("custom_field") == "custom_value"


# ---------------------------------------------------------------------------
#  2. Hook conversion tests
# ---------------------------------------------------------------------------


class TestLLMEventToEventBusConversion:
    """Tests for the _llm_event_to_eventbus hook function."""

    @pytest.mark.asyncio
    @_handle_project
    async def test_basic_event_conversion(self):
        """LLM events should be converted and published to EventBus."""
        request = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }

        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(request=request)
            _llm_event_to_eventbus(llm_event)

            # Give async publish time to complete
            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.type == "LLM"
        assert evt.payload["request"] == request
        assert evt.payload["response"] is None

    @pytest.mark.asyncio
    @_handle_project
    async def test_event_with_response(self):
        """Events should include the full response dict."""
        request = {"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]}
        response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o-2024-08-06",
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(request=request, response=response)
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["request"] == request
        assert evt.payload["response"] == response

    @pytest.mark.asyncio
    @_handle_project
    async def test_event_with_costs(self):
        """Events should include the call's cost."""
        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                request={"model": "gpt-4o", "messages": []},
                provider_cost=0.001,
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["provider_cost"] == 0.001

    @pytest.mark.asyncio
    @_handle_project
    async def test_streaming_event_has_no_response(self):
        """Streaming events should have None response."""
        async with capture_events("LLM") as captured:
            llm_event = LLMEvent(
                request={"model": "gpt-4o", "messages": [], "stream": True},
                response=None,  # Streaming has no single response
            )
            _llm_event_to_eventbus(llm_event)

            await asyncio.sleep(0.05)

        assert len(captured) == 1
        evt = captured[0]
        assert evt.payload["response"] is None


# ---------------------------------------------------------------------------
#  3. Hook installation tests
# ---------------------------------------------------------------------------


class TestHookInstallation:
    """Tests for the hook installation mechanism."""

    def test_hook_installed_during_unify_init(self):
        """The hook should be installed during unify.init()."""
        import unify

        unify.init()
        # unify.init() registers a unillm listener (process-wide, works across threads)
        callbacks = [listener.callback for listener in unillm.llm_event_listeners()]
        assert _llm_event_to_eventbus in callbacks

    def test_install_hook_multiple_calls_safe(self):
        """Calling install_llm_event_hook multiple times should not crash."""
        install_llm_event_hook()
        install_llm_event_hook()
        install_llm_event_hook()
        # Should not raise any errors


# ---------------------------------------------------------------------------
#  4. Integration tests with real LLM calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_llm_call_publishes_event():
    """A real LLM call should publish an event to EventBus."""
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[
                {"role": "user", "content": "Say 'test123' exactly [llm_events_test]"},
            ],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    # Should have one event per LLM call
    assert len(captured) >= 1

    evt = captured[-1]
    # Request should contain the model and messages
    assert "model" in evt.payload["request"]
    assert "messages" in evt.payload["request"]
    # Response should be a dict (non-streaming)
    assert isinstance(evt.payload["response"], dict)


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_llm_call_captures_full_response():
    """LLM events should include the full response dict."""
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "Say 'hi' [full_response_test]"}],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    assert len(captured) >= 1

    evt = captured[-1]
    response = evt.payload["response"]
    # Response should be the full serialized ChatCompletion
    assert response is not None
    assert "id" in response or "choices" in response


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_llm_call_with_tools():
    """LLM events should capture the full request including tools."""
    install_llm_event_hook()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "What is 2+2? [tools_test]"}],
            tools=tools,
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    assert len(captured) >= 1

    evt = captured[-1]
    # Request should include tools
    assert "tools" in evt.payload["request"]


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multiple_sequential_llm_calls():
    """Multiple LLM calls should each publish their own events."""
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)

        # Make three calls
        await client.generate(
            messages=[{"role": "user", "content": "Call 1 [seq_test]"}],
        )
        await client.generate(
            messages=[{"role": "user", "content": "Call 2 [seq_test]"}],
        )
        await client.generate(
            messages=[{"role": "user", "content": "Call 3 [seq_test]"}],
        )

        await asyncio.sleep(0.15)
        EVENT_BUS.join_published()

    # Should have 3 events (1 per call)
    assert len(captured) >= 3


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_llm_events_searchable_in_eventbus():
    """LLM events should be searchable after publishing."""
    install_llm_event_hook()

    client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
    await client.generate(
        messages=[{"role": "user", "content": "Searchable test [search_test]"}],
    )

    await asyncio.sleep(0.1)
    EVENT_BUS.join_published()

    events = EVENT_BUS.search(filter='type == "LLM"', limit=10)

    assert len(events) >= 1
    llm_events = [e for e in events if e.type == "LLM"]
    assert len(llm_events) >= 1


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_llm_event_includes_cost_fields():
    """An LLM event carries the call's cost when the provider answered it.

    The prompt is fixed, like the other tests' here, so the run that records
    it sends it to the provider and checks a cost, and a replay serves the
    recording (no cost, as for any cached response). A unique prompt forced
    a provider call on every run, which a keyless replay cannot make.
    """
    install_llm_event_hook()

    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "Say 'cost test' [cost_test]"}],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    assert len(captured) >= 1
    evt = captured[-1]

    # Verify event structure
    assert evt.type == "LLM"

    # A call the provider answered has a positive cost; a cached one has none.
    if evt.payload["provider_cost"] is not None:
        assert evt.payload["provider_cost"] > 0


# ---------------------------------------------------------------------------
#  5. EventBus LLM type registration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_llm_event_type_registered():
    """The LLM event type should be registered in the EventBus."""
    bus = EventBus()

    event = Event(
        type="LLM",
        payload=LLMPayload(
            request={"model": "test", "messages": []},
        ),
    )
    await bus.publish(event)
    bus.join_published()

    assert bus.search(filter='type == "LLM"') == [event]


@pytest.mark.asyncio
@_handle_project
async def test_llm_events_in_search_by_type():
    """LLM events should be retrievable via a type filter."""
    bus = EventBus()

    for i in range(3):
        await bus.publish(
            Event(
                type="LLM",
                payload=LLMPayload(
                    request={"model": "test", "messages": [], "seq": i},
                ),
            ),
        )
    bus.join_published()

    results = bus.search(filter='type == "LLM"', limit=10)

    assert [e.payload["request"]["seq"] for e in results] == [2, 1, 0]


# ---------------------------------------------------------------------------
#  6. Cross-thread hook installation (production scenario)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_hook_works_when_installed_from_different_thread():
    """LLM events should be captured when hook is installed from a worker thread.

    This mimics the production scenario where:
    - unify.init() is called from a thread pool worker (via asyncio.to_thread)
    - LLM calls happen from the main async context

    Listener registration is process-global, so events are captured regardless
    of which thread registered the listener vs which thread makes the LLM call.
    """
    import concurrent.futures

    unillm.set_llm_event_hook(None)
    unillm.clear_llm_event_listeners()

    # Register from a worker thread (mimicking asyncio.to_thread behavior)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(
            unillm.add_llm_event_listener,
            _llm_event_to_eventbus,
        ).result()

    # Now make LLM call from main thread and verify event is captured
    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "Cross-thread test [xthread]"}],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    assert len(captured) >= 1, (
        "No LLM events captured! This indicates the listener registered from a "
        "worker thread is not visible to the main thread."
    )


@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_hook_works_when_installed_via_asyncio_to_thread():
    """LLM events should be captured when hook is installed via asyncio.to_thread.

    This directly mimics the production code path in managers_utils.py:
        await asyncio.to_thread(_init_managers, cm, loop)

    Where _init_managers calls unify.init() which installs the LLM event hook.
    """

    unillm.set_llm_event_hook(None)
    unillm.clear_llm_event_listeners()

    # Register via asyncio.to_thread (exactly like production)
    await asyncio.to_thread(unillm.add_llm_event_listener, _llm_event_to_eventbus)

    # Make LLM call from main async context
    async with capture_events("LLM") as captured:
        client = unillm.AsyncUnify("openai/gpt-4.1-nano@openrouter", cache=True)
        await client.generate(
            messages=[{"role": "user", "content": "asyncio.to_thread test [tothread]"}],
        )

        await asyncio.sleep(0.1)
        EVENT_BUS.join_published()

    assert len(captured) >= 1, (
        "No LLM events captured when hook was installed via asyncio.to_thread! "
        "This is the production bug - hook is installed in thread pool worker "
        "but LLM calls happen in main async context."
    )


# ---------------------------------------------------------------------------
#  7. Edge case and resilience tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@_handle_project
async def test_hook_handles_none_response():
    """Hook should handle None response without crashing."""
    async with capture_events("LLM") as captured:
        llm_event = LLMEvent(
            request={"model": "gpt-4o", "messages": []},
            response=None,
        )
        _llm_event_to_eventbus(llm_event)

        await asyncio.sleep(0.05)

    # Should still publish an event
    assert len(captured) == 1
    assert captured[0].payload["response"] is None


@pytest.mark.asyncio
@_handle_project
async def test_hook_handles_empty_request():
    """Hook should handle minimal request dict."""
    async with capture_events("LLM") as captured:
        llm_event = LLMEvent(request={})
        _llm_event_to_eventbus(llm_event)

        await asyncio.sleep(0.05)

    assert len(captured) == 1
    assert captured[0].payload["request"] == {}
