"""Tests for LLM event integration with EventBus.

These tests verify that:
1. The LLMPayload model correctly captures full request/response data
2. The hook converts unillm LLMEvent to EventBus events
3. LLM events are published to EventBus during calls through unillm (the
   provider is scripted, tests/scripted_model.py)
4. Cost information is properly captured
"""

from __future__ import annotations

import asyncio
import json

import pytest
import unillm
from openai.types.chat import ChatCompletion
from unillm import LLMEvent

from tests.helpers import _handle_project, capture_events
from tests.scripted_model import ScriptedModel, reply, scripted
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
#  4. Integration tests: a call through unillm publishes its event
# ---------------------------------------------------------------------------
#
# The provider is the scripted transport (tests/scripted_model.py), below
# unillm's request building and event emission: each call goes through the
# client as in production and emits its LLMEvent, and nothing leaves the
# process. Clients are built with cache=False, so every call reaches the
# transport (a cached reply is not the subject here).

_MODEL = "openai/gpt-4.1-nano@openrouter"


def _client():
    return unillm.AsyncUnify(_MODEL, cache=False)


def _usage_with_cost(cost: float) -> ChatCompletion:
    """A reply whose usage carries OpenRouter's charged amount (``usage.cost``)."""
    data = reply("cost test").model_dump()
    data["usage"]["cost"] = cost
    return ChatCompletion.model_validate(data)


def _user_text(event) -> str:
    return json.dumps(event.payload["request"]["messages"][-1]["content"])


@pytest.mark.asyncio
@_handle_project
async def test_llm_call_publishes_event():
    """A call through unillm publishes an LLM event to the EventBus."""
    install_llm_event_hook()
    model = ScriptedModel(query_llm=[reply("test123")])

    with scripted(model):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[
                    {"role": "user", "content": "Say 'test123' exactly [llm_events_test]"},
                ],
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    # One event for the one call.
    assert len(captured) == 1

    evt = captured[-1]
    # Request should contain the model and messages
    assert "model" in evt.payload["request"]
    assert "llm_events_test" in _user_text(evt)
    # Response should be a dict (non-streaming)
    assert isinstance(evt.payload["response"], dict)
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_llm_call_captures_full_response():
    """LLM events should include the full response dict."""
    install_llm_event_hook()
    model = ScriptedModel(query_llm=[reply("hi there")])

    with scripted(model):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[{"role": "user", "content": "Say 'hi' [full_response_test]"}],
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    assert len(captured) == 1

    response = captured[-1].payload["response"]
    # The full serialized ChatCompletion: its id, the reply and the usage.
    assert response["id"] == "cmpl-scripted"
    assert response["choices"][0]["message"]["content"] == "hi there"
    assert response["usage"]["prompt_tokens"] == 100


@pytest.mark.asyncio
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
    model = ScriptedModel(actor=[reply(calls=[("search", {})])])

    with scripted(model):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[{"role": "user", "content": "What is 2+2? [tools_test]"}],
                tools=tools,
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    assert len(captured) == 1

    evt = captured[-1]
    # Request should include tools
    assert [t["function"]["name"] for t in evt.payload["request"]["tools"]] == [
        "search",
    ]
    message = evt.payload["response"]["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "search"


@pytest.mark.asyncio
@_handle_project
async def test_multiple_sequential_llm_calls():
    """Multiple LLM calls should each publish their own events."""
    install_llm_event_hook()
    model = ScriptedModel(query_llm=[reply("one"), reply("two"), reply("three")])

    with scripted(model):
        async with capture_events("LLM") as captured:
            client = _client()

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

    # One event per call, in the order of the calls.
    assert len(captured) == 3
    for n, evt in enumerate(captured, start=1):
        assert f"Call {n} [seq_test]" in _user_text(evt)
    assert [
        e.payload["response"]["choices"][0]["message"]["content"] for e in captured
    ] == ["one", "two", "three"]
    model.assert_used_up()


@pytest.mark.asyncio
@_handle_project
async def test_llm_events_searchable_in_eventbus():
    """LLM events should be searchable after publishing."""
    install_llm_event_hook()
    model = ScriptedModel(query_llm=[reply("found")])

    with scripted(model):
        await _client().generate(
            messages=[{"role": "user", "content": "Searchable test [search_test]"}],
        )

    await asyncio.sleep(0.1)
    EVENT_BUS.join_published()

    events = EVENT_BUS.search(filter='type == "LLM"', limit=10)

    llm_events = [e for e in events if e.type == "LLM"]
    assert any("search_test" in _user_text(e) for e in llm_events), llm_events


@pytest.mark.asyncio
@_handle_project
async def test_llm_event_includes_cost_fields():
    """An LLM event carries the call's cost when the provider answered it.

    OpenRouter reports what it charged on ``usage.cost``; the scripted reply
    carries one, and the event reports it as ``provider_cost``.
    """
    install_llm_event_hook()
    model = ScriptedModel(query_llm=[_usage_with_cost(0.000042)])

    with scripted(model):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[{"role": "user", "content": "Say 'cost test' [cost_test]"}],
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    assert len(captured) == 1
    evt = captured[-1]

    # Verify event structure
    assert evt.type == "LLM"
    assert evt.payload["provider_cost"] == pytest.approx(0.000042)


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
    with scripted(ScriptedModel(query_llm=[reply("seen")])):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[{"role": "user", "content": "Cross-thread test [xthread]"}],
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    assert len(captured) == 1, (
        "No LLM events captured! This indicates the listener registered from a "
        "worker thread is not visible to the main thread."
    )


@pytest.mark.asyncio
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
    with scripted(ScriptedModel(query_llm=[reply("seen")])):
        async with capture_events("LLM") as captured:
            await _client().generate(
                messages=[
                    {"role": "user", "content": "asyncio.to_thread test [tothread]"},
                ],
            )

            await asyncio.sleep(0.1)
            EVENT_BUS.join_published()

    assert len(captured) == 1, (
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
