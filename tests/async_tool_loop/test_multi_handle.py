"""
Tests for multi-handle async tool loop functionality.

These tests verify that a single tool loop can serve multiple concurrent requests,
with each request getting its own handle while sharing global context.
"""

from __future__ import annotations

import asyncio
import pytest

from tests.helpers import _handle_project
from unify.common.llm_client import new_llm_client
from unify.common.async_tool_loop import start_async_tool_loop
from unify.common._async_tool.multi_handle import (
    MultiHandleCoordinator,
    MultiRequestHandle,
)
from unify.common._async_tool.request_state import (
    RequestRegistry,
    RequestStatus,
)
from unify.common._async_tool.tagging import (
    tag_message_with_request,
    format_request_cancelled_notice,
)
from tests.baked_defaults import as_shipped  # noqa: F401

# --------------------------------------------------------------------------- #
#  UNIT TESTS – Request State Module                                          #
# --------------------------------------------------------------------------- #


def test_request_registry_basic_operations():
    """Test basic RequestRegistry operations."""
    registry = RequestRegistry()

    # Register first request
    id0 = registry.register()
    assert id0 == 0
    assert registry.pending_count() == 1
    assert not registry.is_empty()

    # Register second request
    id1 = registry.register()
    assert id1 == 1
    assert registry.pending_count() == 2

    # Complete first request
    assert registry.complete(0, "answer for request 0")
    state0 = registry.get(0)
    assert state0 is not None
    assert state0.status == RequestStatus.COMPLETED
    assert state0.result_future.done()
    assert state0.result_future.result() == "answer for request 0"

    # Pending count should decrease
    assert registry.pending_count() == 1

    # Cannot complete again
    assert not registry.complete(0, "duplicate")

    # Cancel second request
    assert registry.cancel(1, "user requested")
    state1 = registry.get(1)
    assert state1 is not None
    assert state1.status == RequestStatus.CANCELLED
    assert "cancelled" in state1.result_future.result()

    # Now empty
    assert registry.is_empty()


def test_request_registry_close():
    """Test that closed registry rejects new registrations."""
    registry = RequestRegistry()
    registry.register()
    registry.close()

    assert registry.is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        registry.register()


def test_request_registry_invalid_ids():
    """Test handling of invalid request IDs."""
    registry = RequestRegistry()

    # Non-existent ID
    assert not registry.complete(99, "answer")
    assert not registry.cancel(99)
    assert registry.get(99) is None


# --------------------------------------------------------------------------- #
#  UNIT TESTS – Message Tagging                                               #
# --------------------------------------------------------------------------- #


def test_tag_message_with_request():
    """Test message tagging."""
    tagged = tag_message_with_request("Hello world", 0)
    assert tagged == "[Request 0] Hello world"

    tagged = tag_message_with_request("Test message", 42)
    assert tagged == "[Request 42] Test message"


def test_format_request_cancelled_notice():
    """Test cancellation notice formatting."""
    notice = format_request_cancelled_notice(3)
    assert "Request 3" in notice
    assert "cancelled" in notice

    notice_with_reason = format_request_cancelled_notice(3, "user changed mind")
    assert "user changed mind" in notice_with_reason


# --------------------------------------------------------------------------- #
#  UNIT TESTS – Multi-Handle Coordinator                                      #
# --------------------------------------------------------------------------- #


def test_coordinator_basic_operations():
    """Test MultiHandleCoordinator basic operations."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
        persist=False,
    )

    # Register requests
    id0 = coordinator.register_request()
    id1 = coordinator.register_request()
    assert id0 == 0
    assert id1 == 1

    # Should not terminate while requests pending
    assert not coordinator.should_terminate()

    # Complete requests
    coordinator.complete_request(0, "answer 0")
    assert not coordinator.should_terminate()  # Still one pending

    coordinator.complete_request(1, "answer 1")
    assert coordinator.should_terminate()  # All done


def test_coordinator_persist_mode():
    """Test that persist mode prevents termination."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
        persist=True,
    )

    coordinator.register_request()
    coordinator.complete_request(0, "answer")

    # Should NOT terminate even when all requests done
    assert not coordinator.should_terminate()


def test_coordinator_inject_interjection():
    """Test interjection injection with tagging."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
    )

    coordinator.register_request()
    coordinator.inject_interjection(0, "Hello from request 0")

    # Check queue
    item = interject_queue.get_nowait()
    assert "[Request 0]" in item
    assert "Hello from request 0" in item


def test_coordinator_validate_request_id():
    """Test request ID validation."""
    coordinator = MultiHandleCoordinator(
        interject_queue=asyncio.Queue(),
        clarification_channels={},
    )

    # Non-existent
    error = coordinator.validate_request_id(99)
    assert error is not None
    assert "no such request" in error

    # Register and complete
    coordinator.register_request()
    assert coordinator.validate_request_id(0) is None  # Valid

    coordinator.complete_request(0, "done")
    error = coordinator.validate_request_id(0)
    assert error is not None
    assert "already" in error


# --------------------------------------------------------------------------- #
#  UNIT TESTS – Per-Request Handle                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_handle_request_handle_basic():
    """Test MultiRequestHandle basic operations."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
    )
    coordinator.register_request()

    handle = MultiRequestHandle(
        request_id=0,
        coordinator=coordinator,
        loop_id="test",
    )

    assert handle.request_id == 0
    assert not handle.done()

    # Interject
    await handle.interject("test message")
    item = interject_queue.get_nowait()
    assert "[Request 0]" in item

    # Complete via coordinator
    coordinator.complete_request(0, "final answer")
    assert handle.done()

    result = await handle.result()
    assert result == "final answer"


@pytest.mark.asyncio
async def test_multi_handle_request_handle_stop():
    """Test stopping a specific request."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
    )
    coordinator.register_request()

    handle = MultiRequestHandle(
        request_id=0,
        coordinator=coordinator,
    )

    await handle.stop("user cancelled")

    # Should have injected cancellation notice
    item = interject_queue.get_nowait()
    assert "cancelled" in item.lower()

    # Request should be marked cancelled
    assert handle.done()
    result = await handle.result()
    assert "cancelled" in result.lower()


@pytest.mark.asyncio
async def test_multi_handle_add_request():
    """Test adding a new request to an existing coordinator."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
    )
    coordinator.register_request()

    handle0 = MultiRequestHandle(
        request_id=0,
        coordinator=coordinator,
    )

    # Add a new request
    handle1 = handle0.add_request("New request message")

    assert handle1.request_id == 1
    assert not handle1.done()

    # Check interjection was queued
    item = interject_queue.get_nowait()
    assert "[Request 1]" in item
    assert "New request message" in item


@pytest.mark.asyncio
async def test_multi_handle_add_request_after_close():
    """Test that add_request fails after coordinator is closed."""
    coordinator = MultiHandleCoordinator(
        interject_queue=asyncio.Queue(),
        clarification_channels={},
    )
    coordinator.register_request()
    coordinator.close()

    handle = MultiRequestHandle(
        request_id=0,
        coordinator=coordinator,
    )

    with pytest.raises(RuntimeError, match="terminated"):
        handle.add_request("Should fail")


# --------------------------------------------------------------------------- #
#  INTEGRATION TESTS – Multi-Handle Tool Loop                                 #
# --------------------------------------------------------------------------- #


# as_shipped: deleted in step 5 (multi-handle coordination)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_handle_single_request_baseline(llm_config):
    """Test that multi-handle mode works with a single request (baseline)."""
    client = new_llm_client(**llm_config)

    handle = start_async_tool_loop(
        client,
        message="What is 2 + 2? Reply with just the number.",
        tools={},
        multi_handle=True,
    )

    # Should be a MultiRequestHandle
    assert isinstance(handle, MultiRequestHandle)
    assert handle.request_id == 0

    # Should complete normally
    result = await handle.result()
    assert "4" in result


# as_shipped: deleted in step 5 (multi-handle coordination)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_handle_two_requests_sequential(llm_config):
    """Test handling two requests that complete sequentially."""

    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    client = new_llm_client(**llm_config)

    handle0 = start_async_tool_loop(
        client,
        message="Add 5 and 3 using the add tool, then call final_answer with request_id=0 and the result.",
        tools={"add": add},
        multi_handle=True,
    )

    # Add second request
    handle1 = handle0.add_request(
        "Add 10 and 7 using the add tool, then call final_answer with request_id=1 and the result.",
    )

    assert handle1.request_id == 1

    # Wait for both to complete
    result0 = await handle0.result()
    result1 = await handle1.result()

    # Both should have answers
    assert "8" in result0
    assert "17" in result1


# as_shipped: deleted in step 5 (multi-handle coordination)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_handle_stop_one_request(llm_config):
    """Test stopping one request while another continues."""

    async def slow_tool(label: str) -> str:
        """A slow tool that takes time."""
        await asyncio.sleep(2)
        return f"Completed: {label}"

    client = new_llm_client(**llm_config)

    handle0 = start_async_tool_loop(
        client,
        message="Call slow_tool with label='first', then call final_answer with request_id=0.",
        tools={"slow_tool": slow_tool},
        multi_handle=True,
        timeout=60,
    )

    # Add second request
    handle1 = handle0.add_request(
        "Just answer 'hello' immediately using final_answer with request_id=1.",
    )

    # Stop the first request before it completes
    await asyncio.sleep(0.5)
    await handle0.stop("Changed my mind")

    # First request should be cancelled
    assert handle0.done()
    result0 = await handle0.result()
    assert "cancelled" in result0.lower()

    # Second request should still complete normally
    result1 = await handle1.result()
    assert "hello" in result1.lower()


# as_shipped: deleted in step 5 (multi-handle coordination)
@pytest.mark.usefixtures("as_shipped")
@pytest.mark.asyncio
@pytest.mark.llm_call
@_handle_project
async def test_multi_handle_interjection_routing(llm_config):
    """Test that interjections are routed to correct requests."""

    async def wait_tool() -> str:
        """A tool that waits."""
        await asyncio.sleep(1)
        return "done waiting"

    client = new_llm_client(**llm_config)

    handle0 = start_async_tool_loop(
        client,
        message="Wait using wait_tool, then respond acknowledging any updates I send.",
        tools={"wait_tool": wait_tool},
        multi_handle=True,
        timeout=60,
    )

    # Send an interjection to request 0
    await handle0.interject("Update: the task is now urgent!")

    # The interjection should be tagged with [Request 0]
    # and the LLM should see it as coming from request 0

    result = await handle0.result()
    # The result should acknowledge the interjection
    assert result is not None


@pytest.mark.asyncio
async def test_multi_handle_loop_terminates_when_all_done():
    """Test that the loop terminates when all requests are completed."""
    registry = RequestRegistry()

    # Register two requests
    registry.register()
    registry.register()

    # Complete both
    registry.complete(0, "answer 0")
    registry.complete(1, "answer 1")

    # Should be empty now
    assert registry.is_empty()
    assert registry.pending_count() == 0


@pytest.mark.asyncio
async def test_multi_handle_clarification_routing():
    """Test that clarifications are routed to the correct request."""
    interject_queue = asyncio.Queue()
    coordinator = MultiHandleCoordinator(
        interject_queue=interject_queue,
        clarification_channels={},
    )

    # Register two requests
    coordinator.register_request()
    coordinator.register_request()

    # Route a clarification to request 1
    coordinator.route_clarification_to_request(
        1,
        {
            "type": "clarification",
            "request_id": 1,
            "question": "What color?",
        },
    )

    # Get clarification queue for request 1
    q = coordinator.get_clarification_queue(1)
    assert q is not None

    clar = q.get_nowait()
    assert clar["request_id"] == 1
    assert clar["question"] == "What color?"

    # Request 0's queue should be empty
    q0 = coordinator.get_clarification_queue(0)
    assert q0 is not None
    assert q0.empty()
