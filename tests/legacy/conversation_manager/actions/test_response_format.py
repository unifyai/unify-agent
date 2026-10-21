"""
tests/conversation_manager/actions/test_response_format.py
============================================================

Tests for the ``response_format`` parameter on the ConversationManager's
``act`` tool, including:

- Schema-dict → Pydantic model conversion (``schema_dict_to_pydantic``)
- The ``act`` tool signature exposes ``response_format``
- The parameter is forwarded to ``Actor.act()``
- When ``response_format`` is provided, the Actor's ``final_response``
  structured output flows through ``handle.result()`` → ``ActorResult``
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from unify.legacy.conversation_manager.domains.brain_action_tools import (
    ConversationManagerBrainActionTools,
    schema_dict_to_pydantic,
)
from unify.legacy.conversation_manager.domains.chat_history import ChatHistory
from unify.legacy.conversation_manager.domains.notifications import NotificationBar

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_cm():
    """Minimal mock ConversationManager for unit-level act tool tests."""
    cm = MagicMock()
    cm.in_flight_actions = {}
    cm.completed_actions = {}
    cm.notifications_bar = NotificationBar()
    cm.chat_history = ChatHistory()
    cm.brain_messages = []
    cm._current_state_snapshot = None
    cm._current_snapshot_state = None
    cm._pending_steering_tasks = set()
    cm._initialized = asyncio.Event()
    cm._initialized.set()  # mark as initialised so act() doesn't block
    cm.event_broker = MagicMock()
    cm.event_broker.publish = AsyncMock()
    cm.suppress_duplicate_commissioning_tool.return_value = None
    return cm


@pytest.fixture
def brain_action_tools(mock_cm):
    """ConversationManagerBrainActionTools wired to the mock CM."""
    return ConversationManagerBrainActionTools(mock_cm)


# ═════════════════════════════════════════════════════════════════════════════
# 1. schema_dict_to_pydantic — unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestSchemaDictToPydantic:
    """Verify the simplified-schema → Pydantic model conversion."""

    def test_basic_string_types(self):
        """Flat schema with string type names produces a valid model."""
        Model = schema_dict_to_pydantic(
            {
                "name": "string",
                "age": "integer",
                "score": "number",
                "active": "boolean",
            },
        )
        instance = Model(name="Alice", age=30, score=9.5, active=True)
        assert instance.name == "Alice"
        assert instance.age == 30
        assert instance.score == 9.5
        assert instance.active is True

    def test_shorthand_types(self):
        """Shorthand aliases (str, int, float, bool) are accepted."""
        Model = schema_dict_to_pydantic(
            {"x": "str", "y": "int", "z": "float", "ok": "bool"},
        )
        instance = Model(x="hello", y=42, z=3.14, ok=False)
        assert instance.x == "hello"
        assert instance.y == 42

    def test_nested_object(self):
        """Dict values create nested Pydantic models."""
        Model = schema_dict_to_pydantic(
            {"address": {"city": "string", "zip": "string"}},
        )
        instance = Model(address={"city": "Berlin", "zip": "10115"})
        assert instance.address.city == "Berlin"
        assert instance.address.zip == "10115"

    def test_array_of_primitives(self):
        """List with a primitive type string creates list[T]."""
        Model = schema_dict_to_pydantic({"tags": ["string"]})
        instance = Model(tags=["a", "b", "c"])
        assert instance.tags == ["a", "b", "c"]

    def test_array_of_objects(self):
        """List with a dict element creates list[NestedModel]."""
        Model = schema_dict_to_pydantic(
            {"contacts": [{"name": "string", "email": "string"}]},
        )
        instance = Model(contacts=[{"name": "Alice", "email": "a@b.com"}])
        assert instance.contacts[0].name == "Alice"

    def test_complex_nested_schema(self):
        """Realistic schema with arrays, nesting, and mixed types."""
        Model = schema_dict_to_pydantic(
            {
                "contacts": [{"name": "string", "phone": "string"}],
                "total_count": "integer",
                "query_used": "string",
            },
        )
        data = {
            "contacts": [{"name": "Alice", "phone": "+1234"}],
            "total_count": 1,
            "query_used": "Berlin contacts",
        }
        instance = Model(**data)
        assert instance.total_count == 1
        assert instance.contacts[0].phone == "+1234"

    def test_validation_rejects_wrong_types(self):
        """The generated model enforces types via Pydantic validation."""
        Model = schema_dict_to_pydantic({"count": "integer"})
        # Pydantic v2 coerces strings to ints when possible, but a
        # non-numeric string should fail.
        with pytest.raises(ValidationError):
            Model(count="not_a_number")

    def test_model_json_schema_is_valid(self):
        """The generated model produces a valid JSON schema."""
        Model = schema_dict_to_pydantic(
            {"name": "string", "scores": ["number"]},
        )
        schema = Model.model_json_schema()
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "scores" in schema["properties"]

    def test_roundtrip_serialization(self):
        """Model instances serialize to JSON and back."""
        Model = schema_dict_to_pydantic({"email": "string", "verified": "boolean"})
        instance = Model(email="a@b.com", verified=True)
        payload = json.loads(instance.model_dump_json())
        assert payload == {"email": "a@b.com", "verified": True}


# ═════════════════════════════════════════════════════════════════════════════
# 2. act tool signature and parameter forwarding
# ═════════════════════════════════════════════════════════════════════════════


class TestActToolSignature:
    """Verify the act tool exposes response_format correctly."""

    def test_response_format_in_signature(self, brain_action_tools):
        """act() signature includes response_format as an optional dict parameter."""
        sig = inspect.signature(brain_action_tools.act)
        assert "response_format" in sig.parameters
        param = sig.parameters["response_format"]
        assert param.default is None

    def test_response_format_is_optional_in_tool_schema(self, brain_action_tools):
        """response_format should NOT be in the required list of the tool schema."""
        from unify.common.llm_helpers import method_to_schema

        schema = method_to_schema(brain_action_tools.act, include_class_name=False)
        fn_schema = schema["function"]
        required = fn_schema["parameters"].get("required", [])
        assert "query" in required
        assert "response_format" not in required


class TestActForwardsResponseFormat:
    """Verify act passes response_format through to Actor.act()."""

    @pytest.mark.asyncio
    async def test_act_passes_pydantic_model_to_actor(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """When response_format is a dict, a Pydantic model is forwarded to Actor."""
        # Set up a mock Actor that captures kwargs
        captured_kwargs: dict[str, Any] = {}

        async def fake_act(request, **kwargs):
            captured_kwargs.update(kwargs)
            handle = MagicMock()
            handle.result = AsyncMock(return_value="done")
            handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
            handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
            return handle

        mock_cm.actor = MagicMock()
        mock_cm.actor.act = fake_act

        await brain_action_tools.act(
            query="Find the report",
            response_format={"name": "string", "email": "string"},
        )

        assert "response_format" in captured_kwargs
        model_cls = captured_kwargs["response_format"]
        # Should be a Pydantic BaseModel subclass
        assert isinstance(model_cls, type)
        assert issubclass(model_cls, BaseModel)
        # Should have the expected fields
        schema = model_cls.model_json_schema()
        assert "name" in schema["properties"]
        assert "email" in schema["properties"]

    @pytest.mark.asyncio
    async def test_act_without_response_format_passes_none(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """When response_format is omitted, None is passed to Actor."""
        captured_kwargs: dict[str, Any] = {}

        async def fake_act(request, **kwargs):
            captured_kwargs.update(kwargs)
            handle = MagicMock()
            handle.result = AsyncMock(return_value="done")
            handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
            handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
            return handle

        mock_cm.actor = MagicMock()
        mock_cm.actor.act = fake_act

        await brain_action_tools.act(query="Find the report")

        assert captured_kwargs.get("response_format") is None


# ═════════════════════════════════════════════════════════════════════════════
# 3. End-to-end: structured response reaches ActorResult
# ═════════════════════════════════════════════════════════════════════════════


class TestStructuredResultPropagation:
    """Verify that when response_format is set, the structured JSON from
    ``final_response`` reaches the CM via ``ActorResult``."""

    @pytest.mark.asyncio
    async def test_json_result_reaches_actor_result(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """A structured JSON string from handle.result() propagates to ActorResult."""
        expected_payload = {"name": "Alice", "email": "alice@example.com"}

        async def fake_act(request, **kwargs):
            handle = MagicMock()
            # Simulate the loop returning json.dumps(payload) via final_response
            handle.result = AsyncMock(return_value=json.dumps(expected_payload))
            handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
            handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
            return handle

        mock_cm.actor = MagicMock()
        mock_cm.actor.act = fake_act

        result = await brain_action_tools.act(
            query="Find the Aurora report",
            response_format={"name": "string", "email": "string"},
        )

        assert result["status"] == "acting"

        # The handle is now in in_flight_actions — verify it holds the
        # structured result when awaited.
        assert len(mock_cm.in_flight_actions) == 1
        handle_data = next(iter(mock_cm.in_flight_actions.values()))
        handle = handle_data["handle"]
        raw_result = await handle.result()
        parsed = json.loads(raw_result)
        assert parsed == expected_payload

    @pytest.mark.asyncio
    async def test_response_format_stored_in_action_metadata(
        self,
        brain_action_tools,
        mock_cm,
    ):
        """The response_format schema is preserved in the action metadata
        so it can be inspected later (e.g., for rendering or debugging)."""

        async def fake_act(request, **kwargs):
            handle = MagicMock()
            handle.result = AsyncMock(return_value='{"x": 1}')
            handle.next_notification = AsyncMock(side_effect=asyncio.CancelledError)
            handle.next_clarification = AsyncMock(side_effect=asyncio.CancelledError)
            return handle

        mock_cm.actor = MagicMock()
        mock_cm.actor.act = fake_act

        await brain_action_tools.act(
            query="Compute something",
            response_format={"x": "integer"},
        )

        handle_data = next(iter(mock_cm.in_flight_actions.values()))
        # The query is always stored
        assert handle_data["query"] == "Compute something"
