"""Symbolic: discovery-first policy gates on FM + GM when present."""

from __future__ import annotations

import json

import pytest
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from unillm.clients.completion_mutator import CompletionMutatorContext

from unify.actor.code_act_actor import (
    CodeActActor,
    _build_discovery_parallel_mutator,
    _default_tool_policy,
    _discovery_preferred_for_schema,
    _is_discovery_gate_schema,
)
from unify.common.llm_helpers import method_to_schema


def _identity_filter(tools):
    return tools


def _unpack(result):
    """Normalize 2-tuple or eager 3-tuple policy returns."""
    mode, tools = result[0], result[1]
    opts = result[2] if len(result) == 3 else None
    return mode, tools, opts


def test_default_tool_policy_requires_both_families_when_present():
    tools = {
        "FunctionManager_search_functions": object(),
        "FunctionManager_list_functions": object(),
        "FunctionManager_filter_functions": object(),
        "FunctionManager_add_functions": object(),
        "GuidanceManager_search": object(),
        "GuidanceManager_filter": object(),
        "GuidanceManager_get_guidance": object(),
        "GuidanceManager_add_guidance": object(),
        "execute_code": object(),
    }
    policy = _default_tool_policy(
        has_fm_tools=True,
        has_gm_tools=True,
        filter_tools=_identity_filter,
    )
    mode, gated, opts = _unpack(policy(0, tools, called_tools=[]))
    assert mode == "required"
    assert opts == {"eager": True}
    assert "execute_code" not in gated
    assert set(gated) == {
        "FunctionManager_search_functions",
        "GuidanceManager_search",
    }
    assert "FunctionManager_add_functions" not in gated
    assert "GuidanceManager_add_guidance" not in gated

    mode, gated, opts = _unpack(
        policy(
            1,
            tools,
            called_tools=["FunctionManager_search_functions"],
        ),
    )
    assert mode == "required"
    assert opts == {"eager": True}
    assert set(gated) == {"GuidanceManager_search"}

    mode, full, opts = _unpack(
        policy(
            2,
            tools,
            called_tools=[
                "FunctionManager_search_functions",
                "GuidanceManager_search",
            ],
        ),
    )
    assert mode == "auto"
    assert opts is None
    assert "execute_code" in full
    assert "GuidanceManager_add_guidance" in full
    assert "FunctionManager_add_functions" in full


def test_default_tool_policy_falls_back_when_preferred_missing():
    tools = {
        "FunctionManager_list_functions": object(),
        "GuidanceManager_filter": object(),
        "execute_code": object(),
    }
    policy = _default_tool_policy(
        has_fm_tools=True,
        has_gm_tools=True,
        filter_tools=_identity_filter,
    )
    mode, gated, opts = _unpack(policy(0, tools, called_tools=[]))
    assert mode == "required"
    assert opts == {"eager": True}
    assert set(gated) == {
        "FunctionManager_list_functions",
        "GuidanceManager_filter",
    }


def test_default_tool_policy_skips_gm_gate_when_absent():
    tools = {
        "FunctionManager_search_functions": object(),
        "FunctionManager_list_functions": object(),
        "FunctionManager_add_functions": object(),
        "execute_code": object(),
    }
    policy = _default_tool_policy(
        has_fm_tools=True,
        has_gm_tools=False,
        filter_tools=_identity_filter,
    )
    mode, gated, opts = _unpack(policy(0, tools, called_tools=[]))
    assert mode == "required"
    assert opts == {"eager": True}
    assert "FunctionManager_add_functions" not in gated
    assert set(gated) == {"FunctionManager_search_functions"}

    mode, full, opts = _unpack(
        policy(
            1,
            tools,
            called_tools=["FunctionManager_search_functions"],
        ),
    )
    assert mode == "auto"
    assert opts is None
    assert "execute_code" in full
    assert "FunctionManager_add_functions" in full


def test_default_tool_policy_passes_through_without_families():
    tools = {"execute_code": object()}
    policy = _default_tool_policy(
        has_fm_tools=False,
        has_gm_tools=False,
        filter_tools=_identity_filter,
    )
    mode, full, opts = _unpack(policy(0, tools, called_tools=[]))
    assert mode == "auto"
    assert opts is None
    assert full == tools


def test_discovery_gate_schema_detection():
    assert _is_discovery_gate_schema(
        [
            "FunctionManager_search_functions",
            "GuidanceManager_search",
            "compress_context",
        ],
    )
    assert _is_discovery_gate_schema(
        [
            "FunctionManager_list_functions",
            "GuidanceManager_get_guidance",
        ],
    )
    assert not _is_discovery_gate_schema(
        [
            "FunctionManager_search_functions",
            "GuidanceManager_search",
            "execute_code",
        ],
    )
    assert not _is_discovery_gate_schema(["FunctionManager_search_functions"])


def test_discovery_preferred_for_schema_orders_families():
    preferred = _discovery_preferred_for_schema(
        [
            "GuidanceManager_search",
            "FunctionManager_search_functions",
            "compress_context",
        ],
    )
    assert [name for name, _args in preferred] == [
        "FunctionManager_search_functions",
        "GuidanceManager_search",
    ]


def _turn_calling(tool_name: str) -> ChatCompletion:
    """A discovery turn in which the model called only ``tool_name``."""
    call = ChatCompletionMessageToolCall(
        id="call_0",
        type="function",
        function=Function(name=tool_name, arguments="{}"),
    )
    message = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[call],
    )
    return ChatCompletion(
        id="turn",
        choices=[Choice(index=0, message=message, finish_reason="tool_calls")],
        created=0,
        model="test",
        object="chat.completion",
    )


@pytest.mark.asyncio
async def test_discovery_mutator_appends_calls_with_each_tools_parameter_names():
    """The tool loop refuses a call carrying an argument its tool does not
    take, so an appended call with a misnamed argument would fail its search."""
    gate = ("FunctionManager_search_functions", "GuidanceManager_search")
    actor = CodeActActor()
    try:
        tools = actor.get_tools("act")
        schemas = {
            name: method_to_schema(getattr(tools[name], "fn", tools[name]), name)
            for name in gate
        }
        context = CompletionMutatorContext(
            provider="openrouter",
            original_tool_choice="required",
            request_kw={"tools": list(schemas.values())},
        )
        mutator = _build_discovery_parallel_mutator()

        appended = {}
        for called in gate:
            turn = mutator(_turn_calling(called), context)
            for call in turn.choices[0].message.tool_calls[1:]:
                function = call["function"]
                appended[function["name"]] = json.loads(function["arguments"])

        assert set(appended) == set(gate)
        for name, args in appended.items():
            params = schemas[name]["function"]["parameters"]["properties"]
            assert args and set(args) <= set(params), (name, args, sorted(params))
    finally:
        await actor.close()
