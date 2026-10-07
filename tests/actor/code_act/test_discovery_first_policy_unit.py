"""Symbolic: discovery-first policy gates on FM + GM when present."""

from __future__ import annotations


from unify.actor.code_act_actor import (
    _default_tool_policy,
    _discovery_preferred_for_schema,
    _is_discovery_gate_schema,
)


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
    # UNIFY_CACHE_DISCIPLINE (on by default) masks the other tools by rule.
    assert opts["eager"] is True and "searched first" in opts["mask_rule"]
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
    # UNIFY_CACHE_DISCIPLINE (on by default) masks the other tools by rule.
    assert opts["eager"] is True and "searched first" in opts["mask_rule"]
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
    # UNIFY_CACHE_DISCIPLINE (on by default) masks the other tools by rule.
    assert opts["eager"] is True and "searched first" in opts["mask_rule"]
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
    # UNIFY_CACHE_DISCIPLINE (on by default) masks the other tools by rule.
    assert opts["eager"] is True and "searched first" in opts["mask_rule"]
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
