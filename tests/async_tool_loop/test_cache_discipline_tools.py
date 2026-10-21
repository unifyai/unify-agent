"""Symbolic: a session keeps one tool list (the cache discipline).

A provider reuses its prompt cache only as far as a request matches the last
one byte for byte, and the tool list comes first. Every actor conversation
used to change it on its second call (5 tools while the discovery gate was
open, 18 after, on AppWorld), so the cached prefix was lost there. Now it is
fixed: the list is computed once and sent unchanged; what a turn does not
allow is refused at call time with the rule that masks it.

The requests are captured at unillm's transport (see
``tests/cache_discipline_helpers.py``).
"""

from __future__ import annotations


import pytest

from tests import cache_discipline_helpers as h


def _names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


def _tool_reply(requests: list[dict], call_id: str) -> str:
    for request in requests:
        for message in request["messages"]:
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                return message["content"]
    raise AssertionError(f"no tool reply for {call_id}")


# ── on ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", h.ONE_SESSION)
async def test_on_the_tool_list_is_identical_on_every_call(scenario):
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    assert len(requests) >= 2
    tool_bytes = {h.request_bytes(r)["tools"] for r in requests}
    assert len(tool_bytes) == 1, [_names(r) for r in requests]


@pytest.mark.asyncio
async def test_on_the_list_holds_every_tool_in_a_fixed_order():
    _result, _counter, requests = await h.scenario_gate()
    assert _names(requests[0]) == [
        # the caller's tools, by name, whatever the turn's policy shows
        "FunctionManager_search_functions",
        "GuidanceManager_search",
        "execute_code",
        # compress_context even on the eager gate turn that withholds it
        "compress_context",
    ]


@pytest.mark.asyncio
async def test_on_a_masked_call_is_refused_with_the_rule_and_not_run():
    result, counter, requests = await h.scenario_gate()
    assert result == "done"
    # The early call was refused; only the call after the gate ran.
    assert counter == {
        "FunctionManager_search_functions": 1,
        "GuidanceManager_search": 1,
        "execute_code": 1,
    }
    refusal = _tool_reply(requests, "call_0")
    assert "`execute_code` is not available right now" in refusal
    assert "the libraries are searched first" in refusal
    assert (
        "Available now: `FunctionManager_search_functions`, "
        "`GuidanceManager_search`."
    ) in refusal
    assert "refused rather than removed" in refusal
    # The gate was still open on the turn after the refusal.
    assert requests[1]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_on_a_context_full_turn_allows_only_compress_context():
    result, counter, requests = await h.scenario_threshold()
    assert result == "stopping"
    assert counter == {"execute_code": 1}
    refusal = _tool_reply(requests, "call_1")
    assert "the context window is nearly full" in refusal
    assert "Available now: `compress_context`" in refusal
    assert "execute_code" in _names(requests[1])


@pytest.mark.asyncio
async def test_on_a_refused_call_does_not_satisfy_a_gate():
    """A call to a masked library tool is not a library search: the gate stays."""
    counter: dict = {}
    tools = h.make_tools(counter)

    async def FunctionManager_add_functions(source: str) -> str:
        """Store a function.

        Args:
            source: The function's source.
        """
        counter["FunctionManager_add_functions"] = 1
        return "stored"

    tools["FunctionManager_add_functions"] = FunctionManager_add_functions
    replies = (
        lambda: h.completion(
            calls=[
                ("FunctionManager_add_functions", {"source": "def f(): pass"}),
                ("GuidanceManager_search", {"k": 1}),
            ],
        ),
        lambda: h.completion(
            calls=[("FunctionManager_search_functions", {"query": "q"})],
        ),
        lambda: h.completion(content="done"),
    )
    with h.scripted(replies) as provider:
        result = await h._run(
            h.new_client(),
            tools,
            "Do the task.",
            tool_policy=h.gate_policy,
        )
    assert result == "done"
    assert "FunctionManager_add_functions" not in counter
    assert provider.requests[1]["tool_choice"] == "required"
    assert "the libraries are searched first" in _tool_reply(
        provider.requests,
        "call_0",
    )
    assert len({h.request_bytes(r)["tools"] for r in provider.requests}) == 1


@pytest.mark.asyncio
async def test_on_the_list_survives_a_later_turn_that_shows_fewer_tools():
    """A policy that narrows the tools on a later turn narrows only what runs."""
    counter: dict = {}
    tools = h.make_tools(counter)

    def narrowing(step, tools_, called):
        if step == 0:
            return "auto", tools_
        return (
            "auto",
            {"execute_code": tools_["execute_code"]},
            {"mask_rules": {"GuidanceManager_search": "searching is over"}},
        )

    replies = (
        lambda: h.completion(calls=[("GuidanceManager_search", {"k": 1})]),
        lambda: h.completion(calls=[("GuidanceManager_search", {"k": 2})]),
        lambda: h.completion(content="done"),
    )
    with h.scripted(replies) as provider:
        await h._run(
            h.new_client(),
            tools,
            "Do the task.",
            tool_policy=narrowing,
        )
    assert counter == {"GuidanceManager_search": 1}
    assert "searching is over" in _tool_reply(provider.requests, "call_1")
    assert len({h.request_bytes(r)["tools"] for r in provider.requests}) == 1


# ── the helpers the loop and the actor share ─────────────────────────────


def test_policy_mask_rules_reads_only_the_optional_keys():
    from unify.common._async_tool.cache_discipline import policy_mask_rules

    assert policy_mask_rules(("auto", {})) == ({}, None)
    assert policy_mask_rules(("auto", {}, True)) == ({}, None)
    assert policy_mask_rules(
        ("required", {}, {"eager": True, "mask_rule": "r", "mask_rules": {"a": "x"}}),
    ) == ({"a": "x"}, "r")
