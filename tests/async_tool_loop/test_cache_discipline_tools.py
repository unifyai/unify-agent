"""Symbolic: ``UNIFY_CACHE_DISCIPLINE`` keeps one tool list per session.

A provider reuses its prompt cache only as far as a request matches the last
one byte for byte, and the tool list comes first. Every actor conversation
used to change it on its second call (5 tools while the discovery gate was
open, 18 after, on AppWorld), so the cached prefix was lost there. With the
switch on the list is computed once and sent unchanged; what a turn does not
allow is refused at call time with the rule that masks it.

The requests are captured at unillm's transport (see
``tests/cache_discipline_helpers.py``); with the switch off they must be the
bytes the upstream commit sends for the same script.
"""

from __future__ import annotations

import json

import pytest

from tests import cache_discipline_helpers as h
from unify.settings import SETTINGS


@pytest.fixture
def discipline(monkeypatch):
    def set_(on: bool) -> None:
        monkeypatch.setattr(SETTINGS, "UNIFY_CACHE_DISCIPLINE", on)

    return set_


def _names(request: dict) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


def _tool_reply(requests: list[dict], call_id: str) -> str:
    for request in requests:
        for message in request["messages"]:
            if message.get("role") == "tool" and message.get("tool_call_id") == call_id:
                return message["content"]
    raise AssertionError(f"no tool reply for {call_id}")


# ── off: exactly as upstream ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(h.SCENARIOS))
async def test_off_every_request_is_byte_identical_to_upstream(discipline, scenario):
    discipline(False)
    golden = json.loads(h.GOLDEN.read_text())[scenario]
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    assert [h.request_bytes(r) for r in requests] == golden


# ── on ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", sorted(h.SCENARIOS))
async def test_on_the_tool_list_is_identical_on_every_call(discipline, scenario):
    discipline(True)
    _result, _counter, requests = await h.SCENARIOS[scenario]()
    assert len(requests) >= 2
    tool_bytes = {h.request_bytes(r)["tools"] for r in requests}
    assert len(tool_bytes) == 1, [_names(r) for r in requests]


@pytest.mark.asyncio
async def test_on_the_list_holds_every_tool_in_a_fixed_order(discipline):
    discipline(True)
    _result, _counter, requests = await h.scenario_gate()
    assert _names(requests[0]) == [
        # the caller's tools, by name, whatever the turn's policy shows
        "FunctionManager_search_functions",
        "GuidanceManager_search",
        "execute_code",
        # compress_context even on the eager gate turn that withholds it
        "compress_context",
        # the loop's own static surface, in the loop's order
        "wait",
        "steer",
        "ask_about_completed_tool",
    ]


@pytest.mark.asyncio
async def test_on_a_masked_call_is_refused_with_the_rule_and_not_run(discipline):
    discipline(True)
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
        "`GuidanceManager_search`, `ask_about_completed_tool`, `steer`, `wait`."
    ) in refusal
    assert "refused rather than removed" in refusal
    # The gate was still open on the turn after the refusal.
    assert requests[1]["tool_choice"] == "required"


@pytest.mark.asyncio
async def test_on_a_context_full_turn_allows_only_compress_context(discipline):
    discipline(True)
    result, counter, requests = await h.scenario_threshold()
    assert result == "stopping"
    assert counter == {"execute_code": 1}
    refusal = _tool_reply(requests, "call_1")
    assert "the context window is nearly full" in refusal
    assert "Available now: `ask_about_completed_tool`, `compress_context`" in refusal
    assert "execute_code" in _names(requests[1])


@pytest.mark.asyncio
async def test_on_a_refused_call_does_not_satisfy_a_gate(discipline):
    """A call to a masked library tool is not a library search: the gate stays."""
    discipline(True)
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
            interrupt_llm_with_interjections=False,
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
async def test_on_the_list_survives_a_later_turn_that_shows_fewer_tools(discipline):
    """A policy that narrows the tools on a later turn narrows only what runs."""
    discipline(True)
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
            interrupt_llm_with_interjections=False,
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


def test_turn_available_tools_is_unset_outside_a_dispatch():
    from unify.common._async_tool import cache_discipline as cd

    assert cd.turn_available_tools() is None
    token = cd.set_turn_available_tools(["a", "b"])
    try:
        assert cd.turn_available_tools() == frozenset({"a", "b"})
    finally:
        cd.reset_turn_available_tools(token)
    assert cd.turn_available_tools() is None


def test_actor_gate_names_the_rule_only_with_the_switch_on(discipline):
    from unify.actor.code_act_actor import _default_tool_policy

    tools = {
        "FunctionManager_search_functions": object(),
        "GuidanceManager_search": object(),
        "execute_code": object(),
    }
    policy = _default_tool_policy(True, True, lambda t: t)
    discipline(False)
    assert policy(0, tools, [])[2] == {"eager": True}
    discipline(True)
    opts = policy(0, tools, [])[2]
    assert opts["eager"] is True
    assert opts["mask_rule"] == (
        "the libraries are searched first -- call "
        "`FunctionManager_search_functions`, `GuidanceManager_search` before "
        "any other tool"
    )


def test_admission_rules_wrap_two_and_three_argument_policies():
    from unify.actor.code_act_actor import _ADMISSION_MASK_RULE, _with_mask_rules

    rules = {"FunctionManager_add_functions": _ADMISSION_MASK_RULE}

    def two(step, tools):
        return "auto", tools

    def three(step, tools, called):
        return "required", {}, {"eager": True, "mask_rules": {"x": "own rule"}}

    mode, visible, opts = _with_mask_rules(two, rules)(0, {"a": 1}, [])
    assert (mode, visible) == ("auto", {"a": 1})
    assert opts == {"mask_rules": rules}
    mode, visible, opts = _with_mask_rules(three, rules)(0, {"a": 1}, ["y"])
    assert mode == "required" and opts["eager"] is True
    assert opts["mask_rules"] == {"x": "own rule", **rules}
    assert "read-only during this task" in _ADMISSION_MASK_RULE


def test_discovery_mutator_reads_the_turns_allowed_tools(discipline):
    """With every tool in the request, the allowed set says it is a gate turn."""
    from unillm.clients.completion_mutator import CompletionMutatorContext

    from unify.actor.code_act_actor import _build_discovery_parallel_mutator
    from unify.common._async_tool import cache_discipline as cd

    every_tool = [
        {"type": "function", "function": {"name": n, "parameters": {}}}
        for n in (
            "FunctionManager_search_functions",
            "GuidanceManager_search",
            "execute_code",
            "compress_context",
        )
    ]
    context = CompletionMutatorContext(
        provider="openrouter",
        original_tool_choice="required",
        request_kw={"tools": every_tool},
    )
    mutator = _build_discovery_parallel_mutator()

    def called(turn) -> list[str]:
        return [
            (c["function"]["name"] if isinstance(c, dict) else c.function.name)
            for c in turn.choices[0].message.tool_calls
        ]

    one_search = lambda: h.completion(  # noqa: E731
        calls=[("GuidanceManager_search", {"k": 1})],
    )
    # Outside a dispatch the full list is not a gate schema: nothing appended.
    assert called(mutator(one_search(), context)) == ["GuidanceManager_search"]
    token = cd.set_turn_available_tools(
        ["FunctionManager_search_functions", "GuidanceManager_search"],
    )
    try:
        assert called(mutator(one_search(), context)) == [
            "GuidanceManager_search",
            "FunctionManager_search_functions",
        ]
    finally:
        cd.reset_turn_available_tools(token)
