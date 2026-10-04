"""Symbolic: ``UNIFY_DELEGATION``: whether a top-level actor gets sub-actors, and their docs.

Every top-level actor is built with the sub-actor environment
(``primitives.actor``), whose 2,338-token ``act`` docstring rides every
main-actor request. On the 4 Oct ARC LOW runs delegation cost 14.4 calls and
18% of USD per episode in the fixed build (34.4 calls, 34% upstream), and 15
of 20 lost episodes delegated an action only the parent could take. ``off``
builds the actor without it; ``on_demand`` keeps it and leaves its docs to
``help()``, whose output is appended like any tool result.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.environments import ActorEnvironment
from unify.actor.environments import actor as actor_env
from unify.settings import SETTINGS

_HEADING = "### `primitives.actor` — Actor Delegation"
_FULL_DOCS_MARK = "When NOT to use"


def _has_sub_actor(envs) -> bool:
    return any(isinstance(e, ActorEnvironment) for e in envs)


@pytest.mark.parametrize(
    "mode, expected",
    [("on", True), ("off", False), ("on_demand", True)],
)
def test_the_top_level_environments_follow_the_switch(monkeypatch, mode, expected):
    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", mode)
    assert _has_sub_actor(actor_env.top_level_environments()) is expected


def test_on_the_top_level_environments_are_as_shipped(monkeypatch):
    from unify.actor.environments import registered_environments

    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", "on")
    envs = actor_env.top_level_environments()
    shipped = [ActorEnvironment(), *registered_environments()]
    assert [type(e) for e in envs] == [type(e) for e in shipped]


async def _run(replies, mode: str, monkeypatch) -> list[dict]:
    """A scripted ``act()`` of a top-level actor, without the discovery gate."""
    from unify.actor.code_act_actor import CodeActActor

    monkeypatch.setattr(SETTINGS, "UNIFY_DELEGATION", mode)
    actor = CodeActActor(
        environments=actor_env.top_level_environments(),
        tool_policy=None,
    )
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act(
                "Answer the request.",
                persist=False,
                can_store=False,
                clarification_enabled=False,
            )
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return provider.requests


def _system(request: dict) -> str:
    return request["messages"][0]["content"]


def _tool_description(request: dict, name: str) -> str:
    for tool in request["tools"] or []:
        if tool["function"]["name"] == name:
            return tool["function"]["description"]
    raise AssertionError(f"{name} not in the request's tools")


_DONE = [lambda: h.completion(content="done")]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_the_prompt_carries_the_delegation_docs(monkeypatch):
    request = (await _run(_DONE, "on", monkeypatch))[0]
    system = _system(request)
    assert _HEADING in system and _FULL_DOCS_MARK in system
    assert "| `primitives` |" in system


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_nothing_in_the_request_offers_a_sub_actor(monkeypatch):
    request = (await _run(_DONE, "off", monkeypatch))[0]
    system = _system(request)
    assert "primitives.actor" not in system
    assert "| `primitives` |" not in system
    assert "sub-agent (`primitives.actor.act`)" not in system
    assert "primitives.actor" not in _tool_description(request, "execute_function")
    assert "primitives.actor" not in json.dumps(request["tools"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_demand_the_prompt_points_at_help(monkeypatch):
    on = _system((await _run(_DONE, "on", monkeypatch))[0])
    request = (await _run(_DONE, "on_demand", monkeypatch))[0]
    system = _system(request)
    assert _HEADING in system
    assert "help(primitives.actor.act)" in system
    assert _FULL_DOCS_MARK not in system
    assert len(system) < len(on) - 5_000
    # The sub-actor is still there.
    assert "| `primitives` |" in system


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_on_demand_help_returns_the_full_docs(monkeypatch):
    replies = [
        lambda: h.completion(
            calls=[
                (
                    "execute_code",
                    {
                        "thought": "read the docs",
                        "code": "help(primitives.actor.act)",
                    },
                ),
            ],
        ),
        *_DONE * 3,
    ]
    requests = await _run(replies, "on_demand", monkeypatch)
    first, later = requests[0], requests[1]
    # Appended as a tool result; the prefix the first request sent is unchanged.
    assert later["messages"][: len(first["messages"])] == first["messages"]
    assert later["tools"] == first["tools"]
    results = [
        json.dumps(m.get("content"))
        for m in later["messages"]
        if m.get("role") == "tool"
    ]
    assert any(_FULL_DOCS_MARK in r for r in results)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_execute_function_refuses_the_sub_actor(monkeypatch):
    replies = [
        lambda: h.completion(
            calls=[
                (
                    "execute_function",
                    {
                        "thought": "delegate",
                        "function_name": "primitives.actor.act",
                        "call_kwargs": {"request": "do it"},
                    },
                ),
            ],
        ),
        *_DONE * 3,
    ]
    requests = await _run(replies, "off", monkeypatch)
    results = " ".join(
        json.dumps(m.get("content"))
        for m in requests[1]["messages"]
        if m.get("role") == "tool"
    )
    assert "runs without sub-actors" in results


@pytest.mark.parametrize(
    "value, expected",
    [
        ("on", "on"),
        ("off", "off"),
        ("on_demand", "on_demand"),
        ("ON_DEMAND", "on_demand"),
        ("1", "on"),
        ("true", "on"),
        ("0", "off"),
        ("false", "off"),
        ("", "on"),
    ],
)
def test_the_setting_parses(value, expected):
    from unify.settings import ProductionSettings

    assert ProductionSettings(UNIFY_DELEGATION=value).UNIFY_DELEGATION == expected


def test_the_setting_rejects_other_values():
    from unify.settings import ProductionSettings

    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_DELEGATION="sometimes")


def test_the_default_is_on():
    from unify.settings import ProductionSettings

    assert ProductionSettings.model_fields["UNIFY_DELEGATION"].default == "on"


def test_the_cli_and_the_conversation_manager_build_from_the_switch():
    """Both top-level builders take their environments from the switch."""
    import inspect

    from unify import cli
    from unify.conversation_manager.domains import managers_utils

    for module in (cli, managers_utils):
        source = inspect.getsource(module)
        assert "top_level_environments()" in source
        assert "ActorEnvironment(), *registered_environments()" not in source
