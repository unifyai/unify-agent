"""Symbolic: a top-level actor gets no sub-actors (``UNIFY_DELEGATION=off``, baked at the code freeze).

Every top-level actor is built with the sub-actor environment
(``primitives.actor``), whose 2,338-token ``act`` docstring rides every
main-actor request. On the 4 Oct ARC LOW runs delegation cost 14.4 calls and
18% of USD per episode in the fixed build (34.4 calls, 34% upstream), and 15
of 20 lost episodes delegated an action only the parent could take. The
actor is now built without it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from tests import cache_discipline_helpers as h
from unify.actor.environments import ActorEnvironment
from unify.actor.environments import actor as actor_env


def test_the_top_level_environments_have_no_sub_actor():
    envs = actor_env.top_level_environments()
    assert not any(isinstance(e, ActorEnvironment) for e in envs)


async def _run(replies) -> list[dict]:
    """A scripted ``act()`` of a top-level actor, without the discovery gate."""
    from unify.actor.code_act_actor import CodeActActor

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
async def test_off_nothing_in_the_request_offers_a_sub_actor():
    request = (await _run(_DONE))[0]
    system = _system(request)
    assert "primitives.actor" not in system
    assert "| `primitives` |" not in system
    assert "sub-agent (`primitives.actor.act`)" not in system
    assert "primitives.actor" not in _tool_description(request, "execute_function")
    assert "primitives.actor" not in json.dumps(request["tools"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_off_execute_function_refuses_the_sub_actor():
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
    requests = await _run(replies)
    results = " ".join(
        json.dumps(m.get("content"))
        for m in requests[1]["messages"]
        if m.get("role") == "tool"
    )
    assert "runs without sub-actors" in results


def test_the_cli_and_the_conversation_manager_build_from_the_switch():
    """Both top-level builders take their environments from the switch."""
    import inspect

    from unify import cli
    from unify.conversation_manager.domains import managers_utils

    for module in (cli, managers_utils):
        source = inspect.getsource(module)
        assert "top_level_environments()" in source
        assert "ActorEnvironment(), *registered_environments()" not in source
