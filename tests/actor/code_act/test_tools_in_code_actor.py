"""The actor's first request: its only model-API tool is the registry's ``execute_code``, the same bytes every agent
sends, and no tool choice is forced."""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from unify.common import tool_registry


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_execute_code_is_the_only_tool(core_world):  # noqa: F811
    with h.scripted((lambda: h.completion(content="done"),)) as provider:
        handle = await new_actor().act("Do the task.", persist=False)
        assert await asyncio.wait_for(handle.result(), 120) == "done"
    first = provider.requests[0]
    assert first["tools"] == tool_registry.json_tools("actor")
    assert first.get("tool_choice") != "required"
