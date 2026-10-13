"""The actor's first request with ``UNIFY_TOOLS_IN_CODE``: off, its tools and request options are byte for byte the
base commit's (digests taken there on the test worker); on, its only model-API tool is the registry's
``execute_code``, the same bytes every agent sends."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.core_world import (  # noqa: F401 (fixtures)
    core_world,
    new_actor,
    world,
)
from tests.helpers import _handle_project
from unify.common import tool_registry
from unify.settings import SETTINGS

#: sha256 prefixes of json.dumps of the first request's ``tools`` and of its other options at 1f7bd11be.
BASE_TOOLS = "8f3d0e1b2b6bad50"
BASE_OPTIONS = "c3bb69ddcb85ed8d"


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()[:16]


async def _first_request() -> dict:
    actor = new_actor(can_store=False)
    try:
        with h.scripted((lambda: h.completion(content="done"),)) as provider:
            handle = await actor.act("Do the task.", persist=False)
            assert await asyncio.wait_for(handle.result(), 120) == "done"
    finally:
        await actor.close()
    return provider.requests[0]


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_with_the_switch_off_the_first_request_is_the_base_commits(core_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    first = await _first_request()
    assert _h(first["tools"]) == BASE_TOOLS
    assert _h({k: v for k, v in first.items() if k not in ("messages", "tools")}) == BASE_OPTIONS


@pytest.mark.asyncio
@pytest.mark.timeout(120)
@_handle_project
async def test_with_the_switch_on_execute_code_is_the_only_tool(core_world, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    first = await _first_request()
    assert first["tools"] == tool_registry.json_tools("actor")
    assert first.get("tool_choice") != "required"
