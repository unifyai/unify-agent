"""Symbolic: ``UNIFY_ENV_CARDS`` reaches a task's first message, and its recorder is left when the handle is built.

The provider is scripted; nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from tests.helpers import _handle_project
from unify.actor import code_act_actor as caa
from unify.actor import env_cards as ec
from unify.settings import SETTINGS

TASK = "Check the weather in Oslo."
MARK = (
    ec.HEADER
    + "### weather\n- `weather.forecast(city)` → {city: str} (worked in 2 sessions)\n"
)


async def _act(task):
    actor = caa.CodeActActor()
    try:
        with h.scripted([lambda: h.completion(content="done")] * 8) as provider:
            handle = await actor.act(task, persist=False)
            await asyncio.wait_for(handle.result(), 60)
    finally:
        await actor.close()
    return h.session_requests(provider.requests)


def _first_user(requests):
    return next(m["content"] for m in requests[0]["messages"] if m["role"] == "user")


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_on_the_section_is_in_the_first_message_and_the_scope_is_left(
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_CARDS", True)
    entered, left = [], []
    monkeypatch.setattr(
        ec,
        "enter",
        lambda request: (entered.append(request) or "scope", MARK),
    )
    monkeypatch.setattr(ec, "leave", lambda scope: left.append(scope))
    requests = await _act(TASK)
    assert "`weather.forecast(city)` → {city: str}" in str(_first_user(requests))
    assert entered and TASK in str(entered[0])
    assert left == ["scope"]


@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_off_no_section():
    assert SETTINGS.UNIFY_ENV_CARDS is False
    requests = await _act(TASK)
    assert "## Environment Notes" not in str(requests[0]["messages"])
