"""Symbolic: an actor at capacity refuses a request quickly and says why."""

import asyncio

import pytest

from unify.actor import code_act_actor
from unify.actor.code_act_actor import CodeActActor


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_act_at_capacity_is_refused_after_the_slot_wait(monkeypatch):
    actor = CodeActActor()
    actor._slots = asyncio.Semaphore(0)
    monkeypatch.setattr(code_act_actor, "SLOT_WAIT_S", 0.01)

    with pytest.raises(RuntimeError) as e:
        await actor.act("hello")

    assert "at capacity" in str(e.value).lower()
