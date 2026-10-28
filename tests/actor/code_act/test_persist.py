"""Symbolic: a persistent act keeps its loop alive between requests.

The model is the scripted transport (``tests/scripted_model.py``): each turn
is a text reply, so the loop parks after it. The loop signals each park with
a ``response`` notification on the handle, which the test waits for instead
of sleeping.
"""

import asyncio

import pytest

from tests.scripted_model import ScriptedModel, reply, scripted
from unify.actor.code_act_actor import CodeActActor


async def _parked_with(handle, content: str) -> None:
    """Wait for the ``response`` notification of a parked turn."""
    while True:
        event = await asyncio.wait_for(handle.next_notification(), timeout=30)
        if event.get("type") == "response":
            assert event.get("content") == content
            return


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_code_act_persist_keeps_loop_alive_until_stopped():
    """
    When persist=True, the underlying async tool loop should not terminate after a
    single assistant message. Instead it should wait for interjections until stopped.
    """
    model = ScriptedModel(actor=[reply("READY"), reply("ACK")])
    actor = CodeActActor(timeout=60)
    with scripted(model):
        handle = await actor.act(
            "Reply with exactly 'READY' and do not call any tools. Then wait for further instructions.",
            clarification_enabled=False,
            persist=True,
        )

        await _parked_with(handle, "READY")
        assert (
            not handle.done()
        ), "persist=True should keep the loop alive after first message"
        assert len(model.of("actor")) == 1

        # A new request wakes the loop; it answers and parks again.
        _ = await handle.submit(
            "Now reply with exactly 'ACK' and keep waiting.",
        )
        await _parked_with(handle, "ACK")
        assert (
            not handle.done()
        ), "persist=True should keep the loop alive after interjections"
        second = model.of("actor")[1]
        assert any(
            m.get("role") == "user" and "exactly 'ACK'" in str(m.get("content"))
            for m in second.messages
        )

        # Stop should terminate the loop, with no further request.
        await asyncio.wait_for(handle.stop("test complete"), timeout=30)
        await asyncio.wait_for(handle.result(), timeout=30)
        assert handle.done()
    assert model.kinds() == ["actor", "actor"]
    model.assert_used_up()
