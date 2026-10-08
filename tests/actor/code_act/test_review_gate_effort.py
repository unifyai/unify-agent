"""Symbolic: the review gate runs at the effort the session ran at.

Reasoning effort is a fixed condition of each comparison (LOW, MEDIUM,
HIGH), never something the harness changes. The gate's one call used to be
pinned to ``low`` whatever the session ran at; its request now carries the
session's own effort, the same field and value the session's requests carry.

The model is the scripted transport (tests/cache_discipline_helpers.py), which
records each request's ``reasoning_effort``; nothing leaves the process.
"""

from __future__ import annotations

import asyncio

import pytest

from tests import cache_discipline_helpers as h
from unify.actor import code_act_actor as caa
from unify.actor import review_gate
from unify.common.llm_client import new_llm_client
from unify.settings import SETTINGS


async def _next(handle, kinds) -> dict:
    while True:
        note = await asyncio.wait_for(handle.next_notification(), 30)
        if isinstance(note, dict) and note.get("type") in kinds:
            return note


async def _session_and_gate(effort):
    """One session turn at *effort*, ended; the gate answers no."""
    actor = caa.CodeActActor()
    replies = (
        lambda: h.completion(content="Done."),
        lambda: h.completion(content='{"review": false, "reason": "one-off"}'),
    )
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    try:
        with h.scripted(replies) as provider:
            from unify.common.async_tool_loop import start_async_tool_loop

            client = new_llm_client(h.MODEL, cache=False, **kwargs)
            if effort is None:
                client.set_reasoning_effort(None)
            client.set_system_message("You are a scripted actor.")
            inner = start_async_tool_loop(
                client,
                "Say done.",
                h.session_tools(actor),
                loop_id="CodeActActor.act",
                log_steps=False,
                timeout=60,
                persist=True,
            )
            handle = caa._StorageCheckHandle(inner=inner, actor=actor)
            await _next(handle, ("response",))
            await handle.stop(caa.SESSION_ENDED)
            note = await _next(
                handle,
                ("storage_review_complete", "storage_review_skipped"),
            )
            await asyncio.wait_for(handle._lifecycle_task, 30)
    finally:
        await actor.close()
    return note, provider.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("effort", ["low", "high", None])
async def test_the_gate_request_carries_the_sessions_effort(monkeypatch, effort):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "")
    # The gate is asked only when the library holds something.
    monkeypatch.setattr(review_gate, "library_is_empty", lambda counts: False)
    note, requests = await _session_and_gate(effort)
    assert note["type"] == "storage_review_skipped", note
    assert len(requests) == 2
    session, gate = requests
    assert gate["messages"][0]["content"] == review_gate.GATE_SYSTEM_PROMPT
    assert session.get("reasoning_effort") == effort
    assert gate.get("reasoning_effort") == session.get("reasoning_effort")


def test_no_effort_is_hard_coded_in_the_gate():
    assert not hasattr(review_gate, "GATE_EFFORT")
