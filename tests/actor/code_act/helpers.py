"""Shared helpers for CodeActActor tests.

Transcript readers pull tool-call names and ``execute_code`` snippets out of
a handle's history. ``WORKER_START_BOUND_S`` and the ``worker_starts`` fixture
keep a sandboxed worker's start out of a test's tight steady-state bound;
``warm_up`` keeps a fresh actor's first-request costs out of it.
"""

from __future__ import annotations

import asyncio
import functools
import json
import time
from typing import Any, Iterator

import pytest

from unify.actor.execution import worker

#: What starting a sandboxed Python worker may take: the worker's own limit
#: (``worker.START_TIMEOUT_S``). The start builds the sandbox policy (the
#: first build scans the interpreter roots for secrets), launches bwrap and
#: the interpreter and waits for its ready line, so on a loaded host it can
#: take seconds. A tight bound is for work on a running worker; a wait that
#: includes a start adds this budget, and a timed region that includes one
#: subtracts the start's measured time (``worker_starts``).
WORKER_START_BOUND_S = worker.START_TIMEOUT_S


class WorkerStarts:
    """When sandboxed Python workers were starting, as measured by the test."""

    def __init__(self) -> None:
        self.spans: list[tuple[float, float]] = []

    @property
    def count(self) -> int:
        return len(self.spans)

    def seconds(self, since: float = float("-inf")) -> float:
        """Wall-clock seconds after *since* (``time.monotonic()``) during which
        a worker was starting; overlapping starts count once."""
        total, covered = 0.0, since
        for began, ended in sorted(self.spans):
            began = max(began, covered)
            if ended > began:
                total += ended - began
                covered = ended
        return total


@pytest.fixture
def worker_starts(monkeypatch) -> WorkerStarts:
    """Time every sandboxed worker start in the test, from the policy build
    to the worker's ready line (``PythonWorker._start``)."""
    starts = WorkerStarts()
    original = worker.PythonWorker._start

    @functools.wraps(original)
    async def timed(self) -> None:
        began = time.monotonic()
        try:
            await original(self)
        finally:
            starts.spans.append((began, time.monotonic()))

    monkeypatch.setattr(worker.PythonWorker, "_start", timed)
    return starts


#: What a throwaway warm-up request (``warm_up``) may take: its cell may
#: start the worker, and the rest is two scripted model calls.
WARM_UP_BOUND_S = WORKER_START_BOUND_S + 20


async def warm_up(actor) -> None:
    """Run one throwaway request on *actor* so a timed request after it runs
    on a warm actor.

    A fresh actor's first request pays one-time costs that are not that
    request's: the first prompt and tool-loop build, the first model-call
    plumbing, imports on first use, and the first cell's worker start and
    first execution (the worker stays with the actor's executor for its later
    requests). On a loaded host these take seconds, so a tight bound over a
    first request measures the host. The warm-up runs one cell and answers in
    text, in its own scripted block, so a timed request's provider sees only
    that request's model calls."""
    from tests import cache_discipline_helpers as h

    replies = [
        h.completion(calls=[("execute_code", {"code": "1"})]),
        h.completion(content="warm"),
    ]
    with h.scripted(replies) as provider:
        handle = await actor.act(
            "Warm up.",
            persist=False,
            clarification_enabled=False,
        )
        assert await asyncio.wait_for(handle.result(), WARM_UP_BOUND_S) == "warm"
        done = getattr(handle, "_completion_event", None)
        if done is not None:
            await asyncio.wait_for(done.wait(), WARM_UP_BOUND_S)
    assert len(provider.requests) == 2


def _iter_tool_calls(chat_history: list[dict[str, Any]]) -> Iterator[dict]:
    for msg in chat_history:
        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    yield tc


def _tool_call_name_and_args(tc: dict) -> tuple[Any, Any]:
    fn = tc.get("function")
    if isinstance(fn, dict):
        return fn.get("name"), fn.get("arguments")
    return tc.get("name"), tc.get("arguments")


def history(handle: Any) -> list[dict[str, Any]]:
    """The conversation of a CodeActActor handle's loop (its client's messages)."""
    return list(getattr(handle._client, "messages", None) or [])


def get_code_act_tool_calls(handle: Any) -> list[str]:
    """Tool-call names from a CodeActActor handle's chat history, in order."""
    names: list[str] = []
    for tc in _iter_tool_calls(history(handle)):
        name, _args = _tool_call_name_and_args(tc)
        if isinstance(name, str):
            names.append(name)
    return names


def extract_code_act_execute_code_snippets(handle: Any) -> list[str]:
    """The ``code`` argument of every ``execute_code`` call in the handle's history."""
    snippets: list[str] = []
    for tc in _iter_tool_calls(history(handle)):
        name, args = _tool_call_name_and_args(tc)
        if name != "execute_code":
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if isinstance(args, dict):
            code = args.get("code")
            if isinstance(code, str) and code.strip():
                snippets.append(code)
    return snippets
